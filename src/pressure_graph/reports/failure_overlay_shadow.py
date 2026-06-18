"""F3 / F5 long-side failure-overlay live shadow recorder.

Shadow-tracks the two v3.5 winners on the v0.9D capacity trade cache as the
cache grows over time. Re-running this module is idempotent: it re-derives
the F3 / F5 decisions for the full cache and rewrites the ledger / daily
summary / status doc. No live execution path is touched — this is a research
shadow ledger, not an order router.

For every long signal in the focal pool, two decisions are recorded:

- **F3 ``no_overflow_only``** — if `failure_recent (S1/S3/S5, 48-bar cooldown)`
  is true AND the signal would have entered the O6 overflow sleeve, the gate
  blocks the overflow allocation. Baseline P2 entries are never blocked.
- **F5 ``cic2_only_no_long``** — if `failure_recent` is true AND the
  signal's `candidate == CIC2_beta_broad`, the long is blocked entirely.

For each blocked signal the ledger captures: the realised PnL of the trade
that DID fire (so we measure what the gate cost or saved), the motif that
triggered the gate, the gate channel, and whether the gate was an
over-gate (blocked a winner).

The companion CSV `failure_overlay_daily_summary.csv` rolls the ledger up
by date × strategy. `failure_overlay_candidate_status.md` carries the
current 7-criterion status against the v3.5 B3 baseline so a future
operator can see whether F3 / F5 still hold their shadow tag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v3_5_failure_risk_layer_bridge import (
    ACTION_ALLOW,
    ACTION_SKIP_FULL,
    ACTION_SKIP_OVERFLOW,
    DEFAULT_COOLDOWN_BARS,
    DEFAULT_LONG_POOL,
    DEFAULT_MAX_POSITIONS,
    TRADE_CACHE_PATH as V35_TRADE_CACHE_PATH,
    V35Config,
    _load_pool,
    _per_row_failure_recent,
    _per_row_motif_attribution,
)
from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    RiskOffConfig,
    stream_risk_off_events,
)
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10d_late_burst_overflow import BASELINE_MAX_POSITIONS

REPORT_ROOT = Path("reports/failure_overlay_shadow")
DEFAULT_TRADE_CACHE = V35_TRADE_CACHE_PATH

STRATEGY_F3 = "F3_no_overflow_only"
STRATEGY_F5 = "F5_cic2_only_no_long"
STRATEGIES: tuple[str, ...] = (STRATEGY_F3, STRATEGY_F5)


@dataclass(frozen=True)
class ShadowConfig:
    """F3 / F5 live shadow driver config — idempotent across re-runs."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    long_pool_name: str = DEFAULT_LONG_POOL
    top_n: int = 30
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    max_positions: int = DEFAULT_MAX_POSITIONS
    baseline_label: str = "B3_P2_max8_plus_ProtectA_cap2_O6"
    strategies: tuple[str, ...] = STRATEGIES
    # An overflow trade is identified by the v0.9D `burst_count_so_far`
    # column meeting the O6 minimum (the v3.5 / v12s3 convention).
    overflow_min_burst_count: int = 9
    # over-gate threshold: a blocked trade whose realised net exceeded this
    # number is flagged as a winner-kill in the ledger.
    over_gate_net_threshold: float = 0.0


# --------------------------------------------------------------------------------------
# Decision logic — produces a per-row decision Series for each strategy.
# --------------------------------------------------------------------------------------


def _flag_overflow_candidate(pool: pd.DataFrame, cfg: ShadowConfig) -> pd.Series:
    """A row is considered an overflow-channel candidate if its
    ``burst_count_so_far`` ≥ ``cfg.overflow_min_burst_count``. This is the same
    test the O6 simulator uses to decide eligibility for the overflow slot.
    """
    if "burst_count_so_far" not in pool.columns:
        return pd.Series(False, index=pool.index)
    return (
        pd.to_numeric(pool["burst_count_so_far"], errors="coerce").fillna(0)
        >= cfg.overflow_min_burst_count
    )


def _f3_decisions(pool: pd.DataFrame, failure_recent: pd.Series, cfg: ShadowConfig) -> pd.Series:
    """F3 = no_overflow_only: skip the overflow slot when failure_recent AND
    the row is an overflow-eligible candidate."""
    overflow_flag = _flag_overflow_candidate(pool, cfg)
    decisions = pd.Series(ACTION_ALLOW, index=pool.index, dtype=object)
    decisions[failure_recent & overflow_flag] = ACTION_SKIP_OVERFLOW
    return decisions


def _f5_decisions(pool: pd.DataFrame, failure_recent: pd.Series, cfg: ShadowConfig) -> pd.Series:
    """F5 = CIC2-only no-long: skip the entire entry when failure_recent AND
    the candidate is CIC2_beta_broad."""
    is_cic2 = pool.get("candidate", pd.Series("", index=pool.index)).astype(str).eq("CIC2_beta_broad")
    decisions = pd.Series(ACTION_ALLOW, index=pool.index, dtype=object)
    decisions[failure_recent & is_cic2] = ACTION_SKIP_FULL
    return decisions


# --------------------------------------------------------------------------------------
# Ledger builder — one row per (pool_row, strategy)
# --------------------------------------------------------------------------------------


def _build_ledger(
    pool: pd.DataFrame,
    events: pd.DataFrame,
    cfg: ShadowConfig,
) -> pd.DataFrame:
    """One row per (pool row, strategy). Captures the gate decision, the
    realised P&L of the trade that DID fire (so we measure the gate's cost
    or saving), the motif that triggered the gate, and the over-gate flag.
    """
    if pool.empty:
        return pd.DataFrame()
    failure_recent = _per_row_failure_recent(pool, events, cfg.cooldown_bars)
    motif_attr = _per_row_motif_attribution(pool, events, cfg.cooldown_bars)
    decisions_by_strategy: dict[str, pd.Series] = {
        STRATEGY_F3: _f3_decisions(pool, failure_recent, cfg),
        STRATEGY_F5: _f5_decisions(pool, failure_recent, cfg),
    }
    overflow_flag = _flag_overflow_candidate(pool, cfg)
    is_cic2 = pool.get("candidate", pd.Series("", index=pool.index)).astype(str).eq("CIC2_beta_broad")
    rows: list[dict[str, object]] = []
    for strategy, decisions in decisions_by_strategy.items():
        for i in range(len(pool)):
            decision = str(decisions.iloc[i])
            realized = float(
                pd.to_numeric(pool["net_return"].iloc[i], errors="coerce") if "net_return" in pool.columns else np.nan
            )
            actual_executed = True  # cache rows are realised trades
            blocked = decision in (ACTION_SKIP_FULL, ACTION_SKIP_OVERFLOW)
            over_gate = blocked and np.isfinite(realized) and realized > cfg.over_gate_net_threshold
            rows.append(
                {
                    "decision_time": pool["signal_time"].iloc[i] if "signal_time" in pool.columns else pd.NaT,
                    "entry_time": pool["entry_time"].iloc[i] if "entry_time" in pool.columns else pd.NaT,
                    "strategy": strategy,
                    "symbol": str(pool["symbol"].iloc[i]) if "symbol" in pool.columns else "",
                    "candidate": str(pool["candidate"].iloc[i]) if "candidate" in pool.columns else "",
                    "is_cic2": bool(is_cic2.iloc[i]),
                    "is_overflow_eligible": bool(overflow_flag.iloc[i]),
                    "decision": decision,
                    "blocked": blocked,
                    "actual_executed": actual_executed,
                    "realized_net_return": realized,
                    "month": str(pool["month"].iloc[i]) if "month" in pool.columns else "",
                    "motif_recent": str(motif_attr.iloc[i]),
                    "failure_recent": bool(failure_recent.iloc[i]),
                    "over_gate": bool(over_gate),
                    "baseline_label": cfg.baseline_label,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Daily summary roll-up — by (date, strategy)
# --------------------------------------------------------------------------------------


def _daily_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    work = ledger.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="coerce")
    work["date"] = work["decision_time"].dt.strftime("%Y-%m-%d")
    work = work.dropna(subset=["date"]).copy()
    work["realized_net_return"] = pd.to_numeric(work.get("realized_net_return", np.nan), errors="coerce").fillna(0.0)
    work["blocked_int"] = work["blocked"].astype(bool).astype(int)
    work["over_gate_int"] = work["over_gate"].astype(bool).astype(int)
    work["gated_net"] = work["realized_net_return"] * work["blocked_int"]
    grouped = (
        work.groupby(["date", "strategy"], dropna=False)
        .agg(
            total_observations=("strategy", "size"),
            blocks=("blocked_int", "sum"),
            gated_realized_sum=("gated_net", "sum"),
            over_gate_count=("over_gate_int", "sum"),
        )
        .reset_index()
    )
    grouped["allowed"] = grouped["total_observations"] - grouped["blocks"]
    grouped["gated_realized_avg"] = np.where(
        grouped["blocks"] > 0, grouped["gated_realized_sum"] / grouped["blocks"], 0.0
    )
    grouped["over_gate_rate"] = np.where(
        grouped["blocks"] > 0, grouped["over_gate_count"] / grouped["blocks"], 0.0
    )
    # Net delta vs no-overlay baseline: we *avoid* the realised return of blocked
    # trades. If avg blocked return is negative, the gate is net-positive.
    grouped["net_delta_vs_baseline"] = -grouped["gated_realized_sum"]
    return grouped


# --------------------------------------------------------------------------------------
# Per-strategy status — aggregate metrics + verdict band
# --------------------------------------------------------------------------------------


def _strategy_status(ledger: pd.DataFrame, cfg: ShadowConfig) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame([{"strategy": s, "blocks": 0} for s in cfg.strategies])
    rows: list[dict[str, object]] = []
    for strategy in cfg.strategies:
        sub = ledger[ledger["strategy"].eq(strategy)]
        blocks = sub[sub["blocked"].astype(bool)]
        n_blocks = int(len(blocks))
        realized_blocks = pd.to_numeric(blocks["realized_net_return"], errors="coerce").dropna()
        gated_sum = float(realized_blocks.sum())
        gated_avg = float(realized_blocks.mean()) if len(realized_blocks) else float("nan")
        over_gate_count = int(blocks["over_gate"].astype(bool).sum())
        over_gate_rate = (over_gate_count / n_blocks) if n_blocks else float("nan")
        net_delta = -gated_sum
        win_share = float((realized_blocks > 0).mean()) if len(realized_blocks) else float("nan")
        loss_share = float((realized_blocks < 0).mean()) if len(realized_blocks) else float("nan")
        first_block = blocks["decision_time"].min() if n_blocks else pd.NaT
        last_block = blocks["decision_time"].max() if n_blocks else pd.NaT
        # 30-day window
        recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30) if n_blocks else None
        recent = (
            blocks[pd.to_datetime(blocks["decision_time"], utc=True, errors="coerce") >= recent_cutoff]
            if recent_cutoff is not None
            else blocks.iloc[0:0]
        )
        recent_blocks = int(len(recent))
        recent_gated_avg = (
            float(pd.to_numeric(recent["realized_net_return"], errors="coerce").mean())
            if len(recent)
            else float("nan")
        )
        rows.append(
            {
                "strategy": strategy,
                "blocks": n_blocks,
                "gated_realized_sum": gated_sum,
                "gated_realized_avg": gated_avg,
                "over_gate_count": over_gate_count,
                "over_gate_rate": over_gate_rate,
                "net_delta_vs_baseline": net_delta,
                "blocked_win_share": win_share,
                "blocked_loss_share": loss_share,
                "first_block_time": first_block,
                "last_block_time": last_block,
                "recent_30d_blocks": recent_blocks,
                "recent_30d_gated_avg": recent_gated_avg,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Verdict writer — candidate_status.md
# --------------------------------------------------------------------------------------


def _verdict_band(row: pd.Series) -> str:
    """Per-strategy verdict band. We want over-gate rate moderate (~30-55%)
    because the gate cuts on a noisy event-recent signal — too high means
    we're killing too many winners; too low means the gate barely fires."""
    n = int(row.get("blocks", 0))
    if n == 0:
        return "no_data"
    gated_avg = float(row.get("gated_realized_avg", float("nan")))
    over_rate = float(row.get("over_gate_rate", float("nan")))
    net_delta = float(row.get("net_delta_vs_baseline", float("nan")))
    if not np.isfinite(gated_avg) or not np.isfinite(net_delta):
        return "insufficient_data"
    if net_delta > 0 and (not np.isfinite(over_rate) or over_rate <= 0.55):
        return "shadow_holds"
    if net_delta > 0 and over_rate > 0.55:
        return "shadow_holds_but_over_gates"
    if net_delta <= 0 and over_rate > 0.55:
        return "demote_over_gating"
    return "neutral_review"


def _write_status_md(report_root: Path, status: pd.DataFrame, ledger: pd.DataFrame, cfg: ShadowConfig) -> Path:
    notes_path = report_root / "failure_overlay_candidate_status.md"
    lines: list[str] = [
        "# Failure Overlay Candidate Status — F3 / F5",
        "",
        "Periodic shadow ledger for the v3.5 long-side risk-layer winners.",
        "F3 = no_overflow only, F5 = CIC2-only no-long. Both gate on failure",
        "motif recency (S1/S3/S5 within 48-bar cooldown) at the long-entry",
        "signal_time. This document is a snapshot; rerun the CLI to refresh.",
        "",
        f"- pool: {cfg.long_pool_name}; cooldown {cfg.cooldown_bars} bars (12h)",
        f"- baseline reference: {cfg.baseline_label}",
        f"- ledger size: {len(ledger)} rows",
        "",
    ]
    if status.empty:
        lines.append("- empty status — re-run after a fresh v0.9D run.")
        notes_path.write_text("\n".join(lines), encoding="utf-8")
        return notes_path

    lines.append("## Per-strategy snapshot")
    for _, row in status.iterrows():
        strategy = str(row["strategy"])
        verdict = _verdict_band(row)
        lines.append(f"### {strategy} → **{verdict}**")
        lines.append(f"- blocks: {int(row.get('blocks', 0))}")
        lines.append(
            f"- gated realised avg: {float(row.get('gated_realized_avg', np.nan)):+.4%} "
            f"(loss share {float(row.get('blocked_loss_share', np.nan)):.2%}, "
            f"win share {float(row.get('blocked_win_share', np.nan)):.2%})"
        )
        lines.append(
            f"- over-gate rate: {float(row.get('over_gate_rate', np.nan)):.2%} "
            f"({int(row.get('over_gate_count', 0))} winners blocked)"
        )
        lines.append(f"- net delta vs no-overlay baseline: {float(row.get('net_delta_vs_baseline', np.nan)):+.4%}")
        recent_blocks = int(row.get("recent_30d_blocks", 0))
        recent_avg = float(row.get("recent_30d_gated_avg", np.nan))
        if recent_blocks:
            lines.append(
                f"- recent 30d: {recent_blocks} blocks, gated avg {recent_avg:+.4%}"
            )
        first = row.get("first_block_time")
        last = row.get("last_block_time")
        if pd.notna(first) and pd.notna(last):
            lines.append(f"- coverage: {pd.Timestamp(first).strftime('%Y-%m-%d')} → {pd.Timestamp(last).strftime('%Y-%m-%d')}")
        lines.append("")

    lines.extend([
        "## Verdict bands",
        "- `shadow_holds` — net delta positive, over-gate rate ≤ 55%. Keep shadowing.",
        "- `shadow_holds_but_over_gates` — net delta positive but blocking many winners; review motif cooldown.",
        "- `demote_over_gating` — net delta ≤ 0 and over-gate rate > 55%; gate likely broken.",
        "- `neutral_review` — sample too thin or signal mixed; check again next cycle.",
        "",
        "## Discipline notes (closure §保持同一套规范)",
        "- as-of feature only ✓ — failure_recent uses feature_time ≤ signal_time.",
        "- realised PnL pulled from the v0.9D capacity trade cache (post-cost net_return).",
        "- F4 (disable Protect_A) and CP60 prefilter are NOT in shadow — both retired in v3.5.",
        "- Tier: research only. No live execution / paper-live wiring is touched.",
    ])
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


# --------------------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------------------


def write_failure_overlay_shadow(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: ShadowConfig = ShadowConfig(),
) -> dict[str, Path]:
    """Refresh the F3 / F5 shadow ledger and emit ledger / summary / status."""
    report_root = ensure_dir(cfg.report_root)
    if not cfg.trade_cache_path.exists():
        notes_path = report_root / "failure_overlay_candidate_status.md"
        notes_path.write_text(
            f"# Failure Overlay Shadow — trade cache not found at {cfg.trade_cache_path}.\n",
            encoding="utf-8",
        )
        return {"failure_overlay_candidate_status": notes_path}

    # Pool: reuse v3.5's loader so the column conventions match.
    v35_cfg = V35Config(
        trade_cache_path=cfg.trade_cache_path,
        long_pool_name=cfg.long_pool_name,
        top_n=cfg.top_n,
        cooldown_bars=cfg.cooldown_bars,
        max_positions=cfg.max_positions,
    )
    pool = _load_pool(v35_cfg)
    if pool.empty:
        notes_path = report_root / "failure_overlay_candidate_status.md"
        notes_path.write_text("# Failure Overlay Shadow — empty long pool.\n", encoding="utf-8")
        return {"failure_overlay_candidate_status": notes_path}

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    events = stream_risk_off_events(
        feature_path,
        rank30,
        rank90,
        symbols,
        config,
        RiskOffConfig(motifs=cfg.motifs, symbol_cooldown_bars=cfg.cooldown_bars),
    )
    ledger = _build_ledger(pool, events, cfg)
    daily = _daily_summary(ledger)
    status = _strategy_status(ledger, cfg)

    outputs = {
        "failure_overlay_shadow_ledger": report_root / "failure_overlay_shadow_ledger.parquet",
        "failure_overlay_daily_summary": report_root / "failure_overlay_daily_summary.csv",
        "failure_overlay_strategy_status": report_root / "failure_overlay_strategy_status.csv",
        "failure_overlay_candidate_status": report_root / "failure_overlay_candidate_status.md",
    }
    # Parquet ledger: keeps types tight and re-runs cheap.
    ledger.to_parquet(outputs["failure_overlay_shadow_ledger"], index=False)
    daily.to_csv(outputs["failure_overlay_daily_summary"], index=False)
    status.to_csv(outputs["failure_overlay_strategy_status"], index=False)
    _write_status_md(report_root, status, ledger, cfg)
    print(
        f"failure_overlay_shadow: ledger {len(ledger)} rows across {len(cfg.strategies)} strategies, "
        f"daily summary {len(daily)} rows",
        flush=True,
    )
    return outputs


__all__ = [
    "REPORT_ROOT",
    "STRATEGIES",
    "STRATEGY_F3",
    "STRATEGY_F5",
    "ShadowConfig",
    "_build_ledger",
    "_daily_summary",
    "_f3_decisions",
    "_f5_decisions",
    "_flag_overflow_candidate",
    "_strategy_status",
    "_verdict_band",
    "write_failure_overlay_shadow",
]
