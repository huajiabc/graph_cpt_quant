"""v4.4 Binance Taker-Buy Fusion Expansion.

This report expands the v4.2 fusion check for Binance taker-buy confirmation
on the existing Bybit CIC/P2 long candidate pool.  It is not a standalone
lead-lag strategy and does not change live decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v40_cross_exchange_lead_lag import (
    DATA_ROOT,
    SOURCE_EXCHANGE,
    TARGET_EXCHANGE,
    _num,
    _select_common_symbols,
)
from pressure_graph.reports.v42_source_attribution_target_fusion import (
    DEFAULT_TRADE_CACHE,
    P2_POOL,
    SOURCE_1M_DATASET,
    TARGET_1M_DATASET,
    _asof_values,
    _has_prior_event,
    _has_source_context,
    _load_p2_trades,
    _mapped_symbol,
    _month_cap_trade,
    _source_dir,
    _source_matrices,
    _target_dir,
)


REPORT_ROOT = Path("reports/v4_4_binance_taker_buy_fusion")


@dataclass(frozen=True)
class V44Config:
    report_root: Path = REPORT_ROOT
    data_root: Path = DATA_ROOT
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    source_exchange: str = SOURCE_EXCHANGE
    target_exchange: str = TARGET_EXCHANGE
    source_dataset_1m: str = SOURCE_1M_DATASET
    target_dataset_1m: str = TARGET_1M_DATASET
    top_n: int = 50
    windows_minutes: tuple[int, ...] = (5, 15, 30)
    checkpoint_live_path: Path = Path("reports/v0_7d2_cic_mir1_paper_live/checkpoint_protection_attribution_live.csv")


def _net30(frame: pd.DataFrame) -> pd.Series:
    if "gross_return" in frame.columns:
        return _num(frame, "gross_return") - 0.006
    return _num(frame, "net_return") - 0.002


def _summary_row(frame: pd.DataFrame, bucket: str, *, window: int | str = "", scope: str = "") -> dict:
    if frame.empty:
        return {
            "scope": scope,
            "window_minutes": window,
            "bucket": bucket,
            "trades": 0,
            "net20": np.nan,
            "net30": np.nan,
            "hit_rate": np.nan,
            "month_cap35_net20": np.nan,
            "max_symbol_contribution": np.nan,
        }
    local = frame.copy()
    local["net20"] = _num(local, "net_return")
    local["net30"] = _net30(local)
    symbol_sum = local.groupby("symbol", dropna=False)["net20"].sum()
    total = float(symbol_sum.sum())
    return {
        "scope": scope,
        "window_minutes": window,
        "bucket": bucket,
        "trades": int(len(local)),
        "net20": float(local["net20"].mean()),
        "net30": float(local["net30"].mean()),
        "hit_rate": float(local["net20"].gt(0).mean()),
        "month_cap35_net20": _month_cap_trade(local, "net20"),
        "max_symbol_contribution": float((symbol_sum / total).abs().max()) if total else np.nan,
    }


def _annotate_taker_context(pool: pd.DataFrame, symbols: list[str], cfg: V44Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pd.DataFrame()
    _, matrices, density = _source_matrices(symbols, cfg)  # type: ignore[arg-type]
    taker_matrix = matrices["taker"]
    rows = []
    coverage = []
    for symbol, group in pool.groupby("symbol", sort=False):
        local = group.copy()
        source_symbols = {
            "real": symbol,
            "random": _mapped_symbol(symbols, symbol, "cyclic"),
            "shuffled": _mapped_symbol(symbols, symbol, "reverse"),
        }
        covered = symbol in taker_matrix.columns
        local["binance_source_covered"] = _has_source_context(taker_matrix, symbol, local["entry_time"]) if covered else False
        coverage.append(
            {
                "symbol": symbol,
                "trades": int(len(local)),
                "source_1m_covered": bool(covered),
                "entry_context_covered_trades": int(local["binance_source_covered"].sum()),
            }
        )
        for label, source_symbol in source_symbols.items():
            source_covered = source_symbol in taker_matrix.columns if source_symbol is not None else False
            for window in cfg.windows_minutes:
                col = f"{label}_taker_buy_prior_{int(window)}m"
                local[col] = (
                    _has_prior_event(taker_matrix, source_symbol, local["entry_time"], int(window))
                    if source_covered
                    else False
                )
        local["binance_market_density_at_entry"] = (
            _asof_values(density, local["entry_time"]).to_numpy()
            if not density.empty
            else np.nan
        )
        rows.append(local)
    annotated = pd.concat(rows, ignore_index=True) if rows else pool.copy()
    return annotated, pd.DataFrame(coverage)


def _with_without_rows(pool: pd.DataFrame, scope: str, mask: pd.Series, windows: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    sample = pool[mask.fillna(False)].copy()
    for window in windows:
        col = f"real_taker_buy_prior_{int(window)}m"
        covered = sample[sample["binance_source_covered"].astype(bool)].copy()
        yes = covered[covered[col].astype(bool)]
        no = covered[~covered[col].astype(bool)]
        yes_row = _summary_row(yes, "with_binance_taker_buy", window=window, scope=scope)
        no_row = _summary_row(no, "without_binance_taker_buy", window=window, scope=scope)
        lift = yes_row["net20"] - no_row["net20"] if not pd.isna(yes_row["net20"]) and not pd.isna(no_row["net20"]) else np.nan
        yes_row["with_minus_without_net20"] = lift
        no_row["with_minus_without_net20"] = lift
        yes_row["coverage_trades"] = int(len(covered))
        no_row["coverage_trades"] = int(len(covered))
        rows.extend([yes_row, no_row])
    return pd.DataFrame(rows)


def _fusion_summary(pool: pd.DataFrame, cfg: V44Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    all_mask = pd.Series(True, index=pool.index)
    summary = _with_without_rows(pool, P2_POOL, all_mask, cfg.windows_minutes)
    cic_rows = []
    for cic_type, mask in {
        "CIC1": pool["candidate"].astype(str).eq("CIC1_beta_extreme"),
        "CIC2": pool["candidate"].astype(str).eq("CIC2_beta_broad"),
    }.items():
        cic_rows.append(_with_without_rows(pool, cic_type, mask, cfg.windows_minutes))
    by_cic = pd.concat(cic_rows, ignore_index=True) if cic_rows else pd.DataFrame()
    o6_mask = _num(pool, "burst_count_so_far").ge(9)
    o6 = _with_without_rows(pool, "O6_eligible_burst_count_ge_9", o6_mask, cfg.windows_minutes)
    return summary, by_cic, o6


def _leave_one_month(pool: pd.DataFrame, cfg: V44Config) -> pd.DataFrame:
    rows = []
    covered = pool[pool["binance_source_covered"].astype(bool)].copy()
    months = sorted(covered["month"].dropna().astype(str).unique())
    for window in cfg.windows_minutes:
        col = f"real_taker_buy_prior_{int(window)}m"
        for removed in ["NONE", *months]:
            sample = covered if removed == "NONE" else covered[~covered["month"].astype(str).eq(removed)]
            yes = sample[sample[col].astype(bool)]
            no = sample[~sample[col].astype(bool)]
            yes_net = float(_num(yes, "net_return").mean()) if len(yes) else np.nan
            no_net = float(_num(no, "net_return").mean()) if len(no) else np.nan
            rows.append(
                {
                    "window_minutes": window,
                    "removed_month": removed,
                    "with_trades": int(len(yes)),
                    "without_trades": int(len(no)),
                    "with_net20": yes_net,
                    "without_net20": no_net,
                    "with_minus_without_net20": yes_net - no_net if not pd.isna(yes_net) and not pd.isna(no_net) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _random_shuffled_controls(pool: pd.DataFrame, cfg: V44Config) -> pd.DataFrame:
    rows = []
    covered = pool[pool["binance_source_covered"].astype(bool)].copy()
    for window in cfg.windows_minutes:
        for label in ["real", "random", "shuffled"]:
            col = f"{label}_taker_buy_prior_{int(window)}m"
            sample = covered[covered[col].astype(bool)]
            row = _summary_row(sample, f"{label}_with_taker_buy", window=window, scope=P2_POOL)
            rows.append(row)
        local = pd.DataFrame(rows)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    piv = out.pivot_table(index="window_minutes", columns="bucket", values="net20", aggfunc="first")
    for control in ["random_with_taker_buy", "shuffled_with_taker_buy"]:
        if "real_with_taker_buy" in piv.columns and control in piv.columns:
            piv[f"real_vs_{control}_lift"] = piv["real_with_taker_buy"] - piv[control]
    return out.merge(piv.reset_index(), on="window_minutes", how="left")


def _cp60_summary(cfg: V44Config) -> pd.DataFrame:
    if not cfg.checkpoint_live_path.exists():
        return pd.DataFrame(
            [{"status": "missing_checkpoint_live_path", "path": str(cfg.checkpoint_live_path), "rows": 0}]
        )
    frame = pd.read_csv(cfg.checkpoint_live_path)
    if frame.empty:
        return pd.DataFrame([{"status": "no_cp60_live_rows", "path": str(cfg.checkpoint_live_path), "rows": 0}])
    return pd.DataFrame(
        [
            {
                "status": "available_but_not_scored_in_v44",
                "path": str(cfg.checkpoint_live_path),
                "rows": int(len(frame)),
            }
        ]
    )


def _write_notes(root: Path, summary: pd.DataFrame, by_cic: pd.DataFrame, controls: pd.DataFrame, cp60: pd.DataFrame) -> None:
    lines = [
        "# v4.4 Binance Taker-Buy Fusion Expansion",
        "",
        "## Scope",
        "- Tests Binance taker-buy prior 5/15/30m as confirmation context for existing Bybit P2/CIC/O6 candidates.",
        "- This is fusion diagnostics only; no live selector is promoted here.",
        "",
        "## P2 Fusion",
    ]
    if summary.empty:
        lines.append("- No P2 fusion rows.")
    else:
        for window in sorted(summary["window_minutes"].dropna().unique()):
            rows = summary[summary["window_minutes"].eq(window)]
            yes = rows[rows["bucket"].eq("with_binance_taker_buy")]
            no = rows[rows["bucket"].eq("without_binance_taker_buy")]
            if yes.empty or no.empty:
                continue
            lines.append(
                f"- {int(window)}m: with={yes.iloc[0].net20:.4%} ({int(yes.iloc[0].trades)} trades), "
                f"without={no.iloc[0].net20:.4%} ({int(no.iloc[0].trades)} trades), "
                f"lift={yes.iloc[0].with_minus_without_net20:.4%}."
            )
    if not by_cic.empty:
        lines.extend(["", "## CIC Split"])
        best = by_cic[by_cic["bucket"].eq("with_binance_taker_buy")].sort_values("net20", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            lines.append(f"- Best with-taker CIC row: {row.scope} {int(row.window_minutes)}m net20={row.net20:.4%}, trades={int(row.trades)}.")
    if not controls.empty:
        lines.extend(["", "## Controls"])
        row = controls[controls["bucket"].eq("real_with_taker_buy")].sort_values("net20", ascending=False).head(1)
        if not row.empty:
            r = row.iloc[0]
            lines.append(
                f"- Best real control row {int(r.window_minutes)}m: net20={r.net20:.4%}, "
                f"real-vs-random={r.get('real_vs_random_with_taker_buy_lift', np.nan):.4%}, "
                f"real-vs-shuffled={r.get('real_vs_shuffled_with_taker_buy_lift', np.nan):.4%}."
            )
    if not cp60.empty:
        lines.extend(["", "## CP60"])
        lines.append(f"- CP60 fusion status: {cp60.iloc[0].get('status', 'unknown')}.")
    lines.extend(
        [
            "",
            "## Decision",
            "- Promote nothing unless with-minus-without is stable by CIC type, month, and random/shuffled controls.",
            "- If the effect remains only a small P2-level split, keep Binance taker-buy as diagnostic context.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v44_binance_taker_buy_fusion(cfg: V44Config = V44Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    symbols, _ = _select_common_symbols(_source_dir(cfg), _target_dir(cfg), cfg.top_n)  # type: ignore[arg-type]
    pool = _load_p2_trades(cfg.trade_cache_path)
    annotated, coverage = _annotate_taker_context(pool, symbols, cfg)
    covered = annotated[annotated.get("binance_source_covered", pd.Series(False, index=annotated.index)).astype(bool)].copy()
    summary, by_cic, o6 = _fusion_summary(covered, cfg)
    leave_one = _leave_one_month(covered, cfg)
    controls = _random_shuffled_controls(covered, cfg)
    cp60 = _cp60_summary(cfg)

    outputs = {
        "binance_taker_buy_fusion_summary": root / "binance_taker_buy_fusion_summary.csv",
        "p2_fusion_by_cic_type": root / "p2_fusion_by_cic_type.csv",
        "o6_fusion_summary": root / "o6_fusion_summary.csv",
        "cp60_fusion_summary": root / "cp60_fusion_summary.csv",
        "month_cap_leave_one_month": root / "month_cap_leave_one_month.csv",
        "random_shuffled_source_control": root / "random_shuffled_source_control.csv",
        "fusion_coverage": root / "fusion_coverage.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    summary.to_csv(outputs["binance_taker_buy_fusion_summary"], index=False)
    by_cic.to_csv(outputs["p2_fusion_by_cic_type"], index=False)
    o6.to_csv(outputs["o6_fusion_summary"], index=False)
    cp60.to_csv(outputs["cp60_fusion_summary"], index=False)
    leave_one.to_csv(outputs["month_cap_leave_one_month"], index=False)
    controls.to_csv(outputs["random_shuffled_source_control"], index=False)
    coverage.to_csv(outputs["fusion_coverage"], index=False)
    _write_notes(root, summary, by_cic, controls, cp60)
    return outputs


__all__ = ["V44Config", "write_v44_binance_taker_buy_fusion"]
