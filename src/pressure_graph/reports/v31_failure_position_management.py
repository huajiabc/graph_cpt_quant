"""v3.1 Failure-Aware Position Management.

v3.0 showed same-symbol no-long risk-off helps raw P2/O6 but is too coarse for
the managed CP60/Protect_A stack. This report asks the next question: when a
failure motif appears while a long is already open, should that position be
pruned immediately or only if it has not worked?
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v12s2_long_risk_off_overlay import RiskOffConfig, stream_risk_off_events
from pressure_graph.reports.v13a_checkpoint_robustness import CORE_MAX_POSITIONS
from pressure_graph.reports.v13d_cp60_context_protection import _portfolio_summary
from pressure_graph.reports.v13e_cp60_beta_protection_stability import (
    _cp60_sample,
    _portfolio_net,
    _prepare_sample_at_cost,
    _simulate_max8_o6,
)
from pressure_graph.reports.v30_symbol_risk_off_overlay import (
    FOCAL_COST_BPS,
    V30Config,
    _o6_summary,
    _prepare_hold_sample,
    _prepare_protect_a_cap2_sample,
    _simulate_architecture,
)


REPORT_ROOT = Path("reports/v3_1_failure_position_management")
PRICE_COLUMNS = ("symbol", "feature_time", "close", "high", "low")


@dataclass(frozen=True)
class V31Config:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    top_n: int = 30
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    symbol_cooldown_bars: int = 48
    risk_event_cache_path: Path = Path("reports/v3_0_symbol_risk_off_overlay/risk_off_events.csv")


def _risk_cfg(cfg: V31Config) -> RiskOffConfig:
    return RiskOffConfig(
        report_root=cfg.report_root,
        top_n=cfg.top_n,
        motifs=cfg.motifs,
        symbol_cooldown_bars=cfg.symbol_cooldown_bars,
    )


def _load_or_stream_events(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V31Config,
) -> pd.DataFrame:
    if cfg.risk_event_cache_path.exists():
        events = pd.read_csv(cfg.risk_event_cache_path)
        if not events.empty:
            events["feature_time"] = pd.to_datetime(events["feature_time"], utc=True, errors="coerce")
            return events.dropna(subset=["feature_time"]).sort_values("feature_time").reset_index(drop=True)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    return stream_risk_off_events(feature_path, rank30, rank90, symbols, config, _risk_cfg(cfg))


def _load_position_price_frame(feature_path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(set(sample["symbol"].dropna().astype(str)))
    start = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce").min() - pd.Timedelta("30min")
    end = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce").max() + pd.Timedelta("30min")
    prices = pd.read_parquet(feature_path, columns=list(PRICE_COLUMNS))
    prices["feature_time"] = pd.to_datetime(prices["feature_time"], utc=True, errors="coerce")
    prices = prices[
        prices["symbol"].astype(str).isin(symbols)
        & prices["feature_time"].ge(start)
        & prices["feature_time"].le(end)
    ].copy()
    for col in ("close", "high", "low"):
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices.dropna(subset=["symbol", "feature_time", "close"]).sort_values(["symbol", "feature_time"])


def _attach_first_failure_event(sample: pd.DataFrame, events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["failure_event_during_position"] = False
    out["failure_event_time"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["failure_event_motif"] = ""
    out["failure_event_price"] = np.nan
    out["failure_event_net_at_cost"] = np.nan
    out["failure_event_delta_vs_keep"] = np.nan
    if out.empty or events.empty:
        return out

    events = events.copy()
    events["feature_time"] = pd.to_datetime(events["feature_time"], utc=True, errors="coerce")
    events = events.dropna(subset=["feature_time"]).sort_values(["symbol", "feature_time"])
    prices = prices.copy().sort_values(["symbol", "feature_time"])

    by_symbol_events = {
        str(sym): group.reset_index(drop=True)
        for sym, group in events.groupby("symbol", sort=False, dropna=False)
    }
    by_symbol_prices = {
        str(sym): group.reset_index(drop=True)
        for sym, group in prices.groupby("symbol", sort=False, dropna=False)
    }

    for idx, row in out.iterrows():
        symbol = str(row.get("symbol", ""))
        symbol_events = by_symbol_events.get(symbol)
        symbol_prices = by_symbol_prices.get(symbol)
        if symbol_events is None or symbol_events.empty or symbol_prices is None or symbol_prices.empty:
            continue
        entry = pd.Timestamp(row["entry_time"])
        planned_exit = pd.Timestamp(row["effective_exit_time"])
        event_times = pd.to_datetime(symbol_events["feature_time"], utc=True, errors="coerce")
        mask = event_times.gt(entry) & event_times.le(planned_exit)
        if not mask.any():
            continue
        event = symbol_events[mask].iloc[0]
        event_time = pd.Timestamp(event["feature_time"])
        price_window = symbol_prices[pd.to_datetime(symbol_prices["feature_time"], utc=True, errors="coerce").le(event_time)]
        if price_window.empty:
            continue
        price = float(price_window.iloc[-1]["close"])
        entry_price = float(pd.to_numeric(row.get("entry_price", np.nan), errors="coerce"))
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        cost_bps = float(pd.to_numeric(row.get("cost_single_side_bps", FOCAL_COST_BPS), errors="coerce"))
        failure_net = price / entry_price - 1.0 - 2.0 * cost_bps / 10_000.0
        keep_net = float(pd.to_numeric(row.get("effective_net_return", np.nan), errors="coerce"))
        out.loc[idx, "failure_event_during_position"] = True
        out.loc[idx, "failure_event_time"] = event_time
        out.loc[idx, "failure_event_motif"] = str(event.get("motif", ""))
        out.loc[idx, "failure_event_price"] = price
        out.loc[idx, "failure_event_net_at_cost"] = failure_net
        out.loc[idx, "failure_event_delta_vs_keep"] = failure_net - keep_net
    return out


def _apply_failure_position_rule(sample: pd.DataFrame, rule_name: str, mode: str) -> pd.DataFrame:
    out = sample.copy()
    event = out["failure_event_during_position"].fillna(False).astype(bool)
    event_net = pd.to_numeric(out["failure_event_net_at_cost"], errors="coerce")
    if mode == "none":
        trigger = pd.Series(False, index=out.index)
    elif mode == "exit_any":
        trigger = event
    elif mode == "exit_if_net_lte_0":
        trigger = event & event_net.le(0.0)
    elif mode == "exit_if_net_lte_minus_0p5":
        trigger = event & event_net.le(-0.005)
    else:
        raise KeyError(mode)
    out["failure_position_rule"] = rule_name
    out["failure_position_exit"] = trigger
    out["pre_failure_effective_exit_time"] = out["effective_exit_time"]
    out["pre_failure_effective_net_return"] = out["effective_net_return"]
    out.loc[trigger, "effective_exit_time"] = out.loc[trigger, "failure_event_time"]
    out.loc[trigger, "effective_net_return"] = out.loc[trigger, "failure_event_net_at_cost"]
    out["failure_exit_delta_vs_keep"] = np.where(
        trigger,
        pd.to_numeric(out["failure_event_net_at_cost"], errors="coerce")
        - pd.to_numeric(out["pre_failure_effective_net_return"], errors="coerce"),
        0.0,
    )
    out["effective_holding_minutes"] = (
        pd.to_datetime(out["effective_exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out


def _summary_with_failure(
    structure_id: str,
    label: str,
    sample: pd.DataFrame,
    *,
    use_o6: bool,
    baseline_net: float | None = None,
    baseline_dd: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    ledger, skipped, summary = _simulate_architecture(sample, structure_id, use_o6=use_o6)
    failure_exits = (
        ledger.get("failure_position_exit", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    delta = pd.to_numeric(ledger.get("failure_exit_delta_vs_keep", pd.Series(0.0, index=ledger.index)), errors="coerce")
    row = {
        "structure_id": structure_id,
        "structure_label": label,
        "selected_trades": int(summary.get("selected_trades", 0)),
        "skipped_trades": int(summary.get("skipped_trades", 0)),
        "portfolio_net20": float(summary.get("portfolio_net20", np.nan)),
        "delta_net20_vs_baseline": float(summary.get("portfolio_net20", np.nan) - baseline_net)
        if baseline_net is not None
        else np.nan,
        "max_drawdown_proxy": float(summary.get("max_drawdown_proxy", np.nan)),
        "delta_drawdown_vs_baseline": float(summary.get("max_drawdown_proxy", np.nan) - baseline_dd)
        if baseline_dd is not None
        else np.nan,
        "worst_month": float(summary.get("worst_month", np.nan)),
        "worst_burst": float(summary.get("worst_burst", np.nan)),
        "cp60_exits_executed": int(summary.get("cp60_exits_executed", 0)),
        "protected_cp60_exits": int(summary.get("protected_cp60_exits", 0)),
        "overflow_trades": int(summary.get("overflow_trades", 0)),
        "failure_events_during_selected": int(
            ledger.get("failure_event_during_position", pd.Series(False, index=ledger.index)).fillna(False).astype(bool).sum()
        )
        if not ledger.empty
        else 0,
        "failure_position_exits": int(failure_exits.sum()) if len(failure_exits) else 0,
        "failure_exit_delta_vs_keep_sum": float(delta[failure_exits].sum()) if len(failure_exits) else 0.0,
        "failure_exit_delta_vs_keep_avg": float(delta[failure_exits].mean()) if len(failure_exits) and failure_exits.any() else np.nan,
    }
    return row, ledger, skipped


def _run_position_management(base_sample: pd.DataFrame, events: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = _attach_first_failure_event(_prepare_hold_sample(base_sample), events, prices)
    cp60 = _attach_first_failure_event(_cp60_sample(base_sample), events, prices)
    protect = _attach_first_failure_event(_prepare_protect_a_cap2_sample(base_sample), events, prices)

    baseline_sample = _apply_failure_position_rule(protect, "B3_no_failure_exit", "none")
    baseline_row, baseline_ledger, _ = _summary_with_failure(
        "B3_P2_MAX8_PROTECT_A_CAP2_O6",
        "P2 max8 + Protect_A cap2 + O6",
        baseline_sample,
        use_o6=True,
    )
    baseline_net = float(baseline_row["portfolio_net20"])
    baseline_dd = float(baseline_row["max_drawdown_proxy"])

    rows = [baseline_row]
    ledgers = [baseline_ledger.assign(position_management_rule="B3_no_failure_exit")]
    specs = [
        ("PM1_hold_exit_any", hold, "exit_any", False, "P2 max8 + failure exit any"),
        ("PM2_cp60_o6_exit_any", cp60, "exit_any", True, "P2 max8 + CP60 + O6 + failure exit any"),
        ("PM3_protect_o6_exit_any", protect, "exit_any", True, "P2 max8 + Protect_A cap2 + O6 + failure exit any"),
        ("PM4_protect_o6_exit_if_net_lte_0", protect, "exit_if_net_lte_0", True, "P2 max8 + Protect_A cap2 + O6 + failure exit if net<=0"),
        ("PM5_protect_o6_exit_if_net_lte_minus_0p5", protect, "exit_if_net_lte_minus_0p5", True, "P2 max8 + Protect_A cap2 + O6 + failure exit if net<=-0.5%"),
    ]
    for structure_id, source, mode, use_o6, label in specs:
        local = _apply_failure_position_rule(source, structure_id, mode)
        row, ledger, _ = _summary_with_failure(
            structure_id,
            label,
            local,
            use_o6=use_o6,
            baseline_net=baseline_net,
            baseline_dd=baseline_dd,
        )
        rows.append(row)
        if not ledger.empty:
            ledgers.append(ledger.assign(position_management_rule=structure_id))
    ledger_all = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return pd.DataFrame(rows), ledger_all


def _failure_exit_ledger(ledger_all: pd.DataFrame) -> pd.DataFrame:
    if ledger_all.empty:
        return pd.DataFrame()
    mask = ledger_all.get("failure_position_exit", pd.Series(False, index=ledger_all.index)).fillna(False).astype(bool)
    out = ledger_all[mask].copy()
    preferred = [
        "position_management_rule",
        "signal_id",
        "trade_key",
        "symbol",
        "candidate",
        "entry_time",
        "failure_event_time",
        "failure_event_motif",
        "failure_event_net_at_cost",
        "pre_failure_effective_net_return",
        "failure_exit_delta_vs_keep",
        "sleeve",
        "exposure_weight",
        "burst_id",
        "burst_count_so_far",
    ]
    cols = [col for col in preferred if col in out.columns]
    return out[cols + [col for col in out.columns if col not in cols]]


def _write_notes(root: Path, summary: pd.DataFrame, failure_ledger: pd.DataFrame) -> None:
    lines = [
        "# v3.1 Failure-Aware Position Management",
        "",
        "Purpose: test whether same-symbol failure motifs should manage already-open CIC longs.",
        "Status: offline audit only. No live, paper-live, or real-live rule changes.",
        "",
        "## Summary",
    ]
    if summary.empty:
        lines.append("- No summary rows.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.structure_id}: net20={row.portfolio_net20:.4%}, "
                f"delta={row.delta_net20_vs_baseline:+.4%}, "
                f"dd_delta={row.delta_drawdown_vs_baseline:+.4%}, "
                f"failure_exits={row.failure_position_exits}, "
                f"exit_delta_sum={row.failure_exit_delta_vs_keep_sum:.4%}."
            )
    lines.extend(["", "## Failure Exit Attribution"])
    if failure_ledger.empty:
        lines.append("- No failure exits triggered.")
    else:
        by_rule = failure_ledger.groupby("position_management_rule")["failure_exit_delta_vs_keep"].agg(["count", "sum", "mean"]).reset_index()
        for row in by_rule.itertuples(index=False):
            lines.append(
                f"- {row.position_management_rule}: exits={int(row.count)}, "
                f"delta_sum={row.sum:.4%}, delta_avg={row.mean:.4%}."
            )
    lines.extend(["", "## Interpretation"])
    managed = summary[summary["structure_id"].astype(str).str.contains("protect_o6", case=False, na=False)]
    if not managed.empty:
        best_delta = float(managed["delta_net20_vs_baseline"].max())
        best_dd = float(managed["delta_drawdown_vs_baseline"].max())
        lines.append(
            f"- best Protect_A-stack failure-exit net delta={best_delta:+.4%}; "
            f"best drawdown delta={best_dd:+.4%}."
        )
        if best_delta < 0 and best_dd > 0:
            lines.append(
                "- Current verdict: failure-aware immediate exit behaves like a drawdown brake, "
                "but it is not a valid portfolio upgrade because it sacrifices too much continuation PnL."
            )
            lines.append(
                "- Next best use: convert failure events into action labels / atlas rows, not a hard "
                "exit rule. Likely best actions remain no-long cooldown or stricter diagnostics."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Exit is strict post-entry as-of: only failure events after entry_time and before the current planned effective exit can act.",
            "- Exit price uses the latest feature close at or before the failure confirmation time.",
            "- This report tests position management, not new short entries.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v31_failure_position_management(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V31Config = V31Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, FOCAL_COST_BPS)
    events = _load_or_stream_events(feature_path, instruments, config, cfg)
    prices = _load_position_price_frame(feature_path, sample)
    summary, ledger_all = _run_position_management(sample, events, prices)
    failure_ledger = _failure_exit_ledger(ledger_all)
    outputs = {
        "position_management_summary": root / "position_management_summary.csv",
        "failure_position_ledger": root / "failure_position_ledger.csv",
        "failure_exit_ledger": root / "failure_exit_ledger.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    summary.to_csv(outputs["position_management_summary"], index=False)
    ledger_all.to_csv(outputs["failure_position_ledger"], index=False)
    failure_ledger.to_csv(outputs["failure_exit_ledger"], index=False)
    _write_notes(root, summary, failure_ledger)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V31Config",
    "write_v31_failure_position_management",
]
