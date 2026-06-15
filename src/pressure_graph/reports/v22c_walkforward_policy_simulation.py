"""v2.2C walk-forward policy simulation for the pre-entry meta-router.

This report converts v2.2B probabilities into a small set of fixed action
policies and replays them through the B4 portfolio architecture. It is still
offline research only: no selector or shadow portfolio is promoted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import _simulate_portfolio
from pressure_graph.reports.v22b_preentry_meta_router import (
    BASE_SPEC,
    REPORT_ROOT as V22B_ROOT,
    _month_cap,
    _num,
    write_v22b_preentry_meta_router,
)


REPORT_ROOT = Path("reports/v2_2c_walkforward_policy_simulation")


@dataclass(frozen=True)
class V22CConfig:
    report_root: Path = REPORT_ROOT
    v22b_root: Path = V22B_ROOT
    denominator: int = BASE_SPEC.max_positions
    reduce_size_multiplier: float = 0.5


def _read_or_build_predictions(cfg: V22CConfig) -> pd.DataFrame:
    path = cfg.v22b_root / "policy_event_predictions.csv"
    if not path.exists():
        write_v22b_preentry_meta_router()
    if not path.exists():
        raise FileNotFoundError(f"Missing v2.2B predictions: {path}")
    out = pd.read_csv(path)
    for col in ("entry_time", "exit_time", "checkpoint_time"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out["entry_month"] = out["entry_month"].astype(str)
    return out


def _period_from_month(month: str) -> str:
    if month < "2026-02":
        return "search"
    if month < "2026-05":
        return "validation"
    return "holdout"


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    periods = frame["entry_month"].astype(str).map(_period_from_month)
    if period == "full":
        return pd.Series(True, index=frame.index)
    return periods.eq(period)


def _low_coimpulse_mask(frame: pd.DataFrame) -> pd.Series:
    market = _num(frame, "market_impulse_density")
    burst = _num(frame, "burst_count_so_far")
    peers = _num(frame, "same_timestamp_peer_count")
    return market.le(0.25) | burst.le(2) | peers.le(2)


def _policy_definitions(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    low = _low_coimpulse_mask(frame)
    return {
        "baseline_B4": {"kind": "baseline", "mask": pd.Series(False, index=frame.index)},
        "logistic_t70_skip": {"kind": "skip", "mask": frame["logistic_p_no_trade"].ge(0.70)},
        "logistic_t80_skip": {"kind": "skip", "mask": frame["logistic_p_no_trade"].ge(0.80)},
        "logistic_t70_reduce50": {"kind": "reduce", "mask": frame["logistic_p_no_trade"].ge(0.70)},
        "logistic_t80_reduce50": {"kind": "reduce", "mask": frame["logistic_p_no_trade"].ge(0.80)},
        "logistic_t70_skip_low_coimpulse": {
            "kind": "skip",
            "mask": frame["logistic_p_no_trade"].ge(0.70) & low,
        },
        "logistic_t80_skip_low_coimpulse": {
            "kind": "skip",
            "mask": frame["logistic_p_no_trade"].ge(0.80) & low,
        },
    }


def _ledger_metrics(ledger: pd.DataFrame, skipped: pd.DataFrame, denominator: int, router_affected: int) -> dict[str, Any]:
    if ledger.empty:
        return {
            "selected_trades": 0,
            "capacity_skipped_trades": int(len(skipped)),
            "router_affected_events": int(router_affected),
            "portfolio_net20": 0.0,
            "month_cap35_net20": np.nan,
            "worst_month_net20": np.nan,
            "worst_burst_net20": np.nan,
            "max_month_contribution": np.nan,
            "avg_exposure_weight": np.nan,
        }
    weighted = _num(ledger, "weighted_return")
    month_returns = weighted.groupby(ledger["entry_month"].astype(str), sort=False).sum() / denominator
    burst_returns = weighted.groupby(ledger["burst_id"].astype(str), sort=False).sum() / denominator
    positive = month_returns[month_returns.gt(0)]
    return {
        "selected_trades": int(len(ledger)),
        "capacity_skipped_trades": int(len(skipped)),
        "router_affected_events": int(router_affected),
        "portfolio_net20": float(weighted.sum() / denominator),
        "month_cap35_net20": _month_cap(month_returns),
        "worst_month_net20": float(month_returns.min()) if len(month_returns) else np.nan,
        "worst_burst_net20": float(burst_returns.min()) if len(burst_returns) else np.nan,
        "max_month_contribution": float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else np.nan,
        "avg_exposure_weight": float(_num(ledger, "exposure_weight").mean()),
    }


def _simulate_policy(frame: pd.DataFrame, policy: dict[str, Any], cfg: V22CConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mask = policy["mask"].reindex(frame.index).fillna(False).astype(bool)
    kind = str(policy["kind"])
    if kind == "skip":
        ledger, skipped = _simulate_portfolio(frame[~mask].copy(), BASE_SPEC)
        metrics = _ledger_metrics(ledger, skipped, cfg.denominator, int(mask.sum()))
        return ledger, skipped, metrics
    ledger, skipped = _simulate_portfolio(frame.copy(), BASE_SPEC)
    if kind == "reduce" and not ledger.empty:
        reduce_keys = set(frame.loc[mask, "trade_key"].astype(str))
        ledger = ledger.copy()
        selected = ledger["trade_key"].astype(str).isin(reduce_keys)
        ledger["router_reduced_size"] = selected
        ledger.loc[selected, "exposure_weight"] = (
            _num(ledger.loc[selected], "exposure_weight") * cfg.reduce_size_multiplier
        )
        ledger.loc[selected, "weighted_return"] = (
            _num(ledger.loc[selected], "weighted_return") * cfg.reduce_size_multiplier
        )
        affected = int(selected.sum())
    else:
        affected = int(mask.sum()) if kind != "baseline" else 0
    metrics = _ledger_metrics(ledger, skipped, cfg.denominator, affected)
    return ledger, skipped, metrics


def _evaluate(frame: pd.DataFrame, cfg: V22CConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_defs = _policy_definitions(frame)
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    baseline_by_period: dict[str, float] = {}
    baseline_monthly: dict[str, float] = {}
    for period in ("search", "validation", "holdout", "full"):
        part = frame[_period_mask(frame, period)].copy()
        _, _, metrics = _simulate_policy(part, policy_defs["baseline_B4"], cfg)
        baseline_by_period[period] = float(metrics["portfolio_net20"])
    for month, month_frame in frame.groupby("entry_month", sort=True):
        _, _, metrics = _simulate_policy(month_frame, policy_defs["baseline_B4"], cfg)
        baseline_monthly[str(month)] = float(metrics["portfolio_net20"])

    for policy_id, policy in policy_defs.items():
        full_ledger, _, full_metrics = _simulate_policy(frame, policy, cfg)
        row = {"policy_id": policy_id, "policy_kind": policy["kind"], **full_metrics}
        row["delta_vs_baseline_net20"] = row["portfolio_net20"] - baseline_by_period["full"]
        if not full_ledger.empty:
            full_ledger = full_ledger.copy()
            full_ledger["policy_id"] = policy_id
            ledgers.append(full_ledger)
        for period in ("search", "validation", "holdout"):
            part = frame[_period_mask(frame, period)].copy()
            _, _, metrics = _simulate_policy(part, policy, cfg)
            row[f"{period}_portfolio_net20"] = metrics["portfolio_net20"]
            row[f"{period}_delta_vs_baseline_net20"] = metrics["portfolio_net20"] - baseline_by_period[period]
        for month, month_frame in frame.groupby("entry_month", sort=True):
            _, _, metrics = _simulate_policy(month_frame, policy, cfg)
            monthly_rows.append(
                {
                    "policy_id": policy_id,
                    "entry_month": month,
                    "period": _period_from_month(str(month)),
                    **metrics,
                    "baseline_month_net20": baseline_monthly[str(month)],
                    "delta_vs_baseline_net20": metrics["portfolio_net20"] - baseline_monthly[str(month)],
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["validation_delta_vs_baseline_net20", "holdout_delta_vs_baseline_net20", "delta_vs_baseline_net20"],
        ascending=[False, False, False],
    )
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return summary, pd.DataFrame(monthly_rows), ledger


def _slice_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "low_coimpulse": _low_coimpulse_mask(frame),
        "normal_or_high_coimpulse": ~_low_coimpulse_mask(frame),
        "high_router_risk_p70": frame["logistic_p_no_trade"].ge(0.70),
        "high_router_risk_p80": frame["logistic_p_no_trade"].ge(0.80),
    }
    for name, mask in masks.items():
        part = frame[mask].copy()
        if part.empty:
            continue
        net = _num(part, "net20")
        rows.append(
            {
                "slice": name,
                "events": int(len(part)),
                "net20_sum": float(net.sum()),
                "net20_avg": float(net.mean()),
                "no_trade_labels": int(part["pre_entry_action_label"].eq("no_trade").sum()),
                "core_trade_labels": int(part["pre_entry_action_label"].eq("core_trade").sum()),
                "holdout_events": int(part["entry_month"].astype(str).map(_period_from_month).eq("holdout").sum()),
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# v2.2C Walk-forward Policy Simulation",
        "",
        "Status: offline policy conversion audit only. No live/shadow selector is promoted.",
        "",
        "## Policies",
    ]
    for row in summary.head(8).itertuples(index=False):
        lines.append(
            f"- {row.policy_id}: full_delta={row.delta_vs_baseline_net20:.4%}, "
            f"validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
            f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, "
            f"affected={row.router_affected_events}."
        )
    promotable = summary[
        ~summary["policy_id"].eq("baseline_B4")
        & summary["delta_vs_baseline_net20"].gt(0)
        & summary["validation_delta_vs_baseline_net20"].gt(0)
        & summary["holdout_delta_vs_baseline_net20"].ge(0)
    ]
    lines.extend(["", "## Decision"])
    if promotable.empty:
        lines.append("- No fixed action policy passed the basic full/validation/holdout guardrail.")
    else:
        lines.append("- Some policies pass the basic guardrail, but v2.2D/E must decide threshold and control robustness.")
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22c_walkforward_policy_simulation(cfg: V22CConfig = V22CConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    frame = _read_or_build_predictions(cfg)
    summary, monthly, ledger = _evaluate(frame, cfg)
    slices = _slice_summary(frame)
    outputs = {
        "policy_vs_benchmark": root / "policy_vs_benchmark.csv",
        "policy_monthly_performance": root / "policy_monthly_performance.csv",
        "policy_action_ledger": root / "policy_action_ledger.csv",
        "low_coimpulse_slice_summary": root / "low_coimpulse_slice_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    summary.to_csv(outputs["policy_vs_benchmark"], index=False)
    monthly.to_csv(outputs["policy_monthly_performance"], index=False)
    ledger.to_csv(outputs["policy_action_ledger"], index=False)
    slices.to_csv(outputs["low_coimpulse_slice_summary"], index=False)
    _notes(root, summary)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22CConfig",
    "write_v22c_walkforward_policy_simulation",
]
