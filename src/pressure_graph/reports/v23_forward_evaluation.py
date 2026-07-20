"""v2.3 Forward Evaluation & Decision Ledger.

This report turns the current paper-live / shadow ledgers into a forward lab.
It does not fit a model, change a portfolio rule, or promote any live action.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet


REPORT_ROOT = Path("reports/v2_3_forward_evaluation_decision_ledger")
SOURCE_ROOT = Path("reports/v0_7d2_cic_mir1_paper_live")


@dataclass(frozen=True)
class V23Config:
    report_root: Path = REPORT_ROOT
    source_root: Path = SOURCE_ROOT
    low_coimpulse_quantile: float = 0.80
    router_high_risk_threshold: float = 0.80
    neutral_delta_abs: float = 0.001


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(default, index=frame.index)), errors="coerce")


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    series = frame.get(col, pd.Series(False, index=frame.index))
    if series.dtype == object:
        text = series.astype(str).str.lower()
        return text.isin(["true", "1", "yes"])
    return series.where(series.notna(), False).astype(bool)


def _dt(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(frame.get(col, pd.Series(pd.NaT, index=frame.index)), utc=True, errors="coerce")


def _weighted_sum(frame: pd.DataFrame, col: str) -> float:
    if frame.empty:
        return 0.0
    weights = _num(frame, "position_size", 1.0).fillna(1.0)
    values = _num(frame, col, 0.0).fillna(0.0)
    return float((values * weights).sum())


def _portfolio_net(frame: pd.DataFrame, col: str, denominator: float = 8.0) -> float:
    return _weighted_sum(frame, col) / denominator if denominator > 0 else _weighted_sum(frame, col)


def _mean(frame: pd.DataFrame, col: str) -> float:
    values = _num(frame, col)
    return float(values.mean()) if values.notna().any() else np.nan


def _as_event_id(frame: pd.DataFrame) -> pd.Series:
    if "signal_id" in frame.columns:
        return frame["signal_id"].astype(str)
    if "trade_id" in frame.columns:
        return frame["trade_id"].astype(str)
    return pd.Series(frame.index.astype(str), index=frame.index)


def _live_architecture_summary(checkpoint: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "structure",
        "trades",
        "net10",
        "net20",
        "net30",
        "core_pnl",
        "overflow_pnl",
        "checkpoint_pnl",
        "protect_counterfactual_pnl",
        "max_exposure",
        "max_concurrent_positions",
        "checkpoint_exits",
        "protected_exits",
        "overflow_trades",
    ]
    if checkpoint.empty:
        return pd.DataFrame(columns=columns)
    data = checkpoint.copy()
    data["entry_date"] = _dt(data, "entry_time").dt.strftime("%Y-%m-%d").fillna("unknown")
    if "portfolio_id" not in data.columns:
        data["portfolio_id"] = "unknown"
    rows: list[dict[str, Any]] = []
    for (date, portfolio_id), group in data.groupby(["entry_date", "portfolio_id"], sort=True, dropna=False):
        selected = group[_bool(group, "selected")] if "selected" in group.columns else group
        core = selected[_bool(selected, "is_core")]
        overflow = selected[_bool(selected, "is_overflow")]
        checkpoint_delta = (
            _num(selected, "effective_net_return_20bp", 0.0).fillna(0.0)
            - _num(selected, "net_if_kept_counterfactual_20bp", 0.0).fillna(0.0)
        )
        checkpoint_pnl = float((checkpoint_delta * _bool(selected, "checkpoint_triggered").astype(float)).sum())
        protected = selected[_bool(selected, "protected_by_beta_high")]
        exposure = _num(selected, "concurrent_positions", 0.0).fillna(0.0) + _num(selected, "position_size", 1.0).fillna(1.0)
        rows.append(
            {
                "date": date,
                "structure": str(portfolio_id),
                "trades": int(len(selected)),
                "net10": _portfolio_net(selected, "effective_net_return_10bp"),
                "net20": _portfolio_net(selected, "effective_net_return_20bp"),
                "net30": _portfolio_net(selected, "effective_net_return_30bp"),
                "core_pnl": _portfolio_net(core, "effective_net_return_20bp"),
                "overflow_pnl": _portfolio_net(overflow, "effective_net_return_20bp"),
                "checkpoint_pnl": checkpoint_pnl / 8.0,
                "protect_counterfactual_pnl": _portfolio_net(protected, "delta_vs_cp60"),
                "max_exposure": float(exposure.max()) if len(exposure) else 0.0,
                "max_concurrent_positions": float(_num(selected, "concurrent_positions", 0.0).max()) if len(selected) else 0.0,
                "checkpoint_exits": int(_bool(selected, "checkpoint_triggered").sum()),
                "protected_exits": int(_bool(selected, "protected_by_beta_high").sum()),
                "overflow_trades": int(_bool(selected, "is_overflow").sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0)) if numeric.notna().any() else 0.0
    if std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (numeric.fillna(numeric.mean()) - float(numeric.mean())) / std


def _regime_diagnostics(checkpoint: pd.DataFrame, router: pd.DataFrame, cfg: V23Config) -> pd.DataFrame:
    columns = [
        "event_id",
        "symbol",
        "market_impulse_density",
        "same_timestamp_peer_count",
        "burst_count_so_far",
        "cluster_density",
        "low_coimpulse_score",
        "logistic_no_trade_prob",
        "risk_state",
        "net20_later",
    ]
    base = checkpoint[checkpoint.get("portfolio_id", pd.Series(dtype=str)).astype(str).eq("P2_MAX8_BASELINE")].copy()
    if base.empty:
        return pd.DataFrame(columns=columns)
    base["event_id"] = _as_event_id(base)
    base["entry_time_norm"] = _dt(base, "entry_time")
    base["same_timestamp_peer_count"] = base.groupby("entry_time_norm")["event_id"].transform("count")
    out = pd.DataFrame(
        {
            "event_id": base["event_id"],
            "symbol": base.get("symbol", pd.Series("", index=base.index)).astype(str),
            "market_impulse_density": _num(base, "volume_impulse_density_at_entry"),
            "same_timestamp_peer_count": _num(base, "same_timestamp_peer_count"),
            "burst_count_so_far": _num(base, "burst_count_so_far"),
            "cluster_density": _num(base, "cluster_impulse_density_at_entry"),
            "net20_later": _num(base, "net_return_20bp"),
        }
    )
    if not router.empty:
        router_small = router[["signal_id", "logistic_no_trade_prob"]].copy() if {"signal_id", "logistic_no_trade_prob"}.issubset(router.columns) else pd.DataFrame()
        if not router_small.empty:
            out = out.merge(router_small.rename(columns={"signal_id": "event_id"}), on="event_id", how="left")
    if "logistic_no_trade_prob" not in out.columns:
        out["logistic_no_trade_prob"] = np.nan
    components = [
        _zscore(out["market_impulse_density"]),
        _zscore(out["same_timestamp_peer_count"]),
        _zscore(out["burst_count_so_far"]),
        _zscore(out["cluster_density"]),
    ]
    out["low_coimpulse_score"] = -sum(components)
    cutoff = out["low_coimpulse_score"].quantile(cfg.low_coimpulse_quantile) if out["low_coimpulse_score"].notna().any() else np.inf
    high_router = pd.to_numeric(out["logistic_no_trade_prob"], errors="coerce").ge(cfg.router_high_risk_threshold)
    low_coimpulse = out["low_coimpulse_score"].ge(float(cutoff))
    out["risk_state"] = np.select(
        [high_router, low_coimpulse],
        ["router_high_risk", "low_coimpulse_high"],
        default="normal_coimpulse",
    )
    return out.reindex(columns=columns)


def _cp60_live_attribution(checkpoint: pd.DataFrame, cfg: V23Config) -> pd.DataFrame:
    columns = [
        "trade_id",
        "checkpoint_time",
        "checkpoint_net",
        "cp60_exit",
        "net_if_exited",
        "net_if_kept",
        "delta_vs_keep",
        "true_good_exit",
        "false_exit",
        "neutral_exit",
        "beta_high",
        "protect_a_would_keep",
    ]
    data = checkpoint[checkpoint.get("portfolio_id", pd.Series(dtype=str)).astype(str).eq("P2_MAX8_CP60")].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    net_exit = _num(data, "net_if_checkpoint_exit_20bp")
    net_keep = _num(data, "net_if_kept_counterfactual_20bp")
    delta = net_exit - net_keep
    cp60_exit = _bool(data, "checkpoint_triggered")
    beta_high = _bool(data, "beta_high_protection")
    out = pd.DataFrame(
        {
            "trade_id": data.get("trade_id", pd.Series("", index=data.index)).astype(str),
            "checkpoint_time": data.get("checkpoint_time", pd.Series("", index=data.index)).astype(str),
            "checkpoint_net": _num(data, "checkpoint_net20"),
            "cp60_exit": cp60_exit,
            "net_if_exited": net_exit,
            "net_if_kept": net_keep,
            "delta_vs_keep": delta,
            "true_good_exit": cp60_exit & delta.gt(cfg.neutral_delta_abs),
            "false_exit": cp60_exit & delta.lt(-cfg.neutral_delta_abs),
            "neutral_exit": cp60_exit & delta.abs().le(cfg.neutral_delta_abs),
            "beta_high": beta_high,
            "protect_a_would_keep": cp60_exit & beta_high,
        }
    )
    return out.reindex(columns=columns)


def _overflow_live_attribution(overflow: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_id",
        "burst_count_so_far",
        "overflow_triggered",
        "overflow_size",
        "overflow_net20",
        "incremental_vs_core",
        "extra_exposure",
    ]
    if overflow.empty:
        return pd.DataFrame(columns=columns)
    triggered = _bool(overflow, "is_overflow")
    net20 = _num(overflow, "net_return_20bp")
    size = _num(overflow, "position_size", 1.0).fillna(1.0)
    out = pd.DataFrame(
        {
            "trade_id": overflow.get("trade_id", pd.Series("", index=overflow.index)).astype(str),
            "burst_count_so_far": _num(overflow, "burst_count_so_far"),
            "overflow_triggered": triggered,
            "overflow_size": np.where(triggered, size, 0.0),
            "overflow_net20": np.where(triggered, net20, np.nan),
            "incremental_vs_core": np.where(triggered, net20 * size, 0.0),
            "extra_exposure": _num(overflow, "extra_exposure", 0.0).fillna(0.0),
        }
    )
    return out.reindex(columns=columns)


def _protect_a_counterfactual_live(checkpoint: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_id",
        "would_protect",
        "protected_reason",
        "cp60_exit_net",
        "keep_counterfactual_net",
        "delta",
        "slot_blocked",
        "missed_trade",
    ]
    data = checkpoint[checkpoint.get("portfolio_id", pd.Series(dtype=str)).astype(str).eq("P2_MAX8_CP60_PROTECT_A_CAP2")].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    would = _bool(data, "cp60_would_exit") & _bool(data, "beta_high_protection")
    out = pd.DataFrame(
        {
            "trade_id": data.get("trade_id", pd.Series("", index=data.index)).astype(str),
            "would_protect": would,
            "protected_reason": np.where(_bool(data, "protected_by_beta_high"), "beta_high_cap2", np.where(would, "beta_high_cap_reached_or_not_selected", "")),
            "cp60_exit_net": _num(data, "counterfactual_cp60_exit_net20").fillna(_num(data, "net_if_checkpoint_exit_20bp")),
            "keep_counterfactual_net": _num(data, "actual_keep_exit_net20").fillna(_num(data, "net_if_kept_counterfactual_20bp")),
            "delta": _num(data, "delta_vs_cp60"),
            "slot_blocked": _num(data, "slot_blocked_minutes", 0.0).fillna(0.0),
            "missed_trade": _bool(data, "missed_trade_due_to_protection"),
        }
    )
    return out.reindex(columns=columns)


def _notes(root: Path, tables: dict[str, pd.DataFrame]) -> None:
    arch = tables.get("live_architecture_summary", pd.DataFrame())
    regime = tables.get("live_regime_diagnostics", pd.DataFrame())
    cp60 = tables.get("cp60_live_attribution", pd.DataFrame())
    protect = tables.get("protect_a_counterfactual_live", pd.DataFrame())
    lines = [
        "# v2.3 Forward Evaluation & Decision Ledger",
        "",
        "Status: forward evaluation only. No selector, checkpoint rule, overflow sleeve, paper-live primary, or real-live permission is changed.",
        "",
        "## Coverage",
        f"- architecture rows: {len(arch)}",
        f"- regime diagnostic events: {len(regime)}",
        f"- cp60 attribution rows: {len(cp60)}",
        f"- protect_a counterfactual rows: {len(protect)}",
        "",
        "## Decision Use",
        "- Use live_architecture_summary.csv to compare P2 / O6 / CP60 / Protect_A structures over future samples.",
        "- Use live_regime_diagnostics.csv to test whether low-coimpulse and logistic high-risk states remain weak out-of-sample.",
        "- Use cp60_live_attribution.csv to split true-good exits, false exits, and neutral exits.",
        "- Use overflow_live_attribution.csv to evaluate O6 incremental net per extra exposure.",
        "- Use protect_a_counterfactual_live.csv to monitor beta-high cap2 protection without changing CP60_all.",
    ]
    if not regime.empty and "risk_state" in regime.columns:
        risk = regime.groupby("risk_state", dropna=False)["net20_later"].agg(["count", "mean"]).reset_index()
        lines.extend(["", "## Current Risk-State Snapshot"])
        for row in risk.itertuples(index=False):
            lines.append(f"- {row.risk_state}: events={int(row.count)}, net20_avg={row.mean:.4%}.")
    root.joinpath("candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v23_forward_evaluation(cfg: V23Config = V23Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    forward_checkpoint = cfg.source_root / "forward" / "checkpoint_trades.parquet"
    forward_overflow = cfg.source_root / "forward" / "overflow_trades.parquet"
    checkpoint = _read_parquet(
        forward_checkpoint if forward_checkpoint.exists() else cfg.source_root / "checkpoint_trade_ledger.parquet"
    )
    overflow = _read_parquet(
        forward_overflow if forward_overflow.exists() else cfg.source_root / "overflow_trade_ledger.parquet"
    )
    if forward_checkpoint.exists() and "timely_forward_observation" in checkpoint.columns:
        checkpoint = checkpoint[checkpoint["timely_forward_observation"].fillna(False).astype(bool)].copy()
    if forward_overflow.exists() and "timely_forward_observation" in overflow.columns:
        overflow = overflow[overflow["timely_forward_observation"].fillna(False).astype(bool)].copy()
    router = _read_parquet(cfg.source_root / "pre_entry_router_counterfactual_live.parquet")
    if router.empty:
        router = _read_csv(cfg.source_root / "pre_entry_router_counterfactual_live.csv")

    tables = {
        "live_architecture_summary": _live_architecture_summary(checkpoint),
        "live_regime_diagnostics": _regime_diagnostics(checkpoint, router, cfg),
        "cp60_live_attribution": _cp60_live_attribution(checkpoint, cfg),
        "overflow_live_attribution": _overflow_live_attribution(overflow),
        "protect_a_counterfactual_live": _protect_a_counterfactual_live(checkpoint),
    }
    outputs = {
        "live_architecture_summary": root / "live_architecture_summary.csv",
        "live_regime_diagnostics": root / "live_regime_diagnostics.csv",
        "cp60_live_attribution": root / "cp60_live_attribution.csv",
        "overflow_live_attribution": root / "overflow_live_attribution.csv",
        "protect_a_counterfactual_live": root / "protect_a_counterfactual_live.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    for name, frame in tables.items():
        frame.to_csv(outputs[name], index=False)
    _notes(root, tables)
    return outputs


__all__ = ["V23Config", "write_v23_forward_evaluation"]
