from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.config.v07a2 import V07A2Config
from pressure_graph.io import ensure_dir, write_parquet


@dataclass(frozen=True)
class LiveGateDecision:
    allow_new_actions: bool
    reasons: tuple[str, ...]
    data_stale: bool
    latest_feature_time: str
    rolling_window: int
    primary_completed_trades: int
    rolling_primary_net10: float | None
    matched_random_completed_trades: int
    rolling_matched_random_net10: float | None
    rolling_baseline_lift10: float | None
    rolling_net_gate_status: str
    baseline_lift_gate_status: str


def _utc(value: object | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _completed_timely(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "timely_forward_observation" in out.columns:
        out = out[out["timely_forward_observation"].fillna(False).astype(bool)]
    if "exit_reason" in out.columns:
        out = out[~out["exit_reason"].fillna("").astype(str).eq("open")]
    time_col = "exit_time" if "exit_time" in out.columns else "entry_time"
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out = out.sort_values(time_col)
    return out


def _mean_tail(frame: pd.DataFrame, col: str, window: int) -> float | None:
    if len(frame) < window or col not in frame.columns:
        return None
    values = pd.to_numeric(frame.tail(window)[col], errors="coerce").dropna()
    if len(values) < window:
        return None
    return float(values.mean())


def evaluate_live_gates(
    prepared: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V07A2Config,
    *,
    now: object | None = None,
) -> LiveGateDecision:
    checked_at = _utc(now)
    feature_time = pd.to_datetime(prepared.get("feature_time"), utc=True, errors="coerce")
    latest = feature_time.max() if not feature_time.empty else pd.NaT
    stale_after = pd.Timedelta(minutes=15 * int(config.stops.stale_data_bars))
    data_stale = bool(pd.isna(latest) or checked_at - latest > stale_after)

    primary = trades.copy()
    if not primary.empty:
        if "portfolio_id" in primary.columns:
            primary = primary[
                primary["portfolio_id"].astype(str).eq(config.forward_primary.portfolio_id)
            ]
        else:
            primary = primary[
                primary.get("candidate", "").astype(str).eq(config.experiment.primary_candidate)
            ]
        if "portfolio_accepted" in primary.columns and "portfolio_id" not in primary.columns:
            primary = primary[primary["portfolio_accepted"].fillna(False).astype(bool)]
    primary = _completed_timely(primary)

    matched = baseline_trades.copy()
    if not matched.empty and "baseline_kind" in matched.columns:
        matched = matched[matched["baseline_kind"].astype(str).eq("matched_random_reclaim")]
    matched = _completed_timely(matched)

    window = int(config.stops.rolling_trade_window)
    primary_net_col = (
        "effective_net_return_10bp"
        if "effective_net_return_10bp" in primary.columns
        else "net_return_10bp"
    )
    primary_net = _mean_tail(primary, primary_net_col, window)
    matched_net = _mean_tail(matched, "net_return_10bp", window)
    lift = primary_net - matched_net if primary_net is not None and matched_net is not None else None

    reasons: list[str] = []
    if data_stale:
        reasons.append("data_stale")

    rolling_status = "disabled"
    if config.stops.pause_if_rolling_10bp_net_lte_zero:
        rolling_status = "insufficient_sample" if primary_net is None else "pass"
        if primary_net is not None and primary_net <= 0.0:
            rolling_status = "blocked"
            reasons.append("rolling_primary_net10_lte_zero")

    baseline_status = "disabled"
    if config.stops.pause_if_baseline_lift_lte_zero:
        baseline_status = "insufficient_sample" if lift is None else "pass"
        if lift is not None and lift <= 0.0:
            baseline_status = "blocked"
            reasons.append("rolling_baseline_lift10_lte_zero")

    return LiveGateDecision(
        allow_new_actions=not reasons,
        reasons=tuple(reasons),
        data_stale=data_stale,
        latest_feature_time="" if pd.isna(latest) else pd.Timestamp(latest).isoformat(),
        rolling_window=window,
        primary_completed_trades=int(len(primary)),
        rolling_primary_net10=primary_net,
        matched_random_completed_trades=int(len(matched)),
        rolling_matched_random_net10=matched_net,
        rolling_baseline_lift10=lift,
        rolling_net_gate_status=rolling_status,
        baseline_lift_gate_status=baseline_status,
    )


def write_live_gate_artifacts(
    *,
    report_root: Path,
    decision: LiveGateDecision,
    cumulative_signals: pd.DataFrame,
    observed_at: object,
    cumulative_primary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    root = ensure_dir(Path(report_root) / "forward")
    status_json = root / "live_gate_status.json"
    status_md = root / "live_gate_status.md"
    primary_path = root / "primary_forward_trades.parquet"
    actionable_path = root / "actionable_entries.parquet"

    status_json.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Live Gate Status",
        "",
        f"- allow_new_actions: {decision.allow_new_actions}",
        f"- reasons: {','.join(decision.reasons) if decision.reasons else 'none'}",
        f"- data_stale: {decision.data_stale}",
        f"- latest_feature_time: {decision.latest_feature_time or 'n/a'}",
        f"- rolling_window: {decision.rolling_window}",
        f"- primary_completed_trades: {decision.primary_completed_trades}",
        f"- rolling_primary_net10: {decision.rolling_primary_net10}",
        f"- rolling_net_gate_status: {decision.rolling_net_gate_status}",
        f"- matched_random_completed_trades: {decision.matched_random_completed_trades}",
        f"- rolling_matched_random_net10: {decision.rolling_matched_random_net10}",
        f"- rolling_baseline_lift10: {decision.rolling_baseline_lift10}",
        f"- baseline_lift_gate_status: {decision.baseline_lift_gate_status}",
    ]
    status_md.write_text("\n".join(lines), encoding="utf-8")

    action_source = cumulative_primary if cumulative_primary is not None else cumulative_signals
    write_parquet(action_source, primary_path)
    actionable = action_source.iloc[0:0].copy()
    if decision.allow_new_actions and not action_source.empty:
        first_seen = pd.to_datetime(action_source.get("first_observed_at_utc"), utc=True, errors="coerce")
        observed = _utc(observed_at)
        timely = action_source.get(
            "timely_forward_observation", pd.Series(False, index=action_source.index)
        ).fillna(False).astype(bool)
        actionable = action_source[first_seen.eq(observed) & timely].copy()
        actionable["live_action_allowed"] = True
    write_parquet(actionable, actionable_path)
    return {
        "live_gate_status_json": status_json,
        "live_gate_status_md": status_md,
        "primary_forward_trades": primary_path,
        "actionable_entries": actionable_path,
        "actionable_signals": actionable_path,
    }


__all__ = ["LiveGateDecision", "evaluate_live_gates", "write_live_gate_artifacts"]
