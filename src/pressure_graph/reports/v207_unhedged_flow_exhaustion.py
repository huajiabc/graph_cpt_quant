"""Post-hoc diagnostic for the unhedged RFX1 receiver bucket."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    EVENT_WIDE,
)
from pressure_graph.reports.v205_aggtrade_flow_exhaustion import (
    CANDIDATE_FEATURES_PATH,
    RECEIVER_FEATURES_PATH,
    build_rfx1_baseline_features,
)


REPORT_ROOT = Path("reports/v20_7_unhedged_flow_exhaustion")
FINDINGS_PATH = Path("docs/v207_unhedged_flow_exhaustion_findings_2026_07_17.md")
CANDIDATE = "RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE"
BTC_CONTROL = "BTC1_EVENT_MATCHED_SOURCE_FADE_CONTROL"


@dataclass(frozen=True)
class V207Config:
    holding_bars: int = 1
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 20_700


def _parse_symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def build_v207_events(
    features: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V207Config = V207Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for item in features.itertuples(index=False):
        source_time = pd.Timestamp(item.feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        receivers = _parse_symbols(item.candidate_receivers)
        if (
            not receivers
            or entry_time not in close.index
            or exit_time not in close.index
        ):
            continue
        prices = close.reindex(
            index=[entry_time, exit_time], columns=[BTC, *receivers]
        )
        if prices.isna().any().any():
            continue
        future = prices.loc[exit_time].div(prices.loc[entry_time]).sub(1.0)
        direction = -float(item.source_sign)
        weights = {symbol: direction / len(receivers) for symbol in receivers}
        contributions = {
            symbol: float(weight * future[symbol])
            for symbol, weight in weights.items()
        }
        receiver_gross = float(sum(contributions.values()))
        btc_gross = float(direction * future[BTC])
        paired_excess = receiver_gross - btc_gross
        rows.append(
            {
                **item._asdict(),
                "candidate": CANDIDATE,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "realized_receiver_count": len(receivers),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "receiver_gross_return": receiver_gross,
                "receiver_primary_net_return": (
                    receiver_gross - cfg.primary_round_trip_cost
                ),
                "receiver_stress_net_return": (
                    receiver_gross - cfg.stress_round_trip_cost
                ),
                "reversed_receiver_primary_net_return": (
                    -receiver_gross - cfg.primary_round_trip_cost
                ),
                "btc_control_gross_return": btc_gross,
                "btc_control_primary_net_return": (
                    btc_gross - cfg.primary_round_trip_cost
                ),
                "paired_receiver_minus_btc_return": paired_excess,
                "break_even_cost_bp": receiver_gross * 10_000,
                "weights": weights,
                "symbol_contributions": contributions,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "community_id"]
    ).reset_index(drop=True)


def summarize_v207(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        local = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_days": local["entry_day"].nunique() if len(local) else 0,
                "active_months": local["entry_month"].nunique() if len(local) else 0,
                "mean_receivers": (
                    float(local["realized_receiver_count"].mean())
                    if len(local)
                    else math.nan
                ),
                "mean_receiver_gross_bp": (
                    float(local["receiver_gross_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_receiver_primary_net_bp": (
                    float(local["receiver_primary_net_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_receiver_stress_net_bp": (
                    float(local["receiver_stress_net_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_btc_control_gross_bp": (
                    float(local["btc_control_gross_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_btc_control_primary_net_bp": (
                    float(local["btc_control_primary_net_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_receiver_minus_btc_bp": (
                    float(local["paired_receiver_minus_btc_return"].mean() * 10_000)
                    if len(local)
                    else math.nan
                ),
                "mean_reversed_primary_net_bp": (
                    float(
                        local["reversed_receiver_primary_net_return"].mean()
                        * 10_000
                    )
                    if len(local)
                    else math.nan
                ),
                "positive_receiver_primary_fraction": (
                    float(local["receiver_primary_net_return"].gt(0).mean())
                    if len(local)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def random_v207_controls(
    observed: pd.DataFrame,
    baseline: pd.DataFrame,
    cfg: V207Config = V207Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    counts = observed.groupby("period").size().to_dict()
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        samples: list[pd.DataFrame] = []
        for period, count in counts.items():
            pool = baseline[baseline["period"].eq(period)]
            indices = rng.choice(pool.index.to_numpy(), size=count, replace=False)
            samples.append(pool.loc[indices])
        sample = pd.concat(samples, ignore_index=True)
        rows.append(
            {
                "iteration": iteration,
                "mean_receiver_gross_return": float(
                    sample["receiver_gross_return"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_v207(
    events: pd.DataFrame,
    cfg: V207Config = V207Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 100)
    net = events["receiver_primary_net_return"].to_numpy(dtype=float)
    paired = events["paired_receiver_minus_btc_return"].to_numpy(dtype=float)
    net_means: list[float] = []
    paired_means: list[float] = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(events), size=len(events))
        net_means.append(float(net[indices].mean()))
        paired_means.append(float(paired[indices].mean()))
    return pd.DataFrame(
        [
            {
                "events": len(events),
                "mean_receiver_primary_net_bp": float(net.mean() * 10_000),
                "lower_95_receiver_primary_net_bp": float(
                    np.quantile(net_means, 0.025) * 10_000
                ),
                "upper_95_receiver_primary_net_bp": float(
                    np.quantile(net_means, 0.975) * 10_000
                ),
                "mean_receiver_minus_btc_bp": float(paired.mean() * 10_000),
                "lower_95_receiver_minus_btc_bp": float(
                    np.quantile(paired_means, 0.025) * 10_000
                ),
                "upper_95_receiver_minus_btc_bp": float(
                    np.quantile(paired_means, 0.975) * 10_000
                ),
            }
        ]
    )


def build_cost_frontier(events: pd.DataFrame) -> pd.DataFrame:
    gross = float(events["receiver_gross_return"].mean())
    return pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "round_trip_cost_bp": cost_bp,
                "events": len(events),
                "mean_gross_bp": gross * 10_000,
                "mean_net_bp": gross * 10_000 - cost_bp,
            }
            for cost_bp in (5, 10, 20, 40)
        ]
    )


def audit_v207(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cfg: V207Config = V207Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_summary = summary[summary["scope"].eq("all")].iloc[0]
    delayed = delayed_summary[delayed_summary["scope"].eq("all")].iloc[0]
    boot = bootstrap.iloc[0]
    period_counts = summary[summary["scope"].ne("all")].set_index("scope")["events"]
    period_net = summary[summary["scope"].ne("all")].set_index("scope")[
        "mean_receiver_primary_net_bp"
    ]
    observed_gross = float(events["receiver_gross_return"].mean())
    percentile = float(
        random_controls["mean_receiver_gross_return"].le(observed_gross).mean()
    )
    checks = {
        "exact_period_counts_30_15_8": (
            period_counts.to_dict()
            == {"development": 30, "validation": 15, "holdout": 8},
            min(period_counts),
            8,
        ),
        "gross_exceeds_20bp_cost": (
            observed_gross > cfg.primary_round_trip_cost,
            observed_gross * 10_000,
            20.0,
        ),
        "all_period_primary_net_positive": (
            float(period_net.min()) > 0,
            float(period_net.min()),
            0.0,
        ),
        "delayed_primary_net_positive": (
            float(delayed["mean_receiver_primary_net_bp"]) > 0,
            float(delayed["mean_receiver_primary_net_bp"]),
            0.0,
        ),
        "random_control_percentile_at_least_0_95": (
            percentile >= 0.95,
            percentile,
            0.95,
        ),
        "bootstrap_receiver_net_lower_95_positive": (
            float(boot["lower_95_receiver_primary_net_bp"]) > 0,
            float(boot["lower_95_receiver_primary_net_bp"]),
            0.0,
        ),
        "mean_receiver_minus_btc_positive": (
            float(all_summary["mean_receiver_minus_btc_bp"]) > 0,
            float(all_summary["mean_receiver_minus_btc_bp"]),
            0.0,
        ),
        "bootstrap_receiver_minus_btc_lower_95_positive": (
            float(boot["lower_95_receiver_minus_btc_bp"]) > 0,
            float(boot["lower_95_receiver_minus_btc_bp"]),
            0.0,
        ),
    }
    gates = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "check": check,
                "passed": bool(values[0]),
                "value": values[1],
                "threshold": values[2],
            }
            for check, values in checks.items()
        ]
    )
    eligible = all(values[0] for values in checks.values())
    outcome = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "events": len(events),
                "mean_receiver_gross_bp": observed_gross * 10_000,
                "mean_receiver_primary_net_bp": float(
                    all_summary["mean_receiver_primary_net_bp"]
                ),
                "mean_btc_control_gross_bp": float(
                    all_summary["mean_btc_control_gross_bp"]
                ),
                "mean_receiver_minus_btc_bp": float(
                    all_summary["mean_receiver_minus_btc_bp"]
                ),
                "random_control_percentile": percentile,
                "eligible_for_natural_forward_observation": eligible,
                "status": (
                    "diagnostic_pass_natural_forward_observation_only"
                    if eligible
                    else "diagnostic_reject"
                ),
            }
        ]
    )
    return gates, outcome


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    eligible = bool(outcome["eligible_for_natural_forward_observation"].iloc[0])
    verdict = (
        "retain_for_natural_forward_observation_only"
        if eligible
        else "reject_unhedged_flow_exhaustion_diagnostic"
    )
    text = [
        "# v20.7 Unhedged Flow-Exhaustion Diagnostic Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This is a post-hoc decomposition diagnostic, not independent alpha "
        "evidence. Passing could only justify untouched natural-forward observation.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v207_diagnostic(
    receiver_features_path: Path = RECEIVER_FEATURES_PATH,
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V207Config = V207Config(),
) -> dict[str, Path]:
    receivers = pd.read_parquet(receiver_features_path)
    candidates = pd.read_parquet(candidate_features_path)
    for frame in (receivers, candidates):
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
    selected = candidates[candidates["candidate"].eq(EVENT_WIDE)].copy()
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    events = build_v207_events(selected, close, cfg)
    summary = summarize_v207(events)
    delayed_events = build_v207_events(selected, close, cfg, entry_delay_bars=1)
    delayed_summary = summarize_v207(delayed_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (2, 4):
        horizon_events = build_v207_events(
            selected, close, cfg, holding_bars=horizon
        )
        local_summary = summarize_v207(horizon_events)
        local_summary["holding_bars"] = horizon
        horizon_frames.append(local_summary)
    horizons = pd.concat(horizon_frames, ignore_index=True)
    baseline_features = build_rfx1_baseline_features(receivers)
    baseline_events = build_v207_events(baseline_features, close, cfg)
    random_controls = random_v207_controls(events, baseline_events, cfg)
    bootstrap = bootstrap_v207(events, cfg)
    cost_frontier = build_cost_frontier(events)
    gates, outcome = audit_v207(
        events, summary, delayed_summary, random_controls, bootstrap, cfg
    )
    root = ensure_dir(report_root)
    outputs = {
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "random_controls": root / "random_controls.parquet",
        "bootstrap": root / "bootstrap_summary.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    for frame, path in (
        (events, outputs["events"]),
        (delayed_events, outputs["delayed_events"]),
    ):
        serial = frame.copy()
        serial["weights"] = serial["weights"].map(_serialize_mapping)
        serial["symbol_contributions"] = serial["symbol_contributions"].map(
            _serialize_mapping
        )
        serial.to_parquet(path, index=False)
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    horizons.to_csv(outputs["horizons"], index=False)
    cost_frontier.to_csv(outputs["cost_frontier"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return outputs


__all__ = [
    "BTC_CONTROL",
    "CANDIDATE",
    "V207Config",
    "audit_v207",
    "bootstrap_v207",
    "build_v207_events",
    "random_v207_controls",
    "summarize_v207",
    "write_v207_diagnostic",
]
