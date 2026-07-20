"""Posthoc reveal for the preregistered community peer-hedge book."""
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
from pressure_graph.reports.v201_reference_price_transmission import (
    COMMUNITY_TRADE,
    REPORT_ROOT as V201_REPORT_ROOT,
)
from pressure_graph.reports.v202_community_peer_hedge_feature_audit import (
    CANDIDATE,
    REPORT_ROOT as V202_REPORT_ROOT,
    build_peer_hedged_weights,
)


REPORT_ROOT = Path("reports/v20_3_community_peer_hedge")
FINDINGS_PATH = Path("docs/v203_community_peer_hedge_findings_2026_07_17.md")
TARGETS_PATH = V202_REPORT_ROOT / "peer_hedged_targets.parquet"


@dataclass(frozen=True)
class V203Config:
    holding_bars: int = 1
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 20_300


def parse_weights(value: object) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in str(value).split("|"):
        if not item:
            continue
        symbol, number = item.rsplit(":", maxsplit=1)
        output[symbol] = float(number)
    return output


def build_v203_events(
    targets: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V203Config = V203Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for item in targets.itertuples(index=False):
        feature_time = pd.Timestamp(item.feature_time)
        entry_time = feature_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        weights = parse_weights(item.weights)
        selected = [name for name in str(item.selected_symbols).split("|") if name]
        peers = [name for name in str(item.peer_symbols).split("|") if name]
        prices = close.reindex(
            index=[entry_time, exit_time], columns=list(weights)
        )
        if prices.isna().any().any():
            continue
        future = prices.loc[exit_time].div(prices.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weights[symbol] * future[symbol]) for symbol in weights
        }
        selected_contribution = float(
            sum(contributions[symbol] for symbol in selected)
        )
        peer_contribution = float(sum(contributions[symbol] for symbol in peers))
        gross = selected_contribution + peer_contribution
        selected_only_gross = float(
            -float(item.source_sign) * future.reindex(selected).mean()
        )
        rows.append(
            {
                **item._asdict(),
                "candidate": CANDIDATE,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "selected_contribution": selected_contribution,
                "peer_contribution": peer_contribution,
                "selected_only_gross_return": selected_only_gross,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": (
                    -gross - cfg.primary_round_trip_cost
                ),
                "symbol_contributions": contributions,
            }
        )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "community_id"]).reset_index(
            drop=True
        )
    return events


def summarize_v203(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(sample),
                "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                "active_months": (
                    sample["entry_month"].nunique() if len(sample) else 0
                ),
                "mean_selected_count": (
                    float(sample["selected_count"].mean())
                    if len(sample)
                    else math.nan
                ),
                "mean_peer_count": (
                    float(sample["peer_count"].mean()) if len(sample) else math.nan
                ),
                "mean_selected_contribution_bp": (
                    float(sample["selected_contribution"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_peer_contribution_bp": (
                    float(sample["peer_contribution"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_selected_only_gross_bp": (
                    float(sample["selected_only_gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_gross_bp": (
                    float(sample["gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_primary_net_bp": (
                    float(sample["primary_net_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_stress_net_bp": (
                    float(sample["stress_net_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "win_rate_primary": (
                    float(sample["primary_net_return"].gt(0).mean())
                    if len(sample)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def random_v203_controls(
    events: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V203Config = V203Config(),
) -> pd.DataFrame:
    contexts: list[dict[str, object]] = []
    for item in events.itertuples(index=False):
        selected = [name for name in str(item.selected_symbols).split("|") if name]
        peers = [name for name in str(item.peer_symbols).split("|") if name]
        members = [*selected, *peers]
        future = close.reindex(
            index=[item.entry_time, item.exit_time], columns=members
        )
        if future.isna().any().any():
            continue
        returns = future.loc[item.exit_time].div(future.loc[item.entry_time]).sub(1.0)
        contexts.append(
            {
                "members": members,
                "selected_count": len(selected),
                "source_sign": float(item.source_sign),
                "returns": returns.reindex(members).to_numpy(dtype=float),
            }
        )
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for iteration in range(cfg.random_iterations):
        values = []
        for context in contexts:
            count = int(context["selected_count"])
            chosen = set(
                rng.choice(len(context["members"]), size=count, replace=False).tolist()
            )
            selected = [
                name for index, name in enumerate(context["members"]) if index in chosen
            ]
            peers = [
                name
                for index, name in enumerate(context["members"])
                if index not in chosen
            ]
            weights = build_peer_hedged_weights(
                selected, peers, float(context["source_sign"])
            )
            lookup = {
                name: float(context["returns"][index])
                for index, name in enumerate(context["members"])
            }
            gross = sum(weight * lookup[symbol] for symbol, weight in weights.items())
            values.append(gross - cfg.primary_round_trip_cost)
        rows.append(
            {
                "iteration": iteration,
                "events": len(values),
                "mean_primary_net_return": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    events: pd.DataFrame,
    cfg: V203Config,
) -> tuple[float, float]:
    blocks = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in events.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + 1)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(blocks), len(blocks))
        means.append(float(np.mean(np.concatenate([blocks[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_month_concentration(events: pd.DataFrame) -> float:
    monthly = events.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def _positive_symbol_concentration(events: pd.DataFrame) -> float:
    values: dict[str, float] = {}
    for contributions in events["symbol_contributions"]:
        for symbol, contribution in dict(contributions).items():
            values[symbol] = values.get(symbol, 0.0) + float(contribution)
    positive = np.asarray([value for value in values.values() if value > 0])
    return float(positive.max() / positive.sum()) if positive.size else math.inf


def audit_v203(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    original_rpt4_primary_bp: float,
    cfg: V203Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = summary.set_index("scope")
    delayed = float(
        delayed_summary.loc[
            delayed_summary["scope"].eq("all"), "mean_primary_net_bp"
        ].iloc[0]
    )
    horizons = horizon_summary[
        horizon_summary["scope"].eq("all")
    ].set_index("holding_bars")
    real_mean = float(events["primary_net_return"].mean())
    reversed_mean = float(events["reversed_primary_net_return"].mean())
    low, high = _bootstrap(events, cfg)
    percentile = float(random_controls["mean_primary_net_return"].le(real_mean).mean())
    positive_side = float(
        events.loc[events["source_sign"].gt(0), "primary_net_return"].mean()
    )
    negative_side = float(
        events.loc[events["source_sign"].lt(0), "primary_net_return"].mean()
    )
    month_concentration = _positive_month_concentration(events)
    symbol_concentration = _positive_symbol_concentration(events)
    checks: dict[str, tuple[bool, float]] = {
        "full_events_200": (len(events) >= 200, float(len(events))),
        "validation_events_50": (
            scoped.loc["validation", "events"] >= 50,
            float(scoped.loc["validation", "events"]),
        ),
        "holdout_events_60": (
            scoped.loc["holdout", "events"] >= 60,
            float(scoped.loc["holdout", "events"]),
        ),
        "active_months_10": (
            scoped.loc["all", "active_months"] >= 10,
            float(scoped.loc["all", "active_months"]),
        ),
        "development_primary_positive": (
            scoped.loc["development", "mean_primary_net_bp"] > 0,
            float(scoped.loc["development", "mean_primary_net_bp"]),
        ),
        "validation_primary_positive": (
            scoped.loc["validation", "mean_primary_net_bp"] > 0,
            float(scoped.loc["validation", "mean_primary_net_bp"]),
        ),
        "holdout_primary_positive": (
            scoped.loc["holdout", "mean_primary_net_bp"] > 0,
            float(scoped.loc["holdout", "mean_primary_net_bp"]),
        ),
        "full_stress_positive": (
            scoped.loc["all", "mean_stress_net_bp"] > 0,
            float(scoped.loc["all", "mean_stress_net_bp"]),
        ),
        "bootstrap_lower_positive": (low > 0, low * 10_000),
        "random_partition_percentile_95": (percentile >= 0.95, percentile),
        "beats_reversed_direction": (
            real_mean > reversed_mean,
            (real_mean - reversed_mean) * 10_000,
        ),
        "beats_one_bar_delay": (
            real_mean * 10_000 > delayed,
            real_mean * 10_000 - delayed,
        ),
        "beats_original_btc_hedge": (
            real_mean * 10_000 > original_rpt4_primary_bp,
            real_mean * 10_000 - original_rpt4_primary_bp,
        ),
        "holding_30m_positive": (
            horizons.loc[2, "mean_primary_net_bp"] > 0,
            float(horizons.loc[2, "mean_primary_net_bp"]),
        ),
        "holding_60m_positive": (
            horizons.loc[4, "mean_primary_net_bp"] > 0,
            float(horizons.loc[4, "mean_primary_net_bp"]),
        ),
        "positive_source_side_positive": (
            positive_side > 0,
            positive_side * 10_000,
        ),
        "negative_source_side_positive": (
            negative_side > 0,
            negative_side * 10_000,
        ),
        "dollar_exposure_1e10": (
            events["net_dollar_exposure"].abs().max() <= 1e-10,
            float(events["net_dollar_exposure"].abs().max()),
        ),
        "gross_notional_drift_1e10": (
            events["gross_notional"].sub(1.0).abs().max() <= 1e-10,
            float(events["gross_notional"].sub(1.0).abs().max()),
        ),
        "positive_month_concentration_35": (
            month_concentration <= 0.35,
            month_concentration,
        ),
        "positive_symbol_concentration_25": (
            symbol_concentration <= 0.25,
            symbol_concentration,
        ),
    }
    all_pass = all(passed for passed, _ in checks.values())
    gates = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "all_gates_pass": all_pass,
            }
            for name, (passed, value) in checks.items()
        ]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "events": len(events),
                "mean_selected_contribution_bp": float(
                    events["selected_contribution"].mean() * 10_000
                ),
                "mean_peer_contribution_bp": float(
                    events["peer_contribution"].mean() * 10_000
                ),
                "mean_selected_only_gross_bp": float(
                    events["selected_only_gross_return"].mean() * 10_000
                ),
                "mean_gross_bp": float(events["gross_return"].mean() * 10_000),
                "break_even_round_trip_cost_bp": float(
                    events["gross_return"].mean() * 10_000
                ),
                "mean_primary_net_bp": real_mean * 10_000,
                "mean_stress_net_bp": float(
                    events["stress_net_return"].mean() * 10_000
                ),
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_partition_percentile": percentile,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "delayed_primary_net_bp": delayed,
                "original_rpt4_primary_net_bp": original_rpt4_primary_bp,
                "positive_source_primary_bp": positive_side * 10_000,
                "negative_source_primary_bp": negative_side * 10_000,
                "positive_month_concentration": month_concentration,
                "positive_symbol_concentration": symbol_concentration,
                "all_gates_pass": all_pass,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "posthoc_offline_discovery_natural_forward_required"
                    if all_pass
                    else "reject_community_peer_hedge"
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
    text = [
        "# v20.3 Community Peer-Hedge Posthoc Findings",
        "",
        f"Verdict: `{outcome['verdict'].iloc[0]}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The reveal follows the frozen v20.3 posthoc preregistration. A passing "
        "result would still require natural-forward evidence and could not be "
        "promoted from this historical sample.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v203_reveal(
    targets_path: Path = TARGETS_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V203Config = V203Config(),
) -> dict[str, Path]:
    targets = pd.read_parquet(targets_path)
    targets["feature_time"] = pd.to_datetime(targets["feature_time"], utc=True)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    events = build_v203_events(targets, close, cfg)
    summary = summarize_v203(events)
    delayed = build_v203_events(targets, close, cfg, entry_delay_bars=1)
    delayed_summary = summarize_v203(delayed)
    horizon_frames = []
    for horizon in (2, 4):
        local = summarize_v203(
            build_v203_events(targets, close, cfg, holding_bars=horizon)
        )
        local["holding_bars"] = horizon
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v203_controls(events, close, cfg)
    original = pd.read_csv(V201_REPORT_ROOT / "candidate_outcome.csv")
    original_rpt4 = float(
        original.loc[
            original["candidate"].eq(COMMUNITY_TRADE), "mean_primary_net_bp"
        ].iloc[0]
    )
    gates, outcome = audit_v203(
        events,
        summary,
        delayed_summary,
        horizon_summary,
        random_controls,
        original_rpt4,
        cfg,
    )
    root = ensure_dir(report_root)
    outputs = {
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_partition_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    serial = events.copy()
    serial["symbol_contributions"] = serial["symbol_contributions"].map(
        _serialize_mapping
    )
    serial.to_parquet(outputs["events"], index=False)
    serial_delayed = delayed.copy()
    serial_delayed["symbol_contributions"] = serial_delayed[
        "symbol_contributions"
    ].map(_serialize_mapping)
    serial_delayed.to_parquet(outputs["delayed_events"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    horizon_summary.to_csv(outputs["horizons"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return outputs


__all__ = [
    "V203Config",
    "audit_v203",
    "build_v203_events",
    "parse_weights",
    "random_v203_controls",
    "summarize_v203",
    "write_v203_reveal",
]
