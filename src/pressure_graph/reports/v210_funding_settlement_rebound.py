"""Preregistered post-settlement rebound reveal for negative-funding buckets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)
from pressure_graph.reports.v209_funding_settlement_feature_audit import (
    CANDIDATES,
    REPORT_ROOT as V209_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v21_0_funding_settlement_rebound")
FINDINGS_PATH = Path("docs/v210_funding_settlement_rebound_findings_2026_07_17.md")
CANDIDATE_FEATURES_PATH = V209_REPORT_ROOT / "candidate_feature_events.parquet"


@dataclass(frozen=True)
class V210Config:
    holding_bars: int = 4
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 21_000
    minimum_events: int = 400
    minimum_period_events: int = 100
    minimum_active_months: int = 10


def _parse_symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def beta_neutral_long_weights(
    symbols: list[str],
    beta: pd.Series,
    direction: float = 1.0,
) -> dict[str, float]:
    if not symbols:
        return {}
    raw = {symbol: direction / len(symbols) for symbol in symbols}
    hedge = -float(sum(raw[symbol] * float(beta[symbol]) for symbol in symbols))
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def build_v210_events(
    candidate_features: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V210Config = V210Config(),
    holding_bars: int | None = None,
    additional_entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    universe = sorted(set(close.columns) - {BTC})
    rows: list[dict[str, object]] = []
    for item in candidate_features.itertuples(index=False):
        settlement_time = pd.Timestamp(item.settlement_time)
        entry_time = pd.Timestamp(item.entry_time) + pd.Timedelta(
            minutes=15 * additional_entry_delay_bars
        )
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        selected = _parse_symbols(item.selection_symbols)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        event_prices = close.reindex(
            index=[entry_time, exit_time], columns=[BTC, *universe]
        )
        month = _month(settlement_time)
        universe_beta = pd.Series(
            {
                symbol: risk_lookup.get((month, symbol), np.nan)
                for symbol in universe
            },
            dtype=float,
        )
        if event_prices.isna().any().any() or universe_beta.isna().any():
            continue
        selected_beta = universe_beta.reindex(selected)
        weights = beta_neutral_long_weights(selected, selected_beta)
        if not weights:
            continue
        future = event_prices.loc[exit_time].div(event_prices.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weight * future[symbol])
            for symbol, weight in weights.items()
        }
        gross = float(sum(contributions.values()))
        rows.append(
            {
                **item._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "additional_entry_delay_bars": additional_entry_delay_bars,
                "holding_bars": horizon,
                "settlement_hour": settlement_time.hour,
                "realized_selection_count": len(selected),
                "btc_hedge_weight": float(weights[BTC]),
                "residual_btc_beta": float(
                    weights[BTC]
                    + sum(
                        weights[symbol] * float(selected_beta[symbol])
                        for symbol in selected
                    )
                ),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "alt_contribution": float(
                    sum(contributions[symbol] for symbol in selected)
                ),
                "btc_hedge_contribution": float(contributions[BTC]),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": -gross - cfg.primary_round_trip_cost,
                "break_even_cost_bp": gross * 10_000,
                "weights": weights,
                "symbol_contributions": contributions,
                "universe_symbols": universe,
                "universe_betas": universe_beta.to_dict(),
                "universe_future_returns": future.reindex([BTC, *universe]).to_dict(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "candidate"]
    ).reset_index(drop=True)


def summarize_v210(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": (
                        sample["entry_month"].nunique() if len(sample) else 0
                    ),
                    "mean_selection_count": (
                        float(sample["realized_selection_count"].mean())
                        if len(sample)
                        else math.nan
                    ),
                    "mean_alt_bp": (
                        float(sample["alt_contribution"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_btc_hedge_bp": (
                        float(sample["btc_hedge_contribution"].mean() * 10_000)
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
                    "mean_reversed_primary_net_bp": (
                        float(sample["reversed_primary_net_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "positive_primary_fraction": (
                        float(sample["primary_net_return"].gt(0).mean())
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_v210_hours(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for hour in (0, 8, 16):
            sample = local[local["settlement_hour"].eq(hour)]
            rows.append(
                {
                    "candidate": candidate,
                    "settlement_hour": hour,
                    "events": len(sample),
                    "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
                    "mean_primary_net_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    ),
                }
            )
    return pd.DataFrame(rows)


def random_v210_controls(
    events: pd.DataFrame,
    cfg: V210Config = V210Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        totals = np.zeros(cfg.random_iterations, dtype=float)
        for item in local.itertuples(index=False):
            symbols = list(item.universe_symbols)
            betas = pd.Series(item.universe_betas, dtype=float).reindex(symbols).to_numpy()
            future_series = pd.Series(item.universe_future_returns, dtype=float)
            future = future_series.reindex(symbols).to_numpy()
            count = int(item.realized_selection_count)
            scores = rng.random((cfg.random_iterations, len(symbols)))
            indices = np.argpartition(scores, count - 1, axis=1)[:, :count]
            mean_beta = betas[indices].mean(axis=1)
            mean_alt_future = future[indices].mean(axis=1)
            hedge = -mean_beta
            gross = (
                mean_alt_future + hedge * float(future_series[BTC])
            ) / (1.0 + np.abs(hedge))
            totals += gross
        means = totals / len(local)
        rows.extend(
            {
                "candidate": candidate,
                "iteration": iteration,
                "mean_gross_return": float(means[iteration]),
            }
            for iteration in range(cfg.random_iterations)
        )
    return pd.DataFrame(rows)


def day_block_bootstrap_v210(
    events: pd.DataFrame,
    cfg: V210Config = V210Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)].copy()
        daily = local.groupby("entry_day")["primary_net_return"].agg(["sum", "count"])
        rng = np.random.default_rng(cfg.seed + 100 + offset)
        means: list[float] = []
        for _ in range(cfg.bootstrap_iterations):
            indices = rng.integers(0, len(daily), size=len(daily))
            sample = daily.iloc[indices]
            means.append(float(sample["sum"].sum() / sample["count"].sum()))
        rows.append(
            {
                "candidate": candidate,
                "events": len(local),
                "active_days": len(daily),
                "mean_primary_net_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "lower_95_primary_net_bp": float(
                    np.quantile(means, 0.025) * 10_000
                ),
                "upper_95_primary_net_bp": float(
                    np.quantile(means, 0.975) * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cost_frontier(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        gross = float(local["gross_return"].mean())
        for cost_bp in (5, 10, 20, 40):
            rows.append(
                {
                    "candidate": candidate,
                    "round_trip_cost_bp": cost_bp,
                    "events": len(local),
                    "mean_gross_bp": gross * 10_000,
                    "mean_net_bp": gross * 10_000 - cost_bp,
                }
            )
    return pd.DataFrame(rows)


def audit_v210(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cfg: V210Config = V210Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        local_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        local_hours = hour_summary[hour_summary["candidate"].eq(candidate)]
        delayed = delayed_summary[
            (delayed_summary["candidate"].eq(candidate))
            & delayed_summary["scope"].eq("all")
        ].iloc[0]
        boot = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        random = random_controls[
            random_controls["candidate"].eq(candidate)
        ]["mean_gross_return"]
        observed_gross = float(local["gross_return"].mean())
        percentile = float(random.le(observed_gross).mean())
        period_events = [
            int(local_summary.loc[period, "events"])
            for period in ("development", "validation", "holdout")
        ]
        period_net = [
            float(local_summary.loc[period, "mean_primary_net_bp"])
            for period in ("development", "validation", "holdout")
        ]
        checks = {
            "minimum_total_events": (
                len(local) >= cfg.minimum_events,
                len(local),
                cfg.minimum_events,
            ),
            "minimum_each_period_events": (
                min(period_events) >= cfg.minimum_period_events,
                min(period_events),
                cfg.minimum_period_events,
            ),
            "minimum_active_months": (
                int(local_summary.loc["all", "active_months"])
                >= cfg.minimum_active_months,
                int(local_summary.loc["all", "active_months"]),
                cfg.minimum_active_months,
            ),
            "gross_exceeds_20bp_cost": (
                observed_gross > cfg.primary_round_trip_cost,
                observed_gross * 10_000,
                20.0,
            ),
            "all_period_primary_net_positive": (
                min(period_net) > 0,
                min(period_net),
                0.0,
            ),
            "all_settlement_hours_primary_net_positive": (
                float(local_hours["mean_primary_net_bp"].min()) > 0,
                float(local_hours["mean_primary_net_bp"].min()),
                0.0,
            ),
            "delayed_primary_net_positive": (
                float(delayed["mean_primary_net_bp"]) > 0,
                float(delayed["mean_primary_net_bp"]),
                0.0,
            ),
            "random_control_percentile_at_least_0_95": (
                percentile >= 0.95,
                percentile,
                0.95,
            ),
            "reversed_primary_net_negative": (
                float(local_summary.loc["all", "mean_reversed_primary_net_bp"]) < 0,
                float(local_summary.loc["all", "mean_reversed_primary_net_bp"]),
                0.0,
            ),
            "day_bootstrap_lower_95_positive": (
                float(boot["lower_95_primary_net_bp"]) > 0,
                float(boot["lower_95_primary_net_bp"]),
                0.0,
            ),
        }
        for check, (passed, value, threshold) in checks.items():
            gate_rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": value,
                    "threshold": threshold,
                }
            )
        eligible = all(passed for passed, _, _ in checks.values())
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(local),
                "mean_gross_bp": observed_gross * 10_000,
                "mean_primary_net_bp": float(
                    local_summary.loc["all", "mean_primary_net_bp"]
                ),
                "break_even_cost_bp": observed_gross * 10_000,
                "random_control_percentile": percentile,
                "day_bootstrap_lower_95_primary_net_bp": float(
                    boot["lower_95_primary_net_bp"]
                ),
                "eligible": eligible,
                "status": (
                    "offline_candidate_natural_forward_only"
                    if eligible
                    else "rejected"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _serialize_symbols(value: list[str]) -> str:
    return "|".join(value)


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_candidate_natural_forward_only"
        if outcome["eligible"].any()
        else "reject_funding_settlement_rebound_candidates"
    )
    text = [
        "# v21.0 Funding-Settlement Rebound Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary[summary["scope"].eq("all")].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        hour_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The reveal follows the frozen v21.0 preregistration. Signal observation "
        "precedes entry by a full 15-minute bar; no funding payment is credited. "
        "Primary/stress results charge 20/40 bp round-trip book costs.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v210_reveal(
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V210Config = V210Config(),
) -> dict[str, Path]:
    candidates = pd.read_parquet(candidate_features_path)
    for column in ("settlement_time", "entry_time", "exit_time"):
        candidates[column] = pd.to_datetime(candidates[column], utc=True)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    risk = build_v187_monthly_risk(
        close.pct_change(fill_method=None),
        candidates["settlement_time"].min(),
        candidates["settlement_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    events = build_v210_events(candidates, risk, close, cfg)
    summary = summarize_v210(events)
    hour_summary = summarize_v210_hours(events)
    delayed_events = build_v210_events(
        candidates, risk, close, cfg, additional_entry_delay_bars=1
    )
    delayed_summary = summarize_v210(delayed_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (2, 8):
        horizon_events = build_v210_events(
            candidates, risk, close, cfg, holding_bars=horizon
        )
        local_summary = summarize_v210(horizon_events)
        local_summary["holding_bars"] = horizon
        horizon_frames.append(local_summary)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v210_controls(events, cfg)
    bootstrap = day_block_bootstrap_v210(events, cfg)
    cost_frontier = build_cost_frontier(events)
    gates, outcome = audit_v210(
        events,
        summary,
        hour_summary,
        delayed_summary,
        random_controls,
        bootstrap,
        cfg,
    )
    root = ensure_dir(report_root)
    outputs = {
        "risk": root / "monthly_btc_risk.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "hours": root / "settlement_hour_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "random_controls": root / "random_controls.parquet",
        "bootstrap": root / "day_block_bootstrap_summary.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    risk.to_parquet(outputs["risk"], index=False)
    for frame, path in (
        (events, outputs["events"]),
        (delayed_events, outputs["delayed_events"]),
    ):
        serial = frame.copy()
        serial["weights"] = serial["weights"].map(_serialize_mapping)
        serial["symbol_contributions"] = serial["symbol_contributions"].map(
            _serialize_mapping
        )
        serial["universe_symbols"] = serial["universe_symbols"].map(
            _serialize_symbols
        )
        serial["universe_betas"] = serial["universe_betas"].map(_serialize_mapping)
        serial["universe_future_returns"] = serial[
            "universe_future_returns"
        ].map(_serialize_mapping)
        serial.to_parquet(path, index=False)
    summary.to_csv(outputs["summary"], index=False)
    hour_summary.to_csv(outputs["hours"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    horizon_summary.to_csv(outputs["horizons"], index=False)
    cost_frontier.to_csv(outputs["cost_frontier"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, summary, hour_summary, findings_path)
    return outputs


__all__ = [
    "V210Config",
    "audit_v210",
    "beta_neutral_long_weights",
    "build_v210_events",
    "day_block_bootstrap_v210",
    "random_v210_controls",
    "summarize_v210",
    "summarize_v210_hours",
    "write_v210_reveal",
]
