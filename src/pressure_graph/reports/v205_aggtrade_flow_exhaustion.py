"""Preregistered reveal for aggTrade flow exhaustion candidates."""
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
from pressure_graph.reports.v201_reference_price_transmission import (
    beta_neutral_weights,
)
from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    CANDIDATES,
    EVENT_WIDE,
    REPORT_ROOT as V204_REPORT_ROOT,
    STATE_SPREAD,
)


REPORT_ROOT = Path("reports/v20_5_aggtrade_flow_exhaustion")
FINDINGS_PATH = Path("docs/v205_aggtrade_flow_exhaustion_findings_2026_07_17.md")
RECEIVER_FEATURES_PATH = V204_REPORT_ROOT / "receiver_features.parquet"
CANDIDATE_FEATURES_PATH = V204_REPORT_ROOT / "candidate_feature_events.parquet"


@dataclass(frozen=True)
class V205Config:
    holding_bars: int = 1
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 20_500
    minimum_events: int = 45
    minimum_period_events: int = 5


def _parse_symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def state_spread_weights(
    exhausted: list[str],
    persistent: list[str],
    source_sign: float,
    beta: pd.Series,
) -> dict[str, float]:
    if not exhausted or not persistent:
        return {}
    raw = {
        symbol: -source_sign * 0.5 / len(exhausted) for symbol in exhausted
    }
    raw.update(
        {
            symbol: source_sign * 0.5 / len(persistent)
            for symbol in persistent
        }
    )
    hedge = -float(sum(weight * float(beta[symbol]) for symbol, weight in raw.items()))
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def _build_one_event(
    item: object,
    risk_lookup: pd.Series,
    close: pd.DataFrame,
    cfg: V205Config,
    holding_bars: int,
    entry_delay_bars: int,
    exhausted_override: list[str] | None = None,
    persistent_override: list[str] | None = None,
) -> dict[str, object] | None:
    candidate = str(item.candidate)
    source_time = pd.Timestamp(item.feature_time)
    entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
    exit_time = entry_time + pd.Timedelta(minutes=15 * holding_bars)
    if candidate == EVENT_WIDE:
        receivers = _parse_symbols(item.candidate_receivers)
        exhausted: list[str] = []
        persistent: list[str] = []
    else:
        exhausted = (
            exhausted_override
            if exhausted_override is not None
            else _parse_symbols(item.strict_exhausted_receivers)
        )
        persistent = (
            persistent_override
            if persistent_override is not None
            else _parse_symbols(item.persistent_receivers)
        )
        receivers = [*exhausted, *persistent]
    if entry_time not in close.index or exit_time not in close.index or not receivers:
        return None
    beta = pd.Series(
        {
            symbol: risk_lookup.get((_month(source_time), symbol), np.nan)
            for symbol in receivers
        },
        dtype=float,
    )
    prices = close.reindex(index=[entry_time, exit_time], columns=[BTC, *receivers])
    if beta.isna().any() or prices.isna().any().any():
        return None
    if candidate == EVENT_WIDE:
        weights = beta_neutral_weights(
            receivers, -float(item.source_sign), beta
        )
    else:
        weights = state_spread_weights(
            exhausted, persistent, float(item.source_sign), beta
        )
    if not weights:
        return None
    future = prices.loc[exit_time].div(prices.loc[entry_time]).sub(1.0)
    contributions = {
        symbol: float(weight * future[symbol])
        for symbol, weight in weights.items()
    }
    gross = float(sum(contributions.values()))
    exhausted_contribution = (
        float(sum(contributions[symbol] for symbol in exhausted))
        if exhausted
        else math.nan
    )
    persistent_contribution = (
        float(sum(contributions[symbol] for symbol in persistent))
        if persistent
        else math.nan
    )
    return {
        **item._asdict(),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_delay_bars": entry_delay_bars,
        "holding_bars": holding_bars,
        "realized_receiver_count": len(receivers),
        "btc_hedge_weight": float(weights[BTC]),
        "residual_btc_beta": float(
            weights[BTC]
            + sum(weights[symbol] * float(beta[symbol]) for symbol in receivers)
        ),
        "gross_notional": float(sum(abs(weight) for weight in weights.values())),
        "all_alt_contribution": float(
            sum(contributions[symbol] for symbol in receivers)
        ),
        "exhausted_contribution": exhausted_contribution,
        "persistent_contribution": persistent_contribution,
        "btc_hedge_contribution": float(contributions[BTC]),
        "gross_return": gross,
        "primary_net_return": gross - cfg.primary_round_trip_cost,
        "stress_net_return": gross - cfg.stress_round_trip_cost,
        "reversed_primary_net_return": -gross - cfg.primary_round_trip_cost,
        "break_even_cost_bp": gross * 10_000,
        "weights": weights,
        "symbol_contributions": contributions,
        "symbol_betas": beta.to_dict(),
        "symbol_future_returns": future.reindex([BTC, *receivers]).to_dict(),
    }


def build_v205_events(
    candidate_features: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V205Config = V205Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    rows = []
    for item in candidate_features.itertuples(index=False):
        row = _build_one_event(
            item,
            risk_lookup,
            close,
            cfg,
            horizon,
            entry_delay_bars,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "candidate", "community_id"]
    ).reset_index(drop=True)


def build_rfx1_baseline_features(receivers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, local in receivers.groupby("source_event_id", sort=True):
        local = local[local["quality_ok"]].sort_values("symbol")
        if len(local) < 3:
            continue
        first = local.iloc[0]
        rows.append(
            {
                "source_event_id": first["source_event_id"],
                "feature_time": first["feature_time"],
                "period": first["period"],
                "entry_day": pd.Timestamp(first["feature_time"]).date(),
                "entry_month": pd.Timestamp(first["feature_time"]).strftime("%Y-%m"),
                "community_id": first["community_id"],
                "source_sign": float(first["source_sign"]),
                "candidate": EVENT_WIDE,
                "candidate_receiver_count": len(local),
                "candidate_receivers": "|".join(local["symbol"].astype(str)),
                "strict_exhausted_receivers": "",
                "persistent_receivers": "",
            }
        )
    return pd.DataFrame(rows)


def summarize_v205(events: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_receivers": (
                        float(sample["realized_receiver_count"].mean())
                        if len(sample)
                        else math.nan
                    ),
                    "mean_all_alt_bp": (
                        float(sample["all_alt_contribution"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_exhausted_bp": (
                        float(sample["exhausted_contribution"].mean() * 10_000)
                        if sample["exhausted_contribution"].notna().any()
                        else math.nan
                    ),
                    "mean_persistent_bp": (
                        float(sample["persistent_contribution"].mean() * 10_000)
                        if sample["persistent_contribution"].notna().any()
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


def build_cost_frontier(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        gross = float(local["gross_return"].mean()) if len(local) else math.nan
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


def random_v205_controls(
    observed: pd.DataFrame,
    baseline_rfx1: pd.DataFrame,
    cfg: V205Config = V205Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    observed_wide = observed[observed["candidate"].eq(EVENT_WIDE)]
    observed_counts = observed_wide.groupby("period").size().to_dict()
    for iteration in range(cfg.random_iterations):
        sampled: list[pd.DataFrame] = []
        for period, count in observed_counts.items():
            pool = baseline_rfx1[baseline_rfx1["period"].eq(period)]
            indices = rng.choice(pool.index.to_numpy(), size=count, replace=False)
            sampled.append(pool.loc[indices])
        random_wide = pd.concat(sampled, ignore_index=True)
        rows.append(
            {
                "candidate": EVENT_WIDE,
                "iteration": iteration,
                "mean_gross_return": float(random_wide["gross_return"].mean()),
            }
        )

    spread = observed[observed["candidate"].eq(STATE_SPREAD)]
    for iteration in range(cfg.random_iterations):
        gross_returns: list[float] = []
        for item in spread.itertuples(index=False):
            symbols = _parse_symbols(item.candidate_receivers)
            shuffled = list(rng.permutation(symbols))
            exhausted_count = int(item.strict_exhausted_count)
            exhausted = shuffled[:exhausted_count]
            persistent = shuffled[exhausted_count:]
            beta = pd.Series(item.symbol_betas, dtype=float).reindex(symbols)
            future = pd.Series(item.symbol_future_returns, dtype=float)
            weights = state_spread_weights(
                exhausted, persistent, float(item.source_sign), beta
            )
            gross_returns.append(
                float(
                    sum(weights[symbol] * future[symbol] for symbol in symbols)
                    + weights[BTC] * future[BTC]
                )
            )
        rows.append(
            {
                "candidate": STATE_SPREAD,
                "iteration": iteration,
                "mean_gross_return": float(np.mean(gross_returns)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_v205(
    events: pd.DataFrame,
    cfg: V205Config = V205Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        values = events.loc[
            events["candidate"].eq(candidate), "primary_net_return"
        ].to_numpy(dtype=float)
        rng = np.random.default_rng(cfg.seed + 100 + offset)
        means = np.array(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(cfg.bootstrap_iterations)
            ]
        )
        rows.append(
            {
                "candidate": candidate,
                "events": len(values),
                "mean_primary_net_bp": float(values.mean() * 10_000),
                "lower_95_primary_net_bp": float(np.quantile(means, 0.025) * 10_000),
                "upper_95_primary_net_bp": float(np.quantile(means, 0.975) * 10_000),
            }
        )
    return pd.DataFrame(rows)


def audit_v205(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cfg: V205Config = V205Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        local_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        delayed = delayed_summary[
            (delayed_summary["candidate"].eq(candidate))
            & (delayed_summary["scope"].eq("all"))
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
            "gross_exceeds_20bp_cost": (
                observed_gross > cfg.primary_round_trip_cost,
                observed_gross * 10_000,
                cfg.primary_round_trip_cost * 10_000,
            ),
            "all_period_primary_net_positive": (
                min(period_net) > 0,
                min(period_net),
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
            "bootstrap_lower_95_positive": (
                float(boot["lower_95_primary_net_bp"]) > 0,
                float(boot["lower_95_primary_net_bp"]),
                0.0,
            ),
        }
        for check, (passed, value, threshold) in checks.items():
            rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": value,
                    "threshold": threshold,
                }
            )
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
                "bootstrap_lower_95_primary_net_bp": float(
                    boot["lower_95_primary_net_bp"]
                ),
                "eligible": all(passed for passed, _, _ in checks.values()),
                "status": (
                    "offline_candidate_natural_forward_only"
                    if all(passed for passed, _, _ in checks.values())
                    else "rejected"
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(outcomes)


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_candidate_natural_forward_only"
        if outcome["eligible"].any()
        else "reject_aggtrade_flow_exhaustion_candidates"
    )
    text = [
        "# v20.5 AggTrade Flow-Exhaustion Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary[summary["scope"].eq("all")].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "The reveal follows the frozen v20.5 preregistration. The primary and "
        "stress results charge 20/40 bp round-trip book costs. A positive result "
        "would remain post-hoc and natural-forward-only.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v205_reveal(
    receiver_features_path: Path = RECEIVER_FEATURES_PATH,
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V205Config = V205Config(),
) -> dict[str, Path]:
    receivers = pd.read_parquet(receiver_features_path)
    candidates = pd.read_parquet(candidate_features_path)
    for frame in (receivers, candidates):
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    returns = close.pct_change(fill_method=None)
    risk = build_v187_monthly_risk(
        returns,
        candidates["feature_time"].min(),
        candidates["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    events = build_v205_events(candidates, risk, close, cfg)
    summary = summarize_v205(events)
    delayed_events = build_v205_events(
        candidates, risk, close, cfg, entry_delay_bars=1
    )
    delayed_summary = summarize_v205(delayed_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (2, 4):
        horizon_events = build_v205_events(
            candidates, risk, close, cfg, holding_bars=horizon
        )
        local_summary = summarize_v205(horizon_events)
        local_summary["holding_bars"] = horizon
        horizon_frames.append(local_summary)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    baseline_features = build_rfx1_baseline_features(receivers)
    baseline_events = build_v205_events(baseline_features, risk, close, cfg)
    random_controls = random_v205_controls(events, baseline_events, cfg)
    bootstrap = bootstrap_v205(events, cfg)
    cost_frontier = build_cost_frontier(events)
    gates, outcome = audit_v205(
        events,
        summary,
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
    risk.to_parquet(outputs["risk"], index=False)
    for frame, path in (
        (events, outputs["events"]),
        (delayed_events, outputs["delayed_events"]),
    ):
        serial = frame.copy()
        for column in (
            "weights",
            "symbol_contributions",
            "symbol_betas",
            "symbol_future_returns",
        ):
            serial[column] = serial[column].map(_serialize_mapping)
        serial.to_parquet(path, index=False)
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    horizon_summary.to_csv(outputs["horizons"], index=False)
    cost_frontier.to_csv(outputs["cost_frontier"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return outputs


__all__ = [
    "V205Config",
    "audit_v205",
    "bootstrap_v205",
    "build_cost_frontier",
    "build_rfx1_baseline_features",
    "build_v205_events",
    "random_v205_controls",
    "state_spread_weights",
    "summarize_v205",
    "write_v205_reveal",
]
