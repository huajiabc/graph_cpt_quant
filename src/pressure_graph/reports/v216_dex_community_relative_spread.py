"""Second-stage DEX-conditioned community relative-spread diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    KLINE_ROOT,
    _month,
    load_v178_market_data,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)
from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    REPORT_ROOT as V212_REPORT_ROOT,
)
from pressure_graph.reports.v215_dex_community_relative_spread_feature_audit import (
    RELATIVE_SPREAD,
    REPORT_ROOT as V215_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v21_6_dex_community_relative_spread")
FINDINGS_PATH = Path("docs/v216_dex_community_relative_spread_findings_2026_07_17.md")
CANDIDATE_FEATURES_PATH = V215_REPORT_ROOT / "relative_spread_feature_events.parquet"
EVENT_FEATURES_PATH = V212_REPORT_ROOT / "dex_community_event_features.parquet"
ELIGIBLE_PERIODS = ("development", "validation", "holdout")


@dataclass(frozen=True)
class V216Config:
    holding_bars: int = 48
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    minimum_leg_size: int = 2
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 21_600
    minimum_events: int = 200
    minimum_period_events: int = 30
    minimum_active_months: int = 8
    maximum_contribution_share: float = 0.35


def _symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def beta_neutral_spread_weights(
    laggards: list[str],
    leaders: list[str],
    beta: pd.Series,
    direction: float,
) -> dict[str, float]:
    if (
        not laggards
        or len(laggards) != len(leaders)
        or set(laggards) & set(leaders)
        or direction not in (-1.0, 1.0)
    ):
        return {}
    raw = {symbol: 0.5 * direction / len(laggards) for symbol in laggards}
    raw.update({symbol: -0.5 * direction / len(leaders) for symbol in leaders})
    hedge = -float(sum(weight * float(beta[symbol]) for symbol, weight in raw.items()))
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def build_v216_events(
    candidate_features: pd.DataFrame,
    event_features: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V216Config = V216Config(),
    *,
    holding_bars: int | None = None,
    additional_entry_delay_bars: int = 0,
    include_transition: bool = False,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    context = event_features.set_index("event_id")["community_peer_symbols"]
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    rows: list[dict[str, object]] = []
    for item in candidate_features.itertuples(index=False):
        if not include_transition and item.period not in ELIGIBLE_PERIODS:
            continue
        feature_time = pd.Timestamp(item.feature_time)
        entry_time = pd.Timestamp(item.entry_time) + pd.Timedelta(
            minutes=15 * additional_entry_delay_bars
        )
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        month = _month(feature_time)
        laggards = _symbols(item.laggard_symbols)
        leaders = _symbols(item.leader_symbols)
        selected = laggards + leaders
        endpoint = close.loc[[entry_time, exit_time], [BTC, *selected]]
        beta = pd.Series(
            {symbol: risk_lookup.get((month, symbol), np.nan) for symbol in selected},
            dtype=float,
        )
        if (
            endpoint.isna().any().any()
            or beta.isna().any()
            or len(laggards) < cfg.minimum_leg_size
            or len(laggards) != len(leaders)
        ):
            continue
        weights = beta_neutral_spread_weights(laggards, leaders, beta, float(item.source_direction))
        if not weights:
            continue
        future = endpoint.loc[exit_time].div(endpoint.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weight * future[symbol]) for symbol, weight in weights.items()
        }
        gross = float(sum(contributions.values()))
        full_pool = _symbols(context.get(item.event_id, ""))
        pool_endpoint = close.loc[
            [entry_time, exit_time],
            [symbol for symbol in full_pool if symbol in close.columns],
        ]
        peer_pool = [
            symbol
            for symbol in full_pool
            if symbol in pool_endpoint.columns
            and pool_endpoint[symbol].notna().all()
            and np.isfinite(risk_lookup.get((month, symbol), np.nan))
        ]
        pool_beta = {symbol: float(risk_lookup.loc[(month, symbol)]) for symbol in peer_pool}
        pool_future = (
            close.loc[exit_time, peer_pool].div(close.loc[entry_time, peer_pool]).sub(1.0).to_dict()
        )
        pool_future[BTC] = float(future[BTC])
        rows.append(
            {
                **item._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_day": entry_time.floor("D"),
                "holding_bars": horizon,
                "additional_entry_delay_bars": additional_entry_delay_bars,
                "realized_leg_size": len(laggards),
                "btc_hedge_weight": float(weights[BTC]),
                "alt_dollar_exposure": float(
                    sum(weight for symbol, weight in weights.items() if symbol != BTC)
                ),
                "residual_btc_beta": float(
                    weights[BTC] + sum(weights[symbol] * float(beta[symbol]) for symbol in selected)
                ),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "laggard_contribution": float(sum(contributions[symbol] for symbol in laggards)),
                "leader_contribution": float(sum(contributions[symbol] for symbol in leaders)),
                "btc_hedge_contribution": float(contributions[BTC]),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": -gross - cfg.primary_round_trip_cost,
                "weights": weights,
                "symbol_contributions": contributions,
                "peer_pool_symbols": peer_pool,
                "peer_pool_betas": pool_beta,
                "peer_pool_future_returns": pool_future,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["entry_time", "event_id"]).reset_index(drop=True)


def summarize_v216(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", *ELIGIBLE_PERIODS):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": RELATIVE_SPREAD,
                "scope": scope,
                "events": len(sample),
                "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                "active_months": sample["entry_month"].nunique() if len(sample) else 0,
                "source_symbols": sample["source_symbol"].nunique() if len(sample) else 0,
                "mean_leg_size": (
                    float(sample["realized_leg_size"].mean()) if len(sample) else math.nan
                ),
                "mean_laggard_bp": (
                    float(sample["laggard_contribution"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_leader_bp": (
                    float(sample["leader_contribution"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_btc_hedge_bp": (
                    float(sample["btc_hedge_contribution"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_gross_bp": (
                    float(sample["gross_return"].mean() * 10_000) if len(sample) else math.nan
                ),
                "mean_primary_net_bp": (
                    float(sample["primary_net_return"].mean() * 10_000) if len(sample) else math.nan
                ),
                "mean_stress_net_bp": (
                    float(sample["stress_net_return"].mean() * 10_000) if len(sample) else math.nan
                ),
                "mean_reversed_primary_net_bp": (
                    float(sample["reversed_primary_net_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "positive_primary_fraction": (
                    float(sample["primary_net_return"].gt(0).mean()) if len(sample) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def random_v216_controls(
    events: pd.DataFrame,
    cfg: V216Config = V216Config(),
) -> pd.DataFrame:
    totals = np.zeros(cfg.random_iterations, dtype=float)
    usable = 0
    rng = np.random.default_rng(cfg.seed)
    for item in events.itertuples(index=False):
        pool = list(item.peer_pool_symbols)
        size = int(item.realized_leg_size)
        if len(pool) < 2 * size:
            continue
        beta = pd.Series(item.peer_pool_betas, dtype=float).reindex(pool).to_numpy()
        future_series = pd.Series(item.peer_pool_future_returns, dtype=float)
        future = future_series.reindex(pool).to_numpy()
        order = np.argsort(rng.random((cfg.random_iterations, len(pool))), axis=1)[:, : 2 * size]
        lag_index = order[:, :size]
        lead_index = order[:, size:]
        raw_beta = (
            0.5
            * float(item.source_direction)
            * (beta[lag_index].mean(axis=1) - beta[lead_index].mean(axis=1))
        )
        raw_alt = (
            0.5
            * float(item.source_direction)
            * (future[lag_index].mean(axis=1) - future[lead_index].mean(axis=1))
        )
        hedge = -raw_beta
        totals += (raw_alt + hedge * float(future_series[BTC])) / (1.0 + np.abs(hedge))
        usable += 1
    means = totals / usable
    return pd.DataFrame(
        {
            "candidate": RELATIVE_SPREAD,
            "iteration": range(cfg.random_iterations),
            "events": usable,
            "mean_gross_return": means,
        }
    )


def day_block_bootstrap_v216(
    events: pd.DataFrame,
    cfg: V216Config = V216Config(),
) -> pd.DataFrame:
    daily = events.groupby("entry_day")["primary_net_return"].agg(["sum", "count"])
    rng = np.random.default_rng(cfg.seed + 100)
    means: list[float] = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(daily), size=len(daily))
        sample = daily.iloc[indices]
        means.append(float(sample["sum"].sum() / sample["count"].sum()))
    return pd.DataFrame(
        [
            {
                "candidate": RELATIVE_SPREAD,
                "events": len(events),
                "active_days": len(daily),
                "mean_primary_net_bp": float(events["primary_net_return"].mean() * 10_000),
                "lower_95_primary_net_bp": float(np.quantile(means, 0.025) * 10_000),
                "upper_95_primary_net_bp": float(np.quantile(means, 0.975) * 10_000),
            }
        ]
    )


def _positive_share(events: pd.DataFrame, column: str) -> float:
    grouped = events.groupby(column)["primary_net_return"].sum().clip(lower=0.0)
    total = float(grouped.sum())
    return float(grouped.max() / total) if total > 0 else 1.0


def concentration_v216(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": RELATIVE_SPREAD,
                "maximum_month_positive_pnl_share": _positive_share(events, "entry_month"),
                "maximum_source_positive_pnl_share": _positive_share(events, "source_symbol"),
            }
        ]
    )


def cost_frontier_v216(events: pd.DataFrame) -> pd.DataFrame:
    gross_bp = float(events["gross_return"].mean() * 10_000)
    return pd.DataFrame(
        [
            {
                "candidate": RELATIVE_SPREAD,
                "round_trip_cost_bp": cost,
                "events": len(events),
                "mean_gross_bp": gross_bp,
                "mean_net_bp": gross_bp - cost,
            }
            for cost in (5, 10, 20, 40)
        ]
    )


def audit_v216(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    concentration: pd.DataFrame,
    cfg: V216Config = V216Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = summary.set_index("scope")
    delayed = delayed_summary.set_index("scope").loc["all"]
    placebo = placebo_summary.set_index("scope").loc["all"]
    boot = bootstrap.iloc[0]
    conc = concentration.iloc[0]
    gross = float(events["gross_return"].mean())
    percentile = float(random_controls["mean_gross_return"].le(gross).mean())
    period_events = [int(table.loc[period, "events"]) for period in ELIGIBLE_PERIODS]
    period_net = [float(table.loc[period, "mean_primary_net_bp"]) for period in ELIGIBLE_PERIODS]
    checks = {
        "minimum_total_events": (
            len(events) >= cfg.minimum_events,
            len(events),
            cfg.minimum_events,
        ),
        "minimum_each_period_events": (
            min(period_events) >= cfg.minimum_period_events,
            min(period_events),
            cfg.minimum_period_events,
        ),
        "minimum_active_months": (
            int(table.loc["all", "active_months"]) >= cfg.minimum_active_months,
            int(table.loc["all", "active_months"]),
            cfg.minimum_active_months,
        ),
        "gross_exceeds_20bp_cost": (
            gross > cfg.primary_round_trip_cost,
            gross * 10_000,
            20.0,
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
        "reversed_primary_net_negative": (
            float(table.loc["all", "mean_reversed_primary_net_bp"]) < 0,
            float(table.loc["all", "mean_reversed_primary_net_bp"]),
            0.0,
        ),
        "random_rank_percentile_at_least_0_95": (
            percentile >= 0.95,
            percentile,
            0.95,
        ),
        "day_bootstrap_lower_95_positive": (
            float(boot["lower_95_primary_net_bp"]) > 0,
            float(boot["lower_95_primary_net_bp"]),
            0.0,
        ),
        "shifted_24h_placebo_net_nonpositive": (
            float(placebo["mean_primary_net_bp"]) <= 0,
            float(placebo["mean_primary_net_bp"]),
            0.0,
        ),
        "month_contribution_at_most_35pct": (
            float(conc["maximum_month_positive_pnl_share"]) <= cfg.maximum_contribution_share,
            float(conc["maximum_month_positive_pnl_share"]),
            cfg.maximum_contribution_share,
        ),
        "source_contribution_at_most_35pct": (
            float(conc["maximum_source_positive_pnl_share"]) <= cfg.maximum_contribution_share,
            float(conc["maximum_source_positive_pnl_share"]),
            cfg.maximum_contribution_share,
        ),
    }
    gates = pd.DataFrame(
        [
            {
                "candidate": RELATIVE_SPREAD,
                "check": check,
                "passed": bool(passed),
                "value": value,
                "threshold": threshold,
            }
            for check, (passed, value, threshold) in checks.items()
        ]
    )
    economically_interesting = all(passed for passed, _, _ in checks.values())
    outcome = pd.DataFrame(
        [
            {
                "candidate": RELATIVE_SPREAD,
                "events": len(events),
                "mean_gross_bp": gross * 10_000,
                "mean_primary_net_bp": float(table.loc["all", "mean_primary_net_bp"]),
                "mean_stress_net_bp": float(table.loc["all", "mean_stress_net_bp"]),
                "random_control_percentile": percentile,
                "day_bootstrap_lower_95_primary_net_bp": float(boot["lower_95_primary_net_bp"]),
                "economically_interesting": economically_interesting,
                "promotion_eligible": False,
                "status": (
                    "research_only_requires_new_natural_forward"
                    if economically_interesting
                    else "rejected"
                ),
            }
        ]
    )
    return gates, outcome


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _serialize_symbols(value: list[str]) -> str:
    return "|".join(value)


def _write_event_frame(frame: pd.DataFrame, path: Path) -> None:
    serial = frame.copy()
    for column in (
        "weights",
        "symbol_contributions",
        "peer_pool_betas",
        "peer_pool_future_returns",
    ):
        serial[column] = serial[column].map(_serialize_mapping)
    serial["peer_pool_symbols"] = serial["peer_pool_symbols"].map(_serialize_symbols)
    serial.to_parquet(path, index=False)


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    path: Path,
) -> None:
    verdict = str(outcome.iloc[0]["status"])
    controls = pd.DataFrame(
        [
            {
                "delayed_gross_bp": delayed_summary.set_index("scope").loc["all", "mean_gross_bp"],
                "delayed_net20_bp": delayed_summary.set_index("scope").loc[
                    "all", "mean_primary_net_bp"
                ],
                "placebo_24h_gross_bp": placebo_summary.set_index("scope").loc[
                    "all", "mean_gross_bp"
                ],
                "placebo_24h_net20_bp": placebo_summary.set_index("scope").loc[
                    "all", "mean_primary_net_bp"
                ],
            }
        ]
    )
    text = [
        "# v21.6 DEX Community Relative-Spread Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Chronological results:",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Timing controls:",
        "",
        controls.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Alternate horizons:",
        "",
        horizon_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Concentration:",
        "",
        concentration.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This is a second-stage same-history diagnostic. Even a passing economic "
        "result is not promotion evidence and requires genuinely new natural-forward data.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v216_reveal(
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    event_features_path: Path = EVENT_FEATURES_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V216Config = V216Config(),
) -> dict[str, Path]:
    candidates = pd.read_parquet(candidate_features_path)
    event_features = pd.read_parquet(event_features_path)
    for frame in (candidates, event_features):
        for column in ("event_time", "event_available_time", "feature_time", "entry_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    close, _ = load_v178_market_data(kline_root)
    risk = build_v187_monthly_risk(
        close.pct_change(fill_method=None),
        candidates["feature_time"].min(),
        candidates["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    events = build_v216_events(candidates, event_features, risk, close, cfg)
    summary = summarize_v216(events)
    delayed_events = build_v216_events(
        candidates,
        event_features,
        risk,
        close,
        cfg,
        additional_entry_delay_bars=1,
    )
    delayed_summary = summarize_v216(delayed_events)
    placebo_events = build_v216_events(
        candidates,
        event_features,
        risk,
        close,
        cfg,
        additional_entry_delay_bars=96,
    )
    placebo_summary = summarize_v216(placebo_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (16, 96):
        local_events = build_v216_events(
            candidates,
            event_features,
            risk,
            close,
            cfg,
            holding_bars=horizon,
        )
        local = summarize_v216(local_events)
        local["holding_bars"] = horizon
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v216_controls(events, cfg)
    bootstrap = day_block_bootstrap_v216(events, cfg)
    concentration = concentration_v216(events)
    cost_frontier = cost_frontier_v216(events)
    gates, outcome = audit_v216(
        events,
        summary,
        delayed_summary,
        placebo_summary,
        random_controls,
        bootstrap,
        concentration,
        cfg,
    )
    root = ensure_dir(report_root)
    outputs = {
        "risk": root / "monthly_btc_risk.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "placebo_events": root / "shifted_24h_placebo_events.parquet",
        "placebo_summary": root / "shifted_24h_placebo_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "random_controls": root / "random_controls.parquet",
        "bootstrap": root / "day_block_bootstrap_summary.csv",
        "concentration": root / "concentration_summary.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    risk.to_parquet(outputs["risk"], index=False)
    for frame, key in (
        (events, "events"),
        (delayed_events, "delayed_events"),
        (placebo_events, "placebo_events"),
    ):
        _write_event_frame(frame, outputs[key])
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    placebo_summary.to_csv(outputs["placebo_summary"], index=False)
    horizon_summary.to_csv(outputs["horizons"], index=False)
    cost_frontier.to_csv(outputs["cost_frontier"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    concentration.to_csv(outputs["concentration"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(
        outcome,
        summary,
        delayed_summary,
        placebo_summary,
        horizon_summary,
        concentration,
        findings_path,
    )
    return outputs


__all__ = [
    "V216Config",
    "audit_v216",
    "beta_neutral_spread_weights",
    "build_v216_events",
    "random_v216_controls",
    "summarize_v216",
    "write_v216_reveal",
]
