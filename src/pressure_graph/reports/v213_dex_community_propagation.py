"""Preregistered DEX-attention to graph-community propagation reveal."""

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
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)
from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    ALL_PEERS,
    CANDIDATES,
    REPORT_ROOT as V212_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v21_3_dex_community_propagation")
FINDINGS_PATH = Path("docs/v213_dex_community_propagation_findings_2026_07_17.md")
CANDIDATE_FEATURES_PATH = V212_REPORT_ROOT / "candidate_feature_events.parquet"
EVENT_FEATURES_PATH = V212_REPORT_ROOT / "dex_community_event_features.parquet"
ELIGIBLE_PERIODS = ("development", "validation", "holdout")
GLOBAL_RANDOM = "GLOBAL_MONTHLY_GRAPH_UNIVERSE"
COMMUNITY_RANDOM = "SAME_COMMUNITY_PEERS"


@dataclass(frozen=True)
class V213Config:
    holding_bars: int = 48
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    minimum_all_peers: int = 4
    minimum_laggard_peers: int = 3
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 21_300
    minimum_events: int = 200
    minimum_period_events: int = 30
    minimum_active_months: int = 8
    maximum_contribution_share: float = 0.35


def _parse_symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def beta_neutral_directional_weights(
    symbols: list[str],
    beta: pd.Series,
    direction: float,
) -> dict[str, float]:
    if not symbols or direction not in (-1.0, 1.0):
        return {}
    raw = {symbol: direction / len(symbols) for symbol in symbols}
    hedge = -float(sum(raw[symbol] * float(beta[symbol]) for symbol in symbols))
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def _minimum_selection(candidate: str, cfg: V213Config) -> int:
    return cfg.minimum_all_peers if candidate == ALL_PEERS else cfg.minimum_laggard_peers


def build_v213_events(
    candidate_features: pd.DataFrame,
    event_features: pd.DataFrame,
    membership: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V213Config = V213Config(),
    *,
    holding_bars: int | None = None,
    additional_entry_delay_bars: int = 0,
    selection_mode: str = "candidate",
    include_transition: bool = False,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    context = event_features.set_index("event_id")["community_peer_symbols"]
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    membership_groups = {
        pd.Timestamp(month): sorted(group["symbol"].astype(str).unique())
        for month, group in membership.groupby("month_start", sort=True)
    }
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
        source = str(item.source_symbol)
        direction = float(item.source_direction)
        feature_selection = _parse_symbols(item.selection_symbols)
        requested = [source] if selection_mode == "source" else feature_selection
        endpoint_prices = close.loc[[entry_time, exit_time]]
        beta_by_symbol = pd.Series(
            {
                symbol: risk_lookup.get((month, symbol), np.nan)
                for symbol in close.columns
                if symbol != BTC
            },
            dtype=float,
        )
        available = [
            symbol
            for symbol in beta_by_symbol.index.astype(str)
            if np.isfinite(beta_by_symbol[symbol]) and endpoint_prices[symbol].notna().all()
        ]
        selected = [symbol for symbol in requested if symbol in available]
        minimum = 1 if selection_mode == "source" else _minimum_selection(item.candidate, cfg)
        if len(selected) < minimum or endpoint_prices[BTC].isna().any():
            continue
        weights = beta_neutral_directional_weights(
            selected, beta_by_symbol.reindex(selected), direction
        )
        if not weights:
            continue
        future = endpoint_prices.loc[exit_time].div(endpoint_prices.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weight * future[symbol]) for symbol, weight in weights.items()
        }
        gross = float(sum(contributions.values()))
        global_pool = [
            symbol
            for symbol in membership_groups.get(month, [])
            if symbol != source and symbol in available
        ]
        community_pool = [
            symbol
            for symbol in _parse_symbols(context.get(item.event_id, ""))
            if symbol != source and symbol in available
        ]
        universe = sorted(set(global_pool) | set(community_pool) | set(selected))
        rows.append(
            {
                **item._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_day": entry_time.floor("D"),
                "selection_mode": selection_mode,
                "additional_entry_delay_bars": additional_entry_delay_bars,
                "holding_bars": horizon,
                "feature_selection_count": len(feature_selection),
                "realized_selection_count": len(selected),
                "realized_selection_symbols": "|".join(sorted(selected)),
                "btc_hedge_weight": float(weights[BTC]),
                "residual_btc_beta": float(
                    weights[BTC]
                    + sum(weights[symbol] * float(beta_by_symbol[symbol]) for symbol in selected)
                ),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "alt_contribution": float(sum(contributions[symbol] for symbol in selected)),
                "btc_hedge_contribution": float(contributions[BTC]),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": -gross - cfg.primary_round_trip_cost,
                "break_even_cost_bp": gross * 10_000,
                "weights": weights,
                "symbol_contributions": contributions,
                "global_pool_symbols": global_pool,
                "community_pool_symbols": community_pool,
                "universe_betas": beta_by_symbol.reindex(universe).to_dict(),
                "universe_future_returns": future.reindex([BTC, *universe]).to_dict(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_time", "candidate", "event_id"])
        .reset_index(drop=True)
    )


def summarize_v213(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for scope in ("all", *ELIGIBLE_PERIODS):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": sample["entry_month"].nunique() if len(sample) else 0,
                    "source_symbols": sample["source_symbol"].nunique() if len(sample) else 0,
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
                        float(sample["gross_return"].mean() * 10_000) if len(sample) else math.nan
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


def build_v213_cost_frontier(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        gross_bp = float(local["gross_return"].mean() * 10_000)
        for cost_bp in (5, 10, 20, 40):
            rows.append(
                {
                    "candidate": candidate,
                    "round_trip_cost_bp": cost_bp,
                    "events": len(local),
                    "mean_gross_bp": gross_bp,
                    "mean_net_bp": gross_bp - cost_bp,
                }
            )
    return pd.DataFrame(rows)


def random_v213_controls(
    events: pd.DataFrame,
    cfg: V213Config = V213Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
        family = GLOBAL_RANDOM if candidate == ALL_PEERS else COMMUNITY_RANDOM
        totals = np.zeros(cfg.random_iterations, dtype=float)
        usable = 0
        rng = np.random.default_rng(cfg.seed + offset)
        for item in local.itertuples(index=False):
            pool = (
                list(item.global_pool_symbols)
                if family == GLOBAL_RANDOM
                else list(item.community_pool_symbols)
            )
            count = int(item.realized_selection_count)
            if len(pool) < count:
                continue
            betas = pd.Series(item.universe_betas, dtype=float).reindex(pool).to_numpy()
            future_series = pd.Series(item.universe_future_returns, dtype=float)
            future = future_series.reindex(pool).to_numpy()
            scores = rng.random((cfg.random_iterations, len(pool)))
            indices = np.argpartition(scores, count - 1, axis=1)[:, :count]
            mean_beta = betas[indices].mean(axis=1)
            mean_alt_future = future[indices].mean(axis=1)
            direction = float(item.source_direction)
            gross = (
                direction
                * (mean_alt_future - mean_beta * float(future_series[BTC]))
                / (1.0 + np.abs(mean_beta))
            )
            totals += gross
            usable += 1
        means = totals / usable
        rows.extend(
            {
                "candidate": candidate,
                "control_family": family,
                "iteration": iteration,
                "events": usable,
                "mean_gross_return": float(means[iteration]),
            }
            for iteration in range(cfg.random_iterations)
        )
    return pd.DataFrame(rows)


def day_block_bootstrap_v213(
    events: pd.DataFrame,
    cfg: V213Config = V213Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
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
                "mean_primary_net_bp": float(local["primary_net_return"].mean() * 10_000),
                "lower_95_primary_net_bp": float(np.quantile(means, 0.025) * 10_000),
                "upper_95_primary_net_bp": float(np.quantile(means, 0.975) * 10_000),
            }
        )
    return pd.DataFrame(rows)


def _positive_contribution_share(events: pd.DataFrame, column: str) -> float:
    grouped = events.groupby(column)["primary_net_return"].sum().clip(lower=0.0)
    total = float(grouped.sum())
    return float(grouped.max() / total) if total > 0 else 1.0


def concentration_v213(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        rows.append(
            {
                "candidate": candidate,
                "maximum_month_positive_pnl_share": _positive_contribution_share(
                    local, "entry_month"
                ),
                "maximum_source_positive_pnl_share": _positive_contribution_share(
                    local, "source_symbol"
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v213(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    concentration: pd.DataFrame,
    cfg: V213Config = V213Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gates: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        table = summary[summary["candidate"].eq(candidate)].set_index("scope")
        delayed = delayed_summary[
            (delayed_summary["candidate"].eq(candidate)) & delayed_summary["scope"].eq("all")
        ].iloc[0]
        placebo = placebo_summary[
            (placebo_summary["candidate"].eq(candidate)) & placebo_summary["scope"].eq("all")
        ].iloc[0]
        source = source_summary[
            (source_summary["candidate"].eq(candidate)) & source_summary["scope"].eq("all")
        ].iloc[0]
        boot = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        conc = concentration[concentration["candidate"].eq(candidate)].iloc[0]
        random = random_controls[random_controls["candidate"].eq(candidate)]["mean_gross_return"]
        gross = float(local["gross_return"].mean())
        percentile = float(random.le(gross).mean())
        period_events = [int(table.loc[period, "events"]) for period in ELIGIBLE_PERIODS]
        period_net = [
            float(table.loc[period, "mean_primary_net_bp"]) for period in ELIGIBLE_PERIODS
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
            "random_control_percentile_at_least_0_95": (
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
            "candidate_gross_exceeds_source_control": (
                gross * 10_000 > float(source["mean_gross_bp"]),
                gross * 10_000 - float(source["mean_gross_bp"]),
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
        for check, (passed, value, threshold) in checks.items():
            gates.append(
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
                "mean_gross_bp": gross * 10_000,
                "mean_primary_net_bp": float(table.loc["all", "mean_primary_net_bp"]),
                "mean_stress_net_bp": float(table.loc["all", "mean_stress_net_bp"]),
                "random_control_percentile": percentile,
                "day_bootstrap_lower_95_primary_net_bp": float(boot["lower_95_primary_net_bp"]),
                "eligible": eligible,
                "status": ("offline_candidate_natural_forward_only" if eligible else "rejected"),
            }
        )
    return pd.DataFrame(gates), pd.DataFrame(outcomes)


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _serialize_symbols(value: list[str]) -> str:
    return "|".join(value)


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_candidate_natural_forward_only"
        if bool(outcome["eligible"].any())
        else "reject_dex_community_propagation_candidates"
    )
    control = delayed_summary[delayed_summary["scope"].eq("all")][
        ["candidate", "mean_gross_bp", "mean_primary_net_bp"]
    ].rename(
        columns={
            "mean_gross_bp": "delayed_gross_bp",
            "mean_primary_net_bp": "delayed_net20_bp",
        }
    )
    control = control.merge(
        placebo_summary[placebo_summary["scope"].eq("all")][
            ["candidate", "mean_gross_bp", "mean_primary_net_bp"]
        ].rename(
            columns={
                "mean_gross_bp": "placebo_24h_gross_bp",
                "mean_primary_net_bp": "placebo_24h_net20_bp",
            }
        ),
        on="candidate",
    ).merge(
        source_summary[source_summary["scope"].eq("all")][
            ["candidate", "mean_gross_bp", "mean_primary_net_bp"]
        ].rename(
            columns={
                "mean_gross_bp": "source_only_gross_bp",
                "mean_primary_net_bp": "source_only_net20_bp",
            }
        ),
        on="candidate",
    )
    text = [
        "# v21.3 DEX Community-Propagation Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Chronological primary results:",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Timing and attribution controls:",
        "",
        control.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Alternate horizons:",
        "",
        horizon_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Concentration:",
        "",
        concentration.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The reveal follows the frozen v21.3 preregistration. The four-event "
        "DEX-vendor transition interval is excluded from eligibility statistics. "
        "Primary/stress book costs are 20/40 bp round trip on unit gross.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def _write_event_frame(frame: pd.DataFrame, path: Path) -> None:
    serial = frame.copy()
    serial["weights"] = serial["weights"].map(_serialize_mapping)
    serial["symbol_contributions"] = serial["symbol_contributions"].map(_serialize_mapping)
    for column in ("global_pool_symbols", "community_pool_symbols"):
        serial[column] = serial[column].map(_serialize_symbols)
    serial["universe_betas"] = serial["universe_betas"].map(_serialize_mapping)
    serial["universe_future_returns"] = serial["universe_future_returns"].map(_serialize_mapping)
    serial.to_parquet(path, index=False)


def write_v213_reveal(
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    event_features_path: Path = EVENT_FEATURES_PATH,
    membership_path: Path = MEMBERSHIP_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V213Config = V213Config(),
) -> dict[str, Path]:
    candidates = pd.read_parquet(candidate_features_path)
    event_features = pd.read_parquet(event_features_path)
    for frame in (candidates, event_features):
        for column in ("event_time", "event_available_time", "feature_time", "entry_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    membership = load_v195_membership(membership_path)
    close, _ = load_v178_market_data(kline_root)
    risk = build_v187_monthly_risk(
        close.pct_change(fill_method=None),
        candidates["feature_time"].min(),
        candidates["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    events = build_v213_events(candidates, event_features, membership, risk, close, cfg)
    summary = summarize_v213(events)
    delayed_events = build_v213_events(
        candidates,
        event_features,
        membership,
        risk,
        close,
        cfg,
        additional_entry_delay_bars=1,
    )
    delayed_summary = summarize_v213(delayed_events)
    placebo_events = build_v213_events(
        candidates,
        event_features,
        membership,
        risk,
        close,
        cfg,
        additional_entry_delay_bars=96,
    )
    placebo_summary = summarize_v213(placebo_events)
    source_events = build_v213_events(
        candidates,
        event_features,
        membership,
        risk,
        close,
        cfg,
        selection_mode="source",
    )
    source_summary = summarize_v213(source_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (16, 96):
        horizon_events = build_v213_events(
            candidates,
            event_features,
            membership,
            risk,
            close,
            cfg,
            holding_bars=horizon,
        )
        local = summarize_v213(horizon_events)
        local["holding_bars"] = horizon
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v213_controls(events, cfg)
    bootstrap = day_block_bootstrap_v213(events, cfg)
    concentration = concentration_v213(events)
    cost_frontier = build_v213_cost_frontier(events)
    gates, outcome = audit_v213(
        events,
        summary,
        delayed_summary,
        placebo_summary,
        source_summary,
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
        "source_events": root / "source_only_events.parquet",
        "source_summary": root / "source_only_summary.csv",
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
        (source_events, "source_events"),
    ):
        _write_event_frame(frame, outputs[key])
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    placebo_summary.to_csv(outputs["placebo_summary"], index=False)
    source_summary.to_csv(outputs["source_summary"], index=False)
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
        source_summary,
        horizon_summary,
        concentration,
        findings_path,
    )
    return outputs


__all__ = [
    "V213Config",
    "audit_v213",
    "beta_neutral_directional_weights",
    "build_v213_events",
    "day_block_bootstrap_v213",
    "random_v213_controls",
    "summarize_v213",
    "write_v213_reveal",
]
