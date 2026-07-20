"""Preregistered reveal for graph-bucket reference-price transmission alpha."""
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
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)
from pressure_graph.reports.v200_reference_price_transmission_feature_audit import (
    COMMUNITY,
    GLOBAL,
    REFERENCE_LAG,
    REPORT_ROOT as V200_REPORT_ROOT,
    TRADE_OVERSHOOT,
)


REPORT_ROOT = Path("reports/v20_1_reference_price_transmission")
FINDINGS_PATH = Path(
    "docs/v201_reference_price_transmission_findings_2026_07_17.md"
)
FEATURE_EVENTS_PATH = V200_REPORT_ROOT / "candidate_feature_events.parquet"
GLOBAL_REFERENCE = "RPT1_GLOBAL_REFERENCE_RESIDUAL_CATCHUP"
GLOBAL_TRADE = "RPT2_GLOBAL_TRADE_OVERSHOOT_FADE"
COMMUNITY_REFERENCE = "RPT3_COMMUNITY_REFERENCE_RESIDUAL_CATCHUP"
COMMUNITY_TRADE = "RPT4_COMMUNITY_TRADE_OVERSHOOT_FADE"
CANDIDATES = (
    GLOBAL_REFERENCE,
    GLOBAL_TRADE,
    COMMUNITY_REFERENCE,
    COMMUNITY_TRADE,
)
CANDIDATE_RULES = {
    GLOBAL_REFERENCE: (GLOBAL, REFERENCE_LAG, "q90", 1.5),
    GLOBAL_TRADE: (GLOBAL, TRADE_OVERSHOOT, "q90", 1.5),
    COMMUNITY_REFERENCE: (COMMUNITY, REFERENCE_LAG, "z2.0", 1.0),
    COMMUNITY_TRADE: (COMMUNITY, TRADE_OVERSHOOT, "z2.0", 1.5),
}


@dataclass(frozen=True)
class V201Config:
    holding_bars: int = 1
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 20_100


def select_v201_feature_events(
    feature_events: pd.DataFrame,
    candidate: str,
) -> pd.DataFrame:
    source_scope, family, source_setting, receiver_threshold = CANDIDATE_RULES[
        candidate
    ]
    selected = feature_events[
        feature_events["source_scope"].eq(source_scope)
        & feature_events["family"].eq(family)
        & feature_events["source_setting"].eq(source_setting)
        & feature_events["receiver_z_threshold"].eq(receiver_threshold)
        & feature_events["receiver_count"].ge(3)
    ].copy()
    selected["candidate"] = candidate
    return selected.sort_values(["feature_time", "community_id"]).reset_index(
        drop=True
    )


def _candidate_direction(candidate: str, source_sign: float) -> float:
    return source_sign if "REFERENCE" in candidate else -source_sign


def beta_neutral_weights(
    receivers: list[str],
    direction: float,
    beta: pd.Series,
) -> dict[str, float]:
    if not receivers:
        return {}
    raw = {symbol: direction / len(receivers) for symbol in receivers}
    hedge = -float(sum(raw[symbol] * float(beta[symbol]) for symbol in receivers))
    gross = float(sum(abs(value) for value in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: value / gross for symbol, value in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def _receivers(value: object) -> list[str]:
    return [name for name in str(value).split("|") if name]


def build_v201_events(
    feature_events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    candidate: str,
    cfg: V201Config = V201Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    selected = select_v201_feature_events(feature_events, candidate)
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    rows: list[dict[str, object]] = []
    for item in selected.itertuples(index=False):
        source_time = pd.Timestamp(item.feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        receivers = _receivers(item.receivers)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        beta = pd.Series(
            {
                symbol: risk_lookup.get((_month(source_time), symbol), np.nan)
                for symbol in receivers
            },
            dtype=float,
        )
        prices = close.reindex(
            index=[entry_time, exit_time], columns=[BTC, *receivers]
        )
        if beta.isna().any() or prices.isna().any().any():
            continue
        direction = _candidate_direction(candidate, float(item.source_sign))
        weights = beta_neutral_weights(receivers, direction, beta)
        if not weights:
            continue
        future = prices.loc[exit_time].div(prices.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weight * future[symbol])
            for symbol, weight in weights.items()
        }
        gross = float(sum(contributions.values()))
        residual_beta = float(
            weights[BTC]
            + sum(weights[symbol] * float(beta[symbol]) for symbol in receivers)
        )
        gross_notional = float(sum(abs(value) for value in weights.values()))
        rows.append(
            {
                **item._asdict(),
                "candidate": candidate,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "trade_direction": direction,
                "realized_receiver_count": len(receivers),
                "btc_hedge_weight": weights[BTC],
                "residual_btc_beta": residual_beta,
                "gross_notional": gross_notional,
                "alt_gross_return": float(
                    sum(contributions[symbol] for symbol in receivers)
                ),
                "btc_hedge_return": float(contributions[BTC]),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": (
                    -gross - cfg.primary_round_trip_cost
                ),
                "weights": weights,
                "symbol_contributions": contributions,
            }
        )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(
            ["entry_time", "candidate", "community_id"]
        ).reset_index(drop=True)
    return events


def summarize_v201(events: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_alt_gross_bp": (
                        float(sample["alt_gross_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_btc_hedge_bp": (
                        float(sample["btc_hedge_return"].mean() * 10_000)
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


def _random_contexts(
    events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[dict[str, object]]:
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    membership_lookup = {
        (pd.Timestamp(month), str(community)): sorted(
            set(group["symbol"].astype(str)) - {BTC}
        )
        for (month, community), group in membership.groupby(
            ["month_start", "community_id"], sort=True
        )
    }
    all_alts = sorted(set(close.columns) - {BTC})
    contexts: list[dict[str, object]] = []
    for item in events.itertuples(index=False):
        source_time = pd.Timestamp(item.feature_time)
        if str(item.source_scope) == GLOBAL:
            pool = all_alts
        else:
            pool = membership_lookup.get(
                (_month(source_time), str(item.community_id)), []
            )
        beta = pd.Series(
            {
                symbol: risk_lookup.get((_month(source_time), symbol), np.nan)
                for symbol in pool
            },
            dtype=float,
        ).dropna()
        future = close.reindex(
            index=[item.entry_time, item.exit_time], columns=[BTC, *beta.index]
        )
        if future[BTC].isna().any():
            continue
        returns = future.loc[item.exit_time].div(future.loc[item.entry_time]).sub(1.0)
        valid = [symbol for symbol in beta.index if np.isfinite(returns[symbol])]
        count = int(item.realized_receiver_count)
        if len(valid) < count:
            continue
        contexts.append(
            {
                "candidate": str(item.candidate),
                "count": count,
                "direction": float(item.trade_direction),
                "returns": returns.reindex(valid).to_numpy(dtype=float),
                "beta": beta.reindex(valid).to_numpy(dtype=float),
                "btc_return": float(returns[BTC]),
            }
        )
    return contexts


def random_v201_controls(
    events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V201Config = V201Config(),
) -> pd.DataFrame:
    contexts = _random_contexts(events, risk, close, membership)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for context in contexts:
            chosen = rng.choice(
                len(context["returns"]), size=int(context["count"]), replace=False
            )
            direction = float(context["direction"])
            raw = np.full(len(chosen), direction / len(chosen))
            hedge = -float(np.sum(raw * context["beta"][chosen]))
            normalizer = float(np.sum(np.abs(raw)) + abs(hedge))
            gross = float(
                (
                    np.sum(raw * context["returns"][chosen])
                    + hedge * float(context["btc_return"])
                )
                / normalizer
            )
            values[str(context["candidate"])].append(
                gross - cfg.primary_round_trip_cost
            )
        means = {
            candidate: (
                float(np.mean(values[candidate])) if values[candidate] else math.nan
            )
            for candidate in CANDIDATES
        }
        for candidate in CANDIDATES:
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": len(values[candidate]),
                    "mean_primary_net_return": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": max(len(values[candidate]) for candidate in CANDIDATES),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    sample: pd.DataFrame,
    cfg: V201Config,
    offset: int,
) -> tuple[float, float]:
    blocks = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    if not blocks:
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + offset)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(blocks), len(blocks))
        means.append(float(np.mean(np.concatenate([blocks[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_month_concentration(sample: pd.DataFrame) -> float:
    monthly = sample.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def _positive_symbol_concentration(sample: pd.DataFrame) -> float:
    contributions: dict[str, float] = {}
    for value in sample["symbol_contributions"]:
        for symbol, contribution in dict(value).items():
            contributions[symbol] = contributions.get(symbol, 0.0) + float(contribution)
    positive = np.asarray(
        [value for value in contributions.values() if value > 0], dtype=float
    )
    return float(positive.max() / positive.sum()) if positive.size else math.inf


def audit_v201(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V201Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"),
        "mean_primary_net_return",
    ].dropna()
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        sample = events[events["candidate"].eq(candidate)]
        scoped = summary[summary["candidate"].eq(candidate)].set_index("scope")
        delayed = delayed_summary[
            delayed_summary["candidate"].eq(candidate)
            & delayed_summary["scope"].eq("all")
        ]
        local_horizons = horizon_summary[
            horizon_summary["candidate"].eq(candidate)
            & horizon_summary["scope"].eq("all")
        ].set_index("holding_bars")
        real_mean = float(sample["primary_net_return"].mean())
        reversed_mean = float(sample["reversed_primary_net_return"].mean())
        delayed_mean = float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
        low, high = _bootstrap(sample, cfg, offset)
        percentile = float(family.le(real_mean).mean())
        month_concentration = _positive_month_concentration(sample)
        symbol_concentration = _positive_symbol_concentration(sample)
        positive_side = sample[sample["source_sign"].gt(0)][
            "primary_net_return"
        ].mean()
        negative_side = sample[sample["source_sign"].lt(0)][
            "primary_net_return"
        ].mean()
        event_minimum = 500 if candidate.startswith("RPT1") or candidate.startswith("RPT2") else 150
        validation_minimum = 100 if event_minimum == 500 else 40
        holdout_minimum = validation_minimum
        checks: dict[str, tuple[bool, float]] = {
            "full_event_minimum": (
                scoped.loc["all", "events"] >= event_minimum,
                float(scoped.loc["all", "events"]),
            ),
            "validation_event_minimum": (
                scoped.loc["validation", "events"] >= validation_minimum,
                float(scoped.loc["validation", "events"]),
            ),
            "holdout_event_minimum": (
                scoped.loc["holdout", "events"] >= holdout_minimum,
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
            "random_family_percentile_95": (percentile >= 0.95, percentile),
            "beats_reversed_direction": (
                real_mean > reversed_mean,
                (real_mean - reversed_mean) * 10_000,
            ),
            "beats_one_bar_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "holding_30m_positive": (
                local_horizons.loc[2, "mean_primary_net_bp"] > 0,
                float(local_horizons.loc[2, "mean_primary_net_bp"]),
            ),
            "holding_60m_positive": (
                local_horizons.loc[4, "mean_primary_net_bp"] > 0,
                float(local_horizons.loc[4, "mean_primary_net_bp"]),
            ),
            "positive_source_side_positive": (
                positive_side > 0,
                float(positive_side * 10_000),
            ),
            "negative_source_side_positive": (
                negative_side > 0,
                float(negative_side * 10_000),
            ),
            "beta_residual_1e10": (
                sample["residual_btc_beta"].abs().max() <= 1e-10,
                float(sample["residual_btc_beta"].abs().max()),
            ),
            "gross_notional_drift_1e10": (
                sample["gross_notional"].sub(1.0).abs().max() <= 1e-10,
                float(sample["gross_notional"].sub(1.0).abs().max()),
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
        eligible = all(passed for passed, _ in checks.values())
        gate_rows.extend(
            {
                "candidate": candidate,
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for name, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(sample),
                "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
                "break_even_round_trip_cost_bp": float(
                    sample["gross_return"].mean() * 10_000
                ),
                "mean_primary_net_bp": real_mean * 10_000,
                "mean_stress_net_bp": float(
                    sample["stress_net_return"].mean() * 10_000
                ),
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_family_percentile": percentile,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "positive_source_primary_bp": float(positive_side * 10_000),
                "negative_source_primary_bp": float(negative_side * 10_000),
                "positive_month_concentration": month_concentration,
                "positive_symbol_concentration": symbol_concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_reference_price_transmission"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_reference_price_transmission_family"
    )
    text = [
        "# v20.1 Reference-Price Transmission Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary[summary["scope"].eq("all")].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "The reveal follows the frozen v20.1 preregistration. Costs are fully "
        "charged as 20/40 bp round-trip book costs; the gross mean is also the "
        "estimated break-even round-trip cost.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v201_reveal(
    feature_events_path: Path = FEATURE_EVENTS_PATH,
    membership_path: Path = MEMBERSHIP_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V201Config = V201Config(),
) -> dict[str, Path]:
    feature_events = pd.read_parquet(feature_events_path)
    feature_events["feature_time"] = pd.to_datetime(
        feature_events["feature_time"], utc=True
    )
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    membership = load_v195_membership(membership_path)
    returns = close.pct_change(fill_method=None)
    risk = build_v187_monthly_risk(
        returns,
        feature_events["feature_time"].min(),
        feature_events["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    event_frames = [
        build_v201_events(feature_events, risk, close, candidate, cfg)
        for candidate in CANDIDATES
    ]
    events = pd.concat(event_frames, ignore_index=True)
    summary = summarize_v201(events)
    delayed = pd.concat(
        [
            build_v201_events(
                feature_events, risk, close, candidate, cfg, entry_delay_bars=1
            )
            for candidate in CANDIDATES
        ],
        ignore_index=True,
    )
    delayed_summary = summarize_v201(delayed)
    horizon_frames = []
    for horizon in (2, 4):
        local_events = pd.concat(
            [
                build_v201_events(
                    feature_events,
                    risk,
                    close,
                    candidate,
                    cfg,
                    holding_bars=horizon,
                )
                for candidate in CANDIDATES
            ],
            ignore_index=True,
        )
        local_summary = summarize_v201(local_events)
        local_summary["holding_bars"] = horizon
        horizon_frames.append(local_summary)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v201_controls(
        events, risk, close, membership, cfg
    )
    gates, outcome = audit_v201(
        events,
        summary,
        delayed_summary,
        horizon_summary,
        random_controls,
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
        "random_controls": root / "random_receiver_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    risk.to_parquet(outputs["risk"], index=False)
    serial_events = events.copy()
    serial_events["weights"] = serial_events["weights"].map(_serialize_mapping)
    serial_events["symbol_contributions"] = serial_events[
        "symbol_contributions"
    ].map(_serialize_mapping)
    serial_events.to_parquet(outputs["events"], index=False)
    serial_delayed = delayed.copy()
    serial_delayed["weights"] = serial_delayed["weights"].map(_serialize_mapping)
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
    "CANDIDATES",
    "CANDIDATE_RULES",
    "COMMUNITY_REFERENCE",
    "COMMUNITY_TRADE",
    "GLOBAL_REFERENCE",
    "GLOBAL_TRADE",
    "V201Config",
    "audit_v201",
    "beta_neutral_weights",
    "build_v201_events",
    "random_v201_controls",
    "select_v201_feature_events",
    "summarize_v201",
    "write_v201_reveal",
]
