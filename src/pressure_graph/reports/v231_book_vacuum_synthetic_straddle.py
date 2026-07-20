"""Preregistered movement-sufficiency test for v22.4 book-vacuum events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.deribit_option_trade_history import inverse_option_price
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    load_v155_hourly_prices,
)
from pressure_graph.reports.v230_book_vacuum_implied_variance_feature_audit import (
    EVENT_PATH as RAW_EVENT_PATH,
    SURFACE_PATH,
    V230Config,
    build_v230_btc_features,
    select_v230_causal_surface,
)


FEATURE_PATH = Path(
    "reports/v23_0_book_vacuum_implied_variance_feature_audit/"
    "causal_implied_variance_features.parquet"
)
REPORT_ROOT = Path("reports/v23_1_book_vacuum_synthetic_straddle")
FINDINGS_PATH = Path(
    "docs/v231_book_vacuum_synthetic_straddle_findings_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v231_book_vacuum_synthetic_straddle_prereg_2026_07_17.md"
)
CANDIDATE = "DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE"
MATCHED_CONTROL = "DVB2_MATCHED_NON_EVENT_SYNTHETIC_ATM_STRADDLE"
FEATURE_SHA256 = "88A24C37339AEB9F26A272E14794375789026D2426A2730949EFA2B45B178C0D"
YEAR_HOURS = 365.25 * 24.0


@dataclass(frozen=True)
class V231Config:
    feature_path: Path = FEATURE_PATH
    raw_event_path: Path = RAW_EVENT_PATH
    surface_path: Path = SURFACE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    horizons: tuple[int, ...] = (1, 4, 8)
    primary_horizon: int = 4
    primary_premium_hurdle: float = 0.01
    stress_premium_hurdle: float = 0.02
    maximum_surface_age_hours: float = 72.0
    event_exclusion_hours: int = 8
    nearest_controls: int = 10
    minimum_controls: int = 5
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v231_inputs(
    cfg: V231Config = V231Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(cfg.feature_path)
    raw_events = pd.read_parquet(cfg.raw_event_path)
    surface = pd.read_parquet(cfg.surface_path)
    prices = load_v155_hourly_prices()
    for frame, columns in (
        (features, ("feature_time", "entry_time", "surface_expiration_time")),
        (raw_events, ("feature_time", "entry_time")),
        (surface, ("surface_date", "feature_time", "expiration_time")),
        (prices, ("feature_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return features, raw_events, surface, prices


def _btc_close(prices: pd.DataFrame) -> pd.Series:
    btc = prices[prices["symbol"].eq("BTCUSDT")].copy()
    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    return (
        btc.dropna(subset=["feature_time", "close"])
        .drop_duplicates("feature_time", keep="last")
        .set_index("feature_time")["close"]
        .sort_index()
    )


def build_v231_control_universe(
    event_features: pd.DataFrame,
    raw_events: pd.DataFrame,
    surface: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V231Config = V231Config(),
) -> pd.DataFrame:
    v230_cfg = V230Config(maximum_surface_age_hours=cfg.maximum_surface_age_hours)
    btc = build_v230_btc_features(prices, v230_cfg)
    causal_surface = select_v230_causal_surface(surface, v230_cfg)
    first_month = event_features["entry_time"].min().floor("D").replace(day=1)
    last_month = event_features["entry_time"].max().floor("D").replace(day=1)
    last_month = last_month + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    universe = btc[
        btc["entry_time"].between(first_month, last_month)
    ].sort_values("entry_time")
    universe = pd.merge_asof(
        universe,
        causal_surface.sort_values("surface_feature_time"),
        left_on="entry_time",
        right_on="surface_feature_time",
        direction="backward",
        allow_exact_matches=True,
    )
    universe["surface_age_hours"] = (
        universe["entry_time"] - universe["surface_feature_time"]
    ).dt.total_seconds() / 3600.0
    universe["entry_dte"] = (
        universe["surface_expiration_time"] - universe["entry_time"]
    ).dt.total_seconds() / 86400.0
    universe["entry_month"] = universe["entry_time"].dt.strftime("%Y-%m")
    universe["utc_hour"] = universe["entry_time"].dt.hour

    btc_times = set(_btc_close(prices).index)
    maximum_horizon = max(cfg.horizons)
    universe["complete_horizon_coverage"] = [
        all(
            time + pd.Timedelta(hours=offset) in btc_times
            for offset in range(1, maximum_horizon + 1)
        )
        for time in universe["entry_time"]
    ]
    excluded = {
        pd.Timestamp(time) + pd.Timedelta(hours=offset)
        for time in raw_events["entry_time"]
        for offset in range(-cfg.event_exclusion_hours, cfg.event_exclusion_hours + 1)
    }
    universe["outside_event_exclusion"] = ~universe["entry_time"].isin(excluded)
    ready = (
        universe["surface_age_hours"].between(
            0.0, cfg.maximum_surface_age_hours
        )
        & universe["causal_atm_iv"].gt(0)
        & universe["entry_spot"].gt(0)
        & universe["prior_24h_sum_squared_log_move"].gt(0)
        & universe["entry_dte"].gt(maximum_horizon / 24.0)
        & universe["complete_horizon_coverage"]
        & universe["outside_event_exclusion"]
    )
    keep = [
        "entry_time",
        "entry_month",
        "utc_hour",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "surface_feature_time",
        "surface_expiration_time",
        "surface_age_hours",
        "entry_dte",
        "causal_atm_iv",
    ]
    return universe.loc[ready, keep].sort_values("entry_time").reset_index(drop=True)


def build_v231_matched_control_pools(
    event_features: pd.DataFrame,
    control_universe: pd.DataFrame,
    cfg: V231Config = V231Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    epsilon = 1e-12
    for event in event_features.sort_values("entry_time").itertuples(index=False):
        local = control_universe[
            control_universe["entry_month"].eq(event.entry_month)
            & control_universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        if local.empty:
            continue
        local["match_distance"] = (
            np.log(local["causal_atm_iv"] / float(event.causal_atm_iv)).abs()
            + np.log(
                (local["prior_24h_sum_squared_log_move"] + epsilon)
                / (float(event.prior_24h_sum_squared_log_move) + epsilon)
            ).abs()
        )
        local = local.sort_values(["match_distance", "entry_time"]).head(
            cfg.nearest_controls
        )
        if len(local) < cfg.minimum_controls:
            continue
        for rank, control in enumerate(local.itertuples(index=False), start=1):
            rows.append(
                {
                    "event_time": event.entry_time,
                    "event_period": event.period,
                    "event_month": event.entry_month,
                    "control_time": control.entry_time,
                    "match_rank": rank,
                    "match_distance": control.match_distance,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["event_time", "match_rank"]
    ).reset_index(drop=True)


def price_v231_synthetic_straddles(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V231Config = V231Config(),
    *,
    candidate: str = CANDIDATE,
) -> pd.DataFrame:
    close = _btc_close(prices)
    hourly_log_move = np.log(close / close.shift(1))
    rows: list[dict[str, object]] = []
    for feature in features.sort_values("entry_time").itertuples(index=False):
        entry_time = pd.Timestamp(feature.entry_time)
        entry_spot = float(feature.entry_spot)
        volatility = float(feature.causal_atm_iv)
        expiry = pd.Timestamp(feature.surface_expiration_time)
        entry_years = (expiry - entry_time).total_seconds() / (
            YEAR_HOURS * 3600.0
        )
        if entry_years <= max(cfg.horizons) / YEAR_HOURS:
            continue
        strike = entry_spot
        entry_btc = inverse_option_price(
            entry_spot, strike, entry_years, volatility, "call"
        ) + inverse_option_price(entry_spot, strike, entry_years, volatility, "put")
        entry_usd = entry_btc * entry_spot
        row = feature._asdict()
        row["candidate"] = candidate
        row["synthetic_strike"] = strike
        row["synthetic_entry_dte"] = entry_years * 365.25
        row["synthetic_entry_premium_btc"] = entry_btc
        row["synthetic_entry_premium_usd"] = entry_usd
        complete = True
        for horizon in cfg.horizons:
            exit_time = entry_time + pd.Timedelta(hours=horizon)
            path_times = [
                entry_time + pd.Timedelta(hours=offset)
                for offset in range(1, horizon + 1)
            ]
            if exit_time not in close.index or any(
                time not in hourly_log_move.index for time in path_times
            ):
                complete = False
                break
            exit_spot = float(close.loc[exit_time])
            exit_years = entry_years - horizon / YEAR_HOURS
            exit_btc = inverse_option_price(
                exit_spot, strike, exit_years, volatility, "call"
            ) + inverse_option_price(
                exit_spot, strike, exit_years, volatility, "put"
            )
            exit_usd = exit_btc * exit_spot
            gross = exit_usd / entry_usd - 1.0
            realized_variance = float(
                np.square(hourly_log_move.loc[path_times].to_numpy(dtype=float)).sum()
            )
            implied_budget = volatility * volatility * horizon / YEAR_HOURS
            row[f"exit_spot_{horizon}h"] = exit_spot
            row[f"absolute_log_move_{horizon}h"] = abs(
                float(np.log(exit_spot / entry_spot))
            )
            row[f"realized_variance_{horizon}h"] = realized_variance
            row[f"implied_variance_budget_{horizon}h"] = implied_budget
            row[f"realized_to_implied_variance_{horizon}h"] = (
                realized_variance / implied_budget
            )
            row[f"synthetic_exit_premium_btc_{horizon}h"] = exit_btc
            row[f"synthetic_exit_premium_usd_{horizon}h"] = exit_usd
            row[f"gross_premium_return_{horizon}h"] = gross
        if not complete:
            continue
        primary = cfg.primary_horizon
        row[f"primary_net_premium_return_{primary}h"] = (
            row[f"gross_premium_return_{primary}h"] - cfg.primary_premium_hurdle
        )
        row[f"stress_net_premium_return_{primary}h"] = (
            row[f"gross_premium_return_{primary}h"] - cfg.stress_premium_hurdle
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def build_v231_random_paths(
    event_outcomes: pd.DataFrame,
    control_outcomes: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V231Config = V231Config(),
) -> pd.DataFrame:
    primary = f"primary_net_premium_return_{cfg.primary_horizon}h"
    event_lookup = event_outcomes.set_index("entry_time")[primary]
    control_lookup = control_outcomes.set_index("entry_time")[primary]
    grouped = {
        event_time: local["control_time"].tolist()
        for event_time, local in pools.groupby("event_time", sort=True)
        if event_time in event_lookup.index
    }
    event_times = sorted(grouped)
    event_mean = float(event_lookup.loc[event_times].mean())
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        sampled = [
            controls[int(rng.integers(0, len(controls)))]
            for controls in grouped.values()
        ]
        control_mean = float(control_lookup.loc[sampled].mean())
        rows.append(
            {
                "iteration": iteration,
                "matched_events": len(event_times),
                "event_mean_primary_net": event_mean,
                "control_mean_primary_net": control_mean,
                "event_minus_control": event_mean - control_mean,
            }
        )
    return pd.DataFrame(rows)


def build_v231_month_bootstrap(
    outcomes: pd.DataFrame,
    cfg: V231Config = V231Config(),
) -> pd.DataFrame:
    column = f"primary_net_premium_return_{cfg.primary_horizon}h"
    by_month = {
        month: local[column].to_numpy(dtype=float)
        for month, local in outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(by_month)
    rng = np.random.default_rng(cfg.seed + 1)
    means = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled_months = rng.choice(months, size=len(months), replace=True)
        values = np.concatenate([by_month[month] for month in sampled_months])
        means.append(
            {
                "iteration": iteration,
                "mean_primary_net_premium_return": float(values.mean()),
            }
        )
    return pd.DataFrame(means)


def summarize_v231(
    outcomes: pd.DataFrame,
    cfg: V231Config = V231Config(),
) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("all", outcomes),
        ("development", outcomes[outcomes["period"].eq("development")]),
        ("validation", outcomes[outcomes["period"].eq("validation")]),
        ("holdout", outcomes[outcomes["period"].eq("holdout")]),
        ("positive_pressure", outcomes[outcomes["signal_direction"].eq(1)]),
        ("negative_pressure", outcomes[outcomes["signal_direction"].eq(-1)]),
    ]
    rows: list[dict[str, object]] = []
    for scope, local in scopes:
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "mean_gross_premium_return_1h_bp": float(
                    local["gross_premium_return_1h"].mean() * 10_000
                ),
                "mean_gross_premium_return_4h_bp": float(
                    local["gross_premium_return_4h"].mean() * 10_000
                ),
                "mean_primary_net_premium_return_4h_bp": float(
                    local["primary_net_premium_return_4h"].mean() * 10_000
                ),
                "mean_stress_net_premium_return_4h_bp": float(
                    local["stress_net_premium_return_4h"].mean() * 10_000
                ),
                "mean_gross_premium_return_8h_bp": float(
                    local["gross_premium_return_8h"].mean() * 10_000
                ),
                "mean_realized_to_implied_variance_4h": float(
                    local["realized_to_implied_variance_4h"].mean()
                ),
                "median_realized_to_implied_variance_4h": float(
                    local["realized_to_implied_variance_4h"].median()
                ),
                "mean_absolute_log_move_4h_bp": float(
                    local["absolute_log_move_4h"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def decide_v231(
    summary: pd.DataFrame,
    random_paths: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    indexed = summary.set_index("scope")
    period_values = indexed.loc[
        ["development", "validation", "holdout"],
        "mean_primary_net_premium_return_4h_bp",
    ]
    all_row = indexed.loc["all"]
    lower = float(
        bootstrap["mean_primary_net_premium_return"].quantile(0.025) * 10_000
    )
    event_mean = float(random_paths["event_mean_primary_net"].iloc[0])
    random_percentile = float(
        random_paths["control_mean_primary_net"].le(event_mean).mean() * 100.0
    )
    gates = {
        "overall_primary_net_positive": float(
            all_row["mean_primary_net_premium_return_4h_bp"]
        )
        > 0,
        "development_validation_holdout_primary_net_positive": bool(
            period_values.gt(0).all()
        ),
        "month_block_bootstrap_lower_above_zero": lower > 0,
        "matched_random_percentile_at_least_90": random_percentile >= 90.0,
        "mean_realized_to_implied_variance_above_one": float(
            all_row["mean_realized_to_implied_variance_4h"]
        )
        > 1.0,
    }
    decision = pd.DataFrame(
        {
            "gate": list(gates),
            "passed": list(gates.values()),
            "observed": [
                float(all_row["mean_primary_net_premium_return_4h_bp"]),
                float(period_values.min()),
                lower,
                random_percentile,
                float(all_row["mean_realized_to_implied_variance_4h"]),
            ],
        }
    )
    verdict = (
        "research_only_movement_sufficiency_supported"
        if bool(decision["passed"].all())
        else "movement_sufficiency_rejected"
    )
    return decision, verdict


def write_v231_book_vacuum_synthetic_straddle(
    cfg: V231Config = V231Config(),
) -> dict[str, Path]:
    if _sha256(cfg.feature_path) != FEATURE_SHA256:
        raise RuntimeError("v23.0 feature hash differs from preregistration")
    event_features, raw_events, surface, prices = load_v231_inputs(cfg)
    universe = build_v231_control_universe(
        event_features, raw_events, surface, prices, cfg
    )
    pools = build_v231_matched_control_pools(event_features, universe, cfg)
    outcomes = price_v231_synthetic_straddles(event_features, prices, cfg)
    control_outcomes = price_v231_synthetic_straddles(
        universe, prices, cfg, candidate=MATCHED_CONTROL
    )
    random_paths = build_v231_random_paths(outcomes, control_outcomes, pools, cfg)
    bootstrap = build_v231_month_bootstrap(outcomes, cfg)
    summary = summarize_v231(outcomes, cfg)
    decision, verdict = decide_v231(summary, random_paths, bootstrap)

    root = ensure_dir(cfg.report_root)
    paths = {
        "outcomes": root / "synthetic_straddle_event_outcomes.parquet",
        "control_universe": root / "causal_control_universe.parquet",
        "control_pools": root / "matched_control_pools.parquet",
        "control_outcomes": root / "synthetic_straddle_control_outcomes.parquet",
        "random_paths": root / "matched_random_paths.parquet",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "summary": root / "result_summary.csv",
        "decision": root / "decision_gates.csv",
        "config": root / "frozen_config.json",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["outcomes"], index=False)
    universe.to_parquet(paths["control_universe"], index=False)
    pools.to_parquet(paths["control_pools"], index=False)
    control_outcomes.to_parquet(paths["control_outcomes"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    summary.to_csv(paths["summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["hashes"].write_text(
        "input,sha256\n"
        + "\n".join(
            f"{path},{_sha256(path)}"
            for path in (cfg.feature_path, cfg.raw_event_path, cfg.surface_path)
        )
        + "\n",
        encoding="utf-8",
    )
    random_percentile = float(decision.loc[
        decision["gate"].eq("matched_random_percentile_at_least_90"), "observed"
    ].iloc[0])
    bootstrap_lower = float(decision.loc[
        decision["gate"].eq("month_block_bootstrap_lower_above_zero"), "observed"
    ].iloc[0])
    matched_events = pools["event_time"].nunique()
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.1 Book-Vacuum Synthetic Straddle Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Matched-event coverage: {matched_events}/{len(outcomes)}.",
                f"Matched random-time percentile: {random_percentile:.2f}.",
                f"Month-block bootstrap 2.5% lower bound: {bootstrap_lower:.4f} bp.",
                "",
                "This is a constant-IV synthetic movement-sufficiency test, not",
                "historical executable option PnL. The local Deribit archive has",
                "trade OHLCV but no synchronized historical bid/ask, and the exact",
                "preselected two-leg 4-hour trade coverage is too sparse for promotion.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V231Config",
    "build_v231_control_universe",
    "build_v231_matched_control_pools",
    "build_v231_month_bootstrap",
    "build_v231_random_paths",
    "decide_v231",
    "price_v231_synthetic_straddles",
    "summarize_v231",
    "write_v231_book_vacuum_synthetic_straddle",
]
