"""Feature-only audit for causal alt-first volatility ignition events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    load_v155_hourly_prices,
)
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)


REPORT_ROOT = Path("reports/v23_20_alt_first_volatility_ignition_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2320_alt_first_volatility_ignition_feature_audit_2026_07_17.md"
)
CANDIDATE = "AVI1_ALT_FIRST_VOLATILITY_IGNITION_BTC_BREAKOUT"


@dataclass(frozen=True)
class V2320Config:
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    normalization_hours: int = 720
    minimum_normalization_hours: int = 480
    alt_bucket_quantile: float = 0.80
    btc_quiet_quantile: float = 0.50
    symbol_shock_z: float = 1.0
    minimum_shocked_symbols: int = 8
    minimum_covered_symbols: int = 14
    cooldown_hours: int = 4
    prior_sigma_hours: int = 24
    barrier_sigma_multiple: float = 0.75
    path_hours: int = 4
    bar_minutes: int = 15
    minimum_events: int = 90
    minimum_period_events: int = 20
    minimum_active_months: int = 12


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def load_v2320_inputs(
    cfg: V2320Config = V2320Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del cfg
    prices = load_v155_hourly_prices()
    prices["feature_time"] = _utc(prices["feature_time"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    bars = load_v233_btc_15m(V233Config())
    return prices, bars


def build_v2320_states(
    prices: pd.DataFrame,
    cfg: V2320Config = V2320Config(),
) -> pd.DataFrame:
    symbols = [BTC, *FROZEN_SYMBOLS]
    matrix = (
        prices.pivot_table(
            index="feature_time",
            columns="symbol",
            values="close",
            aggfunc="last",
            observed=True,
        )
        .sort_index()[symbols]
    )
    log_returns = np.log(matrix / matrix.shift(1))
    trailing_sigma = (
        log_returns.rolling(
            cfg.normalization_hours,
            min_periods=cfg.minimum_normalization_hours,
        )
        .std(ddof=1)
        .shift(1)
    )
    standardized_abs_move = (log_returns / trailing_sigma).abs()
    alt_z = standardized_abs_move[list(FROZEN_SYMBOLS)]
    alt_bucket_shock = alt_z.median(axis=1, skipna=True)
    prior_alt_threshold = (
        alt_bucket_shock.rolling(
            cfg.normalization_hours,
            min_periods=cfg.minimum_normalization_hours,
        )
        .quantile(cfg.alt_bucket_quantile)
        .shift(1)
    )
    btc_abs_z = standardized_abs_move[BTC]
    prior_btc_quiet_threshold = (
        btc_abs_z.rolling(
            cfg.normalization_hours,
            min_periods=cfg.minimum_normalization_hours,
        )
        .quantile(cfg.btc_quiet_quantile)
        .shift(1)
    )
    states = pd.DataFrame(
        {
            "decision_time": matrix.index,
            "covered_symbols": alt_z.notna().sum(axis=1).to_numpy(int),
            "shocked_symbols": alt_z.ge(cfg.symbol_shock_z).sum(axis=1).to_numpy(int),
            "alt_bucket_shock_z": alt_bucket_shock.to_numpy(float),
            "prior_alt_bucket_shock_threshold": prior_alt_threshold.to_numpy(float),
            "btc_abs_move_z": btc_abs_z.to_numpy(float),
            "prior_btc_quiet_threshold": prior_btc_quiet_threshold.to_numpy(float),
        }
    )
    states["alt_shock_ready"] = (
        states["covered_symbols"].ge(cfg.minimum_covered_symbols)
        & states["shocked_symbols"].ge(cfg.minimum_shocked_symbols)
        & states["alt_bucket_shock_z"].ge(
            states["prior_alt_bucket_shock_threshold"]
        )
    )
    states["btc_still_quiet"] = states["btc_abs_move_z"].le(
        states["prior_btc_quiet_threshold"]
    )
    states["ignition_state"] = states["alt_shock_ready"] & states["btc_still_quiet"]
    states["state_transition"] = states["ignition_state"] & ~states[
        "ignition_state"
    ].shift(1, fill_value=False)
    return states


def select_v2320_events(
    states: pd.DataFrame,
    cfg: V2320Config = V2320Config(),
) -> pd.DataFrame:
    starts = states.loc[states["state_transition"]].copy()
    selected: list[int] = []
    last_time: pd.Timestamp | None = None
    for index, row in starts.iterrows():
        time = pd.Timestamp(row["decision_time"])
        if last_time is None or time - last_time >= pd.Timedelta(
            hours=cfg.cooldown_hours
        ):
            selected.append(index)
            last_time = time
    events = starts.loc[selected].copy().reset_index(drop=True)
    events["candidate"] = CANDIDATE
    events["feature_time"] = events["decision_time"]
    events["entry_time"] = events["decision_time"]
    events["entry_month"] = events["entry_time"].dt.strftime("%Y-%m")
    events["period"] = np.select(
        [
            events["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            events["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return events


def build_v2320_features(
    prices: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2320Config = V2320Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = build_v2320_states(prices, cfg)
    events = select_v2320_events(states, cfg)
    hourly = build_v233_hourly_context(
        bars,
        V233Config(prior_hours=cfg.prior_sigma_hours),
    )
    features = events.merge(hourly, on="entry_time", how="left", validate="one_to_one")
    expected_bars = cfg.path_hours * 60 // cfg.bar_minutes
    available = set(bars["bar_open_time"])
    features["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset) in available
            for offset in range(expected_bars)
        )
        for entry in features["entry_time"]
    ]
    features = features.loc[
        features["entry_spot"].gt(0)
        & features["causal_hourly_sigma"].gt(0)
        & features["path_timestamp_count"].eq(expected_bars)
    ].copy()
    width = cfg.barrier_sigma_multiple * features["causal_hourly_sigma"]
    features["upper_stop_price"] = features["entry_spot"] * np.exp(width)
    features["lower_stop_price"] = features["entry_spot"] * np.exp(-width)
    keep = [
        "candidate",
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "covered_symbols",
        "shocked_symbols",
        "alt_bucket_shock_z",
        "prior_alt_bucket_shock_threshold",
        "btc_abs_move_z",
        "prior_btc_quiet_threshold",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "causal_hourly_sigma",
        "upper_stop_price",
        "lower_stop_price",
        "path_timestamp_count",
    ]
    return states, features[keep].sort_values("entry_time").reset_index(drop=True)


def summarize_v2320(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features.loc[features["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "median_shocked_symbols": float(local["shocked_symbols"].median()),
                "median_alt_bucket_shock_z": float(local["alt_bucket_shock_z"].median()),
                "median_btc_abs_move_z": float(local["btc_abs_move_z"].median()),
                "median_barrier_width_bp": float(
                    (
                        np.log(local["upper_stop_price"] / local["entry_spot"])
                        * 10_000
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v2320(
    states: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2320Config = V2320Config(),
) -> pd.DataFrame:
    period_counts = features["period"].value_counts()
    event_states = states.set_index("decision_time").loc[features["entry_time"]]
    gaps = features["entry_time"].diff().dropna()
    expected_bars = cfg.path_hours * 60 // cfg.bar_minutes
    checks = {
        "minimum_total_events": len(features) >= cfg.minimum_events,
        "minimum_period_events": all(
            int(period_counts.get(period, 0)) >= cfg.minimum_period_events
            for period in ("development", "validation", "holdout")
        ),
        "minimum_active_months": features["entry_month"].nunique()
        >= cfg.minimum_active_months,
        "minimum_symbol_coverage": features["covered_symbols"].ge(
            cfg.minimum_covered_symbols
        ).all(),
        "minimum_shock_breadth": features["shocked_symbols"].ge(
            cfg.minimum_shocked_symbols
        ).all(),
        "alt_bucket_above_causal_q80": features["alt_bucket_shock_z"].ge(
            features["prior_alt_bucket_shock_threshold"]
        ).all(),
        "btc_below_causal_median": features["btc_abs_move_z"].le(
            features["prior_btc_quiet_threshold"]
        ).all(),
        "every_event_is_state_transition": event_states["state_transition"].all(),
        "four_hour_cooldown": gaps.ge(pd.Timedelta(hours=cfg.cooldown_hours)).all(),
        "feature_time_equals_entry_time": features["feature_time"].eq(
            features["entry_time"]
        ).all(),
        "complete_four_hour_paths": features["path_timestamp_count"].eq(
            expected_bars
        ).all(),
        "causal_sigma_positive": features["causal_hourly_sigma"].gt(0).all(),
        "barrier_width_exact": np.allclose(
            np.log(features["upper_stop_price"] / features["entry_spot"]),
            cfg.barrier_sigma_multiple * features["causal_hourly_sigma"],
        )
        and np.allclose(
            np.log(features["entry_spot"] / features["lower_stop_price"]),
            cfg.barrier_sigma_multiple * features["causal_hourly_sigma"],
        ),
        "summary_reconciles": int(summary.loc[summary["scope"].eq("all"), "events"].iloc[0])
        == len(features),
        "no_outcome_columns": {
            "triggered",
            "trigger_time",
            "exit_spot",
            "gross_return",
            "primary_net_return",
            "stress_net_return",
        }.isdisjoint(features.columns),
    }
    return pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )


def feature_hash_v2320(features: pd.DataFrame) -> str:
    payload = features.sort_values("entry_time").to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_v2320_alt_first_volatility_ignition_feature_audit(
    cfg: V2320Config = V2320Config(),
) -> dict[str, Path]:
    prices, bars = load_v2320_inputs(cfg)
    states, features = build_v2320_features(prices, bars, cfg)
    summary = summarize_v2320(features)
    checks = audit_v2320(states, features, summary, cfg)
    feature_hash = feature_hash_v2320(features)
    root = ensure_dir(cfg.report_root)
    paths = {
        "states": root / "hourly_ignition_states.parquet",
        "features": root / "alt_first_ignition_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    states.to_parquet(paths["states"], index=False)
    features.to_parquet(paths["features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    passed = bool(checks["passed"].all())
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_hash": feature_hash,
                "all_checks_passed": passed,
                "outcomes_loaded": False,
                "config": {
                    **asdict(cfg),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "feature_viable_freeze_alt_first_ignition" if passed else "feature_audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.20 Alt-First Volatility Ignition Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Feature hash: `{feature_hash}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The signal uses only completed hourly prices and prior rolling",
                "normalization windows. It captures broad alt volatility while BTC",
                "remains below its own causal median shock. No post-entry price or",
                "return outcome was used to select the 100 events.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2320Config",
    "audit_v2320",
    "build_v2320_features",
    "build_v2320_states",
    "feature_hash_v2320",
    "load_v2320_inputs",
    "select_v2320_events",
    "summarize_v2320",
    "write_v2320_alt_first_volatility_ignition_feature_audit",
]
