"""Feature-only audit for causal implied variance at v22.4 vacuum events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    load_v155_hourly_prices,
)


EVENT_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
SURFACE_PATH = Path(
    "data/external/deribit_monthly_option_trades/daily_trade_surface.parquet"
)
BTC_HISTORY_PATH = Path("data/raw/bybit/klines/BTCUSDT.parquet")
BTC_EXTENSION_PATH = Path(
    "data/external/recent_perp_carry/bybit_klines_15m/BTCUSDT.parquet"
)
REPORT_ROOT = Path("reports/v23_0_book_vacuum_implied_variance_feature_audit")
FINDINGS_PATH = Path(
    "docs/v230_book_vacuum_implied_variance_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVB2_BOOK_VACUUM_CAUSAL_IMPLIED_VARIANCE"


@dataclass(frozen=True)
class V230Config:
    event_path: Path = EVENT_PATH
    surface_path: Path = SURFACE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_dte: float = 7.0
    maximum_dte: float = 45.0
    target_dte: float = 21.0
    maximum_surface_age_hours: float = 72.0
    prior_variance_hours: int = 24
    minimum_events: int = 120
    minimum_period_events: int = 30
    minimum_active_months: int = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v230_inputs(
    cfg: V230Config = V230Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(cfg.event_path)
    surface = pd.read_parquet(cfg.surface_path)
    prices = load_v155_hourly_prices()
    for frame, columns in (
        (events, ("feature_time", "entry_time")),
        (surface, ("surface_date", "feature_time", "expiration_time")),
        (prices, ("feature_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return events, surface, prices


def select_v230_causal_surface(
    surface: pd.DataFrame,
    cfg: V230Config = V230Config(),
) -> pd.DataFrame:
    eligible = surface[
        surface["quality_pass"].fillna(False)
        & surface["dte"].between(cfg.minimum_dte, cfg.maximum_dte)
        & surface["atm_iv"].gt(0)
    ].copy()
    eligible["target_dte_gap"] = (eligible["dte"] - cfg.target_dte).abs()
    selected = (
        eligible.sort_values(
            ["feature_time", "target_dte_gap", "expiration_time"]
        )
        .drop_duplicates("feature_time", keep="first")
        .rename(
            columns={
                "feature_time": "surface_feature_time",
                "surface_date": "surface_source_date",
                "expiration_time": "surface_expiration_time",
                "dte": "surface_dte",
                "atm_iv": "causal_atm_iv",
            }
        )
    )
    keep = [
        "surface_feature_time",
        "surface_source_date",
        "surface_expiration_time",
        "surface_dte",
        "causal_atm_iv",
        "target_dte_gap",
        "contract_count",
        "strike_count",
        "call_contracts",
        "put_contracts",
        "active_hours",
        "total_volume",
    ]
    return selected[keep].sort_values("surface_feature_time").reset_index(drop=True)


def build_v230_btc_features(
    prices: pd.DataFrame,
    cfg: V230Config = V230Config(),
) -> pd.DataFrame:
    btc = (
        prices[prices["symbol"].eq(BTC)][["feature_time", "close"]]
        .drop_duplicates("feature_time", keep="last")
        .sort_values("feature_time")
        .rename(columns={"feature_time": "entry_time", "close": "entry_spot"})
        .reset_index(drop=True)
    )
    btc["entry_spot"] = pd.to_numeric(btc["entry_spot"], errors="coerce")
    log_move = np.log(btc["entry_spot"] / btc["entry_spot"].shift(1))
    btc["prior_24h_sum_squared_log_move"] = log_move.rolling(
        cfg.prior_variance_hours,
        min_periods=cfg.prior_variance_hours,
    ).apply(lambda values: float(np.square(values).sum()), raw=True)
    return btc


def build_v230_event_features(
    events: pd.DataFrame,
    surface: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V230Config = V230Config(),
) -> pd.DataFrame:
    selected_surface = select_v230_causal_surface(surface, cfg)
    btc = build_v230_btc_features(prices, cfg)
    output = events.sort_values("entry_time").merge(
        btc,
        on="entry_time",
        how="left",
        validate="one_to_one",
    )
    output = pd.merge_asof(
        output.sort_values("entry_time"),
        selected_surface.sort_values("surface_feature_time"),
        left_on="entry_time",
        right_on="surface_feature_time",
        direction="backward",
        allow_exact_matches=True,
    )
    output["surface_age_hours"] = (
        output["entry_time"] - output["surface_feature_time"]
    ).dt.total_seconds() / 3600.0
    ready = (
        output["surface_age_hours"].between(
            0.0, cfg.maximum_surface_age_hours
        )
        & output["causal_atm_iv"].gt(0)
        & output["entry_spot"].gt(0)
        & output["prior_24h_sum_squared_log_move"].ge(0)
    )
    output = output[ready].copy()
    output["candidate"] = CANDIDATE
    columns = [
        "candidate",
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "signal_direction",
        "bucket_pressure",
        "prior_abs_pressure_threshold",
        "directional_symbol_count",
        "withdrawing_symbol_count",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "surface_feature_time",
        "surface_source_date",
        "surface_expiration_time",
        "surface_age_hours",
        "surface_dte",
        "causal_atm_iv",
        "target_dte_gap",
        "contract_count",
        "strike_count",
        "call_contracts",
        "put_contracts",
        "active_hours",
        "total_volume",
    ]
    return output[columns].sort_values("entry_time").reset_index(drop=True)


def summarize_v230(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features[features["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "median_surface_age_hours": float(local["surface_age_hours"].median()),
                "median_surface_dte": float(local["surface_dte"].median()),
                "median_causal_atm_iv": float(local["causal_atm_iv"].median()),
                "median_prior_24h_squared_move": float(
                    local["prior_24h_sum_squared_log_move"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v230_features(
    raw_events: pd.DataFrame,
    raw_surface: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V230Config = V230Config(),
) -> pd.DataFrame:
    eligible = raw_surface[
        raw_surface["quality_pass"].fillna(False)
        & raw_surface["dte"].between(cfg.minimum_dte, cfg.maximum_dte)
        & raw_surface["atm_iv"].gt(0)
    ].copy()
    eligible["minimum_gap"] = (eligible["dte"] - cfg.target_dte).abs()
    minimum_gap = eligible.groupby("feature_time")["minimum_gap"].min()
    observed_gap = features["surface_feature_time"].map(minimum_gap)
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    periods = summary[summary["scope"].ne("all")]
    columns = " ".join(features.columns).lower()
    event_keys = set(pd.to_datetime(raw_events["entry_time"], utc=True))
    checks = {
        "event_keys_unique": features["entry_time"].is_unique,
        "all_features_are_frozen_v224_events": set(features["entry_time"]).issubset(
            event_keys
        ),
        "surface_is_known_by_entry": bool(
            features["surface_feature_time"].le(features["entry_time"]).all()
        ),
        "surface_freshness_at_most_72h": bool(
            features["surface_age_hours"].between(
                0.0, cfg.maximum_surface_age_hours
            ).all()
        ),
        "surface_dte_in_frozen_7_45_window": bool(
            features["surface_dte"].between(
                cfg.minimum_dte, cfg.maximum_dte
            ).all()
        ),
        "surface_expiry_is_after_entry": bool(
            features["surface_expiration_time"].gt(features["entry_time"]).all()
        ),
        "closest_available_surface_to_21d": bool(
            np.allclose(features["target_dte_gap"], observed_gap, atol=1e-12)
        ),
        "causal_iv_positive_and_finite": bool(
            np.isfinite(features["causal_atm_iv"]).all()
            and features["causal_atm_iv"].gt(0).all()
        ),
        "entry_spot_positive_and_finite": bool(
            np.isfinite(features["entry_spot"]).all()
            and features["entry_spot"].gt(0).all()
        ),
        "prior_24h_variance_is_causal_and_finite": bool(
            np.isfinite(features["prior_24h_sum_squared_log_move"]).all()
            and features["prior_24h_sum_squared_log_move"].ge(0).all()
        ),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_period_events": bool(
            periods["events"].ge(cfg.minimum_period_events).all()
        ),
        "minimum_active_months": int(all_row["active_months"])
        >= cfg.minimum_active_months,
        "no_future_outcome_columns": not any(
            token in columns for token in ("future", "exit", "pnl", "gross", "net")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v230_book_vacuum_implied_variance_feature_audit(
    cfg: V230Config = V230Config(),
) -> dict[str, Path]:
    events, surface, prices = load_v230_inputs(cfg)
    features = build_v230_event_features(events, surface, prices, cfg)
    summary = summarize_v230(features)
    checks = audit_v230_features(events, surface, features, summary, cfg)
    hash_paths = [
        cfg.event_path,
        cfg.surface_path,
        BTC_HISTORY_PATH,
        BTC_EXTENSION_PATH,
    ]
    hashes = pd.DataFrame(
        [{"input": str(path), "sha256": _sha256(path)} for path in hash_paths]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "causal_implied_variance_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    features.to_parquet(paths["features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    verdict = (
        "feature_viable_freeze_causal_implied_variance_test"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.0 Book-Vacuum Causal Implied-Variance Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".6f"),
                "",
                "Each frozen v22.4 event uses only the latest completed Deribit",
                "daily trade surface known at entry, capped at 72 hours of age.",
                "Within the causal surface timestamp, the quality-passing 7--45 DTE",
                "row closest to 21 DTE supplies ATM IV. Entry BTC and the trailing",
                "24-hour sum of squared log moves are also known at the event time.",
                "",
                "No post-entry spot, option value, return, PnL, or event outcome was",
                "loaded. Historical option trade bars remain signal-only because no",
                "historical bid/ask archive is available.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V230Config",
    "audit_v230_features",
    "build_v230_btc_features",
    "build_v230_event_features",
    "select_v230_causal_surface",
    "summarize_v230",
    "write_v230_book_vacuum_implied_variance_feature_audit",
]
