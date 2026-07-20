"""Feature-only audit for a causal BTC OCO breakout after v22.4 events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


EVENT_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
BTC_HISTORY_PATH = Path("data/raw/bybit/klines/BTCUSDT.parquet")
BTC_EXTENSION_PATH = Path(
    "data/external/recent_perp_carry/bybit_klines_15m/BTCUSDT.parquet"
)
REPORT_ROOT = Path("reports/v23_3_book_vacuum_oco_breakout_feature_audit")
FINDINGS_PATH = Path(
    "docs/v233_book_vacuum_oco_breakout_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT"


@dataclass(frozen=True)
class V233Config:
    event_path: Path = EVENT_PATH
    btc_history_path: Path = BTC_HISTORY_PATH
    btc_extension_path: Path = BTC_EXTENSION_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prior_hours: int = 24
    barrier_sigma_multiple: float = 1.0
    path_hours: int = 4
    bar_minutes: int = 15
    minimum_events: int = 150
    minimum_period_events: int = 45
    minimum_active_months: int = 11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v233_btc_15m(cfg: V233Config = V233Config()) -> pd.DataFrame:
    columns = [
        "symbol",
        "exchange",
        "bar_open_time",
        "bar_close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in (cfg.btc_history_path, cfg.btc_extension_path)
    ]
    bars = pd.concat(frames, ignore_index=True)
    for column in ("bar_open_time", "bar_close_time"):
        bars[column] = pd.to_datetime(bars[column], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return (
        bars[
            bars["symbol"].eq("BTCUSDT")
            & bars["exchange"].eq("bybit")
        ]
        .dropna(
            subset=[
                "bar_open_time",
                "bar_close_time",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .drop_duplicates("bar_open_time", keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )


def build_v233_hourly_context(
    bars: pd.DataFrame,
    cfg: V233Config = V233Config(),
) -> pd.DataFrame:
    hourly = bars[
        bars["bar_close_time"].dt.minute.eq(0)
        & bars["bar_close_time"].dt.second.eq(0)
    ][["bar_close_time", "close"]].rename(
        columns={"bar_close_time": "entry_time", "close": "entry_spot"}
    )
    hourly = (
        hourly.drop_duplicates("entry_time", keep="last")
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    log_move = np.log(hourly["entry_spot"] / hourly["entry_spot"].shift(1))
    hourly["prior_24h_sum_squared_log_move"] = log_move.rolling(
        cfg.prior_hours, min_periods=cfg.prior_hours
    ).apply(lambda values: float(np.square(values).sum()), raw=True)
    hourly["causal_hourly_sigma"] = np.sqrt(
        hourly["prior_24h_sum_squared_log_move"] / cfg.prior_hours
    )
    return hourly


def build_v233_event_features(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V233Config = V233Config(),
) -> pd.DataFrame:
    hourly = build_v233_hourly_context(bars, cfg)
    output = events.merge(
        hourly,
        on="entry_time",
        how="left",
        validate="one_to_one",
    )
    path_bars = cfg.path_hours * 60 // cfg.bar_minutes
    available_times = set(bars["bar_open_time"])
    output["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset)
            in available_times
            for offset in range(path_bars)
        )
        for entry in output["entry_time"]
    ]
    ready = (
        output["entry_spot"].gt(0)
        & output["causal_hourly_sigma"].gt(0)
        & output["path_timestamp_count"].eq(path_bars)
    )
    output = output[ready].copy()
    width = cfg.barrier_sigma_multiple * output["causal_hourly_sigma"]
    output["upper_stop_price"] = output["entry_spot"] * np.exp(width)
    output["lower_stop_price"] = output["entry_spot"] * np.exp(-width)
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
        "covered_symbols",
        "directional_symbol_count",
        "directional_breadth",
        "withdrawing_symbol_count",
        "withdrawal_breadth",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "causal_hourly_sigma",
        "upper_stop_price",
        "lower_stop_price",
        "path_timestamp_count",
    ]
    return output[columns].sort_values("entry_time").reset_index(drop=True)


def summarize_v233(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features[features["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "long_pressure_events": int(local["signal_direction"].eq(1).sum()),
                "short_pressure_events": int(local["signal_direction"].eq(-1).sum()),
                "median_hourly_sigma_bp": float(
                    local["causal_hourly_sigma"].median() * 10_000
                ),
                "median_barrier_distance_bp": float(
                    np.log(local["upper_stop_price"] / local["entry_spot"]).median()
                    * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v233_features(
    raw_events: pd.DataFrame,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V233Config = V233Config(),
) -> pd.DataFrame:
    expected_bars = cfg.path_hours * 60 // cfg.bar_minutes
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    periods = summary[summary["scope"].ne("all")]
    columns = " ".join(features.columns).lower()
    checks = {
        "btc_15m_keys_unique": bars["bar_open_time"].is_unique,
        "btc_15m_bars_are_exactly_15_minutes": bool(
            bars["bar_close_time"]
            .sub(bars["bar_open_time"])
            .eq(pd.Timedelta(minutes=cfg.bar_minutes))
            .all()
        ),
        "ohlc_geometry_valid": bool(
            bars["high"].ge(bars[["open", "close"]].max(axis=1)).all()
            and bars["low"].le(bars[["open", "close"]].min(axis=1)).all()
            and bars["high"].ge(bars["low"]).all()
        ),
        "event_keys_match_frozen_v224": set(features["entry_time"])
        == set(raw_events["entry_time"]),
        "entry_equals_completed_feature_hour": bool(
            features["entry_time"].eq(features["feature_time"]).all()
        ),
        "trailing_24h_variance_positive_and_finite": bool(
            np.isfinite(features["prior_24h_sum_squared_log_move"]).all()
            and features["prior_24h_sum_squared_log_move"].gt(0).all()
        ),
        "causal_sigma_identity_exact": bool(
            np.allclose(
                features["causal_hourly_sigma"],
                np.sqrt(features["prior_24h_sum_squared_log_move"] / cfg.prior_hours),
                atol=1e-12,
            )
        ),
        "barrier_width_is_frozen_one_sigma": bool(
            np.allclose(
                np.log(features["upper_stop_price"] / features["entry_spot"]),
                cfg.barrier_sigma_multiple * features["causal_hourly_sigma"],
                atol=1e-12,
            )
            and np.allclose(
                np.log(features["entry_spot"] / features["lower_stop_price"]),
                cfg.barrier_sigma_multiple * features["causal_hourly_sigma"],
                atol=1e-12,
            )
        ),
        "complete_16_bar_timestamp_paths": bool(
            features["path_timestamp_count"].eq(expected_bars).all()
        ),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_period_events": bool(
            periods["events"].ge(cfg.minimum_period_events).all()
        ),
        "minimum_active_months": int(all_row["active_months"])
        >= cfg.minimum_active_months,
        "no_post_entry_outcome_columns": not any(
            token in columns
            for token in (
                "trigger",
                "fill",
                "exit",
                "future",
                "return",
                "pnl",
                "gross",
                "net",
            )
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v233_book_vacuum_oco_breakout_feature_audit(
    cfg: V233Config = V233Config(),
) -> dict[str, Path]:
    events = pd.read_parquet(cfg.event_path)
    for column in ("feature_time", "entry_time"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    bars = load_v233_btc_15m(cfg)
    features = build_v233_event_features(events, bars, cfg)
    summary = summarize_v233(features)
    checks = audit_v233_features(events, bars, features, summary, cfg)
    hashes = pd.DataFrame(
        [
            {"input": str(path), "sha256": _sha256(path)}
            for path in (
                cfg.event_path,
                cfg.btc_history_path,
                cfg.btc_extension_path,
            )
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "oco_breakout_features.parquet",
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
        "feature_viable_freeze_one_sigma_oco_breakout"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.3 Book-Vacuum OCO Breakout Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "At each frozen v22.4 event, the completed-hour BTC close and the",
                "trailing 24 completed hourly log moves define a causal one-hour",
                "sigma. Symmetric stops are frozen at plus/minus one sigma. All 16",
                "subsequent 15-minute timestamps must exist, but their highs, lows,",
                "trigger states, direction, fills, and returns were not loaded.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V233Config",
    "audit_v233_features",
    "build_v233_event_features",
    "build_v233_hourly_context",
    "load_v233_btc_15m",
    "summarize_v233",
    "write_v233_book_vacuum_oco_breakout_feature_audit",
]
