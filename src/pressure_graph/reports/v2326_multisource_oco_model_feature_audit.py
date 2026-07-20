"""Outcome-free multisource features for a temporal OCO selection model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)


BASE_FEATURE_PATH = Path(
    "reports/v23_3_book_vacuum_oco_breakout_feature_audit/"
    "oco_breakout_features.parquet"
)
METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
REPORT_ROOT = Path("reports/v23_26_multisource_oco_model_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2326_multisource_oco_model_feature_audit_2026_07_17.md"
)
CANDIDATE = "MSM1_MULTISOURCE_INTERACTION_RIDGE_OCO_SELECTOR"
MODEL_FEATURES = (
    "signal_direction",
    "pressure_excess",
    "directional_breadth",
    "withdrawal_breadth",
    "causal_hourly_sigma",
    "btc_return_1h",
    "alt_taker_log_median",
    "alt_taker_buy_breadth",
    "alt_oi_change_median",
    "alt_oi_build_breadth",
    "alt_top_position_change_median",
    "alt_top_position_build_breadth",
    "alt_top_size_bias_median",
    "btc_taker_log",
    "btc_oi_change",
    "btc_top_position_change",
    "btc_top_size_bias",
    "utc_hour_sin",
    "utc_hour_cos",
)


@dataclass(frozen=True)
class V2326Config:
    base_feature_path: Path = BASE_FEATURE_PATH
    metrics_root: Path = METRICS_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_metric_symbols: int = 15
    minimum_events: int = 159
    minimum_period_events: int = 45
    minimum_active_months: int = 11


def _load_metric_frames(cfg: V2326Config) -> dict[str, pd.DataFrame]:
    symbols = [*FROZEN_SYMBOLS, "BTCUSDT"]
    columns = [
        "create_time",
        "sum_open_interest",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    output = {}
    for symbol in symbols:
        frame = pd.read_parquet(cfg.metrics_root / f"{symbol}.parquet", columns=columns)
        frame["create_time"] = pd.to_datetime(
            frame["create_time"], utc=True, errors="raise"
        )
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        output[symbol] = (
            frame.drop_duplicates("create_time", keep="last")
            .set_index("create_time")
            .sort_index()
        )
    return output


def load_v2326_inputs(
    cfg: V2326Config = V2326Config(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    base = pd.read_parquet(cfg.base_feature_path)
    for column in ("feature_time", "entry_time"):
        base[column] = pd.to_datetime(base[column], utc=True, errors="raise")
    metrics = _load_metric_frames(cfg)
    bars = load_v233_btc_15m(V233Config())
    return base.sort_values("entry_time").reset_index(drop=True), metrics, bars


def build_v2326_features(
    base: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    bars: pd.DataFrame,
    cfg: V2326Config = V2326Config(),
) -> pd.DataFrame:
    hourly = build_v233_hourly_context(bars, V233Config()).sort_values("entry_time")
    hourly["btc_return_1h"] = np.log(
        hourly["entry_spot"] / hourly["entry_spot"].shift(1)
    )
    btc_return = hourly.set_index("entry_time")["btc_return_1h"]
    rows = []
    for event in base.itertuples(index=False):
        time = pd.Timestamp(event.entry_time)
        prior = time - pd.Timedelta(minutes=5)
        alt_rows = []
        for symbol in FROZEN_SYMBOLS:
            frame = metrics[symbol]
            if time not in frame.index or prior not in frame.index:
                continue
            current = frame.loc[time]
            previous = frame.loc[prior]
            required = [
                current["sum_open_interest"],
                previous["sum_open_interest"],
                current["count_toptrader_long_short_ratio"],
                current["sum_toptrader_long_short_ratio"],
                previous["sum_toptrader_long_short_ratio"],
                current["sum_taker_long_short_vol_ratio"],
            ]
            if not all(np.isfinite(value) and value > 0 for value in required):
                continue
            alt_rows.append(
                {
                    "taker": float(np.log(current["sum_taker_long_short_vol_ratio"])),
                    "oi_change": float(
                        np.log(current["sum_open_interest"] / previous["sum_open_interest"])
                    ),
                    "top_change": float(
                        np.log(
                            current["sum_toptrader_long_short_ratio"]
                            / previous["sum_toptrader_long_short_ratio"]
                        )
                    ),
                    "top_size_bias": float(
                        np.log(
                            current["sum_toptrader_long_short_ratio"]
                            / current["count_toptrader_long_short_ratio"]
                        )
                    ),
                }
            )
        btc_frame = metrics["BTCUSDT"]
        if time not in btc_frame.index or prior not in btc_frame.index:
            continue
        btc_now = btc_frame.loc[time]
        btc_prior = btc_frame.loc[prior]
        btc_required = [
            btc_now["sum_open_interest"],
            btc_prior["sum_open_interest"],
            btc_now["count_toptrader_long_short_ratio"],
            btc_now["sum_toptrader_long_short_ratio"],
            btc_prior["sum_toptrader_long_short_ratio"],
            btc_now["sum_taker_long_short_vol_ratio"],
        ]
        if len(alt_rows) < cfg.minimum_metric_symbols or not all(
            np.isfinite(value) and value > 0 for value in btc_required
        ):
            continue
        alt = pd.DataFrame(alt_rows)
        row = event._asdict()
        row.update(
            {
                "candidate": CANDIDATE,
                "metric_feature_time": time,
                "metric_symbol_count": len(alt),
                "pressure_excess": abs(float(event.bucket_pressure))
                / float(event.prior_abs_pressure_threshold)
                - 1.0,
                "btc_return_1h": float(btc_return.at[time]),
                "alt_taker_log_median": float(alt["taker"].median()),
                "alt_taker_buy_breadth": float(alt["taker"].gt(0).mean()),
                "alt_oi_change_median": float(alt["oi_change"].median()),
                "alt_oi_build_breadth": float(alt["oi_change"].gt(0).mean()),
                "alt_top_position_change_median": float(alt["top_change"].median()),
                "alt_top_position_build_breadth": float(
                    alt["top_change"].gt(0).mean()
                ),
                "alt_top_size_bias_median": float(alt["top_size_bias"].median()),
                "btc_taker_log": float(
                    np.log(btc_now["sum_taker_long_short_vol_ratio"])
                ),
                "btc_oi_change": float(
                    np.log(
                        btc_now["sum_open_interest"]
                        / btc_prior["sum_open_interest"]
                    )
                ),
                "btc_top_position_change": float(
                    np.log(
                        btc_now["sum_toptrader_long_short_ratio"]
                        / btc_prior["sum_toptrader_long_short_ratio"]
                    )
                ),
                "btc_top_size_bias": float(
                    np.log(
                        btc_now["sum_toptrader_long_short_ratio"]
                        / btc_now["count_toptrader_long_short_ratio"]
                    )
                ),
                "utc_hour_sin": float(np.sin(2 * np.pi * time.hour / 24)),
                "utc_hour_cos": float(np.cos(2 * np.pi * time.hour / 24)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def summarize_v2326(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features.loc[features["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "minimum_alt_metric_symbols": int(
                    local["metric_symbol_count"].min()
                ),
                "positive_pressure_fraction": float(
                    local["signal_direction"].eq(1).mean()
                ),
                "median_pressure_excess": float(local["pressure_excess"].median()),
                "median_alt_taker_buy_breadth": float(
                    local["alt_taker_buy_breadth"].median()
                ),
                "median_alt_oi_build_breadth": float(
                    local["alt_oi_build_breadth"].median()
                ),
                "median_causal_sigma_bp": float(
                    local["causal_hourly_sigma"].median() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v2326(
    base: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2326Config = V2326Config(),
) -> pd.DataFrame:
    period_counts = features["period"].value_counts()
    values = features[list(MODEL_FEATURES)].to_numpy(float)
    checks = {
        "all_159_base_events_retained": len(base) == 159 and len(features) == 159,
        "minimum_period_events": all(
            int(period_counts.get(period, 0)) >= cfg.minimum_period_events
            for period in ("development", "validation", "holdout")
        ),
        "minimum_active_months": features["entry_month"].nunique()
        >= cfg.minimum_active_months,
        "at_least_15_of_16_exact_alt_metric_coverage": features[
            "metric_symbol_count"
        ].between(cfg.minimum_metric_symbols, 16).all(),
        "at_most_one_event_below_16_alt_symbols": features[
            "metric_symbol_count"
        ].lt(16).sum()
        <= 1,
        "metric_time_equals_entry": features["metric_feature_time"].eq(
            features["entry_time"]
        ).all(),
        "all_model_features_finite": np.isfinite(values).all(),
        "pressure_excess_nonnegative": features["pressure_excess"].ge(0).all(),
        "breadths_between_zero_and_one": all(
            features[column].between(0, 1).all()
            for column in (
                "directional_breadth",
                "withdrawal_breadth",
                "alt_taker_buy_breadth",
                "alt_oi_build_breadth",
                "alt_top_position_build_breadth",
            )
        ),
        "causal_sigma_positive": features["causal_hourly_sigma"].gt(0).all(),
        "cyclic_hour_features_exact": np.allclose(
            np.square(features["utc_hour_sin"])
            + np.square(features["utc_hour_cos"]),
            1.0,
        ),
        "model_feature_count_19": len(MODEL_FEATURES) == 19,
        "entry_times_unique": features["entry_time"].is_unique,
        "summary_reconciles": int(summary.loc[summary["scope"].eq("all"), "events"].iloc[0])
        == len(features),
        "outcome_columns_absent": {
            "gross_return",
            "primary_net_return",
            "stress_net_return",
            "profitable",
        }.isdisjoint(features.columns),
    }
    return pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )


def feature_hash_v2326(features: pd.DataFrame) -> str:
    payload = features.sort_values("entry_time").to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_v2326_multisource_oco_model_feature_audit(
    cfg: V2326Config = V2326Config(),
) -> dict[str, Path]:
    base, metrics, bars = load_v2326_inputs(cfg)
    features = build_v2326_features(base, metrics, bars, cfg)
    summary = summarize_v2326(features)
    checks = audit_v2326(base, features, summary, cfg)
    feature_hash = feature_hash_v2326(features)
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "multisource_model_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
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
                "model_features": list(MODEL_FEATURES),
                "outcomes_loaded": False,
                "config": {
                    **asdict(cfg),
                    "base_feature_path": str(cfg.base_feature_path),
                    "metrics_root": str(cfg.metrics_root),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "feature_viable_freeze_multisource_model" if passed else "feature_audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.26 Multisource OCO Model Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Feature hash: `{feature_hash}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The 19 fixed features combine book state, causal BTC volatility and",
                "return, same-timestamp 15-of-16-or-better taker/OI/top-position",
                "state, BTC derivatives state, and UTC-hour cyclic terms. One",
                "validation event uses 15 symbols because XLM lacks that exact 5-minute",
                "record; no future or stale fill is used. No OCO return was loaded.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "MODEL_FEATURES",
    "V2326Config",
    "audit_v2326",
    "build_v2326_features",
    "feature_hash_v2326",
    "load_v2326_inputs",
    "summarize_v2326",
    "write_v2326_multisource_oco_model_feature_audit",
]
