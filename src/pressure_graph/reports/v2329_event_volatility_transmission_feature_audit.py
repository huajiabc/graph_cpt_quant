"""Outcome-free event features for continuous alt-to-BTC volatility transmission."""

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


BASE_FEATURE_PATH = Path(
    "reports/v23_26_multisource_oco_model_feature_audit/"
    "multisource_model_features.parquet"
)
REPORT_ROOT = Path("reports/v23_29_event_volatility_transmission_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2329_event_volatility_transmission_feature_audit_2026_07_17.md"
)
CANDIDATE = "EVT1_EVENT_ALT_TO_BTC_VOLATILITY_TRANSMISSION"
VOLATILITY_FEATURES = (
    "alt_abs_z_median",
    "alt_abs_z_dispersion",
    "alt_shock_breadth_z1",
    "alt_shock_breadth_z2",
    "alt_positive_return_breadth",
    "alt_directional_coherence",
    "btc_abs_z",
    "alt_btc_abs_z_gap",
    "alt_rv_acceleration_median",
    "alt_rv_acceleration_breadth",
    "alt_residual_abs_z_median",
    "alt_residual_shock_breadth",
    "directed_edge_fraction",
    "directed_edge_weight_mean",
    "leader_shock_score",
    "leader_shock_breadth",
    "btc_receiver_gap",
)


@dataclass(frozen=True)
class V2329Config:
    base_feature_path: Path = BASE_FEATURE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    lookback_hours: int = 720
    minimum_history_hours: int = 480
    shrinkage_n: int = 500
    top_leaders: int = 4
    minimum_events: int = 159
    minimum_period_events: int = 45
    minimum_active_months: int = 11


def load_v2329_inputs(
    cfg: V2329Config = V2329Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(cfg.base_feature_path)
    for column in ("feature_time", "entry_time", "metric_feature_time"):
        base[column] = pd.to_datetime(base[column], utc=True, errors="raise")
    prices = load_v155_hourly_prices()
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="raise"
    )
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.loc[prices["symbol"].isin((BTC, *FROZEN_SYMBOLS))]
    return base.sort_values("entry_time").reset_index(drop=True), prices


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 100:
        return 0.0
    x = left[finite]
    y = right[finite]
    if np.isclose(x.std(ddof=1), 0.0) or np.isclose(y.std(ddof=1), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _event_features(
    entry: pd.Timestamp,
    returns: pd.DataFrame,
    cfg: V2329Config,
) -> dict[str, float] | None:
    symbols = [BTC, *FROZEN_SYMBOLS]
    if entry not in returns.index:
        return None
    history = returns.loc[
        (returns.index >= entry - pd.Timedelta(hours=cfg.lookback_hours))
        & (returns.index < entry),
        symbols,
    ].dropna(how="any")
    current = returns.loc[entry, symbols]
    if len(history) < cfg.minimum_history_hours or not np.isfinite(current).all():
        return None
    scale = history.std(ddof=1).replace(0.0, np.nan)
    current_z = current / scale
    if not np.isfinite(current_z).all():
        return None
    alt_z = current_z[list(FROZEN_SYMBOLS)]
    alt_abs = alt_z.abs()
    absolute_median = float(alt_abs.median())
    signed_median = float(alt_z.median())
    directional_coherence = (
        abs(signed_median) / absolute_median if absolute_median > 0 else 0.0
    )

    window_24 = returns.loc[
        (returns.index > entry - pd.Timedelta(hours=24))
        & (returns.index <= entry),
        symbols,
    ]
    if len(window_24) != 24 or window_24.isna().any().any():
        return None
    rv_4 = window_24.tail(4).pow(2).sum().pow(0.5)
    rv_24_scaled = window_24.pow(2).sum().pow(0.5) * np.sqrt(4.0 / 24.0)
    acceleration = (rv_4 / rv_24_scaled.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    alt_acceleration = acceleration[list(FROZEN_SYMBOLS)]
    if alt_acceleration.isna().any():
        return None

    btc_history = history[BTC]
    btc_variance = float(btc_history.var(ddof=0))
    if not np.isfinite(btc_variance) or btc_variance <= 0:
        return None
    alt_history = history[list(FROZEN_SYMBOLS)]
    betas = alt_history.apply(
        lambda values: float(values.cov(btc_history, ddof=0) / btc_variance)
    )
    residual_history = alt_history.subtract(
        btc_history.to_numpy()[:, None] * betas.to_numpy()[None, :]
    )
    residual_scale = residual_history.std(ddof=1).replace(0.0, np.nan)
    current_residual = current[list(FROZEN_SYMBOLS)] - betas * current[BTC]
    current_residual_z = (current_residual / residual_scale).abs()
    if not np.isfinite(current_residual_z).all():
        return None

    residual_shock_history = residual_history.abs().divide(residual_scale)
    btc_shock_history = (btc_history / float(scale[BTC])).abs()
    edge_rows = []
    shrinkage = np.sqrt(len(history) / (len(history) + cfg.shrinkage_n))
    for symbol in FROZEN_SYMBOLS:
        shock = residual_shock_history[symbol].to_numpy(float)
        btc_shock = btc_shock_history.to_numpy(float)
        forward = _correlation(shock[:-1], btc_shock[1:])
        reverse = _correlation(btc_shock[:-1], shock[1:])
        advantage = forward - reverse
        if forward > 0 and advantage > 0:
            edge_rows.append((symbol, advantage * shrinkage))
    edge_rows.sort(key=lambda item: item[1], reverse=True)
    leaders = edge_rows[: cfg.top_leaders]
    if leaders:
        weights = np.asarray([weight for _, weight in leaders], dtype=float)
        shocks = np.asarray(
            [current_residual_z[symbol] for symbol, _ in leaders], dtype=float
        )
        leader_score = float(np.average(shocks, weights=weights))
        leader_breadth = float(np.mean(shocks >= 1.0))
        edge_weight_mean = float(np.mean([weight for _, weight in edge_rows]))
    else:
        leader_score = 0.0
        leader_breadth = 0.0
        edge_weight_mean = 0.0
    btc_abs_z = float(abs(current_z[BTC]))
    return {
        "price_history_hours": int(len(history)),
        "price_symbol_count": int(len(symbols) - 1),
        "alt_abs_z_median": absolute_median,
        "alt_abs_z_dispersion": float(alt_abs.std(ddof=1)),
        "alt_shock_breadth_z1": float(alt_abs.ge(1.0).mean()),
        "alt_shock_breadth_z2": float(alt_abs.ge(2.0).mean()),
        "alt_positive_return_breadth": float(alt_z.gt(0.0).mean()),
        "alt_directional_coherence": float(directional_coherence),
        "btc_abs_z": btc_abs_z,
        "alt_btc_abs_z_gap": absolute_median - btc_abs_z,
        "alt_rv_acceleration_median": float(alt_acceleration.median()),
        "alt_rv_acceleration_breadth": float(alt_acceleration.gt(1.0).mean()),
        "alt_residual_abs_z_median": float(current_residual_z.median()),
        "alt_residual_shock_breadth": float(current_residual_z.ge(1.0).mean()),
        "directed_edge_fraction": float(len(edge_rows) / len(FROZEN_SYMBOLS)),
        "directed_edge_weight_mean": edge_weight_mean,
        "leader_shock_score": leader_score,
        "leader_shock_breadth": leader_breadth,
        "btc_receiver_gap": leader_score - btc_abs_z,
    }


def build_v2329_features(
    base: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V2329Config = V2329Config(),
) -> pd.DataFrame:
    matrix = (
        prices.pivot_table(
            index="feature_time",
            columns="symbol",
            values="close",
            aggfunc="last",
            observed=True,
        )
        .sort_index()
        [[BTC, *FROZEN_SYMBOLS]]
    )
    returns = np.log(matrix / matrix.shift(1))
    rows = []
    keep = [
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "signal_direction",
        "bucket_pressure",
        "prior_abs_pressure_threshold",
    ]
    for event in base.itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        values = _event_features(entry, returns, cfg)
        if values is None:
            continue
        row = {column: getattr(event, column) for column in keep}
        row.update(
            {
                "candidate": CANDIDATE,
                "price_feature_time": entry,
                **values,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def summarize_v2329(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features.loc[features["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "minimum_history_hours": int(local["price_history_hours"].min()),
                "median_alt_abs_z": float(local["alt_abs_z_median"].median()),
                "median_alt_btc_abs_z_gap": float(
                    local["alt_btc_abs_z_gap"].median()
                ),
                "median_rv_acceleration": float(
                    local["alt_rv_acceleration_median"].median()
                ),
                "median_directed_edge_fraction": float(
                    local["directed_edge_fraction"].median()
                ),
                "median_btc_receiver_gap": float(local["btc_receiver_gap"].median()),
            }
        )
    return pd.DataFrame(rows)


def audit_v2329(
    base: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2329Config = V2329Config(),
) -> pd.DataFrame:
    period_counts = features["period"].value_counts()
    values = features[list(VOLATILITY_FEATURES)].to_numpy(float)
    breadth_columns = [
        "alt_shock_breadth_z1",
        "alt_shock_breadth_z2",
        "alt_positive_return_breadth",
        "alt_rv_acceleration_breadth",
        "alt_residual_shock_breadth",
        "directed_edge_fraction",
        "leader_shock_breadth",
    ]
    checks = {
        "all_159_base_events_retained": len(base) == 159 and len(features) == 159,
        "minimum_period_events": all(
            int(period_counts.get(period, 0)) >= cfg.minimum_period_events
            for period in ("development", "validation", "holdout")
        ),
        "minimum_active_months": features["entry_month"].nunique()
        >= cfg.minimum_active_months,
        "full_16_alt_price_coverage": features["price_symbol_count"].eq(16).all(),
        "minimum_causal_history": features["price_history_hours"].ge(
            cfg.minimum_history_hours
        ).all(),
        "price_time_equals_entry": features["price_feature_time"].eq(
            features["entry_time"]
        ).all(),
        "all_17_features_finite": np.isfinite(values).all(),
        "breadths_between_zero_and_one": all(
            features[column].between(0.0, 1.0).all() for column in breadth_columns
        ),
        "shock_breadth_nested": features["alt_shock_breadth_z2"].le(
            features["alt_shock_breadth_z1"]
        ).all(),
        "nonnegative_scale_features": features[
            [
                "alt_abs_z_median",
                "alt_abs_z_dispersion",
                "btc_abs_z",
                "alt_rv_acceleration_median",
                "alt_residual_abs_z_median",
                "directed_edge_weight_mean",
                "leader_shock_score",
            ]
        ].ge(0.0).all().all(),
        "entry_times_unique": features["entry_time"].is_unique,
        "summary_reconciles": int(summary.loc[summary["scope"].eq("all"), "events"].iloc[0])
        == len(features),
        "outcomes_absent": {
            "triggered",
            "gross_return",
            "primary_net_return",
            "stress_net_return",
        }.isdisjoint(features.columns),
    }
    return pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )


def feature_hash_v2329(features: pd.DataFrame) -> str:
    payload = features.sort_values("entry_time").to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_v2329_event_volatility_transmission_feature_audit(
    cfg: V2329Config = V2329Config(),
) -> dict[str, Path]:
    base, prices = load_v2329_inputs(cfg)
    features = build_v2329_features(base, prices, cfg)
    summary = summarize_v2329(features)
    checks = audit_v2329(base, features, summary, cfg)
    feature_hash = feature_hash_v2329(features)
    passed = bool(checks["passed"].all())
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "event_volatility_transmission_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    features.to_parquet(paths["features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_hash": feature_hash,
                "all_checks_passed": passed,
                "volatility_features": list(VOLATILITY_FEATURES),
                "outcomes_loaded": False,
                "config": {
                    **asdict(cfg),
                    "base_feature_path": str(cfg.base_feature_path),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "feature_viable_freeze_direct_volatility_selector" if passed else "feature_audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.29 Event Volatility-Transmission Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Feature hash: `{feature_hash}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The 17 features use the current completed hourly return and strictly",
                "prior 30-day normalization/lead-lag history. No payoff was loaded.",
                "",
                "No PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "VOLATILITY_FEATURES",
    "V2329Config",
    "audit_v2329",
    "build_v2329_features",
    "feature_hash_v2329",
    "load_v2329_inputs",
    "write_v2329_event_volatility_transmission_feature_audit",
]
