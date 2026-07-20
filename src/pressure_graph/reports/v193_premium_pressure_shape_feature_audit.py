"""Feature-only audit for premium-index OHLC pressure shapes and propagation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC, V185Config
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_ohlc_panels,
)


REPORT_ROOT = Path("reports/v19_3_premium_pressure_shape_feature_audit")
FINDINGS_PATH = Path(
    "docs/v193_premium_pressure_shape_feature_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V193FeatureConfig(V185Config):
    body_z_threshold: float = 0.50
    close_location_threshold: float = 0.50
    receiver_range_z_threshold: float = 1.00
    minimum_receivers: int = 5


def build_v193_pressure_features(
    price_close: pd.DataFrame,
    premium_ohlc: dict[str, pd.DataFrame],
    cfg: V193FeatureConfig = V193FeatureConfig(),
) -> dict[str, pd.DataFrame]:
    exact = {
        field: panel.reindex(index=price_close.index, columns=price_close.columns)
        for field, panel in premium_ohlc.items()
    }
    premium_range = exact["high"] - exact["low"]
    range_mean = (
        premium_range.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .mean()
    )
    range_scale = (
        premium_range.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .std(ddof=1)
    )
    range_z = (premium_range - range_mean).div(range_scale.where(range_scale.gt(0)))

    premium_body = exact["close"] - exact["open"]
    body_scale = (
        premium_body.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .std(ddof=1)
    )
    body_z = premium_body.div(body_scale.where(body_scale.gt(0)))
    close_location = (
        2.0 * (exact["close"] - exact["low"]).div(premium_range.where(
            premium_range.gt(0)
        ))
        - 1.0
    ).clip(lower=-1.0, upper=1.0)
    price_return = price_close.pct_change(fill_method=None)
    price_abs = price_return[BTC].abs()
    price_thresholds = pd.DataFrame(
        {
            "q85": price_abs.shift(1).rolling(
                cfg.source_lookback_bars, min_periods=cfg.source_min_bars
            ).quantile(0.85),
            "q90": price_abs.shift(1).rolling(
                cfg.source_lookback_bars, min_periods=cfg.source_min_bars
            ).quantile(0.90),
        },
        index=price_close.index,
    )
    return {
        "premium_range": premium_range,
        "range_z": range_z,
        "body_z": body_z,
        "close_location": close_location,
        "price_return": price_return,
        "price_thresholds": price_thresholds,
    }


def build_v193_feature_coverage(
    price_close: pd.DataFrame,
    premium_ohlc: dict[str, pd.DataFrame],
    cfg: V193FeatureConfig = V193FeatureConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    features = build_v193_pressure_features(price_close, premium_ohlc, cfg)
    price_return = features["price_return"]
    source_sign = np.sign(price_return[BTC])
    btc_aligned_body = source_sign * features["body_z"][BTC]
    btc_aligned_location = source_sign * features["close_location"][BTC]

    aligned_body = features["body_z"].mul(source_sign, axis=0)
    aligned_location = features["close_location"].mul(source_sign, axis=0)
    receiver_transfer = (
        features["range_z"].ge(cfg.receiver_range_z_threshold)
        & aligned_body.ge(cfg.body_z_threshold)
        & aligned_location.ge(cfg.close_location_threshold)
    )
    receiver_absorption = (
        features["range_z"].ge(cfg.receiver_range_z_threshold)
        & aligned_body.le(-cfg.body_z_threshold)
        & aligned_location.le(-cfg.close_location_threshold)
    )
    transfer_count = receiver_transfer.drop(columns=[BTC]).sum(axis=1)
    absorption_count = receiver_absorption.drop(columns=[BTC]).sum(axis=1)

    base_events = pd.DataFrame(
        {
            "feature_time": price_close.index,
            "source_feature_time": price_close.index,
            "btc_return_15m": price_return[BTC].to_numpy(),
            "source_sign": source_sign.to_numpy(),
            "btc_premium_range_z": features["range_z"][BTC].to_numpy(),
            "btc_aligned_body_z": btc_aligned_body.to_numpy(),
            "btc_aligned_close_location": btc_aligned_location.to_numpy(),
            "eligible_transfer_receivers": transfer_count.to_numpy(),
            "eligible_absorption_receivers": absorption_count.to_numpy(),
            "period": price_close.index.map(_period),
        }
    )
    for label in ("q85", "q90"):
        base_events[f"btc_abs_return_{label}"] = features["price_thresholds"][
            label
        ].to_numpy()

    rows: list[dict[str, object]] = []
    for price_quantile in ("q85", "q90"):
        price_shock = base_events["btc_return_15m"].abs().ge(
            base_events[f"btc_abs_return_{price_quantile}"]
        )
        for range_threshold in (1.0, 1.5, 2.0):
            range_shock = base_events["btc_premium_range_z"].ge(range_threshold)
            for shape, body_mask, location_mask, receiver_column in (
                (
                    "through_pressure",
                    base_events["btc_aligned_body_z"].ge(cfg.body_z_threshold),
                    base_events["btc_aligned_close_location"].ge(
                        cfg.close_location_threshold
                    ),
                    "eligible_transfer_receivers",
                ),
                (
                    "opposing_absorption",
                    base_events["btc_aligned_body_z"].le(-cfg.body_z_threshold),
                    base_events["btc_aligned_close_location"].le(
                        -cfg.close_location_threshold
                    ),
                    "eligible_absorption_receivers",
                ),
            ):
                selected = base_events[
                    price_shock & range_shock & body_mask & location_mask
                ]
                bucket = selected[selected[receiver_column].ge(cfg.minimum_receivers)]
                rows.append(
                    {
                        "price_quantile": price_quantile,
                        "btc_range_z_threshold": range_threshold,
                        "shape": shape,
                        "events": len(selected),
                        "development_events": int(
                            selected["period"].eq("development").sum()
                        ),
                        "validation_events": int(
                            selected["period"].eq("validation").sum()
                        ),
                        "holdout_events": int(
                            selected["period"].eq("holdout").sum()
                        ),
                        "median_btc_range_z": float(
                            selected["btc_premium_range_z"].median()
                        ),
                        "median_aligned_body_z": float(
                            selected["btc_aligned_body_z"].median()
                        ),
                        "median_aligned_close_location": float(
                            selected["btc_aligned_close_location"].median()
                        ),
                        "median_eligible_receivers": float(
                            selected[receiver_column].median()
                        ),
                        "bucket_events_min5": len(bucket),
                        "bucket_development_events": int(
                            bucket["period"].eq("development").sum()
                        ),
                        "bucket_validation_events": int(
                            bucket["period"].eq("validation").sum()
                        ),
                        "bucket_holdout_events": int(
                            bucket["period"].eq("holdout").sum()
                        ),
                    }
                )
    return pd.DataFrame(rows), base_events, features


def _write_findings(coverage: pd.DataFrame, path: Path) -> None:
    text = [
        "# v19.3 Premium Pressure-Shape Feature-Only Audit",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Range and body scales use shifted prior-30-day windows with at least",
        "20 days of history. Close location uses only the completed premium-index",
        "bar. No future candidate return was calculated or inspected in this audit.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v193_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V193FeatureConfig = V193FeatureConfig(),
) -> dict[str, Path]:
    price_close, _ = load_v184_exact_panels(metrics_root, kline_root)
    premium_ohlc = load_v190_premium_ohlc_panels(premium_root)
    coverage, events, features = build_v193_feature_coverage(
        price_close, premium_ohlc, cfg
    )
    root = ensure_dir(report_root)
    outputs = {
        "coverage": root / "feature_coverage.csv",
        "events": root / "feature_events.parquet",
        "range_z": root / "premium_range_z.parquet",
        "body_z": root / "premium_body_z.parquet",
        "close_location": root / "premium_close_location.parquet",
        "findings": findings_path,
    }
    coverage.to_csv(outputs["coverage"], index=False)
    events.to_parquet(outputs["events"], index=False)
    features["range_z"].to_parquet(outputs["range_z"])
    features["body_z"].to_parquet(outputs["body_z"])
    features["close_location"].to_parquet(outputs["close_location"])
    _write_findings(coverage, findings_path)
    return outputs


__all__ = [
    "V193FeatureConfig",
    "build_v193_feature_coverage",
    "build_v193_pressure_features",
    "write_v193_feature_audit",
]
