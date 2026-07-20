"""Feature-only coverage audit for premium innovation during OI unwind events."""
from __future__ import annotations

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
from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BTC,
    UNWIND,
    V185Config,
    build_v185_source_signals,
)
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_panel,
)


REPORT_ROOT = Path("reports/v19_1_premium_innovation_feature_audit")
FINDINGS_PATH = Path(
    "docs/v191_premium_innovation_feature_audit_2026_07_17.md"
)


def build_v191_premium_features(
    close: pd.DataFrame,
    premium: pd.DataFrame,
    cfg: V185Config = V185Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = premium.reindex(index=close.index, columns=close.columns)
    innovation = exact.diff()
    scale = (
        innovation.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .std(ddof=1)
    )
    innovation_z = innovation.div(scale.where(scale.gt(0)))
    btc_direction = np.sign(close[BTC].pct_change(fill_method=None))
    aligned = innovation_z.mul(btc_direction, axis=0)
    valid = aligned.notna().sum(axis=1)
    transmitted = aligned.gt(1.0).sum(axis=1)
    breadth = pd.DataFrame(
        {
            "premium_valid_symbols": valid,
            "premium_transmitted_symbols": transmitted,
            "premium_innovation_breadth": transmitted.div(valid.where(valid.gt(0))),
        },
        index=close.index,
    )
    breadth["breadth_q70"] = (
        breadth["premium_innovation_breadth"]
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(0.70)
    )
    return innovation_z, breadth


def build_v191_feature_coverage(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    premium: pd.DataFrame,
    cfg: V185Config = V185Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    innovation_z, breadth = build_v191_premium_features(close, premium, cfg)
    rows: list[dict[str, object]] = []
    event_rows: list[pd.DataFrame] = []
    for source_quantile in (0.85, 0.90):
        signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=source_quantile
        )
        signals = signals[signals["kind"].eq(UNWIND)].copy()
        signals["period"] = signals["feature_time"].map(_period)
        signals["btc_premium_innovation_z"] = signals["feature_time"].map(
            innovation_z[BTC]
        )
        signals["aligned_btc_premium_z"] = (
            signals["source_sign"] * signals["btc_premium_innovation_z"]
        )
        signals["premium_innovation_breadth"] = signals["feature_time"].map(
            breadth["premium_innovation_breadth"]
        )
        signals["premium_breadth_q70"] = signals["feature_time"].map(
            breadth["breadth_q70"]
        )
        receiver_counts = []
        for event in signals.itertuples(index=False):
            aligned = float(event.source_sign) * innovation_z.loc[event.feature_time]
            receiver_counts.append(int(aligned.drop(labels=[BTC]).gt(1.0).sum()))
        signals["eligible_premium_receivers"] = receiver_counts
        signals["source_return_quantile"] = source_quantile
        event_rows.append(signals)
        base_shock = signals["aligned_btc_premium_z"].ge(1.0)
        broad = base_shock & signals["premium_innovation_breadth"].ge(
            signals["premium_breadth_q70"]
        )
        for source_side, side_mask in (
            ("long_liquidation", signals["source_sign"].lt(0)),
            ("short_cover", signals["source_sign"].gt(0)),
            ("all", signals["source_sign"].ne(0)),
        ):
            for filter_name, filter_mask in (
                ("btc_premium_shock", base_shock),
                ("broad_premium_shock", broad),
            ):
                local = signals[side_mask & filter_mask.fillna(False)]
                bucket_local = local[local["eligible_premium_receivers"].ge(5)]
                rows.append(
                    {
                        "source_return_quantile": source_quantile,
                        "source_side": source_side,
                        "filter": filter_name,
                        "events": len(local),
                        "development_events": int(
                            local["period"].eq("development").sum()
                        ),
                        "validation_events": int(
                            local["period"].eq("validation").sum()
                        ),
                        "holdout_events": int(
                            local["period"].eq("holdout").sum()
                        ),
                        "median_aligned_btc_premium_z": float(
                            local["aligned_btc_premium_z"].median()
                        ),
                        "median_premium_breadth": float(
                            local["premium_innovation_breadth"].median()
                        ),
                        "median_eligible_receivers": float(
                            local["eligible_premium_receivers"].median()
                        ),
                        "bucket_events_min5": int(
                            len(bucket_local)
                        ),
                        "bucket_development_events": int(
                            bucket_local["period"].eq("development").sum()
                        ),
                        "bucket_validation_events": int(
                            bucket_local["period"].eq("validation").sum()
                        ),
                        "bucket_holdout_events": int(
                            bucket_local["period"].eq("holdout").sum()
                        ),
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.concat(event_rows, ignore_index=True),
        breadth.reset_index(),
    )


def _write_findings(coverage: pd.DataFrame, path: Path) -> None:
    text = [
        "# v19.1 Premium-Innovation Feature-Only Audit",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Premium innovation is an exact 15-minute close difference divided by",
        "shifted prior-30-day volatility. No future candidate return was calculated",
        "or inspected in this audit.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v191_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V185Config = V185Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    premium = load_v190_premium_panel(premium_root)
    coverage, events, breadth = build_v191_feature_coverage(
        close, panels, premium, cfg
    )
    root = ensure_dir(report_root)
    outputs = {
        "coverage": root / "feature_coverage.csv",
        "events": root / "feature_events.parquet",
        "breadth": root / "premium_innovation_breadth.parquet",
        "findings": findings_path,
    }
    coverage.to_csv(outputs["coverage"], index=False)
    events.to_parquet(outputs["events"], index=False)
    breadth.to_parquet(outputs["breadth"], index=False)
    _write_findings(coverage, findings_path)
    return outputs


__all__ = [
    "build_v191_feature_coverage",
    "build_v191_premium_features",
    "write_v191_feature_audit",
]
