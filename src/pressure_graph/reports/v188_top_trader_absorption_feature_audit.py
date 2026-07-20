"""Feature-only coverage audit for top-trader absorption hypotheses."""
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


REPORT_ROOT = Path("reports/v18_8_top_trader_absorption_feature_audit")
FINDINGS_PATH = Path(
    "docs/v188_top_trader_absorption_feature_audit_2026_07_16.md"
)


def build_v188_feature_coverage(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg: V185Config = V185Config(),
) -> pd.DataFrame:
    btc_return = close[BTC].pct_change(fill_method=None)
    returns = close.pct_change(fill_method=None)
    direction = np.sign(btc_return)
    top_position = np.log(
        panels["sum_toptrader_long_short_ratio"].where(
            panels["sum_toptrader_long_short_ratio"].gt(0)
        )
    )
    flow = np.log(
        panels["sum_taker_long_short_vol_ratio"].where(
            panels["sum_taker_long_short_vol_ratio"].gt(0)
        )
    )
    oi_change = np.log(
        panels["sum_open_interest"].where(panels["sum_open_interest"].gt(0))
    ).diff()
    top_change = top_position.diff()
    btc_absorption = -direction * top_change[BTC]

    rows: list[dict[str, object]] = []
    for source_quantile in (0.85, 0.90):
        signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=source_quantile
        )
        signals = signals[signals["kind"].eq(UNWIND)].copy()
        for absorption_quantile in (0.50, 0.55, 0.60, 0.65):
            threshold = (
                btc_absorption.shift(1)
                .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
                .quantile(absorption_quantile)
            )
            local = signals[
                signals["feature_time"].map(btc_absorption).ge(
                    signals["feature_time"].map(threshold)
                )
            ].copy()
            local["period"] = local["feature_time"].map(_period)
            eligible_counts = []
            for event in local.itertuples(index=False):
                timestamp = pd.Timestamp(event.feature_time)
                source_sign = float(event.source_sign)
                eligible = (
                    (-source_sign * top_change.loc[timestamp]).gt(0)
                    & (source_sign * flow.loc[timestamp]).gt(0)
                    & (-oi_change.loc[timestamp]).gt(0)
                    & (source_sign * returns.loc[timestamp]).gt(0)
                ).drop(labels=[BTC], errors="ignore")
                eligible_counts.append(int(eligible.sum()))
            local["eligible_receivers"] = eligible_counts
            rows.append(
                {
                    "source_return_quantile": source_quantile,
                    "absorption_quantile": absorption_quantile,
                    "direct_events": len(local),
                    "development_events": int(
                        local["period"].eq("development").sum()
                    ),
                    "validation_events": int(
                        local["period"].eq("validation").sum()
                    ),
                    "holdout_events": int(local["period"].eq("holdout").sum()),
                    "bucket_events_min5": int(
                        local["eligible_receivers"].ge(5).sum()
                    ),
                    "eligible_receivers_median": (
                        float(local["eligible_receivers"].median())
                        if len(local)
                        else np.nan
                    ),
                    "eligible_receivers_minimum": (
                        int(local["eligible_receivers"].min()) if len(local) else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_findings(coverage: pd.DataFrame, path: Path) -> None:
    text = [
        "# v18.8 Top-Trader Absorption Feature-Only Audit",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This audit inspects only contemporaneous/lagged features and event counts;",
        "no future candidate return was calculated or inspected.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v188_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V185Config = V185Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    coverage = build_v188_feature_coverage(close, panels, cfg)
    root = ensure_dir(report_root)
    outputs = {"coverage": root / "feature_coverage.csv", "findings": findings_path}
    coverage.to_csv(outputs["coverage"], index=False)
    _write_findings(coverage, findings_path)
    return outputs


__all__ = ["build_v188_feature_coverage", "write_v188_feature_audit"]
