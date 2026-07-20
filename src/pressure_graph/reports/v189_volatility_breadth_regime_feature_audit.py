"""Feature-only coverage audit for cross-asset volatility breadth regimes."""
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
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)


REPORT_ROOT = Path("reports/v18_9_volatility_breadth_regime_feature_audit")
FINDINGS_PATH = Path(
    "docs/v189_volatility_breadth_regime_feature_audit_2026_07_16.md"
)


def build_v189_breadth(
    returns: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for month, local_risk in risk.groupby("risk_month", sort=True):
        start = pd.Timestamp(month)
        end = start + pd.offsets.MonthBegin(1)
        month_returns = returns.loc[
            (returns.index >= start) & (returns.index < end)
        ]
        names = local_risk["receiver"].astype(str).tolist()
        volatility = local_risk.set_index("receiver").reindex(names)[
            "return_volatility"
        ]
        standardized = month_returns[names].div(volatility, axis=1)
        aligned = standardized.mul(np.sign(month_returns[BTC]), axis=0)
        valid = aligned.notna().sum(axis=1)
        transmitted = aligned.gt(1.0).sum(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "feature_time": month_returns.index,
                    "breadth_valid_receivers": valid,
                    "breadth_transmitted_receivers": transmitted,
                    "volatility_breadth": transmitted.div(valid.where(valid.gt(0))),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).set_index("feature_time").sort_index()


def build_v189_feature_coverage(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg: V185Config = V185Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    seed_signals = build_v185_source_signals(
        close, panels, cfg, return_quantile=0.85
    )
    seed_signals = seed_signals[seed_signals["kind"].eq(UNWIND)]
    risk = build_v187_monthly_risk(
        returns,
        seed_signals["feature_time"].min(),
        seed_signals["feature_time"].max(),
    )
    breadth = build_v189_breadth(returns, risk)
    breadth["breadth_q30"] = (
        breadth["volatility_breadth"]
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(0.30)
    )
    breadth["breadth_q70"] = (
        breadth["volatility_breadth"]
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(0.70)
    )
    rows: list[dict[str, object]] = []
    for source_quantile in (0.85, 0.90):
        signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=source_quantile
        )
        signals = signals[signals["kind"].eq(UNWIND)].copy()
        signals["period"] = signals["feature_time"].map(_period)
        signals["breadth"] = signals["feature_time"].map(
            breadth["volatility_breadth"]
        )
        signals["q30"] = signals["feature_time"].map(breadth["breadth_q30"])
        signals["q70"] = signals["feature_time"].map(breadth["breadth_q70"])
        regimes = {
            "low_breadth_exhaustion": signals["breadth"].le(signals["q30"]),
            "high_breadth_cascade": signals["breadth"].ge(signals["q70"]),
        }
        for regime, mask in regimes.items():
            local = signals[mask.fillna(False)]
            rows.append(
                {
                    "source_return_quantile": source_quantile,
                    "regime": regime,
                    "events": len(local),
                    "development_events": int(
                        local["period"].eq("development").sum()
                    ),
                    "validation_events": int(
                        local["period"].eq("validation").sum()
                    ),
                    "holdout_events": int(local["period"].eq("holdout").sum()),
                    "median_breadth": float(local["breadth"].median()),
                    "median_transmitted_receivers": float(
                        local["feature_time"].map(
                            breadth["breadth_transmitted_receivers"]
                        ).median()
                    ),
                    "median_valid_receivers": float(
                        local["feature_time"].map(
                            breadth["breadth_valid_receivers"]
                        ).median()
                    ),
                }
            )
    return pd.DataFrame(rows), breadth.reset_index()


def _write_findings(coverage: pd.DataFrame, path: Path) -> None:
    text = [
        "# v18.9 Volatility-Breadth Regime Feature-Only Audit",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Breadth uses only event-time returns standardized by prior-month risk",
        "estimates, and q30/q70 thresholds use shifted prior-30-day breadth.",
        "No future candidate return was calculated or inspected.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v189_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V185Config = V185Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    coverage, breadth = build_v189_feature_coverage(close, panels, cfg)
    root = ensure_dir(report_root)
    outputs = {
        "coverage": root / "feature_coverage.csv",
        "breadth": root / "volatility_breadth.parquet",
        "findings": findings_path,
    }
    coverage.to_csv(outputs["coverage"], index=False)
    breadth.to_parquet(outputs["breadth"], index=False)
    _write_findings(coverage, findings_path)
    return outputs


__all__ = [
    "build_v189_breadth",
    "build_v189_feature_coverage",
    "write_v189_feature_audit",
]
