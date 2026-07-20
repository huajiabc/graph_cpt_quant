"""Data-quality audit for BTC-inclusive Binance five-minute metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
KLINE_ROOT = Path("data/raw/binance/klines")
REPORT_ROOT = Path("reports/v18_4_btc_inclusive_metrics_audit")
FINDINGS_PATH = Path("docs/v184_btc_inclusive_metrics_audit_2026_07_16.md")
FIELDS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
BTC = "BTCUSDT"
EXCLUDED = {"XAUTUSDT"}


def load_v184_exact_panels(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_symbols = {path.stem.upper() for path in metrics_root.glob("*.parquet")}
    kline_symbols = {path.stem.upper() for path in kline_root.glob("*.parquet")}
    symbols = sorted((metric_symbols & kline_symbols) - EXCLUDED)
    close_frames: list[pd.Series] = []
    for symbol in symbols:
        frame = pd.read_parquet(
            kline_root / f"{symbol}.parquet",
            columns=["bar_close_time", "close"],
        )
        frame["bar_close_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        series = (
            frame.dropna(subset=["bar_close_time"])
            .drop_duplicates("bar_close_time", keep="last")
            .set_index("bar_close_time")["close"]
            .rename(symbol)
        )
        close_frames.append(series)
    close = pd.concat(close_frames, axis=1).sort_index()
    metric_start = pd.Timestamp("2025-07-01 00:00", tz="UTC")
    metric_end = pd.Timestamp("2026-07-14 23:55", tz="UTC")
    times = close.index[(close.index >= metric_start) & (close.index <= metric_end)]
    close = close.reindex(times)
    panels = {
        field: pd.DataFrame(index=times, columns=symbols, dtype=float)
        for field in FIELDS
    }
    for symbol in symbols:
        frame = pd.read_parquet(
            metrics_root / f"{symbol}.parquet",
            columns=["create_time", *FIELDS],
        )
        frame["create_time"] = pd.to_datetime(
            frame["create_time"], utc=True, errors="coerce"
        )
        frame = (
            frame.dropna(subset=["create_time"])
            .drop_duplicates("create_time", keep="last")
            .set_index("create_time")
        )
        exact = frame.reindex(times)
        for field in FIELDS:
            panels[field][symbol] = pd.to_numeric(exact[field], errors="coerce")
    close.index.name = "feature_time"
    for panel in panels.values():
        panel.index.name = "feature_time"
    return close, panels


def _symbol_quality(metrics_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["create_time", "source_day", *FIELDS])
        time = pd.DatetimeIndex(pd.to_datetime(frame["create_time"], utc=True))
        regular = pd.date_range(time.min(), time.max(), freq="5min", tz="UTC")
        row: dict[str, object] = {
            "symbol": path.stem.upper(),
            "rows": len(frame),
            "unique_times": time.nunique(),
            "duplicate_times": int(time.duplicated().sum()),
            "first_time": time.min(),
            "last_time": time.max(),
            "source_days": int(pd.to_datetime(frame["source_day"]).dt.normalize().nunique()),
            "regular_grid_points": len(regular),
            "missing_grid_points": len(regular.difference(time)),
            "grid_coverage": time.nunique() / len(regular),
        }
        for field in FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce")
            row[f"{field}_missing"] = int(values.isna().sum())
            row[f"{field}_nonfinite"] = int(np.isinf(values).sum())
            row[f"{field}_nonpositive"] = int(values.le(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def audit_v184(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(metrics_root / "manifest.csv")
    coverage = json.loads((metrics_root / "coverage.json").read_text(encoding="utf-8"))
    quality = _symbol_quality(metrics_root)
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    taker_coverage = panels["sum_taker_long_short_vol_ratio"].notna().sum(axis=1)
    minimum_field_coverage = pd.concat(
        [panel.notna().sum(axis=1).rename(field) for field, panel in panels.items()],
        axis=1,
    ).min(axis=1)
    breadth = pd.DataFrame(
        {
            "feature_time": close.index,
            "price_symbols": close.notna().sum(axis=1).to_numpy(),
            "taker_metric_symbols": taker_coverage.to_numpy(),
            "minimum_field_symbols": minimum_field_coverage.to_numpy(),
            "btc_taker_present": panels["sum_taker_long_short_vol_ratio"][BTC]
            .notna()
            .to_numpy(),
            "btc_all_fields_present": pd.concat(
                [panel[BTC].notna() for panel in panels.values()], axis=1
            )
            .all(axis=1)
            .to_numpy(),
        }
    )
    btc = quality[quality["symbol"].eq(BTC)].iloc[0]
    ratio_missing = sum(
        int(btc[f"{field}_missing"])
        for field in FIELDS
        if "ratio" in field
    )
    checks = {
        "manifest_expected_symbols_73": len(manifest) == 73,
        "manifest_covered_symbols_72": manifest["rows"].gt(0).sum() == 72,
        "manifest_mnt_explicit_missing": bool(
            manifest.loc[manifest["bybit_symbol"].eq("MNTUSDT"), "rows"].eq(0).all()
        ),
        "coverage_full_directory_mode": coverage["config"].get("inventory_mode")
        == "full_directory",
        "btc_source_days_379": int(btc["source_days"]) == 379,
        "btc_unique_no_duplicates": int(btc["duplicate_times"]) == 0,
        "btc_grid_coverage_999": float(btc["grid_coverage"]) >= 0.999,
        "btc_taker_no_missing": int(
            btc["sum_taker_long_short_vol_ratio_missing"]
        )
        == 0,
        "btc_ratio_missing_below_01pct": ratio_missing / int(btc["rows"]) < 0.001,
        "btc_numeric_no_infinite": all(
            int(btc[f"{field}_nonfinite"]) == 0 for field in FIELDS
        ),
        "kline_metric_intersection_45": len(close.columns) >= 45,
        "exact_breadth_40_998": float(taker_coverage.ge(40).mean()) >= 0.998,
        "all_field_breadth_40_998": float(minimum_field_coverage.ge(40).mean())
        >= 0.998,
        "btc_exact_close_coverage_999": float(breadth["btc_taker_present"].mean())
        >= 0.999,
    }
    audit = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed)}
            for name, passed in checks.items()
        ]
    )
    audit["verdict"] = (
        "audit_pass_btc_inclusive_metrics_ready"
        if audit["passed"].all()
        else "audit_failure_metrics_not_ready"
    )
    return audit, quality, breadth


def _write_findings(
    audit: pd.DataFrame,
    quality: pd.DataFrame,
    breadth: pd.DataFrame,
    path: Path,
) -> None:
    btc = quality[quality["symbol"].eq(BTC)].iloc[0]
    text = [
        "# v18.4 BTC-Inclusive Binance Metrics Audit",
        "",
        f"Verdict: `{audit['verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}.",
        "",
        f"BTC rows: {int(btc['rows']):,}; source days: {int(btc['source_days'])}; "
        f"five-minute grid coverage: {float(btc['grid_coverage']):.6%}.",
        "",
        f"Exact 15-minute panel symbols: {int(breadth['price_symbols'].max())}; "
        f"bars: {len(breadth):,}; bars with >=40 taker symbols: "
        f"{int(breadth['taker_metric_symbols'].ge(40).sum()):,}.",
        "",
        "No forward fill is used. A metric is available only when its archived",
        "timestamp exactly equals the completed 15-minute price-bar close.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v184_btc_inclusive_metrics_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, quality, breadth = audit_v184()
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "symbol_quality": root / "symbol_quality.csv",
        "exact_breadth": root / "exact_15m_breadth.parquet",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    quality.to_csv(outputs["symbol_quality"], index=False)
    breadth.to_parquet(outputs["exact_breadth"], index=False)
    _write_findings(audit, quality, breadth, findings_path)
    return outputs


__all__ = [
    "FIELDS",
    "audit_v184",
    "load_v184_exact_panels",
    "write_v184_btc_inclusive_metrics_audit",
]
