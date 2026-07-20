"""Data-quality audit for checksummed Binance USD-M premium-index klines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)


PREMIUM_ROOT = Path("data/external/binance_um_premium_15m")
REPORT_ROOT = Path("reports/v19_0_binance_premium_index_audit")
FINDINGS_PATH = Path("docs/v190_binance_premium_index_audit_2026_07_17.md")
BTC = "BTCUSDT"
TON = "TONUSDT"


def load_v190_premium_panel(
    premium_root: Path = PREMIUM_ROOT,
) -> pd.DataFrame:
    series = []
    for path in sorted(premium_root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["feature_time", "close"])
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        local = (
            frame.dropna(subset=["feature_time"])
            .drop_duplicates("feature_time", keep="last")
            .set_index("feature_time")["close"]
            .rename(path.stem.upper())
        )
        series.append(local)
    panel = pd.concat(series, axis=1).sort_index()
    panel.index.name = "feature_time"
    return panel


def load_v190_premium_ohlc_panels(
    premium_root: Path = PREMIUM_ROOT,
) -> dict[str, pd.DataFrame]:
    """Load exact premium-index OHLC values without filling missing timestamps."""
    series: dict[str, list[pd.Series]] = {
        field: [] for field in ("open", "high", "low", "close")
    }
    for path in sorted(premium_root.glob("*.parquet")):
        frame = pd.read_parquet(
            path, columns=["feature_time", "open", "high", "low", "close"]
        )
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        frame = (
            frame.dropna(subset=["feature_time"])
            .drop_duplicates("feature_time", keep="last")
            .set_index("feature_time")
        )
        for field in series:
            series[field].append(
                pd.to_numeric(frame[field], errors="coerce").rename(
                    path.stem.upper()
                )
            )
    panels = {
        field: pd.concat(values, axis=1).sort_index()
        for field, values in series.items()
    }
    for panel in panels.values():
        panel.index.name = "feature_time"
    return panels


def _symbol_quality(premium_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(premium_root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        time = pd.DatetimeIndex(pd.to_datetime(frame["feature_time"], utc=True))
        regular = pd.date_range(time.min(), time.max(), freq="15min", tz="UTC")
        ohlc = frame[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        finite = np.isfinite(ohlc)
        high_floor = ohlc[["open", "close", "low"]].max(axis=1)
        low_ceiling = ohlc[["open", "close", "high"]].min(axis=1)
        rows.append(
            {
                "symbol": path.stem.upper(),
                "rows": len(frame),
                "unique_times": time.nunique(),
                "duplicates": int(time.duplicated().sum()),
                "first_time": time.min(),
                "last_time": time.max(),
                "regular_grid_points": len(regular),
                "missing_grid_points": len(regular.difference(time)),
                "grid_coverage": time.nunique() / len(regular),
                "ohlc_missing": int(ohlc.isna().sum().sum()),
                "ohlc_nonfinite": int((~finite).sum().sum()),
                "ohlc_range_bad": int(
                    (ohlc["high"].lt(high_floor) | ohlc["low"].gt(low_ceiling)).sum()
                ),
                "close_abs_max": float(ohlc["close"].abs().max()),
                "source_archives": int(frame["source_archive"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def audit_v190(
    premium_root: Path = PREMIUM_ROOT,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(premium_root / "manifest.csv")
    download = pd.read_csv(premium_root / "last_download.csv")
    quality = _symbol_quality(premium_root)
    premium = load_v190_premium_panel(premium_root)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    exact = premium.reindex(close.index)
    exact_breadth = exact.notna().sum(axis=1)
    research_rows = len(close)
    full_exact_rows = int(exact_breadth.eq(len(close.columns)).sum())
    at_least_45 = int(exact_breadth.ge(45).sum())

    full_symbols = quality[quality["symbol"].ne(TON)]
    ton = quality[quality["symbol"].eq(TON)].iloc[0]
    checks = {
        "manifest_symbols_46": len(manifest) == 46,
        "all_symbols_covered": manifest["rows"].gt(0).all(),
        "last_download_symbols_46": len(download) == 46,
        "all_downloaded_archives_checksummed": download[
            "downloaded_archives"
        ].eq(download["verified_archives"]).all(),
        "verified_archives_1182": int(download["verified_archives"].sum()) == 1182,
        "premium_files_46": len(quality) == 46,
        "timestamps_unique": quality["duplicates"].eq(0).all(),
        "ohlc_complete_finite": quality["ohlc_missing"].eq(0).all()
        and quality["ohlc_nonfinite"].eq(0).all(),
        "ohlc_ranges_valid": quality["ohlc_range_bad"].eq(0).all(),
        "full_symbols_common_rows": full_symbols["rows"].nunique() == 1
        and int(full_symbols["rows"].iloc[0]) == 36_288,
        "full_symbols_single_official_day_gap": full_symbols[
            "missing_grid_points"
        ].eq(96).all(),
        "ton_partial_stop_explicit": int(ton["rows"]) == 34_848
        and pd.Timestamp(ton["last_time"]) == pd.Timestamp(
            "2026-06-29 00:00:00+00:00"
        ),
        "price_premium_symbols_exact_46": set(close.columns) == set(premium.columns),
        "research_exact_coverage_995": at_least_45 / research_rows >= 0.995,
        "btc_exact_coverage_995": exact[BTC].notna().mean() >= 0.995,
    }
    audit = pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(passed),
                "value": {
                    "manifest_symbols_46": len(manifest),
                    "all_symbols_covered": int(manifest["rows"].gt(0).sum()),
                    "last_download_symbols_46": len(download),
                    "all_downloaded_archives_checksummed": int(
                        download["verified_archives"].sum()
                    ),
                    "verified_archives_1182": int(
                        download["verified_archives"].sum()
                    ),
                    "premium_files_46": len(quality),
                    "timestamps_unique": int(quality["duplicates"].sum()),
                    "ohlc_complete_finite": int(
                        quality["ohlc_missing"].sum()
                        + quality["ohlc_nonfinite"].sum()
                    ),
                    "ohlc_ranges_valid": int(quality["ohlc_range_bad"].sum()),
                    "full_symbols_common_rows": int(full_symbols["rows"].min()),
                    "full_symbols_single_official_day_gap": int(
                        full_symbols["missing_grid_points"].max()
                    ),
                    "ton_partial_stop_explicit": int(ton["rows"]),
                    "price_premium_symbols_exact_46": len(
                        set(close.columns) & set(premium.columns)
                    ),
                    "research_exact_coverage_995": at_least_45 / research_rows,
                    "btc_exact_coverage_995": float(exact[BTC].notna().mean()),
                }[name],
            }
            for name, passed in checks.items()
        ]
    )
    audit["research_rows"] = research_rows
    audit["full_exact_rows"] = full_exact_rows
    audit["at_least_45_rows"] = at_least_45
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_premium_index_ready",
        "audit_failure_requires_investigation",
    )
    breadth = pd.DataFrame(
        {
            "feature_time": close.index,
            "price_symbols": close.notna().sum(axis=1).to_numpy(),
            "premium_symbols": exact_breadth.to_numpy(),
            "btc_premium_present": exact[BTC].notna().to_numpy(),
        }
    )
    return audit, quality, breadth


def _write_findings(
    audit: pd.DataFrame,
    quality: pd.DataFrame,
    path: Path,
) -> None:
    failed = audit[~audit["passed"]]
    ton = quality[quality["symbol"].eq(TON)].iloc[0]
    text = [
        "# v19.0 Binance Premium-Index Data Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        f"TON is explicitly partial through {ton['last_time']}; other symbols share",
        "the official 2026-06-29 archive gap. Exact timestamps are retained without",
        "forward filling. No future strategy return was inspected in this audit.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v190_premium_audit(
    premium_root: Path = PREMIUM_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, quality, breadth = audit_v190(premium_root)
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "quality": root / "symbol_quality.csv",
        "breadth": root / "exact_15m_breadth.parquet",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    breadth.to_parquet(outputs["breadth"], index=False)
    _write_findings(audit, quality, findings_path)
    return outputs


__all__ = [
    "PREMIUM_ROOT",
    "audit_v190",
    "load_v190_premium_ohlc_panels",
    "load_v190_premium_panel",
    "write_v190_premium_audit",
]
