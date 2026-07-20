"""Data-quality and identity audit for Binance mark/index price klines."""
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
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_panel,
)


MARK_ROOT = Path("data/external/binance_um_mark_price_15m")
INDEX_ROOT = Path("data/external/binance_um_index_price_15m")
REPORT_ROOT = Path("reports/v19_9_binance_reference_price_audit")
FINDINGS_PATH = Path("docs/v199_binance_reference_price_audit_2026_07_17.md")
FIELDS = ("open", "high", "low", "close")


def load_reference_ohlc_panels(root: Path) -> dict[str, pd.DataFrame]:
    """Load exact completed-bar reference OHLC without filling timestamps."""
    series: dict[str, list[pd.Series]] = {field: [] for field in FIELDS}
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["feature_time", *FIELDS])
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        frame = (
            frame.dropna(subset=["feature_time"])
            .drop_duplicates("feature_time", keep="last")
            .set_index("feature_time")
        )
        for field in FIELDS:
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


def _symbol_quality(root: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=["feature_time", "source_archive", "dataset", *FIELDS],
        )
        time = pd.DatetimeIndex(
            pd.to_datetime(frame["feature_time"], utc=True, errors="coerce")
        )
        ohlc = frame[list(FIELDS)].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(ohlc)
        regular = pd.date_range(time.min(), time.max(), freq="15min", tz="UTC")
        high_floor = ohlc[["open", "close", "low"]].max(axis=1)
        low_ceiling = ohlc[["open", "close", "high"]].min(axis=1)
        rows.append(
            {
                "reference": label,
                "symbol": path.stem.upper(),
                "dataset": "|".join(sorted(frame["dataset"].dropna().unique())),
                "rows": len(frame),
                "unique_times": time.nunique(),
                "duplicates": int(time.duplicated().sum()),
                "first_time": time.min(),
                "last_time": time.max(),
                "regular_grid_points": len(regular),
                "missing_grid_points": len(regular.difference(time)),
                "source_archives": int(frame["source_archive"].nunique()),
                "ohlc_missing": int(ohlc.isna().sum().sum()),
                "ohlc_nonfinite": int((~finite).sum().sum()),
                "ohlc_nonpositive": int(ohlc.le(0).sum().sum()),
                "ohlc_range_bad": int(
                    (
                        ohlc["high"].lt(high_floor)
                        | ohlc["low"].gt(low_ceiling)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_reference_relationships(
    futures_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    index_close: pd.DataFrame,
    premium_close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe exact-time relationships without reading future returns."""
    times = (
        futures_close.index.intersection(mark_close.index)
        .intersection(index_close.index)
        .intersection(premium_close.index)
    )
    symbols = sorted(
        set(futures_close.columns)
        & set(mark_close.columns)
        & set(index_close.columns)
        & set(premium_close.columns)
    )
    futures = futures_close.reindex(index=times, columns=symbols)
    mark = mark_close.reindex(index=times, columns=symbols)
    index = index_close.reindex(index=times, columns=symbols)
    premium = premium_close.reindex(index=times, columns=symbols)
    implied = mark.div(index).sub(1.0)

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        local = pd.concat(
            {
                "futures": futures[symbol],
                "mark": mark[symbol],
                "index": index[symbol],
                "premium": premium[symbol],
                "implied": implied[symbol],
            },
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        delta = local["implied"].sub(local["premium"]).abs()
        rows.append(
            {
                "symbol": symbol,
                "aligned_points": len(local),
                "implied_premium_correlation": local["implied"].corr(
                    local["premium"]
                ),
                "median_abs_implied_premium_difference": float(delta.median()),
                "p99_abs_implied_premium_difference": float(delta.quantile(0.99)),
                "max_abs_implied_premium_difference": float(delta.max()),
                "median_abs_mark_futures_difference": float(
                    local["mark"].div(local["futures"]).sub(1.0).abs().median()
                ),
                "p99_abs_mark_futures_difference": float(
                    local["mark"]
                    .div(local["futures"])
                    .sub(1.0)
                    .abs()
                    .quantile(0.99)
                ),
                "median_abs_index_futures_difference": float(
                    local["index"].div(local["futures"]).sub(1.0).abs().median()
                ),
                "p99_abs_index_futures_difference": float(
                    local["index"]
                    .div(local["futures"])
                    .sub(1.0)
                    .abs()
                    .quantile(0.99)
                ),
            }
        )
    by_symbol = pd.DataFrame(rows)

    valid = (
        np.isfinite(implied.to_numpy())
        & np.isfinite(premium.to_numpy())
        & np.isfinite(futures.to_numpy())
        & np.isfinite(mark.to_numpy())
        & np.isfinite(index.to_numpy())
    )
    implied_values = implied.to_numpy()[valid]
    premium_values = premium.to_numpy()[valid]
    difference_values = np.abs(implied_values - premium_values)
    mark_futures = np.abs(mark.to_numpy()[valid] / futures.to_numpy()[valid] - 1.0)
    index_futures = np.abs(index.to_numpy()[valid] / futures.to_numpy()[valid] - 1.0)
    aggregate = pd.DataFrame(
        [
            {
                "aligned_times": len(times),
                "aligned_symbols": len(symbols),
                "aligned_points": int(valid.sum()),
                "expected_points": len(times) * len(symbols),
                "aligned_point_coverage": float(valid.mean()),
                "implied_premium_correlation": float(
                    np.corrcoef(implied_values, premium_values)[0, 1]
                ),
                "median_symbol_implied_premium_correlation": float(
                    by_symbol["implied_premium_correlation"].median()
                ),
                "median_abs_implied_premium_difference": float(
                    np.median(difference_values)
                ),
                "p99_abs_implied_premium_difference": float(
                    np.quantile(difference_values, 0.99)
                ),
                "median_abs_mark_futures_difference": float(
                    np.median(mark_futures)
                ),
                "p99_abs_mark_futures_difference": float(
                    np.quantile(mark_futures, 0.99)
                ),
                "median_abs_index_futures_difference": float(
                    np.median(index_futures)
                ),
                "p99_abs_index_futures_difference": float(
                    np.quantile(index_futures, 0.99)
                ),
            }
        ]
    )
    return aggregate, by_symbol


def audit_v199(
    mark_root: Path = MARK_ROOT,
    index_root: Path = INDEX_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mark_manifest = pd.read_csv(mark_root / "manifest.csv")
    index_manifest = pd.read_csv(index_root / "manifest.csv")
    mark_download = pd.read_csv(mark_root / "last_download.csv")
    index_download = pd.read_csv(index_root / "last_download.csv")
    quality = pd.concat(
        [
            _symbol_quality(mark_root, "mark"),
            _symbol_quality(index_root, "index"),
        ],
        ignore_index=True,
    )
    mark = load_reference_ohlc_panels(mark_root)
    index = load_reference_ohlc_panels(index_root)
    premium = load_v190_premium_panel(premium_root)
    futures, _ = load_v184_exact_panels(metrics_root, kline_root)
    aggregate, by_symbol = summarize_reference_relationships(
        futures, mark["close"], index["close"], premium
    )
    stats = aggregate.iloc[0]
    mark_quality = quality[quality["reference"].eq("mark")]
    index_quality = quality[quality["reference"].eq("index")]
    expected_symbols = set(futures.columns)
    shared_symbols = (
        expected_symbols
        & set(mark["close"].columns)
        & set(index["close"].columns)
        & set(premium.columns)
    )
    checks: dict[str, tuple[bool, object]] = {
        "manifests_have_46_symbols": (
            len(mark_manifest) == 46 and len(index_manifest) == 46,
            f"{len(mark_manifest)}|{len(index_manifest)}",
        ),
        "downloads_have_46_symbols": (
            len(mark_download) == 46 and len(index_download) == 46,
            f"{len(mark_download)}|{len(index_download)}",
        ),
        "archives_all_checksummed": (
            mark_download["downloaded_archives"].eq(
                mark_download["verified_archives"]
            ).all()
            and index_download["downloaded_archives"].eq(
                index_download["verified_archives"]
            ).all(),
            f"{int(mark_download['verified_archives'].sum())}|"
            f"{int(index_download['verified_archives'].sum())}",
        ),
        "archives_690_each_no_missing": (
            int(mark_download["verified_archives"].sum()) == 690
            and int(index_download["verified_archives"].sum()) == 690
            and int(mark_download["missing_archives"].sum()) == 0
            and int(index_download["missing_archives"].sum()) == 0,
            f"{int(mark_download['missing_archives'].sum())}|"
            f"{int(index_download['missing_archives'].sum())}",
        ),
        "parquet_files_46_each": (
            len(mark_quality) == 46 and len(index_quality) == 46,
            f"{len(mark_quality)}|{len(index_quality)}",
        ),
        "rows_1497024_each": (
            int(mark_quality["rows"].sum()) == 1_497_024
            and int(index_quality["rows"].sum()) == 1_497_024,
            f"{int(mark_quality['rows'].sum())}|{int(index_quality['rows'].sum())}",
        ),
        "rows_32544_per_symbol": (
            quality["rows"].eq(32_544).all(),
            int(quality["rows"].min()),
        ),
        "dataset_labels_exact": (
            mark_quality["dataset"].eq("markPriceKlines").all()
            and index_quality["dataset"].eq("indexPriceKlines").all(),
            f"{mark_quality['dataset'].nunique()}|{index_quality['dataset'].nunique()}",
        ),
        "timestamps_unique_regular": (
            quality["duplicates"].eq(0).all()
            and quality["missing_grid_points"].eq(0).all(),
            int(quality["duplicates"].sum() + quality["missing_grid_points"].sum()),
        ),
        "ohlc_complete_finite_positive": (
            quality["ohlc_missing"].eq(0).all()
            and quality["ohlc_nonfinite"].eq(0).all()
            and quality["ohlc_nonpositive"].eq(0).all(),
            int(
                quality["ohlc_missing"].sum()
                + quality["ohlc_nonfinite"].sum()
                + quality["ohlc_nonpositive"].sum()
            ),
        ),
        "ohlc_ranges_valid": (
            quality["ohlc_range_bad"].eq(0).all(),
            int(quality["ohlc_range_bad"].sum()),
        ),
        "symbol_sets_exact_46": (
            len(shared_symbols) == 46 and shared_symbols == expected_symbols,
            len(shared_symbols),
        ),
        "research_exact_points_complete": (
            int(stats["aligned_times"]) == 32_459
            and int(stats["aligned_symbols"]) == 46
            and float(stats["aligned_point_coverage"]) == 1.0,
            int(stats["aligned_points"]),
        ),
        "implied_premium_relationship_strong": (
            float(stats["implied_premium_correlation"]) >= 0.90
            and float(stats["median_symbol_implied_premium_correlation"]) >= 0.80,
            float(stats["implied_premium_correlation"]),
        ),
        "implied_premium_not_numerically_identical": (
            float(stats["median_abs_implied_premium_difference"]) > 0
            and float(stats["median_abs_implied_premium_difference"]) <= 0.0002
            and float(stats["p99_abs_implied_premium_difference"]) <= 0.002,
            float(stats["median_abs_implied_premium_difference"]),
        ),
        "mark_tracks_futures_close": (
            float(stats["median_abs_mark_futures_difference"]) <= 0.0002
            and float(stats["p99_abs_mark_futures_difference"]) <= 0.002,
            float(stats["p99_abs_mark_futures_difference"]),
        ),
    }
    audit = pd.DataFrame(
        [
            {"check": name, "passed": bool(result), "value": value}
            for name, (result, value) in checks.items()
        ]
    )
    audit["verdict"] = (
        "audit_pass_reference_prices_ready_distinct_from_premium"
        if audit["passed"].all()
        else "audit_failure_reference_prices_require_investigation"
    )
    return audit, quality, aggregate, by_symbol


def _write_findings(
    audit: pd.DataFrame,
    aggregate: pd.DataFrame,
    by_symbol: pd.DataFrame,
    path: Path,
) -> None:
    stats = aggregate.iloc[0]
    weakest = by_symbol.nsmallest(5, "implied_premium_correlation")[
        [
            "symbol",
            "implied_premium_correlation",
            "median_abs_implied_premium_difference",
            "p99_abs_implied_premium_difference",
        ]
    ]
    failed = audit[~audit["passed"]]
    text = [
        "# v19.9 Binance Mark/Index Reference-Price Audit",
        "",
        f"Verdict: `{audit['verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        f"Exact research overlap: {int(stats['aligned_times']):,} bars x "
        f"{int(stats['aligned_symbols'])} symbols = "
        f"{int(stats['aligned_points']):,} complete points.",
        "",
        f"The close-to-close implied basis `mark/index - 1` has aggregate "
        f"correlation {float(stats['implied_premium_correlation']):.6f} with the "
        "official premium-index close. Its median absolute difference is "
        f"{float(stats['median_abs_implied_premium_difference']):.8f}, and its "
        f"99th percentile is {float(stats['p99_abs_implied_premium_difference']):.8f}.",
        "",
        "The two series are therefore related but not numerically interchangeable. "
        "Later feature work must measure incremental information relative to premium "
        "innovations instead of counting both as independent alpha.",
        "",
        "Weakest per-symbol implied/premium relationships (retained, not removed):",
        "",
        weakest.to_markdown(index=False, floatfmt=".8f"),
        "",
        "All timestamps are exact completed-bar times; no fill and no future return "
        "was used. No live, PaperLive, application, leverage, remote, or order scope "
        "changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v199_reference_price_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, quality, aggregate, by_symbol = audit_v199()
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "quality": root / "symbol_quality.csv",
        "aggregate": root / "reference_relationship_summary.csv",
        "by_symbol": root / "reference_relationship_by_symbol.csv",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    by_symbol.to_csv(outputs["by_symbol"], index=False)
    _write_findings(audit, aggregate, by_symbol, findings_path)
    return outputs


__all__ = [
    "INDEX_ROOT",
    "MARK_ROOT",
    "audit_v199",
    "load_reference_ohlc_panels",
    "summarize_reference_relationships",
    "write_v199_reference_price_audit",
]
