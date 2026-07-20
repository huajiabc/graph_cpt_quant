"""Causal hourly features from cached Binance USD-M book-depth archives."""
from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


DATA_ROOT = Path("data/external/binance_um_book_depth")
TARGET_BANDS = (1.0, 5.0)


@dataclass(frozen=True)
class BinanceBookDepthHourlyConfig:
    data_root: Path = DATA_ROOT
    workers: int = 8
    minimum_snapshots: int = 90


@dataclass(frozen=True)
class BinanceBookDepthHourlyResult:
    symbol: str
    archives: int
    invalid_archives: int
    rows: int
    primary_valid_hours: int
    first_decision_time: str | None
    last_decision_time: str | None
    output_path: str | None
    error: str | None = None


def _label(band: float) -> str:
    return str(band).replace(".", "p")


def parse_hourly_book_depth_archive(path: Path) -> pd.DataFrame:
    """Aggregate strict prior-hour snapshot intervals from one daily archive."""
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV, found {csv_names!r}")
        frame = pd.read_csv(
            archive.open(csv_names[0]),
            usecols=["timestamp", "percentage", "notional"],
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["percentage"] = pd.to_numeric(frame["percentage"], errors="coerce")
    frame["notional"] = pd.to_numeric(frame["notional"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "percentage", "notional"])
    outputs = []
    for band in TARGET_BANDS:
        local = frame[frame["percentage"].isin([-band, band])]
        pivot = local.pivot_table(
            index="timestamp",
            columns="percentage",
            values="notional",
            aggfunc="last",
            observed=True,
        )
        if -band not in pivot.columns or band not in pivot.columns:
            continue
        denominator = pivot[-band] + pivot[band]
        imbalance = ((pivot[-band] - pivot[band]) / denominator).where(denominator.gt(0))
        local_hourly = pd.DataFrame(
            {
                "imbalance": imbalance.replace([np.inf, -np.inf], np.nan),
                "total_notional": denominator.where(denominator.gt(0)),
            }
        ).dropna()
        local_hourly = local_hourly.reset_index()
        local_hourly["decision_time"] = (
            local_hourly["timestamp"].dt.floor("h") + pd.Timedelta(hours=1)
        )
        label = _label(band)
        summary = local_hourly.groupby("decision_time", observed=True).agg(
            **{
                f"notional_imbalance_{label}_median": ("imbalance", "median"),
                f"notional_imbalance_{label}_mean": ("imbalance", "mean"),
                f"notional_imbalance_{label}_std": (
                    "imbalance",
                    lambda series: series.std(ddof=0),
                ),
                f"notional_imbalance_{label}_valid_snapshots": ("imbalance", "count"),
                f"total_notional_{label}_median": ("total_notional", "median"),
                f"total_notional_{label}_mean": ("total_notional", "mean"),
                f"total_notional_{label}_std": (
                    "total_notional",
                    lambda series: series.std(ddof=0),
                ),
            }
        )
        outputs.append(summary.reset_index())
    if not outputs:
        return pd.DataFrame()
    hourly = outputs[0]
    for output in outputs[1:]:
        hourly = hourly.merge(output, on="decision_time", how="outer", validate="one_to_one")
    hourly["source_day"] = pd.Timestamp(frame["timestamp"].min()).floor("D")
    return hourly.sort_values("decision_time").reset_index(drop=True)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def build_hourly_book_depth_symbol(
    symbol: str,
    cfg: BinanceBookDepthHourlyConfig = BinanceBookDepthHourlyConfig(),
) -> BinanceBookDepthHourlyResult:
    raw_root = cfg.data_root / "raw" / symbol
    paths = sorted(raw_root.rglob("*.zip"))
    hashes = pd.read_parquet(
        cfg.data_root / "daily_features" / f"{symbol}.parquet",
        columns=["source_day", "archive_sha256"],
    )
    hashes["source_day"] = pd.to_datetime(hashes["source_day"], utc=True, errors="coerce")
    hash_lookup = hashes.set_index("source_day")["archive_sha256"].to_dict()
    frames = []
    invalid = 0
    for path in paths:
        try:
            frame = parse_hourly_book_depth_archive(path)
            if frame.empty:
                invalid += 1
                continue
            frame["symbol"] = symbol
            frame["archive_sha256"] = frame["source_day"].map(hash_lookup)
            frames.append(frame)
        except Exception:
            invalid += 1
    if not frames:
        return BinanceBookDepthHourlyResult(
            symbol, len(paths), invalid, 0, 0, None, None, None, "no_valid_archives"
        )
    hourly = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["symbol", "decision_time"], keep="last")
        .sort_values("decision_time")
        .reset_index(drop=True)
    )
    primary_count = "notional_imbalance_1p0_valid_snapshots"
    hourly["primary_coverage_pass"] = hourly[primary_count].ge(cfg.minimum_snapshots)
    output_path = cfg.data_root / "hourly_features" / f"{symbol}.parquet"
    _atomic_write_parquet(hourly, output_path)
    return BinanceBookDepthHourlyResult(
        symbol=symbol,
        archives=len(paths),
        invalid_archives=invalid,
        rows=len(hourly),
        primary_valid_hours=int(hourly["primary_coverage_pass"].sum()),
        first_decision_time=hourly["decision_time"].min().isoformat(),
        last_decision_time=hourly["decision_time"].max().isoformat(),
        output_path=str(output_path),
    )


def build_hourly_book_depth_panel(
    symbols: list[str],
    cfg: BinanceBookDepthHourlyConfig = BinanceBookDepthHourlyConfig(),
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    results = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(build_hourly_book_depth_symbol, symbol, cfg): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = BinanceBookDepthHourlyResult(
                    symbol,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(
                f"hourly bookDepth: {completed}/{len(normalized)} {symbol} "
                f"hours={result.rows} valid={result.primary_valid_hours} "
                f"invalid={result.invalid_archives} error={result.error or '-'}",
                flush=True,
            )
    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values("symbol")
    root = ensure_dir(cfg.data_root / "hourly_features")
    manifest.to_csv(root / "manifest.csv", index=False)
    coverage = {
        "symbols": len(normalized),
        "covered_symbols": int(manifest["rows"].gt(0).sum()),
        "rows": int(manifest["rows"].sum()),
        "primary_valid_hours": int(manifest["primary_valid_hours"].sum()),
        "minimum_snapshots": cfg.minimum_snapshots,
    }
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return manifest.reset_index(drop=True)
