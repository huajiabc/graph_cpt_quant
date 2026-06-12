"""Memory-bounded v0.3 feature build for cgroup-limited boxes.

The monolithic ``build_feature_table`` + ``add_future_labels`` path holds the
full universe in memory (≈12 GB table with multi-copy transform peaks for a
369-symbol year), which gets OOM-killed inside an 18 GB memcg. This module
produces the same parquet by building per-symbol batches in worker processes
and streaming the parts into one file.

Parity notes:
- All per-symbol features and labels reuse the exact monolithic functions.
- The monolithic BTC volatility percentile is computed over the symbol-major
  concatenated frame, so every non-first symbol's segment starts with the
  rolling window still holding the previous segment's tail (an order artifact).
  This is reproduced exactly for gap-free symbols via a "first segment" and a
  "subsequent segment" percentile timeline; symbols with intra-year bar gaps
  may deviate inside their first ``BTC_VOL_WINDOW`` bars only — an explicit
  approximation of an artifact, not of a semantic feature.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressure_graph.config import ExperimentConfig
from pressure_graph.features.build import (
    BTC_VOL_MIN_PERIODS,
    BTC_VOL_WINDOW,
    add_core_features,
    align_funding,
    align_open_interest,
    btc_market_state_labels,
    btc_vol_regime_labels,
    ensure_bar_times,
    rolling_percentile_current_vs_prior,
)
from pressure_graph.io import read_parquet, write_parquet
from pressure_graph.labels.future import add_future_labels
from pressure_graph.universe.selection import apply_universe_flags

LIGHT_KLINE_COLUMNS = ["exchange", "symbol", "bar_open_time", "close", "volume", "turnover"]
FLAG_COLUMNS = ["universe_static_current_top30", "universe_dynamic_monthly_top30"]
BTC_BASE_COLUMNS = ["exchange", "bar_open_time", "btc_ret_1h", "btc_ret_4h", "btc_volatility_4h"]


@dataclass(frozen=True)
class StreamingBuildConfig:
    batch_size: int = 12
    workers: int = 5


def _symbol_files(kline_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(kline_dir.glob("*.parquet"))}


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = [field.name for field in pq.read_schema(path)]
    selected = [col for col in columns if col in available]
    return pq.read_table(path, columns=selected).to_pandas()


def _light_klines(kline_files: dict[str, Path]) -> pd.DataFrame:
    frames = [_read_columns(path, LIGHT_KLINE_COLUMNS) for path in kline_files.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _universe_flags(
    kline_files: dict[str, Path],
    instruments: pd.DataFrame,
    tickers: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    light = _light_klines(kline_files)
    if light.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "bar_open_time", *FLAG_COLUMNS])
    flagged = apply_universe_flags(light, instruments, tickers, config)
    flags = flagged[["exchange", "symbol", "bar_open_time", *FLAG_COLUMNS]].copy()
    flags["bar_open_time"] = pd.to_datetime(flags["bar_open_time"], utc=True)
    return flags


def _btc_state_tables(
    btc_kline_path: Path | None,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BTC context tables for the globally-first segment and for all others."""
    if btc_kline_path is None or not btc_kline_path.exists():
        empty = pd.DataFrame(columns=BTC_BASE_COLUMNS)
        return empty, empty
    btc = add_core_features(read_parquet(btc_kline_path), config)
    base = btc[["exchange", "bar_open_time", "ret_1h", "ret_4h", "volatility_4h"]].rename(
        columns={
            "ret_1h": "btc_ret_1h",
            "ret_4h": "btc_ret_4h",
            "volatility_4h": "btc_volatility_4h",
        }
    )
    base = base.sort_values("bar_open_time").reset_index(drop=True)
    vol = base["btc_volatility_4h"]
    pct_first = rolling_percentile_current_vs_prior(vol, BTC_VOL_WINDOW, BTC_VOL_MIN_PERIODS)
    carried = pd.concat([vol.tail(BTC_VOL_WINDOW), vol], ignore_index=True)
    pct_rest = rolling_percentile_current_vs_prior(
        carried, BTC_VOL_WINDOW, BTC_VOL_MIN_PERIODS
    ).iloc[len(carried) - len(vol):].reset_index(drop=True)
    first = base.copy()
    first["btc_volatility_percentile"] = pct_first.to_numpy()
    rest = base.copy()
    rest["btc_volatility_percentile"] = pct_rest.to_numpy()
    return first, rest


def _attach_btc_state(bars: pd.DataFrame, btc_table: pd.DataFrame) -> pd.DataFrame:
    out = bars.merge(
        btc_table[[*BTC_BASE_COLUMNS, "btc_volatility_percentile"]],
        on=["exchange", "bar_open_time"],
        how="left",
    )
    out["btc_market_state"] = btc_market_state_labels(out["btc_ret_4h"])
    out["btc_vol_regime"] = btc_vol_regime_labels(out["btc_volatility_percentile"])
    return out


def _build_symbol_batch(
    symbols: list[str],
    first_symbol: str,
    data_root: Path,
    exchange: str,
    config: ExperimentConfig,
    flags: pd.DataFrame,
    btc_first: pd.DataFrame,
    btc_rest: pd.DataFrame,
    part_path: Path,
) -> str:
    raw_root = data_root / "raw" / exchange
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        kline_path = raw_root / "klines" / f"{symbol}.parquet"
        if not kline_path.exists():
            continue
        bars = read_parquet(kline_path)
        if bars.empty:
            continue
        bars = ensure_bar_times(bars, config.experiment.base_interval)
        bars["bar_open_time"] = pd.to_datetime(bars["bar_open_time"], utc=True)
        symbol_flags = flags[flags["symbol"].eq(symbol)]
        bars = bars.merge(
            symbol_flags[["exchange", "symbol", "bar_open_time", *FLAG_COLUMNS]],
            on=["exchange", "symbol", "bar_open_time"],
            how="left",
        )
        for col in FLAG_COLUMNS:
            bars[col] = bars[col].fillna(False).astype(bool)
        funding = _read_optional(raw_root / "funding" / f"{symbol}.parquet")
        oi = _read_optional(raw_root / "open_interest" / f"{symbol}.parquet")
        instruments = _read_optional(raw_root / "instruments.parquet")
        bars = align_funding(bars, funding, instruments, config)
        bars = align_open_interest(bars, oi)
        bars = add_core_features(bars, config)
        btc_table = btc_first if symbol == first_symbol else btc_rest
        bars = _attach_btc_state(bars, btc_table)
        bars = add_future_labels(bars)
        frames.append(bars)
    if not frames:
        return ""
    part = pd.concat(frames, ignore_index=True)
    part = part.sort_values(["exchange", "symbol", "bar_open_time"]).reset_index(drop=True)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(part, part_path)
    return str(part_path)


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _merge_parts(part_paths: list[Path], output: Path) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    schema: pa.Schema | None = None
    try:
        for path in part_paths:
            table = pq.read_table(path)
            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(output, schema)
            elif table.schema.names != schema.names:
                raise ValueError(
                    f"part schema mismatch: {path} has {table.schema.names[:5]}..."
                )
            else:
                table = table.cast(schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def build_v03_features_streaming(
    config: ExperimentConfig,
    streaming: StreamingBuildConfig = StreamingBuildConfig(),
    exchange: str = "bybit",
) -> Path:
    data_root = config.paths.data_root
    raw_root = data_root / "raw" / exchange
    kline_files = _symbol_files(raw_root / "klines")
    if not kline_files:
        raise FileNotFoundError(f"No {exchange} raw klines found. Run collect first.")
    instruments = _read_optional(raw_root / "instruments.parquet")
    tickers = _read_optional(raw_root / "tickers.parquet")

    flags = _universe_flags(kline_files, instruments, tickers, config)
    symbols = sorted(kline_files)
    first_symbol = symbols[0]
    btc_first, btc_rest = _btc_state_tables(kline_files.get("BTCUSDT"), config)

    output = data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.parent / "_streaming_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale in parts_dir.glob("part_*.parquet"):
        stale.unlink()

    batches = [
        symbols[idx : idx + streaming.batch_size]
        for idx in range(0, len(symbols), streaming.batch_size)
    ]
    part_paths: list[Path] = []
    jobs = []
    for batch_idx, batch in enumerate(batches):
        part_path = parts_dir / f"part_{batch_idx:04d}.parquet"
        batch_flags = flags[flags["symbol"].isin(batch)]
        jobs.append(
            (
                batch,
                first_symbol,
                data_root,
                exchange,
                config,
                batch_flags,
                btc_first,
                btc_rest,
                part_path,
            )
        )

    if streaming.workers <= 1:
        results = [_build_symbol_batch(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=streaming.workers) as pool:
            futures = [pool.submit(_build_symbol_batch, *job) for job in jobs]
            results = []
            for done, future in enumerate(futures, start=1):
                results.append(future.result())
                print(f"[streaming-build] batch {done}/{len(futures)} done", flush=True)
    part_paths = [Path(result) for result in results if result]

    rows = _merge_parts(part_paths, output)
    for path in part_paths:
        path.unlink()
    parts_dir.rmdir()
    print(f"[streaming-build] wrote {rows:,} rows -> {output}", flush=True)
    return output


__all__ = ["StreamingBuildConfig", "build_v03_features_streaming"]
