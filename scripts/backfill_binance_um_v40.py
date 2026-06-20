from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from pressure_graph.clients.binance import BinanceClient
from pressure_graph.io import ensure_dir, write_parquet


BINANCE_PUBLIC_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
BINANCE_PUBLIC_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/klines"
KLINE_COLUMNS = [
    "bar_open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_raw",
    "turnover",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _utc(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _bybit_rank(data_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((data_root / "raw" / "bybit" / "klines").glob("*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["bar_open_time", "bar_close_time", "turnover"])
        except Exception:  # noqa: BLE001 - one corrupted local symbol must not kill rank construction
            continue
        if frame.empty:
            continue
        rows.append(
            {
                "symbol": path.stem.upper(),
                "target_rows": int(len(frame)),
                "target_turnover_sum": float(pd.to_numeric(frame["turnover"], errors="coerce").sum()),
                "target_start": pd.to_datetime(frame["bar_open_time"], utc=True, errors="coerce").min(),
                "target_end": pd.to_datetime(frame["bar_close_time"], utc=True, errors="coerce").max(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target_turnover_sum", "symbol"], ascending=[False, True])


def _existing_status(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {"existing_rows": 0, "existing_start": pd.NaT, "existing_end": pd.NaT, "has_taker_buy": False}
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return {"existing_rows": 0, "existing_start": pd.NaT, "existing_end": pd.NaT, "has_taker_buy": False}
    times = pd.to_datetime(frame.get("bar_close_time"), utc=True, errors="coerce")
    return {
        "existing_rows": int(len(frame)),
        "existing_start": times.min(),
        "existing_end": times.max(),
        "has_taker_buy": "taker_buy_base" in frame.columns and pd.to_numeric(
            frame.get("taker_buy_base"), errors="coerce"
        ).notna().any(),
        "covers_requested_window": bool(times.min() <= start and times.max() >= end),
    }


def _select_symbols(rank: pd.DataFrame, binance_symbols: set[str], top_n: int) -> pd.DataFrame:
    local = rank.copy()
    local["binance_symbol"] = local["symbol"]
    local["binance_exact_available"] = local["binance_symbol"].isin(binance_symbols)
    selected = local[local["binance_exact_available"]].head(int(top_n)).copy()
    selected["selected_rank"] = range(1, len(selected) + 1)
    return selected


def _month_stamps(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start.tz_convert(None).to_period("M"), end=end.tz_convert(None).to_period("M"), freq="M")
    return [str(month) for month in months]


def _monthly_url(symbol: str, stamp: str) -> str:
    return f"{BINANCE_PUBLIC_BASE}/{symbol}/15m/{symbol}-15m-{stamp}.zip"


def _daily_url(symbol: str, stamp: str) -> str:
    return f"{BINANCE_PUBLIC_DAILY_BASE}/{symbol}/15m/{symbol}-15m-{stamp}.zip"


def _monthly_zip_path(data_root: Path, symbol: str, stamp: str) -> Path:
    return data_root / "raw" / "binance" / "klines_monthly_zips" / symbol / f"{symbol}-15m-{stamp}.zip"


def _daily_zip_path(data_root: Path, symbol: str, stamp: str) -> Path:
    return data_root / "raw" / "binance" / "klines_daily_zips" / symbol / f"{symbol}-15m-{stamp}.zip"


def _download_monthly_zip(symbol: str, stamp: str, data_root: Path) -> Path | None:
    path = _monthly_zip_path(data_root, symbol, stamp)
    if path.exists() and path.stat().st_size > 0:
        return path
    missing = path.with_suffix(".zip.missing")
    if missing.exists():
        return None
    ensure_dir(path.parent)
    request = Request(_monthly_url(symbol, stamp), headers={"User-Agent": "pressure-graph/0.1 v40-backfill"})
    try:
        with urlopen(request, timeout=90) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            missing.write_text("404", encoding="utf-8")
            return None
        raise
    tmp = path.with_suffix(".zip.part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def _download_daily_zip(symbol: str, stamp: str, data_root: Path) -> Path | None:
    path = _daily_zip_path(data_root, symbol, stamp)
    if path.exists() and path.stat().st_size > 0:
        return path
    missing = path.with_suffix(".zip.missing")
    if missing.exists():
        return None
    ensure_dir(path.parent)
    request = Request(_daily_url(symbol, stamp), headers={"User-Agent": "pressure-graph/0.1 v40-backfill"})
    try:
        with urlopen(request, timeout=90) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            missing.write_text("404", encoding="utf-8")
            return None
        raise
    tmp = path.with_suffix(".zip.part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def _read_monthly_zip(path: Path, symbol: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None, low_memory=False)
    if frame.empty or len(frame.columns) < len(KLINE_COLUMNS):
        return pd.DataFrame()
    frame = frame.iloc[:, : len(KLINE_COLUMNS)].copy()
    frame.columns = KLINE_COLUMNS
    open_raw = pd.to_numeric(frame["bar_open_time"], errors="coerce")
    median = float(open_raw.dropna().median()) if open_raw.notna().any() else 0.0
    unit = "us" if median > 1e14 else "ms"
    out = pd.DataFrame(index=frame.index)
    out["exchange"] = "binance"
    out["symbol"] = symbol
    out["bar_open_time"] = pd.to_datetime(open_raw, unit=unit, utc=True, errors="coerce")
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["bar_close_time"] = out["bar_open_time"] + pd.Timedelta(minutes=15)
    return out.dropna(subset=["bar_open_time", "close"])


def _day_stamps(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    days = pd.date_range(
        start=start.tz_convert(None).normalize(),
        end=end.tz_convert(None).normalize(),
        freq="D",
    )
    return [day.strftime("%Y-%m-%d") for day in days]


def _download_symbol_archive(symbol: str, start: pd.Timestamp, end: pd.Timestamp, data_root: Path) -> dict:
    frames: list[pd.DataFrame] = []
    missing_months: list[str] = []
    try:
        for stamp in _month_stamps(start, end):
            zip_path = _download_monthly_zip(symbol, stamp, data_root)
            if zip_path is None:
                missing_months.append(stamp)
                continue
            frame = _read_monthly_zip(zip_path, symbol)
            if not frame.empty:
                frames.append(frame)
        if missing_months:
            for stamp in _day_stamps(start, end):
                if stamp[:7] not in set(missing_months):
                    continue
                zip_path = _download_daily_zip(symbol, stamp, data_root)
                if zip_path is None:
                    continue
                frame = _read_monthly_zip(zip_path, symbol)
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return {
                "symbol": symbol,
                "status": "empty_archive",
                "downloaded_rows": 0,
                "missing_months": ",".join(missing_months),
            }
        out = pd.concat(frames, ignore_index=True).sort_values("bar_open_time")
        out = out[(out["bar_close_time"] >= start) & (out["bar_open_time"] <= end)].copy()
        out = out.drop_duplicates(["symbol", "bar_open_time"])
        if out.empty:
            return {
                "symbol": symbol,
                "status": "empty_after_window_filter",
                "downloaded_rows": 0,
                "missing_months": ",".join(missing_months),
            }
        write_parquet(out, data_root / "raw" / "binance" / "klines" / f"{symbol}.parquet")
        return {
            "symbol": symbol,
            "status": "ok",
            "downloaded_rows": int(len(out)),
            "downloaded_start": out["bar_close_time"].min(),
            "downloaded_end": out["bar_close_time"].max(),
            "has_taker_buy": True,
            "missing_months": ",".join(missing_months),
        }
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "status": "failed", "error": repr(exc), "downloaded_rows": 0}


def _download_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp, data_root: Path) -> dict:
    client = BinanceClient()
    path = data_root / "raw" / "binance" / "klines" / f"{symbol}.parquet"
    try:
        frame = client.klines(symbol, start, end, "15m")
        if frame.empty:
            return {"symbol": symbol, "status": "empty", "downloaded_rows": 0}
        write_parquet(frame, path)
        return {
            "symbol": symbol,
            "status": "ok",
            "downloaded_rows": int(len(frame)),
            "downloaded_start": frame["bar_close_time"].min(),
            "downloaded_end": frame["bar_close_time"].max(),
            "has_taker_buy": "taker_buy_base" in frame.columns,
        }
    except Exception as exc:  # noqa: BLE001 - keep resumable manifest
        return {"symbol": symbol, "status": "failed", "error": repr(exc), "downloaded_rows": 0}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Binance UM 15m klines for v4.0 cross-exchange research.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/v4_0_cross_exchange_lead_lag"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--source", choices=["archive", "api"], default="archive")
    args = parser.parse_args()

    rank = _bybit_rank(args.data_root)
    if rank.empty:
        raise FileNotFoundError("No Bybit kline files found under data/raw/bybit/klines.")
    out_dir = ensure_dir(args.report_root)
    selection_cache = out_dir / "binance_backfill_selected_symbols.csv"
    if selection_cache.exists():
        selected = pd.read_csv(selection_cache)
        selected = selected.head(int(args.top_n)).copy()
    else:
        try:
            client = BinanceClient()
            try:
                instruments = client.instruments()
            finally:
                client.close()
            binance_symbols = set(instruments.get("symbol", pd.Series(dtype=str)).astype(str).str.upper())
            selected = _select_symbols(rank, binance_symbols, args.top_n)
        except Exception:
            selected = rank.head(int(args.top_n)).copy()
            selected["binance_symbol"] = selected["symbol"]
            selected["binance_exact_available"] = True
            selected["selected_rank"] = range(1, len(selected) + 1)
    if selected.empty:
        raise RuntimeError("No exact-common Bybit/Binance symbols found.")

    start = _utc(args.start) or pd.to_datetime(selected["target_start"], utc=True, errors="coerce").min()
    end = _utc(args.end) or pd.to_datetime(selected["target_end"], utc=True, errors="coerce").max()
    ensure_dir(args.data_root / "raw" / "binance" / "klines")

    manifest_rows: list[dict] = []
    to_fetch: list[str] = []
    for row in selected.itertuples(index=False):
        symbol = str(row.binance_symbol)
        path = args.data_root / "raw" / "binance" / "klines" / f"{symbol}.parquet"
        status = _existing_status(path, start, end)
        should_fetch = args.refresh_existing or not (
            status.get("covers_requested_window", False) and status.get("has_taker_buy", False)
        )
        manifest_rows.append(
            {
                "symbol": symbol,
                "selected_rank": int(row.selected_rank),
                "target_turnover_sum": float(row.target_turnover_sum),
                "target_start": row.target_start,
                "target_end": row.target_end,
                "requested_start": start,
                "requested_end": end,
                "fetch_planned": should_fetch,
                **status,
            }
        )
        if should_fetch:
            to_fetch.append(symbol)

    print(f"[v4.0 backfill] selected={len(selected)} fetch={len(to_fetch)} window={start} -> {end}", flush=True)
    results: dict[str, dict] = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _download_symbol_archive if args.source == "archive" else _download_symbol,
                    symbol,
                    start,
                    end,
                    args.data_root,
                ): symbol
                for symbol in to_fetch
            }
            for future in as_completed(futures):
                symbol = futures[future]
                result = future.result()
                results[symbol] = result
                print(
                    f"[v4.0 backfill] {symbol} {result.get('status')} rows={result.get('downloaded_rows', 0)}",
                    flush=True,
                )

    manifest = pd.DataFrame(manifest_rows)
    for symbol, result in results.items():
        for key, value in result.items():
            manifest.loc[manifest["symbol"].eq(symbol), key] = value
    manifest["final_status"] = manifest.get("status", pd.Series(index=manifest.index, dtype=object)).fillna(
        manifest["fetch_planned"].map({True: "not_fetched", False: "cached"})
    )
    manifest.to_csv(out_dir / "binance_backfill_manifest.csv", index=False)
    selected.to_csv(out_dir / "binance_backfill_selected_symbols.csv", index=False)
    print(f"[v4.0 backfill] manifest={out_dir / 'binance_backfill_manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
