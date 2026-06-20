from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from pressure_graph.clients.bybit import BybitClient
from pressure_graph.io import ensure_dir, write_parquet


BINANCE_MONTHLY_1M = "https://data.binance.vision/data/futures/um/monthly/klines"
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


def _month_start_end(stamp: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    period = pd.Period(stamp, freq="M")
    start = pd.Timestamp(period.start_time, tz="UTC")
    end = pd.Timestamp(period.end_time, tz="UTC").ceil("min")
    return start, end


def _selected_symbols(report_root: Path, top_n: int) -> list[str]:
    path = report_root / "binance_backfill_selected_symbols.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/backfill_binance_um_v40.py first.")
    frame = pd.read_csv(path)
    col = "binance_symbol" if "binance_symbol" in frame.columns else "symbol"
    return frame[col].astype(str).str.upper().head(int(top_n)).tolist()


def _binance_zip_url(symbol: str, stamp: str) -> str:
    return f"{BINANCE_MONTHLY_1M}/{symbol}/1m/{symbol}-1m-{stamp}.zip"


def _binance_zip_path(data_root: Path, symbol: str, stamp: str) -> Path:
    return data_root / "raw" / "binance" / "klines_1m_v4_zips" / symbol / f"{symbol}-1m-{stamp}.zip"


def _download_binance_zip(data_root: Path, symbol: str, stamp: str) -> Path | None:
    path = _binance_zip_path(data_root, symbol, stamp)
    if path.exists() and path.stat().st_size > 0:
        return path
    missing = path.with_suffix(".zip.missing")
    if missing.exists():
        return None
    ensure_dir(path.parent)
    request = Request(_binance_zip_url(symbol, stamp), headers={"User-Agent": "pressure-graph/0.1 v41-1m"})
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


def _read_binance_zip(path: Path, symbol: str) -> pd.DataFrame:
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
    raw_time = pd.to_numeric(frame["bar_open_time"], errors="coerce")
    unit = "us" if float(raw_time.dropna().median()) > 1e14 else "ms"
    out = pd.DataFrame(index=frame.index)
    out["exchange"] = "binance"
    out["symbol"] = symbol
    out["bar_open_time"] = pd.to_datetime(raw_time, unit=unit, utc=True, errors="coerce")
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
    out["bar_close_time"] = out["bar_open_time"] + pd.Timedelta(minutes=1)
    return out.dropna(subset=["bar_open_time", "close"])


def _append_month(existing: pd.DataFrame, month_frame: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [existing, month_frame] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("bar_open_time")
        .drop_duplicates(["symbol", "bar_open_time"], keep="last")
        .reset_index(drop=True)
    )


def _read_existing(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _fetch_binance_symbol(data_root: Path, symbol: str, months: list[str], refresh: bool) -> dict:
    out_path = data_root / "raw" / "binance" / "klines_1m_v4" / f"{symbol}.parquet"
    existing = pd.DataFrame() if refresh else _read_existing(out_path)
    frames = [existing] if not existing.empty else []
    missing: list[str] = []
    try:
        for stamp in months:
            zip_path = _download_binance_zip(data_root, symbol, stamp)
            if zip_path is None:
                missing.append(stamp)
                continue
            frame = _read_binance_zip(zip_path, symbol)
            if not frame.empty:
                frames.append(frame)
        out = _append_month(pd.DataFrame(), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        if out.empty:
            return {"exchange": "binance", "symbol": symbol, "status": "empty", "rows": 0, "missing_months": ",".join(missing)}
        write_parquet(out, out_path)
        return {
            "exchange": "binance",
            "symbol": symbol,
            "status": "ok",
            "rows": int(len(out)),
            "start": out["bar_close_time"].min(),
            "end": out["bar_close_time"].max(),
            "missing_months": ",".join(missing),
        }
    except Exception as exc:  # noqa: BLE001
        return {"exchange": "binance", "symbol": symbol, "status": "failed", "rows": 0, "error": repr(exc)}


def _fetch_bybit_symbol(data_root: Path, symbol: str, months: list[str], refresh: bool, base_url: str) -> dict:
    out_path = data_root / "raw" / "bybit" / "klines_1m_v4" / f"{symbol}.parquet"
    existing = pd.DataFrame() if refresh else _read_existing(out_path)
    frames = [existing] if not existing.empty else []
    client = BybitClient(base_url, "linear")
    try:
        for stamp in months:
            start, end = _month_start_end(stamp)
            frames.append(client.klines(symbol, start, end, "1m"))
        out = _append_month(pd.DataFrame(), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        if out.empty:
            return {"exchange": "bybit", "symbol": symbol, "status": "empty", "rows": 0}
        write_parquet(out, out_path)
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "status": "ok",
            "rows": int(len(out)),
            "start": out["bar_close_time"].min(),
            "end": out["bar_close_time"].max(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"exchange": "bybit", "symbol": symbol, "status": "failed", "rows": 0, "error": repr(exc)}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill focused 1m cross-exchange samples for v4.1 diagnostics.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/v4_1_cross_exchange_lead_lag_diagnostics"))
    parser.add_argument("--v40-report-root", type=Path, default=Path("reports/v4_0_cross_exchange_lead_lag"))
    parser.add_argument("--months", type=str, default="2025-08,2025-10,2026-05")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exchange", choices=["both", "binance", "bybit"], default="both")
    parser.add_argument("--bybit-base-url", type=str, default="https://api.bybit-tr.com")
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()

    months = [month.strip() for month in args.months.split(",") if month.strip()]
    symbols = _selected_symbols(args.v40_report_root, args.top_n)
    ensure_dir(args.report_root)
    ensure_dir(args.data_root / "raw" / "binance" / "klines_1m_v4")
    ensure_dir(args.data_root / "raw" / "bybit" / "klines_1m_v4")
    jobs = []
    if args.exchange in {"both", "binance"}:
        jobs.extend(("binance", symbol) for symbol in symbols)
    if args.exchange in {"both", "bybit"}:
        jobs.extend(("bybit", symbol) for symbol in symbols)

    print(f"[v4.1 1m] jobs={len(jobs)} symbols={len(symbols)} months={','.join(months)}", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {}
        for exchange, symbol in jobs:
            if exchange == "binance":
                future = executor.submit(_fetch_binance_symbol, args.data_root, symbol, months, args.refresh_existing)
            else:
                future = executor.submit(
                    _fetch_bybit_symbol,
                    args.data_root,
                    symbol,
                    months,
                    args.refresh_existing,
                    args.bybit_base_url,
                )
            futures[future] = (exchange, symbol)
        for future in as_completed(futures):
            exchange, symbol = futures[future]
            result = future.result()
            rows.append(result)
            print(f"[v4.1 1m] {exchange} {symbol} {result.get('status')} rows={result.get('rows', 0)}", flush=True)
    manifest = pd.DataFrame(rows)
    manifest_path = args.report_root / "one_min_backfill_manifest.csv"
    if manifest_path.exists():
        try:
            existing = pd.read_csv(manifest_path)
            manifest = pd.concat([existing, manifest], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    manifest = (
        manifest.sort_values(["exchange", "symbol"])
        .drop_duplicates(["exchange", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"[v4.1 1m] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
