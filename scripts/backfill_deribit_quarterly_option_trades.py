from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pressure_graph.clients.deribit import DeribitClient
from pressure_graph.deribit_option_trade_history import (
    DeribitOptionSurfaceConfig,
    build_daily_option_surface,
    candidate_strikes,
    monthly_expiries,
    normalize_active_trade_bars,
    option_instrument,
    quarterly_expiries,
)
from pressure_graph.io import ensure_dir


DEFAULT_PERPETUAL = Path(
    "data/external/orthogonal_volatility/deribit_perpetual_1h/BTC-PERPETUAL.parquet"
)
DEFAULT_OUTPUT = Path("data/external/deribit_quarterly_option_trades")
SOURCE_ENDPOINT = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
_THREAD_LOCAL = threading.local()


def _thread_client() -> DeribitClient:
    client = getattr(_THREAD_LOCAL, "deribit_client", None)
    if client is None:
        client = DeribitClient()
        _THREAD_LOCAL.deribit_client = client
    return client


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _underlying(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["bar_open_time", "open", "close"]
    )
    frame["bar_open_time"] = pd.to_datetime(
        frame["bar_open_time"], utc=True, errors="coerce"
    )
    return (
        frame.dropna(subset=["bar_open_time", "open", "close"])
        .drop_duplicates("bar_open_time", keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )


def _query_one(
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    underlying: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    try:
        raw = _thread_client().chart_data(
            instrument, start, end, resolution_minutes=60
        )
    except Exception as exc:
        return (
            {
                "instrument_name": instrument,
                "query_start": start,
                "query_end": end,
                "status": "invalid_or_error",
                "chart_rows": 0,
                "active_rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            },
            pd.DataFrame(),
        )
    if raw.empty:
        return (
            {
                "instrument_name": instrument,
                "query_start": start,
                "query_end": end,
                "status": "no_data",
                "chart_rows": 0,
                "active_rows": 0,
                "error": None,
            },
            pd.DataFrame(),
        )
    active = normalize_active_trade_bars(raw, underlying, instrument)
    return (
        {
            "instrument_name": instrument,
            "query_start": start,
            "query_end": end,
            "status": "active" if not active.empty else "forward_fill_only",
            "chart_rows": int(len(raw)),
            "active_rows": int(len(active)),
            "error": None,
        },
        active,
    )


def backfill(
    perpetual_path: Path = DEFAULT_PERPETUAL,
    output_root: Path = DEFAULT_OUTPUT,
    start: pd.Timestamp = pd.Timestamp("2021-03-01", tz="UTC"),
    end: pd.Timestamp = pd.Timestamp("2026-07-01", tz="UTC"),
    workers: int = 4,
    calendar: str = "quarterly",
) -> dict[str, Path]:
    underlying = _underlying(perpetual_path)
    minimum_time = pd.Timestamp(underlying["bar_open_time"].min())
    maximum_time = pd.Timestamp(underlying["bar_open_time"].max())
    tasks: list[tuple[str, pd.Timestamp, pd.Timestamp, float]] = []
    expiry_builder = monthly_expiries if calendar == "monthly" else quarterly_expiries
    for expiry in expiry_builder(start, end):
        query_start = max(expiry - pd.Timedelta(days=45), minimum_time)
        query_end = min(expiry, maximum_time)
        if query_start >= query_end:
            continue
        reference = underlying[underlying["bar_open_time"].le(query_start)]
        if reference.empty:
            reference_price = float(underlying.iloc[0]["close"])
        else:
            reference_price = float(reference.iloc[-1]["close"])
        for strike in candidate_strikes(reference_price):
            for option_type in ("C", "P"):
                tasks.append(
                    (
                        option_instrument(expiry, strike, option_type),
                        query_start,
                        query_end,
                        reference_price,
                    )
                )

    manifests: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_query_one, instrument, query_start, query_end, underlying): (
                instrument,
                reference_price,
            )
            for instrument, query_start, query_end, reference_price in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            instrument, reference_price = futures[future]
            manifest, active = future.result()
            manifest["reference_price"] = reference_price
            manifests.append(manifest)
            if not active.empty:
                frames.append(active)
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"deribit-options: {completed}/{len(tasks)} "
                    f"active_contracts={sum(row['status'] == 'active' for row in manifests)}",
                    flush=True,
                )

    manifest = pd.DataFrame(manifests).sort_values("instrument_name").reset_index(
        drop=True
    )
    active = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["instrument_name", "bar_open_time"], keep="last")
        .sort_values(["bar_open_time", "instrument_name"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    surface = build_daily_option_surface(active)
    outputs = {
        "manifest": output_root / "query_manifest.csv",
        "active_bars": output_root / "active_hourly_trade_bars.parquet",
        "surface": output_root / "daily_trade_surface.parquet",
        "coverage": output_root / "coverage.json",
    }
    _atomic_csv(manifest, outputs["manifest"])
    _atomic_parquet(active, outputs["active_bars"])
    _atomic_parquet(surface, outputs["surface"])
    coverage = {
        "source_endpoint": SOURCE_ENDPOINT,
        "expiry_calendar": calendar,
        "history_semantics": "trade_ohlcv_only; zero-volume forward fills excluded",
        "execution_boundary": "signal research only; no historical bid/ask reconstruction",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "queried_contracts": int(len(manifest)),
        "active_contracts": int(manifest["status"].eq("active").sum()),
        "forward_fill_only_contracts": int(
            manifest["status"].eq("forward_fill_only").sum()
        ),
        "no_data_contracts": int(manifest["status"].eq("no_data").sum()),
        "invalid_or_error_contracts": int(
            manifest["status"].eq("invalid_or_error").sum()
        ),
        "active_hourly_rows": int(len(active)),
        "surface_rows": int(len(surface)),
        "quality_surface_rows": int(surface.get("quality_pass", pd.Series(dtype=bool)).sum()),
        "surface_first_time": (
            surface["feature_time"].min().isoformat() if not surface.empty else None
        ),
        "surface_last_time": (
            surface["feature_time"].max().isoformat() if not surface.empty else None
        ),
        "surface_config": asdict(DeribitOptionSurfaceConfig()),
        "sha256": {
            key: _sha256(path)
            for key, path in outputs.items()
            if key != "coverage"
        },
    }
    ensure_dir(output_root)
    outputs["coverage"].write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    print(json.dumps(coverage, indent=2), flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill archived Deribit BTC quarterly option trade bars."
    )
    parser.add_argument("--perpetual-path", type=Path, default=DEFAULT_PERPETUAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2021-03-01T00:00:00Z")
    parser.add_argument("--end", default="2026-07-01T00:00:00Z")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--calendar", choices=("quarterly", "monthly"), default="quarterly"
    )
    args = parser.parse_args()
    backfill(
        perpetual_path=args.perpetual_path,
        output_root=args.output_root,
        start=pd.Timestamp(args.start).tz_convert("UTC"),
        end=pd.Timestamp(args.end).tz_convert("UTC"),
        workers=args.workers,
        calendar=args.calendar,
    )


if __name__ == "__main__":
    main()
