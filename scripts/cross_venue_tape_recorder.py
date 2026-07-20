from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.clients.binance import BinanceClient  # noqa: E402
from pressure_graph.cross_venue_tape import (  # noqa: E402
    MinuteBarAccumulator,
    parse_binance_message,
    parse_bybit_message,
    select_common_symbols,
    write_bar_fragment,
    write_coverage_report,
)
from pressure_graph.io import ensure_dir  # noqa: E402


BINANCE_WS = "wss://fstream.binance.com/market/stream?streams={streams}"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
RECEIVE_TIMEOUT_SECONDS = 90


def _session_event(path: Path, event: str, **fields: object) -> None:
    ensure_dir(path.parent)
    row = {"time": __import__("pandas").Timestamp.now(tz="UTC").isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


async def _bybit_ping(websocket: object) -> None:
    while True:
        await asyncio.sleep(20)
        await websocket.send(json.dumps({"op": "ping"}))


async def _venue_loop(
    venue: str,
    symbols: list[str],
    accumulator: MinuteBarAccumulator,
    session_log: Path,
) -> None:
    from websockets import connect

    parser = parse_binance_message if venue == "binance" else parse_bybit_message
    backoff = 1
    while True:
        session_id = f"{venue}-{uuid.uuid4().hex[:12]}"
        ping_task: asyncio.Task[None] | None = None
        try:
            if venue == "binance":
                streams = "/".join(f"{symbol.lower()}@aggTrade" for symbol in symbols)
                url = BINANCE_WS.format(streams=streams)
            else:
                url = BYBIT_WS
            _session_event(session_log, "connecting", venue=venue, session_id=session_id)
            async with connect(url, ping_interval=20, ping_timeout=20, max_queue=100_000) as websocket:
                if venue == "bybit":
                    topics = [f"publicTrade.{symbol}" for symbol in symbols]
                    for idx in range(0, len(topics), 10):
                        await websocket.send(json.dumps({"op": "subscribe", "args": topics[idx : idx + 10]}))
                    ping_task = asyncio.create_task(_bybit_ping(websocket))
                _session_event(session_log, "connected", venue=venue, session_id=session_id)
                backoff = 1
                first_trade_logged = False
                while True:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=RECEIVE_TIMEOUT_SECONDS,
                    )
                    trades = parser(message)
                    for trade in trades:
                        accumulator.add_trade(trade, session_id)
                    if trades and not first_trade_logged:
                        _session_event(
                            session_log,
                            "first_trade",
                            venue=venue,
                            session_id=session_id,
                            symbol=trades[0]["symbol"],
                            trade_time=str(trades[0]["timestamp"]),
                        )
                        first_trade_logged = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorder must reconnect after any socket failure
            _session_event(
                session_log,
                "disconnected",
                venue=venue,
                session_id=session_id,
                error=f"{type(exc).__name__}:{exc}",
                retry_seconds=backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
        finally:
            if ping_task is not None:
                ping_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await ping_task


async def _flush_loop(
    accumulator: MinuteBarAccumulator,
    root: Path,
    report_root: Path,
    flush_seconds: int,
    coverage_hours: int,
    coverage_refresh_seconds: int,
    session_log: Path,
) -> None:
    import pandas as pd

    sequence = 0
    last_coverage_time: pd.Timestamp | None = None
    while True:
        await asyncio.sleep(flush_seconds)
        cutoff = pd.Timestamp.now(tz="UTC").floor("1min")
        bars = accumulator.drain(cutoff, bar_complete=True)
        if bars.empty:
            continue
        sequence += 1
        fragment_id = f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{sequence:08d}"
        fragment_path = write_bar_fragment(bars, root, fragment_id)
        _session_event(
            session_log,
            "bar_flush",
            cutoff=cutoff.isoformat(),
            rows=len(bars),
            fragment=str(fragment_path) if fragment_path is not None else None,
        )
        if (
            last_coverage_time is None
            or cutoff - last_coverage_time
            >= pd.Timedelta(seconds=coverage_refresh_seconds)
        ):
            await asyncio.to_thread(
                write_coverage_report,
                root,
                report_root,
                coverage_hours,
                cutoff,
            )
            last_coverage_time = cutoff


async def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    report_root = Path(args.report_root)
    session_log = report_root / "session_events.jsonl"
    if args.symbols:
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    else:
        instruments = BinanceClient().instruments()
        common = set(instruments.loc[instruments["symbol"].astype(str).str.endswith("USDT"), "symbol"].astype(str))
        symbols = select_common_symbols(Path(args.live_feature_path), common, args.max_symbols)
    if not symbols:
        raise RuntimeError("no common Binance/Bybit symbols selected")
    _session_event(session_log, "recorder_start", symbols=symbols, max_symbols=args.max_symbols)
    accumulator = MinuteBarAccumulator()
    tasks = [
        asyncio.create_task(_venue_loop("binance", symbols, accumulator, session_log)),
        asyncio.create_task(_venue_loop("bybit", symbols, accumulator, session_log)),
        asyncio.create_task(
            _flush_loop(
                accumulator,
                root,
                report_root,
                args.flush_seconds,
                args.coverage_hours,
                args.coverage_refresh_seconds,
                session_log,
            )
        ),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record synchronized Binance/Bybit public trade bars.")
    parser.add_argument("--root", default="data/orderflow/v9_6_cross_venue")
    parser.add_argument("--report-root", default="reports/v9_6_cross_venue_tape")
    parser.add_argument(
        "--live-feature-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
    )
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--flush-seconds", type=int, default=60)
    parser.add_argument("--coverage-hours", type=int, default=24)
    parser.add_argument("--coverage-refresh-seconds", type=int, default=300)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
