from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.paper_live.fss3 import (
    build_fss3_hourly_prices,
    latest_monday,
    load_fss3_live_config,
    load_fss3_membership,
    write_fss3_forward_shadow,
)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _append_pruned(
    existing: pd.DataFrame,
    current: pd.DataFrame,
    key: list[str],
    time_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    frames = [frame for frame in (existing, current) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    output[time_column] = pd.to_datetime(
        output[time_column], utc=True, errors="coerce"
    )
    return (
        output[output[time_column].ge(cutoff)]
        .sort_values(key)
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )


def _start_for_symbol(
    path: Path,
    time_column: str,
    end: pd.Timestamp,
    history_days: int,
) -> pd.Timestamp:
    floor = end - pd.Timedelta(days=history_days)
    if not path.exists():
        return floor
    data = pd.read_parquet(path, columns=[time_column])
    if data.empty:
        return floor
    latest = pd.to_datetime(data[time_column], utc=True, errors="coerce").max()
    return floor if pd.isna(latest) else max(floor, latest - pd.Timedelta(hours=2))


def _concat(root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _decision_exists(report_root: Path, decision: pd.Timestamp) -> bool:
    path = report_root / "forward" / "decisions.parquet"
    if not path.exists():
        return False
    decisions = pd.read_parquet(path, columns=["decision_time"])
    times = pd.to_datetime(decisions["decision_time"], utc=True, errors="coerce")
    return bool(times.eq(decision).any())


def _should_refresh(cfg, observed: pd.Timestamp, force: bool) -> bool:
    if force:
        return True
    if not _decision_exists(cfg.report_root, latest_monday(observed)):
        return True
    status_path = cfg.live_root / "raw_refresh_status.json"
    if not status_path.exists():
        return True
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    last = pd.Timestamp(payload["refreshed_at_utc"])
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    return observed - last >= pd.Timedelta(minutes=cfg.refresh_interval_minutes)


def refresh_fss3_raw(
    config_path: str | Path,
    observed_at: pd.Timestamp,
) -> None:
    cfg = load_fss3_live_config(config_path)
    base = load_config(cfg.base_config)
    membership = load_fss3_membership(cfg.membership_path)
    month = pd.Timestamp(
        year=observed_at.year, month=observed_at.month, day=1, tz="UTC"
    )
    symbols = sorted(
        set(
            membership.loc[
                membership["month_start"].eq(month), "symbol"
            ].astype(str)
        )
        | {"BTCUSDT"}
    )
    if len(symbols) <= 1:
        raise RuntimeError(f"FSS3 membership is unavailable for {month:%Y-%m}")
    raw_root = ensure_dir(cfg.live_root / "raw" / "bybit")
    kline_root = ensure_dir(raw_root / "klines_1h")
    funding_root = ensure_dir(raw_root / "funding")
    latest_closed_open = observed_at.floor("h") - pd.Timedelta(hours=1)
    cutoff = observed_at - pd.Timedelta(days=cfg.history_days)
    funding_history_days = max(16, cfg.funding_lookback_days * 2 + 2)
    funding_cutoff = observed_at - pd.Timedelta(days=funding_history_days)
    client = BybitClient(
        str(base.exchanges.bybit.base_url), base.exchanges.bybit.category
    )
    errors: list[str] = []
    try:
        instruments = client.instruments(base.exchanges.bybit.settle_coin)
        write_parquet(instruments, raw_root / "instruments.parquet")
        active = set(instruments.get("symbol", pd.Series(dtype=str)).astype(str))
        inactive = sorted(set(symbols) - active)
        if inactive:
            errors.append("inactive_members:" + "|".join(inactive))
        for index, symbol in enumerate(symbols, start=1):
            print(f"fss3 refresh {index}/{len(symbols)} {symbol}", flush=True)
            kline_path = kline_root / f"{symbol}.parquet"
            funding_path = funding_root / f"{symbol}.parquet"
            try:
                kline_start = _start_for_symbol(
                    kline_path,
                    "bar_open_time",
                    latest_closed_open,
                    cfg.history_days,
                )
                funding_start = _start_for_symbol(
                    funding_path,
                    "funding_time",
                    observed_at,
                    funding_history_days,
                )
                klines = client.klines(
                    symbol,
                    kline_start,
                    latest_closed_open,
                    cfg.price_interval,
                )
                funding = client.funding_history(
                    symbol, funding_start, observed_at
                )
                merged_klines = _append_pruned(
                    _read_optional(kline_path),
                    klines,
                    ["exchange", "symbol", "bar_open_time"],
                    "bar_open_time",
                    cutoff - pd.Timedelta(hours=1),
                )
                if not merged_klines.empty:
                    merged_klines = merged_klines[
                        pd.to_datetime(
                            merged_klines["bar_close_time"],
                            utc=True,
                            errors="coerce",
                        ).le(observed_at)
                    ]
                merged_funding = _append_pruned(
                    _read_optional(funding_path),
                    funding,
                    ["exchange", "symbol", "funding_time"],
                    "funding_time",
                    funding_cutoff,
                )
                write_parquet(merged_klines, kline_path)
                write_parquet(merged_funding, funding_path)
            except Exception as exc:  # fail closed later with explicit telemetry
                errors.append(f"{symbol}:{type(exc).__name__}:{exc}")
    finally:
        client.close()
    status = {
        "refreshed_at_utc": observed_at.isoformat(),
        "symbols": len(symbols),
        "errors": errors,
        "real_orders_allowed": False,
    }
    (cfg.live_root / "raw_refresh_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one stateful v14.9 FSS3 record-only forward refresh."
    )
    parser.add_argument(
        "--config", default="configs/v14_9_fss3_forward_shadow.yaml"
    )
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    cfg = load_fss3_live_config(args.config)
    observed = pd.Timestamp.now(tz="UTC")
    if (
        not args.skip_refresh
        and _should_refresh(cfg, observed, args.force_refresh)
    ):
        refresh_fss3_raw(args.config, observed)
    kline_root = cfg.live_root / "raw" / "bybit" / "klines_1h"
    funding_root = cfg.live_root / "raw" / "bybit" / "funding"
    klines = _concat(kline_root)
    funding = _concat(funding_root)
    if klines.empty or funding.empty:
        raise FileNotFoundError("FSS3 live kline/funding data are unavailable")
    prices = build_fss3_hourly_prices(klines)
    membership = load_fss3_membership(cfg.membership_path)
    outputs = write_fss3_forward_shadow(
        funding, prices, membership, cfg, observed_at=observed
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
