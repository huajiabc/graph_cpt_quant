from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

from pressure_graph.binance_um_carry_history import (
    BinanceUmCarryConfig,
    fetch_um_funding,
)
from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.paper_live.cm2 import (
    latest_monday,
    load_cm2_live_config,
    load_membership,
    write_cm2_forward_shadow,
)
from pressure_graph.recent_perp_carry_history import fetch_binance_um_klines


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _merge(
    existing: pd.DataFrame,
    current: pd.DataFrame,
    *,
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
        .dropna(subset=[time_column])
        .sort_values(key)
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )


def _start(
    path: Path,
    time_column: str,
    observed: pd.Timestamp,
    history_days: int,
    overlap: pd.Timedelta,
) -> pd.Timestamp:
    floor = observed - pd.Timedelta(days=history_days)
    if not path.exists():
        return floor
    data = pd.read_parquet(path, columns=[time_column])
    if data.empty:
        return floor
    latest = pd.to_datetime(data[time_column], utc=True, errors="coerce").max()
    return floor if pd.isna(latest) else max(floor, latest - overlap)


def _refresh_symbol(
    symbol: str,
    observed: pd.Timestamp,
    cfg,
    bybit_base_url: str,
    bybit_category: str,
) -> dict[str, object]:
    root = cfg.live_root / "raw"
    paths = {
        "bybit_kline": root / "bybit" / "klines_1h" / f"{symbol}.parquet",
        "bybit_funding": root / "bybit" / "funding" / f"{symbol}.parquet",
        "binance_kline": root / "binance" / "klines_1h" / f"{symbol}.parquet",
        "binance_funding": root / "binance" / "funding" / f"{symbol}.parquet",
    }
    latest_closed_open = observed.floor("h") - pd.Timedelta(hours=1)
    kline_start = min(
        _start(
            paths["bybit_kline"],
            "bar_open_time",
            observed,
            cfg.history_days,
            pd.Timedelta(hours=2),
        ),
        _start(
            paths["binance_kline"],
            "bar_open_time",
            observed,
            cfg.history_days,
            pd.Timedelta(hours=2),
        ),
    )
    funding_days = cfg.funding_lookback_days + 3
    funding_start = min(
        _start(
            paths["bybit_funding"],
            "funding_time",
            observed,
            funding_days,
            pd.Timedelta(hours=8),
        ),
        _start(
            paths["binance_funding"],
            "funding_time",
            observed,
            funding_days,
            pd.Timedelta(hours=8),
        ),
    )
    errors: list[str] = []
    bybit = BybitClient(bybit_base_url, bybit_category)
    try:
        bybit_klines = bybit.klines(
            symbol, kline_start, latest_closed_open, "1h"
        )
        bybit_funding = bybit.funding_history(
            symbol, funding_start, observed
        )
    except Exception as exc:
        bybit_klines = pd.DataFrame()
        bybit_funding = pd.DataFrame()
        errors.append(f"bybit:{type(exc).__name__}:{exc}")
    finally:
        bybit.close()
    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-forward-shadow/1.0"},
    ) as client:
        try:
            binance_klines = fetch_binance_um_klines(
                symbol, kline_start, observed.floor("h"), client
            )
            binance_funding = fetch_um_funding(
                symbol,
                funding_start.date(),
                observed.date(),
                client,
                BinanceUmCarryConfig(timeout_seconds=60),
            )
        except Exception as exc:
            binance_klines = pd.DataFrame()
            binance_funding = pd.DataFrame()
            errors.append(f"binance:{type(exc).__name__}:{exc}")
    cutoff = observed - pd.Timedelta(days=cfg.history_days)
    funding_cutoff = observed - pd.Timedelta(days=funding_days)
    outputs = {
        "bybit_kline": _merge(
            _read(paths["bybit_kline"]),
            bybit_klines,
            key=["exchange", "symbol", "bar_open_time"],
            time_column="bar_open_time",
            cutoff=cutoff - pd.Timedelta(hours=1),
        ),
        "bybit_funding": _merge(
            _read(paths["bybit_funding"]),
            bybit_funding,
            key=["exchange", "symbol", "funding_time"],
            time_column="funding_time",
            cutoff=funding_cutoff,
        ),
        "binance_kline": _merge(
            _read(paths["binance_kline"]),
            binance_klines,
            key=["bybit_symbol", "bar_open_time"],
            time_column="bar_open_time",
            cutoff=cutoff - pd.Timedelta(hours=1),
        ),
        "binance_funding": _merge(
            _read(paths["binance_funding"]),
            binance_funding,
            key=["bybit_symbol", "funding_time"],
            time_column="funding_time",
            cutoff=funding_cutoff,
        ),
    }
    for name, frame in outputs.items():
        if not frame.empty:
            write_parquet(frame, paths[name])
    return {
        "symbol": symbol,
        "bybit_kline_rows": len(outputs["bybit_kline"]),
        "bybit_funding_rows": len(outputs["bybit_funding"]),
        "binance_kline_rows": len(outputs["binance_kline"]),
        "binance_funding_rows": len(outputs["binance_funding"]),
        "error": "|".join(errors) or None,
    }


def _concat(root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _should_refresh(cfg, observed: pd.Timestamp, force: bool) -> bool:
    if force:
        return True
    decisions_path = cfg.report_root / "tg1" / "forward" / "decisions.parquet"
    if not decisions_path.exists():
        return True
    decisions = pd.read_parquet(decisions_path, columns=["decision_time"])
    due = latest_monday(observed)
    if not pd.to_datetime(
        decisions["decision_time"], utc=True, errors="coerce"
    ).eq(due).any():
        return True
    status_path = cfg.live_root / "raw_refresh_status.json"
    if not status_path.exists():
        return True
    last = pd.Timestamp(
        json.loads(status_path.read_text(encoding="utf-8"))[
            "refreshed_at_utc"
        ]
    )
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    return observed - last >= pd.Timedelta(minutes=cfg.refresh_interval_minutes)


def refresh_raw(config_path: str | Path, observed: pd.Timestamp) -> None:
    cfg = load_cm2_live_config(config_path)
    base = load_config(cfg.base_config)
    membership = load_membership(cfg.membership_path)
    month = pd.Timestamp(
        year=observed.year, month=observed.month, day=1, tz="UTC"
    )
    symbols = sorted(
        membership.loc[membership["month_start"].eq(month), "symbol"]
        .astype(str)
        .unique()
    )
    if len(symbols) < cfg.bucket_size:
        raise RuntimeError(f"CM2/TG1 membership unavailable for {month:%Y-%m}")
    ensure_dir(cfg.live_root / "raw")
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                _refresh_symbol,
                symbol,
                observed,
                cfg,
                str(base.exchanges.bybit.base_url),
                base.exchanges.bybit.category,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "symbol": symbol,
                    "bybit_kline_rows": 0,
                    "bybit_funding_rows": 0,
                    "binance_kline_rows": 0,
                    "binance_funding_rows": 0,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            results.append(result)
            print(
                f"cm2 raw {completed}/{len(symbols)} {symbol} "
                f"error={result['error'] or '-'}",
                flush=True,
            )
    manifest = pd.DataFrame(results).sort_values("symbol")
    manifest.to_csv(cfg.live_root / "raw_manifest.csv", index=False)
    status = {
        "refreshed_at_utc": observed.isoformat(),
        "symbols": len(symbols),
        "complete_symbols": int(
            (
                manifest["bybit_kline_rows"].gt(0)
                & manifest["bybit_funding_rows"].gt(0)
                & manifest["binance_kline_rows"].gt(0)
                & manifest["binance_funding_rows"].gt(0)
            ).sum()
        ),
        "errors": manifest.loc[
            manifest["error"].notna(), ["symbol", "error"]
        ].to_dict("records"),
        "real_orders_allowed": False,
    }
    (cfg.live_root / "raw_refresh_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_inputs(cfg):
    raw = cfg.live_root / "raw"
    bybit_klines = _concat(raw / "bybit" / "klines_1h")
    bybit_funding = _concat(raw / "bybit" / "funding")
    binance_klines = _concat(raw / "binance" / "klines_1h")
    binance_funding = _concat(raw / "binance" / "funding")
    if any(
        frame.empty
        for frame in (
            bybit_klines,
            bybit_funding,
            binance_klines,
            binance_funding,
        )
    ):
        raise FileNotFoundError("CM2/TG1 live raw inputs are incomplete")
    bybit_prices = bybit_klines[
        ["symbol", "bar_close_time", "close"]
    ].rename(columns={"bar_close_time": "feature_time"})
    binance_prices = binance_klines[
        ["bybit_symbol", "feature_time", "close"]
    ].rename(
        columns={"bybit_symbol": "symbol", "close": "binance_close"}
    )
    if "symbol" not in bybit_funding and "bybit_symbol" in bybit_funding:
        bybit_funding = bybit_funding.rename(
            columns={"bybit_symbol": "symbol"}
        )
    if "symbol" not in binance_funding and "bybit_symbol" in binance_funding:
        binance_funding = binance_funding.rename(
            columns={"bybit_symbol": "symbol"}
        )
    return bybit_funding, binance_funding, bybit_prices, binance_prices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one exact TG1 sleeve and fixed 80/20 CM2 shadow."
    )
    parser.add_argument(
        "--config", default="configs/v16_5_cm2_forward_shadow.yaml"
    )
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    cfg = load_cm2_live_config(args.config)
    observed = pd.Timestamp.now(tz="UTC")
    if (
        not args.skip_refresh
        and _should_refresh(cfg, observed, args.force_refresh)
    ):
        refresh_raw(args.config, observed)
    inputs = _load_inputs(cfg)
    membership = load_membership(cfg.membership_path)
    outputs = write_cm2_forward_shadow(
        *inputs, membership, cfg, observed_at=observed
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
