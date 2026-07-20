from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.io import write_parquet
from pressure_graph.paper_live.v112 import (
    build_v112_return_panel,
    load_v112_live_config,
    write_v112_paper_live,
)
from v06a3_live_once import (
    _append_pruned,
    _floor_15m,
    _read_optional,
    _start_for_symbol,
)


def refresh_v112_klines(config_path: str | Path) -> pd.DataFrame:
    cfg = load_v112_live_config(config_path)
    base = load_config(cfg.base_config)
    end = _floor_15m()
    cutoff = end - pd.Timedelta(days=cfg.history_days)
    root = cfg.live_root / "raw" / "bybit" / "klines"
    client = BybitClient(str(base.exchanges.bybit.base_url), base.exchanges.bybit.category)
    try:
        refresh_symbols = tuple(dict.fromkeys([*cfg.symbols, *cfg.fallback_candidates]))
        for index, symbol in enumerate(refresh_symbols, start=1):
            path = root / f"{symbol}.parquet"
            start = _start_for_symbol(path, "bar_open_time", end, cfg.history_days)
            print(f"v112 klines {index}/{len(refresh_symbols)} {symbol}", flush=True)
            current = client.klines(symbol, start, end, base.experiment.base_interval)
            merged = _append_pruned(
                _read_optional(path),
                current,
                ["exchange", "symbol", "bar_open_time"],
                "bar_open_time",
                cutoff,
            )
            if not merged.empty:
                merged = merged[
                    pd.to_datetime(merged["bar_open_time"], utc=True, errors="coerce").le(end)
                ]
            write_parquet(merged, path)
    finally:
        client.close()
    frames = [pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v11.2 topology PaperLive shadow refresh.")
    parser.add_argument(
        "--config", default="configs/v11_2_high_vol_topology_paper_live.yaml"
    )
    parser.add_argument("--skip-refresh", action="store_true")
    args = parser.parse_args()
    cfg = load_v112_live_config(args.config)
    if args.skip_refresh:
        root = cfg.live_root / "raw" / "bybit" / "klines"
        frames = [pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))]
        klines = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        klines = refresh_v112_klines(args.config)
    if klines.empty:
        raise FileNotFoundError("No v11.2 live klines are available.")
    panel = build_v112_return_panel(klines)
    outputs = write_v112_paper_live(panel, cfg)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
