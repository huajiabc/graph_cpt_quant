"""Readiness and quality gate for the forward cross-venue flow graph."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.cross_venue_tape import combine_bar_fragments
from pressure_graph.io import ensure_dir


TAPE_ROOT = Path("data/orderflow/v9_6_cross_venue")
REPORT_ROOT = Path("reports/v10_7_cross_venue_flow_graph_readiness")
ADMISSIBLE_START = pd.Timestamp("2026-07-13T11:01:00Z")
FROZEN_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "HYPEUSDT",
    "XRPUSDT",
    "ZECUSDT",
    "DOGEUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "ONDOUSDT",
    "1000PEPEUSDT",
    "XLMUSDT",
    "XAUTUSDT",
    "ADAUSDT",
    "WLDUSDT",
    "TAOUSDT",
    "FARTCOINUSDT",
    "LINKUSDT",
    "BNBUSDT",
    "ENAUSDT",
)


@dataclass(frozen=True)
class V107Config:
    tape_root: Path = TAPE_ROOT
    report_root: Path = REPORT_ROOT
    admissible_start: pd.Timestamp = ADMISSIBLE_START
    frozen_symbols: tuple[str, ...] = FROZEN_SYMBOLS
    required_calendar_days: int = 90
    required_months: int = 3
    required_usable_symbols: int = 15
    required_full_days_per_symbol: int = 80
    synchronized_ratio_floor: float = 0.95
    stale_seconds: float = 10.0
    minimum_evaluated_day_minutes: int = 1_200


def inventory_v107_fragments(root: Path = TAPE_ROOT) -> pd.DataFrame:
    bar_root = root / "bars_1m"
    rows = []
    if not bar_root.exists():
        return pd.DataFrame(columns=["day", "path", "bytes", "modified_utc"])
    for path in sorted(bar_root.glob("*/*.parquet")):
        stat = path.stat()
        rows.append(
            {
                "day": path.parent.name,
                "path": str(path),
                "bytes": int(stat.st_size),
                "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def load_v107_fragments(root: Path = TAPE_ROOT) -> pd.DataFrame:
    inventory = inventory_v107_fragments(root)
    if inventory.empty:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in inventory["path"]]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combine_bar_fragments(raw) if not raw.empty else raw


def build_v107_synchronized_panel(
    bars: pd.DataFrame,
    cfg: V107Config = V107Config(),
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    data = bars.copy()
    data["bar_open_time"] = pd.to_datetime(
        data["bar_open_time"], utc=True, errors="coerce"
    )
    data = data[
        data["bar_open_time"].ge(cfg.admissible_start)
        & data["symbol"].astype(str).isin(cfg.frozen_symbols)
        & data["exchange"].astype(str).isin(["binance", "bybit"])
        & data["bar_complete"].fillna(False).astype(bool)
        & pd.to_numeric(data["event_lag_seconds"], errors="coerce").le(
            cfg.stale_seconds
        )
    ].copy()
    if data.empty:
        return pd.DataFrame()
    value_columns = [
        "buy_sell_imbalance",
        "turnover",
        "close",
        "event_lag_seconds",
    ]
    wide = data.pivot_table(
        index=["symbol", "bar_open_time"],
        columns="exchange",
        values=value_columns,
        aggfunc="last",
        observed=True,
    )
    required = [(column, exchange) for column in value_columns for exchange in ("binance", "bybit")]
    if any(column not in wide.columns for column in required):
        return pd.DataFrame()
    wide = wide.dropna(subset=required).reset_index()
    out = wide[["symbol", "bar_open_time"]].copy()
    for column in value_columns:
        out[f"binance_{column}"] = wide[(column, "binance")].to_numpy()
        out[f"bybit_{column}"] = wide[(column, "bybit")].to_numpy()
    out["cross_venue_imbalance"] = out[
        ["binance_buy_sell_imbalance", "bybit_buy_sell_imbalance"]
    ].mean(axis=1)
    out["flow_sign_agreement"] = np.sign(out["binance_buy_sell_imbalance"]).eq(
        np.sign(out["bybit_buy_sell_imbalance"])
    )
    out["flow_spread"] = (
        out["binance_buy_sell_imbalance"] - out["bybit_buy_sell_imbalance"]
    )
    return out.sort_values(["bar_open_time", "symbol"]).reset_index(drop=True)


def _symbol_day_coverage(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: V107Config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_day = cfg.admissible_start.floor("D")
    last_day = as_of.floor("D")
    for day in pd.date_range(first_day, last_day, freq="1D", tz="UTC"):
        start = max(day, cfg.admissible_start)
        end = min(day + pd.Timedelta(days=1), as_of)
        expected = max(int((end - start) / pd.Timedelta(minutes=1)), 0)
        if expected < cfg.minimum_evaluated_day_minutes:
            continue
        day_panel = panel[
            panel["bar_open_time"].ge(start) & panel["bar_open_time"].lt(end)
        ]
        for symbol in cfg.frozen_symbols:
            sample = day_panel[day_panel["symbol"].eq(symbol)]
            observed = int(sample["bar_open_time"].nunique())
            rows.append(
                {
                    "day": day,
                    "symbol": symbol,
                    "expected_minutes": expected,
                    "synchronized_minutes": observed,
                    "synchronized_ratio": observed / expected if expected else np.nan,
                    "coverage_pass": bool(
                        expected > 0
                        and observed / expected >= cfg.synchronized_ratio_floor
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_v107_readiness(
    panel: pd.DataFrame,
    cfg: V107Config = V107Config(),
    as_of: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    coverage = _symbol_day_coverage(panel, now, cfg) if not panel.empty else pd.DataFrame()
    elapsed_days = float((now - cfg.admissible_start) / pd.Timedelta(days=1))
    observed_months = (
        int(panel["bar_open_time"].dt.strftime("%Y-%m").nunique())
        if not panel.empty
        else 0
    )
    if coverage.empty:
        usable_symbols = 0
        min_ratio = 0.0
        quality_pass = False
        minimum_days = 0
    else:
        symbol_stats = coverage.groupby("symbol").agg(
            evaluated_days=("day", "nunique"),
            passing_days=("coverage_pass", "sum"),
            minimum_ratio=("synchronized_ratio", "min"),
        )
        usable = symbol_stats[
            symbol_stats["passing_days"].ge(cfg.required_full_days_per_symbol)
            & symbol_stats["minimum_ratio"].ge(cfg.synchronized_ratio_floor)
        ]
        usable_symbols = int(len(usable))
        min_ratio = float(coverage["synchronized_ratio"].min())
        quality_pass = bool(coverage["coverage_pass"].all())
        minimum_days = int(symbol_stats["passing_days"].min())
    gates = {
        "calendar_days_90": elapsed_days >= cfg.required_calendar_days,
        "calendar_months_3": observed_months >= cfg.required_months,
        "usable_symbols_15": usable_symbols >= cfg.required_usable_symbols,
        "all_symbol_days_sync_95pct": quality_pass,
        "full_days_per_symbol_80": minimum_days >= cfg.required_full_days_per_symbol,
    }
    local_available = not panel.empty
    if not local_available:
        status = "DATA_UNAVAILABLE_LOCAL"
    elif not quality_pass and not coverage.empty:
        status = "DATA_QUALITY_FAIL"
    elif all(gates.values()):
        status = "RESEARCH_GATE_OPEN"
    else:
        status = "DATA_ACCUMULATING"
    readiness = pd.DataFrame(
        [
            {
                "status": status,
                "as_of": now,
                "admissible_start": cfg.admissible_start,
                "earliest_time_gate": cfg.admissible_start
                + pd.Timedelta(days=cfg.required_calendar_days),
                "elapsed_calendar_days": elapsed_days,
                "observed_months": observed_months,
                "panel_rows": int(len(panel)),
                "observed_symbols": int(panel["symbol"].nunique())
                if not panel.empty
                else 0,
                "usable_symbols": usable_symbols,
                "minimum_synchronized_ratio": min_ratio,
                "minimum_passing_days_per_symbol": minimum_days,
                **{f"gate_{name}": bool(value) for name, value in gates.items()},
                "alpha_verdict_allowed": bool(status == "RESEARCH_GATE_OPEN"),
            }
        ]
    )
    return readiness, coverage


def write_v107_flow_graph_readiness(
    cfg: V107Config = V107Config(),
    as_of: object | None = None,
) -> dict[str, Path]:
    inventory = inventory_v107_fragments(cfg.tape_root)
    bars = load_v107_fragments(cfg.tape_root) if not inventory.empty else pd.DataFrame()
    panel = build_v107_synchronized_panel(bars, cfg)
    readiness, coverage = evaluate_v107_readiness(panel, cfg, as_of)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "inventory": root / "fragment_inventory.csv",
        "synchronized_panel": root / "latest_synchronized_panel.parquet",
        "coverage": root / "symbol_day_coverage.csv",
        "readiness": root / "readiness.csv",
        "status": root / "status.md",
    }
    inventory.to_csv(outputs["inventory"], index=False)
    panel.to_parquet(outputs["synchronized_panel"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    readiness.to_csv(outputs["readiness"], index=False)
    row = readiness.iloc[0]
    lines = [
        "# v10.7 Cross-Venue Flow Graph Readiness",
        "",
        f"Status: `{row['status']}`.",
        f"- admissible start: {row['admissible_start']}",
        f"- earliest time-only gate: {row['earliest_time_gate']}",
        f"- elapsed days: {row['elapsed_calendar_days']:.2f} / {cfg.required_calendar_days}",
        f"- synchronized panel rows: {int(row['panel_rows'])}",
        f"- observed / usable symbols: {int(row['observed_symbols'])} / {int(row['usable_symbols'])}",
        f"- alpha verdict allowed: {bool(row['alpha_verdict_allowed'])}",
        "- PaperLive and live permissions remain unchanged.",
    ]
    outputs["status"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
