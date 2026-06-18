"""v7S Direction D2 — Relative-Value Pair Short with CVD confirmation (77.docx §8).

The original v7S Direction D (5 candidates × 3 horizons × 2 confirmation
modes) was rejected on all 30 cells: every pair hedge HURT the naked
short, beta and BTC/ETH were too correlated, basket was catastrophic
(4-leg cost dominated any signal). The 77 docx explicitly says: do
NOT iterate the old D version. Open D2.

The D2 chain is structurally different:

    beta overextended relative to BTC/ETH/sector
  + beta CVD weakening (per-bar continuous Binance UM CVD)
  + hedge leg CVD stable/strong
  + beta failed follow-through
  -> short beta / long hedge

What changes vs D:
- The naive D used PRICE-LEVEL relative overextension (ret_4h_percentile).
  D2 additionally requires DIRECTIONAL FLOW divergence (beta CVD weakening
  while hedge CVD is stable/strong). This is the cross-section flow
  asymmetry that the relative-value pair hypothesis needs to make sense.
- The CVD reads come from the continuous Binance UM backfill
  (binance_continuous_cvd) at the SAME bar as the entry signal. No more
  CIC-event-anchored data gap.

The CVD gates fail closed when the continuous shard for the bar is
absent — this is the right default while the backfill ramps up.
Direction D2 is therefore strictly DATA-DEPENDENT on P1; on a box
where only a sample of days has been backfilled, D2 will fire only
in those days.

Two candidates this commit:
- ``D2_btc_cvd_pair`` — hedge = BTCUSDT
- ``D2_eth_cvd_pair`` — hedge = ETHUSDT

Holding horizons: h4 (16 bars), h12 (48 bars), h24 (96 bars).
Execution: 2-leg pair short, same cost model as v7s Direction D
pair candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.binance_continuous_cvd import CONTINUOUS_ROOT
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v12s_short_motif_atlas import _f

REPORT_ROOT = Path("reports/v7s_short_alpha/D2_cvd_pair")

DIRECTION_D2 = "D2_cvd_pair"
CANDIDATE_D2_BTC = "D2_btc_cvd_pair"
CANDIDATE_D2_ETH = "D2_eth_cvd_pair"
D2_CANDIDATES: tuple[str, ...] = (CANDIDATE_D2_BTC, CANDIDATE_D2_ETH)

# Map candidate -> hedge symbol.
HEDGE_FOR_CANDIDATE: dict[str, str] = {
    CANDIDATE_D2_BTC: "BTCUSDT",
    CANDIDATE_D2_ETH: "ETHUSDT",
}


@dataclass(frozen=True)
class D2Config:
    """Knobs for Direction D2."""

    report_root: Path = REPORT_ROOT
    continuous_root: Path = CONTINUOUS_ROOT
    top_n: int = 30

    # Distribution gates.
    d2_lookback_bars: int = 16
    d2_overextended_pct: float = 95.0
    d2_relative_overperf_min: float = 0.02  # beta - hedge return ≥ 2 %
    d2_reclaim_tolerance: float = 0.015  # 1.5 % below lookback high
    d2_cooldown_bars: int = 24

    # CVD confirmation thresholds.
    d2_cvd_window_minutes: int = 15  # which CVD bar size to read (5 / 15)
    d2_beta_cvd_max: float = -0.05  # beta's buy_sell_imbalance ≤ -0.05 → weakening
    d2_hedge_cvd_min: float = -0.05  # hedge's buy_sell_imbalance ≥ -0.05 → stable/strong

    # Fixed pair holding horizons in bars (15-min bars: 16/48/96 = 4h/12h/24h).
    d2_holding_horizons_bars: tuple[int, ...] = (16, 48, 96)

    # Hedge exclusion: do not run D2 on hedge symbols.
    d2_exclude_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")


# --------------------------------------------------------------------------------------
# CVD lookups
# --------------------------------------------------------------------------------------


def _continuous_shard_path(
    continuous_root: Path,
    symbol: str,
    day: date,
    bar_size: str,
) -> Path:
    yyyymm = f"{day:%Y-%m}"
    return continuous_root / symbol / bar_size / f"{yyyymm}.parquet"


def _bar_size_label(window_minutes: int) -> str:
    return f"{window_minutes}min"


def build_cvd_lookup(
    symbol: str,
    bar_size: str,
    continuous_root: Path = CONTINUOUS_ROOT,
) -> Callable[[pd.Timestamp], dict | None]:
    """Return a (bar_open_time → row) lookup over the symbol's continuous
    CVD shards. Missing shards are tolerated — the lookup returns ``None``
    for any timestamp outside the loaded sample."""
    sym_dir = continuous_root / symbol / bar_size
    if not sym_dir.exists():
        empty: dict[int, dict] = {}

        def empty_lookup(ts: pd.Timestamp) -> dict | None:
            return None

        return empty_lookup
    frames: list[pd.DataFrame] = []
    for shard in sorted(sym_dir.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(shard))
        except Exception:
            continue
    if not frames:
        def empty_lookup(ts: pd.Timestamp) -> dict | None:
            return None

        return empty_lookup
    full = pd.concat(frames, ignore_index=True)
    if "bar_open_time" not in full.columns:
        def empty_lookup(ts: pd.Timestamp) -> dict | None:
            return None

        return empty_lookup
    full["bar_open_time"] = pd.to_datetime(full["bar_open_time"], utc=True, errors="coerce")
    full = full.dropna(subset=["bar_open_time"])
    by_ns: dict[int, dict] = {
        int(t.value): row for t, row in zip(full["bar_open_time"], full.to_dict(orient="records"))
    }

    def lookup(ts: pd.Timestamp) -> dict | None:
        return by_ns.get(int(pd.Timestamp(ts).floor(bar_size).value))

    return lookup


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------


def _gate_beta_overextended(group: pd.DataFrame, idx: int, threshold: float, cfg: D2Config) -> bool:
    """Symbol's ``ret_4h_percentile`` reached ``threshold`` somewhere in lookback."""
    if "ret_4h_percentile" not in group.columns:
        return False
    pct = _f(group, "ret_4h_percentile")
    if idx >= len(pct):
        return False
    start = max(0, idx - cfg.d2_lookback_bars)
    window = pct[start : idx + 1]
    if window.size == 0:
        return False
    return bool(np.nanmax(window) >= threshold)


def _gate_relative_overperf(
    group: pd.DataFrame,
    idx: int,
    hedge_lookup: Callable[[pd.Timestamp], dict | None] | None,
    cfg: D2Config,
) -> bool:
    """beta_return_lookback - hedge_return_lookback ≥ d2_relative_overperf_min."""
    close = _f(group, "close")
    if idx >= len(close) or idx < cfg.d2_lookback_bars:
        return False
    if not (np.isfinite(close[idx - cfg.d2_lookback_bars]) and np.isfinite(close[idx])):
        return False
    beta_ret = (close[idx] - close[idx - cfg.d2_lookback_bars]) / close[idx - cfg.d2_lookback_bars]
    # Pull BTC return from the joined column (already on every symbol's group).
    if "btc_ret_4h" in group.columns:
        btc_ret_arr = _f(group, "btc_ret_4h")
        hedge_ret = float(btc_ret_arr[idx]) if idx < len(btc_ret_arr) and np.isfinite(btc_ret_arr[idx]) else float("nan")
    else:
        hedge_ret = float("nan")
    if not np.isfinite(hedge_ret):
        return False
    return bool(beta_ret - hedge_ret >= cfg.d2_relative_overperf_min)


def _gate_beta_failed_followthrough(group: pd.DataFrame, idx: int, cfg: D2Config) -> bool:
    """Close is ≥ ``d2_reclaim_tolerance`` below lookback high."""
    close = _f(group, "close")
    high = _f(group, "high")
    if idx >= len(close) or idx >= len(high) or idx < cfg.d2_lookback_bars:
        return False
    start = max(0, idx - cfg.d2_lookback_bars)
    lookback_high = float(np.nanmax(high[start : idx + 1]))
    cur_close = float(close[idx])
    if not (np.isfinite(lookback_high) and np.isfinite(cur_close)) or lookback_high <= 0:
        return False
    drop = (lookback_high - cur_close) / lookback_high
    return bool(drop >= cfg.d2_reclaim_tolerance)


def _gate_cvd_divergence(
    bar_open_time: pd.Timestamp,
    beta_cvd_lookup: Callable[[pd.Timestamp], dict | None],
    hedge_cvd_lookup: Callable[[pd.Timestamp], dict | None],
    cfg: D2Config,
) -> tuple[bool, str]:
    """beta's CVD weakening AND hedge's CVD stable-or-strong at signal bar.

    Returns (passed, audit_reason). Fails closed when either CVD lookup
    is empty — D2 is data-dependent on the P1 continuous backfill.
    """
    beta_payload = beta_cvd_lookup(bar_open_time)
    if beta_payload is None:
        return False, "beta_cvd_missing"
    hedge_payload = hedge_cvd_lookup(bar_open_time)
    if hedge_payload is None:
        return False, "hedge_cvd_missing"
    beta_imb = beta_payload.get("buy_sell_imbalance")
    hedge_imb = hedge_payload.get("buy_sell_imbalance")
    if not (np.isfinite(beta_imb) and np.isfinite(hedge_imb)):
        return False, "cvd_imbalance_nan"
    if beta_imb > cfg.d2_beta_cvd_max:
        return False, "beta_cvd_not_weakening"
    if hedge_imb < cfg.d2_hedge_cvd_min:
        return False, "hedge_cvd_not_stable"
    return True, "ok"


# --------------------------------------------------------------------------------------
# Signal emission + pair execution
# --------------------------------------------------------------------------------------


def _emit_d2_signals(
    group: pd.DataFrame,
    candidate_code: str,
    beta_cvd_lookup: Callable[[pd.Timestamp], dict | None],
    hedge_cvd_lookup: Callable[[pd.Timestamp], dict | None],
    cfg: D2Config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    symbol = str(group["symbol"].iloc[0]) if len(group) else ""
    if not symbol or symbol in set(cfg.d2_exclude_symbols):
        return rows
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    n = len(group)
    last_fire = -1_000_000
    audit_buckets: dict[str, int] = {}
    for idx in range(cfg.d2_lookback_bars, n - 1):
        if idx - last_fire < cfg.d2_cooldown_bars:
            continue
        if not _gate_beta_overextended(group, idx, cfg.d2_overextended_pct, cfg):
            continue
        if not _gate_relative_overperf(group, idx, hedge_cvd_lookup, cfg):
            continue
        if not _gate_beta_failed_followthrough(group, idx, cfg):
            continue
        ts = bar_open_time.iloc[idx]
        cvd_ok, audit = _gate_cvd_divergence(ts, beta_cvd_lookup, hedge_cvd_lookup, cfg)
        audit_buckets[audit] = audit_buckets.get(audit, 0) + 1
        if not cvd_ok:
            continue
        entry_idx = idx + 1
        if entry_idx >= n:
            continue
        rows.append({
            "direction": DIRECTION_D2,
            "candidate_code": candidate_code,
            "sleeve_code": candidate_code,
            "exchange": str(group.iloc[idx].get("exchange", "")),
            "symbol": symbol,
            "anchor_idx": int(idx),
            "entry_idx": int(entry_idx),
            "signal_time": feature_time.iloc[idx],
            "entry_time": bar_open_time.iloc[entry_idx],
            "month": (
                feature_time.iloc[idx].strftime("%Y-%m")
                if pd.notna(feature_time.iloc[idx])
                else ""
            ),
            "audit_reason": audit,
        })
        last_fire = idx
    return rows


def _execute_d2_pair(
    group: pd.DataFrame,
    hedge_group: pd.DataFrame,
    signal: dict,
    cfg: D2Config,
) -> list[dict[str, object]]:
    """Per signal, emit one row per fixed horizon."""
    entry_idx = int(signal["entry_idx"])
    open_arr = _f(group, "open")
    close_arr = _f(group, "close")
    if entry_idx < 0 or entry_idx >= len(open_arr):
        return []
    symbol_entry = float(open_arr[entry_idx])
    if not np.isfinite(symbol_entry) or symbol_entry <= 0:
        return []
    entry_time = pd.Timestamp(signal["entry_time"])
    if pd.isna(entry_time):
        return []
    # Hedge entry: same bar_open_time
    hedge_open_arr = _f(hedge_group, "open")
    hedge_close_arr = _f(hedge_group, "close")
    hedge_open_times = pd.to_datetime(hedge_group["bar_open_time"], utc=True, errors="coerce")
    ns_to_idx = {int(t.value): i for i, t in enumerate(hedge_open_times) if pd.notna(t)}
    hedge_entry_idx = ns_to_idx.get(int(entry_time.value), -1)
    if hedge_entry_idx < 0 or hedge_entry_idx >= len(hedge_open_arr):
        return []
    hedge_entry = float(hedge_open_arr[hedge_entry_idx])
    if not np.isfinite(hedge_entry) or hedge_entry <= 0:
        return []
    rows: list[dict[str, object]] = []
    for horizon in cfg.d2_holding_horizons_bars:
        sym_exit_idx = entry_idx + horizon
        hed_exit_idx = hedge_entry_idx + horizon
        if sym_exit_idx >= len(close_arr) or hed_exit_idx >= len(hedge_close_arr):
            continue
        sym_exit = float(close_arr[sym_exit_idx])
        hed_exit = float(hedge_close_arr[hed_exit_idx])
        if not (np.isfinite(sym_exit) and np.isfinite(hed_exit)):
            continue
        beta_short_ret = (symbol_entry - sym_exit) / symbol_entry
        hedge_long_ret = (hed_exit - hedge_entry) / hedge_entry
        pair_gross = beta_short_ret + hedge_long_ret
        # 2-leg cost (matches D1 cost model).
        pair_net20 = pair_gross - 4.0 * 20.0 / 10_000.0
        pair_net30 = pair_gross - 4.0 * 30.0 / 10_000.0
        out = dict(signal)
        out.update({
            "execution": f"h{horizon * 15 // 60}",
            "holding_bars": int(horizon),
            "gross_return": float(pair_gross),
            "beta_short_gross": float(beta_short_ret),
            "hedge_long_gross": float(hedge_long_ret),
            "net20": float(pair_net20),
            "net30": float(pair_net30),
            "exit_reason": "horizon",
        })
        rows.append(out)
    return rows


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def collect_direction_d2(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: D2Config | None = None,
) -> pd.DataFrame:
    """Stream Direction D2 over the universe × CVD-loaded days only.

    Rows fire only where:
    1. The symbol's continuous CVD has the bar.
    2. The hedge's continuous CVD has the bar.
    Otherwise the CVD gate fails closed.
    """
    cfg = cfg or D2Config()
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )

    bar_size = _bar_size_label(cfg.d2_cvd_window_minutes)
    cvd_lookups: dict[str, Callable[[pd.Timestamp], dict | None]] = {}
    hedge_groups: dict[str, pd.DataFrame] = {}
    for hedge_sym in set(HEDGE_FOR_CANDIDATE.values()):
        cvd_lookups[hedge_sym] = build_cvd_lookup(hedge_sym, bar_size, cfg.continuous_root)
        hedge_group = _read_symbol_features(feature_path, rank30, rank90, hedge_sym, config)
        if not hedge_group.empty:
            hedge_groups[hedge_sym] = hedge_group.sort_values("bar_open_time").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for i, symbol in enumerate(symbols, start=1):
        if symbol in set(cfg.d2_exclude_symbols):
            continue
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        beta_cvd = build_cvd_lookup(symbol, bar_size, cfg.continuous_root)
        for cand_code in D2_CANDIDATES:
            hedge_sym = HEDGE_FOR_CANDIDATE[cand_code]
            hedge_group = hedge_groups.get(hedge_sym)
            if hedge_group is None:
                continue
            signals = _emit_d2_signals(group, cand_code, beta_cvd, cvd_lookups[hedge_sym], cfg)
            for sig in signals:
                rows.extend(_execute_d2_pair(group, hedge_group, sig, cfg))
        if i % 25 == 0:
            print(f"v7S D2: {i}/{len(symbols)} symbols, {len(rows)} executions", flush=True)
    return pd.DataFrame(rows)


def write_direction_d2(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: D2Config | None = None,
) -> dict[str, Path]:
    cfg = cfg or D2Config()
    report_root = ensure_dir(cfg.report_root)
    if not feature_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(report_root / "short_trades.csv", index=False)
        return {"trades": report_root / "short_trades.csv"}
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    trades = collect_direction_d2(feature_path, rank30, rank90, config, cfg)
    trades_path = report_root / "short_trades.csv"
    trades.to_csv(trades_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not trades.empty:
        for (cand, exe), sub in trades.groupby(["candidate_code", "execution"]):
            nets = pd.to_numeric(sub["net20"], errors="coerce")
            summary_rows.append({
                "candidate_code": cand,
                "execution": exe,
                "n_trades": int(len(sub)),
                "mean_gross": float(pd.to_numeric(sub["gross_return"], errors="coerce").mean()),
                "mean_net20": float(nets.mean()),
                "mean_net30": float(pd.to_numeric(sub["net30"], errors="coerce").mean()),
                "win_rate_net20": float((nets > 0).mean()),
            })
    summary_path = report_root / "short_candidate_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return {"trades": trades_path, "summary": summary_path}


__all__ = [
    "D2Config",
    "DIRECTION_D2",
    "CANDIDATE_D2_BTC",
    "CANDIDATE_D2_ETH",
    "D2_CANDIDATES",
    "HEDGE_FOR_CANDIDATE",
    "REPORT_ROOT",
    "build_cvd_lookup",
    "collect_direction_d2",
    "write_direction_d2",
]
