"""v1.2s Short Motif Atlas — phase-1 failure-path research for the short side.

Follows the short instruction doc. Core stance: shorts are NOT inverted longs;
the edge (if any) lives in *strong-structure failure / crowded unwind*, and the
overriding risk is the short squeeze. This atlas does not produce a strategy.
It scores a small set of failure motifs against matched baselines, splits by BTC
regime, and — per the doc's most important ask — compares every short against
simply *not being long*.

Tier: research only. No shadow, paper-live, or real-live wiring.

Motifs (per-symbol, computed from the existing v0.3 feature stream):
- S1 failed_reclaim_short      bullish shock -> pullback -> reclaim attempt -> failed reclaim
- S2 extreme_exhaustion_short  extreme strength + upper-wick rejection -> breakdown (per-symbol proxy)
- S3 crowded_long_unwind_short high funding + high OI delta + stalled price -> support break
- S5 btc_down_breakdown_short  BTC_down + symbol breakdown -> failed bounce -> lower low

S4 (leader -> beta failure) needs the cross-sectional graph and is deferred to a
later cross-system pass; including it here as a per-symbol proxy would misrepresent
a graph claim, which the doc explicitly warns against.

Engineering parity with the long side: same primary key (exchange, symbol,
feature_time), strict as-of (every gate/entry uses only bars with
feature_time <= decision bar), same cost grid (10/20/30/50 bp) plus a short
slippage add-on, same output filenames, same month-cap / symbol-contribution
guards.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.short_execution import (
    ShortExitRule,
    short_net_return,
    simulate_short_exit,
)
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v06a1 import _read_symbol_features


REPORT_ROOT = Path("reports/v1_2s_short_motif_atlas")
COST_GRID_BPS = (10.0, 20.0, 30.0, 50.0)
FOCAL_COST = 20.0
MONTH_CAP = 0.35
TOP_N = 30  # doc §5: Top30 liquidity only for phase 1.
REGIMES = ("BTC_down", "BTC_chop", "BTC_up")


@dataclass(frozen=True)
class ShortAtlasConfig:
    report_root: Path = REPORT_ROOT
    top_n: int = TOP_N
    extra_slippage_bps: float = 5.0  # doc §5: shorts carry higher slippage.
    take_profit: float = 0.03
    stop_loss: float = 0.025
    max_hold_bars: int = 16  # doc §5: shorter validity window than the long book.
    cooldown_bars: int = 16
    motifs: tuple[str, ...] = ("S1", "S2", "S3", "S5")


# --- motif parameter blocks ------------------------------------------------


@dataclass(frozen=True)
class MotifParams:
    code: str
    name: str
    pullback_pct: float = 0.010
    pullback_valid: int = 6
    reclaim_valid: int = 6
    fail_valid: int = 4
    extreme_pct: float = 90.0
    shock_z: float = 2.0
    wick_thresh: float = 0.45
    exhaust_valid: int = 6
    breakdown_valid: int = 6
    funding_hi: float = 80.0
    oi_hi: float = 70.0
    stall_pct: float = 55.0
    support_lookback: int = 8
    btc_breakdown_lookback: int = 8
    bounce_valid: int = 6


MOTIF_PARAMS = {
    "S1": MotifParams("S1", "failed_reclaim_short"),
    "S2": MotifParams("S2", "extreme_exhaustion_short"),
    "S3": MotifParams("S3", "crowded_long_unwind_short"),
    "S5": MotifParams("S5", "btc_down_breakdown_short"),
}


# --- numeric helpers -------------------------------------------------------


def _f(group: pd.DataFrame, col: str) -> np.ndarray:
    if col not in group.columns:
        return np.full(len(group), np.nan)
    return pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)


def _b(group: pd.DataFrame, col: str) -> np.ndarray:
    if col not in group.columns:
        return np.zeros(len(group), dtype=bool)
    return group[col].fillna(False).to_numpy(dtype=bool)


def _seed_from(text: str) -> int:
    """Deterministic seed (crc32) so matched-random baselines reproduce."""
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


# --- motif detectors -------------------------------------------------------
# Each returns a list of (confirmation_idx, anchor_idx). Short entry is taken at
# confirmation_idx + 1 (next bar open). anchor_idx marks the raw trigger bar and
# feeds the entry-only baseline. Detectors never read past confirmation_idx.


def detect_s1_failed_reclaim(group: pd.DataFrame, p: MotifParams) -> list[tuple[int, int]]:
    close = _f(group, "close")
    low = _f(group, "low")
    shock = _b(group, "bullish_volume_shock_event")
    n = len(group)
    out: list[tuple[int, int]] = []
    for i in range(n):
        if not shock[i] or not np.isfinite(close[i]):
            continue
        level = close[i]
        pullback_at = -1
        for j in range(i + 1, min(i + p.pullback_valid + 1, n)):
            if np.isfinite(low[j]) and low[j] <= level * (1.0 - p.pullback_pct):
                pullback_at = j
                break
        if pullback_at < 0:
            continue
        reclaim_at = -1
        for k in range(pullback_at, min(i + p.pullback_valid + p.reclaim_valid + 1, n)):
            if np.isfinite(close[k]) and close[k] >= level:
                reclaim_at = k
                break
        if reclaim_at < 0:
            continue
        for m in range(reclaim_at + 1, min(reclaim_at + p.fail_valid + 1, n)):
            if np.isfinite(close[m]) and close[m] < level:
                out.append((m, i))
                break
    return out


def detect_s2_extreme_exhaustion(group: pd.DataFrame, p: MotifParams) -> list[tuple[int, int]]:
    close = _f(group, "close")
    open_ = _f(group, "open")
    low = _f(group, "low")
    ret_pct = _f(group, "ret_4h_percentile")
    vol_z = _f(group, "volume_z_4h")
    wick = _f(group, "upper_wick_ratio")
    n = len(group)
    out: list[tuple[int, int]] = []
    for i in range(n):
        if not (np.isfinite(ret_pct[i]) and ret_pct[i] >= p.extreme_pct):
            continue
        if not (np.isfinite(vol_z[i]) and vol_z[i] >= p.shock_z):
            continue
        exhaust_at = -1
        for e in range(i + 1, min(i + p.exhaust_valid + 1, n)):
            rejection = (
                np.isfinite(wick[e])
                and wick[e] >= p.wick_thresh
                and np.isfinite(close[e])
                and np.isfinite(open_[e])
                and close[e] < open_[e]
            )
            if rejection:
                exhaust_at = e
                break
        if exhaust_at < 0:
            continue
        ref_low = low[exhaust_at]
        for bdx in range(exhaust_at + 1, min(exhaust_at + p.breakdown_valid + 1, n)):
            if np.isfinite(close[bdx]) and np.isfinite(ref_low) and close[bdx] < ref_low:
                out.append((bdx, i))
                break
    return out


def detect_s3_crowded_long_unwind(group: pd.DataFrame, p: MotifParams) -> list[tuple[int, int]]:
    close = _f(group, "close")
    low = _f(group, "low")
    funding_pct = _f(group, "funding_percentile")
    oi_pct = _f(group, "oi_value_delta_4h_percentile")
    ret_pct = _f(group, "ret_4h_percentile")
    btc_up = _b(group, "gate_BTC_up")
    n = len(group)
    out: list[tuple[int, int]] = []
    for i in range(p.support_lookback, n):
        crowded = (
            np.isfinite(funding_pct[i]) and funding_pct[i] >= p.funding_hi
            and np.isfinite(oi_pct[i]) and oi_pct[i] >= p.oi_hi
            and np.isfinite(ret_pct[i]) and ret_pct[i] <= p.stall_pct
            and not btc_up[i]
        )
        if not crowded:
            continue
        support = np.nanmin(low[i - p.support_lookback : i + 1])
        if not np.isfinite(support):
            continue
        for bdx in range(i + 1, min(i + p.breakdown_valid + 1, n)):
            if np.isfinite(close[bdx]) and close[bdx] < support:
                out.append((bdx, i))
                break
    return out


def detect_s5_btc_down_breakdown(group: pd.DataFrame, p: MotifParams) -> list[tuple[int, int]]:
    close = _f(group, "close")
    open_ = _f(group, "open")
    high = _f(group, "high")
    low = _f(group, "low")
    btc_down = _b(group, "gate_BTC_down")
    n = len(group)
    out: list[tuple[int, int]] = []
    for i in range(p.btc_breakdown_lookback, n):
        if not btc_down[i]:
            continue
        prior_low = np.nanmin(low[i - p.btc_breakdown_lookback : i])
        if not (np.isfinite(close[i]) and np.isfinite(prior_low) and close[i] < prior_low):
            continue
        bounce_at = -1
        for f_idx in range(i + 1, min(i + p.bounce_valid + 1, n)):
            failed_bounce = (
                np.isfinite(high[f_idx]) and high[f_idx] < high[i]
                and np.isfinite(close[f_idx]) and np.isfinite(open_[f_idx])
                and close[f_idx] < open_[f_idx]
            )
            if failed_bounce:
                bounce_at = f_idx
                break
        if bounce_at < 0:
            continue
        for bdx in range(bounce_at + 1, min(bounce_at + p.bounce_valid + 1, n)):
            if np.isfinite(low[bdx]) and np.isfinite(low[i]) and low[bdx] < low[i]:
                out.append((bdx, i))
                break
    return out


DETECTORS = {
    "S1": detect_s1_failed_reclaim,
    "S2": detect_s2_extreme_exhaustion,
    "S3": detect_s3_crowded_long_unwind,
    "S5": detect_s5_btc_down_breakdown,
}


# --- trade construction ----------------------------------------------------


def _forward_long_return(group: pd.DataFrame, entry_idx: int, exit_idx: int) -> float:
    """Long return over the same window — feeds the 'just don't be long' test."""
    entry = float(group.iloc[entry_idx]["open"])
    exit_close = float(group.iloc[exit_idx]["close"])
    if not np.isfinite(entry) or entry <= 0:
        return np.nan
    return exit_close / entry - 1.0


def _short_trades_from_signals(
    group: pd.DataFrame,
    signals: list[tuple[int, int]],
    cfg: ShortAtlasConfig,
    *,
    role: str,
) -> list[dict[str, object]]:
    """Run the squeeze-aware short simulator over one symbol's signal list."""
    rule = ShortExitRule(cfg.take_profit, cfg.stop_loss, cfg.max_hold_bars)
    rows: list[dict[str, object]] = []
    active_until = -1
    n = len(group)
    for confirmation_idx, anchor_idx in sorted(signals):
        entry_idx = confirmation_idx + 1
        if entry_idx >= n or entry_idx <= active_until:
            continue
        entry_price = float(group.iloc[entry_idx]["open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        exit = simulate_short_exit(group, entry_idx, entry_price, rule)
        signal_row = group.iloc[confirmation_idx]
        rows.append(
            {
                "exchange": str(group.iloc[entry_idx]["exchange"]),
                "symbol": str(group.iloc[entry_idx]["symbol"]),
                "role": role,
                "signal_time": pd.Timestamp(signal_row["feature_time"]),
                "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
                "exit_time": pd.Timestamp(group.iloc[exit.exit_idx]["bar_close_time"]),
                "month": pd.Timestamp(signal_row["feature_time"]).strftime("%Y-%m"),
                "btc_state": str(signal_row.get("btc_market_state", "")),
                "entry_price": entry_price,
                "exit_price": exit.exit_price,
                "gross_return": exit.gross_return,
                "exit_reason": exit.exit_reason,
                "holding_bars": exit.holding_bars,
                "max_adverse_excursion": exit.max_adverse_excursion,
                "max_favorable_excursion": exit.max_favorable_excursion,
                "squeezed": exit.squeezed,
                "forward_long_return": _forward_long_return(group, entry_idx, exit.exit_idx),
            }
        )
        active_until = exit.exit_idx + cfg.cooldown_bars
    return rows


def _eligible_indices(group: pd.DataFrame, cfg: ShortAtlasConfig) -> np.ndarray:
    warmup = _b(group, "warmup_complete") if "warmup_complete" in group.columns else np.ones(len(group), bool)
    rank = _f(group, "dynamic_all_rank")
    eligible = warmup & np.isfinite(rank) & (rank <= cfg.top_n)
    return np.flatnonzero(eligible)


def _matched_random_signals(
    group: pd.DataFrame,
    cfg: ShortAtlasConfig,
    motif_code: str,
    count: int,
) -> list[tuple[int, int]]:
    """Same-symbol random bars, count-matched to the real motif (null distribution)."""
    if count <= 0:
        return []
    pool = _eligible_indices(group, cfg)
    pool = pool[pool < len(group) - 1]
    if len(pool) == 0:
        return []
    symbol = str(group.iloc[0]["symbol"]) if len(group) else ""
    rng = np.random.default_rng(_seed_from(f"{symbol}:{motif_code}:matched_random"))
    take = min(count, len(pool))
    picked = rng.choice(pool, size=take, replace=False)
    return [(int(idx), int(idx)) for idx in picked]


def _entry_only_signals(group: pd.DataFrame, motif_code: str, p: MotifParams, cfg: ShortAtlasConfig) -> list[tuple[int, int]]:
    """The raw trigger WITHOUT the failure confirmation — isolates the gate's lift."""
    n = len(group)
    if motif_code == "S1":
        anchors = np.flatnonzero(_b(group, "bullish_volume_shock_event"))
    elif motif_code == "S2":
        ret_pct = _f(group, "ret_4h_percentile")
        vol_z = _f(group, "volume_z_4h")
        anchors = np.flatnonzero((ret_pct >= p.extreme_pct) & (vol_z >= p.shock_z))
    elif motif_code == "S3":
        funding_pct = _f(group, "funding_percentile")
        oi_pct = _f(group, "oi_value_delta_4h_percentile")
        ret_pct = _f(group, "ret_4h_percentile")
        btc_up = _b(group, "gate_BTC_up")
        anchors = np.flatnonzero(
            (funding_pct >= p.funding_hi) & (oi_pct >= p.oi_hi) & (ret_pct <= p.stall_pct) & (~btc_up)
        )
    elif motif_code == "S5":
        anchors = np.flatnonzero(_b(group, "gate_BTC_down"))
    else:
        anchors = np.array([], dtype=int)
    return [(int(a), int(a)) for a in anchors if a + 1 < n]


def _plain_drop_signals(
    group: pd.DataFrame,
    cfg: ShortAtlasConfig,
    motif_code: str,
    count: int,
) -> list[tuple[int, int]]:
    """Negative control: '普通下跌后做空' — short after a plain ~1h drop, no failure
    structure. Count-matched per symbol so the motif must beat naive breakdown chasing.
    """
    if count <= 0:
        return []
    close = _f(group, "close")
    eligible = set(int(i) for i in _eligible_indices(group, cfg))
    candidates = [
        i for i in range(4, len(group) - 1)
        if i in eligible and np.isfinite(close[i]) and np.isfinite(close[i - 4]) and close[i] < close[i - 4] * 0.99
    ]
    if not candidates:
        return []
    symbol = str(group.iloc[0]["symbol"]) if len(group) else ""
    rng = np.random.default_rng(_seed_from(f"{symbol}:{motif_code}:plain_drop"))
    take = min(count, len(candidates))
    picked = rng.choice(np.array(candidates), size=take, replace=False)
    return [(int(idx), int(idx)) for idx in picked]


# --- per-symbol streaming driver ------------------------------------------


def _collect_symbol_trades(group: pd.DataFrame, cfg: ShortAtlasConfig) -> list[dict[str, object]]:
    group = group.sort_values("bar_open_time").reset_index(drop=True)
    all_rows: list[dict[str, object]] = []
    for motif_code in cfg.motifs:
        params = MOTIF_PARAMS[motif_code]
        real = DETECTORS[motif_code](group, params)
        real_count = len(real)
        roles = {
            f"{motif_code}:real": real,
            f"{motif_code}:entry_only": _entry_only_signals(group, motif_code, params, cfg),
            f"{motif_code}:matched_random": _matched_random_signals(group, cfg, motif_code, real_count),
            f"{motif_code}:plain_drop": _plain_drop_signals(group, cfg, motif_code, real_count),
        }
        for role, signals in roles.items():
            rows = _short_trades_from_signals(group, signals, cfg, role=role)
            for row in rows:
                row["motif"] = motif_code
            all_rows.extend(rows)
    return all_rows


def _stream_short_trades(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
    cfg: ShortAtlasConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= cfg.top_n].copy()
        if data.empty:
            continue
        rows.extend(_collect_symbol_trades(data, cfg))
        if idx % 25 == 0:
            print(f"v1.2s short atlas: {idx}/{len(symbols)} symbols, {len(rows)} trades", flush=True)
    return pd.DataFrame(rows)


# --- aggregation -----------------------------------------------------------


def _net_at_cost(gross: pd.Series, cost: float, cfg: ShortAtlasConfig) -> pd.Series:
    round_trip = 2.0 * (cost + cfg.extra_slippage_bps) / 10_000.0
    return pd.to_numeric(gross, errors="coerce") - round_trip


def _month_capped_total(sample: pd.DataFrame, net_col: str) -> float:
    if sample.empty:
        return np.nan
    total = pd.to_numeric(sample[net_col], errors="coerce").sum()
    cap_value = total * MONTH_CAP if total > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", sort=False, dropna=False):
        value = pd.to_numeric(group[net_col], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped))


def _max_symbol_share(sample: pd.DataFrame, net_col: str) -> float:
    if sample.empty:
        return np.nan
    grouped = pd.to_numeric(sample[net_col], errors="coerce").groupby(sample["symbol"], sort=False).sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _candidate_summary(trades: pd.DataFrame, cfg: ShortAtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    real = trades[trades["role"].astype(str).str.endswith(":real")]
    for motif_code in cfg.motifs:
        sample = real[real["motif"].eq(motif_code)].copy()
        if sample.empty:
            continue
        gross = pd.to_numeric(sample["gross_return"], errors="coerce")
        mae = pd.to_numeric(sample["max_adverse_excursion"], errors="coerce")
        mfe = pd.to_numeric(sample["max_favorable_excursion"], errors="coerce")
        squeezed = sample["squeezed"].fillna(False).astype(bool)
        fwd_long = pd.to_numeric(sample["forward_long_return"], errors="coerce")
        avoided_long_loss = (-fwd_long).clip(lower=0.0)
        for cost in COST_GRID_BPS:
            net = _net_at_cost(sample["gross_return"], cost, cfg)
            sample[f"_net_{int(cost)}"] = net
            entry_only = trades[trades["role"].eq(f"{motif_code}:entry_only")]
            entry_only_net = _net_at_cost(entry_only["gross_return"], cost, cfg).mean() if not entry_only.empty else np.nan
            rows.append(
                {
                    "motif": motif_code,
                    "name": MOTIF_PARAMS[motif_code].name,
                    "cost_single_side_bps": cost,
                    "extra_slippage_bps": cfg.extra_slippage_bps,
                    "trades": int(len(sample)),
                    "short_net": float(net.mean()),
                    "short_net_total": float(net.sum()),
                    "gross_mean": float(gross.mean()),
                    "hit_down_3pct_rate": float((mfe <= -0.03).mean()),
                    "hit_down_5pct_rate": float((mfe <= -0.05).mean()),
                    "squeeze_out_rate": float(squeezed.mean()),
                    "avg_max_adverse_excursion": float(mae.mean()),
                    "avg_max_favorable_excursion": float(mfe.mean()),
                    "win_rate": float((net > 0).mean()),
                    "baseline_lift_vs_entry_only": float(net.mean() - entry_only_net)
                    if np.isfinite(entry_only_net)
                    else np.nan,
                    "month_cap_net_total": _month_capped_total(
                        sample.assign(_net=net.to_numpy()), "_net"
                    ),
                    "max_symbol_contribution": _max_symbol_share(sample.assign(_net=net.to_numpy()), "_net"),
                    "avg_short_net_minus_avoided_long_loss": float(net.mean() - avoided_long_loss.mean()),
                    "avoided_long_loss_mean": float(avoided_long_loss.mean()),
                }
            )
    return pd.DataFrame(rows)


def _baseline_comparison(trades: pd.DataFrame, cfg: ShortAtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    role_kinds = ["real", "entry_only", "matched_random", "plain_drop"]
    for motif_code in cfg.motifs:
        nets: dict[str, float] = {}
        counts: dict[str, int] = {}
        for kind in role_kinds:
            sample = trades[trades["role"].eq(f"{motif_code}:{kind}")]
            counts[kind] = int(len(sample))
            nets[kind] = float(_net_at_cost(sample["gross_return"], FOCAL_COST, cfg).mean()) if not sample.empty else np.nan
        real_net = nets.get("real", np.nan)
        rows.append(
            {
                "motif": motif_code,
                "name": MOTIF_PARAMS[motif_code].name,
                "cost_single_side_bps": FOCAL_COST,
                "real_trades": counts.get("real", 0),
                "real_net20": real_net,
                "entry_only_net20": nets.get("entry_only", np.nan),
                "matched_random_net20": nets.get("matched_random", np.nan),
                "plain_drop_net20": nets.get("plain_drop", np.nan),
                "lift_vs_entry_only": real_net - nets.get("entry_only", np.nan),
                "lift_vs_matched_random": real_net - nets.get("matched_random", np.nan),
                "lift_vs_plain_drop": real_net - nets.get("plain_drop", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _regime_split(trades: pd.DataFrame, cfg: ShortAtlasConfig) -> pd.DataFrame:
    real = trades[trades["role"].astype(str).str.endswith(":real")]
    rows: list[dict[str, object]] = []
    for motif_code in cfg.motifs:
        sample = real[real["motif"].eq(motif_code)]
        for regime in REGIMES:
            regime_sample = sample[sample["btc_state"].eq(regime)]
            net = _net_at_cost(regime_sample["gross_return"], FOCAL_COST, cfg)
            rows.append(
                {
                    "motif": motif_code,
                    "btc_state": regime,
                    "trades": int(len(regime_sample)),
                    "short_net20": float(net.mean()) if len(net) else np.nan,
                    "squeeze_out_rate": float(regime_sample["squeezed"].fillna(False).astype(bool).mean())
                    if len(regime_sample)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _month_cap_table(trades: pd.DataFrame, cfg: ShortAtlasConfig) -> pd.DataFrame:
    real = trades[trades["role"].astype(str).str.endswith(":real")]
    rows: list[dict[str, object]] = []
    for motif_code in cfg.motifs:
        sample = real[real["motif"].eq(motif_code)].copy()
        if sample.empty:
            continue
        net = _net_at_cost(sample["gross_return"], FOCAL_COST, cfg)
        sample = sample.assign(_net=net.to_numpy())
        raw_total = float(net.sum())
        capped_total = _month_capped_total(sample, "_net")
        rows.append(
            {
                "motif": motif_code,
                "cost_single_side_bps": FOCAL_COST,
                "raw_net_total": raw_total,
                "month_capped_net_total": capped_total,
                "cap_haircut": raw_total - capped_total,
                "months": int(sample["month"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _symbol_contribution(trades: pd.DataFrame, cfg: ShortAtlasConfig) -> pd.DataFrame:
    real = trades[trades["role"].astype(str).str.endswith(":real")]
    rows: list[dict[str, object]] = []
    for motif_code in cfg.motifs:
        sample = real[real["motif"].eq(motif_code)].copy()
        if sample.empty:
            continue
        net = _net_at_cost(sample["gross_return"], FOCAL_COST, cfg)
        sample = sample.assign(_net=net.to_numpy())
        grouped = sample.groupby("symbol", sort=False)["_net"].agg(["sum", "count"])
        total = grouped["sum"].sum()
        for symbol, row in grouped.iterrows():
            rows.append(
                {
                    "motif": motif_code,
                    "symbol": symbol,
                    "trades": int(row["count"]),
                    "net20_total": float(row["sum"]),
                    "contribution_share": float(row["sum"] / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["motif", "net20_total"], ascending=[True, False])


# --- notes -----------------------------------------------------------------


def _write_notes(
    report_root: Path,
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    regime: pd.DataFrame,
    cfg: ShortAtlasConfig,
) -> None:
    lines = [
        "# v1.2s Short Motif Atlas",
        "",
        "Phase-1 failure-path research for the short side. Tier: research only — no",
        "shadow, paper-live, or real-live wiring. Following the short instruction doc:",
        "shorts are strong->weak failure unwinds, not inverted longs, and squeeze",
        "avoidance dominates. A motif is only interesting if it beats ALL of:",
        "entry-only, matched-random, and plain-drop (普通下跌后做空) baselines, survives",
        "month-cap + symbol-contribution, and is not dominated by simply not being long.",
        "",
        f"- cost grid: {', '.join(f'{int(c)}bp' for c in COST_GRID_BPS)} + {cfg.extra_slippage_bps:.0f}bp short slippage add-on",
        f"- short exit: tp={cfg.take_profit:.1%} (down), stop={cfg.stop_loss:.1%} (up, squeeze), max_hold={cfg.max_hold_bars} bars",
        f"- universe: Top{cfg.top_n} liquidity only (doc §5)",
        "",
    ]
    focal = summary[summary["cost_single_side_bps"].eq(FOCAL_COST)] if not summary.empty else pd.DataFrame()
    if not focal.empty:
        lines.append("## Motif scorecard (20bp + short slippage)")
        for row in focal.sort_values("short_net", ascending=False).itertuples(index=False):
            verdict = _motif_verdict(row, baselines)
            lines.append(
                f"- **{row.motif} {row.name}**: trades={row.trades}, short_net={row.short_net:.4%}, "
                f"win={row.win_rate:.1%}, hit_down_3pct={row.hit_down_3pct_rate:.1%}, "
                f"squeeze_out={row.squeeze_out_rate:.1%}, MAE={row.avg_max_adverse_excursion:.2%}, "
                f"lift_vs_entry_only={row.baseline_lift_vs_entry_only:+.4%}. {verdict}"
            )
        lines.append("")
    if not baselines.empty:
        lines.append("## Baseline comparison (real vs controls, 20bp)")
        for row in baselines.itertuples(index=False):
            lines.append(
                f"- {row.motif}: real={row.real_net20:.4%} | entry_only={row.entry_only_net20:.4%} "
                f"| matched_random={row.matched_random_net20:.4%} | plain_drop={row.plain_drop_net20:.4%} "
                f"(lift vs plain_drop={row.lift_vs_plain_drop:+.4%})."
            )
        lines.append("")
    if not focal.empty:
        lines.append("## Short vs just-not-being-long (doc §8, the decisive test)")
        for row in focal.sort_values("avg_short_net_minus_avoided_long_loss", ascending=False).itertuples(index=False):
            better = row.avg_short_net_minus_avoided_long_loss > 0
            lines.append(
                f"- {row.motif}: short_net={row.short_net:.4%} vs avoided_long_loss={row.avoided_long_loss_mean:.4%} "
                f"-> {'shorting adds edge beyond risk-off' if better else 'best use is long risk-off, not shorting'}."
            )
        lines.append("")
    if not regime.empty:
        lines.append("## Regime split (squeeze sanity)")
        for motif_code in cfg.motifs:
            motif_rows = regime[regime["motif"].eq(motif_code)]
            cells = ", ".join(
                f"{r.btc_state}: net={r.short_net20:.3%}/sq={r.squeeze_out_rate:.0%}/n={r.trades}"
                for r in motif_rows.itertuples(index=False)
            )
            lines.append(f"- {motif_code}: {cells}")
        lines.append("")
    lines.extend(
        [
            "## Discipline",
            "- Every entry uses only bars with feature_time <= the decision bar (strict as-of).",
            "- Matched-random and plain-drop baselines are count-matched per symbol with a",
            "  deterministic crc32 seed, so the null distributions reproduce.",
            "- A motif with positive short_net but no lift over plain_drop is breakdown-chasing,",
            "  not failure-edge; it does not advance to shadow.",
            "- No paper-live / real-live permission changes.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _motif_verdict(row, baselines: pd.DataFrame) -> str:
    base = baselines[baselines["motif"].eq(row.motif)]
    if base.empty:
        return ""
    b = base.iloc[0]
    if not np.isfinite(row.short_net) or row.short_net <= 0:
        return "No standalone short edge."
    if not (b["lift_vs_plain_drop"] > 0 and b["lift_vs_matched_random"] > 0 and b["lift_vs_entry_only"] > 0):
        return "Positive but does not beat all baselines — likely breakdown-chasing."
    return "Beats all baselines — candidate for deeper study (still research-only)."


# --- entry point -----------------------------------------------------------


def write_v12s_short_motif_atlas(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: ShortAtlasConfig = ShortAtlasConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    trades = _stream_short_trades(feature_path, rank30, rank90, symbols, config, cfg)

    summary = _candidate_summary(trades, cfg)
    baselines = _baseline_comparison(trades, cfg)
    regime = _regime_split(trades, cfg)
    month_cap = _month_cap_table(trades, cfg)
    symbol_contribution = _symbol_contribution(trades, cfg)

    outputs = {
        "candidate_summary": report_root / "candidate_summary.csv",
        "baseline_comparison": report_root / "baseline_comparison.csv",
        "regime_split": report_root / "regime_split.csv",
        "month_cap": report_root / "month_cap.csv",
        "symbol_contribution": report_root / "symbol_contribution.csv",
        "trades": report_root / "short_motif_trades.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    baselines.to_csv(outputs["baseline_comparison"], index=False)
    regime.to_csv(outputs["regime_split"], index=False)
    month_cap.to_csv(outputs["month_cap"], index=False)
    symbol_contribution.to_csv(outputs["symbol_contribution"], index=False)
    trades.to_csv(outputs["trades"], index=False)
    _write_notes(report_root, summary, baselines, regime, cfg)
    return outputs


__all__ = [
    "COST_GRID_BPS",
    "MOTIF_PARAMS",
    "REPORT_ROOT",
    "ShortAtlasConfig",
    "detect_s1_failed_reclaim",
    "detect_s2_extreme_exhaustion",
    "detect_s3_crowded_long_unwind",
    "detect_s5_btc_down_breakdown",
    "write_v12s_short_motif_atlas",
]
