"""v3.4 True Short Sleeve — narrow, "rare-but-clean" short windows.

Source: short_instructment3(v3.4).docx. The v1.2s atlas already proved there is
no standalone short edge in the raw failure motifs (value was long risk-off,
not shorting). v3.4 asks the next question: after the failure event, is there a
*small* sub-population where a real short pays — failed-bounce confirmed, market
unsupportive, squeeze risk visibly capped — that justifies a tiny short sleeve?

Three candidate sleeves (each tested in two parameter forms = 6 sleeves total):

- SS1 Failed Reclaim Breakdown Short:
    S1/S3/S5 failure -> attempted reclaim -> reclaim failed -> price breaks
    below the failed-reclaim low. Gates: BTC_not_up (SS1A) or low_coimpulse
    (SS1B). The point is to wait for the *breakdown confirmation*, not short
    the failure itself (which is just shorting at the top of a strong name).

- SS2 Crowded Long Stall -> Unwind Short:
    High funding + high OI + stalled price + failed reclaim + (BTC_down |
    low_coimpulse) + volume shock losing follow-through -> short. The S3 motif
    already captures the crowded-long stall; SS2 adds the failed-followthrough
    + macro context so we don't short in strong trend mid-leg.

- SS3 CIC Failure Short:
    The reverse of CP60. A CIC candidate (CIC1_beta_extreme / CIC2_beta_broad)
    that fails to follow through within ~1h and would be exited by CP60 — is
    that weak-position name also a short? SS3A uses entry as the breakdown
    reference; SS3B uses the pullback low and requires no Protect_A.

Pass-criteria (docx §6) — every sleeve must clear *all* 10 hurdles, with the
decisive comparison being short_vs_no_long: if no-long cooldown beats opening
short, the signal stays as risk-off, not a short.

Tier: research only. No shadow / paper-live / real-live wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.backtest.short_execution import ShortExitRule, simulate_short_exit
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v12s_short_motif_atlas import (
    DETECTORS as MOTIF_DETECTORS,
    MOTIF_PARAMS,
    _b,
    _f,
)

# Deferred imports — pulled into the streaming driver only, so unit tests that
# build a synthetic ``group`` frame can exercise gates / signal emission / 3-
# action compare without dragging in the heavier feature stack (and the local
# numpy/scipy ABI mismatch that breaks v06a1 in the dev env).


REPORT_ROOT = Path("reports/v3_4_true_short_sleeve")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
LONG_POOL_NAME = "P2_CIC1_CIC2_COMBINED"

# Docx §5: fixed short execution rules — no per-sleeve optimization.
FAST_RULE = ShortExitRule(take_profit=0.03, stop_loss=0.02, max_hold_bars=16)
SWING_RULE = ShortExitRule(take_profit=0.05, stop_loss=0.03, max_hold_bars=48)

COST_GRID_BPS = (10.0, 20.0, 30.0, 50.0)
EXTRA_SLIPPAGE_GRID_BPS = (5.0, 10.0)
FOCAL_COST_BPS = 20.0
FOCAL_EXTRA_SLIPPAGE_BPS = 5.0
MONTH_CAP = 0.35  # docx §6.4
MIN_SAMPLES = 50  # docx §6.9


@dataclass(frozen=True)
class SleeveSpec:
    """A v3.4 short sleeve definition.

    The sleeve fires a short entry when *all* of these hold:
      1. one of ``source_motifs`` confirms at bar i (or, for SS3, a CIC
         candidate exists and its CP60 exit gate is met),
      2. the breakdown confirmation walker finds ``close < reference_low``
         within ``breakdown_valid_bars`` after the motif confirmation,
      3. every gate in ``gates`` passes at the breakdown bar.

    Gate names are resolved by ``_GATE_REGISTRY``. An unrecognised or missing
    gate fails closed (no entry) — better to skip than to fake-pass.
    """

    code: str
    name: str
    description: str
    source: str  # "motif" or "cic_candidate"
    source_motifs: tuple[str, ...] = ()
    breakdown_reference: str = "reclaim_low"  # "reclaim_low" | "pullback_low" | "entry" | "anchor_low"
    breakdown_valid_bars: int = 8
    gates: tuple[str, ...] = ()
    cooldown_bars: int = 16  # docx §3.cooldown after a sleeve fire (per symbol).


SLEEVES: tuple[SleeveSpec, ...] = (
    SleeveSpec(
        code="SS1A",
        name="failed_reclaim_breakdown_btc_not_up",
        description="S1/S3/S5 failure + failed reclaim breakdown + BTC_not_up",
        source="motif",
        source_motifs=("S1", "S3", "S5"),
        breakdown_reference="reclaim_low",
        gates=("btc_not_up",),
    ),
    SleeveSpec(
        code="SS1B",
        name="failed_reclaim_breakdown_low_coimpulse",
        description="S1/S3/S5 failure + failed reclaim breakdown + low co-impulse density",
        source="motif",
        source_motifs=("S1", "S3", "S5"),
        breakdown_reference="reclaim_low",
        gates=("low_coimpulse",),
    ),
    SleeveSpec(
        code="SS2A",
        name="crowded_long_stall_btc_down",
        description="Crowded long + price stall + failed reclaim + BTC_down",
        source="motif",
        source_motifs=("S3",),
        breakdown_reference="reclaim_low",
        gates=("btc_down", "price_stall", "failed_followthrough"),
    ),
    SleeveSpec(
        code="SS2B",
        name="crowded_long_stall_low_coimpulse_breakdown",
        description="Crowded long + price stall + low co-impulse + breakdown",
        source="motif",
        source_motifs=("S3",),
        breakdown_reference="reclaim_low",
        gates=("low_coimpulse", "price_stall", "failed_followthrough"),
    ),
    SleeveSpec(
        code="SS3A",
        name="cic_failure_breakdown_density_fading",
        description="CIC candidate + CP60-exit + breakdown below entry + density fading",
        source="cic_candidate",
        breakdown_reference="entry",
        breakdown_valid_bars=12,
        gates=("cp60_would_exit", "density_fading"),
    ),
    SleeveSpec(
        code="SS3B",
        name="cic_failure_breakdown_no_protect_a",
        description="CIC candidate + CP60-exit + breakdown below pullback low + no Protect_A",
        source="cic_candidate",
        breakdown_reference="pullback_low",
        breakdown_valid_bars=12,
        gates=("cp60_would_exit", "no_protect_a"),
    ),
)


@dataclass(frozen=True)
class V34Config:
    """v3.4 driver config — research run with the docx-mandated grids."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    top_n: int = 30
    long_pool_name: str = LONG_POOL_NAME
    sleeves: tuple[SleeveSpec, ...] = field(default_factory=lambda: SLEEVES)
    fast_rule: ShortExitRule = FAST_RULE
    swing_rule: ShortExitRule = SWING_RULE
    cost_grid_bps: tuple[float, ...] = COST_GRID_BPS
    extra_slippage_grid_bps: tuple[float, ...] = EXTRA_SLIPPAGE_GRID_BPS
    focal_cost_bps: float = FOCAL_COST_BPS
    focal_extra_slippage_bps: float = FOCAL_EXTRA_SLIPPAGE_BPS
    month_cap: float = MONTH_CAP
    min_samples: int = MIN_SAMPLES
    no_long_cooldown_bars: int = 16  # 4h — same as Fast short max_hold, the natural unit.
    coimpulse_low_percentile: float = 40.0  # below this -> "low co-impulse"
    density_fading_lookback: int = 4
    cp60_window_bars: int = 60
    cp60_exit_threshold: float = 0.005  # CP60 weak position: |ret| < 0.5% over 1h after entry
    follow_through_volume_z: float = 2.0
    follow_through_ret_max: float = 0.005


# --------------------------------------------------------------------------------------
# Gate registry — each gate is a function(group, idx, cfg) -> bool.
# Missing source columns fail-closed (no entry). Keep these intentionally thin so
# they read column-by-column rather than building heavy frames per-call.
# --------------------------------------------------------------------------------------


def _gate_btc_not_up(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    btc_up = _b(group, "gate_BTC_up")
    return idx < len(btc_up) and not bool(btc_up[idx])


def _gate_btc_down(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    btc_down = _b(group, "gate_BTC_down")
    return idx < len(btc_down) and bool(btc_down[idx])


def _gate_low_coimpulse(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    """Co-impulse / market-density is below the configured percentile.

    Uses ``co_impulse_density`` (graph P0a density) if present, else falls back
    to ``volume_impulse_density``. Threshold is a percentile across the symbol's
    own history up to ``idx`` so the gate stays meaningful on universes where
    one symbol dominates density. Missing column -> gate fails closed.
    """
    for col in ("co_impulse_density", "volume_impulse_density"):
        if col in group.columns:
            series = pd.to_numeric(group[col], errors="coerce")
            value = series.iloc[idx]
            if not np.isfinite(value):
                return False
            window = series.iloc[: idx + 1].dropna()
            if len(window) < 8:
                return False
            threshold = float(np.percentile(window, cfg.coimpulse_low_percentile))
            return bool(float(value) <= threshold)
    return False


def _gate_price_stall(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    ret_pct = _f(group, "ret_4h_percentile")
    if idx >= len(ret_pct):
        return False
    return bool(np.isfinite(ret_pct[idx]) and ret_pct[idx] <= 55.0)  # docx §SS2 stall threshold.


def _gate_failed_followthrough(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    """High volume but small price move — the volume shock did not pay."""
    vol_z = _f(group, "volume_z_4h")
    ret_1h = _f(group, "ret_1h")
    if idx >= len(vol_z) or idx >= len(ret_1h):
        return False
    return bool(
        np.isfinite(vol_z[idx])
        and np.isfinite(ret_1h[idx])
        and vol_z[idx] >= cfg.follow_through_volume_z
        and abs(ret_1h[idx]) <= cfg.follow_through_ret_max
    )


def _gate_density_fading(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    """Density at idx is below the density window-lookback average — fading."""
    for col in ("co_impulse_density", "volume_impulse_density"):
        if col in group.columns:
            series = pd.to_numeric(group[col], errors="coerce")
            value = series.iloc[idx]
            window_start = max(0, idx - cfg.density_fading_lookback)
            window = series.iloc[window_start:idx]
            avg = window.mean()
            if not (np.isfinite(value) and np.isfinite(avg)):
                return False
            return bool(value < avg)
    return False


def _gate_no_protect_a(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    """``Protect_A`` is the live high-beta protection flag. If it is set, the
    long is already shielded; opening a short here would over-stack the bet.
    Missing column means Protect_A was not in the feature build — gate passes
    (no protection to fight against), but this is logged via ``audit_reason``.
    """
    if "gate_Protect_A" in group.columns:
        flags = _b(group, "gate_Protect_A")
        return idx < len(flags) and not bool(flags[idx])
    return True


def _gate_cp60_would_exit(group: pd.DataFrame, idx: int, cfg: V34Config) -> bool:
    """CP60 fires when a name's |return| over the last hour is below threshold.

    A "would-exit" gate: the long has been weak-stagnant for the CP60 window.
    Computed locally (4 bars of 15m = 1h proxy). Missing close column fails closed.
    """
    close = _f(group, "close")
    if idx < 4 or idx >= len(close):
        return False
    entry = close[idx - 4]
    current = close[idx]
    if not (np.isfinite(entry) and np.isfinite(current) and entry > 0):
        return False
    return bool(abs((current / entry) - 1.0) <= cfg.cp60_exit_threshold)


_GATE_REGISTRY: dict[str, Callable[[pd.DataFrame, int, V34Config], bool]] = {
    "btc_not_up": _gate_btc_not_up,
    "btc_down": _gate_btc_down,
    "low_coimpulse": _gate_low_coimpulse,
    "price_stall": _gate_price_stall,
    "failed_followthrough": _gate_failed_followthrough,
    "density_fading": _gate_density_fading,
    "no_protect_a": _gate_no_protect_a,
    "cp60_would_exit": _gate_cp60_would_exit,
}


def _apply_gates(group: pd.DataFrame, idx: int, sleeve: SleeveSpec, cfg: V34Config) -> tuple[bool, str]:
    """Evaluate every gate in ``sleeve.gates``. Returns (passed, fail_reason)."""
    for gate_name in sleeve.gates:
        fn = _GATE_REGISTRY.get(gate_name)
        if fn is None:
            return False, f"unknown_gate:{gate_name}"
        if not fn(group, idx, cfg):
            return False, f"gate_failed:{gate_name}"
    return True, ""


# --------------------------------------------------------------------------------------
# Motif streaming + breakdown confirmation walker
# --------------------------------------------------------------------------------------


def _motif_anchors(group: pd.DataFrame, motif_code: str) -> list[tuple[int, int]]:
    """Run a v1.2s motif detector and return its (confirmation_idx, anchor_idx) list."""
    detector = MOTIF_DETECTORS.get(motif_code)
    if detector is None:
        return []
    params = MOTIF_PARAMS[motif_code]
    return detector(group, params)


def _reference_low_for(
    group: pd.DataFrame,
    anchor_idx: int,
    confirmation_idx: int,
    reference: str,
) -> tuple[float, int]:
    """The low price that the breakdown must close below.

    Returns (level, reference_bar). For sleeves that don't have a natural low
    (e.g. SS3 with reference=entry), the entry price is used.
    """
    low = _f(group, "low")
    close = _f(group, "close")
    if reference == "reclaim_low":
        # The lowest low between the failure anchor and the confirmation bar.
        window = low[anchor_idx : confirmation_idx + 1]
        level = float(np.nanmin(window)) if window.size else float("nan")
        ref_bar = anchor_idx + int(np.nanargmin(window)) if window.size else confirmation_idx
    elif reference == "pullback_low":
        # The lowest low between confirmation and 3 bars after (pullback after failure).
        end = min(confirmation_idx + 4, len(group))
        window = low[confirmation_idx:end]
        level = float(np.nanmin(window)) if window.size else float("nan")
        ref_bar = confirmation_idx + int(np.nanargmin(window)) if window.size else confirmation_idx
    elif reference == "entry":
        # The confirmation bar's close itself — anything below is a "below entry".
        level = float(close[confirmation_idx]) if confirmation_idx < len(close) else float("nan")
        ref_bar = confirmation_idx
    else:
        level = float("nan")
        ref_bar = confirmation_idx
    return level, ref_bar


def _find_breakdown(
    group: pd.DataFrame,
    start_idx: int,
    reference_level: float,
    valid_bars: int,
) -> int:
    """Walk forward; return the first bar index where close < reference_level, else -1."""
    close = _f(group, "close")
    n = len(close)
    if not np.isfinite(reference_level):
        return -1
    end = min(start_idx + valid_bars + 1, n)
    for j in range(start_idx + 1, end):
        if np.isfinite(close[j]) and close[j] < reference_level:
            return j
    return -1


def _emit_motif_sleeve_signals(
    group: pd.DataFrame,
    sleeve: SleeveSpec,
    cfg: V34Config,
) -> list[dict[str, object]]:
    """Combine motif confirmations + breakdown walker + gates -> short entry rows."""
    rows: list[dict[str, object]] = []
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    n = len(group)
    if sleeve.source != "motif":
        return rows
    active_until = -1
    for motif_code in sleeve.source_motifs:
        for confirmation_idx, anchor_idx in _motif_anchors(group, motif_code):
            if confirmation_idx >= n - 1:
                continue
            level, ref_bar = _reference_low_for(group, anchor_idx, confirmation_idx, sleeve.breakdown_reference)
            break_idx = _find_breakdown(group, confirmation_idx, level, sleeve.breakdown_valid_bars)
            if break_idx < 0:
                continue
            passed, fail_reason = _apply_gates(group, break_idx, sleeve, cfg)
            if not passed:
                continue
            entry_idx = break_idx + 1
            if entry_idx >= n or entry_idx <= active_until:
                continue
            rows.append(
                _signal_row(
                    group,
                    sleeve,
                    motif_code=motif_code,
                    anchor_idx=anchor_idx,
                    confirmation_idx=confirmation_idx,
                    break_idx=break_idx,
                    entry_idx=entry_idx,
                    reference_level=level,
                    feature_time=feature_time,
                    bar_open_time=bar_open_time,
                )
            )
            active_until = entry_idx + sleeve.cooldown_bars
    return rows


def _signal_row(
    group: pd.DataFrame,
    sleeve: SleeveSpec,
    *,
    motif_code: str,
    anchor_idx: int,
    confirmation_idx: int,
    break_idx: int,
    entry_idx: int,
    reference_level: float,
    feature_time: pd.Series,
    bar_open_time: pd.Series,
) -> dict[str, object]:
    """Pack one short signal — execution + 3-action attach happens downstream."""
    row = group.iloc[entry_idx]
    return {
        "sleeve_code": sleeve.code,
        "sleeve_name": sleeve.name,
        "exchange": str(row.get("exchange", "")),
        "symbol": str(row.get("symbol", "")),
        "motif_code": motif_code,
        "anchor_idx": int(anchor_idx),
        "confirmation_idx": int(confirmation_idx),
        "break_idx": int(break_idx),
        "entry_idx": int(entry_idx),
        "anchor_feature_time": feature_time.iloc[anchor_idx] if anchor_idx < len(feature_time) else pd.NaT,
        "signal_time": feature_time.iloc[break_idx] if break_idx < len(feature_time) else pd.NaT,
        "entry_time": bar_open_time.iloc[entry_idx] if entry_idx < len(bar_open_time) else pd.NaT,
        "reference_low": float(reference_level),
        "month": (
            feature_time.iloc[break_idx].strftime("%Y-%m")
            if break_idx < len(feature_time) and pd.notna(feature_time.iloc[break_idx])
            else ""
        ),
        "btc_state": str(row.get("btc_market_state", "")),
    }


# --------------------------------------------------------------------------------------
# CIC-candidate sleeve (SS3)
# --------------------------------------------------------------------------------------


def _emit_cic_sleeve_signals(
    group: pd.DataFrame,
    sleeve: SleeveSpec,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V34Config,
) -> list[dict[str, object]]:
    """SS3: spawn short candidates from long-side CIC entries that fail follow-through.

    For each long entry in ``cic_long_index`` matching this symbol, we look for
    the breakdown reference (entry close or pullback low) within
    ``breakdown_valid_bars`` and require ``cp60_would_exit`` + the sleeve gates.

    The long index is built once from the trade cache so each symbol's pass
    here is O(symbol bars + cic entries on this symbol).
    """
    rows: list[dict[str, object]] = []
    if sleeve.source != "cic_candidate" or not cic_long_index:
        return rows
    symbol = str(group["symbol"].iloc[0]) if len(group) else ""
    if not symbol:
        return rows
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    n = len(group)
    feature_ns = feature_time.astype("int64").to_numpy()
    active_until = -1
    for (sym_key, ts_key), payload in cic_long_index.items():
        if sym_key != symbol:
            continue
        anchor_ns = int(pd.Timestamp(ts_key).value)
        confirmation_idx = int(np.searchsorted(feature_ns, anchor_ns, side="left"))
        if confirmation_idx >= n - 1:
            continue
        ref_low = float(payload.get("pullback_low", float("nan")))
        if sleeve.breakdown_reference == "entry":
            close = _f(group, "close")
            level = float(close[confirmation_idx])
        else:
            level = ref_low if np.isfinite(ref_low) else (
                float(np.nanmin(_f(group, "low")[confirmation_idx : confirmation_idx + 4]))
            )
        break_idx = _find_breakdown(group, confirmation_idx, level, sleeve.breakdown_valid_bars)
        if break_idx < 0:
            continue
        passed, fail_reason = _apply_gates(group, break_idx, sleeve, cfg)
        if not passed:
            continue
        entry_idx = break_idx + 1
        if entry_idx >= n or entry_idx <= active_until:
            continue
        rows.append(
            _signal_row(
                group,
                sleeve,
                motif_code=str(payload.get("candidate", "CIC")),
                anchor_idx=confirmation_idx,
                confirmation_idx=confirmation_idx,
                break_idx=break_idx,
                entry_idx=entry_idx,
                reference_level=level,
                feature_time=feature_time,
                bar_open_time=bar_open_time,
            )
        )
        active_until = entry_idx + sleeve.cooldown_bars
    return rows


def _build_cic_long_index(trade_cache: pd.DataFrame, cfg: V34Config) -> dict[tuple[str, pd.Timestamp], dict]:
    """Index v0.9D long trades by (symbol, signal_time) -> {candidate, entry, exit, pullback_low}.

    Used by SS3 to seed short candidates and by ``_three_action_compare`` to
    match A/B/C actions against the actual long P&L. Empty cache -> empty dict.
    """
    from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool  # deferred

    if trade_cache.empty:
        return {}
    pool = _focus_pool(trade_cache, cfg.long_pool_name)
    if pool.empty:
        return {}
    pool = pool.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["signal_time"])
    index: dict[tuple[str, pd.Timestamp], dict] = {}
    for row in pool.itertuples(index=False):
        symbol = str(getattr(row, "symbol", ""))
        ts = pd.Timestamp(getattr(row, "signal_time"))
        if not symbol or pd.isna(ts):
            continue
        index[(symbol, ts)] = {
            "candidate": str(getattr(row, "candidate", "")),
            "net_return": float(getattr(row, "net_return", float("nan"))),
            "entry_time": getattr(row, "entry_time", pd.NaT),
            "exit_time": getattr(row, "exit_time", pd.NaT),
            "pullback_low": float(getattr(row, "pullback_low", float("nan")))
            if hasattr(row, "pullback_low")
            else float("nan"),
            "month": pd.Timestamp(getattr(row, "signal_time")).strftime("%Y-%m"),
        }
    return index


# --------------------------------------------------------------------------------------
# Short execution (Fast + Swing) per signal
# --------------------------------------------------------------------------------------


def _execute_signal(group: pd.DataFrame, signal: dict, rule: ShortExitRule, label: str) -> dict:
    """Run one short execution for one signal row. Returns enriched row."""
    entry_idx = int(signal["entry_idx"])
    if entry_idx >= len(group):
        return {**signal, "execution": label, "gross_return": np.nan, "squeezed": False}
    entry_price = float(group.iloc[entry_idx]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {**signal, "execution": label, "gross_return": np.nan, "squeezed": False}
    result = simulate_short_exit(group, entry_idx, entry_price, rule)
    return {
        **signal,
        "execution": label,
        "entry_price": entry_price,
        "exit_price": result.exit_price,
        "exit_reason": result.exit_reason,
        "gross_return": float(result.gross_return),
        "max_adverse_excursion": float(result.max_adverse_excursion),
        "max_favorable_excursion": float(result.max_favorable_excursion),
        "squeezed": bool(result.squeezed),
        "holding_bars": int(result.holding_bars),
        "exit_time": (
            pd.to_datetime(group.iloc[result.exit_idx].get("bar_close_time", pd.NaT), utc=True, errors="coerce")
        ),
    }


def _net_at_cost(gross: pd.Series, cost_bps: float, extra_slippage_bps: float) -> pd.Series:
    """Round-trip net = gross - 2 * (cost + slippage) bp."""
    round_trip = 2.0 * (cost_bps + extra_slippage_bps) / 10_000.0
    return pd.to_numeric(gross, errors="coerce") - round_trip


# --------------------------------------------------------------------------------------
# 3-action comparison: A no-action / B no-long cooldown / C open-short
# --------------------------------------------------------------------------------------


def _three_action_compare(
    signals_df: pd.DataFrame,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V34Config,
) -> pd.DataFrame:
    """For every signal, attach long-action counterfactuals A and B alongside C.

    A no_action_net: the realized long net20 of any long at (sym, t_signal..t_signal+cd]
    B no_long_net: -A_net (avoided loss / sacrificed gain) — same magnitude flipped
    C short_net: short execution P&L at the focal cost grid

    Per the docx §7 decisive test:
      if portfolio(B) > portfolio(C)  -> signal value is risk-off, not shorting
      if portfolio(C) > portfolio(B)  -> sleeve has true short value
    """
    if signals_df.empty:
        return signals_df.copy()
    cd_ns = int(pd.Timedelta(minutes=15 * cfg.no_long_cooldown_bars).value)
    # Group cic_long_index by symbol for fast lookup.
    by_symbol: dict[str, list[tuple[int, dict]]] = {}
    for (sym, ts), payload in cic_long_index.items():
        if pd.notna(ts):
            by_symbol.setdefault(sym, []).append((int(pd.Timestamp(ts).value), payload))
    for sym in list(by_symbol.keys()):
        by_symbol[sym].sort(key=lambda kv: kv[0])
    no_action_net: list[float] = []
    no_long_net: list[float] = []
    long_match_count: list[int] = []
    for row in signals_df.itertuples(index=False):
        sym = str(row.symbol)
        ts_ns = int(pd.Timestamp(row.signal_time).value) if pd.notna(row.signal_time) else 0
        if not ts_ns or sym not in by_symbol:
            no_action_net.append(0.0)
            no_long_net.append(0.0)
            long_match_count.append(0)
            continue
        # Match any long in (signal_time, signal_time + cooldown].
        items = by_symbol[sym]
        ns_keys = [k for k, _ in items]
        left = int(np.searchsorted(ns_keys, ts_ns, side="right"))
        right = int(np.searchsorted(ns_keys, ts_ns + cd_ns, side="right"))
        if right <= left:
            no_action_net.append(0.0)
            no_long_net.append(0.0)
            long_match_count.append(0)
            continue
        # Sum the realized long net20 across all matched longs (usually 0 or 1).
        nets = [items[i][1].get("net_return", 0.0) for i in range(left, right)]
        nets = [float(x) if np.isfinite(x) else 0.0 for x in nets]
        a = float(sum(nets))
        no_action_net.append(a)
        no_long_net.append(-a)
        long_match_count.append(right - left)
    out = signals_df.copy()
    out["no_action_long_net"] = no_action_net
    out["no_long_net"] = no_long_net
    out["matched_longs"] = long_match_count
    return out


# --------------------------------------------------------------------------------------
# Per-symbol streaming driver
# --------------------------------------------------------------------------------------


def _collect_symbol_signals(
    group: pd.DataFrame,
    sleeves: tuple[SleeveSpec, ...],
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V34Config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    group = group.sort_values("bar_open_time").reset_index(drop=True)
    for sleeve in sleeves:
        if sleeve.source == "motif":
            rows.extend(_emit_motif_sleeve_signals(group, sleeve, cfg))
        elif sleeve.source == "cic_candidate":
            rows.extend(_emit_cic_sleeve_signals(group, sleeve, cic_long_index, cfg))
    if not rows:
        return rows
    # Attach Fast + Swing executions per signal row.
    enriched: list[dict[str, object]] = []
    for signal in rows:
        for label, rule in (("fast", cfg.fast_rule), ("swing", cfg.swing_rule)):
            enriched.append(_execute_signal(group, signal, rule, label))
    return enriched


def _stream_sleeve_signals(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V34Config,
) -> pd.DataFrame:
    """Iterate symbols, collect short signals + executions across all sleeves."""
    from pressure_graph.reports.v06a1 import _read_symbol_features  # deferred — see header note

    rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= cfg.top_n].copy()
        if data.empty:
            continue
        rows.extend(_collect_symbol_signals(data, cfg.sleeves, cic_long_index, cfg))
        if idx % 25 == 0:
            print(f"v3.4 sleeve scan: {idx}/{len(symbols)} symbols, {len(rows)} sleeve trades", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Aggregation — the 9 instruction-mandated CSV tables (§6).
# --------------------------------------------------------------------------------------


def _month_capped_total(sample: pd.DataFrame, net_col: str, cap: float) -> float:
    if sample.empty:
        return float("nan")
    total = pd.to_numeric(sample[net_col], errors="coerce").sum()
    if not np.isfinite(total) or total <= 0:
        return float(total) if np.isfinite(total) else float("nan")
    cap_value = total * cap
    capped = []
    for _, g in sample.groupby("month", sort=False, dropna=False):
        v = pd.to_numeric(g[net_col], errors="coerce").sum()
        capped.append(min(v, cap_value) if v > 0 else v)
    return float(np.sum(capped))


def _max_symbol_share(sample: pd.DataFrame, net_col: str) -> float:
    if sample.empty:
        return float("nan")
    grouped = pd.to_numeric(sample[net_col], errors="coerce").groupby(sample["symbol"], sort=False).sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else float("nan")


def _candidate_summary(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ].copy()
            if sample.empty:
                continue
            for cost in cfg.cost_grid_bps:
                for slip in cfg.extra_slippage_grid_bps:
                    net = _net_at_cost(sample["gross_return"], cost, slip)
                    sample_n = sample.assign(_net=net.to_numpy())
                    rows.append(
                        {
                            "sleeve_code": sleeve.code,
                            "sleeve_name": sleeve.name,
                            "execution": execution,
                            "cost_single_side_bps": cost,
                            "extra_slippage_bps": slip,
                            "trades": int(len(sample)),
                            "short_net_mean": float(net.mean()),
                            "short_net_total": float(net.sum()),
                            "gross_mean": float(pd.to_numeric(sample["gross_return"], errors="coerce").mean()),
                            "win_rate": float((net > 0).mean()),
                            "squeeze_out_rate": float(sample["squeezed"].fillna(False).astype(bool).mean()),
                            "avg_mae": float(pd.to_numeric(sample["max_adverse_excursion"], errors="coerce").mean()),
                            "month_capped_net": _month_capped_total(sample_n, "_net", cfg.month_cap),
                            "max_symbol_share": _max_symbol_share(sample_n, "_net"),
                            "sample_passes_min": bool(len(sample) >= cfg.min_samples),
                        }
                    )
    return pd.DataFrame(rows)


def _first_touch_summary(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    """Tabulate stop / take-profit / max-hold mix per sleeve (the short_first_touch view)."""
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ]
            if sample.empty:
                continue
            reasons = sample["exit_reason"].astype(str).value_counts(normalize=True).to_dict()
            rows.append(
                {
                    "sleeve_code": sleeve.code,
                    "execution": execution,
                    "trades": int(len(sample)),
                    "rate_take_profit": float(reasons.get("take_profit", 0.0) + reasons.get("tp_ambiguous", 0.0)),
                    "rate_stop": float(reasons.get("stop", 0.0) + reasons.get("stop_ambiguous", 0.0)),
                    "rate_max_hold": float(reasons.get("max_hold", 0.0)),
                    "rate_squeezed": float(sample["squeezed"].fillna(False).astype(bool).mean()),
                }
            )
    return pd.DataFrame(rows)


def _clean_hit_summary(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    """The docx-mandated clean_short_hit cut: target hit AND not squeezed first."""
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ]
            if sample.empty:
                continue
            # take_profit reason AND not squeezed -> clean hit.
            tp_rows = sample[sample["exit_reason"].astype(str).str.startswith("take_profit")]
            clean = tp_rows[~tp_rows["squeezed"].fillna(False).astype(bool)]
            net = _net_at_cost(clean["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps)
            base_net = _net_at_cost(sample["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps)
            rows.append(
                {
                    "sleeve_code": sleeve.code,
                    "execution": execution,
                    "trades_total": int(len(sample)),
                    "trades_clean": int(len(clean)),
                    "clean_hit_rate": float(len(clean) / len(sample)) if len(sample) else float("nan"),
                    "clean_net_mean": float(net.mean()) if len(net) else float("nan"),
                    "baseline_net_mean": float(base_net.mean()) if len(base_net) else float("nan"),
                    "clean_lift_vs_baseline": (float(net.mean()) - float(base_net.mean()))
                    if len(net) and len(base_net)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _cost_stress(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ]
            if sample.empty:
                continue
            for cost in cfg.cost_grid_bps:
                for slip in cfg.extra_slippage_grid_bps:
                    net = _net_at_cost(sample["gross_return"], cost, slip)
                    rows.append(
                        {
                            "sleeve_code": sleeve.code,
                            "execution": execution,
                            "cost_single_side_bps": cost,
                            "extra_slippage_bps": slip,
                            "trades": int(len(sample)),
                            "net_mean": float(net.mean()) if len(net) else float("nan"),
                            "net_total": float(net.sum()) if len(net) else float("nan"),
                            "positive_after_costs": bool(net.sum() > 0) if len(net) else False,
                        }
                    )
    return pd.DataFrame(rows)


def _regime_split(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ]
            if sample.empty:
                continue
            for regime in ("BTC_down", "BTC_chop", "BTC_up"):
                rs = sample[sample["btc_state"].eq(regime)]
                net = _net_at_cost(rs["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps)
                rows.append(
                    {
                        "sleeve_code": sleeve.code,
                        "execution": execution,
                        "btc_state": regime,
                        "trades": int(len(rs)),
                        "net_mean": float(net.mean()) if len(net) else float("nan"),
                        "squeeze_out_rate": float(rs["squeezed"].fillna(False).astype(bool).mean())
                        if len(rs)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _month_symbol_attribution(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        sample = trades[
            (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq("fast"))
        ].copy()
        if sample.empty:
            continue
        net = _net_at_cost(sample["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps)
        sample["_net"] = net.to_numpy()
        for (month, symbol), g in sample.groupby(["month", "symbol"], sort=False, dropna=False):
            rows.append(
                {
                    "sleeve_code": sleeve.code,
                    "month": month,
                    "symbol": symbol,
                    "trades": int(len(g)),
                    "net_total": float(pd.to_numeric(g["_net"], errors="coerce").sum()),
                }
            )
    return pd.DataFrame(rows)


def _short_vs_no_long(trades: pd.DataFrame, cfg: V34Config) -> pd.DataFrame:
    """The decisive table (docx §7).

    For each sleeve at the focal cost grid, compute:
      A_total: sum of would-be long net20 across signals (long stays as-is)
      B_total: -A_total (no-long cooldown — long avoided)
      C_total: sum of short net20 (open short)
      decision: short_value = C - B > 0 means shorting beats risk-off here.
    """
    rows: list[dict[str, object]] = []
    if trades.empty or "no_action_long_net" not in trades.columns:
        return pd.DataFrame()
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ]
            if sample.empty:
                continue
            c_net = _net_at_cost(sample["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps)
            a_total = float(pd.to_numeric(sample["no_action_long_net"], errors="coerce").sum())
            b_total = -a_total
            c_total = float(c_net.sum())
            short_value = c_total - b_total
            verdict = (
                "true_short_value"
                if short_value > 0 and c_total > 0
                else "risk_off_value" if b_total > 0 and b_total >= c_total
                else "no_value"
            )
            rows.append(
                {
                    "sleeve_code": sleeve.code,
                    "execution": execution,
                    "trades": int(len(sample)),
                    "A_no_action_long_total": a_total,
                    "B_no_long_cooldown_total": b_total,
                    "C_open_short_total": c_total,
                    "C_minus_B": short_value,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def _short_as_hedge(trades: pd.DataFrame, cic_long_index: dict, cfg: V34Config) -> pd.DataFrame:
    """Hedge value: does the sleeve PnL correlate with long-book down days?

    A small positive hedge correlation can be valuable even when the sleeve's
    standalone alpha is mediocre — but only as a small sleeve, per docx §9.
    """
    rows: list[dict[str, object]] = []
    if trades.empty or not cic_long_index:
        return pd.DataFrame()
    # Build a daily long PnL series from the index.
    long_rows = [
        {"date": pd.Timestamp(v.get("entry_time")).floor("D"), "net": float(v.get("net_return", 0.0))}
        for v in cic_long_index.values()
        if pd.notna(v.get("entry_time"))
    ]
    if not long_rows:
        return pd.DataFrame()
    long_daily = (
        pd.DataFrame(long_rows)
        .groupby("date", sort=True)["net"]
        .sum()
        .reset_index()
        .rename(columns={"net": "long_net"})
    )
    for sleeve in cfg.sleeves:
        for execution in ("fast", "swing"):
            sample = trades[
                (trades["sleeve_code"].eq(sleeve.code)) & (trades["execution"].eq(execution))
            ].copy()
            if sample.empty:
                continue
            sample["date"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce").dt.floor("D")
            sample["net"] = _net_at_cost(sample["gross_return"], cfg.focal_cost_bps, cfg.focal_extra_slippage_bps).to_numpy()
            short_daily = sample.groupby("date", sort=True)["net"].sum().reset_index().rename(columns={"net": "short_net"})
            merged = long_daily.merge(short_daily, on="date", how="left").fillna({"short_net": 0.0})
            down_days = merged[merged["long_net"] < 0]
            if len(merged) < 8:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(merged["long_net"].to_numpy(), merged["short_net"].to_numpy())[0, 1])
            rows.append(
                {
                    "sleeve_code": sleeve.code,
                    "execution": execution,
                    "long_days": int(len(merged)),
                    "long_down_days": int(len(down_days)),
                    "short_net_on_long_down_days": float(down_days["short_net"].sum()) if len(down_days) else 0.0,
                    "short_net_on_long_up_days": float(merged[merged["long_net"] >= 0]["short_net"].sum()),
                    "corr_long_short": corr,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Notes writer
# --------------------------------------------------------------------------------------


def _write_notes(
    report_root: Path,
    candidate_summary: pd.DataFrame,
    short_vs_no_long: pd.DataFrame,
    cfg: V34Config,
) -> None:
    """Stamp the v3.4 verdict against the docx §6 pass criteria."""
    lines: list[str] = [
        "# v3.4 True Short Sleeve",
        "",
        "Source: short_instructment3(v3.4).docx. Research only — no shadow,",
        "paper-live, or real-live wiring.",
        "",
        f"- sleeves tested: {', '.join(s.code for s in cfg.sleeves)}",
        f"- executions: fast (tp=3%/sl=2%/4h) + swing (tp=5%/sl=3%/12h)",
        f"- focal cost grid: {int(cfg.focal_cost_bps)}bp + {int(cfg.focal_extra_slippage_bps)}bp short slippage",
        f"- min sample threshold: {cfg.min_samples} per sleeve",
        "",
    ]

    if not candidate_summary.empty:
        focal = candidate_summary[
            (candidate_summary["cost_single_side_bps"].eq(cfg.focal_cost_bps))
            & (candidate_summary["extra_slippage_bps"].eq(cfg.focal_extra_slippage_bps))
        ]
        if not focal.empty:
            lines.append("## Per-sleeve focal scoreboard (20bp + 5bp short slippage)")
            for row in focal.sort_values("short_net_mean", ascending=False).itertuples(index=False):
                pass_min = "PASS" if row.sample_passes_min else "low_n"
                lines.append(
                    f"- **{row.sleeve_code}/{row.execution}**: n={row.trades} ({pass_min}), "
                    f"net_mean={row.short_net_mean:.4%}, win={row.win_rate:.1%}, "
                    f"squeeze={row.squeeze_out_rate:.1%}, month_cap={row.month_capped_net:.4f}, "
                    f"max_sym_share={row.max_symbol_share:.1%}."
                )
            lines.append("")

    if not short_vs_no_long.empty:
        lines.append("## Decisive: short vs no-long (docx §7)")
        for row in short_vs_no_long.itertuples(index=False):
            lines.append(
                f"- **{row.sleeve_code}/{row.execution}**: A_long={row.A_no_action_long_total:+.4f}, "
                f"B_no_long={row.B_no_long_cooldown_total:+.4f}, C_short={row.C_open_short_total:+.4f}, "
                f"C-B={row.C_minus_B:+.4f} -> **{row.verdict}**."
            )
        lines.append("")

    # Verdict synthesis.
    promote: list[str] = []
    risk_off_only: list[str] = []
    drop: list[str] = []
    if not short_vs_no_long.empty:
        for row in short_vs_no_long.itertuples(index=False):
            if row.verdict == "true_short_value":
                promote.append(f"{row.sleeve_code}/{row.execution}")
            elif row.verdict == "risk_off_value":
                risk_off_only.append(f"{row.sleeve_code}/{row.execution}")
            else:
                drop.append(f"{row.sleeve_code}/{row.execution}")

    lines.extend(
        [
            "## Verdict",
            f"- candidates for promotion to small short sleeve: {', '.join(promote) or 'none'}",
            f"- risk-off value only (stay in v1.2s2/v1.2s3 gate, do not short): {', '.join(risk_off_only) or 'none'}",
            f"- no value, drop: {', '.join(drop) or 'none'}",
            "",
            "## Discipline",
            "- Every short entry waits for the breakdown bar — never the failure event itself.",
            "- 3-action compare is the gate: C-B > 0 *and* C-total > 0 *and* sample >= 50.",
            "- Hedge value is reported separately. Hedge-only sleeves stay tiny (docx §9).",
            "- No paper-live / real-live permission changes.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Main entry point — wires the full pipeline.
# --------------------------------------------------------------------------------------


def write_v3_4_true_short_sleeve(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V34Config = V34Config(),
) -> dict[str, Path]:
    """Run the v3.4 true-short-sleeve research pipeline end-to-end.

    Produces the 9 docx-mandated CSVs + candidate_notes.md. Missing trade cache
    is allowed (SS3 sleeves and 3-action / hedge tables emit empty), so the
    motif-only sleeves still run for partial environments.
    """
    from pressure_graph.reports.v06c import _rank_inputs  # deferred — see header note
    from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool  # deferred

    report_root = ensure_dir(cfg.report_root)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )

    trade_cache = (
        read_parquet(cfg.trade_cache_path) if cfg.trade_cache_path.exists() else pd.DataFrame()
    )
    cic_long_index = _build_cic_long_index(trade_cache, cfg) if not trade_cache.empty else {}

    trades = _stream_sleeve_signals(
        feature_path, rank30, rank90, symbols, config, cic_long_index, cfg
    )
    trades = _three_action_compare(trades, cic_long_index, cfg) if not trades.empty else trades

    candidate_summary = _candidate_summary(trades, cfg)
    first_touch = _first_touch_summary(trades, cfg)
    clean_hit = _clean_hit_summary(trades, cfg)
    cost_stress = _cost_stress(trades, cfg)
    regime_split = _regime_split(trades, cfg)
    month_symbol = _month_symbol_attribution(trades, cfg)
    short_vs_no_long = _short_vs_no_long(trades, cfg)
    hedge = _short_as_hedge(trades, cic_long_index, cfg)

    outputs = {
        "trades": report_root / "short_sleeve_trades.csv",
        "candidate_summary": report_root / "short_candidate_summary.csv",
        "first_touch_summary": report_root / "short_first_touch_summary.csv",
        "clean_hit_summary": report_root / "short_clean_hit_summary.csv",
        "cost_stress": report_root / "short_cost_stress.csv",
        "regime_split": report_root / "short_regime_split.csv",
        "month_symbol_attribution": report_root / "short_month_symbol_attribution.csv",
        "short_vs_no_long_comparison": report_root / "short_vs_no_long_comparison.csv",
        "short_as_hedge_summary": report_root / "short_as_hedge_summary.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }

    trades.to_csv(outputs["trades"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    first_touch.to_csv(outputs["first_touch_summary"], index=False)
    clean_hit.to_csv(outputs["clean_hit_summary"], index=False)
    cost_stress.to_csv(outputs["cost_stress"], index=False)
    regime_split.to_csv(outputs["regime_split"], index=False)
    month_symbol.to_csv(outputs["month_symbol_attribution"], index=False)
    short_vs_no_long.to_csv(outputs["short_vs_no_long_comparison"], index=False)
    hedge.to_csv(outputs["short_as_hedge_summary"], index=False)
    _write_notes(report_root, candidate_summary, short_vs_no_long, cfg)
    return outputs


__all__ = [
    "FAST_RULE",
    "REPORT_ROOT",
    "SLEEVES",
    "SWING_RULE",
    "SleeveSpec",
    "V34Config",
    "TRADE_CACHE_PATH",
    "_GATE_REGISTRY",
    "_apply_gates",
    "_build_cic_long_index",
    "_candidate_summary",
    "_clean_hit_summary",
    "_collect_symbol_signals",
    "_cost_stress",
    "_emit_cic_sleeve_signals",
    "_emit_motif_sleeve_signals",
    "_execute_signal",
    "_find_breakdown",
    "_first_touch_summary",
    "_month_symbol_attribution",
    "_net_at_cost",
    "_reference_low_for",
    "_regime_split",
    "_short_as_hedge",
    "_short_vs_no_long",
    "_stream_sleeve_signals",
    "_three_action_compare",
    "_write_notes",
    "write_v3_4_true_short_sleeve",
]
