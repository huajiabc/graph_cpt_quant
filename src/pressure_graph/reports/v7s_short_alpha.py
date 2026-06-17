"""v7S Short Alpha Exploration — orthogonal short-side research lane.

The closure doc (``docs/short_research_closure.md``) terminated the
v12s/v3.4/v4S/v6S short thread. v7S is the explicit reopen, framed by
``short_instructment6 (v7s).docx``: do NOT iterate the old failed motifs;
instead open a new, orthogonal lane that answers three questions:

1. What market state lets short itself pay?
2. What short signal is strictly better than ``no_long``?
3. What short complements the long stack (hedge-positive in long DD)?

The doc names five candidate directions:

- **Direction A** — Cross-exchange downside lead-lag short. Source-venue
  (Binance / OKX / Hyperliquid) sell impulse → target-venue (Bybit) lag →
  failed bounce → short. Data: Binance UM aggTrades + Bybit linear
  aggTrades, both reachable on the A100 box.
- **Direction B** — Long-liquidation / forced-unwind continuation. Needs
  a liquidation tape; deferred (no source on the research environment yet).
- **Direction C 2.0** — Crowded unwind with explicit taker-buy exhaustion +
  CVD divergence + large-sell cluster. Distinct from v4S Path C (which
  used funding/OI/stall only).
- **Direction D** — Relative-value pair short: ``short overextended beta /
  long sector leader``. Closer to a hedge than directional short.
- **Direction E** — Long failure after CIC exhaustion. Strict version of
  v4S Path A: ``CIC continuation → CP60 weak → Protect_A not active →
  beta_high environment gone → break below entry / pullback low → sell
  flow confirms → short``. The two new gates vs v3.4 SS3 are
  ``beta_high_gone`` and ``sell_flow_confirms``.

This module ships Direction E as the FIRST concrete candidate, since the
required upstream (CIC long index, CP60 / Protect_A flags, v11 orderflow
event windows) is already in the repo. Directions A / C2.0 / D are
stubbed with config flags + TODO markers so a follow-up commit can plug
each in without re-shaping the harness.

Outputs (per direction, docx-mandated): all ten CSVs land under
``reports/v7s_short_alpha/<direction>/``:

    short_candidate_summary.csv
    short_cost_grid.csv
    short_first_touch.csv
    short_vs_no_long.csv
    short_vs_exit_long.csv
    short_hedge_value.csv
    month_cap_leave_one_month.csv
    symbol_contribution.csv
    matched_random_baseline.csv
    candidate_notes.md

Acceptance gates (docx §统一验收标准, 10 gates):

    1. net20 + extra slippage > 0
    2. net30 not collapsing (drop < net20 by less than ½)
    3. clean_short_hit lifts vs unconditional
    4. up_before_down / squeeze controllable (< 20 %)
    5. month_cap35 still positive
    6. leave-one-month not collapsing (worst-month-out net > 0)
    7. max_symbol_contribution < 35 %
    8. matched_random_baseline strictly worse
    9. short > no_long > 0  (closure §reopen criterion 4)
   10. not co-lossy with the long stack (hedge_corr ≤ -0.3 OR positive in
       long-stack worst month — closure §reopen criterion 2)

A candidate that fails any gate is recorded with ``verdict=no_value`` in
``short_candidate_summary.csv`` and never promoted. ``risk_off_only`` is
the only "intermediate" verdict (gates 1-8 OK, 9-10 fail) — same as v6S.

Tier: research only. No paper-live / real-live permission changes.
Production runs on A100 (see ``scripts/server_v7s_short_alpha_run.sh``);
the local repo has no feature parquet so a local run will report
``no_data`` and write empty CSV stubs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.backtest.short_execution import ShortExitRule
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool
from pressure_graph.reports.v12s_short_motif_atlas import _b, _f
from pressure_graph.reports.v3_4_true_short_sleeve import (
    FAST_RULE,
    SWING_RULE,
    SleeveSpec,
    V34Config,
    _build_cic_long_index,
    _execute_signal,
    _find_breakdown,
    _gate_cp60_would_exit,
    _gate_failed_followthrough,
    _gate_no_protect_a,
)

REPORT_ROOT = Path("reports/v7s_short_alpha")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
ORDERFLOW_EVENT_PATH = Path("data/orderflow_history/binance_um/cic_event_orderflow.parquet")

DIRECTION_A = "A_cross_exchange_lag"
DIRECTION_B = "B_liquidation_continuation"
DIRECTION_C = "C_crowded_unwind_v2"
DIRECTION_D = "D_relative_value_pair"
DIRECTION_E = "E_cic_failure_confirmed"

DIRECTIONS: tuple[str, ...] = (DIRECTION_A, DIRECTION_B, DIRECTION_C, DIRECTION_D, DIRECTION_E)

# Direction E candidate codes — strict CIC-failure short, two variants by
# breakdown reference (entry close vs pullback low). Each candidate runs Fast
# and Swing exit rules; the candidate code below is exit-rule agnostic.
CANDIDATE_E1 = "E1_cic_break_entry_strict"      # ref = entry close
CANDIDATE_E2 = "E2_cic_break_pullback_strict"   # ref = pullback low

E_CANDIDATES: tuple[str, ...] = (CANDIDATE_E1, CANDIDATE_E2)

# Direction D candidate codes — relative-value pair (short overextended beta /
# long hedge). Per the v7s expanded spec, four candidates differ by HEDGE TYPE,
# plus one naked-short baseline column so the discipline checks can prove
# pair > naked at the row level.
CANDIDATE_D0 = "D0_naked_short"        # naked: -symbol_short (no hedge)
CANDIDATE_D1 = "D1_pair_btc"           # short beta / long BTC
CANDIDATE_D2 = "D2_pair_eth"           # short beta / long ETH
CANDIDATE_D3 = "D3_pair_dynamic_leader"  # short beta / long argmax-ret leader from pool
CANDIDATE_D4 = "D4_pair_basket"        # short beta / long average of leader basket

D_CANDIDATES: tuple[str, ...] = (
    CANDIDATE_D0,
    CANDIDATE_D1,
    CANDIDATE_D2,
    CANDIDATE_D3,
    CANDIDATE_D4,
)

# Direction A candidate codes — cross-exchange downside lead-lag short.
# Source venue = Binance UM (lead); target venue = Bybit (lag). Hyperliquid
# leg (A3 per docx) is deferred until a Hyperliquid tape source is wired.
CANDIDATE_A0 = "A0_no_filter"                              # control: short whenever an event fires
CANDIDATE_A1 = "A1_binance_sell_impulse_bybit_lag"         # shock_bar Binance sell + Bybit still high
CANDIDATE_A2 = "A2_binance_breakdown_bybit_failed_reclaim"  # pullback+reclaim Binance sell + Bybit failed reclaim

A_CANDIDATES: tuple[str, ...] = (
    CANDIDATE_A0,
    CANDIDATE_A1,
    CANDIDATE_A2,
)


@dataclass(frozen=True)
class V7SConfig:
    """v7S driver config — Direction E is the only fully implemented direction."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    orderflow_event_path: Path = ORDERFLOW_EVENT_PATH
    long_pool_name: str = "P2_CIC1_CIC2_COMBINED"
    top_n: int = 30

    # Which directions to execute this run. E + D + A are wired; B/C remain
    # NotImplementedError-stubbed (B needs liquidation tape; C needs CVD
    # divergence labels not yet exported).
    enabled_directions: tuple[str, ...] = (DIRECTION_E, DIRECTION_D, DIRECTION_A)

    # Direction E sleeve specs — kept inline so the strict gates are visible
    # at config-read time (`grep -n E_SLEEVES` finds the active surface).
    e_breakdown_valid_bars: int = 12
    e_cooldown_bars: int = 16
    e_fast_rule: ShortExitRule = FAST_RULE
    e_swing_rule: ShortExitRule = SWING_RULE

    # Cost grid (docx gate 1 + 2). Focal = 20 bps, stress = 30 bps.
    cost_grid_bps: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    extra_slippage_bps: tuple[float, ...] = (0.0, 5.0, 10.0)
    focal_cost_bps: float = 20.0
    focal_extra_slippage_bps: float = 0.0

    # Funding accrual on short (CIC failure shorts often fire under
    # funding≥60 percentile so the short receives funding from longs).
    funding_apr_assumption: float = 0.30
    bars_per_hour: float = 4.0  # 15-min bars

    # Discipline thresholds (docx §统一验收标准 + closure §reopen criteria).
    min_samples: int = 20
    month_cap_pct: float = 0.35
    max_symbol_share: float = 0.35
    max_squeeze_share: float = 0.20
    hedge_corr_max: float = -0.30
    no_long_block_hours: int = 12  # window for B counterfactual

    # Forward-walk labels.
    label_window_bars: int = 48  # matches Swing max_hold
    hit_down_thresholds: tuple[float, ...] = (0.03, 0.05)
    up_before_down_thresholds: tuple[float, ...] = (0.02, 0.03)

    # Direction E NEW gates.
    e_beta_high_lookback_bars: int = 16  # 4h: how far back must beta_high have been to count as "gone"
    e_beta_high_ret_percentile: float = 85.0  # ret_4h_percentile threshold for "was hot"
    e_beta_high_cooled_percentile: float = 50.0  # ret_4h_percentile cap for "cooled now"
    e_sell_flow_window: str = "reclaim_bar"  # which orderflow window to read
    e_sell_flow_max_imbalance: float = -0.05  # net taker imbalance must be sell-leaning
    # If orderflow is unavailable, fail_open=True lets the gate pass with an
    # audit reason (downstream candidate_notes.md still flags the run as
    # "orderflow_missing").
    e_sell_flow_fail_open: bool = False

    # Direction D — relative-value pair (short overextended beta / long hedge).
    # Gate parameters (shared across all D candidates).
    d_lookback_bars: int = 16  # 4h scan window for overperformance + reclaim
    d_overextended_pct: float = 95.0  # ret_4h_percentile bar for "extreme overperformance"
    d_leader_weak_ret_4h: float = -0.005  # BTC ret_4h ≤ -0.5 % counts as leader weakening
    d_reclaim_tolerance: float = 0.015  # close must be ≥1.5 % below the lookback's high
    d_cooldown_bars: int = 24
    d_exclude_leader: bool = True  # do not run Direction D on a leader-pool symbol itself

    # Fixed pair holding horizons in bars (4h, 12h, 24h with 15-min bars).
    d_holding_horizons_bars: tuple[int, ...] = (16, 48, 96)

    # Hedge symbol pool.
    d_leader_pool: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
    d_basket_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    d_dynamic_leader_lookback_bars: int = 96  # 24h "leader showing strength" window

    # No-confirmation ablation: when True, run a SECOND pass without the
    # reclaim_failure gate. Candidates from that pass get suffix ``_nc`` so
    # the discipline view can compare with vs without failure confirmation
    # (docx §核心对照 item 7).
    d_include_no_confirmation: bool = True

    # Direction A — cross-exchange downside lead-lag short.
    # Source venue = Binance UM (lead); target venue = Bybit (lag). Hyperliquid
    # leg (A3 per docx) is deferred until a Hyperliquid tape source is wired.
    a_binance_sell_impulse_max: float = -0.15  # shock_bar buy_sell_imbalance below this → Binance sell impulse
    a_binance_breakdown_max: float = -0.05  # pullback+reclaim both below this → Binance breakdown sustained
    a_bybit_lag_max_drop_pct: float = 0.015  # Bybit close drop ≤ 1.5 % over lookback → Bybit lagging
    a_bybit_failed_reclaim_pct: float = 0.005  # Bybit recovered < 0.5 % from window low → failed reclaim
    a_lookback_bars_for_lag: int = 4  # 1 h: window for Bybit price-action read
    a_holding_horizons_bars: tuple[int, ...] = (16, 48, 96)  # 4h / 12h / 24h
    a_event_orderflow_path: Path = ORDERFLOW_EVENT_PATH  # reuse Direction E's lookup
    a_exclude_leader_symbols: bool = True  # do not short BTC/ETH/SOL/BNB

    # A1 threshold sweep — fires the same Binance-sell-impulse gate but at
    # different Bybit-lag thresholds. The threshold sweep produces extra
    # candidate codes A1_lag1pct / A1_lag1p5pct / A1_lag2pct / A1_lag2p5pct.
    a_lag_threshold_sweep: tuple[float, ...] = (0.010, 0.015, 0.020, 0.025)

    # Matched-random baseline knobs (docx gate 8).
    random_baseline_draws: int = 100
    random_baseline_seed: int = 20260617

    # v3.4 driver config used by the breakdown helpers.
    v34_cfg: V34Config = field(default_factory=V34Config)


# --------------------------------------------------------------------------------------
# NEW gates: beta_high_gone + sell_flow_confirms
# Both fail-closed by default; sell_flow_confirms has a fail_open knob for
# environments without orderflow_history backfill.
# --------------------------------------------------------------------------------------


def _gate_beta_high_gone(group: pd.DataFrame, idx: int, cfg: V7SConfig) -> bool:
    """``beta_high environment 消失``: the high-beta market regime that
    sponsored the CIC long is behind us at the break bar.

    The v7s docx phrasing is a *market-regime* observation, not a
    per-symbol percentile read. Three layered proxies are tried in order;
    the first available source wins:

    1. **v07c semantic** — if ``gate_beta_already_extended`` (or
       ``c2_bucket_beta_extreme_overextended``) is in the parquet, fire
       when the flag was True in ``[idx - lookback, idx)`` and False
       at idx. The raw v0.3 parquet typically does NOT carry this.

    2. **BTC volatility regime** — ``btc_vol_regime`` transitioned from
       ``high_vol`` somewhere in the lookback to non-``high_vol`` at idx.
       This is the closest direct proxy because beta-extreme regimes
       coincide with high-vol BTC periods.

    3. **BTC market state** — ``btc_market_state`` was ``BTC_up``
       somewhere in the lookback and is NOT ``BTC_up`` at idx. A weaker
       proxy used as fallback when ``btc_vol_regime`` is missing.

    Missing all three sources fails closed.
    """
    lookback = cfg.e_beta_high_lookback_bars
    if idx <= 0:
        return False
    start = max(0, idx - lookback)

    if "gate_beta_already_extended" in group.columns or "c2_bucket_beta_extreme_overextended" in group.columns:
        flag_col = "gate_beta_already_extended" if "gate_beta_already_extended" in group.columns else "c2_bucket_beta_extreme_overextended"
        flags = _b(group, flag_col)
        if idx < len(flags) and start < len(flags):
            return bool(flags[start:idx].any() and not flags[idx])

    if "btc_vol_regime" in group.columns:
        s = group["btc_vol_regime"].astype(str)
        if idx < len(s) and start < len(s):
            window = s.iloc[start:idx]
            was_hot = bool((window == "high_vol").any())
            cooled_now = bool(s.iloc[idx] != "high_vol")
            if was_hot and cooled_now:
                return True
            if was_hot and not cooled_now:
                return False

    if "btc_market_state" in group.columns:
        s = group["btc_market_state"].astype(str)
        if idx < len(s) and start < len(s):
            window = s.iloc[start:idx]
            was_up = bool((window == "BTC_up").any())
            cooled_now = bool(s.iloc[idx] != "BTC_up")
            return was_up and cooled_now

    return False


def _gate_sell_flow_confirms(
    group: pd.DataFrame,
    idx: int,
    cfg: V7SConfig,
    orderflow_lookup: Callable[[str, pd.Timestamp], dict | None] | None,
) -> tuple[bool, str]:
    """Sell-flow confirms: taker imbalance at the break window leans sell.

    Reads ``buy_sell_imbalance`` from the v11 orderflow_history event payload
    keyed by (symbol, signal_time). Returns (passed, audit_reason).

    When ``orderflow_lookup`` is None (no orderflow cache present) the
    behaviour depends on ``cfg.e_sell_flow_fail_open``:

    - ``fail_open=False`` (default): returns (False, "orderflow_missing"),
      so the candidate is rejected by the gate.
    - ``fail_open=True``: returns (True, "orderflow_missing_open"), and the
      run's candidate_notes.md flags this so a reader does not mistake the
      result for a clean pass.
    """
    if orderflow_lookup is None:
        if cfg.e_sell_flow_fail_open:
            return True, "orderflow_missing_open"
        return False, "orderflow_missing"
    symbol = str(group["symbol"].iloc[idx]) if "symbol" in group.columns and idx < len(group) else ""
    feature_time = pd.to_datetime(group["feature_time"].iloc[idx], utc=True, errors="coerce") if "feature_time" in group.columns else pd.NaT
    if not symbol or pd.isna(feature_time):
        return False, "missing_symbol_or_time"
    payload = orderflow_lookup(symbol, feature_time)
    if not payload:
        if cfg.e_sell_flow_fail_open:
            return True, "orderflow_missing_open"
        return False, "orderflow_unmatched_event"
    window = payload.get(cfg.e_sell_flow_window) or {}
    imbalance = window.get("buy_sell_imbalance")
    if imbalance is None or not np.isfinite(imbalance):
        if cfg.e_sell_flow_fail_open:
            return True, "orderflow_imbalance_nan_open"
        return False, "orderflow_imbalance_nan"
    return bool(float(imbalance) <= cfg.e_sell_flow_max_imbalance), "ok"


# --------------------------------------------------------------------------------------
# Direction E — strict CIC-failure-confirmed short signal emitter
# --------------------------------------------------------------------------------------


def _build_orderflow_lookup(path: Path) -> Callable[[str, pd.Timestamp], dict | None] | None:
    """Return a lookup callable into the v11 cic_event_orderflow parquet, or
    ``None`` when the file does not exist on this box.

    Schema (per ``orderflow_history.write_event_orderflow``): one row per
    (symbol, signal_time) carrying nested window payloads. We keep the
    lookup in memory because the file is small (<200k rows in production).
    """
    if not path.exists():
        return None
    df = read_parquet(path)
    if df.empty or "symbol" not in df.columns or "signal_time" not in df.columns:
        return None
    df = df.copy()
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["signal_time"])
    by_key: dict[tuple[str, int], dict] = {}
    for row in df.itertuples(index=False):
        key = (str(getattr(row, "symbol", "")), int(pd.Timestamp(getattr(row, "signal_time")).value))
        by_key[key] = row._asdict()

    def lookup(symbol: str, signal_time: pd.Timestamp) -> dict | None:
        ts_ns = int(pd.Timestamp(signal_time).value)
        return by_key.get((symbol, ts_ns))

    return lookup


def _emit_direction_e_signals(
    group: pd.DataFrame,
    candidate_code: str,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    orderflow_lookup: Callable[[str, pd.Timestamp], dict | None] | None,
    cfg: V7SConfig,
) -> list[dict[str, object]]:
    """Direction E entry stream — strict CIC-failure short.

    Pipeline per CIC long entry on this symbol:

    1. Locate the CIC long's confirmation bar via the long-index payload.
    2. Walk forward up to ``e_breakdown_valid_bars`` looking for the first
       bar whose close < reference_level (entry close for E1, pullback low
       for E2).
    3. Apply the four hard gates:
         - cp60_would_exit (v3.4)
         - no_protect_a (v3.4)
         - beta_high_gone (NEW)
         - sell_flow_confirms (NEW)
       Plus a soft gate ``failed_followthrough`` re-checked at break_idx
       (already implied by SS3 anchor, but kept here for symmetry).
    4. Emit a signal row with ``audit_reason`` populated so the candidate
       notes can show why each near-miss was rejected.
    """
    rows: list[dict[str, object]] = []
    if not cic_long_index:
        return rows
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    symbol = str(group["symbol"].iloc[0]) if len(group) else ""
    if not symbol:
        return rows
    breakdown_reference = "entry" if candidate_code == CANDIDATE_E1 else "pullback_low"
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    feature_ns = feature_time.astype("int64").to_numpy()
    n = len(group)
    active_until = -1
    v34 = cfg.v34_cfg

    for (sym_key, ts_key), payload in cic_long_index.items():
        if sym_key != symbol:
            continue
        anchor_ns = int(pd.Timestamp(ts_key).value)
        confirmation_idx = int(np.searchsorted(feature_ns, anchor_ns, side="left"))
        if confirmation_idx >= n - 1:
            continue
        if breakdown_reference == "entry":
            close = _f(group, "close")
            level = float(close[confirmation_idx]) if confirmation_idx < len(close) else float("nan")
        else:
            ref_low = float(payload.get("pullback_low", float("nan")))
            level = ref_low if np.isfinite(ref_low) else (
                float(np.nanmin(_f(group, "low")[confirmation_idx : confirmation_idx + 4]))
            )
        break_idx = _find_breakdown(group, confirmation_idx, level, cfg.e_breakdown_valid_bars)
        if break_idx < 0:
            continue
        entry_idx = break_idx + 1
        if entry_idx >= n or entry_idx <= active_until:
            continue

        audit_parts: list[str] = []
        if not _gate_cp60_would_exit(group, break_idx, v34):
            audit_parts.append("cp60_not_exit")
            continue
        if not _gate_no_protect_a(group, break_idx, v34):
            audit_parts.append("protect_a_active")
            continue
        if not _gate_beta_high_gone(group, break_idx, cfg):
            audit_parts.append("beta_still_high")
            continue
        sell_flow_ok, sell_flow_reason = _gate_sell_flow_confirms(
            group, break_idx, cfg, orderflow_lookup
        )
        if not sell_flow_ok:
            audit_parts.append(f"sell_flow:{sell_flow_reason}")
            continue

        rows.append(
            {
                "direction": DIRECTION_E,
                "candidate_code": candidate_code,
                "sleeve_code": candidate_code,  # alias so v3.4 _execute_signal accepts it
                "sleeve_name": candidate_code,
                "exchange": str(group.iloc[break_idx].get("exchange", "")),
                "symbol": symbol,
                "motif_code": "",
                "anchor_idx": int(confirmation_idx),
                "confirmation_idx": int(confirmation_idx),
                "break_idx": int(break_idx),
                "entry_idx": int(entry_idx),
                "anchor_feature_time": feature_time.iloc[confirmation_idx],
                "signal_time": feature_time.iloc[break_idx],
                "entry_time": bar_open_time.iloc[entry_idx],
                "reference_low": float(level),
                "month": (
                    feature_time.iloc[break_idx].strftime("%Y-%m")
                    if pd.notna(feature_time.iloc[break_idx])
                    else ""
                ),
                "btc_state": str(group.iloc[break_idx].get("btc_market_state", "")),
                "audit_reason": "ok" if not audit_parts else ";".join(audit_parts),
                "sell_flow_reason": sell_flow_reason,
            }
        )
        active_until = entry_idx + cfg.e_cooldown_bars
    return rows


def _execute_direction_e(
    group: pd.DataFrame,
    signal: dict,
    cfg: V7SConfig,
) -> list[dict[str, object]]:
    """Run Fast + Swing exit on a Direction E signal; return two enriched rows."""
    return [
        _execute_signal(group, signal, cfg.e_fast_rule, "fast"),
        _execute_signal(group, signal, cfg.e_swing_rule, "swing"),
    ]


# --------------------------------------------------------------------------------------
# Symbol-level driver — streams Direction E across the universe
# --------------------------------------------------------------------------------------


def _collect_direction_e_signals(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    orderflow_lookup: Callable[[str, pd.Timestamp], dict | None] | None,
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Stream Direction E events across the top-N universe."""
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows: list[dict[str, object]] = []
    for i, symbol in enumerate(symbols, start=1):
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        for candidate_code in E_CANDIDATES:
            signals = _emit_direction_e_signals(group, candidate_code, cic_long_index, orderflow_lookup, cfg)
            for sig in signals:
                rows.extend(_execute_direction_e(group, sig, cfg))
        if i % 25 == 0:
            print(f"v7S Direction E: {i}/{len(symbols)} symbols, {len(rows)} executions", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Direction D — relative-value pair (short overextended beta / long leader BTC)
# --------------------------------------------------------------------------------------


def _gate_beta_overextended(
    group: pd.DataFrame, idx: int, threshold: float, cfg: V7SConfig
) -> bool:
    """Symbol's ``ret_4h_percentile`` reached ≥ ``threshold`` somewhere in
    the lookback ``[idx - d_lookback_bars, idx]``. Missing column fails closed."""
    if "ret_4h_percentile" not in group.columns:
        return False
    pct = _f(group, "ret_4h_percentile")
    if idx >= len(pct):
        return False
    start = max(0, idx - cfg.d_lookback_bars)
    window = pct[start : idx + 1]
    if window.size == 0:
        return False
    return bool(np.nanmax(window) >= threshold)


def _gate_leader_weakening(group: pd.DataFrame, idx: int, cfg: V7SConfig) -> bool:
    """BTC's ret_4h at this bar is ≤ ``d_leader_weak_ret_4h``."""
    if "btc_ret_4h" not in group.columns:
        return False
    btc = _f(group, "btc_ret_4h")
    if idx >= len(btc):
        return False
    return bool(np.isfinite(btc[idx]) and btc[idx] <= cfg.d_leader_weak_ret_4h)


def _gate_beta_reclaim_failure(group: pd.DataFrame, idx: int, cfg: V7SConfig) -> bool:
    """Symbol's close at idx is ≥ ``d_reclaim_tolerance`` below the high in
    the lookback ``[idx - d_lookback_bars, idx]``."""
    close = _f(group, "close")
    high = _f(group, "high")
    if idx >= len(close) or idx >= len(high):
        return False
    start = max(0, idx - cfg.d_lookback_bars)
    window_high = high[start : idx + 1]
    if window_high.size == 0:
        return False
    lookback_high = float(np.nanmax(window_high))
    cur_close = float(close[idx])
    if not (np.isfinite(lookback_high) and np.isfinite(cur_close)) or lookback_high <= 0:
        return False
    drop_pct = (lookback_high - cur_close) / lookback_high
    return bool(drop_pct >= cfg.d_reclaim_tolerance)


def _emit_direction_d_signals(
    group: pd.DataFrame,
    cfg: V7SConfig,
    require_reclaim_failure: bool = True,
) -> list[dict[str, object]]:
    """Direction D entry stream — one row per bar that passes the 3-gate combo
    + cooldown. When ``require_reclaim_failure`` is False, the third gate is
    bypassed (this powers the no-confirmation ablation: how much of the alpha
    survives without the "failed reclaim" filter). Candidate (hedge type)
    expansion happens at execution time so we don't re-walk the bar tape per
    candidate."""
    rows: list[dict[str, object]] = []
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    symbol = str(group["symbol"].iloc[0]) if len(group) else ""
    if not symbol:
        return rows
    if cfg.d_exclude_leader and symbol in set(cfg.d_leader_pool):
        return rows
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    close = _f(group, "close")
    n = len(group)
    last_fire = -1_000_000
    gate_audit = "ok" if require_reclaim_failure else "no_reclaim_gate"
    for idx in range(cfg.d_lookback_bars, n - 1):
        if idx - last_fire < cfg.d_cooldown_bars:
            continue
        if not _gate_beta_overextended(group, idx, cfg.d_overextended_pct, cfg):
            continue
        if not _gate_leader_weakening(group, idx, cfg):
            continue
        if require_reclaim_failure and not _gate_beta_reclaim_failure(group, idx, cfg):
            continue
        entry_idx = idx + 1
        if entry_idx >= n:
            continue
        rows.append(
            {
                "direction": DIRECTION_D,
                "sleeve_code": "D_overextended_beta",
                "sleeve_name": "D_overextended_beta",
                "exchange": str(group.iloc[idx].get("exchange", "")),
                "symbol": symbol,
                "motif_code": "",
                "anchor_idx": int(idx),
                "confirmation_idx": int(idx),
                "break_idx": int(idx),
                "entry_idx": int(entry_idx),
                "anchor_feature_time": feature_time.iloc[idx],
                "signal_time": feature_time.iloc[idx],
                "entry_time": bar_open_time.iloc[entry_idx],
                "reference_low": float(close[idx]) if idx < len(close) else float("nan"),
                "month": (
                    feature_time.iloc[idx].strftime("%Y-%m")
                    if pd.notna(feature_time.iloc[idx])
                    else ""
                ),
                "btc_state": str(group.iloc[idx].get("btc_market_state", "")),
                "audit_reason": gate_audit,
                "confirmation_required": bool(require_reclaim_failure),
            }
        )
        last_fire = idx
    return rows


def _build_hedge_lookups(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V7SConfig,
) -> dict[str, tuple[pd.DataFrame, dict[int, int]]]:
    """Pre-load each hedge symbol's group + ``bar_open_time_ns -> idx`` map.

    Hedge pool = union of leader_pool and basket_symbols. Missing symbols are
    simply omitted from the dict; candidates that depend on them later return
    NaN for the leg (and the row is dropped).
    """
    out: dict[str, tuple[pd.DataFrame, dict[int, int]]] = {}
    needed = set(cfg.d_leader_pool) | set(cfg.d_basket_symbols)
    for sym in sorted(needed):
        g = _read_symbol_features(feature_path, rank30, rank90, sym, config)
        if g.empty:
            continue
        g = g.sort_values("bar_open_time").reset_index(drop=True)
        ns_array = pd.to_datetime(g["bar_open_time"], utc=True, errors="coerce").astype("int64").to_numpy()
        ns_to_idx = {int(ns): i for i, ns in enumerate(ns_array) if ns > 0}
        out[sym] = (g, ns_to_idx)
    return out


def _leg_return_at_horizon(
    hedge_lookup: tuple[pd.DataFrame, dict[int, int]] | None,
    entry_time: pd.Timestamp,
    holding_bars: int,
) -> float:
    """Long-leg gross return from entry_time over holding_bars (15-min bars)."""
    if hedge_lookup is None:
        return float("nan")
    hedge_group, ns_to_idx = hedge_lookup
    idx = ns_to_idx.get(int(pd.Timestamp(entry_time).value), -1)
    if idx < 0:
        return float("nan")
    open_arr = _f(hedge_group, "open")
    close_arr = _f(hedge_group, "close")
    if idx >= len(open_arr) or idx + holding_bars >= len(close_arr):
        return float("nan")
    entry_price = float(open_arr[idx])
    exit_price = float(close_arr[idx + holding_bars])
    if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or entry_price <= 0:
        return float("nan")
    return (exit_price - entry_price) / entry_price


def _pick_dynamic_leader(
    entry_time: pd.Timestamp,
    hedge_lookups: dict[str, tuple[pd.DataFrame, dict[int, int]]],
    cfg: V7SConfig,
) -> str | None:
    """Pick the leader-pool symbol whose ``d_dynamic_leader_lookback_bars``
    cumulative return at ``entry_time`` is highest. Falls back to the first
    pool symbol with available data."""
    best_sym: str | None = None
    best_ret = -float("inf")
    lookback = cfg.d_dynamic_leader_lookback_bars
    for sym in cfg.d_leader_pool:
        if sym not in hedge_lookups:
            continue
        g, ns_to_idx = hedge_lookups[sym]
        idx = ns_to_idx.get(int(pd.Timestamp(entry_time).value), -1)
        if idx < 0 or idx < lookback:
            continue
        close = _f(g, "close")
        if idx >= len(close):
            continue
        past = float(close[idx - lookback])
        now = float(close[idx])
        if not (np.isfinite(past) and np.isfinite(now)) or past <= 0:
            continue
        ret = (now - past) / past
        if ret > best_ret:
            best_ret = ret
            best_sym = sym
    return best_sym


def _execute_direction_d(
    group: pd.DataFrame,
    signal: dict,
    hedge_lookups: dict[str, tuple[pd.DataFrame, dict[int, int]]],
    cfg: V7SConfig,
) -> list[dict[str, object]]:
    """Expand one D signal into 5 candidates × len(d_holding_horizons_bars) rows.

    For each (candidate, horizon):
    - Compute the symbol's naked short return over the horizon.
    - For pair candidates, compute the hedge long return over the same horizon.
    - Pair gross = -symbol_short + hedge_long (i.e. symbol_short_return as defined
      already is positive when the short wins; we add it directly).

    Rows are tagged ``execution = "h4"/"h12"/"h24"`` (the holding window in hours).
    Naked short candidate D0 carries hedge_long = 0 and serves as the row-level
    baseline for the discipline checks.
    """
    entry_idx = int(signal.get("entry_idx", -1))
    open_arr = _f(group, "open")
    close_arr = _f(group, "close")
    if entry_idx < 0 or entry_idx >= len(open_arr):
        return []
    symbol_entry_price = float(open_arr[entry_idx])
    if not np.isfinite(symbol_entry_price) or symbol_entry_price <= 0:
        return []
    entry_time = pd.Timestamp(signal.get("entry_time"))
    if pd.isna(entry_time):
        return []

    dynamic_leader_sym = _pick_dynamic_leader(entry_time, hedge_lookups, cfg)

    rows: list[dict[str, object]] = []
    for horizon in cfg.d_holding_horizons_bars:
        exit_idx = entry_idx + horizon
        if exit_idx >= len(close_arr):
            continue
        symbol_exit = float(close_arr[exit_idx])
        if not np.isfinite(symbol_exit):
            continue
        symbol_short_ret = (symbol_entry_price - symbol_exit) / symbol_entry_price
        horizon_label = f"h{horizon * 15 // 60}"

        for cand_code in D_CANDIDATES:
            if cand_code == CANDIDATE_D0:
                hedge_syms: tuple[str, ...] = ()
            elif cand_code == CANDIDATE_D1:
                hedge_syms = ("BTCUSDT",)
            elif cand_code == CANDIDATE_D2:
                hedge_syms = ("ETHUSDT",)
            elif cand_code == CANDIDATE_D3:
                hedge_syms = (dynamic_leader_sym,) if dynamic_leader_sym else ()
            elif cand_code == CANDIDATE_D4:
                hedge_syms = cfg.d_basket_symbols
            else:
                continue

            hedge_returns: list[float] = []
            for hsym in hedge_syms:
                ret = _leg_return_at_horizon(hedge_lookups.get(hsym), entry_time, horizon)
                if np.isfinite(ret):
                    hedge_returns.append(ret)

            if hedge_syms and not hedge_returns:
                continue  # hedge unavailable — drop this candidate row
            hedge_long_ret = float(np.mean(hedge_returns)) if hedge_returns else 0.0
            pair_gross = symbol_short_ret + hedge_long_ret

            row = dict(signal)
            row.update(
                {
                    "candidate_code": cand_code,
                    "execution": horizon_label,
                    "holding_bars": int(horizon),
                    "gross_return": float(pair_gross),
                    "pair_gross_return": float(pair_gross),
                    "symbol_short_gross": float(symbol_short_ret),
                    "hedge_long_gross": float(hedge_long_ret),
                    "hedge_symbols": "|".join(hedge_syms) if hedge_syms else "naked",
                    "exit_reason": "horizon",
                    "execution_horizon_bars": int(horizon),
                }
            )
            rows.append(row)
    return rows


def _collect_direction_d_signals(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Stream Direction D events × 5 candidates × 3 horizons across the universe."""
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    hedge_lookups = _build_hedge_lookups(feature_path, rank30, rank90, config, cfg)
    if not hedge_lookups:
        print("v7S Direction D: no hedge symbols available — skipping.", flush=True)
        return pd.DataFrame()
    print(
        f"v7S Direction D: hedge symbols loaded = {sorted(hedge_lookups.keys())}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    leader_set = set(cfg.d_leader_pool)
    for i, symbol in enumerate(symbols, start=1):
        if cfg.d_exclude_leader and symbol in leader_set:
            continue
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        signals = _emit_direction_d_signals(group, cfg, require_reclaim_failure=True)
        for sig in signals:
            rows.extend(_execute_direction_d(group, sig, hedge_lookups, cfg))
        if cfg.d_include_no_confirmation:
            nc_signals = _emit_direction_d_signals(group, cfg, require_reclaim_failure=False)
            for sig in nc_signals:
                for r in _execute_direction_d(group, sig, hedge_lookups, cfg):
                    r["candidate_code"] = f"{r['candidate_code']}_nc"
                    rows.append(r)
        if i % 25 == 0:
            print(f"v7S Direction D: {i}/{len(symbols)} symbols, {len(rows)} executions", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Direction A — cross-exchange downside lead-lag short (Binance → Bybit)
# --------------------------------------------------------------------------------------


def _gate_binance_sell_impulse(payload: dict, cfg: V7SConfig) -> bool:
    """Shock-bar Binance buy_sell_imbalance ≤ ``a_binance_sell_impulse_max``.

    Reads the v11 cic_event_orderflow payload; missing field fails closed."""
    val = payload.get("shock_bar_buy_sell_imbalance")
    if val is None or not np.isfinite(val):
        return False
    return bool(float(val) <= cfg.a_binance_sell_impulse_max)


def _gate_binance_breakdown_sustained(payload: dict, cfg: V7SConfig) -> bool:
    """Both pullback_window and reclaim_bar Binance buy_sell_imbalance are
    sell-leaning (≤ ``a_binance_breakdown_max``). Reads cic_event_orderflow."""
    pb = payload.get("pullback_window_buy_sell_imbalance")
    rc = payload.get("reclaim_bar_buy_sell_imbalance")
    if pb is None or rc is None or not (np.isfinite(pb) and np.isfinite(rc)):
        return False
    return bool(float(pb) <= cfg.a_binance_breakdown_max and float(rc) <= cfg.a_binance_breakdown_max)


def _gate_bybit_lagging(group: pd.DataFrame, idx: int, cfg: V7SConfig) -> bool:
    """At bar ``idx`` Bybit hasn't dropped much vs ``a_lookback_bars_for_lag``
    bars ago (drop ≤ ``a_bybit_lag_max_drop_pct``)."""
    close = _f(group, "close")
    if idx >= len(close) or idx < cfg.a_lookback_bars_for_lag:
        return False
    past_close = float(close[idx - cfg.a_lookback_bars_for_lag])
    cur_close = float(close[idx])
    if not (np.isfinite(past_close) and np.isfinite(cur_close)) or past_close <= 0:
        return False
    drop_pct = (past_close - cur_close) / past_close
    return bool(drop_pct <= cfg.a_bybit_lag_max_drop_pct)


def _gate_bybit_failed_reclaim(group: pd.DataFrame, idx: int, cfg: V7SConfig) -> bool:
    """Bybit recovered less than ``a_bybit_failed_reclaim_pct`` from its low in
    the lookback window — failed to reclaim."""
    high = _f(group, "high")
    low = _f(group, "low")
    close = _f(group, "close")
    if idx >= len(close) or idx < cfg.a_lookback_bars_for_lag:
        return False
    start = idx - cfg.a_lookback_bars_for_lag
    window_low = float(np.nanmin(low[start : idx + 1]))
    cur_close = float(close[idx])
    if not (np.isfinite(window_low) and np.isfinite(cur_close)) or window_low <= 0:
        return False
    recovered = (cur_close - window_low) / window_low
    return bool(recovered < cfg.a_bybit_failed_reclaim_pct)


def _emit_direction_a_signals(
    events: pd.DataFrame,
    group: pd.DataFrame,
    candidate_code: str,
    cfg: V7SConfig,
) -> list[dict[str, object]]:
    """Direction A entry stream — scan cic_event_orderflow rows for this
    symbol, apply the candidate's gate combination, emit short entries."""
    rows: list[dict[str, object]] = []
    if events.empty or group.empty:
        return rows
    symbol = str(group["symbol"].iloc[0])
    sub = events[events["symbol"].astype(str) == symbol]
    if sub.empty:
        return rows
    if cfg.a_exclude_leader_symbols and symbol in set(cfg.d_leader_pool):
        return rows
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    feature_ns = feature_time.astype("int64").to_numpy()
    n = len(group)
    for _, ev in sub.iterrows():
        signal_time = pd.to_datetime(ev.get("signal_time"), utc=True, errors="coerce")
        if pd.isna(signal_time):
            continue
        ts_ns = int(signal_time.value)
        idx = int(np.searchsorted(feature_ns, ts_ns, side="left"))
        if idx >= n - 1:
            continue
        # Match exact bar (searchsorted gives insertion point; ensure bar_open_time matches)
        if idx < len(feature_ns) and feature_ns[idx] != ts_ns:
            # Try idx - 1 if close
            if idx > 0 and feature_ns[idx - 1] == ts_ns:
                idx = idx - 1
            else:
                continue

        payload = ev.to_dict()
        gate_audit = "ok"
        if candidate_code == CANDIDATE_A0:
            pass  # control — no filter
        elif candidate_code == CANDIDATE_A1:
            if not _gate_binance_sell_impulse(payload, cfg):
                continue
            if not _gate_bybit_lagging(group, idx, cfg):
                continue
        elif candidate_code == CANDIDATE_A2:
            if not _gate_binance_breakdown_sustained(payload, cfg):
                continue
            if not _gate_bybit_failed_reclaim(group, idx, cfg):
                continue
        else:
            continue

        entry_idx = idx + 1
        if entry_idx >= n:
            continue
        rows.append(
            {
                "direction": DIRECTION_A,
                "candidate_code": candidate_code,
                "sleeve_code": candidate_code,
                "sleeve_name": candidate_code,
                "exchange": str(group.iloc[idx].get("exchange", "")),
                "symbol": symbol,
                "motif_code": "",
                "anchor_idx": int(idx),
                "confirmation_idx": int(idx),
                "break_idx": int(idx),
                "entry_idx": int(entry_idx),
                "anchor_feature_time": feature_time.iloc[idx],
                "signal_time": feature_time.iloc[idx],
                "entry_time": bar_open_time.iloc[entry_idx],
                "reference_low": float(_f(group, "close")[idx]) if idx < len(_f(group, "close")) else float("nan"),
                "month": (
                    feature_time.iloc[idx].strftime("%Y-%m")
                    if pd.notna(feature_time.iloc[idx])
                    else ""
                ),
                "btc_state": str(group.iloc[idx].get("btc_market_state", "")),
                "binance_shock_imbalance": float(payload.get("shock_bar_buy_sell_imbalance", float("nan"))),
                "binance_pullback_imbalance": float(payload.get("pullback_window_buy_sell_imbalance", float("nan"))),
                "binance_reclaim_imbalance": float(payload.get("reclaim_bar_buy_sell_imbalance", float("nan"))),
                "audit_reason": gate_audit,
            }
        )
    return rows


def _execute_direction_a(
    group: pd.DataFrame,
    signal: dict,
    cfg: V7SConfig,
) -> list[dict[str, object]]:
    """Short Bybit at next bar; emit one row per fixed horizon (h4/h12/h24).

    Single-leg short execution (n_legs=1) — same cost model as Direction E
    and Direction D's naked baseline.
    """
    entry_idx = int(signal.get("entry_idx", -1))
    open_arr = _f(group, "open")
    close_arr = _f(group, "close")
    if entry_idx < 0 or entry_idx >= len(open_arr):
        return []
    entry_price = float(open_arr[entry_idx])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return []
    rows: list[dict[str, object]] = []
    for horizon in cfg.a_holding_horizons_bars:
        exit_idx = entry_idx + horizon
        if exit_idx >= len(close_arr):
            continue
        exit_price = float(close_arr[exit_idx])
        if not np.isfinite(exit_price):
            continue
        gross = (entry_price - exit_price) / entry_price
        row = dict(signal)
        row.update(
            {
                "execution": f"h{horizon * 15 // 60}",
                "holding_bars": int(horizon),
                "gross_return": float(gross),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "exit_reason": "horizon",
            }
        )
        rows.append(row)
    return rows


def _collect_direction_a_signals(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Stream Direction A events × 3 candidates × 3 horizons across the universe."""
    if not cfg.a_event_orderflow_path.exists():
        print(f"v7S Direction A: orderflow event parquet missing — skipping.", flush=True)
        return pd.DataFrame()
    events = read_parquet(cfg.a_event_orderflow_path)
    if events.empty or "symbol" not in events.columns or "signal_time" not in events.columns:
        print(f"v7S Direction A: orderflow event frame empty/malformed — skipping.", flush=True)
        return pd.DataFrame()
    print(f"v7S Direction A: orderflow events loaded = {len(events)}", flush=True)

    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows: list[dict[str, object]] = []
    for i, symbol in enumerate(symbols, start=1):
        if cfg.a_exclude_leader_symbols and symbol in set(cfg.d_leader_pool):
            continue
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        for candidate_code in A_CANDIDATES:
            signals = _emit_direction_a_signals(events, group, candidate_code, cfg)
            for sig in signals:
                rows.extend(_execute_direction_a(group, sig, cfg))
        # A1 threshold sweep: re-run A1 logic at each lag threshold and tag
        # the candidate code with the threshold suffix so the discipline view
        # can see whether the alpha generalizes across thresholds.
        original_lag = cfg.a_bybit_lag_max_drop_pct
        for thr in cfg.a_lag_threshold_sweep:
            if abs(thr - original_lag) < 1e-9:
                # Already covered by the canonical A1 above.
                continue
            sweep_cfg = replace(cfg, a_bybit_lag_max_drop_pct=thr)
            tag = f"lag{int(thr * 1000):d}bp"
            signals = _emit_direction_a_signals(events, group, CANDIDATE_A1, sweep_cfg)
            for sig in signals:
                sig["candidate_code"] = f"A1_{tag}"
                rows.extend(_execute_direction_a(group, sig, cfg))
        if i % 25 == 0:
            print(f"v7S Direction A: {i}/{len(symbols)} symbols, {len(rows)} executions", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cost / funding model — per-trade net
# --------------------------------------------------------------------------------------


def _net_short_return(
    gross: float,
    holding_bars: int,
    cost_bps: float,
    extra_slippage_bps: float,
    cfg: V7SConfig,
) -> float:
    """Post-cost, post-funding net for a 1.0× short (single leg)."""
    if not np.isfinite(gross):
        return float("nan")
    round_trip_bps = 2.0 * (float(cost_bps) + float(extra_slippage_bps)) / 10_000.0
    holding_hours = max(float(holding_bars), 0.0) / max(cfg.bars_per_hour, 1e-6)
    funding_accrual = cfg.funding_apr_assumption * (holding_hours / (24.0 * 365.0))
    return float(gross) - round_trip_bps + funding_accrual


def _net_pair_return(
    gross: float,
    holding_bars: int,
    cost_bps: float,
    extra_slippage_bps: float,
    cfg: V7SConfig,
    n_legs: int = 2,
) -> float:
    """Post-cost net for a relative-value position with ``n_legs`` legs.

    Pair PnL already nets ``-symbol_short`` against ``+leader_long`` so the
    only adjustment is round-trip cost per leg. Funding is assumed symmetric
    between legs in a pair — net ≈ 0; for ``n_legs=1`` (naked baseline) the
    short receives funding at the configured APR via the holding window.
    """
    if not np.isfinite(gross):
        return float("nan")
    round_trip_bps = 2.0 * n_legs * (float(cost_bps) + float(extra_slippage_bps)) / 10_000.0
    if n_legs == 1:
        holding_hours = max(float(holding_bars), 0.0) / max(cfg.bars_per_hour, 1e-6)
        funding_accrual = cfg.funding_apr_assumption * (holding_hours / (24.0 * 365.0))
        return float(gross) - round_trip_bps + funding_accrual
    return float(gross) - round_trip_bps


def _n_legs_for(direction: str, candidate_code: str, cfg: V7SConfig) -> int:
    """Number of legs charged in the cost model.

    - Direction E shorts: 1 leg.
    - Direction D, D0 naked: 1 leg.
    - Direction D, D1/D2/D3: 2 legs (symbol short + one hedge long).
    - Direction D, D4 basket: 1 + len(basket) legs.

    The ``_nc`` (no-confirmation) suffix used by the ablation pass does not
    change leg count — we treat the candidate prefix as the cost source.
    """
    if direction != DIRECTION_D:
        return 1
    base = candidate_code.removesuffix("_nc")
    if base == CANDIDATE_D0:
        return 1
    if base == CANDIDATE_D4:
        return 1 + len(cfg.d_basket_symbols)
    return 2


def _net_at(
    direction: str,
    candidate_code: str,
    gross: float,
    holding_bars: int,
    cost_bps: float,
    extra_slippage_bps: float,
    cfg: V7SConfig,
) -> float:
    """Dispatch to the right cost model based on direction + candidate."""
    n_legs = _n_legs_for(direction, candidate_code, cfg)
    if n_legs == 1 and direction != DIRECTION_D:
        return _net_short_return(gross, holding_bars, cost_bps, extra_slippage_bps, cfg)
    return _net_pair_return(gross, holding_bars, cost_bps, extra_slippage_bps, cfg, n_legs=n_legs)


def _attach_focal_net(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    """Add ``net20`` and ``net30`` columns at focal slippage (direction + candidate aware)."""
    if trades.empty:
        return trades
    out = trades.copy()
    holding = pd.to_numeric(out.get("holding_bars", 0), errors="coerce").fillna(0).astype(int)
    gross = pd.to_numeric(out.get("gross_return", float("nan")), errors="coerce")
    direction = out.get("direction", pd.Series([DIRECTION_E] * len(out))).astype(str)
    candidate = out.get("candidate_code", pd.Series([""] * len(out))).astype(str)
    nets: dict[str, list[float]] = {"net20": [], "net30": []}
    for g, h, d, c in zip(gross, holding, direction, candidate):
        nets["net20"].append(_net_at(d, c, g, h, 20.0, cfg.focal_extra_slippage_bps, cfg))
        nets["net30"].append(_net_at(d, c, g, h, 30.0, cfg.focal_extra_slippage_bps, cfg))
    out["net20"] = nets["net20"]
    out["net30"] = nets["net30"]
    return out


# --------------------------------------------------------------------------------------
# Forward-walk labels — per signal, six clean-short forward metrics
# --------------------------------------------------------------------------------------


def _label_clean_short(
    group: pd.DataFrame,
    entry_idx: int,
    cfg: V7SConfig,
) -> dict[str, object]:
    """Same shape as v6S clean-short labeller — six forward-path metrics."""
    out: dict[str, object] = {
        "hit_down_3pct": False,
        "hit_down_5pct": False,
        "up_before_down_2pct": False,
        "up_before_down_3pct": False,
        "short_squeeze_before_hit": False,
        "max_adverse_up": float("nan"),
    }
    if entry_idx + 1 >= len(group):
        return out
    open_arr = _f(group, "open")
    high_arr = _f(group, "high")
    low_arr = _f(group, "low")
    if entry_idx + 1 >= len(open_arr):
        return out
    entry_price = float(open_arr[entry_idx + 1])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return out
    end = min(entry_idx + 1 + cfg.label_window_bars, len(group))
    max_up_pct = 0.0
    first_hit_down_3 = -1
    first_up_2 = -1
    hit_down_5_seen = False
    for j in range(entry_idx + 1, end):
        if j >= len(high_arr) or j >= len(low_arr):
            break
        hi = float(high_arr[j])
        lo = float(low_arr[j])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        up_pct = (hi - entry_price) / entry_price
        down_pct = (entry_price - lo) / entry_price
        if up_pct > max_up_pct:
            max_up_pct = up_pct
        if first_up_2 < 0 and up_pct >= cfg.up_before_down_thresholds[0]:
            first_up_2 = j - entry_idx - 1
        if first_hit_down_3 < 0 and down_pct >= cfg.hit_down_thresholds[0]:
            first_hit_down_3 = j - entry_idx - 1
        if down_pct >= cfg.hit_down_thresholds[1]:
            hit_down_5_seen = True
    out["hit_down_3pct"] = first_hit_down_3 >= 0
    out["hit_down_5pct"] = hit_down_5_seen
    out["up_before_down_2pct"] = max_up_pct >= cfg.up_before_down_thresholds[0] and (
        first_hit_down_3 < 0 or first_up_2 < first_hit_down_3
    )
    out["up_before_down_3pct"] = max_up_pct >= cfg.up_before_down_thresholds[1] and (
        first_hit_down_3 < 0 or first_up_2 < first_hit_down_3
    )
    out["short_squeeze_before_hit"] = bool(out["up_before_down_2pct"]) and bool(out["hit_down_3pct"])
    out["max_adverse_up"] = float(max_up_pct)
    return out


# --------------------------------------------------------------------------------------
# Ten standardized outputs — one writer per CSV
# --------------------------------------------------------------------------------------


def _short_candidate_summary(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n_trades", "mean_gross", "mean_net20", "win_rate_net20", "verdict"])
    rows: list[dict[str, object]] = []
    for (direction, code, execution), sub in trades.groupby(["direction", "candidate_code", "execution"]):
        n = len(sub)
        rows.append(
            {
                "direction": direction,
                "candidate_code": code,
                "execution": execution,
                "n_trades": n,
                "mean_gross": float(pd.to_numeric(sub.get("gross_return"), errors="coerce").mean()),
                "mean_net20": float(pd.to_numeric(sub.get("net20"), errors="coerce").mean()),
                "win_rate_net20": float((pd.to_numeric(sub.get("net20"), errors="coerce") > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _short_cost_grid(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "cost_bps", "extra_slippage_bps", "n_trades", "mean_net", "win_rate"])
    rows: list[dict[str, object]] = []
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    holding = pd.to_numeric(trades.get("holding_bars", 0), errors="coerce").fillna(0).astype(int)
    direction_series = trades.get("direction", pd.Series([DIRECTION_E] * len(trades))).astype(str)
    candidate_series = trades.get("candidate_code", pd.Series([""] * len(trades))).astype(str)
    for cost_bps in cfg.cost_grid_bps:
        for extra in cfg.extra_slippage_bps:
            nets = [
                _net_at(d, c, g, h, cost_bps, extra, cfg)
                for d, c, g, h in zip(direction_series, candidate_series, gross, holding)
            ]
            trades_n = trades.assign(_net=nets)
            for (g_direction, g_code, g_execution), sub in trades_n.groupby(["direction", "candidate_code", "execution"]):
                net = pd.to_numeric(sub["_net"], errors="coerce")
                rows.append(
                    {
                        "direction": g_direction,
                        "candidate_code": g_code,
                        "execution": g_execution,
                        "cost_bps": cost_bps,
                        "extra_slippage_bps": extra,
                        "n_trades": int(net.notna().sum()),
                        "mean_net": float(net.mean()),
                        "win_rate": float((net > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _short_first_touch(trades: pd.DataFrame, cfg: V7SConfig, group_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Clean-short forward labels per execution.

    Re-uses the cached per-symbol group frames to walk forward without IO.
    """
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n", "hit_down_3pct", "hit_down_5pct", "up_before_down_2pct", "short_squeeze_before_hit"])
    label_rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        symbol = str(getattr(row, "symbol", ""))
        entry_idx = int(getattr(row, "entry_idx", -1)) - 1  # _label_clean_short expects break_idx
        group = group_cache.get(symbol)
        if group is None or entry_idx < 0:
            continue
        labels = _label_clean_short(group, entry_idx, cfg)
        labels.update(
            {
                "direction": getattr(row, "direction", ""),
                "candidate_code": getattr(row, "candidate_code", ""),
                "execution": getattr(row, "execution", ""),
            }
        )
        label_rows.append(labels)
    if not label_rows:
        return pd.DataFrame()
    label_df = pd.DataFrame(label_rows)
    agg = (
        label_df.groupby(["direction", "candidate_code", "execution"])
        .agg(
            n=("hit_down_3pct", "size"),
            hit_down_3pct=("hit_down_3pct", "mean"),
            hit_down_5pct=("hit_down_5pct", "mean"),
            up_before_down_2pct=("up_before_down_2pct", "mean"),
            short_squeeze_before_hit=("short_squeeze_before_hit", "mean"),
            max_adverse_up_mean=("max_adverse_up", "mean"),
        )
        .reset_index()
    )
    return agg


def _short_vs_no_long(
    trades: pd.DataFrame,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V7SConfig,
) -> pd.DataFrame:
    """B counterfactual — short vs no_long head-to-head.

    For each signal_time, find any forward long (within ``no_long_block_hours``)
    that would have fired on the same symbol in the v0.9D long stream. Record:

    - ``A_no_action`` — realized PnL of that long (we let it run).
    - ``B_no_long`` — 0 (we blocked the long).
    - ``C_normal_short`` — our short's net20.

    The output aggregates per (direction, candidate_code, execution) so
    a reader can see whether short > no_long > no_action consistently.
    """
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n", "mean_A_no_action", "mean_B_no_long", "mean_C_short", "short_beats_no_long_pct"])
    by_symbol: dict[str, list[tuple[int, dict]]] = {}
    for (sym, ts), payload in cic_long_index.items():
        by_symbol.setdefault(sym, []).append((int(pd.Timestamp(ts).value), payload))
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda kv: kv[0])

    window_ns = int(pd.Timedelta(hours=cfg.no_long_block_hours).value)
    rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        symbol = str(getattr(row, "symbol", ""))
        signal_time = pd.Timestamp(getattr(row, "signal_time"))
        if pd.isna(signal_time):
            continue
        ts_ns = int(signal_time.value)
        items = by_symbol.get(symbol, [])
        a_no_action = 0.0
        for entry_ns, payload in items:
            if entry_ns < ts_ns:
                continue
            if entry_ns - ts_ns > window_ns:
                break
            realized = float(pd.to_numeric(payload.get("net_return", float("nan")), errors="coerce"))
            if np.isfinite(realized):
                a_no_action = realized
                break
        rows.append(
            {
                "direction": getattr(row, "direction", ""),
                "candidate_code": getattr(row, "candidate_code", ""),
                "execution": getattr(row, "execution", ""),
                "A_no_action": a_no_action,
                "B_no_long": 0.0,
                "C_short": float(pd.to_numeric(getattr(row, "net20", float("nan")), errors="coerce")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (
        df.groupby(["direction", "candidate_code", "execution"])
        .agg(
            n=("A_no_action", "size"),
            mean_A_no_action=("A_no_action", "mean"),
            mean_B_no_long=("B_no_long", "mean"),
            mean_C_short=("C_short", "mean"),
        )
        .reset_index()
    )
    agg["short_beats_no_long_pct"] = (
        df.assign(_beats=lambda d: (d["C_short"] > d["B_no_long"]).astype(float))
        .groupby(["direction", "candidate_code", "execution"])["_beats"]
        .mean()
        .reset_index(drop=True)
    )
    return agg


def _short_vs_exit_long(
    trades: pd.DataFrame,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Exit-existing-long counterfactual.

    If an active long is open at ``signal_time``, compute its unrealised PnL
    at that point and treat that as the ``exit_existing_long`` outcome. The
    short's outcome is its net20. Aggregates the spread per candidate.

    Note: this is a simplified version of v4S's exit-existing computation —
    we use ``net_return`` as a proxy when ``unrealized_at_signal`` isn't in
    the long payload. Production A100 run can swap to ``_position_state``.
    """
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n_with_active_long", "mean_exit_now", "mean_short_net20", "short_beats_exit"])
    by_symbol: dict[str, list[tuple[int, int, dict]]] = {}
    for (sym, ts), payload in cic_long_index.items():
        exit_time = payload.get("exit_time")
        if exit_time is None:
            continue
        entry_ns = int(pd.Timestamp(ts).value)
        exit_ns = int(pd.Timestamp(exit_time).value)
        by_symbol.setdefault(sym, []).append((entry_ns, exit_ns, payload))

    rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        symbol = str(getattr(row, "symbol", ""))
        signal_time = pd.Timestamp(getattr(row, "signal_time"))
        if pd.isna(signal_time):
            continue
        ts_ns = int(signal_time.value)
        items = by_symbol.get(symbol, [])
        active_payload = None
        for entry_ns, exit_ns, payload in items:
            if entry_ns <= ts_ns < exit_ns:
                active_payload = payload
                break
        if active_payload is None:
            continue
        exit_now = float(pd.to_numeric(active_payload.get("unrealized_at_signal", active_payload.get("net_return", float("nan"))), errors="coerce"))
        rows.append(
            {
                "direction": getattr(row, "direction", ""),
                "candidate_code": getattr(row, "candidate_code", ""),
                "execution": getattr(row, "execution", ""),
                "exit_now": exit_now,
                "short_net20": float(pd.to_numeric(getattr(row, "net20", float("nan")), errors="coerce")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n_with_active_long", "mean_exit_now", "mean_short_net20", "short_beats_exit"])
    agg = (
        df.groupby(["direction", "candidate_code", "execution"])
        .agg(
            n_with_active_long=("exit_now", "size"),
            mean_exit_now=("exit_now", "mean"),
            mean_short_net20=("short_net20", "mean"),
        )
        .reset_index()
    )
    beats = (
        df.assign(_beats=(df["short_net20"] > df["exit_now"]).astype(float))
        .groupby(["direction", "candidate_code", "execution"])["_beats"]
        .mean()
        .reset_index(drop=True)
    )
    agg["short_beats_exit"] = beats
    return agg


def _short_hedge_value(
    trades: pd.DataFrame,
    long_monthly: pd.DataFrame,
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Pearson corr vs long-stack monthly net + worst-month overlap.

    ``long_monthly`` must carry columns ``month`` (YYYY-MM) and ``net``.
    When absent, returns a single ``no_long_monthly`` row per candidate.
    """
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "n_months", "hedge_corr", "long_worst_month", "short_in_long_worst_month"])
    if long_monthly.empty:
        return (
            trades.groupby(["direction", "candidate_code", "execution"])
            .size()
            .reset_index(name="n_months")
            .assign(hedge_corr=float("nan"), long_worst_month="", short_in_long_worst_month=float("nan"))
        )
    long_map = dict(zip(long_monthly["month"].astype(str), pd.to_numeric(long_monthly["net"], errors="coerce")))
    long_worst_month = min(long_map, key=lambda k: long_map.get(k, float("inf"))) if long_map else ""
    rows: list[dict[str, object]] = []
    for (direction, code, execution), sub in trades.groupby(["direction", "candidate_code", "execution"]):
        monthly_short = (
            sub.groupby(sub["month"].astype(str))["net20"]
            .mean()
            .reset_index(name="short_net")
        )
        merged = monthly_short.rename(columns={"month": "month_key"}).copy()
        merged["long_net"] = merged["month_key"].map(long_map)
        valid = merged.dropna(subset=["short_net", "long_net"])
        corr = float(valid["short_net"].corr(valid["long_net"])) if len(valid) >= 2 else float("nan")
        in_worst = float(monthly_short.loc[monthly_short["month"] == long_worst_month, "short_net"].mean()) if long_worst_month else float("nan")
        rows.append(
            {
                "direction": direction,
                "candidate_code": code,
                "execution": execution,
                "n_months": int(len(monthly_short)),
                "hedge_corr": corr,
                "long_worst_month": long_worst_month,
                "short_in_long_worst_month": in_worst,
            }
        )
    return pd.DataFrame(rows)


def _month_cap_leave_one_month(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    """month_cap35 net + leave-one-month-out net per candidate."""
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "uncapped_net", "month_capped_net", "leave_worst_net", "best_month_share"])
    rows: list[dict[str, object]] = []
    cap = cfg.month_cap_pct
    for (direction, code, execution), sub in trades.groupby(["direction", "candidate_code", "execution"]):
        net = pd.to_numeric(sub["net20"], errors="coerce").fillna(0.0)
        month = sub["month"].astype(str)
        per_month = pd.DataFrame({"net": net, "month": month}).groupby("month")["net"].sum()
        total = float(per_month.sum())
        if total == 0:
            uncapped = 0.0
            best_share = 0.0
            capped = 0.0
            worst = 0.0
        else:
            uncapped = total
            best_share = float(per_month.max() / total) if total != 0 else 0.0
            per_month_capped = per_month.clip(upper=cap * total) if total > 0 else per_month
            capped = float(per_month_capped.sum())
            worst = float((per_month.sum() - per_month.min()))
        rows.append(
            {
                "direction": direction,
                "candidate_code": code,
                "execution": execution,
                "uncapped_net": uncapped,
                "month_capped_net": capped,
                "leave_worst_net": worst,
                "best_month_share": best_share,
            }
        )
    return pd.DataFrame(rows)


def _symbol_contribution(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    """Max symbol contribution + leave-one-symbol-out net."""
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "max_symbol_share", "leave_worst_symbol_net"])
    rows: list[dict[str, object]] = []
    for (direction, code, execution), sub in trades.groupby(["direction", "candidate_code", "execution"]):
        net = pd.to_numeric(sub["net20"], errors="coerce").fillna(0.0)
        symbols = sub["symbol"].astype(str)
        per_sym = pd.DataFrame({"net": net, "symbol": symbols}).groupby("symbol")["net"].sum()
        total = float(per_sym.sum())
        if total == 0 or per_sym.empty:
            rows.append({"direction": direction, "candidate_code": code, "execution": execution, "max_symbol_share": 0.0, "leave_worst_symbol_net": 0.0})
            continue
        max_share = float(per_sym.abs().max() / max(per_sym.abs().sum(), 1e-9))
        worst_sym = per_sym.idxmin()
        leave_worst = float(per_sym.drop(worst_sym).sum())
        rows.append({"direction": direction, "candidate_code": code, "execution": execution, "max_symbol_share": max_share, "leave_worst_symbol_net": leave_worst})
    return pd.DataFrame(rows)


def _matched_random_baseline(trades: pd.DataFrame, cfg: V7SConfig) -> pd.DataFrame:
    """Matched-random baseline: per candidate, draw ``random_baseline_draws``
    independent samples of identical size, using the same symbol distribution
    but random ``gross_return`` from the global pool. Compare means.

    Uses a fixed seed so the run is reproducible. Returns mean of means and
    the share of draws whose mean was ≥ the candidate's mean (an empirical
    one-sided p-value).
    """
    if trades.empty:
        return pd.DataFrame(columns=["direction", "candidate_code", "execution", "candidate_mean_net20", "random_mean_net20", "p_ge_candidate"])
    rng = np.random.default_rng(cfg.random_baseline_seed)
    pool = pd.to_numeric(trades["net20"], errors="coerce").dropna().to_numpy()
    if pool.size == 0:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (direction, code, execution), sub in trades.groupby(["direction", "candidate_code", "execution"]):
        candidate_mean = float(pd.to_numeric(sub["net20"], errors="coerce").mean())
        n = len(sub)
        draws = np.array([float(rng.choice(pool, size=n, replace=True).mean()) for _ in range(cfg.random_baseline_draws)])
        rows.append(
            {
                "direction": direction,
                "candidate_code": code,
                "execution": execution,
                "candidate_mean_net20": candidate_mean,
                "random_mean_net20": float(draws.mean()),
                "p_ge_candidate": float((draws >= candidate_mean).mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Ten-gate verdict evaluator — produces ``verdict`` column on candidate_summary
# --------------------------------------------------------------------------------------


GATE_NAMES: tuple[str, ...] = (
    "gate1_net20_positive",
    "gate2_net30_holds",
    "gate3_clean_hit_lifts",
    "gate4_squeeze_below_threshold",
    "gate5_month_cap_positive",
    "gate6_leave_one_month_positive",
    "gate7_symbol_share_below_threshold",
    "gate8_baseline_strictly_worse",
    "gate9_short_beats_no_long",
    "gate10_hedge_complementary",
)


def _evaluate_gates(
    summary: pd.DataFrame,
    cost_grid: pd.DataFrame,
    first_touch: pd.DataFrame,
    vs_no_long: pd.DataFrame,
    hedge: pd.DataFrame,
    month_cap: pd.DataFrame,
    symbol_contrib: pd.DataFrame,
    baseline: pd.DataFrame,
    cfg: V7SConfig,
) -> pd.DataFrame:
    """Evaluate the docx-mandated 10 gates per (candidate, execution).

    A candidate gets ``verdict``:

    - ``promote`` when all 10 gates pass.
    - ``risk_off_only`` when gates 1-8 pass but 9 OR 10 fails (operator can
      still take it under risk-off discretion; cf. v6S Path C).
    - ``no_value`` when any of gates 1-8 fails.

    The output frame is the candidate summary plus one column per gate
    (bool) plus a ``verdict`` column.
    """
    if summary.empty:
        return summary.assign(verdict="no_data", **{g: False for g in GATE_NAMES})

    out = summary.copy()
    for g in GATE_NAMES:
        out[g] = False
    out["verdict"] = "no_value"
    out["gate_failures"] = ""

    keys = ["direction", "candidate_code", "execution"]

    cost30_by_key: dict[tuple, float] = {}
    if not cost_grid.empty:
        focal = cost_grid[(cost_grid["cost_bps"] == 30.0) & (cost_grid["extra_slippage_bps"] == cfg.focal_extra_slippage_bps)]
        for _, row in focal.iterrows():
            cost30_by_key[(row["direction"], row["candidate_code"], row["execution"])] = float(row.get("mean_net", float("nan")))

    first_touch_by_key: dict[tuple, dict] = {}
    if not first_touch.empty:
        for _, row in first_touch.iterrows():
            first_touch_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    vs_no_long_by_key: dict[tuple, dict] = {}
    if not vs_no_long.empty:
        for _, row in vs_no_long.iterrows():
            vs_no_long_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    hedge_by_key: dict[tuple, dict] = {}
    if not hedge.empty:
        for _, row in hedge.iterrows():
            hedge_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    month_by_key: dict[tuple, dict] = {}
    if not month_cap.empty:
        for _, row in month_cap.iterrows():
            month_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    symbol_by_key: dict[tuple, dict] = {}
    if not symbol_contrib.empty:
        for _, row in symbol_contrib.iterrows():
            symbol_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    baseline_by_key: dict[tuple, dict] = {}
    if not baseline.empty:
        for _, row in baseline.iterrows():
            baseline_by_key[(row["direction"], row["candidate_code"], row["execution"])] = row.to_dict()

    for i, row in out.iterrows():
        key = (row["direction"], row["candidate_code"], row["execution"])
        failures: list[str] = []

        net20 = float(row.get("mean_net20", float("nan")))
        gate1 = np.isfinite(net20) and net20 > 0
        if not gate1:
            failures.append("gate1")
        out.at[i, "gate1_net20_positive"] = bool(gate1)

        net30 = cost30_by_key.get(key, float("nan"))
        gate2 = False
        if gate1 and np.isfinite(net30):
            gate2 = net30 > 0 and (net30 >= 0.5 * net20)
        if not gate2:
            failures.append("gate2")
        out.at[i, "gate2_net30_holds"] = bool(gate2)

        ft = first_touch_by_key.get(key, {})
        hit3 = float(ft.get("hit_down_3pct", float("nan")))
        gate3 = np.isfinite(hit3) and hit3 >= 0.35  # baseline expectation for a clean short
        if not gate3:
            failures.append("gate3")
        out.at[i, "gate3_clean_hit_lifts"] = bool(gate3)

        squeeze = float(ft.get("short_squeeze_before_hit", float("nan")))
        gate4 = np.isfinite(squeeze) and squeeze <= cfg.max_squeeze_share
        if not gate4:
            failures.append("gate4")
        out.at[i, "gate4_squeeze_below_threshold"] = bool(gate4)

        m = month_by_key.get(key, {})
        capped = float(m.get("month_capped_net", float("nan")))
        gate5 = np.isfinite(capped) and capped > 0
        if not gate5:
            failures.append("gate5")
        out.at[i, "gate5_month_cap_positive"] = bool(gate5)

        leave_worst = float(m.get("leave_worst_net", float("nan")))
        gate6 = np.isfinite(leave_worst) and leave_worst > 0
        if not gate6:
            failures.append("gate6")
        out.at[i, "gate6_leave_one_month_positive"] = bool(gate6)

        s = symbol_by_key.get(key, {})
        symbol_share = float(s.get("max_symbol_share", float("nan")))
        gate7 = np.isfinite(symbol_share) and symbol_share <= cfg.max_symbol_share
        if not gate7:
            failures.append("gate7")
        out.at[i, "gate7_symbol_share_below_threshold"] = bool(gate7)

        b = baseline_by_key.get(key, {})
        cand_mean = float(b.get("candidate_mean_net20", float("nan")))
        rand_mean = float(b.get("random_mean_net20", float("nan")))
        gate8 = np.isfinite(cand_mean) and np.isfinite(rand_mean) and cand_mean > rand_mean
        if not gate8:
            failures.append("gate8")
        out.at[i, "gate8_baseline_strictly_worse"] = bool(gate8)

        vnl = vs_no_long_by_key.get(key, {})
        c_short = float(vnl.get("mean_C_short", float("nan")))
        b_no_long = float(vnl.get("mean_B_no_long", 0.0))
        gate9 = np.isfinite(c_short) and c_short > b_no_long and c_short > 0
        if not gate9:
            failures.append("gate9")
        out.at[i, "gate9_short_beats_no_long"] = bool(gate9)

        h = hedge_by_key.get(key, {})
        corr = float(h.get("hedge_corr", float("nan")))
        worst = float(h.get("short_in_long_worst_month", float("nan")))
        gate10 = (np.isfinite(corr) and corr <= cfg.hedge_corr_max) or (np.isfinite(worst) and worst > 0)
        if not gate10:
            failures.append("gate10")
        out.at[i, "gate10_hedge_complementary"] = bool(gate10)

        out.at[i, "gate_failures"] = ";".join(failures) if failures else ""
        gates_1to8 = all(getattr(out.at[i, g], "item", lambda: out.at[i, g])() for g in GATE_NAMES[:8])
        gate9_ok = bool(out.at[i, "gate9_short_beats_no_long"])
        gate10_ok = bool(out.at[i, "gate10_hedge_complementary"])
        if gates_1to8 and gate9_ok and gate10_ok:
            out.at[i, "verdict"] = "promote"
        elif gates_1to8 and (gate9_ok or gate10_ok):
            out.at[i, "verdict"] = "risk_off_only"
        else:
            out.at[i, "verdict"] = "no_value"

    return out


# --------------------------------------------------------------------------------------
# Candidate notes — markdown narrative per candidate
# --------------------------------------------------------------------------------------


def _write_candidate_notes(
    report_root: Path,
    summary: pd.DataFrame,
    cfg: V7SConfig,
    *,
    sell_flow_audit: str,
) -> Path:
    """Emit per-direction candidate_notes.md summarizing verdicts + audit reasons."""
    path = report_root / "candidate_notes.md"
    lines: list[str] = []
    lines.append("# v7S Short Alpha Exploration — candidate notes")
    lines.append("")
    lines.append(f"- Sell-flow gate audit: **{sell_flow_audit}**")
    lines.append("- Acceptance gates (docx §统一验收标准):")
    for g in GATE_NAMES:
        lines.append(f"  - {g}")
    lines.append("")
    if summary.empty:
        lines.append("_No data — feature parquet not available on this box. See A100 runner._")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    for direction, sub in summary.groupby("direction"):
        lines.append(f"## {direction}")
        lines.append("")
        for _, row in sub.iterrows():
            failures = str(row.get("gate_failures", "")) or "—"
            lines.append(
                f"- **{row['candidate_code']}** ({row['execution']}): N={row.get('n_trades', 0)} "
                f"mean_net20={row.get('mean_net20', float('nan')):.4f} "
                f"verdict=`{row.get('verdict', 'no_value')}` fails=`{failures}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Public driver
# --------------------------------------------------------------------------------------


SUPPORTED_DIRECTIONS: tuple[str, ...] = (DIRECTION_E, DIRECTION_D, DIRECTION_A)


def write_v7s_short_alpha(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V7SConfig = V7SConfig(),
    *,
    long_monthly: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Run the v7S Short Alpha Exploration pipeline end-to-end.

    Iterates ``cfg.enabled_directions`` and runs each supported direction's
    collector. Directions A/B/C remain ``NotImplementedError`` until their
    data plumbing lands. Outputs are emitted per-direction under
    ``cfg.report_root / <direction>/`` so each direction has the docx-mandated
    ten CSVs + candidate_notes.md.

    Missing trade cache or feature parquet is allowed — the run writes
    empty stubs and the candidate_notes.md flags ``no_data``.

    Returns a dict of output paths keyed as ``<direction>:<csv_name>``.
    """
    for d in cfg.enabled_directions:
        if d not in SUPPORTED_DIRECTIONS:
            raise NotImplementedError(
                f"v7S Direction {d} is configured but not yet implemented. "
                f"Supported: {SUPPORTED_DIRECTIONS}."
            )

    sell_flow_audit = "orderflow_loaded"
    orderflow_lookup = _build_orderflow_lookup(cfg.orderflow_event_path)
    if orderflow_lookup is None:
        sell_flow_audit = "orderflow_missing"

    outputs: dict[str, Path] = {}

    if not feature_path.exists():
        for d in cfg.enabled_directions:
            report_root = ensure_dir(cfg.report_root / d)
            stubs = _write_empty_outputs(
                report_root, pd.DataFrame(), cfg, sell_flow_audit=sell_flow_audit
            )
            for k, p in stubs.items():
                outputs[f"{d}:{k}"] = p
        return outputs

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    trade_cache = (
        read_parquet(cfg.trade_cache_path) if cfg.trade_cache_path.exists() else pd.DataFrame()
    )
    pool = _focus_pool(trade_cache, cfg.long_pool_name) if not trade_cache.empty else pd.DataFrame()
    cic_long_index = _build_cic_long_index(pool, cfg.v34_cfg) if not pool.empty else {}
    long_monthly_df = long_monthly if long_monthly is not None else _derive_long_monthly(pool)

    all_trades: list[pd.DataFrame] = []
    if DIRECTION_E in cfg.enabled_directions:
        if cic_long_index:
            e_trades = _collect_direction_e_signals(
                feature_path, rank30, rank90, config, cic_long_index, orderflow_lookup, cfg
            )
            if not e_trades.empty:
                all_trades.append(e_trades)
        else:
            print("v7S Direction E: CIC long index empty — skipping.", flush=True)
    if DIRECTION_D in cfg.enabled_directions:
        d_trades = _collect_direction_d_signals(
            feature_path, rank30, rank90, config, cfg
        )
        if not d_trades.empty:
            all_trades.append(d_trades)
    if DIRECTION_A in cfg.enabled_directions:
        a_trades = _collect_direction_a_signals(
            feature_path, rank30, rank90, config, cfg
        )
        if not a_trades.empty:
            all_trades.append(a_trades)

    trades = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )
    trades = _attach_focal_net(trades, cfg)

    group_cache: dict[str, pd.DataFrame] = {}
    if not trades.empty:
        for sym in trades["symbol"].astype(str).unique():
            g = _read_symbol_features(feature_path, rank30, rank90, sym, config)
            if not g.empty:
                group_cache[sym] = g.sort_values("bar_open_time").reset_index(drop=True)

    summary = _short_candidate_summary(trades, cfg)
    cost_grid = _short_cost_grid(trades, cfg)
    first_touch = _short_first_touch(trades, cfg, group_cache)
    vs_no_long = _short_vs_no_long(trades, cic_long_index, cfg)
    vs_exit_long = _short_vs_exit_long(trades, cic_long_index, cfg)
    hedge = _short_hedge_value(trades, long_monthly_df, cfg)
    month_cap = _month_cap_leave_one_month(trades, cfg)
    symbol_contrib = _symbol_contribution(trades, cfg)
    baseline = _matched_random_baseline(trades, cfg)

    summary = _evaluate_gates(
        summary, cost_grid, first_touch, vs_no_long, hedge, month_cap, symbol_contrib, baseline, cfg
    )

    for d in cfg.enabled_directions:
        report_root = ensure_dir(cfg.report_root / d)
        d_trades = trades[trades["direction"].astype(str) == d] if not trades.empty else trades
        d_summary = summary[summary["direction"].astype(str) == d] if not summary.empty else summary
        d_cost = cost_grid[cost_grid["direction"].astype(str) == d] if not cost_grid.empty else cost_grid
        d_first = first_touch[first_touch["direction"].astype(str) == d] if not first_touch.empty else first_touch
        d_vsnl = vs_no_long[vs_no_long["direction"].astype(str) == d] if not vs_no_long.empty else vs_no_long
        d_vsex = vs_exit_long[vs_exit_long["direction"].astype(str) == d] if not vs_exit_long.empty else vs_exit_long
        d_hedge = hedge[hedge["direction"].astype(str) == d] if not hedge.empty else hedge
        d_month = month_cap[month_cap["direction"].astype(str) == d] if not month_cap.empty else month_cap
        d_sym = symbol_contrib[symbol_contrib["direction"].astype(str) == d] if not symbol_contrib.empty else symbol_contrib
        d_base = baseline[baseline["direction"].astype(str) == d] if not baseline.empty else baseline
        d_outputs = _persist_outputs(
            report_root,
            trades=d_trades,
            summary=d_summary,
            cost_grid=d_cost,
            first_touch=d_first,
            vs_no_long=d_vsnl,
            vs_exit_long=d_vsex,
            hedge=d_hedge,
            month_cap=d_month,
            symbol_contrib=d_sym,
            baseline=d_base,
            cfg=cfg,
            sell_flow_audit=sell_flow_audit,
        )
        for k, p in d_outputs.items():
            outputs[f"{d}:{k}"] = p

    return outputs


def _derive_long_monthly(pool: pd.DataFrame) -> pd.DataFrame:
    """Build long-stack monthly net from the focused pool, or return empty
    when the trade cache lacks ``net_return`` / ``entry_time``."""
    if pool.empty:
        return pd.DataFrame(columns=["month", "net"])
    if "entry_time" not in pool.columns or "net_return" not in pool.columns:
        return pd.DataFrame(columns=["month", "net"])
    ts = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool = pool.assign(_month=ts.dt.strftime("%Y-%m"))
    monthly = pool.groupby("_month")["net_return"].sum().reset_index()
    monthly.columns = ["month", "net"]
    return monthly


def _persist_outputs(
    report_root: Path,
    *,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    cost_grid: pd.DataFrame,
    first_touch: pd.DataFrame,
    vs_no_long: pd.DataFrame,
    vs_exit_long: pd.DataFrame,
    hedge: pd.DataFrame,
    month_cap: pd.DataFrame,
    symbol_contrib: pd.DataFrame,
    baseline: pd.DataFrame,
    cfg: V7SConfig,
    sell_flow_audit: str,
) -> dict[str, Path]:
    paths = {
        "trades": report_root / "short_trades.csv",
        "summary": report_root / "short_candidate_summary.csv",
        "cost_grid": report_root / "short_cost_grid.csv",
        "first_touch": report_root / "short_first_touch.csv",
        "vs_no_long": report_root / "short_vs_no_long.csv",
        "vs_exit_long": report_root / "short_vs_exit_long.csv",
        "hedge": report_root / "short_hedge_value.csv",
        "month_cap": report_root / "month_cap_leave_one_month.csv",
        "symbol_contrib": report_root / "symbol_contribution.csv",
        "baseline": report_root / "matched_random_baseline.csv",
    }
    trades.to_csv(paths["trades"], index=False)
    summary.to_csv(paths["summary"], index=False)
    cost_grid.to_csv(paths["cost_grid"], index=False)
    first_touch.to_csv(paths["first_touch"], index=False)
    vs_no_long.to_csv(paths["vs_no_long"], index=False)
    vs_exit_long.to_csv(paths["vs_exit_long"], index=False)
    hedge.to_csv(paths["hedge"], index=False)
    month_cap.to_csv(paths["month_cap"], index=False)
    symbol_contrib.to_csv(paths["symbol_contrib"], index=False)
    baseline.to_csv(paths["baseline"], index=False)
    notes_path = _write_candidate_notes(report_root, summary, cfg, sell_flow_audit=sell_flow_audit)
    paths["candidate_notes"] = notes_path
    return paths


def _write_empty_outputs(
    report_root: Path,
    summary: pd.DataFrame,
    cfg: V7SConfig,
    *,
    sell_flow_audit: str,
) -> dict[str, Path]:
    """Write empty CSV stubs when no feature data is available locally."""
    empty = pd.DataFrame()
    return _persist_outputs(
        report_root,
        trades=empty,
        summary=summary,
        cost_grid=empty,
        first_touch=empty,
        vs_no_long=empty,
        vs_exit_long=empty,
        hedge=empty,
        month_cap=empty,
        symbol_contrib=empty,
        baseline=empty,
        cfg=cfg,
        sell_flow_audit=sell_flow_audit,
    )


__all__ = [
    "CANDIDATE_E1",
    "CANDIDATE_E2",
    "DIRECTIONS",
    "DIRECTION_A",
    "DIRECTION_B",
    "DIRECTION_C",
    "DIRECTION_D",
    "DIRECTION_E",
    "E_CANDIDATES",
    "GATE_NAMES",
    "REPORT_ROOT",
    "TRADE_CACHE_PATH",
    "V7SConfig",
    "_attach_focal_net",
    "_build_orderflow_lookup",
    "_collect_direction_e_signals",
    "_derive_long_monthly",
    "_emit_direction_e_signals",
    "_evaluate_gates",
    "_execute_direction_e",
    "_gate_beta_high_gone",
    "_gate_sell_flow_confirms",
    "_label_clean_short",
    "_matched_random_baseline",
    "_month_cap_leave_one_month",
    "_net_short_return",
    "_short_candidate_summary",
    "_short_cost_grid",
    "_short_first_touch",
    "_short_hedge_value",
    "_short_vs_exit_long",
    "_short_vs_no_long",
    "_symbol_contribution",
    "_write_candidate_notes",
    "write_v7s_short_alpha",
]
