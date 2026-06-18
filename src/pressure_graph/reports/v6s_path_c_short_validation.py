"""v6S Path C Short Validation — discipline-grade test of the one v4S survivor.

v4S identified exactly one short cell that beats `no_long`:
``crowded_long_stall + BTC weakness + Swing rule`` (N=63, mean +1.52%, win
61.9% at 20 bps focal cost). v6S is the discipline round: take three fixed
candidates atop the same Path C event stream and verify whether the alpha
holds under the instructment5 §discipline matrix.

**Three fixed candidates (do NOT expand)**:

- ``S_C1`` — crowded stall + BTC weakness, normal_short (1.0× size), Swing exit.
- ``S_C2`` — same, small_short (0.5× size), Swing exit.
- ``S_C3a`` — same trigger, passive ``no_long`` action (no short opened;
  outcome is the realised P&L of any new long blocked in a 12h forward
  window). The critical control — only graduate S-C1/S-C2 to paper-shadow if
  they clearly beat S-C3a.
- ``S_C3b`` — same trigger, passive ``no_overflow`` action (outcome = the
  realised P&L of any O6-overflow long blocked in the same window). Lighter
  cut than S-C3a, often equally protective.

**Discipline applied per candidate**:

1. Cost pressure — 10 / 20 / 30 / 50 bps × extra slippage 0 / 5 / 10 bps.
   Funding accrual baked in: at funding ≥ 70 pct, short receives ≈30% APR
   funding per ``funding_apr_assumption`` (configurable). Reject if positive
   only at the cheapest cell.
2. Monthly stability — month-cap 35%, leave-one-month-out, worst / best
   month contribution. With N=63 a single month can drive the result.
3. Symbol stability — max symbol contribution, leave-one-symbol-out. Same
   reason.
4. Clean short labels — six forward-path metrics per event:
   ``hit_down_3pct``, ``hit_down_5pct``, ``up_before_down_2pct``,
   ``up_before_down_3pct``, ``short_squeeze_before_hit``, ``max_adverse_up``.
   The squeeze risk question: did price rip up before resolving down?
5. ``no_long`` head-to-head — per event, compare ``A no_action`` (let any
   matched long ride), ``B no_long`` (block forward longs), ``C normal_short``
   (S-C1), ``D small_short`` (S-C2). Promote a short only if it clearly beats
   B.
6. Hedge value — correlation of candidate P&L vs the long stack's monthly
   drawdown; worst-DD-month overlap. Even a flat short can be worth shipping
   if it pays in the long stack's bad months.

Outputs under ``reports/v6s_path_c_short_validation/``: eight CSVs plus
``candidate_notes.md`` carrying the seven-criterion auto-verdict.

Tier: research only — no paper-live / real-live wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
from pressure_graph.reports.v3_4_true_short_sleeve import (
    SWING_RULE,
    V34Config,
    _build_cic_long_index,
    _execute_signal,
)
from pressure_graph.reports.v4s_failure_state_graph import (
    PATH_C,
    V4SConfig,
    _collect_symbol_states,
)
from pressure_graph.reports.v12s_short_motif_atlas import _f

REPORT_ROOT = Path("reports/v6s_path_c_short_validation")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")

CANDIDATE_SC1 = "S_C1_normal_short_swing"
CANDIDATE_SC2 = "S_C2_small_short_swing"
CANDIDATE_SC3A = "S_C3a_no_long"
CANDIDATE_SC3B = "S_C3b_no_overflow"

CANDIDATES: tuple[str, ...] = (CANDIDATE_SC1, CANDIDATE_SC2, CANDIDATE_SC3A, CANDIDATE_SC3B)


@dataclass(frozen=True)
class V6SConfig:
    """v6S driver config."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    long_pool_name: str = "P2_CIC1_CIC2_COMBINED"
    top_n: int = 30

    # Inherit v4S Path C combo defaults verbatim so the event stream matches.
    v4s_cfg: V4SConfig = field(default_factory=V4SConfig)

    # Short sizing
    normal_size: float = 1.0
    small_size: float = 0.5
    exit_rule: ShortExitRule = SWING_RULE

    # Cost-stress grid
    cost_grid_bps: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    extra_slippage_bps: tuple[float, ...] = (0.0, 5.0, 10.0)
    focal_cost_bps: float = 20.0
    focal_extra_slippage_bps: float = 0.0

    # Funding model — Path C fires when funding_percentile≥70 so the regime is
    # funding-positive on average. Estimate APR % and apply pro-rata to
    # holding hours.
    funding_apr_assumption: float = 0.30  # 30% APR; user can override.
    bars_per_hour: float = 4.0  # 15-min bars

    # Stability checks
    min_samples: int = 20
    month_cap_pct: float = 0.35
    no_long_block_hours: int = 12  # 12h forward window for B counterfactual

    # Clean-short forward-walk window (in bars)
    label_window_bars: int = 48  # matches Swing max_hold
    hit_down_thresholds: tuple[float, ...] = (0.03, 0.05)
    up_before_down_thresholds: tuple[float, ...] = (0.02, 0.03)


# --------------------------------------------------------------------------------------
# Event collection — runs the v4S Path C detector across the universe.
# --------------------------------------------------------------------------------------


def _collect_path_c_events(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V6SConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Stream Path C events across the top-N universe.

    Re-uses the v4S Path C combo gate so v6S's event population is
    bit-identical to the v4S production sample. Returns (events_df,
    per_symbol_group_cache). The group cache lets downstream forward-walk
    labelling read close/high/low without re-IO.
    """
    cic_long_index: dict = {}  # path C does not use it, but _collect_symbol_states needs the param
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows: list[dict[str, object]] = []
    group_cache: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols, start=1):
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        group_cache[symbol] = group
        states = _collect_symbol_states(group, cic_long_index, cfg.v4s_cfg)
        for state in states:
            if state.get("path") != PATH_C:
                continue
            rows.append(state)
        if i % 25 == 0:
            print(f"v6S Path C events: {i}/{len(symbols)} symbols, {len(rows)} events", flush=True)
    return pd.DataFrame(rows), group_cache


# --------------------------------------------------------------------------------------
# Clean-short forward labels — walk each event's group forward up to window.
# --------------------------------------------------------------------------------------


def _label_clean_short(
    group: pd.DataFrame,
    entry_idx: int,
    cfg: V6SConfig,
) -> dict[str, object]:
    """Six forward-path labels per short entry.

    The walker stops at min(entry_idx + label_window_bars, len(group)).
    All percentages are signed relative to entry price (positive = bar's
    close above entry, i.e. adverse for short).
    """
    out: dict[str, object] = {
        "hit_down_3pct": False,
        "hit_down_5pct": False,
        "up_before_down_2pct": False,
        "up_before_down_3pct": False,
        "short_squeeze_before_hit": False,
        "max_adverse_up": float("nan"),
        "first_hit_down_3pct_bar": -1,
        "first_up_2pct_bar": -1,
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
    out["first_hit_down_3pct_bar"] = int(first_hit_down_3)
    out["first_up_2pct_bar"] = int(first_up_2)
    return out


# --------------------------------------------------------------------------------------
# Cost / funding model — convert gross short return to net per (cost, slippage)
# --------------------------------------------------------------------------------------


def _net_short_return(
    gross: float,
    holding_bars: int,
    cost_bps: float,
    extra_slippage_bps: float,
    cfg: V6SConfig,
) -> float:
    """Return the post-cost, post-funding net for a 1.0× short.

    Funding model: at Path C entry the funding_percentile is ≥ 70, so shorts
    *receive* funding from longs at the configured APR. Accrual is pro-rata
    on holding hours (bars / bars_per_hour).
    """
    if not np.isfinite(gross):
        return float("nan")
    round_trip_bps = 2.0 * (float(cost_bps) + float(extra_slippage_bps)) / 10_000.0
    holding_hours = max(float(holding_bars), 0.0) / max(cfg.bars_per_hour, 1e-6)
    funding_accrual = cfg.funding_apr_assumption * (holding_hours / (24.0 * 365.0))
    return float(gross) - round_trip_bps + funding_accrual


# --------------------------------------------------------------------------------------
# Forward-long matching — used for A / B / S-C3 outcomes.
# --------------------------------------------------------------------------------------


def _forward_long_match(
    symbol: str,
    signal_time: pd.Timestamp,
    by_symbol_cache: dict[str, list[tuple[int, dict]]],
    window_minutes: int,
) -> list[dict]:
    """Every v0.9D long whose entry_time falls in (signal_time, signal_time+window]."""
    items = by_symbol_cache.get(symbol)
    if not items:
        return []
    sig_ns = int(pd.Timestamp(signal_time).value)
    window_ns = int(window_minutes * 60 * 1e9)
    matches: list[dict] = []
    for entry_ns, payload in items:
        if entry_ns <= sig_ns:
            continue
        if entry_ns - sig_ns > window_ns:
            break
        matches.append(payload)
    return matches


def _is_overflow_long(payload: dict) -> bool:
    sleeve = str(payload.get("sleeve_kind", "") or payload.get("sleeve", ""))
    return sleeve == "overflow"


# --------------------------------------------------------------------------------------
# Per-event candidate outcomes (S-C1 / S-C2 / S-C3a / S-C3b)
# --------------------------------------------------------------------------------------


def _candidate_outcomes_for_event(
    event: dict,
    by_symbol_cache: dict[str, list[tuple[int, dict]]],
    cfg: V6SConfig,
) -> dict[str, object]:
    """Compute one row per event with all four candidate outcomes + the
    A / B counterfactuals at the focal cost grid."""
    symbol = str(event.get("symbol", ""))
    signal_time = pd.Timestamp(event.get("signal_time")) if event.get("signal_time") is not None else pd.NaT
    gross = float(pd.to_numeric(event.get("gross_return", np.nan), errors="coerce"))
    holding_bars = int(pd.to_numeric(event.get("holding_bars", 0), errors="coerce") or 0)
    matches = _forward_long_match(
        symbol, signal_time, by_symbol_cache, cfg.no_long_block_hours * 60
    ) if pd.notna(signal_time) else []
    overflow_nets = [
        float(pd.to_numeric(m.get("net_return", 0.0), errors="coerce") or 0.0)
        for m in matches if _is_overflow_long(m)
    ]
    all_nets = [
        float(pd.to_numeric(m.get("net_return", 0.0), errors="coerce") or 0.0) for m in matches
    ]
    a_no_action = float(sum(all_nets)) if all_nets else 0.0
    b_no_long = -a_no_action
    sc3a_no_long_net = b_no_long
    sc3b_no_overflow_net = -float(sum(overflow_nets)) if overflow_nets else 0.0

    short_net = _net_short_return(
        gross, holding_bars, cfg.focal_cost_bps, cfg.focal_extra_slippage_bps, cfg
    )
    sc1_net = short_net * cfg.normal_size
    sc2_net = short_net * cfg.small_size
    return {
        "symbol": symbol,
        "signal_time": signal_time,
        "month": str(event.get("month", "")),
        "execution": str(event.get("execution", "")),
        "gross_return": gross,
        "holding_bars": holding_bars,
        "squeezed": bool(event.get("squeezed", False)),
        "matched_long_count": len(matches),
        "overflow_long_count": len(overflow_nets),
        "A_no_action": a_no_action,
        "B_no_long": b_no_long,
        "C_normal_short": sc1_net,
        "D_small_short": sc2_net,
        CANDIDATE_SC1: sc1_net,
        CANDIDATE_SC2: sc2_net,
        CANDIDATE_SC3A: sc3a_no_long_net,
        CANDIDATE_SC3B: sc3b_no_overflow_net,
    }


# --------------------------------------------------------------------------------------
# Cost-stress table — cost × slippage matrix per candidate, swing exit only
# --------------------------------------------------------------------------------------


def _cost_stress_table(events: pd.DataFrame, cfg: V6SConfig) -> pd.DataFrame:
    """One row per (candidate, cost_bps, extra_slippage_bps) cell."""
    if events.empty:
        return events
    short_events = events[events["execution"].astype(str).eq("swing")].copy()
    if short_events.empty:
        return short_events
    short_events["gross"] = pd.to_numeric(short_events["gross_return"], errors="coerce")
    short_events["hold"] = pd.to_numeric(short_events["holding_bars"], errors="coerce").fillna(0)
    rows: list[dict[str, object]] = []
    for cost in cfg.cost_grid_bps:
        for slip in cfg.extra_slippage_bps:
            nets_normal = [
                _net_short_return(g, h, cost, slip, cfg) * cfg.normal_size
                for g, h in zip(short_events["gross"], short_events["hold"])
            ]
            nets_small = [
                _net_short_return(g, h, cost, slip, cfg) * cfg.small_size
                for g, h in zip(short_events["gross"], short_events["hold"])
            ]
            finite_n = [x for x in nets_normal if np.isfinite(x)]
            finite_s = [x for x in nets_small if np.isfinite(x)]
            rows.append(
                {
                    "candidate": CANDIDATE_SC1,
                    "cost_bps": cost,
                    "extra_slippage_bps": slip,
                    "sample_size": len(finite_n),
                    "mean_net": float(np.mean(finite_n)) if finite_n else float("nan"),
                    "median_net": float(np.median(finite_n)) if finite_n else float("nan"),
                    "win_rate": float(np.mean(np.array(finite_n) > 0)) if finite_n else float("nan"),
                    "sum_net": float(np.sum(finite_n)),
                }
            )
            rows.append(
                {
                    "candidate": CANDIDATE_SC2,
                    "cost_bps": cost,
                    "extra_slippage_bps": slip,
                    "sample_size": len(finite_s),
                    "mean_net": float(np.mean(finite_s)) if finite_s else float("nan"),
                    "median_net": float(np.median(finite_s)) if finite_s else float("nan"),
                    "win_rate": float(np.mean(np.array(finite_s) > 0)) if finite_s else float("nan"),
                    "sum_net": float(np.sum(finite_s)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Month + symbol stability — leave-one-out and concentration concentration
# --------------------------------------------------------------------------------------


def _month_cap_total(values: pd.Series, months: pd.Series, cap_pct: float) -> float:
    """Sum of values where any single month is capped at ``cap_pct`` of total."""
    if values.empty:
        return 0.0
    raw_total = float(values.sum())
    if raw_total == 0:
        return 0.0
    cap_value = abs(raw_total) * cap_pct
    by_month = values.groupby(months).sum()
    clipped = by_month.clip(lower=-cap_value, upper=cap_value)
    return float(clipped.sum())


def _stability_per_candidate(
    outcomes: pd.DataFrame, candidate: str, cfg: V6SConfig
) -> dict[str, object]:
    """Compute month-cap, leave-one-month-out, leave-one-symbol-out, and the
    worst / best month contribution shares for one candidate."""
    if outcomes.empty:
        return {"candidate": candidate, "sample_size": 0}
    values = pd.to_numeric(outcomes[candidate], errors="coerce").fillna(0.0)
    months = outcomes["month"].astype(str)
    symbols = outcomes["symbol"].astype(str)
    raw_sum = float(values.sum())
    n = int(len(values))
    capped_sum = _month_cap_total(values, months, cfg.month_cap_pct)
    by_month_sum = values.groupby(months).sum()
    by_month_share = (by_month_sum / raw_sum) if raw_sum != 0 else by_month_sum * 0
    worst_month = str(by_month_sum.idxmin()) if not by_month_sum.empty else ""
    best_month = str(by_month_sum.idxmax()) if not by_month_sum.empty else ""
    best_month_contribution = float(by_month_share.max()) if not by_month_share.empty else float("nan")
    worst_month_contribution = float(by_month_share.min()) if not by_month_share.empty else float("nan")
    loo_month = {
        str(m): float(raw_sum - by_month_sum.loc[m]) / max(n - len(values[months.eq(m)]), 1)
        for m in by_month_sum.index
    }
    worst_loo_month_mean = min(loo_month.values()) if loo_month else float("nan")
    by_symbol_sum = values.groupby(symbols).sum()
    by_symbol_share = (by_symbol_sum / raw_sum) if raw_sum != 0 else by_symbol_sum * 0
    max_symbol_contribution = float(by_symbol_share.max()) if not by_symbol_share.empty else float("nan")
    top_symbol = str(by_symbol_sum.idxmax()) if not by_symbol_sum.empty else ""
    loo_symbol = {
        str(s): float(raw_sum - by_symbol_sum.loc[s]) / max(n - len(values[symbols.eq(s)]), 1)
        for s in by_symbol_sum.index
    }
    worst_loo_symbol_mean = min(loo_symbol.values()) if loo_symbol else float("nan")
    return {
        "candidate": candidate,
        "sample_size": n,
        "raw_mean": float(values.mean()) if n else float("nan"),
        "raw_sum": raw_sum,
        "month_cap_sum": capped_sum,
        "best_month": best_month,
        "best_month_contribution": best_month_contribution,
        "worst_month": worst_month,
        "worst_month_contribution": worst_month_contribution,
        "worst_loo_month_mean": worst_loo_month_mean,
        "top_symbol": top_symbol,
        "max_symbol_contribution": max_symbol_contribution,
        "worst_loo_symbol_mean": worst_loo_symbol_mean,
        "unique_months": int(len(by_month_sum)),
        "unique_symbols": int(len(by_symbol_sum)),
    }


def _stability_table(outcomes: pd.DataFrame, cfg: V6SConfig) -> pd.DataFrame:
    rows = [
        _stability_per_candidate(outcomes, candidate, cfg)
        for candidate in CANDIDATES
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Hedge value — correlation of candidate PnL vs long-stack monthly drawdown.
# --------------------------------------------------------------------------------------


def _long_stack_monthly_dd(trade_pool: pd.DataFrame) -> pd.Series:
    """Monthly drawdown proxy from the trade cache (per-month net return sum)."""
    if trade_pool.empty:
        return pd.Series(dtype=float)
    work = trade_pool.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["entry_time"]).copy()
    if work.empty:
        return pd.Series(dtype=float)
    work["month"] = work["entry_time"].dt.strftime("%Y-%m")
    work["net_return"] = pd.to_numeric(work.get("net_return", 0.0), errors="coerce").fillna(0.0)
    by_month = work.groupby("month")["net_return"].sum().sort_index()
    return by_month


def _hedge_correlation_per_candidate(
    outcomes: pd.DataFrame, monthly_long: pd.Series, candidate: str
) -> dict[str, object]:
    """Pearson and Spearman correlation of candidate's per-month PnL vs the
    long stack's per-month net return. Also report worst-DD-month overlap."""
    if outcomes.empty or monthly_long.empty:
        return {"candidate": candidate, "months": 0}
    values = pd.to_numeric(outcomes[candidate], errors="coerce").fillna(0.0)
    months = outcomes["month"].astype(str)
    by_month = values.groupby(months).sum().sort_index()
    common = monthly_long.index.intersection(by_month.index)
    if len(common) < 3:
        return {
            "candidate": candidate,
            "months": int(len(common)),
            "pearson_corr": float("nan"),
            "spearman_corr": float("nan"),
            "worst_long_month": str(monthly_long.idxmin()) if not monthly_long.empty else "",
            "candidate_pnl_in_worst_long_month": float("nan"),
            "worst_long_month_long_pnl": float(monthly_long.min()) if not monthly_long.empty else float("nan"),
        }
    a = monthly_long.loc[common]
    b = by_month.loc[common]
    pearson = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    try:
        spearman = float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    except Exception:
        spearman = float("nan")
    worst_long_month = str(a.idxmin())
    cand_in_worst = float(b.loc[worst_long_month]) if worst_long_month in b.index else float("nan")
    return {
        "candidate": candidate,
        "months": int(len(common)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "worst_long_month": worst_long_month,
        "worst_long_month_long_pnl": float(a.loc[worst_long_month]),
        "candidate_pnl_in_worst_long_month": cand_in_worst,
    }


def _hedge_table(outcomes: pd.DataFrame, monthly_long: pd.Series) -> pd.DataFrame:
    rows = [
        _hedge_correlation_per_candidate(outcomes, monthly_long, candidate)
        for candidate in CANDIDATES
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Clean-short forward-label table — one row per (event, candidate) for shorts
# --------------------------------------------------------------------------------------


def _clean_short_table(
    events: pd.DataFrame, group_cache: dict[str, pd.DataFrame], cfg: V6SConfig
) -> pd.DataFrame:
    """Apply the 6-label forward walker to every swing-execution event."""
    if events.empty:
        return events
    swing = events[events["execution"].astype(str).eq("swing")].copy()
    if swing.empty:
        return swing
    rows: list[dict[str, object]] = []
    for event in swing.to_dict(orient="records"):
        symbol = str(event.get("symbol", ""))
        group = group_cache.get(symbol)
        if group is None or group.empty:
            continue
        confirmation_idx = int(pd.to_numeric(event.get("confirmation_idx", -1), errors="coerce") or -1)
        if confirmation_idx < 0:
            continue
        labels = _label_clean_short(group, confirmation_idx, cfg)
        rows.append(
            {
                "symbol": symbol,
                "signal_time": event.get("signal_time"),
                "month": event.get("month", ""),
                "holding_bars": event.get("holding_bars", 0),
                "gross_return": event.get("gross_return", float("nan")),
                "squeezed": bool(event.get("squeezed", False)),
                **labels,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.agg(
            sample=("hit_down_3pct", "size"),
            hit_down_3pct_rate=("hit_down_3pct", "mean"),
            hit_down_5pct_rate=("hit_down_5pct", "mean"),
            up_before_down_2pct_rate=("up_before_down_2pct", "mean"),
            up_before_down_3pct_rate=("up_before_down_3pct", "mean"),
            short_squeeze_before_hit_rate=("short_squeeze_before_hit", "mean"),
            mean_max_adverse_up=("max_adverse_up", "mean"),
        )
    )
    # Return the per-event frame; the summary row is computed by the caller.
    return df


# --------------------------------------------------------------------------------------
# A/B/C/D head-to-head + candidate summary
# --------------------------------------------------------------------------------------


def _abcd_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per-event 4-way comparison; flag whether shorts strictly beat ``B_no_long``."""
    if outcomes.empty:
        return outcomes
    work = outcomes[
        ["symbol", "signal_time", "month", "A_no_action", "B_no_long", "C_normal_short", "D_small_short"]
    ].copy()
    work["C_minus_B"] = pd.to_numeric(work["C_normal_short"], errors="coerce") - pd.to_numeric(
        work["B_no_long"], errors="coerce"
    )
    work["D_minus_B"] = pd.to_numeric(work["D_small_short"], errors="coerce") - pd.to_numeric(
        work["B_no_long"], errors="coerce"
    )
    work["C_beats_B"] = work["C_minus_B"] > 0
    work["D_beats_B"] = work["D_minus_B"] > 0
    return work


def _candidate_summary(outcomes: pd.DataFrame, cfg: V6SConfig) -> pd.DataFrame:
    """Per-candidate headline metrics (mean, win rate, sample, vs no_long)."""
    if outcomes.empty:
        return outcomes
    no_long_mean = float(pd.to_numeric(outcomes["B_no_long"], errors="coerce").mean())
    no_long_sum = float(pd.to_numeric(outcomes["B_no_long"], errors="coerce").sum())
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        values = pd.to_numeric(outcomes[candidate], errors="coerce").dropna()
        rows.append(
            {
                "candidate": candidate,
                "sample_size": int(len(values)),
                "mean_net": float(values.mean()) if len(values) else float("nan"),
                "median_net": float(values.median()) if len(values) else float("nan"),
                "sum_net": float(values.sum()),
                "win_rate": float((values > 0).mean()) if len(values) else float("nan"),
                "vs_no_long_mean": float(values.mean()) - no_long_mean if len(values) else float("nan"),
                "vs_no_long_sum": float(values.sum()) - no_long_sum,
                "beats_no_long": bool(float(values.mean()) > no_long_mean + 1e-6) if len(values) else False,
                "meets_min_samples": int(len(values)) >= cfg.min_samples,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# candidate_notes.md — apply the seven-criterion discipline gate
# --------------------------------------------------------------------------------------


_DISCIPLINE_CRITERIA = (
    "cost_stress_robust",
    "month_stability",
    "symbol_stability",
    "clean_short_squeeze_ok",
    "beats_no_long",
    "hedge_value",
    "min_samples",
)


def _verdict_for_short_candidate(
    candidate: str,
    summary_row: pd.Series,
    stability_row: pd.Series,
    cost_stress: pd.DataFrame,
    clean_short_summary: dict[str, float],
    hedge_row: pd.Series,
    cfg: V6SConfig,
) -> tuple[str, dict[str, bool]]:
    """Combine the seven criteria into a single verdict ∈ {ship_to_shadow,
    risk_off_only, reject}."""
    flags: dict[str, bool] = {}
    flags["min_samples"] = bool(summary_row.get("meets_min_samples", False))
    flags["beats_no_long"] = bool(summary_row.get("beats_no_long", False))
    cost_band = cost_stress[cost_stress["candidate"].eq(candidate)]
    if cost_band.empty:
        flags["cost_stress_robust"] = False
    else:
        worst_cell = cost_band["mean_net"].min()
        flags["cost_stress_robust"] = bool(np.isfinite(worst_cell) and worst_cell > 0)
    flags["month_stability"] = (
        bool(stability_row.get("best_month_contribution", 1.0) <= cfg.month_cap_pct)
        and bool(stability_row.get("worst_loo_month_mean", float("-inf")) > 0)
    )
    flags["symbol_stability"] = (
        bool(stability_row.get("max_symbol_contribution", 1.0) <= cfg.month_cap_pct)
        and bool(stability_row.get("worst_loo_symbol_mean", float("-inf")) > 0)
    )
    flags["clean_short_squeeze_ok"] = bool(
        clean_short_summary.get("short_squeeze_before_hit_rate", 1.0) <= 0.20
    )
    hedge_corr = float(hedge_row.get("pearson_corr", float("nan")))
    cand_in_worst = float(hedge_row.get("candidate_pnl_in_worst_long_month", float("nan")))
    flags["hedge_value"] = bool(
        (np.isfinite(hedge_corr) and hedge_corr <= -0.3)
        or (np.isfinite(cand_in_worst) and cand_in_worst > 0)
    )
    n_pass = sum(1 for v in flags.values() if v)
    if n_pass == len(_DISCIPLINE_CRITERIA):
        return "ship_to_shadow", flags
    if flags["min_samples"] and flags["beats_no_long"] and flags["cost_stress_robust"]:
        return "risk_off_only", flags
    return "reject", flags


def _write_notes(
    report_root: Path,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    cost_stress: pd.DataFrame,
    clean_short_rows: pd.DataFrame,
    hedge: pd.DataFrame,
    abcd: pd.DataFrame,
    cfg: V6SConfig,
) -> Path:
    notes_path = report_root / "candidate_notes.md"
    lines: list[str] = [
        "# v6S Path C Short Validation — candidate notes",
        "",
        "Discipline round atop the one v4S survivor (Path C × Swing). Each",
        "candidate must clear all seven criteria to graduate to paper-shadow:",
        "- min sample size",
        "- beats B_no_long head-to-head",
        "- cost-stress robust (mean_net > 0 at the worst cost / slippage cell)",
        "- month stability (best month contribution ≤ 35%, worst-leave-one-month-out mean > 0)",
        "- symbol stability (max symbol contribution ≤ 35%, worst-leave-one-symbol-out mean > 0)",
        "- clean-short squeeze rate ≤ 20%",
        "- hedge value (Pearson corr vs long-stack monthly net ≤ −0.3, or candidate PnL > 0 in long-stack worst month)",
        "",
        f"- focal cost: {cfg.focal_cost_bps:.1f} bps + slip {cfg.focal_extra_slippage_bps:.1f} bps; funding APR {cfg.funding_apr_assumption:.1%}",
        f"- cost grid: {cfg.cost_grid_bps}; slippage grid: {cfg.extra_slippage_bps}",
        f"- candidates: {', '.join(CANDIDATES)}",
        "",
    ]
    clean_summary: dict[str, float] = {}
    if not clean_short_rows.empty:
        clean_summary = {
            "hit_down_3pct_rate": float(clean_short_rows["hit_down_3pct"].mean()),
            "hit_down_5pct_rate": float(clean_short_rows["hit_down_5pct"].mean()),
            "up_before_down_2pct_rate": float(clean_short_rows["up_before_down_2pct"].mean()),
            "up_before_down_3pct_rate": float(clean_short_rows["up_before_down_3pct"].mean()),
            "short_squeeze_before_hit_rate": float(clean_short_rows["short_squeeze_before_hit"].mean()),
            "mean_max_adverse_up": float(clean_short_rows["max_adverse_up"].mean()),
        }
        lines.append("## Clean-short label headline (Swing exit, all events)")
        for k, v in clean_summary.items():
            lines.append(f"- {k}: {v:.2%}")
        lines.append("")

    if summary.empty:
        lines.append("- empty event set; rerun once features are available.")
        notes_path.write_text("\n".join(lines), encoding="utf-8")
        return notes_path

    lines.append("## Per-candidate verdict")
    for _, row in summary.iterrows():
        candidate = str(row["candidate"])
        stability_row = stability[stability["candidate"].eq(candidate)]
        stab = stability_row.iloc[0] if not stability_row.empty else pd.Series(dtype=object)
        hedge_row = hedge[hedge["candidate"].eq(candidate)]
        hedg = hedge_row.iloc[0] if not hedge_row.empty else pd.Series(dtype=object)
        verdict, flags = _verdict_for_short_candidate(
            candidate, row, stab, cost_stress, clean_summary, hedg, cfg
        )
        lines.append(
            f"- **{candidate}**: N={int(row['sample_size'])}, mean={float(row['mean_net']):+.4%}, "
            f"win={float(row['win_rate']):.2%}, vs_no_long={float(row['vs_no_long_mean']):+.4%}"
        )
        lines.append(f"  - month contrib={float(stab.get('best_month_contribution', np.nan)):.2%} max symbol={float(stab.get('max_symbol_contribution', np.nan)):.2%}")
        lines.append(f"  - hedge corr={float(hedg.get('pearson_corr', np.nan)):+.3f} worst-long-month PnL={float(hedg.get('candidate_pnl_in_worst_long_month', np.nan)):+.4%}")
        flags_str = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in flags.items())
        lines.append(f"  - {flags_str} → **{verdict}**")
        lines.append("")

    lines.extend([
        "## Decision rule (instructment §5)",
        "- ship_to_shadow ← all 7 criteria pass.",
        "- risk_off_only ← passes min_samples + beats_no_long + cost_stress only;",
        "  ship as a *risk-mode option*, not a default sleeve.",
        "- reject otherwise; do not investigate further.",
        "",
        "## Discipline checklist (instructment §保持同一套规范)",
        "- as-of features only ✓ — Path C detector and short execution both walk forward strictly.",
        "- cost grid 10/20/30/50 bps × slippage 0/5/10 bps ✓",
        "- funding accrual baked in at funding_apr_assumption ✓",
        "- month-cap + leave-one-month-out + worst_month + best_month_contribution ✓",
        "- max_symbol_contribution + leave-one-symbol-out ✓",
        "- clean-short labels: hit_down_3pct / hit_down_5pct / up_before_down / short_squeeze_before_hit / max_adverse_up ✓",
        "- A/B/C/D head-to-head ✓",
        "- hedge value vs long-stack monthly DD ✓",
        "- no real-live without forward sample ✓ — tier: research only.",
    ])
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


# --------------------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------------------


def write_v6s_path_c_short_validation(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V6SConfig = V6SConfig(),
) -> dict[str, Path]:
    """Produce the eight instruction-mandated CSVs + candidate_notes.md verdict."""
    report_root = ensure_dir(cfg.report_root)
    if not cfg.trade_cache_path.exists():
        notes_path = report_root / "candidate_notes.md"
        notes_path.write_text(
            f"# v6S — long trade cache not found at {cfg.trade_cache_path}. Run run-v09d first.\n",
            encoding="utf-8",
        )
        return {"candidate_notes": notes_path}

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    events, group_cache = _collect_path_c_events(feature_path, rank30, rank90, config, cfg)
    if events.empty:
        notes_path = report_root / "candidate_notes.md"
        notes_path.write_text("# v6S — no Path C events fired.\n", encoding="utf-8")
        return {"candidate_notes": notes_path}

    trade_cache = read_parquet(cfg.trade_cache_path)
    pool = _focus_pool(trade_cache, cfg.long_pool_name) if "candidate" in trade_cache.columns else trade_cache
    cic_long_index = _build_cic_long_index(pool, cfg.v4s_cfg.v34_cfg)
    by_symbol_cache: dict[str, list[tuple[int, dict]]] = {}
    for (sym, ts), payload in cic_long_index.items():
        if pd.notna(ts):
            by_symbol_cache.setdefault(sym, []).append((int(pd.Timestamp(ts).value), payload))
    for sym in by_symbol_cache:
        by_symbol_cache[sym].sort(key=lambda kv: kv[0])

    swing_events = events[events["execution"].astype(str).eq("swing")].copy()
    outcomes = pd.DataFrame(
        [_candidate_outcomes_for_event(event, by_symbol_cache, cfg) for event in swing_events.to_dict(orient="records")]
    )

    summary = _candidate_summary(outcomes, cfg)
    stability = _stability_table(outcomes, cfg)
    cost_stress = _cost_stress_table(events, cfg)
    clean_short_rows = _clean_short_table(events, group_cache, cfg)
    monthly_long = _long_stack_monthly_dd(pool)
    hedge = _hedge_table(outcomes, monthly_long)
    abcd = _abcd_table(outcomes)

    outputs = {
        "v6s_pathc_summary": report_root / "v6s_pathc_summary.csv",
        "v6s_pathc_cost_stress": report_root / "v6s_pathc_cost_stress.csv",
        "v6s_pathc_month_symbol_stability": report_root / "v6s_pathc_month_symbol_stability.csv",
        "v6s_pathc_clean_short": report_root / "v6s_pathc_clean_short.csv",
        "v6s_pathc_no_long_compare": report_root / "v6s_pathc_no_long_compare.csv",
        "v6s_pathc_hedge": report_root / "v6s_pathc_hedge.csv",
        "v6s_pathc_outcomes_per_event": report_root / "v6s_pathc_outcomes_per_event.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["v6s_pathc_summary"], index=False)
    cost_stress.to_csv(outputs["v6s_pathc_cost_stress"], index=False)
    stability.to_csv(outputs["v6s_pathc_month_symbol_stability"], index=False)
    clean_short_rows.to_csv(outputs["v6s_pathc_clean_short"], index=False)
    abcd.to_csv(outputs["v6s_pathc_no_long_compare"], index=False)
    hedge.to_csv(outputs["v6s_pathc_hedge"], index=False)
    outcomes.to_csv(outputs["v6s_pathc_outcomes_per_event"], index=False)
    _write_notes(report_root, summary, stability, cost_stress, clean_short_rows, hedge, abcd, cfg)
    print(
        f"v6S: wrote {len(outcomes)} outcome events, {len(cost_stress)} cost-stress cells, "
        f"clean-short labels for {len(clean_short_rows)} events",
        flush=True,
    )
    return outputs


__all__ = [
    "CANDIDATES",
    "CANDIDATE_SC1",
    "CANDIDATE_SC2",
    "CANDIDATE_SC3A",
    "CANDIDATE_SC3B",
    "REPORT_ROOT",
    "TRADE_CACHE_PATH",
    "V6SConfig",
    "_abcd_table",
    "_candidate_outcomes_for_event",
    "_candidate_summary",
    "_clean_short_table",
    "_collect_path_c_events",
    "_cost_stress_table",
    "_forward_long_match",
    "_hedge_table",
    "_label_clean_short",
    "_long_stack_monthly_dd",
    "_net_short_return",
    "_stability_table",
    "write_v6s_path_c_short_validation",
]
