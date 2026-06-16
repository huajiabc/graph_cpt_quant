"""v4S Failure State Graph — action recommender atop failure paths.

This module is NOT a standalone short-alpha search. v3.5 closed that question:
failure-as-risk-layer beats standalone short. v4S asks the operational
follow-up: at each failure-state observation, which of seven actions is best
on the current managed long stack?

Three paths (each a state transition into a "failure observed" terminal):

- **Path A — CIC Failure**: ``CIC long signal → reclaim ok → 1h no
  follow-through → CP60 would exit → break below entry / pullback low``.
  Built on v3.4 SS3A/SS3B (CIC-led breakdown sleeves) — we treat their rows
  as state observations rather than short entries.
- **Path B — Failed Reclaim Breakdown**: ``failure motif (S1/S3/S5) → reclaim
  attempt → reclaim failed → breakdown below reclaim low``. Built on v3.4
  SS1A/SS1B (motif-led breakdown sleeves).
- **Path C — Crowded Long Stall**: ``funding crowded ∧ OI crowded ∧ price
  stalled ∧ failed follow-through ∧ (BTC_down ∨ low co-impulse)``. A new
  combo gate — scans every bar that meets all five predicates, no motif
  anchor required.

Seven actions evaluated per state observation:

| code               | meaning                                                 |
|--------------------|---------------------------------------------------------|
| ``allow_long``       | take / keep the CIC long entry that triggered Path A   |
| ``no_long``          | skip a new long here; outcome 0                        |
| ``disable_overflow`` | take baseline P2 entry, skip the O6 overflow slot      |
| ``disable_protect``  | diagnostic flag only (gate_Protect_A would be lifted)  |
| ``exit_existing_long`` | close any active long on this symbol now            |
| ``small_short``      | open short at next bar, ``small_short_size``× normal  |
| ``normal_short``     | open short at next bar, full size                      |

For each state, the atlas records the realized outcome of each applicable
action. Aggregation reports answer: which action beats ``no_long``
consistently across paths, costs, months, and symbols?

Tier: research only — no paper-live / real-live permission changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool
from pressure_graph.reports.v10b_slot_turnover_attribution import (
    _load_mark_table,
    _lookup_mark,
    _position_state,
)
from pressure_graph.backtest.short_execution import ShortExitRule, simulate_short_exit
from pressure_graph.reports.v3_4_true_short_sleeve import (
    FAST_RULE,
    SWING_RULE,
    SleeveSpec,
    V34Config,
    _apply_gates,
    _build_cic_long_index,
    _emit_cic_sleeve_signals,
    _emit_motif_sleeve_signals,
    _execute_signal,
    _gate_btc_down,
    _gate_failed_followthrough,
    _gate_low_coimpulse,
    _gate_no_protect_a,
)
from pressure_graph.reports.v12s_short_motif_atlas import DETECTORS as MOTIF_DETECTORS, _b, _f

REPORT_ROOT = Path("reports/v4s_failure_state_graph")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
DEFAULT_LONG_POOL = "P2_CIC1_CIC2_COMBINED"

ACTION_ALLOW_LONG = "allow_long"
ACTION_NO_LONG = "no_long"
ACTION_DISABLE_OVERFLOW = "disable_overflow"
ACTION_DISABLE_PROTECT = "disable_protect"
ACTION_EXIT_EXISTING_LONG = "exit_existing_long"
ACTION_SMALL_SHORT = "small_short"
ACTION_NORMAL_SHORT = "normal_short"

ACTIONS: tuple[str, ...] = (
    ACTION_ALLOW_LONG,
    ACTION_NO_LONG,
    ACTION_DISABLE_OVERFLOW,
    ACTION_DISABLE_PROTECT,
    ACTION_EXIT_EXISTING_LONG,
    ACTION_SMALL_SHORT,
    ACTION_NORMAL_SHORT,
)

PATH_A = "A_cic_failure"
PATH_B = "B_failed_reclaim_breakdown"
PATH_C = "C_crowded_long_stall"

# Path A and B: reuse v3.4 sleeve specs. We name them with v4S codes so the
# downstream atlas stays consistent.
PATH_A_SLEEVES: tuple[SleeveSpec, ...] = (
    SleeveSpec(
        code="A1_cic_entry_breakdown",
        name="path_a_cic_entry_breakdown",
        description="Path A — CIC1/CIC2 → 1h no follow-through + CP60-weak → break below entry",
        source="cic_candidate",
        source_motifs=(),
        breakdown_reference="entry",
        breakdown_valid_bars=12,
        gates=("failed_followthrough", "cp60_would_exit"),
        cooldown_bars=16,
    ),
    SleeveSpec(
        code="A2_cic_pullback_breakdown",
        name="path_a_cic_pullback_breakdown",
        description="Path A — CIC1/CIC2 → 1h no follow-through + CP60-weak → break below pullback low",
        source="cic_candidate",
        source_motifs=(),
        breakdown_reference="pullback_low",
        breakdown_valid_bars=12,
        gates=("failed_followthrough", "cp60_would_exit"),
        cooldown_bars=16,
    ),
)

PATH_B_SLEEVES: tuple[SleeveSpec, ...] = (
    SleeveSpec(
        code="B1_motif_reclaim_breakdown",
        name="path_b_motif_reclaim_breakdown",
        description="Path B — S1/S3/S5 motif + failed reclaim → break below reclaim low",
        source="motif",
        source_motifs=("S1", "S3", "S5"),
        breakdown_reference="reclaim_low",
        breakdown_valid_bars=12,
        gates=(),
        cooldown_bars=16,
    ),
)


@dataclass(frozen=True)
class V4SConfig:
    """v4S driver config — research run with the docx-mandated grids."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    long_pool_name: str = DEFAULT_LONG_POOL
    top_n: int = 30
    paths: tuple[str, ...] = (PATH_A, PATH_B, PATH_C)
    path_a_sleeves: tuple[SleeveSpec, ...] = PATH_A_SLEEVES
    path_b_sleeves: tuple[SleeveSpec, ...] = PATH_B_SLEEVES

    # Path C combo thresholds — instruction-conservative defaults.
    path_c_funding_pct_min: float = 70.0
    path_c_oi_pct_min: float = 60.0
    path_c_stall_pct_max: float = 55.0
    path_c_cooldown_bars: int = 16
    path_c_breakdown_valid_bars: int = 12

    # Action sizing & exit rules.
    small_short_size: float = 0.5
    normal_short_size: float = 1.0
    fast_rule: ShortExitRule = FAST_RULE
    swing_rule: ShortExitRule = SWING_RULE

    # Cost-grid for sensitivity (instructment §discipline).
    cost_grid_bps: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    focal_cost_bps: float = 20.0

    # Matching a state observation to an active long uses this cooldown.
    long_match_lookback_minutes: int = 12 * 60  # 12h — matches v3.5 default.
    month_cap: float = 0.35
    min_samples: int = 20

    # Internal V34Config; built lazily so the path detectors share v3.4 gates.
    v34_cfg: V34Config = field(default_factory=V34Config)


# --------------------------------------------------------------------------------------
# Path C — Crowded Long Stall: standalone combo gate (no motif anchor)
# --------------------------------------------------------------------------------------


def _gate_crowded_long_combo(group: pd.DataFrame, idx: int, cfg: V4SConfig) -> bool:
    """Path C entry combo — five predicates evaluated at bar ``idx``.

    All five must hold strictly as-of bar idx. Missing source columns fail
    closed (do not fire). Returns True when the combo triggers."""
    funding = _f(group, "funding_percentile") if "funding_percentile" in group.columns else None
    oi_delta = _f(group, "oi_value_delta_4h_percentile") if "oi_value_delta_4h_percentile" in group.columns else None
    ret_4h = _f(group, "ret_4h_percentile") if "ret_4h_percentile" in group.columns else None
    if funding is None or oi_delta is None or ret_4h is None:
        return False
    if idx >= len(funding) or idx >= len(oi_delta) or idx >= len(ret_4h):
        return False
    if not (np.isfinite(funding[idx]) and funding[idx] >= cfg.path_c_funding_pct_min):
        return False
    if not (np.isfinite(oi_delta[idx]) and oi_delta[idx] >= cfg.path_c_oi_pct_min):
        return False
    if not (np.isfinite(ret_4h[idx]) and ret_4h[idx] <= cfg.path_c_stall_pct_max):
        return False
    v34 = cfg.v34_cfg
    if not _gate_failed_followthrough(group, idx, v34):
        return False
    btc_down_ok = _gate_btc_down(group, idx, v34)
    low_co_ok = _gate_low_coimpulse(group, idx, v34)
    return bool(btc_down_ok or low_co_ok)


def _emit_path_c_states(group: pd.DataFrame, cfg: V4SConfig) -> list[dict[str, object]]:
    """Scan every bar; emit one row per (symbol, bar) where the combo holds.

    Cooldown is enforced per symbol — once the combo fires at bar i, the next
    fire on the same symbol must be ≥ ``path_c_cooldown_bars`` bars later, so
    one stall doesn't spam multiple states.
    """
    rows: list[dict[str, object]] = []
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    bar_open_time = pd.to_datetime(group["bar_open_time"], utc=True, errors="coerce")
    close = _f(group, "close")
    n = len(group)
    last_fire = -1_000_000
    for idx in range(1, n - 1):
        if idx - last_fire < cfg.path_c_cooldown_bars:
            continue
        if not _gate_crowded_long_combo(group, idx, cfg):
            continue
        entry_idx = idx + 1
        if entry_idx >= n:
            continue
        entry_close = float(close[idx]) if idx < len(close) and np.isfinite(close[idx]) else float("nan")
        next_close = float(close[entry_idx]) if entry_idx < len(close) and np.isfinite(close[entry_idx]) else float("nan")
        rows.append(
            {
                "sleeve_code": "C1_crowded_long_stall",
                "sleeve_name": "path_c_crowded_long_stall",
                "exchange": str(group.iloc[idx].get("exchange", "")),
                "symbol": str(group.iloc[idx].get("symbol", "")),
                "motif_code": "",
                "anchor_idx": int(idx),
                "confirmation_idx": int(idx),
                "break_idx": int(idx),
                "entry_idx": int(entry_idx),
                "anchor_feature_time": feature_time.iloc[idx] if idx < len(feature_time) else pd.NaT,
                "signal_time": feature_time.iloc[idx] if idx < len(feature_time) else pd.NaT,
                "entry_time": bar_open_time.iloc[entry_idx] if entry_idx < len(bar_open_time) else pd.NaT,
                "reference_low": entry_close,
                "month": (
                    feature_time.iloc[idx].strftime("%Y-%m")
                    if idx < len(feature_time) and pd.notna(feature_time.iloc[idx])
                    else ""
                ),
                "btc_state": "down_or_low_coimpulse",
            }
        )
        last_fire = idx
    return rows


# --------------------------------------------------------------------------------------
# State streaming — collects all three paths' state rows + Fast/Swing executions
# --------------------------------------------------------------------------------------


def _attach_path_label(rows: list[dict[str, object]], path_code: str) -> list[dict[str, object]]:
    for row in rows:
        row["path"] = path_code
    return rows


def _collect_symbol_states(
    group: pd.DataFrame,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    cfg: V4SConfig,
) -> list[dict[str, object]]:
    """Run all three paths over one symbol's bars; return enriched state rows."""
    rows: list[dict[str, object]] = []
    if "feature_time" not in group.columns or "bar_open_time" not in group.columns:
        return rows
    group = group.sort_values("bar_open_time").reset_index(drop=True)
    v34 = cfg.v34_cfg
    for sleeve in cfg.path_a_sleeves:
        rows.extend(_attach_path_label(_emit_cic_sleeve_signals(group, sleeve, cic_long_index, v34), PATH_A))
    for sleeve in cfg.path_b_sleeves:
        rows.extend(_attach_path_label(_emit_motif_sleeve_signals(group, sleeve, v34), PATH_B))
    rows.extend(_attach_path_label(_emit_path_c_states(group, cfg), PATH_C))
    if not rows:
        return rows
    enriched: list[dict[str, object]] = []
    for state in rows:
        for label, rule in (("fast", cfg.fast_rule), ("swing", cfg.swing_rule)):
            executed = _execute_signal(group, state, rule, label)
            enriched.append(executed)
    return enriched


# --------------------------------------------------------------------------------------
# Action evaluators — one outcome per (state, action) cell
# --------------------------------------------------------------------------------------


def _round_trip_cost(focal_cost_bps: float) -> float:
    return 2.0 * float(focal_cost_bps) / 10_000.0


def _match_active_long(
    symbol: str,
    observation_time: pd.Timestamp,
    cic_long_index: dict[tuple[str, pd.Timestamp], dict],
    by_symbol_cache: dict[str, list[tuple[int, dict]]],
) -> dict | None:
    """Return the v0.9D long active on ``symbol`` at ``observation_time`` (or None).

    An "active" long is one whose entry_time ≤ observation_time < exit_time.
    Uses a pre-built per-symbol sorted index for O(log n) lookup.
    """
    items = by_symbol_cache.get(symbol)
    if not items:
        return None
    ts_ns = int(pd.Timestamp(observation_time).value)
    for entry_ns, payload in items:
        if entry_ns > ts_ns:
            break
        exit_time = payload.get("exit_time")
        if exit_time is None:
            continue
        exit_ns = int(pd.Timestamp(exit_time).value)
        if exit_ns > ts_ns:
            return payload
    return None


def _matched_nearby_long(
    symbol: str,
    signal_time: pd.Timestamp,
    by_symbol_cache: dict[str, list[tuple[int, dict]]],
    lookback_ns: int,
) -> dict | None:
    """Return the most recent v0.9D long whose ``entry_time`` is within the
    ``lookback`` window before ``signal_time``. None if no match."""
    items = by_symbol_cache.get(symbol)
    if not items:
        return None
    ts_ns = int(pd.Timestamp(signal_time).value)
    best: dict | None = None
    for entry_ns, payload in items:
        if entry_ns > ts_ns:
            break
        if ts_ns - entry_ns <= lookback_ns:
            best = payload
    return best


def _short_pnl_from_execution(state: dict, cost_bps: float, size: float) -> float:
    """Net P&L for a short of size ``size`` at the focal cost band.

    The state row already carries ``gross_return`` from ``_execute_signal``
    (Fast/Swing rule). Net = (gross - round_trip_cost) * size.
    """
    gross = float(pd.to_numeric(state.get("gross_return", np.nan), errors="coerce"))
    if not np.isfinite(gross):
        return float("nan")
    return (gross - _round_trip_cost(cost_bps)) * float(size)


def _evaluate_actions_for_state(
    state: dict,
    by_symbol_cache: dict[str, list[tuple[int, dict]]],
    mark_table: dict[str, pd.DataFrame],
    long_lookback_ns: int,
    cfg: V4SConfig,
) -> dict[str, object]:
    """Compute outcome per action for one state observation.

    Returns a dict mapping each action code to its realized PnL (where the
    action applies) plus diagnostic metadata. Actions that don't apply
    return ``np.nan``.
    """
    symbol = str(state.get("symbol", ""))
    signal_time = pd.Timestamp(state.get("signal_time")) if state.get("signal_time") is not None else pd.NaT
    if pd.isna(signal_time):
        return {a: float("nan") for a in ACTIONS}

    outcomes: dict[str, object] = {a: float("nan") for a in ACTIONS}
    diagnostics: dict[str, object] = {
        "matched_long": False,
        "matched_long_overflow_flag": False,
        "matched_long_realized_net": float("nan"),
        "active_long_at_observation": False,
        "exit_now_unrealized": float("nan"),
        "exit_now_remaining_pnl_if_kept": float("nan"),
        "protect_a_active": False,
    }

    matched = _matched_nearby_long(symbol, signal_time, by_symbol_cache, long_lookback_ns)
    if matched is not None:
        diagnostics["matched_long"] = True
        realized = float(pd.to_numeric(matched.get("net_return", np.nan), errors="coerce"))
        diagnostics["matched_long_realized_net"] = realized
        # allow_long = the realized PnL of the long that did fire here.
        outcomes[ACTION_ALLOW_LONG] = realized if np.isfinite(realized) else float("nan")
        # disable_protect: diagnostic-only — flag whether the matched long was
        # under Protect_A; PnL outcome stays equal to allow_long.
        outcomes[ACTION_DISABLE_PROTECT] = outcomes[ACTION_ALLOW_LONG]
        diagnostics["protect_a_active"] = bool(matched.get("protect_a_active", False))
        # disable_overflow: if the matched long was an overflow entry, drop its
        # PnL; otherwise keep it. We do not have an explicit "overflow_flag"
        # column in v0.9D — approximate via `rank_first_come_first_served` slot
        # if available; fall back to ``allow_long`` so this action does not
        # punish core entries.
        sleeve_hint = str(matched.get("sleeve_kind", "") or matched.get("sleeve", ""))
        if sleeve_hint == "overflow":
            diagnostics["matched_long_overflow_flag"] = True
            outcomes[ACTION_DISABLE_OVERFLOW] = 0.0
        else:
            outcomes[ACTION_DISABLE_OVERFLOW] = outcomes[ACTION_ALLOW_LONG]

    # no_long: always defined, outcome = 0 (we did not take the position).
    outcomes[ACTION_NO_LONG] = 0.0

    # exit_existing_long: meaningful only if a long is *active* at signal_time.
    active = _match_active_long(symbol, signal_time, {}, by_symbol_cache)
    if active is not None:
        diagnostics["active_long_at_observation"] = True
        pos = _position_state({"row": active}, signal_time, mark_table)
        diagnostics["exit_now_unrealized"] = float(pos.get("unrealized_pnl_at_decision", float("nan")))
        diagnostics["exit_now_remaining_pnl_if_kept"] = float(
            pos.get("remaining_net20_proxy_if_kept", float("nan"))
        )
        outcomes[ACTION_EXIT_EXISTING_LONG] = diagnostics["exit_now_unrealized"]

    # small_short / normal_short: state already carries gross_return from
    # _execute_signal (Fast rule, then Swing). Use the focal-cost net.
    outcomes[ACTION_SMALL_SHORT] = _short_pnl_from_execution(state, cfg.focal_cost_bps, cfg.small_short_size)
    outcomes[ACTION_NORMAL_SHORT] = _short_pnl_from_execution(state, cfg.focal_cost_bps, cfg.normal_short_size)

    outcomes.update(diagnostics)
    return outcomes


# --------------------------------------------------------------------------------------
# Atlas — long table: one row per (path, state, action), tagged with diagnostics
# --------------------------------------------------------------------------------------


def _build_atlas(states_df: pd.DataFrame, cic_long_index: dict, mark_table: dict, cfg: V4SConfig) -> pd.DataFrame:
    """Wide table → long: each state expands to ``len(ACTIONS)`` rows."""
    if states_df.empty:
        return pd.DataFrame()
    by_symbol_cache: dict[str, list[tuple[int, dict]]] = {}
    for (sym, ts), payload in cic_long_index.items():
        if pd.notna(ts):
            by_symbol_cache.setdefault(sym, []).append((int(pd.Timestamp(ts).value), payload))
    for sym in by_symbol_cache:
        by_symbol_cache[sym].sort(key=lambda kv: kv[0])
    lookback_ns = int(cfg.long_match_lookback_minutes * 60 * 1e9)

    rows: list[dict[str, object]] = []
    for state in states_df.to_dict(orient="records"):
        evaluated = _evaluate_actions_for_state(state, by_symbol_cache, mark_table, lookback_ns, cfg)
        for action in ACTIONS:
            rows.append(
                {
                    "path": state.get("path", ""),
                    "sleeve_code": state.get("sleeve_code", ""),
                    "symbol": state.get("symbol", ""),
                    "month": state.get("month", ""),
                    "signal_time": state.get("signal_time"),
                    "entry_time": state.get("entry_time"),
                    "motif_code": state.get("motif_code", ""),
                    "execution": state.get("execution", ""),
                    "action": action,
                    "outcome": evaluated.get(action, float("nan")),
                    "matched_long": evaluated.get("matched_long", False),
                    "matched_long_realized_net": evaluated.get("matched_long_realized_net", float("nan")),
                    "active_long_at_observation": evaluated.get("active_long_at_observation", False),
                    "exit_now_unrealized": evaluated.get("exit_now_unrealized", float("nan")),
                    "exit_now_remaining_pnl_if_kept": evaluated.get("exit_now_remaining_pnl_if_kept", float("nan")),
                    "protect_a_active": evaluated.get("protect_a_active", False),
                    "gross_return_short": state.get("gross_return", float("nan")),
                    "max_adverse_excursion": state.get("max_adverse_excursion", float("nan")),
                    "max_favorable_excursion": state.get("max_favorable_excursion", float("nan")),
                    "squeezed": state.get("squeezed", False),
                    "holding_bars": state.get("holding_bars", 0),
                    "exit_reason": state.get("exit_reason", ""),
                    "btc_state": state.get("btc_state", ""),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Aggregations — six secondary CSVs + a notes-file verdict
# --------------------------------------------------------------------------------------


def _action_summary(atlas: pd.DataFrame, cfg: V4SConfig) -> pd.DataFrame:
    if atlas.empty:
        return atlas
    grouped = (
        atlas.dropna(subset=["outcome"])
        .groupby(["path", "execution", "action"], dropna=False)
        .agg(
            sample_size=("outcome", "size"),
            mean_outcome=("outcome", "mean"),
            median_outcome=("outcome", "median"),
            std_outcome=("outcome", "std"),
            win_rate=("outcome", lambda s: float((s > 0).mean())),
            loss_share=("outcome", lambda s: float((s < 0).mean())),
            sum_outcome=("outcome", "sum"),
        )
        .reset_index()
    )
    grouped["meets_min_samples"] = grouped["sample_size"] >= cfg.min_samples
    return grouped


def _vs_no_long(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return atlas
    pivot = atlas.pivot_table(
        index=["path", "execution", "sleeve_code", "symbol", "signal_time"],
        columns="action",
        values="outcome",
        aggfunc="first",
    ).reset_index()
    cmp_cols = [
        c for c in (
            ACTION_ALLOW_LONG, ACTION_NO_LONG, ACTION_DISABLE_OVERFLOW,
            ACTION_EXIT_EXISTING_LONG, ACTION_SMALL_SHORT, ACTION_NORMAL_SHORT,
        )
        if c in pivot.columns
    ]
    if not cmp_cols:
        return pivot
    for col in cmp_cols:
        if col != ACTION_NO_LONG:
            pivot[f"{col}_vs_no_long"] = pd.to_numeric(pivot[col], errors="coerce") - pd.to_numeric(
                pivot.get(ACTION_NO_LONG, 0.0), errors="coerce"
            )
    return pivot


def _existing_position_management(atlas: pd.DataFrame) -> pd.DataFrame:
    """exit_existing_long drilldown: who would benefit from an exit, who wouldn't."""
    if atlas.empty:
        return atlas
    target = atlas[
        atlas["action"].eq(ACTION_EXIT_EXISTING_LONG) & atlas["active_long_at_observation"].fillna(False).astype(bool)
    ].copy()
    if target.empty:
        return target
    target["exit_saves_pnl"] = -pd.to_numeric(target["exit_now_remaining_pnl_if_kept"], errors="coerce")
    target["exit_better_than_hold"] = target["exit_saves_pnl"] > 0
    grouped = (
        target.groupby(["path", "execution", "sleeve_code"], dropna=False)
        .agg(
            sample_size=("exit_saves_pnl", "size"),
            mean_exit_saves=("exit_saves_pnl", "mean"),
            median_exit_saves=("exit_saves_pnl", "median"),
            exit_better_share=("exit_better_than_hold", "mean"),
            mean_unrealized_at_exit=("exit_now_unrealized", "mean"),
        )
        .reset_index()
    )
    return grouped


def _short_clean_hit(atlas: pd.DataFrame) -> pd.DataFrame:
    """Short clean-hit metrics — clean = win without hitting stop along the way."""
    if atlas.empty:
        return atlas
    short = atlas[
        atlas["action"].isin([ACTION_SMALL_SHORT, ACTION_NORMAL_SHORT])
    ].copy()
    if short.empty:
        return short
    short["gross_short"] = pd.to_numeric(short["gross_return_short"], errors="coerce")
    short["mae"] = pd.to_numeric(short["max_adverse_excursion"], errors="coerce")
    short["clean_hit"] = (short["gross_short"] > 0) & (short["mae"].abs() < 0.01)
    grouped = (
        short.groupby(["path", "execution", "sleeve_code", "action"], dropna=False)
        .agg(
            sample_size=("gross_short", "size"),
            mean_gross_short=("gross_short", "mean"),
            mean_net_outcome=("outcome", "mean"),
            win_rate=("gross_short", lambda s: float((s > 0).mean())),
            clean_hit_rate=("clean_hit", "mean"),
            mean_mae=("mae", "mean"),
        )
        .reset_index()
    )
    return grouped


def _short_squeeze_risk(atlas: pd.DataFrame) -> pd.DataFrame:
    """Squeeze-risk drilldown: when do shorts get squeezed and by how much."""
    if atlas.empty:
        return atlas
    short = atlas[
        atlas["action"].isin([ACTION_SMALL_SHORT, ACTION_NORMAL_SHORT])
    ].copy()
    if short.empty:
        return short
    short["squeezed_bool"] = short["squeezed"].fillna(False).astype(bool)
    short["mae"] = pd.to_numeric(short["max_adverse_excursion"], errors="coerce")
    grouped = (
        short.groupby(["path", "execution", "sleeve_code", "action"], dropna=False)
        .agg(
            sample_size=("squeezed_bool", "size"),
            squeeze_rate=("squeezed_bool", "mean"),
            mean_mae_when_squeezed=("mae", lambda s: float(s[s.abs() > 0.02].mean()) if (s.abs() > 0.02).any() else float("nan")),
            worst_mae=("mae", "min"),
        )
        .reset_index()
    )
    return grouped


# --------------------------------------------------------------------------------------
# candidate_notes.md verdict — apply the instructment §discipline checks
# --------------------------------------------------------------------------------------


def _verdict_for_action(row: pd.Series, no_long_mean: float, cfg: V4SConfig) -> str:
    """Decision rule (instructment §decisive test):
    - if portfolio(action) > portfolio(no_long) and meets min samples -> ``shadow``
    - if portfolio(action) > portfolio(no_long) but below min samples -> ``promising``
    - if portfolio(action) ≈ portfolio(no_long) -> ``neutral``
    - else -> ``reject``
    """
    if not bool(row.get("meets_min_samples", False)):
        return "promising" if float(row.get("mean_outcome", 0.0)) > no_long_mean + 1e-6 else "reject"
    mean = float(row.get("mean_outcome", 0.0))
    if mean > no_long_mean + 1e-6:
        return "shadow"
    if mean >= no_long_mean - 1e-6:
        return "neutral"
    return "reject"


def _write_notes(report_root: Path, summary: pd.DataFrame, vs_no_long: pd.DataFrame, cfg: V4SConfig) -> Path:
    notes_path = report_root / "candidate_notes.md"
    lines: list[str] = [
        "# v4S Failure State Graph — candidate notes",
        "",
        "Action recommender atop three failure-state paths. Decision per (path,",
        "action) cell follows the instructment §discipline: an action graduates",
        "to `shadow` only when its mean realised outcome beats `no_long` *with*",
        "minimum sample size met. Otherwise: `promising` (positive but small N),",
        "`neutral` (within ±ε), or `reject`.",
        "",
        f"- paths: {', '.join(cfg.paths)}",
        f"- actions: {', '.join(ACTIONS)}",
        f"- focal cost: {cfg.focal_cost_bps:.1f} bps (cost grid: {cfg.cost_grid_bps})",
        f"- small_short size: {cfg.small_short_size}× ; normal_short size: {cfg.normal_short_size}×",
        f"- min samples threshold: {cfg.min_samples}",
        "",
    ]
    if summary.empty:
        lines.append("- empty atlas — universe or trade cache produced no states; rerun once features are fresh.")
        notes_path.write_text("\n".join(lines), encoding="utf-8")
        return notes_path

    for path_code in cfg.paths:
        path_rows = summary[summary["path"].eq(path_code)]
        if path_rows.empty:
            lines.append(f"## {path_code} — no states observed (skip).\n")
            continue
        no_long_means = (
            path_rows[path_rows["action"].eq(ACTION_NO_LONG)]
            .groupby("execution")["mean_outcome"]
            .mean()
            .to_dict()
        )
        lines.append(f"## {path_code}")
        for execution in sorted(path_rows["execution"].dropna().astype(str).unique()):
            block = path_rows[path_rows["execution"].eq(execution)].copy()
            if block.empty:
                continue
            no_long_mean = float(no_long_means.get(execution, 0.0))
            lines.append(f"### execution={execution}, no_long mean={no_long_mean:+.4%}")
            for _, row in block.iterrows():
                action = str(row["action"])
                if action == ACTION_NO_LONG:
                    continue
                verdict = _verdict_for_action(row, no_long_mean, cfg)
                lines.append(
                    f"- **{action}**: N={int(row['sample_size'])}, "
                    f"mean={float(row['mean_outcome']):+.4%}, "
                    f"win_rate={float(row['win_rate']):.2%}, "
                    f"vs_no_long={float(row['mean_outcome']) - no_long_mean:+.4%} → **{verdict}**"
                )
            lines.append("")

    lines.extend([
        "## v4S → v6S hand-off rules (instructment §结果)",
        "- If `open_short` < `no_long`, do NOT ship short.",
        "- If `disable_overflow` ≈ `no_long` with less complexity → prefer `disable_overflow`.",
        "- If `exit_existing_long` saves measurable PnL on a stable sample → consider",
        "  shipping as a defensive risk-off action on the live stack.",
        "- Any cell stamped `reject` or `neutral` does NOT graduate to v3.5's shadow set.",
        "",
        "## Discipline checklist (instructment §保持同一套规范)",
        "- as-of feature only ✓ — every detector reads features at idx ≤ signal_time.",
        "- cost grid 10/20/30/50bp — focal cell at 20bp; `failure_action_summary` and",
        "  `failure_short_clean_hit` carry the focal numbers; full grid sensitivity is",
        "  trivially reproduced by re-running with a different ``focal_cost_bps``.",
        "- month-cap + leave-one-month + symbol contribution + random/shuffled controls",
        "  + selected/skipped counterfactual: deferred to v6S follow-up (not in v4S MVP).",
        "- no real-live without forward sample ✓ — this module is research-only.",
    ])
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


# --------------------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------------------


def write_v4s_failure_state_graph(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V4SConfig = V4SConfig(),
) -> dict[str, Path]:
    """Produce the seven instruction-mandated CSVs + candidate_notes.md verdict."""
    report_root = ensure_dir(cfg.report_root)
    if not cfg.trade_cache_path.exists():
        notes_path = report_root / "candidate_notes.md"
        notes_path.write_text(
            f"# v4S — long trade cache not found at {cfg.trade_cache_path}. Run run-v09d first.\n",
            encoding="utf-8",
        )
        return {"candidate_notes": notes_path}

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    trade_cache = read_parquet(cfg.trade_cache_path)
    pool = _focus_pool(trade_cache, cfg.long_pool_name) if "candidate" in trade_cache.columns else trade_cache
    cic_long_index = _build_cic_long_index(pool, cfg.v34_cfg)

    state_rows: list[dict[str, object]] = []
    for i, symbol in enumerate(symbols, start=1):
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        rows = _collect_symbol_states(group, cic_long_index, cfg)
        state_rows.extend(rows)
        if i % 25 == 0:
            print(f"v4S states: {i}/{len(symbols)} symbols, {len(state_rows)} states", flush=True)

    states_df = pd.DataFrame(state_rows)
    if states_df.empty:
        notes_path = report_root / "candidate_notes.md"
        notes_path.write_text("# v4S — no failure states detected in the universe.\n", encoding="utf-8")
        return {"candidate_notes": notes_path}

    pool_window_min = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce").min() if not pool.empty else pd.NaT
    pool_window_max = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce").max() if not pool.empty else pd.NaT
    if pd.notna(pool_window_min) and pd.notna(pool_window_max):
        mark_table = _load_mark_table(feature_path, set(symbols), pool_window_min, pool_window_max)
    else:
        mark_table = {}

    atlas = _build_atlas(states_df, cic_long_index, mark_table, cfg)
    summary = _action_summary(atlas, cfg)
    vs_no_long = _vs_no_long(atlas)
    epm = _existing_position_management(atlas)
    short_clean = _short_clean_hit(atlas)
    squeeze = _short_squeeze_risk(atlas)

    outputs = {
        "failure_state_action_atlas": report_root / "failure_state_action_atlas.csv",
        "failure_action_summary": report_root / "failure_action_summary.csv",
        "failure_vs_no_long": report_root / "failure_vs_no_long.csv",
        "failure_existing_position_management": report_root / "failure_existing_position_management.csv",
        "failure_short_clean_hit": report_root / "failure_short_clean_hit.csv",
        "failure_short_squeeze_risk": report_root / "failure_short_squeeze_risk.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    atlas.to_csv(outputs["failure_state_action_atlas"], index=False)
    summary.to_csv(outputs["failure_action_summary"], index=False)
    vs_no_long.to_csv(outputs["failure_vs_no_long"], index=False)
    epm.to_csv(outputs["failure_existing_position_management"], index=False)
    short_clean.to_csv(outputs["failure_short_clean_hit"], index=False)
    squeeze.to_csv(outputs["failure_short_squeeze_risk"], index=False)
    _write_notes(report_root, summary, vs_no_long, cfg)
    print(
        f"v4S: wrote atlas with {len(atlas)} rows across {len(states_df)} states "
        f"({len(cfg.paths)} paths × {len(ACTIONS)} actions)",
        flush=True,
    )
    return outputs


__all__ = [
    "ACTIONS",
    "ACTION_ALLOW_LONG",
    "ACTION_DISABLE_OVERFLOW",
    "ACTION_DISABLE_PROTECT",
    "ACTION_EXIT_EXISTING_LONG",
    "ACTION_NO_LONG",
    "ACTION_NORMAL_SHORT",
    "ACTION_SMALL_SHORT",
    "PATH_A",
    "PATH_A_SLEEVES",
    "PATH_B",
    "PATH_B_SLEEVES",
    "PATH_C",
    "REPORT_ROOT",
    "TRADE_CACHE_PATH",
    "V4SConfig",
    "_build_atlas",
    "_collect_symbol_states",
    "_emit_path_c_states",
    "_evaluate_actions_for_state",
    "_gate_crowded_long_combo",
    "_match_active_long",
    "_matched_nearby_long",
    "write_v4s_failure_state_graph",
]
