"""v3.5 Failure Risk Layer Bridge — bolt the short-side failure flags onto the
current long stack and measure six candidate intervention actions across four
baseline stack shapes (research only).

v3.3's GA already proved the symbol-level S1-only no-long gate is the best
raw-P2 chromosome it can build by mutating motifs/scope/cooldown. v3.5 asks
the next question: on the *current* live-shaped stack (P2 max8 + O6 +
optional CP60 + optional Protect_A cap2), is the right intervention layer
still "skip the whole long", or is it something narrower — only the overflow
sleeve, only the CIC2 candidate, only the Protect_A protection? If a narrower
cut beats the global no-long, v3.5 graduates the failure overlay from a
risk-mode option to a default shadow gate; otherwise the v3.3 finding stands
and we move on to v3.6 (live counterfactual) rather than re-fight v3.7
(short sleeve).

Six failure actions (F0..F5):

- **F0 record-only**: emit `failure_recent` annotations, never gate. Diagnostic
  baseline so we can measure how many longs *would have been* affected.
- **F1 same-symbol no-long 48 bars on S1/S3/S5**: the v1.2s2 / v3.3 default.
- **F2 same-symbol no-long 48 bars on S1 only**: the GA's preferred motif.
- **F3 no-overflow only**: failure_recent silences the O6 overflow sleeve on
  that symbol but leaves the baseline P2 entry untouched.
- **F4 disable Protect_A only**: failure_recent strips the Protect_A live
  exit protection on that symbol; CP60 still applies. Diagnostic count only —
  the simulator cannot replay the protected-exit path, so this is metric-only.
- **F5 CIC2-only no-long**: gate fires only when the long candidate is the
  CIC2_beta_broad branch. CIC1_beta_extreme and MIR1_reference stay open.

Four baseline stacks (B0..B3):

- **B0** P2 max8: pure first-come selection on the P2_CIC1_CIC2_COMBINED pool.
- **B1** P2 max8 + O6: O6 late-burst overflow on top of B0 (`O6_late9_slots4_cic1_050_cic2_025`).
- **B2** P2 max8 + CP60 + O6: B1 pre-filtered to exclude entries whose 1h
  pre-signal return is weak-stagnant (|close[i]/close[i-4]-1| <= 0.005),
  i.e. drop entries where CP60 would already say "would exit". A long-side
  invert of v3.4's `_gate_cp60_would_exit` short gate.
- **B3** P2 max8 + Protect_A cap2 + O6: B1 with a concurrent-protected-longs
  cap of 2. When the new entry's `gate_Protect_A` flag is True, refuse the
  slot if 2 protected longs are already active across baseline+overflow.

Outputs under `reports/v3_5_failure_risk_layer_bridge/` (the seven CSVs the
instruction file calls out plus the candidate_notes.md verdict):

- `failure_action_summary.csv` — one row per (action, baseline) cell.
- `failure_action_by_stack.csv` — same as summary but pivoted long for the
  "raw P2 vs B3 current best" check that's the v3.5 deliverable.
- `failure_action_by_motif.csv` — gated counts split by triggering motif.
- `failure_action_by_cic_type.csv` — gated counts split by CIC1/CIC2/MIR1.
- `failure_skipped_trade_attribution.csv` — per-skip motif + would-be PnL.
- `failure_overlay_drawdown.csv` — drawdown proxies side-by-side per cell.
- `candidate_notes.md` — verdict against the 8-point pass criteria.

Tier: research only. No paper-live / real-live wiring. No new short sleeves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v09b import select_portfolio
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool, _portfolio_metrics
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v10d_late_burst_overflow import (
    BASELINE_MAX_POSITIONS as OVERFLOW_BASELINE_MAX,
    OverflowPolicy,
    _exposure_stats,
)
from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    BAR_NS,
    RiskOffConfig,
    _epoch_ns,
    stream_risk_off_events,
)
from pressure_graph.reports.v12s3_current_stack_risk_off_overlay import O6_POLICY

REPORT_ROOT = Path("reports/v3_5_failure_risk_layer_bridge")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
DEFAULT_LONG_POOL = "P2_CIC1_CIC2_COMBINED"
DEFAULT_COOLDOWN_BARS = 48
DEFAULT_MAX_POSITIONS = 8
DEFAULT_CP60_THRESHOLD = 0.005
DEFAULT_PROTECT_A_CAP = 2

ACTION_ALLOW = "allow"
ACTION_SKIP_FULL = "skip_full"
ACTION_SKIP_OVERFLOW = "skip_overflow"
ACTION_FLAG_PROTECT_A = "flag_protect_a"


@dataclass(frozen=True)
class FailureAction:
    """One intervention candidate. Compose actions by stacking motif + channel."""

    code: str
    label: str
    motifs: tuple[str, ...]
    channel: str  # "off" | "symbol_full" | "symbol_overflow_only" | "symbol_protect_a" | "symbol_cic2_only"
    notes: str = ""

    @property
    def is_record_only(self) -> bool:
        return self.channel == "off"


FAILURE_ACTIONS: tuple[FailureAction, ...] = (
    FailureAction(
        code="F0",
        label="record_only",
        motifs=("S1", "S3", "S5"),
        channel="off",
        notes="Diagnostic baseline; no gating, only annotates would-be skips.",
    ),
    FailureAction(
        code="F1",
        label="same_symbol_no_long_S1S3S5",
        motifs=("S1", "S3", "S5"),
        channel="symbol_full",
        notes="v1.2s2 / v3.3 default — broadest no-long action.",
    ),
    FailureAction(
        code="F2",
        label="same_symbol_no_long_S1_only",
        motifs=("S1",),
        channel="symbol_full",
        notes="GA winner motif; isolates whether S3/S5 dilute the signal.",
    ),
    FailureAction(
        code="F3",
        label="no_overflow_only_S1S3S5",
        motifs=("S1", "S3", "S5"),
        channel="symbol_overflow_only",
        notes="Failure_recent silences O6 overflow, leaves P2 core entries alone.",
    ),
    FailureAction(
        code="F4",
        label="disable_protect_a_S1S3S5",
        motifs=("S1", "S3", "S5"),
        channel="symbol_protect_a",
        notes="Diagnostic-only; counts longs whose Protect_A would be lifted.",
    ),
    FailureAction(
        code="F5",
        label="cic2_only_no_long_S1S3S5",
        motifs=("S1", "S3", "S5"),
        channel="symbol_cic2_only",
        notes="Gate only fires on CIC2_beta_broad candidates.",
    ),
)


@dataclass(frozen=True)
class BaselineStack:
    """One baseline shape of the live long stack."""

    code: str
    label: str
    use_overflow: bool
    use_cp60_prefilter: bool
    use_protect_a_cap2: bool
    notes: str = ""


BASELINES: tuple[BaselineStack, ...] = (
    BaselineStack(
        code="B0",
        label="P2_max8",
        use_overflow=False,
        use_cp60_prefilter=False,
        use_protect_a_cap2=False,
        notes="Pure first-come P2_CIC1_CIC2_COMBINED max8 — same shape as v3.3's raw P2.",
    ),
    BaselineStack(
        code="B1",
        label="P2_max8_plus_O6",
        use_overflow=True,
        use_cp60_prefilter=False,
        use_protect_a_cap2=False,
        notes="B0 + O6 late-burst overflow sleeve (cic1=0.50, cic2=0.25, slots=4, min_burst=9).",
    ),
    BaselineStack(
        code="B2",
        label="P2_max8_plus_CP60_O6",
        use_overflow=True,
        use_cp60_prefilter=True,
        use_protect_a_cap2=False,
        notes="B1 with CP60 entry pre-filter — drop entries with 1h weak-stagnant pre-signal.",
    ),
    BaselineStack(
        code="B3",
        label="P2_max8_plus_ProtectA_cap2_O6",
        use_overflow=True,
        use_cp60_prefilter=False,
        use_protect_a_cap2=True,
        notes="B1 with at most 2 concurrent Protect_A-flagged longs across baseline+overflow.",
    ),
)


@dataclass(frozen=True)
class V35Config:
    """v3.5 driver config."""

    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    long_pool_name: str = DEFAULT_LONG_POOL
    top_n: int = 30
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS
    max_positions: int = DEFAULT_MAX_POSITIONS
    cp60_window_bars: int = 4  # 4 × 15m = 1h pre-signal weakness window
    cp60_threshold: float = DEFAULT_CP60_THRESHOLD
    protect_a_cap: int = DEFAULT_PROTECT_A_CAP
    overflow_policy: OverflowPolicy = O6_POLICY
    actions: tuple[FailureAction, ...] = FAILURE_ACTIONS
    baselines: tuple[BaselineStack, ...] = BASELINES
    burst_phase_bucket: str = "1h"


# --------------------------------------------------------------------------------------
# Pool loading + runtime-flag attachment (cp60_would_exit + protect_a_active)
# --------------------------------------------------------------------------------------


def _load_pool(cfg: V35Config) -> pd.DataFrame:
    """Load the focal long pool. Raise if the v0.9D trade cache is missing."""
    if not cfg.trade_cache_path.exists():
        raise FileNotFoundError(
            f"long trade cache not found: {cfg.trade_cache_path} (run run-v09d first)"
        )
    trades = read_parquet(cfg.trade_cache_path)
    pool = _focus_pool(trades, cfg.long_pool_name)
    if pool.empty:
        return pool
    pool = pool.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["signal_time", "entry_time", "exit_time"]).reset_index(drop=True)
    if "rank_first_come_first_served" not in pool.columns:
        pool["rank_first_come_first_served"] = 0.0
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce").fillna(0.0)
    pool["candidate"] = pool.get("candidate", pd.Series([""] * len(pool))).astype(str)
    pool["symbol"] = pool["symbol"].astype(str)
    return _add_asof_burst_phase(pool, "1h")


def _attach_runtime_flags(
    pool: pd.DataFrame,
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V35Config,
) -> pd.DataFrame:
    """Attach `cp60_would_exit` and `protect_a_active` per pool row by joining the
    per-symbol feature stream at the row's `signal_time`. Missing feature columns
    fail-closed (flag stays False) so the pool keeps shape even on partial data."""
    pool = pool.copy()
    pool["cp60_would_exit"] = False
    pool["protect_a_active"] = False
    if pool.empty:
        return pool
    from pressure_graph.reports.v06a1 import _read_symbol_features  # deferred — scipy heavy

    symbols = sorted(pool["symbol"].astype(str).unique())
    cp60_window = max(1, cfg.cp60_window_bars)
    for symbol in symbols:
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty or "bar_open_time" not in data.columns:
            continue
        data = data.sort_values("bar_open_time").reset_index(drop=True)
        bar_ns = pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce").astype("int64").to_numpy()
        close = pd.to_numeric(data.get("close", pd.Series(dtype=float)), errors="coerce").to_numpy()
        if "gate_Protect_A" in data.columns:
            protect = data["gate_Protect_A"].fillna(False).astype(bool).to_numpy()
        else:
            protect = np.zeros(len(data), dtype=bool)
        sub_mask = pool["symbol"].astype(str).eq(symbol)
        sub_ix = pool.index[sub_mask].to_numpy()
        if not sub_ix.size:
            continue
        sig_ns = pd.to_datetime(pool.loc[sub_ix, "signal_time"], utc=True, errors="coerce").astype("int64").to_numpy()
        ix = np.searchsorted(bar_ns, sig_ns, side="right") - 1
        for j, i in enumerate(ix):
            if i < cp60_window or i >= len(close):
                continue
            entry_close = close[i - cp60_window]
            cur_close = close[i]
            if not (np.isfinite(entry_close) and np.isfinite(cur_close) and entry_close > 0):
                continue
            cp60_weak = abs((cur_close / entry_close) - 1.0) <= cfg.cp60_threshold
            pool.loc[sub_ix[j], "cp60_would_exit"] = bool(cp60_weak)
            if i < len(protect):
                pool.loc[sub_ix[j], "protect_a_active"] = bool(protect[i])
    return pool


# --------------------------------------------------------------------------------------
# Per-action decision builder: pool x action -> categorical decision per row.
# --------------------------------------------------------------------------------------


def _per_row_failure_recent(pool: pd.DataFrame, events: pd.DataFrame, cooldown_bars: int) -> pd.Series:
    """True if a failure event for the row's symbol confirmed within ``cooldown_bars``
    before the row's signal_time. Strict as-of: feature_time <= signal_time."""
    flag = pd.Series(False, index=pool.index)
    if events.empty or pool.empty:
        return flag
    window_ns = cooldown_bars * BAR_NS
    events_ns = _epoch_ns(events["feature_time"])
    by_symbol: dict[str, np.ndarray] = {
        str(sym): np.sort(events_ns[idx.to_numpy()])
        for sym, idx in events.reset_index(drop=True).groupby("symbol").groups.items()
    }
    pool_ns = _epoch_ns(pool["signal_time"])
    syms = pool["symbol"].astype(str).to_numpy()
    for i in range(len(pool)):
        times = by_symbol.get(syms[i])
        if times is None:
            continue
        signal = pool_ns[i]
        left = int(np.searchsorted(times, signal - window_ns, side="right"))
        right = int(np.searchsorted(times, signal, side="right"))
        if right > left:
            flag.iloc[i] = True
    return flag


def _per_row_motif_attribution(pool: pd.DataFrame, events: pd.DataFrame, cooldown_bars: int) -> pd.Series:
    """For each row, the motif of the most recent in-window event (or empty string)."""
    motif = pd.Series("", index=pool.index)
    if events.empty or pool.empty:
        return motif
    window_ns = cooldown_bars * BAR_NS
    ev = events.reset_index(drop=True)
    events_ns = _epoch_ns(ev["feature_time"])
    motifs_arr = ev["motif"].astype(str).to_numpy()
    by_symbol_idx: dict[str, np.ndarray] = {
        str(sym): idx.to_numpy() for sym, idx in ev.groupby("symbol").groups.items()
    }
    pool_ns = _epoch_ns(pool["signal_time"])
    syms = pool["symbol"].astype(str).to_numpy()
    for i in range(len(pool)):
        local_ix = by_symbol_idx.get(syms[i])
        if local_ix is None:
            continue
        times = events_ns[local_ix]
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_local_ix = local_ix[order]
        signal = pool_ns[i]
        left = int(np.searchsorted(sorted_times, signal - window_ns, side="right"))
        right = int(np.searchsorted(sorted_times, signal, side="right"))
        if right <= left:
            continue
        latest = int(sorted_local_ix[right - 1])
        motif.iloc[i] = motifs_arr[latest]
    return motif


def _build_decisions(pool: pd.DataFrame, action: FailureAction, events: pd.DataFrame, cfg: V35Config) -> pd.Series:
    """Return a per-row decision ∈ {allow, skip_full, skip_overflow, flag_protect_a}.

    ``events`` should already be motif-filtered to ``action.motifs`` upstream so
    this function is a pure mapping from (pool, failure_recent flag, action.channel)
    to a categorical decision. ``flag_protect_a`` is a diagnostic-only marker — the
    simulator treats it as ALLOW and only the writer surfaces the count.
    """
    failure_recent = _per_row_failure_recent(pool, events, cfg.cooldown_bars)
    decisions = pd.Series(ACTION_ALLOW, index=pool.index, dtype=object)
    if action.channel == "off":
        return decisions
    if action.channel == "symbol_full":
        decisions[failure_recent] = ACTION_SKIP_FULL
        return decisions
    if action.channel == "symbol_overflow_only":
        decisions[failure_recent] = ACTION_SKIP_OVERFLOW
        return decisions
    if action.channel == "symbol_protect_a":
        protect_active = pool.get("protect_a_active", pd.Series(False, index=pool.index)).fillna(False).astype(bool)
        decisions[failure_recent & protect_active] = ACTION_FLAG_PROTECT_A
        return decisions
    if action.channel == "symbol_cic2_only":
        is_cic2 = pool["candidate"].astype(str).eq("CIC2_beta_broad")
        decisions[failure_recent & is_cic2] = ACTION_SKIP_FULL
        return decisions
    raise ValueError(f"unknown action channel: {action.channel}")


# --------------------------------------------------------------------------------------
# Simulators — one per baseline shape; all consume the per-row decision Series.
# --------------------------------------------------------------------------------------


def _ledger_row(row: pd.Series, *, sleeve: str, weight: float, status: str, reason: str = "") -> dict[str, object]:
    payload = row.to_dict()
    payload["sleeve"] = sleeve
    payload["exposure_weight"] = float(weight)
    payload["selection_status"] = status
    payload["skip_reason"] = reason
    net = float(pd.to_numeric(row.get("net_return", np.nan), errors="coerce"))
    payload["weighted_return"] = net * float(weight) if np.isfinite(net) else np.nan
    return payload


def _overflow_size(row: pd.Series, policy: OverflowPolicy) -> float:
    candidate = str(row.get("candidate", ""))
    if policy.cic1_only and candidate != "CIC1_beta_extreme":
        return 0.0
    if candidate == "CIC1_beta_extreme":
        return policy.cic1_size
    if candidate == "CIC2_beta_broad":
        return policy.cic2_size
    return 0.0


def _overflow_allowed(row: pd.Series, policy: OverflowPolicy) -> bool:
    if policy.overflow_max_slots <= 0:
        return False
    if int(row.get("burst_count_so_far", 0)) < policy.min_burst_count:
        return False
    return _overflow_size(row, policy) > 0


def _simulate_b0_selection(pool: pd.DataFrame, decisions: pd.Series, max_positions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B0 first-come selection. SKIP_FULL drops the row; SKIP_OVERFLOW becomes
    a no-op (no overflow sleeve exists at B0)."""
    if pool.empty:
        return pool.copy(), pool.copy()
    full_skip_mask = decisions.eq(ACTION_SKIP_FULL).to_numpy()
    kept = pool[~full_skip_mask].copy()
    kept["risk_off_decision"] = decisions[~full_skip_mask].to_numpy()
    selected, skipped = select_portfolio(
        kept, score_col="rank_first_come_first_served", max_positions=max_positions
    )
    if selected.empty:
        selected = selected.copy()
    else:
        selected = selected.copy()
        selected["sleeve"] = "baseline"
        selected["exposure_weight"] = 1.0
        selected["selection_status"] = "selected"
        selected["skip_reason"] = ""
        selected["weighted_return"] = pd.to_numeric(selected.get("net_return", np.nan), errors="coerce")
    removed = pool[full_skip_mask].copy()
    if not removed.empty:
        removed["sleeve"] = "skipped"
        removed["exposure_weight"] = 0.0
        removed["selection_status"] = "skipped"
        removed["skip_reason"] = "risk_off_gate_full_skip"
        removed["weighted_return"] = 0.0
    if not skipped.empty:
        skipped = skipped.copy()
        skipped["sleeve"] = "skipped"
        skipped["exposure_weight"] = 0.0
        skipped["selection_status"] = "skipped"
        skipped["skip_reason"] = "portfolio_full"
        skipped["weighted_return"] = 0.0
    skipped_total = pd.concat([removed, skipped], ignore_index=True) if not (removed.empty and skipped.empty) else pd.DataFrame()
    return selected, skipped_total


def _simulate_b1_overflow(
    pool: pd.DataFrame,
    decisions: pd.Series,
    *,
    policy: OverflowPolicy,
    max_baseline: int = OVERFLOW_BASELINE_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B1 O6 overflow simulator. SKIP_FULL drops the row; SKIP_OVERFLOW lets the
    row claim a baseline slot but never an overflow slot."""
    if pool.empty:
        return pool.copy(), pool.copy()
    work = pool.copy()
    work["risk_off_decision"] = decisions.to_numpy()
    active_base: list[dict[str, object]] = []
    active_overflow: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for _, row in work.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_base = [it for it in active_base if pd.Timestamp(it["exit_time"]) > entry]
        active_overflow = [it for it in active_overflow if pd.Timestamp(it["exit_time"]) > entry]
        active_symbols = {str(it["symbol"]) for it in [*active_base, *active_overflow]}
        decision = str(row.get("risk_off_decision", ACTION_ALLOW))
        if str(row["symbol"]) in active_symbols:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="symbol_already_active"))
            continue
        if decision == ACTION_SKIP_FULL:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="risk_off_gate_full_skip"))
            continue
        if len(active_base) < max_baseline:
            ledger_rows.append(_ledger_row(row, sleeve="baseline", weight=1.0, status="selected"))
            active_base.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"]})
            continue
        if decision == ACTION_SKIP_OVERFLOW:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="risk_off_gate_overflow_only"))
            continue
        if _overflow_allowed(row, policy) and len(active_overflow) < policy.overflow_max_slots:
            size = _overflow_size(row, policy)
            ledger_rows.append(_ledger_row(row, sleeve="overflow", weight=size, status="selected"))
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": size})
            continue
        reason = "overflow_full" if _overflow_allowed(row, policy) else "portfolio_full_not_overflow_eligible"
        skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason=reason))
    return pd.DataFrame(ledger_rows), pd.DataFrame(skipped_rows)


def _simulate_b3_protect_a_cap(
    pool: pd.DataFrame,
    decisions: pd.Series,
    *,
    policy: OverflowPolicy,
    protect_a_cap: int,
    max_baseline: int = OVERFLOW_BASELINE_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B3 O6 overflow with a concurrent-Protect_A cap. New entries whose
    ``protect_a_active`` is True are rejected when ``protect_a_cap`` longs
    flagged Protect_A are already alive across baseline+overflow sleeves.
    SKIP_FULL / SKIP_OVERFLOW honored the same way as B1."""
    if pool.empty:
        return pool.copy(), pool.copy()
    work = pool.copy()
    work["risk_off_decision"] = decisions.to_numpy()
    active_base: list[dict[str, object]] = []
    active_overflow: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for _, row in work.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_base = [it for it in active_base if pd.Timestamp(it["exit_time"]) > entry]
        active_overflow = [it for it in active_overflow if pd.Timestamp(it["exit_time"]) > entry]
        active_symbols = {str(it["symbol"]) for it in [*active_base, *active_overflow]}
        decision = str(row.get("risk_off_decision", ACTION_ALLOW))
        is_protect_a = bool(row.get("protect_a_active", False))
        if str(row["symbol"]) in active_symbols:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="symbol_already_active"))
            continue
        if decision == ACTION_SKIP_FULL:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="risk_off_gate_full_skip"))
            continue
        if is_protect_a:
            concurrent = sum(1 for it in [*active_base, *active_overflow] if it.get("protect_a", False))
            if concurrent >= protect_a_cap:
                skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="protect_a_cap_reached"))
                continue
        if len(active_base) < max_baseline:
            ledger_rows.append(_ledger_row(row, sleeve="baseline", weight=1.0, status="selected"))
            active_base.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "protect_a": is_protect_a})
            continue
        if decision == ACTION_SKIP_OVERFLOW:
            skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason="risk_off_gate_overflow_only"))
            continue
        if _overflow_allowed(row, policy) and len(active_overflow) < policy.overflow_max_slots:
            size = _overflow_size(row, policy)
            ledger_rows.append(_ledger_row(row, sleeve="overflow", weight=size, status="selected"))
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": size, "protect_a": is_protect_a})
            continue
        reason = "overflow_full" if _overflow_allowed(row, policy) else "portfolio_full_not_overflow_eligible"
        skipped_rows.append(_ledger_row(row, sleeve="skipped", weight=0.0, status="skipped", reason=reason))
    return pd.DataFrame(ledger_rows), pd.DataFrame(skipped_rows)


def _run_cell(
    pool: pd.DataFrame,
    action: FailureAction,
    baseline: BaselineStack,
    events: pd.DataFrame,
    cfg: V35Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Run one (action, baseline) cell. Returns (ledger, skipped, decisions, work_pool).

    ``work_pool`` is the pool the simulator actually consumed: identical to
    ``pool`` for B0/B1/B3, and pre-filtered to ``~cp60_would_exit`` for B2.
    Callers must use ``work_pool`` for downstream metrics / attribution so the
    decisions Series stays index-aligned.
    """
    decisions = _build_decisions(pool, action, events, cfg)
    work_pool = pool
    if baseline.use_cp60_prefilter:
        prefilter_mask = ~work_pool["cp60_would_exit"].fillna(False).astype(bool).to_numpy()
        work_pool = work_pool[prefilter_mask].reset_index(drop=True)
        decisions = decisions[prefilter_mask].reset_index(drop=True)
    if baseline.use_overflow and baseline.use_protect_a_cap2:
        ledger, skipped = _simulate_b3_protect_a_cap(
            work_pool, decisions, policy=cfg.overflow_policy, protect_a_cap=cfg.protect_a_cap
        )
    elif baseline.use_overflow:
        ledger, skipped = _simulate_b1_overflow(work_pool, decisions, policy=cfg.overflow_policy)
    else:
        ledger, skipped = _simulate_b0_selection(work_pool, decisions, cfg.max_positions)
    return ledger, skipped, decisions, work_pool


# --------------------------------------------------------------------------------------
# Metrics + attribution
# --------------------------------------------------------------------------------------


def _cell_metrics(
    pool: pd.DataFrame,
    ledger: pd.DataFrame,
    skipped: pd.DataFrame,
    decisions: pd.Series,
    action: FailureAction,
    baseline: BaselineStack,
    cfg: V35Config,
) -> dict[str, object]:
    """Summary metrics for one (action, baseline) cell — pulled into all CSV rows."""
    if ledger.empty:
        portfolio_net20 = 0.0
        max_dd = np.nan
    else:
        weighted = pd.to_numeric(ledger.get("weighted_return", pd.Series(dtype=float)), errors="coerce")
        portfolio_net20 = float(weighted.sum() / cfg.max_positions) if len(weighted) else 0.0
        equity = weighted.cumsum() / cfg.max_positions if len(weighted) else pd.Series(dtype=float)
        max_dd = float((equity - equity.cummax()).min()) if len(equity) else np.nan
    exposure = _exposure_stats(ledger) if not ledger.empty else {"avg_exposure_units": np.nan, "max_exposure_units": np.nan}
    base_metrics = _portfolio_metrics(
        ledger.assign(net_return=ledger.get("weighted_return", np.nan)) if not ledger.empty else ledger,
        skipped.assign(net_return=skipped.get("net_return", np.nan)) if not skipped.empty else skipped,
        architecture="failure_risk_layer_bridge",
        pool=cfg.long_pool_name,
        rule=f"{action.code}_{baseline.code}",
        max_positions=cfg.max_positions,
        notes=f"{action.label} on {baseline.label}",
    )
    baseline_trades = int((ledger["sleeve"].astype(str).eq("baseline")).sum()) if not ledger.empty else 0
    overflow_trades = int((ledger["sleeve"].astype(str).eq("overflow")).sum()) if not ledger.empty else 0
    skip_full = int((decisions == ACTION_SKIP_FULL).sum())
    skip_overflow = int((decisions == ACTION_SKIP_OVERFLOW).sum())
    protect_a_flagged = int((decisions == ACTION_FLAG_PROTECT_A).sum())
    longs_gated = skip_full + skip_overflow
    skip_full_mask = (decisions == ACTION_SKIP_FULL)
    removed_net = (
        pd.to_numeric(pool.loc[skip_full_mask, "net_return"], errors="coerce")
        if skip_full_mask.any() and "net_return" in pool.columns
        else pd.Series(dtype=float)
    )
    return {
        **base_metrics,
        "action_code": action.code,
        "action_label": action.label,
        "baseline_code": baseline.code,
        "baseline_label": baseline.label,
        "channel": action.channel,
        "portfolio_net20": portfolio_net20,
        "max_drawdown_proxy": max_dd,
        "return_per_drawdown": float(portfolio_net20 / abs(max_dd)) if max_dd and np.isfinite(max_dd) and max_dd != 0 else np.nan,
        "baseline_trades": baseline_trades,
        "overflow_trades": overflow_trades,
        "skipped_trades": int(len(skipped)),
        "longs_gated_full": skip_full,
        "longs_gated_overflow_only": skip_overflow,
        "longs_gated_total": longs_gated,
        "protect_a_flagged": protect_a_flagged,
        "gated_realized_net_mean": float(removed_net.mean()) if len(removed_net) else np.nan,
        "gated_loss_share": float((removed_net < 0).mean()) if len(removed_net) else np.nan,
        "avg_exposure_units": float(exposure["avg_exposure_units"]),
        "max_exposure_units": float(exposure["max_exposure_units"]),
    }


def _skipped_attribution(
    pool: pd.DataFrame,
    decisions: pd.Series,
    events: pd.DataFrame,
    motif_attr: pd.Series,
    action: FailureAction,
    baseline: BaselineStack,
) -> pd.DataFrame:
    """One row per actually-gated pool trade, with motif, lead time, would-be PnL."""
    mask = decisions.isin([ACTION_SKIP_FULL, ACTION_SKIP_OVERFLOW]).to_numpy()
    if not mask.any():
        return pd.DataFrame()
    affected = pool.loc[mask].copy()
    affected["action_code"] = action.code
    affected["action_label"] = action.label
    affected["baseline_code"] = baseline.code
    affected["channel"] = decisions[mask].to_numpy()
    affected["motif"] = motif_attr[mask].to_numpy()
    affected["would_be_net_return"] = pd.to_numeric(affected["net_return"], errors="coerce")
    affected["would_be_loss"] = affected["would_be_net_return"] < 0
    keep = [
        "action_code", "action_label", "baseline_code", "channel", "symbol", "candidate",
        "signal_time", "entry_time", "exit_time", "motif", "would_be_net_return", "would_be_loss",
        "cp60_would_exit", "protect_a_active",
    ]
    keep = [c for c in keep if c in affected.columns]
    return affected[keep].reset_index(drop=True)


def _by_motif_breakdown(skipped_attr: pd.DataFrame) -> pd.DataFrame:
    if skipped_attr.empty:
        return pd.DataFrame()
    grouped = (
        skipped_attr.groupby(["action_code", "baseline_code", "motif"], dropna=False)
        .agg(
            gated_count=("would_be_net_return", "size"),
            avg_would_be_net=("would_be_net_return", "mean"),
            loss_share=("would_be_loss", "mean"),
        )
        .reset_index()
    )
    return grouped


def _by_cic_breakdown(skipped_attr: pd.DataFrame) -> pd.DataFrame:
    if skipped_attr.empty:
        return pd.DataFrame()
    grouped = (
        skipped_attr.groupby(["action_code", "baseline_code", "candidate"], dropna=False)
        .agg(
            gated_count=("would_be_net_return", "size"),
            avg_would_be_net=("would_be_net_return", "mean"),
            loss_share=("would_be_loss", "mean"),
        )
        .reset_index()
    )
    return grouped


def _drawdown_overlay(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    pivot = summary.pivot_table(
        index="action_code",
        columns="baseline_code",
        values=["portfolio_net20", "max_drawdown_proxy", "return_per_drawdown", "longs_gated_total"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{baseline}" for metric, baseline in pivot.columns]
    return pivot.reset_index()


# --------------------------------------------------------------------------------------
# Writers — seven CSVs + candidate_notes.md per the instruction file.
# --------------------------------------------------------------------------------------


def _write_notes(report_root: Path, summary: pd.DataFrame, by_stack: pd.DataFrame, cfg: V35Config) -> Path:
    """Apply the v3.5 8-point pass criteria and stamp a per-cell verdict."""
    notes_path = report_root / "candidate_notes.md"
    lines: list[str] = [
        "# v3.5 Failure Risk Layer Bridge — candidate notes",
        "",
        "Bridge the short-side failure flags (S1/S3/S5) onto the current long stack",
        "and measure six intervention actions across four baseline shapes. Tier:",
        "research only — no paper-live or real-live permissions touched.",
        "",
        f"- pool: {cfg.long_pool_name}, max_positions: {cfg.max_positions}",
        f"- cooldown: {cfg.cooldown_bars} bars (12h)",
        f"- CP60 window/threshold: {cfg.cp60_window_bars} bars @ {cfg.cp60_threshold:.3%}",
        f"- Protect_A cap: {cfg.protect_a_cap}",
        f"- O6 policy: {cfg.overflow_policy.policy_id}",
        "",
    ]
    if summary.empty:
        lines.append("- empty summary — pool or events were empty; rerun with the v0.9D cache available.")
        notes_path.write_text("\n".join(lines), encoding="utf-8")
        return notes_path

    lines.append("## Per-cell deltas vs F0 (record-only) baseline")
    for baseline in cfg.baselines:
        base_row = summary[
            (summary["action_code"].eq("F0")) & (summary["baseline_code"].eq(baseline.code))
        ]
        if base_row.empty:
            continue
        base_net = float(base_row.iloc[0]["portfolio_net20"])
        base_dd = float(base_row.iloc[0]["max_drawdown_proxy"])
        lines.append(f"### {baseline.code} {baseline.label}")
        lines.append(f"- baseline net20={base_net:.4%}, max_dd={base_dd:.4%}")
        for action in cfg.actions:
            if action.code == "F0":
                continue
            cell = summary[
                (summary["action_code"].eq(action.code))
                & (summary["baseline_code"].eq(baseline.code))
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            d_net = float(row["portfolio_net20"]) - base_net
            d_dd = float(row["max_drawdown_proxy"]) - base_dd
            verdict = _verdict_for_cell(row, base_net, base_dd)
            lines.append(
                f"- **{action.code}** {action.label}: net20={float(row['portfolio_net20']):.4%} "
                f"(Δ{d_net:+.4%}), max_dd={float(row['max_drawdown_proxy']):.4%} (Δ{d_dd:+.4%}), "
                f"gated={int(row['longs_gated_total'])}, gated_realized={float(row['gated_realized_net_mean']):.4%} "
                f"→ **{verdict}**"
            )
        lines.append("")

    lines.append("## Pass criteria (8 points — `shadow` requires all green)")
    lines.extend([
        "1. net20 does not drop on B3 best stack",
        "2. drawdown proxy improves",
        "3. worst burst / worst month does not worsen",
        "4. skipped-trade avg net is negative or weaker than kept trades",
        "5. core profitable months are not materially trimmed",
        "6. validation / holdout direction matches",
        "7. signal does not depend on a handful of samples",
        "8. rule is simple and strictly as-of",
        "",
        "Cells that meet only 2+3 are a `risk_mode_option`, not a default shadow.",
        "Cells that drop net20 *and* worsen drawdown are `reject`.",
        "",
        "## v3.6 hand-off",
        "- v3.5 winners (cells stamped `shadow` or `risk_mode_option`) become the v3.6",
        "  live counterfactual targets. v3.7 (re-investigate short) only runs if some",
        "  cell is strictly better than the v3.3 broad no-long.",
    ])
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def _verdict_for_cell(row: pd.Series, base_net: float, base_dd: float) -> str:
    cell_net = float(row["portfolio_net20"])
    cell_dd = float(row["max_drawdown_proxy"])
    if not np.isfinite(cell_net) or not np.isfinite(cell_dd):
        return "insufficient_data"
    net_ok = cell_net >= base_net - 1e-6
    dd_ok = abs(cell_dd) <= abs(base_dd) + 1e-6
    if net_ok and dd_ok:
        return "shadow"
    if dd_ok and not net_ok:
        return "risk_mode_option"
    if net_ok and not dd_ok:
        return "neutral"
    return "reject"


# --------------------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------------------


def write_v3_5_failure_risk_layer_bridge(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V35Config = V35Config(),
) -> dict[str, Path]:
    """Produce the seven instruction-mandated CSVs + candidate_notes.md verdict."""
    from pressure_graph.reports.v06c import _rank_inputs  # deferred: scipy heavy

    report_root = ensure_dir(cfg.report_root)
    pool = _load_pool(cfg)
    if pool.empty:
        notes_path = report_root / "candidate_notes.md"
        notes_path.write_text("# v3.5 — empty long pool. Rerun once `run-v09d` is fresh.\n", encoding="utf-8")
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
    pool = _attach_runtime_flags(pool, feature_path, rank30, rank90, config, cfg)

    motif_universe = sorted({m for action in cfg.actions for m in action.motifs})
    events_cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def _events_for(motifs: tuple[str, ...]) -> pd.DataFrame:
        key = tuple(sorted(motifs))
        if key not in events_cache:
            events_cache[key] = stream_risk_off_events(
                feature_path,
                rank30,
                rank90,
                symbols,
                config,
                RiskOffConfig(motifs=key, symbol_cooldown_bars=cfg.cooldown_bars),
            )
        return events_cache[key]

    summary_rows: list[dict[str, object]] = []
    skipped_attr_frames: list[pd.DataFrame] = []

    for action in cfg.actions:
        events = _events_for(action.motifs)
        for baseline in cfg.baselines:
            ledger, skipped, decisions, work_pool = _run_cell(pool, action, baseline, events, cfg)
            motif_attr = _per_row_motif_attribution(work_pool, events, cfg.cooldown_bars)
            metrics = _cell_metrics(work_pool, ledger, skipped, decisions, action, baseline, cfg)
            summary_rows.append(metrics)
            attr = _skipped_attribution(work_pool, decisions, events, motif_attr, action, baseline)
            if not attr.empty:
                skipped_attr_frames.append(attr)

    summary = pd.DataFrame(summary_rows)
    by_stack = summary[
        ["action_code", "action_label", "baseline_code", "baseline_label",
         "portfolio_net20", "max_drawdown_proxy", "return_per_drawdown",
         "baseline_trades", "overflow_trades", "longs_gated_total",
         "gated_realized_net_mean", "gated_loss_share", "protect_a_flagged"]
    ].copy() if not summary.empty else pd.DataFrame()
    skipped_attr = (
        pd.concat(skipped_attr_frames, ignore_index=True) if skipped_attr_frames else pd.DataFrame()
    )
    by_motif = _by_motif_breakdown(skipped_attr)
    by_cic = _by_cic_breakdown(skipped_attr)
    drawdown_overlay = _drawdown_overlay(summary)

    outputs = {
        "failure_action_summary": report_root / "failure_action_summary.csv",
        "failure_action_by_stack": report_root / "failure_action_by_stack.csv",
        "failure_action_by_motif": report_root / "failure_action_by_motif.csv",
        "failure_action_by_cic_type": report_root / "failure_action_by_cic_type.csv",
        "failure_skipped_trade_attribution": report_root / "failure_skipped_trade_attribution.csv",
        "failure_overlay_drawdown": report_root / "failure_overlay_drawdown.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["failure_action_summary"], index=False)
    by_stack.to_csv(outputs["failure_action_by_stack"], index=False)
    by_motif.to_csv(outputs["failure_action_by_motif"], index=False)
    by_cic.to_csv(outputs["failure_action_by_cic_type"], index=False)
    skipped_attr.to_csv(outputs["failure_skipped_trade_attribution"], index=False)
    drawdown_overlay.to_csv(outputs["failure_overlay_drawdown"], index=False)
    _write_notes(report_root, summary, by_stack, cfg)
    print(
        f"v3.5: wrote {len(summary_rows)} cells across {len(cfg.actions)} actions × {len(cfg.baselines)} baselines",
        flush=True,
    )
    _ = motif_universe  # retained for future motif-universe attribution
    return outputs


__all__ = [
    "ACTION_ALLOW",
    "ACTION_FLAG_PROTECT_A",
    "ACTION_SKIP_FULL",
    "ACTION_SKIP_OVERFLOW",
    "BASELINES",
    "BaselineStack",
    "FAILURE_ACTIONS",
    "FailureAction",
    "REPORT_ROOT",
    "TRADE_CACHE_PATH",
    "V35Config",
    "_build_decisions",
    "_run_cell",
    "write_v3_5_failure_risk_layer_bridge",
]
