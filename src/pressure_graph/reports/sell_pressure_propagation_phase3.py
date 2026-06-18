"""Sell-Pressure Propagation Map — Phase 3 (prompt9 docx 2026-06-18).

Goal docx: "下一步最合理：Phase 3 DOGE Edge Validation". Phase 2C found ONE
asymmetric cross-venue edge — ``bybit:DOGEUSDT → binance_um:DOGEUSDT`` short
lead at lag 5/15/30 min — significant on a one-month October 2025 sample but
not directly tradeable (gross 15bp < 35bp cost, adverse 42%).

Phase 3 is **validation**, not new search. The docx is explicit:

- **3A time stability** — re-measure the edge across multiple months. Is
  October 2025 a regime event, or is the lead persistent month-to-month?
- **3B basket extension** — test the same head-to-head on a DOGE-like meme
  basket (SHIB/PEPE/WIF/BONK/FLOKI/1000PEPE/1000BONK/ORDI/PENGU). Is the
  Bybit→Binance lead a DOGE-only fluke or a meme/retail-beta phenomenon?
- **3C direction stability** — beyond the existing shuffled-null, three extra
  controls: signed-null (random ±1 on each event response), venue-symmetry
  null (pool both directions then relabel), random-symbol source. The edge
  must beat ALL controls.
- **Execution repair E0..E3** — diagnostic only. Four entry filters; do not
  promote unless every gate passes.

Even if every gate passes, the docx is explicit that the edge is a
"propagation diagnostic graph node", NOT a real short. ``register_propagation_graph_node`` produces the artefact.

This module imports only from ``sell_pressure_propagation`` so the underlying
event-detection, response-grid, shuffled-null and CI machinery is shared.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.sell_pressure_propagation import (
    DEFAULT_LAG_WINDOWS_MIN,
    EVENT_CODES,
    PropagationConfig,
    ResponseGrids,
    VenueSymbolSpec,
    _draw_shuffled_means,
    _index_align_event_positions,
    _safe_vwap,
    build_propagation_map_multi_venue,
    detect_events,
    intersect_symbol_windows,
    load_continuous_cvd,
    measure_edge,
    precompute_response_grid,
    restrict_to_window,
)


# -- Public configuration ----------------------------------------------------------

#: 10 DOGE-like meme / retail-beta symbols named in the goal docx.
DEFAULT_DOGE_LIKE_BASKET: tuple[str, ...] = (
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "1000PEPEUSDT",
    "1000BONKUSDT",
    "ORDIUSDT",
    "PENGUUSDT",
)

#: Cost band the docx benchmarks against (gross short return must exceed this
#: for execution repair to count as a pass). 35 bp = 30 bp fee+slip + 5 bp
#: noise; matches the existing closure-doc gate3 probe in Phase 1.
DEFAULT_COST_BPS: float = 35.0

#: Phase 3 verdict gates (docx-aligned). Edge passes "tradeable" only if every
#: gate is True. Otherwise it stays as a diagnostic graph node.
@dataclass(frozen=True)
class Phase3Gates:
    min_months_positive_ci: int = 3  # at least 3 distinct months with ci_low > 0
    max_month_share_of_events: float = 0.45  # no single month carries > 45% of events
    max_adverse_rate: float = 0.40
    cost_bps: float = DEFAULT_COST_BPS
    direction_must_beat_all_nulls: bool = True


# -- Data discovery ----------------------------------------------------------------


def discover_venue_symbol_specs(
    venue_roots: dict[str, Path],
    candidates: tuple[str, ...],
    bar_size_minutes: int,
) -> tuple[VenueSymbolSpec, ...]:
    """Return one VenueSymbolSpec per (venue, symbol) where the parquet folder
    exists on disk. ``venue_roots`` maps a short venue tag (e.g. ``"binance_um"``,
    ``"bybit"``) to its continuous-CVD root. Missing folders are silently
    dropped — the caller can compare the returned tuple against the requested
    basket to surface gaps."""
    out: list[VenueSymbolSpec] = []
    for venue, root in venue_roots.items():
        root = Path(root)
        for symbol in candidates:
            sym_dir = root / symbol / f"{bar_size_minutes}min"
            if sym_dir.exists() and any(sym_dir.glob("*.parquet")):
                out.append(
                    VenueSymbolSpec(
                        label=f"{venue}:{symbol}",
                        venue=venue,
                        cvd_root=root,
                        symbol=symbol,
                    )
                )
    return tuple(out)


def discover_dual_venue_basket(
    binance_root: Path,
    bybit_root: Path,
    candidates: tuple[str, ...],
    bar_size_minutes: int,
) -> tuple[tuple[VenueSymbolSpec, ...], tuple[str, ...]]:
    """Return ``(specs, missing)`` where ``specs`` covers every symbol present
    on BOTH venues and ``missing`` lists the candidate symbols that are
    incomplete (i.e. one or both venues lack a parquet folder)."""
    bin_specs = discover_venue_symbol_specs(
        {"binance_um": binance_root}, candidates, bar_size_minutes
    )
    byb_specs = discover_venue_symbol_specs(
        {"bybit": bybit_root}, candidates, bar_size_minutes
    )
    bin_syms = {s.symbol for s in bin_specs}
    byb_syms = {s.symbol for s in byb_specs}
    both = bin_syms & byb_syms
    missing = tuple(sym for sym in candidates if sym not in both)
    specs = tuple(s for s in (*bin_specs, *byb_specs) if s.symbol in both)
    return specs, missing


def available_months_for_spec(spec: VenueSymbolSpec, bar_size_minutes: int) -> tuple[str, ...]:
    """List 'YYYY-MM' month tags present on disk for one spec."""
    sym_dir = Path(spec.cvd_root) / spec.symbol / f"{bar_size_minutes}min"
    if not sym_dir.exists():
        return ()
    months: set[str] = set()
    for parquet in sym_dir.glob("*.parquet"):
        stem = parquet.stem
        # Accept 'YYYY-MM' shaped stems only — defensive against stray files.
        if len(stem) == 7 and stem[4] == "-" and stem[:4].isdigit() and stem[5:].isdigit():
            months.add(stem)
    return tuple(sorted(months))


def intersect_months(specs: tuple[VenueSymbolSpec, ...], bar_size_minutes: int) -> tuple[str, ...]:
    """Return months present on EVERY spec. Empty tuple → no overlap."""
    if not specs:
        return ()
    months = set(available_months_for_spec(specs[0], bar_size_minutes))
    for spec in specs[1:]:
        months &= set(available_months_for_spec(spec, bar_size_minutes))
    return tuple(sorted(months))


# -- Phase 3A: per-month edge measurement ------------------------------------------


@dataclass(frozen=True)
class FocusEdge:
    """One directed edge the Phase 3 driver evaluates per month."""

    source_label: str
    target_label: str
    event_type: str
    lag_minutes: tuple[int, ...] = DEFAULT_LAG_WINDOWS_MIN


def _month_mask(timestamps: pd.Series, month: str) -> pd.Series:
    """Boolean mask over ``timestamps`` matching the YYYY-MM key."""
    ts = timestamps
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    keys = ts.dt.strftime("%Y-%m")
    return keys.eq(month)


def measure_edge_for_month(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    event_mask: pd.Series,
    event_name: str,
    month: str,
    lag_minutes: int,
    cfg: PropagationConfig,
    rng: np.random.Generator,
    response_grid: ResponseGrids | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    """Run ``measure_edge`` on the subset of events that fall within ``month``.

    The shuffled-null is drawn from eligible positions inside the SAME month —
    this keeps regime context (volatility, basis, funding) comparable between
    observed and null, which matters far more than pool size at month-scale.
    """
    month_keep = _month_mask(source_df["bar_open_time"], month)
    sub_mask = event_mask & month_keep
    row = measure_edge(
        source_df=source_df,
        target_df=target_df,
        event_mask=sub_mask,
        event_name=event_name,
        lag_minutes=lag_minutes,
        cfg=cfg,
        rng=rng,
        response_grid=response_grid,
    )
    short = row["short_return_mean"]
    cost_net = float("nan") if pd.isna(short) else float(short - cost_bps / 1e4)
    return {
        "month": month,
        "event_type": event_name,
        "lag_minutes": lag_minutes,
        "n_events": row["n_events"],
        "short_return_mean": short,
        "adverse_rate": row["adverse_upsample"],
        "bootstrap_ci_low": row["bootstrap_ci_low"],
        "edge_strength": row["edge_strength"],
        "cost_net": cost_net,
    }


def build_per_month_edge_table(
    specs: tuple[VenueSymbolSpec, ...],
    focus: tuple[FocusEdge, ...],
    cfg: PropagationConfig,
    cost_bps: float = DEFAULT_COST_BPS,
    months: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """For each focus edge × month, compute one summary row.

    If ``months`` is None the function uses every month found on disk for the
    source spec; the response grid is re-used across months because it depends
    on the target frame, not on the event subset.
    """
    if not specs:
        return pd.DataFrame()
    cvd_by_label: dict[str, pd.DataFrame] = {
        spec.label: load_continuous_cvd(spec.cvd_root, spec.symbol, cfg.bar_size_minutes)
        for spec in specs
    }
    start, end = intersect_symbol_windows(cvd_by_label)
    cvd_by_label = {k: restrict_to_window(v, start, end) for k, v in cvd_by_label.items()}
    events_by_label = {label: detect_events(df, cfg) for label, df in cvd_by_label.items()}

    if months is None:
        months_from_data: set[str] = set()
        for df in cvd_by_label.values():
            ts = df["bar_open_time"]
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            months_from_data.update(ts.dt.strftime("%Y-%m").unique().tolist())
        months = tuple(sorted(months_from_data))

    rng = np.random.default_rng(cfg.rng_seed)
    rows: list[dict] = []
    for edge in focus:
        if edge.source_label not in cvd_by_label or edge.target_label not in cvd_by_label:
            continue
        source_df = cvd_by_label[edge.source_label]
        target_df = cvd_by_label[edge.target_label]
        event_mask = events_by_label[edge.source_label][edge.event_type]
        for lag_minutes in edge.lag_minutes:
            lag_bars = lag_minutes // cfg.bar_size_minutes
            grid = precompute_response_grid(target_df, lag_bars, cfg)
            for month in months:
                row = measure_edge_for_month(
                    source_df=source_df,
                    target_df=target_df,
                    event_mask=event_mask,
                    event_name=edge.event_type,
                    month=month,
                    lag_minutes=lag_minutes,
                    cfg=cfg,
                    rng=rng,
                    response_grid=grid,
                    cost_bps=cost_bps,
                )
                row["source"] = edge.source_label
                row["target"] = edge.target_label
                rows.append(row)
    cols = [
        "source",
        "target",
        "month",
        "event_type",
        "lag_minutes",
        "n_events",
        "short_return_mean",
        "adverse_rate",
        "bootstrap_ci_low",
        "edge_strength",
        "cost_net",
    ]
    return pd.DataFrame(rows, columns=cols)


def phase3a_verdict(per_month: pd.DataFrame, gates: Phase3Gates) -> dict:
    """Apply Phase 3A gates to the per-month table. Considers the headline lag
    (5 min E1) by default — the lag where Phase 2C found the strongest signal.

    Returns a dict the summary writer can fold into the markdown.
    """
    if per_month.empty:
        return {"status": "no_data", "reason": "empty per-month table"}
    headline = per_month[
        (per_month["event_type"] == "E1") & (per_month["lag_minutes"] == 5)
    ]
    if headline.empty:
        return {"status": "no_data", "reason": "no E1/lag-5m rows"}
    positive_months = headline[headline["bootstrap_ci_low"] > 0]
    n_positive_months = int(positive_months["month"].nunique())
    total_events = int(headline["n_events"].sum())
    if total_events <= 0:
        return {"status": "no_data", "reason": "no E1/lag-5m events across months"}
    month_event_share = (
        headline.groupby("month")["n_events"].sum() / total_events
    )
    max_share = float(month_event_share.max()) if not month_event_share.empty else 1.0
    mean_adverse = float(headline["adverse_rate"].mean())
    passes_months = n_positive_months >= gates.min_months_positive_ci
    passes_concentration = max_share <= gates.max_month_share_of_events
    passes_adverse = mean_adverse <= gates.max_adverse_rate
    if passes_months and passes_concentration and passes_adverse:
        status = "stable"
    elif n_positive_months <= 1:
        status = "regime_event"
    else:
        status = "weak"
    return {
        "status": status,
        "n_months_evaluated": int(headline["month"].nunique()),
        "n_months_positive_ci": n_positive_months,
        "max_single_month_event_share": max_share,
        "mean_adverse_rate": mean_adverse,
        "passes_months_gate": passes_months,
        "passes_concentration_gate": passes_concentration,
        "passes_adverse_gate": passes_adverse,
    }


# -- Phase 3B: basket extension ----------------------------------------------------


def summarize_head_to_head(edge_map: pd.DataFrame, source_venue: str, target_venue: str) -> pd.DataFrame:
    """Filter ``edge_map`` (multi-venue) to head-to-head edges in the
    ``source_venue → target_venue`` direction (same symbol) and return one
    row per (symbol, event_type, lag_minutes)."""
    if edge_map.empty:
        return pd.DataFrame()
    head = edge_map[
        (edge_map["path"] == "cross_venue")
        & (edge_map["source_venue"] == source_venue)
        & (edge_map["target_venue"] == target_venue)
    ].copy()
    return head.sort_values(
        ["source_symbol", "event_type", "lag_minutes"]
    ).reset_index(drop=True)


def phase3b_verdict(
    forward_head: pd.DataFrame,
    reverse_head: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    """For each symbol, classify the cross-venue edge as one of:

    - ``shared`` — forward AND reverse both show positive short_return at headline lag → likely market drift, not propagation.
    - ``asymmetric_forward`` — forward positive, reverse negative → real lead-lag.
    - ``asymmetric_reverse`` — reverse positive, forward negative.
    - ``flat`` — neither direction has a clear edge.

    Returns ``{"per_symbol": [...], "asymmetric_forward_count": int, ...}``.
    """
    if forward_head.empty and reverse_head.empty:
        return {"per_symbol": [], "asymmetric_forward_count": 0, "shared_count": 0, "flat_count": 0}

    headline_filter = (
        lambda df: df[(df["event_type"] == "E1") & (df["lag_minutes"] == 5)] if not df.empty else df
    )
    fwd = headline_filter(forward_head)
    rev = headline_filter(reverse_head)
    symbols = sorted(set(fwd.get("source_symbol", pd.Series(dtype=str))).union(
        set(rev.get("source_symbol", pd.Series(dtype=str)))
    ))
    per_symbol: list[dict] = []
    cost_threshold = cost_bps / 1e4
    for sym in symbols:
        fwd_row = fwd[fwd["source_symbol"] == sym]
        rev_row = rev[rev["source_symbol"] == sym]
        f_short = float(fwd_row["short_return_mean"].iloc[0]) if not fwd_row.empty else float("nan")
        r_short = float(rev_row["short_return_mean"].iloc[0]) if not rev_row.empty else float("nan")
        f_ci = float(fwd_row["bootstrap_ci_low"].iloc[0]) if not fwd_row.empty else float("nan")
        r_ci = float(rev_row["bootstrap_ci_low"].iloc[0]) if not rev_row.empty else float("nan")
        if pd.isna(f_short) or pd.isna(r_short):
            label = "incomplete"
        elif f_short > 0 and r_short > 0:
            label = "shared"
        elif f_short > 0 and f_ci > 0 and r_short <= 0:
            label = "asymmetric_forward"
        elif r_short > 0 and r_ci > 0 and f_short <= 0:
            label = "asymmetric_reverse"
        else:
            label = "flat"
        per_symbol.append(
            {
                "symbol": sym,
                "forward_short_return": f_short,
                "forward_ci_low": f_ci,
                "forward_passes_cost": bool(f_short > cost_threshold),
                "reverse_short_return": r_short,
                "reverse_ci_low": r_ci,
                "label": label,
            }
        )
    counts = pd.Series([row["label"] for row in per_symbol]).value_counts().to_dict()
    return {
        "per_symbol": per_symbol,
        "asymmetric_forward_count": int(counts.get("asymmetric_forward", 0)),
        "asymmetric_reverse_count": int(counts.get("asymmetric_reverse", 0)),
        "shared_count": int(counts.get("shared", 0)),
        "flat_count": int(counts.get("flat", 0)),
        "incomplete_count": int(counts.get("incomplete", 0)),
    }


# -- Phase 3C: directional / source-symbol nulls ------------------------------------


def _event_positions_on_target(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    event_mask: pd.Series,
    response_grid: ResponseGrids,
) -> np.ndarray:
    """Translate source event positions to target positions where the response
    is measurable (response_grid.short_return is not NaN). A thin wrapper over
    the private helper so the null builders can reuse it without depending on
    private symbols."""
    positions = _index_align_event_positions(source_df, target_df, event_mask)
    if positions.size == 0:
        return positions
    measurable = ~np.isnan(response_grid.short_return)
    return positions[measurable[positions]]


def signed_random_null(
    response: np.ndarray,
    event_positions: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sign-flip null: at the OBSERVED event positions, multiply each event's
    short_return by random ±1 (Rademacher) and return the bootstrap means.

    If the observed mean is truly positive (consistent downside propagation),
    it will be far above the upper tail of this null. If the observed mean is
    driven by a few large bidirectional events (long tail), the null will
    overlap.
    """
    if event_positions.size == 0:
        return np.empty(0, dtype=float)
    vals = response[event_positions]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.empty(0, dtype=float)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, vals.size))
    return np.nanmean(vals * signs, axis=1)


def venue_symmetry_null(
    forward_response: np.ndarray,
    forward_event_positions: np.ndarray,
    reverse_response: np.ndarray,
    reverse_event_positions: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pool the observed event responses from BOTH directions, then randomly
    relabel direction per iteration (forward = first n_forward draws from the
    permuted pool, take its mean). If the observed forward mean is just a
    relabelling of a directionally-symmetric distribution, this null will
    overlap the observed; if there's a real venue lead-lag, the observed
    forward mean sits above the null's 90th percentile.
    """
    fwd_vals = forward_response[forward_event_positions]
    rev_vals = reverse_response[reverse_event_positions]
    pooled = np.concatenate([fwd_vals, rev_vals])
    pooled = pooled[~np.isnan(pooled)]
    n_fwd = int(np.sum(~np.isnan(fwd_vals)))
    if pooled.size == 0 or n_fwd == 0:
        return np.empty(0, dtype=float)
    means = np.empty(iterations)
    for k in range(iterations):
        idx = rng.permutation(pooled.size)
        means[k] = float(np.mean(pooled[idx[:n_fwd]]))
    return means


def random_symbol_source_null(
    target_response: np.ndarray,
    other_source_event_positions: dict[str, np.ndarray],
    n_events: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace the source symbol's events with another symbol's events at the
    same venue. ``other_source_event_positions`` is a dict
    ``{symbol_label: event_positions_on_target_frame}``. Each iteration picks
    one off-symbol pool uniformly and draws ``n_events`` from it.

    If the observed forward edge is propagation specifically from the source
    symbol (Bybit:DOGE), the observed mean stays above this null; if the
    observed is just "any meme on Bybit dumps → Binance follows", the null
    overlaps.
    """
    keys = list(other_source_event_positions.keys())
    if not keys or n_events <= 0:
        return np.empty(0, dtype=float)
    means: list[float] = []
    for _ in range(iterations):
        key = keys[int(rng.integers(0, len(keys)))]
        pool = other_source_event_positions[key]
        if pool.size == 0:
            continue
        idx = rng.integers(0, pool.size, size=n_events)
        vals = target_response[pool[idx]]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        means.append(float(np.mean(vals)))
    return np.array(means, dtype=float)


@dataclass(frozen=True)
class DirectionControls:
    """Per-edge bundle of the three Phase 3C nulls + the observed mean."""

    observed_short_return: float
    n_events: int
    signed_null_p90: float
    signed_null_mean: float
    venue_symmetry_null_p90: float
    venue_symmetry_null_mean: float
    random_symbol_null_p90: float
    random_symbol_null_mean: float
    cost_bps: float

    @property
    def cost_threshold(self) -> float:
        return float(self.cost_bps / 1e4)

    def beats_signed(self) -> bool:
        return self.observed_short_return > self.signed_null_p90

    def beats_venue_symmetry(self) -> bool:
        return self.observed_short_return > self.venue_symmetry_null_p90

    def beats_random_symbol(self) -> bool:
        return self.observed_short_return > self.random_symbol_null_p90

    def passes_all(self) -> bool:
        return (
            self.beats_signed()
            and self.beats_venue_symmetry()
            and self.beats_random_symbol()
            and self.observed_short_return > self.cost_threshold
        )


def build_directional_controls(
    forward_response: np.ndarray,
    forward_event_positions: np.ndarray,
    reverse_response: np.ndarray,
    reverse_event_positions: np.ndarray,
    other_source_event_positions: dict[str, np.ndarray],
    iterations: int,
    rng: np.random.Generator,
    cost_bps: float = DEFAULT_COST_BPS,
) -> DirectionControls:
    """Compute the three nulls and the observed forward mean for one edge."""
    fwd_vals = forward_response[forward_event_positions]
    fwd_vals = fwd_vals[~np.isnan(fwd_vals)]
    observed = float(np.mean(fwd_vals)) if fwd_vals.size else float("nan")

    signed = signed_random_null(forward_response, forward_event_positions, iterations, rng)
    venue = venue_symmetry_null(
        forward_response,
        forward_event_positions,
        reverse_response,
        reverse_event_positions,
        iterations,
        rng,
    )
    rand_sym = random_symbol_source_null(
        forward_response,
        other_source_event_positions,
        n_events=int(fwd_vals.size),
        iterations=iterations,
        rng=rng,
    )

    def _p90_mean(arr: np.ndarray) -> tuple[float, float]:
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(arr, 90)), float(np.mean(arr))

    signed_p90, signed_mean = _p90_mean(signed)
    venue_p90, venue_mean = _p90_mean(venue)
    rand_p90, rand_mean = _p90_mean(rand_sym)

    return DirectionControls(
        observed_short_return=observed,
        n_events=int(fwd_vals.size),
        signed_null_p90=signed_p90,
        signed_null_mean=signed_mean,
        venue_symmetry_null_p90=venue_p90,
        venue_symmetry_null_mean=venue_mean,
        random_symbol_null_p90=rand_p90,
        random_symbol_null_mean=rand_mean,
        cost_bps=cost_bps,
    )


# -- Execution-repair diagnostics (E0..E3) -----------------------------------------


@dataclass(frozen=True)
class ExecutionRepairConfig:
    """All thresholds live here so synthetic tests can dial them in. None of
    these are tuned — they are the docx-named gates literally."""

    # E1: failed bounce confirmation
    bounce_lookahead_bars: int = 3  # 15 min at 5min bars
    bounce_floor_bps: float = 5.0
    failure_drawdown_bps: float = 3.0

    # E2: target micro breakdown
    breakdown_lookback_bars: int = 6  # 30 min recent low window
    breakdown_break_bps: float = 5.0

    # E3: target not-yet-moved filter
    notyet_lookback_bars: int = 3
    notyet_source_drop_bps: float = 20.0  # source must have dropped ≥20bp
    notyet_target_drop_ceil_bps: float = 5.0  # target must have dropped ≤5bp


def _bps(x: float) -> float:
    return float(x) / 1e4


def apply_execution_repair(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_event_mask: pd.Series,
    mode: str,
    cfg: PropagationConfig,
    repair_cfg: ExecutionRepairConfig,
) -> pd.Series:
    """Return a refined boolean event mask (indexed on source_df) for one
    execution repair mode. Modes:

    - ``E0`` — baseline, mask unchanged
    - ``E1`` — failed bounce confirmation on the target
    - ``E2`` — target micro breakdown
    - ``E3`` — target not-yet-moved filter
    """
    if mode == "E0":
        return source_event_mask
    skip = max(0, int(cfg.entry_skip_bars))

    src_times = source_df["bar_open_time"]
    tgt_index = pd.Series(np.arange(len(target_df)), index=target_df["bar_open_time"])
    aligned = tgt_index.reindex(src_times).to_numpy(dtype=float)

    target_vwap = _safe_vwap(target_df).to_numpy()
    source_vwap = _safe_vwap(source_df).to_numpy()

    keep = source_event_mask.to_numpy().copy()
    n_src = len(source_df)
    n_tgt = len(target_df)

    for i in range(n_src):
        if not keep[i]:
            continue
        ti = aligned[i]
        if np.isnan(ti):
            keep[i] = False
            continue
        ti = int(ti)
        start = ti + skip
        if start >= n_tgt:
            keep[i] = False
            continue
        v0 = target_vwap[start]
        if np.isnan(v0) or v0 == 0:
            keep[i] = False
            continue

        if mode == "E1":
            end = min(n_tgt, start + 1 + repair_cfg.bounce_lookahead_bars)
            window = target_vwap[start:end]
            if window.size == 0:
                keep[i] = False
                continue
            peak = float(np.nanmax(window))
            tail = window[-1]
            had_bounce = peak >= v0 * (1.0 + _bps(repair_cfg.bounce_floor_bps))
            failed = tail <= peak * (1.0 - _bps(repair_cfg.failure_drawdown_bps))
            keep[i] = had_bounce and failed
        elif mode == "E2":
            lookback_start = max(0, start - repair_cfg.breakdown_lookback_bars)
            window = target_vwap[lookback_start : start + 1]
            if window.size == 0:
                keep[i] = False
                continue
            recent_low = float(np.nanmin(window[:-1])) if window.size > 1 else float(window[0])
            if np.isnan(recent_low) or recent_low == 0:
                keep[i] = False
                continue
            keep[i] = v0 <= recent_low * (1.0 - _bps(repair_cfg.breakdown_break_bps))
        elif mode == "E3":
            # Anchor on BOTH sides at the same wall-clock bar (event_bar minus
            # K), not at start (which includes the post-event skip). This
            # keeps the source/target drift comparison symmetric.
            src_anchor_idx = max(0, i - repair_cfg.notyet_lookback_bars)
            tgt_anchor_idx = max(0, ti - repair_cfg.notyet_lookback_bars)
            src_anchor = float(source_vwap[src_anchor_idx])
            tgt_anchor = float(target_vwap[tgt_anchor_idx])
            if np.isnan(src_anchor) or src_anchor == 0 or np.isnan(tgt_anchor) or tgt_anchor == 0:
                keep[i] = False
                continue
            src_drift = source_vwap[i] / src_anchor - 1.0
            tgt_drift = v0 / tgt_anchor - 1.0
            src_already_dropped = src_drift <= -_bps(repair_cfg.notyet_source_drop_bps)
            tgt_not_yet = tgt_drift >= -_bps(repair_cfg.notyet_target_drop_ceil_bps)
            keep[i] = bool(src_already_dropped and tgt_not_yet)
        else:
            raise ValueError(f"unknown execution-repair mode: {mode}")
    return pd.Series(keep, index=source_event_mask.index)


def run_execution_repair_diagnostic(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_events: pd.Series,
    event_name: str,
    lag_minutes: int,
    cfg: PropagationConfig,
    repair_cfg: ExecutionRepairConfig,
    rng: np.random.Generator,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Evaluate ``measure_edge`` under each repair mode and return one row per
    mode. Diagnostic only — the docx forbids promoting on these results."""
    lag_bars = lag_minutes // cfg.bar_size_minutes
    grid = precompute_response_grid(target_df, lag_bars, cfg)
    rows = []
    for mode in ("E0", "E1", "E2", "E3"):
        refined_mask = apply_execution_repair(
            source_df, target_df, source_events, mode, cfg, repair_cfg
        )
        row = measure_edge(
            source_df=source_df,
            target_df=target_df,
            event_mask=refined_mask,
            event_name=event_name,
            lag_minutes=lag_minutes,
            cfg=cfg,
            rng=rng,
            response_grid=grid,
        )
        short = row["short_return_mean"]
        cost_net = float("nan") if pd.isna(short) else float(short - cost_bps / 1e4)
        rows.append(
            {
                "mode": mode,
                "n_events": row["n_events"],
                "short_return_mean": short,
                "adverse_rate": row["adverse_upsample"],
                "bootstrap_ci_low": row["bootstrap_ci_low"],
                "cost_net": cost_net,
            }
        )
    return pd.DataFrame(rows)


# -- Graph node registry -----------------------------------------------------------


def register_propagation_graph_node(
    edge_id: str,
    source_label: str,
    target_label: str,
    lags_minutes: tuple[int, ...],
    statistically_real: bool,
    tradeable_after_cost: bool,
    notes: str = "",
) -> dict:
    """Build the diagnostic graph-node dict the docx prescribes.

    Per the docx, the node is **not** a live short candidate even when every
    Phase 3 gate passes; it is a context node used elsewhere in the graph
    (long-side risk warning, basket risk context, future short candidate
    source). The caller is responsible for writing the dict to disk.
    """
    return {
        "edge_id": edge_id,
        "edge": f"{source_label} -> {target_label}",
        "edge_type": "downside_sell_pressure",
        "lag_minutes": list(lags_minutes),
        "status_statistically_real": bool(statistically_real),
        "status_tradeable_after_cost": bool(tradeable_after_cost),
        "use_cases": [
            f"{target_label.split(':')[-1]} long risk warning",
            "DOGE-related meme basket risk context",
            "cross-venue propagation graph node",
            "future short candidate source edge",
        ],
        "notes": notes,
    }


# -- Top-level driver --------------------------------------------------------------


@dataclass(frozen=True)
class Phase3Inputs:
    binance_root: Path
    bybit_root: Path
    basket: tuple[str, ...] = DEFAULT_DOGE_LIKE_BASKET
    primary_source_venue: str = "bybit"
    primary_target_venue: str = "binance_um"
    primary_symbol: str = "DOGEUSDT"
    bar_size_minutes: int = 5
    cost_bps: float = DEFAULT_COST_BPS
    cfg: PropagationConfig = field(default_factory=PropagationConfig)
    repair_cfg: ExecutionRepairConfig = field(default_factory=ExecutionRepairConfig)
    gates: Phase3Gates = field(default_factory=Phase3Gates)
    null_iterations: int = 1000


def run_phase3(inputs: Phase3Inputs, report_root: Path) -> dict[str, Path]:
    """Top-level Phase 3 driver. Runs every stage that has on-disk data and
    surfaces the missing-data gaps explicitly in the summary so the operator
    knows what backfill is required to close them.
    """
    ensure_dir(report_root)
    cfg = inputs.cfg
    cost_bps = inputs.cost_bps

    # --- Phase 3B inventory: discover the dual-venue basket on disk -----------------
    basket_specs, missing_symbols = discover_dual_venue_basket(
        inputs.binance_root, inputs.bybit_root, inputs.basket, cfg.bar_size_minutes
    )
    primary_basket_symbol = inputs.primary_symbol
    primary_source_label = f"{inputs.primary_source_venue}:{primary_basket_symbol}"
    primary_target_label = f"{inputs.primary_target_venue}:{primary_basket_symbol}"

    inventory_rows: list[dict] = []
    for spec in basket_specs:
        months = available_months_for_spec(spec, cfg.bar_size_minutes)
        inventory_rows.append(
            {
                "label": spec.label,
                "venue": spec.venue,
                "symbol": spec.symbol,
                "months_on_disk": ",".join(months),
                "n_months_on_disk": len(months),
            }
        )
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_path = report_root / "dual_venue_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)

    # --- Phase 3A: per-month edge ------------------------------------------------
    primary_specs = tuple(
        s for s in basket_specs if s.symbol == primary_basket_symbol
    )
    per_month_df = pd.DataFrame()
    phase3a = {"status": "no_data", "reason": "primary symbol not on both venues"}
    if len(primary_specs) >= 2:
        focus_lags = inputs.cfg.lag_windows_min
        focus = (
            FocusEdge(
                source_label=primary_source_label,
                target_label=primary_target_label,
                event_type="E1",
                lag_minutes=focus_lags,
            ),
            FocusEdge(
                source_label=primary_target_label,
                target_label=primary_source_label,
                event_type="E1",
                lag_minutes=focus_lags,
            ),
        )
        per_month_df = build_per_month_edge_table(
            primary_specs, focus, cfg, cost_bps=cost_bps
        )
        forward_per_month = per_month_df[
            (per_month_df["source"] == primary_source_label)
            & (per_month_df["target"] == primary_target_label)
        ].copy()
        phase3a = phase3a_verdict(forward_per_month, inputs.gates)
    per_month_path = report_root / "phase3a_per_month.csv"
    per_month_df.to_csv(per_month_path, index=False)

    # --- Phase 3B: basket head-to-head --------------------------------------------
    basket_edge_map = pd.DataFrame()
    forward_head = pd.DataFrame()
    reverse_head = pd.DataFrame()
    phase3b = {"per_symbol": [], "asymmetric_forward_count": 0}
    if basket_specs and any(s.venue == inputs.primary_source_venue for s in basket_specs):
        basket_edge_map = build_propagation_map_multi_venue(basket_specs, cfg)
        forward_head = summarize_head_to_head(
            basket_edge_map, inputs.primary_source_venue, inputs.primary_target_venue
        )
        reverse_head = summarize_head_to_head(
            basket_edge_map, inputs.primary_target_venue, inputs.primary_source_venue
        )
        phase3b = phase3b_verdict(forward_head, reverse_head, cost_bps=cost_bps)
    basket_map_path = report_root / "phase3b_basket_edge_map.csv"
    basket_edge_map.to_csv(basket_map_path, index=False)
    forward_head_path = report_root / "phase3b_forward_head_to_head.csv"
    forward_head.to_csv(forward_head_path, index=False)
    reverse_head_path = report_root / "phase3b_reverse_head_to_head.csv"
    reverse_head.to_csv(reverse_head_path, index=False)
    per_symbol_path = report_root / "phase3b_per_symbol_verdict.csv"
    pd.DataFrame(phase3b.get("per_symbol", [])).to_csv(per_symbol_path, index=False)

    # --- Phase 3C: directional controls for primary edge ---------------------------
    controls_df = pd.DataFrame()
    phase3c_pass = False
    if len(primary_specs) >= 2:
        cvd_by_label = {
            s.label: load_continuous_cvd(s.cvd_root, s.symbol, cfg.bar_size_minutes)
            for s in basket_specs
        }
        start, end = intersect_symbol_windows(cvd_by_label)
        cvd_by_label = {k: restrict_to_window(v, start, end) for k, v in cvd_by_label.items()}
        events_by_label = {label: detect_events(df, cfg) for label, df in cvd_by_label.items()}
        rng = np.random.default_rng(cfg.rng_seed)

        controls_rows = []
        for lag_minutes in (5, 15, 30):
            lag_bars = lag_minutes // cfg.bar_size_minutes
            target_df = cvd_by_label[primary_target_label]
            reverse_target_df = cvd_by_label[primary_source_label]
            forward_grid = precompute_response_grid(target_df, lag_bars, cfg)
            reverse_grid = precompute_response_grid(reverse_target_df, lag_bars, cfg)

            source_df = cvd_by_label[primary_source_label]
            forward_mask = events_by_label[primary_source_label]["E1"]
            forward_positions = _event_positions_on_target(
                source_df, target_df, forward_mask, forward_grid
            )

            reverse_source_df = cvd_by_label[primary_target_label]
            reverse_mask = events_by_label[primary_target_label]["E1"]
            reverse_positions = _event_positions_on_target(
                reverse_source_df, reverse_target_df, reverse_mask, reverse_grid
            )

            other_pools: dict[str, np.ndarray] = {}
            for label, ev in events_by_label.items():
                if label in (primary_source_label, primary_target_label):
                    continue
                if label.split(":")[0] != inputs.primary_source_venue:
                    continue
                src_df_other = cvd_by_label[label]
                pool = _event_positions_on_target(
                    src_df_other, target_df, ev["E1"], forward_grid
                )
                if pool.size > 0:
                    other_pools[label] = pool

            controls = build_directional_controls(
                forward_response=forward_grid.short_return,
                forward_event_positions=forward_positions,
                reverse_response=reverse_grid.short_return,
                reverse_event_positions=reverse_positions,
                other_source_event_positions=other_pools,
                iterations=inputs.null_iterations,
                rng=rng,
                cost_bps=cost_bps,
            )
            controls_rows.append(
                {
                    "lag_minutes": lag_minutes,
                    "n_events": controls.n_events,
                    "observed_short_return": controls.observed_short_return,
                    "signed_null_p90": controls.signed_null_p90,
                    "signed_null_mean": controls.signed_null_mean,
                    "venue_symmetry_null_p90": controls.venue_symmetry_null_p90,
                    "venue_symmetry_null_mean": controls.venue_symmetry_null_mean,
                    "random_symbol_null_p90": controls.random_symbol_null_p90,
                    "random_symbol_null_mean": controls.random_symbol_null_mean,
                    "beats_signed": controls.beats_signed(),
                    "beats_venue_symmetry": controls.beats_venue_symmetry(),
                    "beats_random_symbol": controls.beats_random_symbol(),
                    "passes_all": controls.passes_all(),
                }
            )
        controls_df = pd.DataFrame(controls_rows)
        if not controls_df.empty:
            phase3c_pass = bool(controls_df.loc[controls_df["lag_minutes"] == 5, "passes_all"].any())
    controls_path = report_root / "phase3c_direction_controls.csv"
    controls_df.to_csv(controls_path, index=False)

    # --- Execution repair diagnostic E0..E3 ---------------------------------------
    repair_df = pd.DataFrame()
    if len(primary_specs) >= 2:
        cvd_by_label = {
            s.label: load_continuous_cvd(s.cvd_root, s.symbol, cfg.bar_size_minutes)
            for s in primary_specs
        }
        start, end = intersect_symbol_windows(cvd_by_label)
        cvd_by_label = {k: restrict_to_window(v, start, end) for k, v in cvd_by_label.items()}
        events_by_label = {label: detect_events(df, cfg) for label, df in cvd_by_label.items()}
        rng = np.random.default_rng(cfg.rng_seed + 1)
        source_df = cvd_by_label[primary_source_label]
        target_df = cvd_by_label[primary_target_label]
        event_mask = events_by_label[primary_source_label]["E1"]
        repair_df = run_execution_repair_diagnostic(
            source_df=source_df,
            target_df=target_df,
            source_events=event_mask,
            event_name="E1",
            lag_minutes=5,
            cfg=cfg,
            repair_cfg=inputs.repair_cfg,
            rng=rng,
            cost_bps=cost_bps,
        )
    repair_path = report_root / "phase3_execution_repair.csv"
    repair_df.to_csv(repair_path, index=False)

    # --- Graph node ----------------------------------------------------------------
    statistically_real = bool(
        not per_month_df.empty
        and phase3c_pass
        and phase3a.get("passes_months_gate", False)
    )
    tradeable_after_cost = bool(
        statistically_real
        and not repair_df.empty
        and bool((repair_df["cost_net"] > 0).any())
    )
    node = register_propagation_graph_node(
        edge_id=f"{primary_source_label}->{primary_target_label}@E1",
        source_label=primary_source_label,
        target_label=primary_target_label,
        lags_minutes=(5, 15, 30),
        statistically_real=statistically_real,
        tradeable_after_cost=tradeable_after_cost,
        notes=(
            "Phase 3 verdict — Phase 3A=" + str(phase3a.get("status"))
            + f"; passes_3c={phase3c_pass}; "
            + f"missing_basket={list(missing_symbols)}"
        ),
    )
    graph_path = report_root / "graph_nodes.json"
    graph_path.write_text(json.dumps([node], indent=2), encoding="utf-8")

    # --- Summary markdown ---------------------------------------------------------
    summary_path = report_root / "summary.md"
    summary_path.write_text(
        _format_summary(
            inputs=inputs,
            basket_specs=basket_specs,
            missing_symbols=missing_symbols,
            per_month_df=per_month_df,
            phase3a=phase3a,
            phase3b=phase3b,
            controls_df=controls_df,
            repair_df=repair_df,
            graph_node=node,
        ),
        encoding="utf-8",
    )

    return {
        "inventory_csv": inventory_path,
        "phase3a_per_month_csv": per_month_path,
        "phase3b_basket_edge_map_csv": basket_map_path,
        "phase3b_forward_head_csv": forward_head_path,
        "phase3b_reverse_head_csv": reverse_head_path,
        "phase3b_per_symbol_csv": per_symbol_path,
        "phase3c_direction_controls_csv": controls_path,
        "phase3_execution_repair_csv": repair_path,
        "graph_nodes_json": graph_path,
        "summary_md": summary_path,
    }


def _format_summary(
    inputs: Phase3Inputs,
    basket_specs: tuple[VenueSymbolSpec, ...],
    missing_symbols: tuple[str, ...],
    per_month_df: pd.DataFrame,
    phase3a: dict,
    phase3b: dict,
    controls_df: pd.DataFrame,
    repair_df: pd.DataFrame,
    graph_node: dict,
) -> str:
    lines: list[str] = [
        "# Sell-Pressure Propagation Map — Phase 3",
        "",
        "_Goal docx prompt9 (2026-06-18): DOGE edge validation. Multi-month + basket + extra controls + execution-repair diagnostic. Even on full pass, the verdict is a propagation diagnostic graph node, not a real short._",
        "",
        "## Phase 3B inventory — dual-venue basket on disk",
        "",
        f"- Symbols requested: {list(inputs.basket)}",
        f"- Symbols with parquets on BOTH venues: "
        f"{sorted({s.symbol for s in basket_specs})}",
        f"- **Missing** (would require backfill): {list(missing_symbols)}",
        "",
        "## Phase 3A — per-month stability of the primary edge",
        "",
        f"- Primary edge: `{inputs.primary_source_venue}:{inputs.primary_symbol}` "
        f"→ `{inputs.primary_target_venue}:{inputs.primary_symbol}`",
        f"- Verdict: **{phase3a.get('status')}** "
        f"(n_months_evaluated={phase3a.get('n_months_evaluated')}, "
        f"n_months_positive_ci={phase3a.get('n_months_positive_ci')}, "
        f"max_single_month_event_share={phase3a.get('max_single_month_event_share')})",
    ]
    if not per_month_df.empty:
        headline = per_month_df[
            (per_month_df["event_type"] == "E1")
            & (per_month_df["lag_minutes"].isin([5, 15, 30]))
        ].copy()
        if not headline.empty:
            lines += ["", headline.round(5).to_markdown(index=False)]
    else:
        lines += [
            "",
            "_Per-month table empty — no dual-venue overlap for the primary symbol._",
        ]
    if phase3a.get("status") == "regime_event":
        lines += [
            "",
            "**Interpretation:** the edge is a 2025-10 regime event, not a stable propagation. "
            "Do not promote. The next-step blocker is Bybit multi-month backfill.",
        ]
    elif phase3a.get("status") == "no_data":
        lines += [
            "",
            "**Interpretation:** insufficient on-disk overlap to evaluate stability. "
            "Bybit DOGEUSDT continuous CVD needs 2025-08, 2025-09, 2025-11 and 2-3 ordinary months "
            "backfilled before the 3A gate can be evaluated. The A100 box has the aggTrades download "
            "infrastructure; see [[live-pipeline-proxy-contract]] for the proxy contract.",
        ]

    lines += [
        "",
        "## Phase 3B — basket head-to-head (forward = bybit→binance)",
        "",
        f"- Asymmetric-forward symbols: {phase3b.get('asymmetric_forward_count')}",
        f"- Shared (likely market drift) symbols: {phase3b.get('shared_count')}",
        f"- Flat: {phase3b.get('flat_count')}",
        f"- Incomplete: {phase3b.get('incomplete_count', 0)}",
    ]
    per_sym = phase3b.get("per_symbol", [])
    if per_sym:
        lines += [
            "",
            pd.DataFrame(per_sym).round(5).to_markdown(index=False),
        ]
    if len(missing_symbols) > 0:
        lines += [
            "",
            f"**Basket coverage gap:** {len(missing_symbols)} of {len(inputs.basket)} "
            "meme symbols are missing on at least one venue. Without these, Phase 3B cannot "
            "tell whether the Bybit→Binance lead is DOGE-only or basket-wide.",
        ]

    lines += [
        "",
        "## Phase 3C — directional controls",
        "",
    ]
    if controls_df.empty:
        lines += ["_No controls computed — primary symbol not on both venues._"]
    else:
        lines += [controls_df.round(5).to_markdown(index=False)]
        lag5 = controls_df[controls_df["lag_minutes"] == 5]
        if not lag5.empty:
            passes = bool(lag5["passes_all"].any())
            lines += [
                "",
                f"**Lag-5m verdict:** observed {'beats every null' if passes else 'fails at least one null'}.",
            ]

    lines += [
        "",
        "## Execution-repair diagnostic (E0 baseline; E1..E3 docx-named filters)",
        "",
    ]
    if repair_df.empty:
        lines += ["_No diagnostic computed._"]
    else:
        lines += [
            repair_df.round(5).to_markdown(index=False),
            "",
            "Diagnostic only — the docx forbids promoting on these results. "
            "A positive cost_net here updates the graph-node `status_tradeable_after_cost` flag "
            "but does not make the edge a live short.",
        ]

    lines += [
        "",
        "## Graph node registered",
        "",
        "```json",
        json.dumps(graph_node, indent=2),
        "```",
        "",
        "Use cases: long-side risk warning (Binance DOGE), meme basket risk context, "
        "and a source vertex for future propagation-graph queries. Not a tradeable short.",
        "",
    ]
    return "\n".join(lines) + "\n"
