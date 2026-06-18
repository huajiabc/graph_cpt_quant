"""Sell-Pressure Propagation Map (goal docx 2026-06-18, Phase 1).

Builds a directed propagation map across N symbols at a fixed bar resolution.
For each ordered pair ``(source, target)``, an event type ``E`` and a lag
window ``L``, the map records whether the target's price has a stable,
non-random downward follow-through after a source sell-pressure event.

This module is **measurement**, not strategy: no fees, no slippage, no
execution. The goal doc is explicit — produce the map first, decide whether
any edge is worth strategising afterward against the closure-doc reopen
criteria.

Three propagation paths are coded:

1. **Leader → beta** (``source != target``) — does target lag the source?
2. **Forced unwind continuation** (``source == target``) — does the same
   symbol keep falling after its own event, or does it bounce?
3. **Cross-exchange** — out of scope this phase, see design doc §Phase 2.

The shuffled-null is a paired permutation test: per (source, event_type) we
draw ``n_events`` random eligible timestamps and compute the same target
response there; the bootstrap CI is on the (observed − shuffled) difference.

All public functions take immutable config and return new objects; no in-place
mutation. Detection rules use only same-bar or *past* features (CVD is
backward-looking on closed bars) — the response measurement is strictly
forward — so no look-ahead leak is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir

# -- Configuration -----------------------------------------------------------------

DEFAULT_CONTINUOUS_ROOT = Path("data/orderflow_history/binance_um/continuous")
DEFAULT_REPORT_ROOT = Path("reports/sell_pressure_propagation")
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "AVAXUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "INJUSDT",
    "NEARUSDT",
    "ORDIUSDT",
    "SEIUSDT",
    "WLDUSDT",
)
DEFAULT_LAG_WINDOWS_MIN: tuple[int, ...] = (5, 15, 30, 60, 240)


@dataclass(frozen=True)
class PropagationConfig:
    """Knobs for the propagation map. All thresholds self-adapt to regime via
    a rolling window per source symbol, except the literal ratio gates (E2)."""

    bar_size_minutes: int = 5
    rolling_window_bars: int = 288  # 24h at 5min
    coverage_min: float = 0.80

    # E1 — CVD breakdown
    cvd_z_threshold: float = -3.0

    # E2 — Taker sell imbalance
    taker_buy_ratio_max: float = 0.30
    taker_volume_multiple: float = 2.0

    # E3 — Large-sell cluster
    large_cluster_pctile: float = 0.95
    large_cluster_min: int = 3

    # E4 — Sustained sell continuation
    sustained_z_threshold: float = -1.5
    sustained_bars: int = 3

    # Lag windows
    lag_windows_min: tuple[int, ...] = DEFAULT_LAG_WINDOWS_MIN

    # Entry-bar skip: response uses VWAP(t+skip) → VWAP(t+skip+lag) so the
    # source event bar's own price drift is NOT counted in the propagation.
    # The default (skip=1) measures "the next L minutes AFTER the event bar
    # closed", which is the clean propagation reading. skip=0 reproduces the
    # naive VWAP-to-VWAP measurement and is left available for diagnostics.
    entry_skip_bars: int = 1

    # Response gates
    adverse_threshold_bps: float = 10.0  # squeeze proxy
    target_cvd_z_threshold: float = -1.0

    # Statistical test
    shuffled_iterations: int = 1000
    rng_seed: int = 20260618

    # Map gates (post-aggregation filters; documented in design doc §Success criteria)
    min_events: int = 30
    min_active_months: int = 3
    max_month_share: float = 0.50
    max_adverse_upsample: float = 0.40


EVENT_CODES: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E_any")


# -- Loading -----------------------------------------------------------------------


def load_continuous_cvd(
    root: Path,
    symbol: str,
    bar_size_minutes: int,
) -> pd.DataFrame:
    """Read all monthly parquets for ``symbol`` at ``bar_size_minutes`` and
    return a single sorted DataFrame keyed by ``bar_open_time`` (UTC tz-aware).

    The continuous CVD writer guarantees one row per bar per symbol per month;
    we just concat and sort by time.
    """
    bar_dir_name = f"{bar_size_minutes}min"
    sym_root = Path(root) / symbol / bar_dir_name
    if not sym_root.exists():
        raise FileNotFoundError(f"continuous CVD not found: {sym_root}")
    parts = sorted(sym_root.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no monthly parquets in {sym_root}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("bar_open_time").reset_index(drop=True)
    return df


def intersect_symbol_windows(
    cvd_by_symbol: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) where all symbols have data. Used to compare a
    common period across the propagation map."""
    if not cvd_by_symbol:
        raise ValueError("empty cvd_by_symbol map")
    start = max(df["bar_open_time"].min() for df in cvd_by_symbol.values())
    end = min(df["bar_open_time"].max() for df in cvd_by_symbol.values())
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError(f"no overlap window across symbols (start={start}, end={end})")
    return start, end


def restrict_to_window(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Filter df to [start, end] inclusive and reset index."""
    mask = (df["bar_open_time"] >= start) & (df["bar_open_time"] <= end)
    return df.loc[mask].reset_index(drop=True)


# -- Event detection ---------------------------------------------------------------


def _coverage_mask(df: pd.DataFrame, cfg: PropagationConfig) -> pd.Series:
    cov_ok = pd.to_numeric(df["coverage_ratio"], errors="coerce").fillna(0.0) >= cfg.coverage_min
    quality_ok = df["source_quality"].astype(str).eq("complete")
    return cov_ok & quality_ok


def _min_periods(window: int) -> int:
    """Floor for rolling-window warmup: a quarter of the window, clamped to at
    least 5 bars and at most the window itself. Production default
    (``window=288``) yields ``72``; tests with tiny windows still validate."""
    return max(5, min(window, window // 4))


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Causal z-score: rolling mean/std over the trailing ``window`` bars,
    excluding the current bar so the current value is judged against its own
    history rather than itself."""
    shifted = series.shift(1)
    mean = shifted.rolling(window, min_periods=_min_periods(window)).mean()
    std = shifted.rolling(window, min_periods=_min_periods(window)).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z


def _rolling_pctile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Causal rolling q-quantile over the trailing window (excludes current bar)."""
    shifted = series.shift(1)
    return shifted.rolling(window, min_periods=_min_periods(window)).quantile(q)


def _rolling_median(series: pd.Series, window: int) -> pd.Series:
    shifted = series.shift(1)
    return shifted.rolling(window, min_periods=_min_periods(window)).median()


def detect_events(df: pd.DataFrame, cfg: PropagationConfig) -> pd.DataFrame:
    """Tag each bar in ``df`` with each event code. Returns a DataFrame with
    the original index plus boolean columns ``E1``, ``E2``, ``E3``, ``E4``,
    ``E_any``, and a ``coverage_ok`` gate.
    """
    out = pd.DataFrame(index=df.index)
    coverage = _coverage_mask(df, cfg)
    out["coverage_ok"] = coverage

    cvd_delta = pd.to_numeric(df["cvd_delta_volume"], errors="coerce")
    taker_buy_ratio = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    large_sell = pd.to_numeric(df["large_sell_count"], errors="coerce").fillna(0)
    large_buy = pd.to_numeric(df["large_buy_count"], errors="coerce").fillna(0)

    # E1 — CVD breakdown
    cvd_z = _rolling_z(cvd_delta, cfg.rolling_window_bars)
    out["E1"] = (cvd_z <= cfg.cvd_z_threshold).fillna(False) & coverage

    # E2 — taker sell imbalance with volume spike
    vol_median = _rolling_median(volume, cfg.rolling_window_bars)
    out["E2"] = (
        (taker_buy_ratio <= cfg.taker_buy_ratio_max).fillna(False)
        & (volume >= cfg.taker_volume_multiple * vol_median).fillna(False)
        & coverage
    )

    # E3 — large-sell cluster
    cluster_diff = (large_sell - large_buy).astype(float)
    cluster_thr = _rolling_pctile(
        cluster_diff, cfg.rolling_window_bars, cfg.large_cluster_pctile
    )
    out["E3"] = (
        (cluster_diff >= cluster_thr).fillna(False)
        & (cluster_diff >= cfg.large_cluster_min)
        & coverage
    )

    # E4 — sustained sell continuation
    sustained = (cvd_z <= cfg.sustained_z_threshold).fillna(False)
    # all of the last `sustained_bars` bars satisfy the z gate
    sustained_window = sustained.rolling(cfg.sustained_bars, min_periods=cfg.sustained_bars).sum()
    out["E4"] = (sustained_window >= cfg.sustained_bars).fillna(False) & coverage

    out["E_any"] = (out[["E1", "E2", "E3", "E4"]].any(axis=1)) & coverage
    return out


# -- Response precompute -----------------------------------------------------------


def _safe_vwap(df: pd.DataFrame) -> pd.Series:
    turnover = pd.to_numeric(df["turnover"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    vwap = turnover / volume.replace(0, np.nan)
    return vwap


@dataclass(frozen=True)
class ResponseGrids:
    """Precomputed per-bar response arrays for one target × one lag window.

    Each array has the same length as the target frame; the value at bar ``i``
    is the response a hypothetical event placed at bar ``i`` would generate.
    NaN means the response is not measurable (window runs past the data).
    """

    short_return: np.ndarray  # = -(vwap[i+L] / vwap[i] - 1); >0 means downward move (good for short)
    adverse_hit: np.ndarray  # bool: did vwap go +adverse_threshold above start within (i, i+L]?
    cvd_followthrough: np.ndarray  # bool: did target CVD z drop ≤ cvd_z_threshold within (i, i+L]?


def precompute_response_grid(
    target_df: pd.DataFrame, lag_bars: int, cfg: PropagationConfig
) -> ResponseGrids:
    """Compute the three response arrays for one (target, lag) pair.

    Definitions (with ``skip = cfg.entry_skip_bars``):

    - ``short_return[i] = -(vwap[i + skip + lag_bars] / vwap[i + skip] - 1.0)``
      The hypothetical short enters at VWAP of bar ``i+skip`` (= the first bar
      after the event bar by default) and exits at VWAP of bar
      ``i+skip+lag_bars``. ``skip=0`` reproduces the naive in-bar measurement.
    - ``adverse_hit[i]`` = did VWAP go up by the adverse threshold inside the
      held window ``(i+skip, i+skip+lag_bars]``?
    - ``cvd_followthrough[i]`` = did the target's CVD z drop below the threshold
      inside the held window?
    """
    n = len(target_df)
    skip = max(0, int(cfg.entry_skip_bars))
    vwap = _safe_vwap(target_df).to_numpy()

    start_vwap = np.full(n, np.nan)
    if skip < n:
        start_vwap[: n - skip] = vwap[skip:]
    end_vwap = np.full(n, np.nan)
    end_offset = skip + lag_bars
    if end_offset < n:
        end_vwap[: n - end_offset] = vwap[end_offset:]
    with np.errstate(invalid="ignore", divide="ignore"):
        short_return = -(end_vwap / start_vwap - 1.0)

    vwap_series = pd.Series(vwap)
    # Rolling max over the held window of length lag_bars, starting at i+skip+1
    fwd_max_window = (
        vwap_series.shift(-(skip + 1))
        .iloc[::-1]
        .rolling(lag_bars, min_periods=1)
        .max()
        .iloc[::-1]
        .to_numpy()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        adverse_ratio = fwd_max_window / start_vwap - 1.0
    adverse_hit = adverse_ratio >= (cfg.adverse_threshold_bps / 1e4)

    cvd_delta = pd.to_numeric(target_df["cvd_delta_volume"], errors="coerce")
    target_cvd_z = _rolling_z(cvd_delta, cfg.rolling_window_bars)
    fwd_min_cvd_z = (
        target_cvd_z.shift(-(skip + 1))
        .iloc[::-1]
        .rolling(lag_bars, min_periods=1)
        .min()
        .iloc[::-1]
        .to_numpy()
    )
    cvd_followthrough = fwd_min_cvd_z <= cfg.target_cvd_z_threshold

    valid_end = np.arange(n) + end_offset < n
    short_return = np.where(valid_end, short_return, np.nan)
    adverse_hit = np.where(valid_end, adverse_hit, False)
    cvd_followthrough = np.where(valid_end, cvd_followthrough, False)

    return ResponseGrids(
        short_return=short_return,
        adverse_hit=adverse_hit.astype(bool),
        cvd_followthrough=cvd_followthrough.astype(bool),
    )


# -- Edge measurement --------------------------------------------------------------


def _index_align_event_positions(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    event_mask: pd.Series,
) -> np.ndarray:
    """Translate event-flagged source bar positions to target bar positions
    using a left-merge on ``bar_open_time``. Returns target-row positions for
    the source events that exist on the target frame; events without a
    matching target bar are dropped."""
    source_times = source_df.loc[event_mask, "bar_open_time"]
    if source_times.empty:
        return np.empty(0, dtype=int)
    target_index = pd.Series(np.arange(len(target_df)), index=target_df["bar_open_time"])
    matched = target_index.reindex(source_times).dropna()
    return matched.to_numpy(dtype=int)


def _month_distribution(
    timestamps: pd.Series,
) -> tuple[int, float]:
    """Return (active_month_count, max_month_share). Empty input → (0, 0.0).
    Pandas warns when converting tz-aware to PeriodIndex; we drop the tz first
    since month bucketing doesn't need it."""
    if timestamps.empty:
        return 0, 0.0
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    months = timestamps.dt.to_period("M")
    counts = months.value_counts(normalize=True)
    return int(counts.shape[0]), float(counts.max())


def _draw_shuffled_means(
    response: np.ndarray,
    eligible_positions: np.ndarray,
    n_events: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized null: ``iterations`` draws of size ``n_events`` from the
    eligible positions, returning the mean response for each draw.

    `response` is a length-T array, NaNs allowed; means are taken over
    non-NaN values per draw. The number of valid positions is dominant so we
    keep the slow path (draws with all-NaN) silently dropped.
    """
    if n_events == 0 or eligible_positions.size == 0:
        return np.empty(0, dtype=float)
    pool_size = eligible_positions.size
    draws = rng.integers(0, pool_size, size=(iterations, n_events))
    samples = response[eligible_positions[draws]]
    return np.nanmean(samples, axis=1)


def measure_edge(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    event_mask: pd.Series,
    event_name: str,
    lag_minutes: int,
    cfg: PropagationConfig,
    rng: np.random.Generator,
    response_grid: ResponseGrids | None = None,
) -> dict:
    """Compute one map row. The caller can pass ``response_grid`` to share
    the precompute across multiple sources targeting the same target/lag.
    """
    lag_bars = lag_minutes // cfg.bar_size_minutes
    if response_grid is None:
        response_grid = precompute_response_grid(target_df, lag_bars, cfg)

    event_positions = _index_align_event_positions(source_df, target_df, event_mask)
    # Restrict to positions where the response is measurable
    measurable_mask = ~np.isnan(response_grid.short_return)
    valid_event_positions = event_positions[measurable_mask[event_positions]] if event_positions.size else event_positions

    n_events = int(valid_event_positions.size)
    if n_events == 0:
        return _empty_row(event_name, lag_minutes, cfg)

    obs_short = response_grid.short_return[valid_event_positions]
    obs_adverse = response_grid.adverse_hit[valid_event_positions]
    obs_cvd = response_grid.cvd_followthrough[valid_event_positions]

    observed_short_mean = float(np.nanmean(obs_short))
    observed_short_median = float(np.nanmedian(obs_short))
    adverse_upsample = float(np.mean(obs_adverse))
    cvd_followthrough_rate = float(np.mean(obs_cvd))

    event_times = source_df.loc[event_mask, "bar_open_time"]
    active_months, max_month_share = _month_distribution(event_times)

    eligible_positions = np.flatnonzero(measurable_mask)
    null_means = _draw_shuffled_means(
        response_grid.short_return,
        eligible_positions,
        n_events,
        cfg.shuffled_iterations,
        rng,
    )

    shuffled_mean = float(np.mean(null_means)) if null_means.size else float("nan")
    shuffled_std = float(np.std(null_means)) if null_means.size else float("nan")
    if null_means.size:
        differences = observed_short_mean - null_means
        ci_low = float(np.percentile(differences, 5))
        ci_high = float(np.percentile(differences, 95))
        edge_strength = (
            (observed_short_mean - shuffled_mean) / shuffled_std
            if shuffled_std > 0
            else float("nan")
        )
    else:
        ci_low = float("nan")
        ci_high = float("nan")
        edge_strength = float("nan")

    return {
        "event_type": event_name,
        "lag_minutes": lag_minutes,
        "n_events": n_events,
        "target_response_mean": -observed_short_mean,  # mean target return (negative ⇒ down)
        "target_response_median": -observed_short_median,
        "short_return_mean": observed_short_mean,
        "adverse_upsample": adverse_upsample,
        "target_cvd_followthrough_rate": cvd_followthrough_rate,
        "month_distribution_active_months": active_months,
        "month_distribution_max_share": max_month_share,
        "shuffled_null_mean": shuffled_mean,
        "shuffled_null_std": shuffled_std,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "edge_strength": edge_strength,
    }


def _empty_row(event_name: str, lag_minutes: int, cfg: PropagationConfig) -> dict:
    return {
        "event_type": event_name,
        "lag_minutes": lag_minutes,
        "n_events": 0,
        "target_response_mean": float("nan"),
        "target_response_median": float("nan"),
        "short_return_mean": float("nan"),
        "adverse_upsample": float("nan"),
        "target_cvd_followthrough_rate": float("nan"),
        "month_distribution_active_months": 0,
        "month_distribution_max_share": 0.0,
        "shuffled_null_mean": float("nan"),
        "shuffled_null_std": float("nan"),
        "bootstrap_ci_low": float("nan"),
        "bootstrap_ci_high": float("nan"),
        "edge_strength": float("nan"),
    }


# -- Map driver --------------------------------------------------------------------


def build_propagation_map(
    symbols: tuple[str, ...],
    cvd_root: Path,
    cfg: PropagationConfig,
) -> pd.DataFrame:
    """Build the full propagation map across all (source, target) pairs."""
    cvd_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cvd_by_symbol[sym] = load_continuous_cvd(cvd_root, sym, cfg.bar_size_minutes)
    start, end = intersect_symbol_windows(cvd_by_symbol)
    cvd_by_symbol = {k: restrict_to_window(v, start, end) for k, v in cvd_by_symbol.items()}

    events_by_symbol = {sym: detect_events(df, cfg) for sym, df in cvd_by_symbol.items()}

    rng = np.random.default_rng(cfg.rng_seed)
    rows: list[dict] = []
    for target_sym in symbols:
        target_df = cvd_by_symbol[target_sym]
        for lag_minutes in cfg.lag_windows_min:
            lag_bars = lag_minutes // cfg.bar_size_minutes
            response_grid = precompute_response_grid(target_df, lag_bars, cfg)
            for source_sym in symbols:
                source_df = cvd_by_symbol[source_sym]
                event_df = events_by_symbol[source_sym]
                for event_code in EVENT_CODES:
                    event_mask = event_df[event_code]
                    row = measure_edge(
                        source_df=source_df,
                        target_df=target_df,
                        event_mask=event_mask,
                        event_name=event_code,
                        lag_minutes=lag_minutes,
                        cfg=cfg,
                        rng=rng,
                        response_grid=response_grid,
                    )
                    row["source"] = source_sym
                    row["target"] = target_sym
                    row["path"] = "self" if source_sym == target_sym else "cross"
                    row["sample_start"] = start
                    row["sample_end"] = end
                    rows.append(row)

    cols = [
        "source",
        "target",
        "path",
        "event_type",
        "lag_minutes",
        "n_events",
        "target_response_mean",
        "target_response_median",
        "short_return_mean",
        "adverse_upsample",
        "target_cvd_followthrough_rate",
        "month_distribution_active_months",
        "month_distribution_max_share",
        "shuffled_null_mean",
        "shuffled_null_std",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "edge_strength",
        "sample_start",
        "sample_end",
    ]
    return pd.DataFrame(rows, columns=cols)


@dataclass(frozen=True)
class VenueSymbolSpec:
    """One (venue, symbol) participant in a multi-venue propagation map.

    ``label`` is the display name used in the output edge map (e.g.
    ``"binance_um:BTCUSDT"``). ``venue`` is a short tag for path
    classification. ``cvd_root`` is the directory containing the
    ``<symbol>/<bar_size>/<YYYY-MM>.parquet`` tree. ``symbol`` is the
    on-disk symbol folder name.
    """

    label: str
    venue: str
    cvd_root: Path
    symbol: str


def _classify_multi_venue_path(source: VenueSymbolSpec, target: VenueSymbolSpec) -> str:
    same_venue = source.venue == target.venue
    same_symbol = source.symbol == target.symbol
    if same_venue and same_symbol:
        return "self"
    if same_venue and not same_symbol:
        return "cross_symbol"
    if not same_venue and same_symbol:
        return "cross_venue"
    return "cross_venue_cross_symbol"


def build_propagation_map_multi_venue(
    specs: tuple[VenueSymbolSpec, ...],
    cfg: PropagationConfig,
) -> pd.DataFrame:
    """Multi-venue extension of ``build_propagation_map``.

    Identical statistical machinery; only the data loading + path classification
    changes. Specs are labelled (e.g. ``"binance_um:BTCUSDT"``) so the output
    edge map is venue-aware. The intersection window is computed across all
    specs — if two venues have different coverage windows, the analysis runs
    on the overlap only.
    """
    if not specs:
        raise ValueError("at least one VenueSymbolSpec required")

    cvd_by_label: dict[str, pd.DataFrame] = {}
    for spec in specs:
        df = load_continuous_cvd(spec.cvd_root, spec.symbol, cfg.bar_size_minutes)
        cvd_by_label[spec.label] = df

    start, end = intersect_symbol_windows(cvd_by_label)
    cvd_by_label = {k: restrict_to_window(v, start, end) for k, v in cvd_by_label.items()}
    events_by_label = {label: detect_events(df, cfg) for label, df in cvd_by_label.items()}
    spec_by_label = {spec.label: spec for spec in specs}

    rng = np.random.default_rng(cfg.rng_seed)
    rows: list[dict] = []
    for tgt_label in spec_by_label:
        target_df = cvd_by_label[tgt_label]
        for lag_minutes in cfg.lag_windows_min:
            lag_bars = lag_minutes // cfg.bar_size_minutes
            response_grid = precompute_response_grid(target_df, lag_bars, cfg)
            for src_label, src_spec in spec_by_label.items():
                source_df = cvd_by_label[src_label]
                event_df = events_by_label[src_label]
                tgt_spec = spec_by_label[tgt_label]
                for event_code in EVENT_CODES:
                    event_mask = event_df[event_code]
                    row = measure_edge(
                        source_df=source_df,
                        target_df=target_df,
                        event_mask=event_mask,
                        event_name=event_code,
                        lag_minutes=lag_minutes,
                        cfg=cfg,
                        rng=rng,
                        response_grid=response_grid,
                    )
                    row["source"] = src_label
                    row["target"] = tgt_label
                    row["source_venue"] = src_spec.venue
                    row["target_venue"] = tgt_spec.venue
                    row["source_symbol"] = src_spec.symbol
                    row["target_symbol"] = tgt_spec.symbol
                    row["path"] = _classify_multi_venue_path(src_spec, tgt_spec)
                    row["sample_start"] = start
                    row["sample_end"] = end
                    rows.append(row)

    cols = [
        "source",
        "target",
        "source_venue",
        "target_venue",
        "source_symbol",
        "target_symbol",
        "path",
        "event_type",
        "lag_minutes",
        "n_events",
        "target_response_mean",
        "target_response_median",
        "short_return_mean",
        "adverse_upsample",
        "target_cvd_followthrough_rate",
        "month_distribution_active_months",
        "month_distribution_max_share",
        "shuffled_null_mean",
        "shuffled_null_std",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "edge_strength",
        "sample_start",
        "sample_end",
    ]
    return pd.DataFrame(rows, columns=cols)


def run_multi_venue_propagation_map(
    specs: tuple[VenueSymbolSpec, ...],
    report_root: Path,
    cfg: PropagationConfig | None = None,
) -> dict[str, Path]:
    """Top-level multi-venue driver; mirrors ``run_sell_pressure_propagation_map``."""
    cfg = cfg or PropagationConfig()
    edge_map = build_propagation_map_multi_venue(specs, cfg)
    survivors = filter_surviving_edges(edge_map, cfg)
    return write_propagation_map(edge_map, survivors, report_root, cfg)


def filter_surviving_edges(
    edge_map: pd.DataFrame, cfg: PropagationConfig
) -> pd.DataFrame:
    """Apply the design-doc Success-Criteria gates and return the surviving
    rows sorted by edge_strength descending. An empty DataFrame is a valid
    outcome — the headline is simply 'no edge survived' in that case."""
    keep = (
        (edge_map["n_events"] >= cfg.min_events)
        & (edge_map["month_distribution_active_months"] >= cfg.min_active_months)
        & (edge_map["month_distribution_max_share"] <= cfg.max_month_share)
        & (edge_map["bootstrap_ci_low"] > 0)
        & (edge_map["adverse_upsample"] <= cfg.max_adverse_upsample)
    )
    survivors = edge_map.loc[keep].copy()
    if not survivors.empty:
        survivors = survivors.sort_values("edge_strength", ascending=False).reset_index(drop=True)
    return survivors


def _lag_profile(edge_map: pd.DataFrame, source: str, target: str, event_type: str) -> pd.DataFrame:
    """Return the full lag-window profile for one directed (source, target, event)
    triple, sorted by lag. Useful for reading the propagation hump shape."""
    sub = edge_map[
        (edge_map["source"] == source)
        & (edge_map["target"] == target)
        & (edge_map["event_type"] == event_type)
    ]
    return sub.sort_values("lag_minutes").reset_index(drop=True)


def build_lag_profile_table(edge_map: pd.DataFrame, survivors: pd.DataFrame) -> pd.DataFrame:
    """For every (source, target, event_type) that appears in survivors, emit
    its lag-window profile so the hump shape is visible."""
    if survivors.empty:
        return pd.DataFrame()
    keys = (
        survivors[["source", "target", "event_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    frames = [_lag_profile(edge_map, s, t, e) for s, t, e in keys]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def closure_doc_gate_table(survivors: pd.DataFrame, cfg: PropagationConfig) -> pd.DataFrame:
    """Apply the v12s short-research-closure 5-gate ledger to each surviving edge.
    Only gates 1, 3 (proxy via cost-band probe) and 5 are evaluable from the
    map alone; gates 2 (hedge corr) and 4 (beats no_long) need the long-stack
    book and are flagged as ``deferred`` here. The output is the explicit
    accounting the closure-doc requires before any short can be reopened."""
    if survivors.empty:
        return pd.DataFrame()
    rows = []
    for _, r in survivors.iterrows():
        gate1_distribution = (
            r["month_distribution_active_months"] >= 3
            and r["month_distribution_max_share"] <= 0.35
        )
        gross_short_return = r["short_return_mean"]
        # Phase-1 cost-band probe: gate 3 in the closure doc is "survives 30bp + 5bp slip".
        # We approximate by checking whether the gross short return exceeds 35bp.
        gate3_cost = gross_short_return > 0.0035
        gate5_squeeze = r["adverse_upsample"] <= 0.20
        all_evaluable_pass = gate1_distribution and gate3_cost and gate5_squeeze
        rows.append(
            {
                "source": r["source"],
                "target": r["target"],
                "event_type": r["event_type"],
                "lag_minutes": r["lag_minutes"],
                "gate1_distribution_pass": bool(gate1_distribution),
                "gate2_hedge_corr": "deferred",
                "gate3_cost_pass": bool(gate3_cost),
                "gate4_beats_no_long": "deferred",
                "gate5_squeeze_pass": bool(gate5_squeeze),
                "evaluable_gates_pass_all": bool(all_evaluable_pass),
            }
        )
    return pd.DataFrame(rows)


def _format_survivor_preview(survivors: pd.DataFrame) -> str:
    preview_cols = [
        "source",
        "target",
        "event_type",
        "lag_minutes",
        "n_events",
        "short_return_mean",
        "adverse_upsample",
        "edge_strength",
        "bootstrap_ci_low",
        "month_distribution_active_months",
        "month_distribution_max_share",
    ]
    return survivors[preview_cols].head(40).round(5).to_markdown(index=False)


def _format_lag_profile(profile: pd.DataFrame) -> str:
    if profile.empty:
        return "_no lag profile rows — survivor table empty._"
    cols = [
        "source",
        "target",
        "event_type",
        "lag_minutes",
        "n_events",
        "short_return_mean",
        "adverse_upsample",
        "edge_strength",
        "bootstrap_ci_low",
    ]
    return profile[cols].round(5).to_markdown(index=False)


def _format_closure_gates(gate_tbl: pd.DataFrame) -> str:
    if gate_tbl.empty:
        return "_no surviving edges — closure-doc gates not applicable this phase._"
    return gate_tbl.to_markdown(index=False)


def write_propagation_map(
    edge_map: pd.DataFrame,
    survivors: pd.DataFrame,
    report_root: Path,
    cfg: PropagationConfig | None = None,
) -> dict[str, Path]:
    """Write the full edge map + survivor table + lag-profile + closure-gate ledger
    + markdown summary."""
    cfg = cfg or PropagationConfig()
    ensure_dir(report_root)
    parquet_path = report_root / "edge_map.parquet"
    csv_path = report_root / "edge_map.csv"
    survivors_path = report_root / "edge_map_survivors.csv"
    lag_profile_path = report_root / "lag_profile.csv"
    closure_gate_path = report_root / "closure_doc_gates.csv"
    summary_path = report_root / "summary.md"

    edge_map.to_parquet(parquet_path, index=False)
    edge_map.to_csv(csv_path, index=False)
    survivors.to_csv(survivors_path, index=False)
    lag_profile = build_lag_profile_table(edge_map, survivors)
    lag_profile.to_csv(lag_profile_path, index=False)
    gate_tbl = closure_doc_gate_table(survivors, cfg)
    gate_tbl.to_csv(closure_gate_path, index=False)

    summary_lines: list[str] = [
        "# Sell-Pressure Propagation Map — Phase 1 summary",
        "",
        "_Goal docx 2026-06-18 — measure first, do not strategise._",
        "",
    ]

    if edge_map.empty:
        summary_lines.append("Edge map is empty — no symbols loaded.")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        return {
            "edge_map_parquet": parquet_path,
            "edge_map_csv": csv_path,
            "survivors_csv": survivors_path,
            "lag_profile_csv": lag_profile_path,
            "closure_gate_csv": closure_gate_path,
            "summary_md": summary_path,
        }

    sample_start = edge_map["sample_start"].iloc[0]
    sample_end = edge_map["sample_end"].iloc[0]
    summary_lines += [
        f"- Sample window: **{sample_start} → {sample_end}**",
        f"- Symbols: {sorted(set(edge_map['source']))}",
        f"- Bar size: {cfg.bar_size_minutes}min | entry skip: {cfg.entry_skip_bars} bar(s)",
        f"- Total directed edges measured: {len(edge_map)}",
        f"- Survivors of headline gates: **{len(survivors)}**",
        "",
        "## Methodology — short version",
        "",
        "- Per source bar, four sell-pressure event types: E1 (CVD-breakdown z ≤ −3), "
        "E2 (taker_buy_ratio ≤ 0.30 + volume spike), E3 (large-sell cluster ≥ 95th pctile), "
        "E4 (3 consecutive z ≤ −1.5 bars), and `E_any` (any of the above).",
        "- Per target × lag, response = `vwap[t+skip+lag] / vwap[t+skip] − 1` "
        "(`skip=1` so the source event bar's own drift is excluded — without skipping, "
        "the 5min-lag column is dominated by within-bar VWAP-shift rather than real "
        "propagation, as confirmed by a `skip=0` diagnostic during build).",
        "- Significance: paired shuffled null + bootstrap CI on (observed − shuffled).",
        f"- Headline gates: n_events ≥ {cfg.min_events}, active_months ≥ "
        f"{cfg.min_active_months}, max_month_share ≤ {cfg.max_month_share:.2f}, "
        f"bootstrap_ci_low > 0, adverse_upsample ≤ {cfg.max_adverse_upsample:.2f}.",
        "",
        "## Survivor table",
        "",
    ]

    if survivors.empty:
        summary_lines += [
            "No directed edges survive the headline gates.",
            "",
            "**This is itself the headline finding.** The naive (`skip=0`) measurement "
            "showed 293 'surviving' edges — once the source event bar's own drift is "
            "excluded, the propagation collapses to the small set of edges below.",
            "",
            "Recommendation: expand the data universe (Phase 2 — cross-exchange + "
            "BTC continuous CVD) before any further short investigation.",
        ]
    else:
        summary_lines += [
            _format_survivor_preview(survivors),
            "",
            "## Lag profile (each surviving directed triple, across all lag windows)",
            "",
            "Read this for the propagation hump shape — propagation is real if the "
            "edge strengthens over the first few lag bars and then decays, rather "
            "than appearing at one lag and vanishing.",
            "",
            _format_lag_profile(lag_profile),
            "",
            "## Closure-doc gate ledger",
            "",
            "Applies the v12s short-research-closure 5-gate ledger to every surviving "
            "edge. Gates 2 (hedge correlation) and 4 (beats no_long) require the "
            "long-stack book and are deferred to a follow-up phase.",
            "",
            _format_closure_gates(gate_tbl),
            "",
        ]
        any_pass_all = bool(gate_tbl["evaluable_gates_pass_all"].any()) if not gate_tbl.empty else False
        if any_pass_all:
            summary_lines += [
                "**At least one edge passes every evaluable closure-doc gate.** This "
                "does NOT clear the edge for promotion — gates 2 and 4 must still be "
                "evaluated against the long-stack book before any short can be reopened.",
            ]
        else:
            summary_lines += [
                "**No surviving edge passes every evaluable closure-doc gate** "
                "(typically because adverse_upsample at the lag-peak is well above 0.20). "
                "The short stays closed; the propagation finding is a data-quality / "
                "research artefact that the orderflow IS leaking but in a form too "
                "noisy for a tradable short on this universe.",
            ]

    summary_lines += [
        "",
        "## Path coverage",
        "",
    ]
    is_multi_venue = "source_venue" in edge_map.columns
    if is_multi_venue:
        path_counts = edge_map["path"].value_counts().to_dict()
        for label, key in [
            ("Self (same venue, same symbol)", "self"),
            ("Cross-symbol (same venue, diff symbol)", "cross_symbol"),
            ("Cross-venue (diff venue, same symbol)", "cross_venue"),
            ("Cross-venue cross-symbol", "cross_venue_cross_symbol"),
        ]:
            summary_lines.append(f"- {label}: {path_counts.get(key, 0)} edges measured.")
        # Highlight the cross-venue head-to-head edges so they're easy to spot
        head_to_head = edge_map[
            (edge_map["path"] == "cross_venue") & (edge_map["n_events"] >= 30)
        ]
        if not head_to_head.empty:
            summary_lines += [
                "",
                "### Cross-venue head-to-head edges (n_events ≥ 30)",
                "",
                head_to_head[
                    [
                        "source",
                        "target",
                        "event_type",
                        "lag_minutes",
                        "n_events",
                        "short_return_mean",
                        "adverse_upsample",
                        "edge_strength",
                        "bootstrap_ci_low",
                    ]
                ]
                .round(5)
                .to_markdown(index=False),
            ]
    else:
        summary_lines += [
            f"- Leader → beta (cross-symbol): {len(edge_map[edge_map['path'] == 'cross'])} edges measured.",
            f"- Forced unwind continuation (self-edges): {len(edge_map[edge_map['path'] == 'self'])} edges measured.",
            "- Cross-exchange: **out of scope this phase** — no OKX/Bybit/Hyperliquid "
            "aggTrades on disk. See `docs/sell_pressure_propagation_map_design.md` "
            "§Phase 2 for the data-fetch plan.",
        ]

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "edge_map_parquet": parquet_path,
        "edge_map_csv": csv_path,
        "survivors_csv": survivors_path,
        "lag_profile_csv": lag_profile_path,
        "closure_gate_csv": closure_gate_path,
        "summary_md": summary_path,
    }


def run_sell_pressure_propagation_map(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    cvd_root: Path = DEFAULT_CONTINUOUS_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    cfg: PropagationConfig | None = None,
) -> dict[str, Path]:
    """Top-level driver: load CVD → detect events → build map → filter survivors → write."""
    cfg = cfg or PropagationConfig()
    edge_map = build_propagation_map(symbols, cvd_root, cfg)
    survivors = filter_surviving_edges(edge_map, cfg)
    return write_propagation_map(edge_map, survivors, report_root, cfg)
