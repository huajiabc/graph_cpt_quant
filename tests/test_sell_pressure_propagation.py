"""Unit + integration tests for the Sell-Pressure Propagation Map.

These tests use synthetic CVD frames (no aggTrades zip dependency) so they
run in <1s on a laptop and prove the building blocks in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.sell_pressure_propagation import (
    EVENT_CODES,
    PropagationConfig,
    ResponseGrids,
    _draw_shuffled_means,
    _month_distribution,
    build_propagation_map,
    detect_events,
    filter_surviving_edges,
    intersect_symbol_windows,
    measure_edge,
    precompute_response_grid,
)


def _make_cvd_frame(
    symbol: str,
    start: str = "2025-06-01 00:00:00",
    n_bars: int = 600,
    base_volume: float = 1000.0,
    base_price: float = 100.0,
    cvd_pattern: np.ndarray | None = None,
    taker_buy_ratio_pattern: np.ndarray | None = None,
    vwap_pattern: np.ndarray | None = None,
    large_sell_pattern: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a synthetic continuous-CVD frame compatible with the loader."""
    ts = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    cvd = (
        cvd_pattern
        if cvd_pattern is not None
        else np.random.default_rng(0).normal(0, 5.0, size=n_bars)
    )
    tb_ratio = (
        taker_buy_ratio_pattern
        if taker_buy_ratio_pattern is not None
        else np.full(n_bars, 0.5)
    )
    vwap = vwap_pattern if vwap_pattern is not None else np.full(n_bars, base_price)
    large_sell = (
        large_sell_pattern if large_sell_pattern is not None else np.zeros(n_bars, dtype=int)
    )

    volume = np.full(n_bars, base_volume)
    turnover = vwap * volume
    buy_volume = volume * tb_ratio
    sell_volume = volume - buy_volume

    return pd.DataFrame(
        {
            "symbol": symbol,
            "bar_open_time": ts,
            "bar_size": "5min",
            "trade_count": np.full(n_bars, 100),
            "volume": volume,
            "turnover": turnover,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_turnover": buy_volume * vwap,
            "sell_turnover": sell_volume * vwap,
            "taker_buy_ratio": tb_ratio,
            "buy_sell_imbalance": 2 * tb_ratio - 1,
            "cvd_delta_volume": cvd,
            "cvd_delta_turnover": cvd * vwap,
            "large_trade_threshold": 50.0,
            "large_buy_count": np.zeros(n_bars, dtype=int),
            "large_sell_count": large_sell,
            "large_buy_turnover": np.zeros(n_bars),
            "large_sell_turnover": large_sell * 100.0,
            "coverage_ratio": np.full(n_bars, 1.0),
            "source_quality": "complete",
        }
    )


# -------------------------------------------------------------------
# Event detection
# -------------------------------------------------------------------


def test_e1_cvd_breakdown_fires_on_real_outlier_bar():
    """The planted -50 sigma outlier must be flagged. Random noise can produce
    its own occasional ≤ -3 sigma bars; that's not a bug, it's the detector
    behaving correctly. We only assert the planted bar is detected."""
    n = 600
    rng = np.random.default_rng(7)
    cvd = rng.normal(0.0, 1.0, size=n)
    cvd[400] = -50.0
    df = _make_cvd_frame("SYNTH", n_bars=n, cvd_pattern=cvd)
    cfg = PropagationConfig(rolling_window_bars=144)
    events = detect_events(df, cfg)
    assert events.loc[400, "E1"]
    # And the rate of unplanted E1 hits over the post-warmup window should be
    # within Gaussian-tail expectations (≤ 1% of bars).
    post_warmup = events.loc[150:, "E1"]
    spurious_rate = (post_warmup.sum() - 1) / max(1, len(post_warmup) - 1)
    assert spurious_rate < 0.01


def test_e2_taker_sell_requires_volume_spike():
    n = 400
    tbr = np.full(n, 0.5)
    tbr[300] = 0.2  # sell-heavy
    df = _make_cvd_frame("SYNTH", n_bars=n, taker_buy_ratio_pattern=tbr)
    cfg = PropagationConfig(rolling_window_bars=144)
    events = detect_events(df, cfg)
    # baseline volume is constant ⇒ no spike ⇒ E2 must not fire
    assert not events["E2"].any()


def test_e2_fires_when_volume_spikes_and_taker_sells():
    n = 400
    tbr = np.full(n, 0.5)
    tbr[300] = 0.2
    df = _make_cvd_frame("SYNTH", n_bars=n, taker_buy_ratio_pattern=tbr)
    df.loc[300, "volume"] = 5_000  # 5× baseline
    df.loc[300, "turnover"] = 5_000 * 100.0
    cfg = PropagationConfig(rolling_window_bars=144)
    events = detect_events(df, cfg)
    assert events.loc[300, "E2"]


def test_e3_large_cluster_requires_min_count_and_pctile():
    n = 500
    cluster = np.zeros(n, dtype=int)
    cluster[400] = 5
    df = _make_cvd_frame("SYNTH", n_bars=n, large_sell_pattern=cluster)
    cfg = PropagationConfig(rolling_window_bars=144, large_cluster_min=3)
    events = detect_events(df, cfg)
    assert events.loc[400, "E3"]


def test_coverage_gate_suppresses_events():
    n = 400
    cvd = np.zeros(n)
    cvd[300] = -50.0
    df = _make_cvd_frame("SYNTH", n_bars=n, cvd_pattern=cvd)
    df.loc[300, "coverage_ratio"] = 0.5  # below default 0.8 gate
    cfg = PropagationConfig(rolling_window_bars=144)
    events = detect_events(df, cfg)
    assert not events.loc[300, "E1"]
    assert not events.loc[300, "E_any"]


# -------------------------------------------------------------------
# Response grid
# -------------------------------------------------------------------


def test_response_grid_short_return_matches_manual_calc():
    """With default ``entry_skip_bars=1``, the response uses
    ``vwap[i+1]`` as start and ``vwap[i+1+lag_bars]`` as end."""
    n = 100
    vwap = np.linspace(100, 90, n)
    df = _make_cvd_frame("SYNTH", n_bars=n, vwap_pattern=vwap)
    cfg = PropagationConfig(rolling_window_bars=20)
    grid = precompute_response_grid(df, lag_bars=10, cfg=cfg)
    expected = -(vwap[11] / vwap[1] - 1.0)
    assert grid.short_return[0] == pytest.approx(expected, rel=1e-6)


def test_response_grid_skip_zero_reproduces_naive_measurement():
    n = 100
    vwap = np.linspace(100, 90, n)
    df = _make_cvd_frame("SYNTH", n_bars=n, vwap_pattern=vwap)
    cfg = PropagationConfig(rolling_window_bars=20, entry_skip_bars=0)
    grid = precompute_response_grid(df, lag_bars=10, cfg=cfg)
    expected = -(vwap[10] / vwap[0] - 1.0)
    assert grid.short_return[0] == pytest.approx(expected, rel=1e-6)


def test_response_grid_handles_tail_with_nans():
    n = 50
    vwap = np.linspace(100, 90, n)
    df = _make_cvd_frame("SYNTH", n_bars=n, vwap_pattern=vwap)
    cfg = PropagationConfig(rolling_window_bars=10)
    grid = precompute_response_grid(df, lag_bars=10, cfg=cfg)
    # last `skip+lag_bars` = 11 positions should be NaN
    assert np.isnan(grid.short_return[-1])
    assert np.isnan(grid.short_return[-11])
    assert not np.isnan(grid.short_return[0])


def test_response_grid_adverse_hit_triggers_on_upward_move():
    n = 30
    vwap = np.full(n, 100.0)
    vwap[6] = 100.5  # +50 bps inside window (i+skip+1 .. i+skip+lag_bars] = (2..11]
    df = _make_cvd_frame("SYNTH", n_bars=n, vwap_pattern=vwap)
    cfg = PropagationConfig(rolling_window_bars=10, adverse_threshold_bps=10.0)
    grid = precompute_response_grid(df, lag_bars=10, cfg=cfg)
    # bar 0 with skip=1: window is (2, 11], contains the spike at bar 6
    assert bool(grid.adverse_hit[0])


# -------------------------------------------------------------------
# Shuffled null
# -------------------------------------------------------------------


def test_shuffled_means_uses_eligible_positions_only():
    rng = np.random.default_rng(0)
    response = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    eligible = np.array([0, 1, 2, 3, 4])  # excludes index 5
    means = _draw_shuffled_means(response, eligible, n_events=3, iterations=200, rng=rng)
    assert means.shape == (200,)
    # All values picked from the 1.0 cohort → mean must be exactly 1.0
    np.testing.assert_allclose(means, 1.0)


def test_shuffled_means_empty_when_no_events():
    rng = np.random.default_rng(0)
    response = np.array([1.0, 2.0, 3.0])
    means = _draw_shuffled_means(response, np.array([0, 1, 2]), 0, 100, rng)
    assert means.size == 0


# -------------------------------------------------------------------
# Month distribution
# -------------------------------------------------------------------


def test_month_distribution_counts_distinct_months():
    times = pd.Series(
        pd.to_datetime(
            ["2025-06-15", "2025-07-01", "2025-08-10", "2025-08-20"], utc=True
        )
    )
    active, max_share = _month_distribution(times)
    assert active == 3
    assert max_share == pytest.approx(0.5)  # August has 2/4


def test_month_distribution_empty():
    times = pd.Series([], dtype="datetime64[ns, UTC]")
    active, max_share = _month_distribution(times)
    assert active == 0
    assert max_share == 0.0


# -------------------------------------------------------------------
# Edge measurement
# -------------------------------------------------------------------


def _build_propagation_pair() -> tuple[pd.DataFrame, pd.DataFrame, PropagationConfig]:
    """Source has CVD breakdowns at known bars; target's VWAP falls 30 bps
    AT BAR b+2 (one bar after the event-bar's close = the start of the
    measurement window with skip=1). This isolates real propagation from
    same-bar drift."""
    n = 600
    rng = np.random.default_rng(11)
    source_cvd = rng.normal(0.0, 1.0, size=n)
    event_bars = [200, 250, 300, 350, 400, 450]
    for b in event_bars:
        source_cvd[b] = -50.0
    source = _make_cvd_frame("SRC", n_bars=n, cvd_pattern=source_cvd)

    target_vwap = np.full(n, 100.0)
    for b in event_bars:
        target_vwap[b + 2 :] *= 0.997  # 30 bps drop persists from bar b+2 onward
    target = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=target_vwap)

    cfg = PropagationConfig(rolling_window_bars=144, shuffled_iterations=200)
    return source, target, cfg


def test_measure_edge_detects_planted_propagation():
    source, target, cfg = _build_propagation_pair()
    events = detect_events(source, cfg)
    rng = np.random.default_rng(0)
    row = measure_edge(
        source_df=source,
        target_df=target,
        event_mask=events["E1"],
        event_name="E1",
        lag_minutes=5,
        cfg=cfg,
        rng=rng,
    )
    assert row["n_events"] == 6
    assert row["short_return_mean"] == pytest.approx(0.003, rel=0.05)
    # Each drop is permanent for this synthetic ⇒ adverse upsample at lag=5 = 0
    assert row["adverse_upsample"] == 0.0


def test_measure_edge_empty_when_no_events():
    source, target, cfg = _build_propagation_pair()
    no_events = pd.Series(False, index=source.index)
    rng = np.random.default_rng(0)
    row = measure_edge(
        source_df=source,
        target_df=target,
        event_mask=no_events,
        event_name="E1",
        lag_minutes=5,
        cfg=cfg,
        rng=rng,
    )
    assert row["n_events"] == 0
    assert np.isnan(row["short_return_mean"])


# -------------------------------------------------------------------
# Map driver end-to-end
# -------------------------------------------------------------------


def test_intersect_symbol_windows_picks_common_overlap():
    a = _make_cvd_frame("A", start="2025-06-01", n_bars=400)
    b = _make_cvd_frame("B", start="2025-06-02", n_bars=400)
    start, end = intersect_symbol_windows({"A": a, "B": b})
    assert start == b["bar_open_time"].min()
    assert end == a["bar_open_time"].max()


def test_build_propagation_map_runs_on_synthetic_universe(tmp_path):
    # Two-symbol universe; CVD spikes on A propagate to B at lag 5m.
    n = 600
    rng = np.random.default_rng(13)
    src_cvd = rng.normal(0.0, 1.0, size=n)  # baseline noise so rolling std > 0
    event_bars = list(range(200, 500, 50))
    for b in event_bars:
        src_cvd[b] = -50.0
    a = _make_cvd_frame("AAAUSDT", n_bars=n, cvd_pattern=src_cvd)
    b_vwap = np.full(n, 100.0)
    for eb in event_bars:
        b_vwap[eb + 2 :] *= 0.998  # 20 bps drop at bar eb+2 (first held bar with skip=1)
    b = _make_cvd_frame("BBBUSDT", n_bars=n, vwap_pattern=b_vwap)

    # Write synthetic monthly parquets into tmp_path/<sym>/5min/
    root = tmp_path / "cvd"
    for sym, frame in [("AAAUSDT", a), ("BBBUSDT", b)]:
        sym_root = root / sym / "5min"
        sym_root.mkdir(parents=True, exist_ok=True)
        # Bucket by month so the loader gets exercised.
        frame_with_month = frame.assign(month=frame["bar_open_time"].dt.strftime("%Y-%m"))
        for month, chunk in frame_with_month.groupby("month"):
            chunk = chunk.drop(columns=["month"])
            chunk.to_parquet(sym_root / f"{month}.parquet", index=False)

    cfg = PropagationConfig(
        rolling_window_bars=144,
        shuffled_iterations=100,
        min_events=3,
    )
    edge_map = build_propagation_map(("AAAUSDT", "BBBUSDT"), root, cfg)
    expected_rows = 2 * 2 * len(cfg.lag_windows_min) * len(EVENT_CODES)
    assert len(edge_map) == expected_rows
    # The A→B at lag 5m / E1 row should pick up the planted propagation
    a_to_b_e1_l5 = edge_map[
        (edge_map["source"] == "AAAUSDT")
        & (edge_map["target"] == "BBBUSDT")
        & (edge_map["event_type"] == "E1")
        & (edge_map["lag_minutes"] == 5)
    ]
    assert len(a_to_b_e1_l5) == 1
    assert a_to_b_e1_l5["short_return_mean"].iloc[0] > 0.0015  # ≥ 15 bps


def test_filter_surviving_edges_rejects_one_month_concentration():
    df = pd.DataFrame(
        [
            # passes: enough events, 3 months, ci low > 0
            dict(
                source="A",
                target="B",
                event_type="E1",
                lag_minutes=5,
                n_events=40,
                month_distribution_active_months=3,
                month_distribution_max_share=0.4,
                bootstrap_ci_low=0.01,
                adverse_upsample=0.2,
                edge_strength=2.0,
            ),
            # fails: only one month with ≥1 event
            dict(
                source="A",
                target="C",
                event_type="E1",
                lag_minutes=5,
                n_events=40,
                month_distribution_active_months=1,
                month_distribution_max_share=1.0,
                bootstrap_ci_low=0.01,
                adverse_upsample=0.2,
                edge_strength=3.0,
            ),
            # fails: ci_low ≤ 0
            dict(
                source="A",
                target="D",
                event_type="E1",
                lag_minutes=5,
                n_events=40,
                month_distribution_active_months=3,
                month_distribution_max_share=0.3,
                bootstrap_ci_low=-0.001,
                adverse_upsample=0.2,
                edge_strength=2.0,
            ),
        ]
    )
    survivors = filter_surviving_edges(df, PropagationConfig())
    assert list(survivors["target"]) == ["B"]


def test_multi_venue_path_classification(tmp_path):
    """Verify cross-venue / cross-symbol / self / cross-venue-cross-symbol
    paths are tagged correctly when running a four-spec map."""
    from pressure_graph.reports.sell_pressure_propagation import (
        VenueSymbolSpec,
        _classify_multi_venue_path,
        build_propagation_map_multi_venue,
    )

    a = VenueSymbolSpec(label="ven_a:BTC", venue="ven_a", cvd_root=tmp_path / "ven_a", symbol="BTC")
    b = VenueSymbolSpec(label="ven_b:BTC", venue="ven_b", cvd_root=tmp_path / "ven_b", symbol="BTC")
    c = VenueSymbolSpec(label="ven_a:ETH", venue="ven_a", cvd_root=tmp_path / "ven_a", symbol="ETH")
    assert _classify_multi_venue_path(a, a) == "self"
    assert _classify_multi_venue_path(a, b) == "cross_venue"
    assert _classify_multi_venue_path(a, c) == "cross_symbol"
    assert _classify_multi_venue_path(b, c) == "cross_venue_cross_symbol"


def test_build_propagation_map_multi_venue_runs_on_synthetic(tmp_path):
    """End-to-end multi-venue smoke. Two venues × one shared symbol.
    Venue A has CVD spikes that propagate to venue B's price after one bar."""
    from pressure_graph.reports.sell_pressure_propagation import (
        VenueSymbolSpec,
        build_propagation_map_multi_venue,
    )

    n = 600
    rng = np.random.default_rng(17)
    src_cvd = rng.normal(0.0, 1.0, size=n)
    event_bars = list(range(200, 500, 50))
    for b in event_bars:
        src_cvd[b] = -50.0
    a_frame = _make_cvd_frame("BTC", n_bars=n, cvd_pattern=src_cvd)

    b_vwap = np.full(n, 100.0)
    for eb in event_bars:
        b_vwap[eb + 2 :] *= 0.998
    b_frame = _make_cvd_frame("BTC", n_bars=n, vwap_pattern=b_vwap)

    for venue, frame in [("ven_a", a_frame), ("ven_b", b_frame)]:
        out_dir = tmp_path / venue / "BTC" / "5min"
        out_dir.mkdir(parents=True, exist_ok=True)
        f = frame.assign(month=frame["bar_open_time"].dt.strftime("%Y-%m"))
        for month, chunk in f.groupby("month"):
            chunk.drop(columns=["month"]).to_parquet(out_dir / f"{month}.parquet", index=False)

    specs = (
        VenueSymbolSpec("ven_a:BTC", "ven_a", tmp_path / "ven_a", "BTC"),
        VenueSymbolSpec("ven_b:BTC", "ven_b", tmp_path / "ven_b", "BTC"),
    )
    cfg = PropagationConfig(rolling_window_bars=144, shuffled_iterations=100, min_events=3)
    edge_map = build_propagation_map_multi_venue(specs, cfg)
    expected_rows = 2 * 2 * len(cfg.lag_windows_min) * len(EVENT_CODES)
    assert len(edge_map) == expected_rows
    cross_row = edge_map[
        (edge_map["source"] == "ven_a:BTC")
        & (edge_map["target"] == "ven_b:BTC")
        & (edge_map["event_type"] == "E1")
        & (edge_map["lag_minutes"] == 5)
    ]
    assert len(cross_row) == 1
    assert cross_row["path"].iloc[0] == "cross_venue"
    assert cross_row["short_return_mean"].iloc[0] > 0.0015


def test_filter_returns_empty_dataframe_when_nothing_survives():
    df = pd.DataFrame(
        [
            dict(
                source="A",
                target="B",
                event_type="E1",
                lag_minutes=5,
                n_events=5,  # below n_events threshold
                month_distribution_active_months=3,
                month_distribution_max_share=0.3,
                bootstrap_ci_low=0.01,
                adverse_upsample=0.2,
                edge_strength=2.0,
            ),
        ]
    )
    survivors = filter_surviving_edges(df, PropagationConfig())
    assert survivors.empty
