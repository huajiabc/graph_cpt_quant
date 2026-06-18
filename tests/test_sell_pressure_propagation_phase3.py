"""Pytest unit + integration tests for Phase 3 of the Sell-Pressure
Propagation Map. Uses only synthetic CVD frames so no parquet fetch is needed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.sell_pressure_propagation import (
    PropagationConfig,
    VenueSymbolSpec,
    detect_events,
    precompute_response_grid,
)
from pressure_graph.reports.sell_pressure_propagation_phase3 import (
    DEFAULT_DOGE_LIKE_BASKET,
    DirectionControls,
    ExecutionRepairConfig,
    FocusEdge,
    Phase3Gates,
    Phase3Inputs,
    _event_positions_on_target,
    _month_mask,
    apply_execution_repair,
    available_months_for_spec,
    build_directional_controls,
    build_per_month_edge_table,
    discover_dual_venue_basket,
    discover_venue_symbol_specs,
    intersect_months,
    measure_edge_for_month,
    phase3a_verdict,
    phase3b_verdict,
    register_propagation_graph_node,
    run_execution_repair_diagnostic,
    run_phase3,
    signed_random_null,
    summarize_head_to_head,
    venue_symmetry_null,
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
    ts = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    cvd = cvd_pattern if cvd_pattern is not None else np.zeros(n_bars)
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


def _write_monthly_parquets(frame: pd.DataFrame, root: Path, symbol: str, bar: int = 5) -> Path:
    out_dir = root / symbol / f"{bar}min"
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = frame.assign(month=frame["bar_open_time"].dt.strftime("%Y-%m"))
    for month, chunk in bucket.groupby("month"):
        chunk.drop(columns=["month"]).to_parquet(out_dir / f"{month}.parquet", index=False)
    return out_dir


# -------------------------------------------------------------------
# Data discovery
# -------------------------------------------------------------------


def test_discover_venue_symbol_specs_skips_missing_folders(tmp_path: Path) -> None:
    binance_root = tmp_path / "binance_um"
    _write_monthly_parquets(_make_cvd_frame("AAAUSDT", n_bars=10), binance_root, "AAAUSDT")
    specs = discover_venue_symbol_specs(
        {"binance_um": binance_root}, ("AAAUSDT", "BBBUSDT"), 5
    )
    assert {s.symbol for s in specs} == {"AAAUSDT"}
    assert specs[0].venue == "binance_um"
    assert specs[0].label == "binance_um:AAAUSDT"


def test_discover_dual_venue_basket_lists_missing_symbols(tmp_path: Path) -> None:
    binance_root = tmp_path / "binance_um"
    bybit_root = tmp_path / "bybit"
    _write_monthly_parquets(_make_cvd_frame("AAAUSDT"), binance_root, "AAAUSDT")
    _write_monthly_parquets(_make_cvd_frame("BBBUSDT"), binance_root, "BBBUSDT")
    _write_monthly_parquets(_make_cvd_frame("AAAUSDT"), bybit_root, "AAAUSDT")
    specs, missing = discover_dual_venue_basket(
        binance_root, bybit_root, ("AAAUSDT", "BBBUSDT", "CCCUSDT"), 5
    )
    assert {s.symbol for s in specs} == {"AAAUSDT"}
    assert tuple(missing) == ("BBBUSDT", "CCCUSDT")


def test_available_months_for_spec_parses_yyyy_mm(tmp_path: Path) -> None:
    binance_root = tmp_path / "binance_um"
    frame = _make_cvd_frame("AAAUSDT", start="2025-06-01", n_bars=288 * 90)  # 3 months at 5min
    _write_monthly_parquets(frame, binance_root, "AAAUSDT")
    spec = VenueSymbolSpec("binance_um:AAAUSDT", "binance_um", binance_root, "AAAUSDT")
    months = available_months_for_spec(spec, 5)
    assert months == ("2025-06", "2025-07", "2025-08")


def test_intersect_months_returns_overlap(tmp_path: Path) -> None:
    binance_root = tmp_path / "binance_um"
    bybit_root = tmp_path / "bybit"
    a = _make_cvd_frame("AAAUSDT", start="2025-06-01", n_bars=288 * 90)
    b = _make_cvd_frame("AAAUSDT", start="2025-07-01", n_bars=288 * 90)
    _write_monthly_parquets(a, binance_root, "AAAUSDT")
    _write_monthly_parquets(b, bybit_root, "AAAUSDT")
    specs = (
        VenueSymbolSpec("binance_um:AAAUSDT", "binance_um", binance_root, "AAAUSDT"),
        VenueSymbolSpec("bybit:AAAUSDT", "bybit", bybit_root, "AAAUSDT"),
    )
    assert intersect_months(specs, 5) == ("2025-07", "2025-08")


# -------------------------------------------------------------------
# Phase 3A — per-month edge
# -------------------------------------------------------------------


def test_month_mask_splits_timestamps_by_calendar_month() -> None:
    ts = pd.Series(
        pd.to_datetime(
            ["2025-06-01", "2025-06-30", "2025-07-01", "2025-08-15"], utc=True
        )
    )
    mask = _month_mask(ts, "2025-06")
    assert mask.tolist() == [True, True, False, False]


def test_measure_edge_for_month_restricts_events_and_computes_cost_net() -> None:
    n = 600
    rng = np.random.default_rng(11)
    src_cvd = rng.normal(0.0, 1.0, size=n)
    # plant 6 events: 3 in June, 3 in July
    event_bars = [200, 220, 240, 400, 420, 440]
    for b in event_bars:
        src_cvd[b] = -50.0
    source = _make_cvd_frame("SRC", start="2025-06-01", n_bars=n, cvd_pattern=src_cvd)
    target_vwap = np.full(n, 100.0)
    for b in event_bars:
        target_vwap[b + 2 :] *= 0.997  # 30bp drop
    target = _make_cvd_frame("TGT", start="2025-06-01", n_bars=n, vwap_pattern=target_vwap)
    cfg = PropagationConfig(rolling_window_bars=144, shuffled_iterations=200)
    events = detect_events(source, cfg)
    rng_meas = np.random.default_rng(0)
    row_june = measure_edge_for_month(
        source_df=source,
        target_df=target,
        event_mask=events["E1"],
        event_name="E1",
        month="2025-06",
        lag_minutes=5,
        cfg=cfg,
        rng=rng_meas,
        cost_bps=35.0,
    )
    # event_bars[0..2] all sit in 2025-06 because n_bars=600 5min = 2.08 days,
    # which means ALL events fall in June; but planted bar 400 is also in June.
    # The synthetic frame doesn't span July; the function correctly returns 0 for July.
    row_july = measure_edge_for_month(
        source_df=source,
        target_df=target,
        event_mask=events["E1"],
        event_name="E1",
        month="2025-07",
        lag_minutes=5,
        cfg=cfg,
        rng=rng_meas,
    )
    assert row_june["n_events"] >= 1
    assert row_june["short_return_mean"] > 0
    assert row_june["cost_net"] == pytest.approx(row_june["short_return_mean"] - 0.0035)
    assert row_july["n_events"] == 0
    assert pd.isna(row_july["cost_net"])


def test_build_per_month_edge_table_emits_one_row_per_focus_lag_month(tmp_path: Path) -> None:
    n = 600
    rng = np.random.default_rng(11)
    src_cvd = rng.normal(0.0, 1.0, size=n)
    for b in [200, 250, 300, 350, 400, 450]:
        src_cvd[b] = -50.0
    src_frame = _make_cvd_frame("AAAUSDT", n_bars=n, cvd_pattern=src_cvd)
    target_vwap = np.full(n, 100.0)
    for b in [200, 250, 300, 350, 400, 450]:
        target_vwap[b + 2 :] *= 0.998
    tgt_frame = _make_cvd_frame("AAAUSDT", n_bars=n, vwap_pattern=target_vwap)

    binance_root = tmp_path / "binance_um"
    bybit_root = tmp_path / "bybit"
    _write_monthly_parquets(tgt_frame, binance_root, "AAAUSDT")
    _write_monthly_parquets(src_frame, bybit_root, "AAAUSDT")
    specs = (
        VenueSymbolSpec("bybit:AAAUSDT", "bybit", bybit_root, "AAAUSDT"),
        VenueSymbolSpec("binance_um:AAAUSDT", "binance_um", binance_root, "AAAUSDT"),
    )
    cfg = PropagationConfig(rolling_window_bars=144, shuffled_iterations=100)
    focus = (
        FocusEdge(
            source_label="bybit:AAAUSDT",
            target_label="binance_um:AAAUSDT",
            event_type="E1",
            lag_minutes=(5, 15),
        ),
    )
    table = build_per_month_edge_table(specs, focus, cfg, cost_bps=35.0)
    # 1 focus × 2 lags × 1 month = 2 rows
    assert len(table) == 2
    assert set(table["lag_minutes"]) == {5, 15}
    assert table["source"].iloc[0] == "bybit:AAAUSDT"
    # Synthetic plants a clean 20bp drop at lag-5; cost_net should be the
    # gross minus 35bp.
    lag5_row = table[table["lag_minutes"] == 5].iloc[0]
    assert lag5_row["cost_net"] == pytest.approx(
        lag5_row["short_return_mean"] - 0.0035
    )


def test_phase3a_verdict_classifies_no_data_when_empty() -> None:
    verdict = phase3a_verdict(pd.DataFrame(), Phase3Gates())
    assert verdict["status"] == "no_data"


def test_phase3a_verdict_marks_regime_event_when_only_one_month_positive() -> None:
    rows = [
        {
            "month": m,
            "event_type": "E1",
            "lag_minutes": 5,
            "n_events": 25,
            "short_return_mean": 0.002 if m == "2025-10" else -0.001,
            "adverse_rate": 0.30 if m == "2025-10" else 0.45,
            "bootstrap_ci_low": 0.001 if m == "2025-10" else -0.005,
            "edge_strength": 2.0,
            "cost_net": -0.001,
            "source": "bybit:DOGEUSDT",
            "target": "binance_um:DOGEUSDT",
        }
        for m in ["2025-08", "2025-09", "2025-10", "2025-11"]
    ]
    verdict = phase3a_verdict(pd.DataFrame(rows), Phase3Gates(min_months_positive_ci=3))
    assert verdict["status"] == "regime_event"
    assert verdict["n_months_positive_ci"] == 1


def test_phase3a_verdict_marks_stable_when_gates_pass() -> None:
    rows = [
        {
            "month": m,
            "event_type": "E1",
            "lag_minutes": 5,
            "n_events": 25,
            "short_return_mean": 0.004,
            "adverse_rate": 0.30,
            "bootstrap_ci_low": 0.002,
            "edge_strength": 3.0,
            "cost_net": 0.0005,
            "source": "bybit:DOGEUSDT",
            "target": "binance_um:DOGEUSDT",
        }
        for m in ["2025-08", "2025-09", "2025-10", "2025-11"]
    ]
    verdict = phase3a_verdict(pd.DataFrame(rows), Phase3Gates(min_months_positive_ci=3))
    assert verdict["status"] == "stable"
    assert verdict["passes_months_gate"]
    assert verdict["passes_concentration_gate"]


# -------------------------------------------------------------------
# Phase 3B — basket head-to-head
# -------------------------------------------------------------------


def test_summarize_head_to_head_filters_to_cross_venue_pairs() -> None:
    df = pd.DataFrame(
        [
            {
                "source": "bybit:AAA",
                "target": "binance_um:AAA",
                "source_venue": "bybit",
                "target_venue": "binance_um",
                "source_symbol": "AAA",
                "target_symbol": "AAA",
                "path": "cross_venue",
                "event_type": "E1",
                "lag_minutes": 5,
                "short_return_mean": 0.002,
                "bootstrap_ci_low": 0.001,
            },
            {
                "source": "binance_um:AAA",
                "target": "bybit:AAA",
                "source_venue": "binance_um",
                "target_venue": "bybit",
                "source_symbol": "AAA",
                "target_symbol": "AAA",
                "path": "cross_venue",
                "event_type": "E1",
                "lag_minutes": 5,
                "short_return_mean": -0.001,
                "bootstrap_ci_low": -0.003,
            },
        ]
    )
    head = summarize_head_to_head(df, "bybit", "binance_um")
    assert len(head) == 1
    assert head["source"].iloc[0] == "bybit:AAA"


def test_phase3b_verdict_flags_asymmetric_forward() -> None:
    fwd = pd.DataFrame(
        [
            {
                "source_symbol": "AAA",
                "event_type": "E1",
                "lag_minutes": 5,
                "short_return_mean": 0.0020,
                "bootstrap_ci_low": 0.0010,
            }
        ]
    )
    rev = pd.DataFrame(
        [
            {
                "source_symbol": "AAA",
                "event_type": "E1",
                "lag_minutes": 5,
                "short_return_mean": -0.0015,
                "bootstrap_ci_low": -0.0030,
            }
        ]
    )
    verdict = phase3b_verdict(fwd, rev, cost_bps=35.0)
    per = verdict["per_symbol"]
    assert per[0]["label"] == "asymmetric_forward"
    assert verdict["asymmetric_forward_count"] == 1


# -------------------------------------------------------------------
# Phase 3C — directional nulls
# -------------------------------------------------------------------


def test_signed_random_null_centers_near_zero() -> None:
    rng = np.random.default_rng(0)
    response = np.full(200, 0.003)  # constant positive
    positions = np.arange(50)
    null = signed_random_null(response, positions, iterations=500, rng=rng)
    assert null.size == 500
    # mean of the null distribution should be near 0 by symmetry
    assert abs(float(np.mean(null))) < 5e-4


def test_signed_random_null_empty_when_no_events() -> None:
    rng = np.random.default_rng(0)
    null = signed_random_null(np.array([1.0, 2.0]), np.empty(0, dtype=int), 100, rng)
    assert null.size == 0


def test_venue_symmetry_null_overlaps_when_pool_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    # forward positions all +0.003, reverse positions all -0.003. Pooled
    # mean = 0. A random partition with n_fwd = n_rev should have mean
    # centered at 0.
    fwd_response = np.full(100, 0.003)
    rev_response = np.full(100, -0.003)
    fwd_pos = np.arange(20)
    rev_pos = np.arange(20)
    null = venue_symmetry_null(
        fwd_response, fwd_pos, rev_response, rev_pos, iterations=400, rng=rng
    )
    assert null.size == 400
    assert abs(float(np.mean(null))) < 5e-4


def test_build_directional_controls_passes_all_when_observed_dominates() -> None:
    rng = np.random.default_rng(0)
    # Target response is +30bp at positions [0..39] (where forward events
    # land) and 0 elsewhere; the off-symbol pool sits in the zero region so
    # the random-symbol null mean is ~0 (cleanly beaten by the observed).
    fwd_response = np.zeros(200)
    fwd_response[:40] = 0.003
    rev_response = np.zeros(200)
    rev_response[:40] = -0.003
    fwd_pos = np.arange(40)
    rev_pos = np.arange(40)
    controls = build_directional_controls(
        forward_response=fwd_response,
        forward_event_positions=fwd_pos,
        reverse_response=rev_response,
        reverse_event_positions=rev_pos,
        other_source_event_positions={"bybit:OTHER": np.arange(60, 140)},
        iterations=400,
        rng=rng,
        cost_bps=10.0,
    )
    assert isinstance(controls, DirectionControls)
    assert controls.observed_short_return == pytest.approx(0.003)
    assert controls.beats_signed()
    assert controls.beats_random_symbol()
    assert controls.passes_all()


# -------------------------------------------------------------------
# Execution-repair E0..E3
# -------------------------------------------------------------------


def _propagation_synthetic_pair() -> tuple[pd.DataFrame, pd.DataFrame, PropagationConfig, pd.Series]:
    n = 600
    rng = np.random.default_rng(7)
    src_cvd = rng.normal(0.0, 1.0, size=n)
    event_bars = [200, 250, 300]
    for b in event_bars:
        src_cvd[b] = -50.0
    src = _make_cvd_frame("SRC", n_bars=n, cvd_pattern=src_cvd)
    cfg = PropagationConfig(rolling_window_bars=144, shuffled_iterations=100)
    events = detect_events(src, cfg)
    tgt_vwap = np.full(n, 100.0)
    return src, _make_cvd_frame("TGT", n_bars=n, vwap_pattern=tgt_vwap), cfg, events["E1"]


def test_apply_execution_repair_e0_returns_mask_unchanged() -> None:
    src, tgt, cfg, mask = _propagation_synthetic_pair()
    refined = apply_execution_repair(src, tgt, mask, "E0", cfg, ExecutionRepairConfig())
    pd.testing.assert_series_equal(refined, mask)


def test_apply_execution_repair_e2_keeps_breakdown_events_only() -> None:
    n = 600
    rng_noise = np.random.default_rng(7)
    src_cvd = rng_noise.normal(0.0, 1.0, size=n)
    src_cvd[300] = -50.0
    src = _make_cvd_frame("SRC", n_bars=n, cvd_pattern=src_cvd)
    cfg = PropagationConfig(rolling_window_bars=144)
    mask = detect_events(src, cfg)["E1"]
    assert mask.loc[300]
    # Target stays flat → no breakdown; E2 must drop the event.
    tgt_flat = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=np.full(n, 100.0))
    repair = ExecutionRepairConfig()
    refined_flat = apply_execution_repair(src, tgt_flat, mask, "E2", cfg, repair)
    assert refined_flat.sum() == 0
    # Target drops 50bp at event-bar+1 → E2 keeps it.
    tgt_drop = np.full(n, 100.0)
    tgt_drop[301:] = 99.0
    tgt_dropping = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=tgt_drop)
    refined_drop = apply_execution_repair(src, tgt_dropping, mask, "E2", cfg, repair)
    assert refined_drop.sum() == 1


def test_apply_execution_repair_e1_requires_bounce_then_failure() -> None:
    n = 600
    rng_noise = np.random.default_rng(11)
    src_cvd = rng_noise.normal(0.0, 1.0, size=n)
    src_cvd[300] = -50.0
    src = _make_cvd_frame("SRC", n_bars=n, cvd_pattern=src_cvd)
    cfg = PropagationConfig(rolling_window_bars=144)
    mask = detect_events(src, cfg)["E1"]
    assert mask.loc[300]
    repair = ExecutionRepairConfig(
        bounce_lookahead_bars=3, bounce_floor_bps=5.0, failure_drawdown_bps=3.0
    )
    # Target: from event-bar+skip=301, VWAP bounces to 100.10 at 302 then
    # tail at 303 = 100.04. That's a 10bp bounce + ~6bp failure. E1 keeps.
    tgt_vwap = np.full(n, 100.0)
    tgt_vwap[302] = 100.10
    tgt_vwap[303] = 100.04
    tgt = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=tgt_vwap)
    refined = apply_execution_repair(src, tgt, mask, "E1", cfg, repair)
    assert refined.sum() == 1
    # Target flat (no bounce) → E1 drops.
    tgt_flat = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=np.full(n, 100.0))
    refined_flat = apply_execution_repair(src, tgt_flat, mask, "E1", cfg, repair)
    assert refined_flat.sum() == 0


def test_apply_execution_repair_e3_requires_source_drop_and_target_lag() -> None:
    n = 600
    rng_noise = np.random.default_rng(17)
    src_cvd = rng_noise.normal(0.0, 1.0, size=n)
    src_cvd[300] = -50.0
    src_vwap = np.full(n, 100.0)
    src_vwap[298:] = 99.5  # source has already dropped 50bp before event_bar+1
    src = _make_cvd_frame(
        "SRC", n_bars=n, cvd_pattern=src_cvd, vwap_pattern=src_vwap
    )
    cfg = PropagationConfig(rolling_window_bars=144)
    mask = detect_events(src, cfg)["E1"]
    assert mask.loc[300]
    repair = ExecutionRepairConfig(
        notyet_lookback_bars=3,
        notyet_source_drop_bps=20.0,
        notyet_target_drop_ceil_bps=5.0,
    )
    # Target flat → tgt_drift = 0 ≥ -5bp; E3 keeps.
    tgt_flat = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=np.full(n, 100.0))
    refined_lag = apply_execution_repair(src, tgt_flat, mask, "E3", cfg, repair)
    assert refined_lag.sum() == 1
    # Target already dropped 50bp → tgt_drift = -50bp < -5bp; E3 drops.
    tgt_drop = np.full(n, 100.0)
    tgt_drop[298:] = 99.5
    tgt_dropping = _make_cvd_frame("TGT", n_bars=n, vwap_pattern=tgt_drop)
    refined_no = apply_execution_repair(src, tgt_dropping, mask, "E3", cfg, repair)
    assert refined_no.sum() == 0


def test_run_execution_repair_diagnostic_returns_four_modes() -> None:
    src, tgt, cfg, mask = _propagation_synthetic_pair()
    rng = np.random.default_rng(0)
    out = run_execution_repair_diagnostic(
        source_df=src,
        target_df=tgt,
        source_events=mask,
        event_name="E1",
        lag_minutes=5,
        cfg=cfg,
        repair_cfg=ExecutionRepairConfig(),
        rng=rng,
        cost_bps=35.0,
    )
    assert list(out["mode"]) == ["E0", "E1", "E2", "E3"]
    assert "cost_net" in out.columns


# -------------------------------------------------------------------
# Graph-node registry
# -------------------------------------------------------------------


def test_register_propagation_graph_node_emits_expected_payload() -> None:
    node = register_propagation_graph_node(
        edge_id="bybit:DOGEUSDT->binance_um:DOGEUSDT@E1",
        source_label="bybit:DOGEUSDT",
        target_label="binance_um:DOGEUSDT",
        lags_minutes=(5, 15, 30),
        statistically_real=True,
        tradeable_after_cost=False,
        notes="test",
    )
    assert node["edge"] == "bybit:DOGEUSDT -> binance_um:DOGEUSDT"
    assert node["edge_type"] == "downside_sell_pressure"
    assert node["status_statistically_real"] is True
    assert node["status_tradeable_after_cost"] is False
    assert any("long risk warning" in uc for uc in node["use_cases"])


# -------------------------------------------------------------------
# Top-level integration test
# -------------------------------------------------------------------


def test_run_phase3_writes_every_artefact_on_synthetic_dual_venue(tmp_path: Path) -> None:
    """End-to-end Phase 3 driver on synthetic Bybit:AAA → Binance:AAA edge.
    Verifies every output file is written and that the graph node is registered.
    """
    n = 600
    rng = np.random.default_rng(13)
    src_cvd = rng.normal(0.0, 1.0, size=n)
    event_bars = list(range(200, 500, 50))
    for b in event_bars:
        src_cvd[b] = -50.0
    src_frame = _make_cvd_frame("AAAUSDT", n_bars=n, cvd_pattern=src_cvd)
    tgt_vwap = np.full(n, 100.0)
    for b in event_bars:
        tgt_vwap[b + 2 :] *= 0.998  # 20bp drop
    tgt_frame = _make_cvd_frame("AAAUSDT", n_bars=n, vwap_pattern=tgt_vwap)

    binance_root = tmp_path / "binance_um"
    bybit_root = tmp_path / "bybit"
    _write_monthly_parquets(tgt_frame, binance_root, "AAAUSDT")
    _write_monthly_parquets(src_frame, bybit_root, "AAAUSDT")

    inputs = Phase3Inputs(
        binance_root=binance_root,
        bybit_root=bybit_root,
        basket=("AAAUSDT", "BBBUSDT"),  # BBBUSDT will be in missing
        primary_source_venue="bybit",
        primary_target_venue="binance_um",
        primary_symbol="AAAUSDT",
        bar_size_minutes=5,
        cost_bps=10.0,  # lower so the synthetic 20bp drop can pass cost gate
        cfg=PropagationConfig(rolling_window_bars=144, shuffled_iterations=100),
        null_iterations=100,
    )
    paths = run_phase3(inputs, tmp_path / "phase3")
    for key in (
        "inventory_csv",
        "phase3a_per_month_csv",
        "phase3b_basket_edge_map_csv",
        "phase3b_forward_head_csv",
        "phase3b_reverse_head_csv",
        "phase3b_per_symbol_csv",
        "phase3c_direction_controls_csv",
        "phase3_execution_repair_csv",
        "graph_nodes_json",
        "summary_md",
    ):
        assert paths[key].exists(), f"missing artefact: {key} ({paths[key]})"
    nodes = json.loads(paths["graph_nodes_json"].read_text())
    assert len(nodes) == 1
    assert nodes[0]["edge_type"] == "downside_sell_pressure"
    assert "BBBUSDT" in paths["summary_md"].read_text(encoding="utf-8")  # missing symbol surfaced

    per_month = pd.read_csv(paths["phase3a_per_month_csv"])
    assert not per_month.empty
    assert per_month["lag_minutes"].isin([5, 15, 30, 60, 240]).all()
