"""Tests for the propagation_risk_overlay consumer.

Uses synthetic CVD frames and synthetic graph_nodes.json payloads so the
unit tests do not require the Phase 3 backfill on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.sell_pressure_propagation import (
    PropagationConfig,
    detect_events,
)
from pressure_graph.reports.propagation_risk_overlay import (
    OverlayInputs,
    PropagationOverlayConfig,
    RiskFlag,
    SEVERITY_ACTIONABLE,
    SEVERITY_ELEVATED,
    SEVERITY_INFORMATIONAL,
    emit_flags_for_node,
    flags_to_dataframe,
    is_long_at,
    load_propagation_nodes,
    node_event_type,
    node_severity,
    run_overlay,
    run_overlay_from_disk,
    summarize_overlay,
    write_overlay_report,
)


def _make_cvd_frame(
    symbol: str,
    start: str = "2025-06-01 00:00:00",
    n_bars: int = 600,
    cvd_pattern: np.ndarray | None = None,
) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    cvd = cvd_pattern if cvd_pattern is not None else np.zeros(n_bars)
    vwap = np.full(n_bars, 100.0)
    volume = np.full(n_bars, 1000.0)
    turnover = vwap * volume
    buy_volume = volume * 0.5
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
            "taker_buy_ratio": np.full(n_bars, 0.5),
            "buy_sell_imbalance": np.full(n_bars, 0.0),
            "cvd_delta_volume": cvd,
            "cvd_delta_turnover": cvd * vwap,
            "large_trade_threshold": 50.0,
            "large_buy_count": np.zeros(n_bars, dtype=int),
            "large_sell_count": np.zeros(n_bars, dtype=int),
            "large_buy_turnover": np.zeros(n_bars),
            "large_sell_turnover": np.zeros(n_bars),
            "coverage_ratio": np.full(n_bars, 1.0),
            "source_quality": "complete",
        }
    )


def _planted_event_cvd(n_bars: int = 600, event_bars: tuple[int, ...] = (200, 300, 400)) -> np.ndarray:
    rng = np.random.default_rng(0)
    cvd = rng.normal(0.0, 1.0, size=n_bars)
    for b in event_bars:
        cvd[b] = -50.0
    return cvd


def _node(
    edge_id: str = "bybit:DOGEUSDT->binance_um:DOGEUSDT@E1",
    statistically_real: bool = False,
    tradeable_after_cost: bool = False,
    lag_minutes: tuple[int, ...] = (5, 15, 30),
) -> dict:
    return {
        "edge_id": edge_id,
        "edge": edge_id.split("@", 1)[0].replace("->", " -> "),
        "edge_type": "downside_sell_pressure",
        "lag_minutes": list(lag_minutes),
        "status_statistically_real": statistically_real,
        "status_tradeable_after_cost": tradeable_after_cost,
        "use_cases": [],
        "notes": "test",
    }


# -------------------------------------------------------------------
# Node parsing + severity mapping
# -------------------------------------------------------------------


def test_node_event_type_extracts_event_class_from_edge_id() -> None:
    assert node_event_type({"edge_id": "bybit:DOGEUSDT->binance_um:DOGEUSDT@E1"}) == "E1"
    assert node_event_type({"edge_id": "bybit:DOGEUSDT->binance_um:DOGEUSDT@E2"}) == "E2"
    # No @ → default E1 (matches docx primary edge)
    assert node_event_type({"edge_id": "no-suffix"}) == "E1"


def test_node_severity_maps_verdict_to_grade_ladder() -> None:
    assert node_severity(_node(statistically_real=False, tradeable_after_cost=False)) == SEVERITY_INFORMATIONAL
    assert node_severity(_node(statistically_real=True, tradeable_after_cost=False)) == SEVERITY_ELEVATED
    assert node_severity(_node(statistically_real=False, tradeable_after_cost=True)) == SEVERITY_ACTIONABLE
    assert node_severity(_node(statistically_real=True, tradeable_after_cost=True)) == SEVERITY_ACTIONABLE


def test_load_propagation_nodes_handles_missing_file(tmp_path: Path) -> None:
    assert load_propagation_nodes(tmp_path / "absent.json") == []


def test_load_propagation_nodes_accepts_list_and_dict(tmp_path: Path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([_node(), _node(edge_id="other->x@E2")]), encoding="utf-8")
    assert len(load_propagation_nodes(list_path)) == 2
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps(_node()), encoding="utf-8")
    assert len(load_propagation_nodes(dict_path)) == 1


# -------------------------------------------------------------------
# Positions probe
# -------------------------------------------------------------------


def test_is_long_at_matches_open_window() -> None:
    pos = pd.DataFrame(
        [
            {
                "symbol": "DOGEUSDT",
                "position_open_time": pd.Timestamp("2025-10-15", tz="UTC"),
                "position_close_time": pd.Timestamp("2025-10-20", tz="UTC"),
            }
        ]
    )
    pos["position_open_time"] = pd.to_datetime(pos["position_open_time"], utc=True)
    pos["position_close_time"] = pd.to_datetime(pos["position_close_time"], utc=True)
    assert is_long_at(pos, "DOGEUSDT", pd.Timestamp("2025-10-17", tz="UTC"))
    assert not is_long_at(pos, "DOGEUSDT", pd.Timestamp("2025-10-21", tz="UTC"))
    assert not is_long_at(pos, "BTCUSDT", pd.Timestamp("2025-10-17", tz="UTC"))


def test_is_long_at_handles_open_ended_position() -> None:
    pos = pd.DataFrame(
        [
            {
                "symbol": "DOGEUSDT",
                "position_open_time": pd.Timestamp("2025-10-15", tz="UTC"),
                "position_close_time": pd.NaT,
            }
        ]
    )
    pos["position_open_time"] = pd.to_datetime(pos["position_open_time"], utc=True)
    pos["position_close_time"] = pd.to_datetime(pos["position_close_time"], utc=True)
    assert is_long_at(pos, "DOGEUSDT", pd.Timestamp("2026-01-01", tz="UTC"))


# -------------------------------------------------------------------
# Flag emission
# -------------------------------------------------------------------


def test_emit_flags_for_node_always_long_one_flag_per_event() -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    node = _node(statistically_real=False)
    overlay_cfg = PropagationOverlayConfig(
        treat_as_always_long=True,
        cfg=PropagationConfig(rolling_window_bars=144),
    )
    flags = emit_flags_for_node(
        node=node, source_cvd=cvd, positions=pd.DataFrame(), overlay_cfg=overlay_cfg
    )
    # 3 planted E1 events → 3 flags (one per event bar)
    assert len(flags) >= 3
    assert all(f.severity == SEVERITY_INFORMATIONAL for f in flags)
    assert all(f.target_long_active for f in flags)
    assert all(f.source_symbol == "DOGEUSDT" for f in flags)


def test_emit_flags_for_node_real_positions_respect_window() -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    bar_at_200 = cvd["bar_open_time"].iloc[200]
    bar_at_300 = cvd["bar_open_time"].iloc[300]
    # Long open during the first event window only
    positions = pd.DataFrame(
        [
            {
                "symbol": "DOGEUSDT",
                "position_open_time": bar_at_200,
                "position_close_time": bar_at_200 + pd.Timedelta(minutes=30),
            }
        ]
    )
    node = _node(statistically_real=False)
    overlay_cfg = PropagationOverlayConfig(
        treat_as_always_long=False,
        flag_lookforward_bars=6,
        cfg=PropagationConfig(rolling_window_bars=144),
    )
    flags = emit_flags_for_node(
        node=node, source_cvd=cvd, positions=positions, overlay_cfg=overlay_cfg
    )
    # All flags should fall in the [200, 200+6] bar window (not 300, not 400)
    times = [f.bar_open_time for f in flags]
    assert all(bar_at_200 <= t <= bar_at_200 + pd.Timedelta(minutes=30) for t in times)


def test_emit_flags_for_node_skips_when_event_type_unknown() -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    node = _node(edge_id="x->y@E_nonexistent")
    overlay_cfg = PropagationOverlayConfig(cfg=PropagationConfig(rolling_window_bars=144))
    flags = emit_flags_for_node(
        node=node, source_cvd=cvd, positions=pd.DataFrame(), overlay_cfg=overlay_cfg
    )
    assert flags == []


def test_flags_to_dataframe_round_trip_columns() -> None:
    flag = RiskFlag(
        edge_id="x->y@E1",
        source_label="bybit:DOGEUSDT",
        target_label="binance_um:DOGEUSDT",
        source_symbol="DOGEUSDT",
        target_symbol="DOGEUSDT",
        event_type="E1",
        bar_open_time=pd.Timestamp("2025-10-15 12:30", tz="UTC"),
        severity=SEVERITY_INFORMATIONAL,
        statistically_real=False,
        tradeable_after_cost=False,
        target_long_active=True,
        lag_minutes=(5, 15, 30),
    )
    df = flags_to_dataframe([flag])
    assert len(df) == 1
    assert df["lag_minutes"].iloc[0] == "5,15,30"
    assert df["severity"].iloc[0] == SEVERITY_INFORMATIONAL


def test_flags_to_dataframe_empty_returns_empty_with_schema() -> None:
    df = flags_to_dataframe([])
    assert df.empty
    assert "edge_id" in df.columns


# -------------------------------------------------------------------
# Overlay driver
# -------------------------------------------------------------------


def test_run_overlay_emits_flags_for_each_matched_node() -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    inputs = OverlayInputs(
        nodes=(_node(), _node(edge_id="bybit:DOGEUSDT->binance_um:DOGEUSDT@E2")),
        source_cvd_by_label={"bybit:DOGEUSDT": cvd},
        positions=None,
        overlay_cfg=PropagationOverlayConfig(
            cfg=PropagationConfig(rolling_window_bars=144)
        ),
    )
    log = run_overlay(inputs)
    # Node 1 (E1) fires on the planted bars; node 2 (E2) requires a volume
    # spike that isn't planted — should be empty. So the log is only E1 flags.
    assert (log["event_type"] == "E1").all()
    assert len(log) >= 3


def test_run_overlay_skips_node_without_source_cvd() -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    inputs = OverlayInputs(
        nodes=(_node(), _node(edge_id="okx:DOGEUSDT->binance_um:DOGEUSDT@E1")),
        source_cvd_by_label={"bybit:DOGEUSDT": cvd},  # no okx CVD provided
        positions=None,
        overlay_cfg=PropagationOverlayConfig(
            cfg=PropagationConfig(rolling_window_bars=144)
        ),
    )
    log = run_overlay(inputs)
    # Only the bybit node contributes; the okx node is silently skipped
    assert (log["source_label"] == "bybit:DOGEUSDT").all()


# -------------------------------------------------------------------
# Aggregation + summary
# -------------------------------------------------------------------


def test_summarize_overlay_aggregates_by_edge_month_severity() -> None:
    log = pd.DataFrame(
        [
            {
                "edge_id": "x->y@E1",
                "source_label": "bybit:DOGEUSDT",
                "target_label": "binance_um:DOGEUSDT",
                "source_symbol": "DOGEUSDT",
                "target_symbol": "DOGEUSDT",
                "event_type": "E1",
                "bar_open_time": pd.Timestamp("2025-10-15", tz="UTC"),
                "severity": SEVERITY_INFORMATIONAL,
                "statistically_real": False,
                "tradeable_after_cost": False,
                "target_long_active": True,
                "lag_minutes": "5,15,30",
            },
            {
                "edge_id": "x->y@E1",
                "source_label": "bybit:DOGEUSDT",
                "target_label": "binance_um:DOGEUSDT",
                "source_symbol": "DOGEUSDT",
                "target_symbol": "DOGEUSDT",
                "event_type": "E1",
                "bar_open_time": pd.Timestamp("2025-10-20", tz="UTC"),
                "severity": SEVERITY_INFORMATIONAL,
                "statistically_real": False,
                "tradeable_after_cost": False,
                "target_long_active": True,
                "lag_minutes": "5,15,30",
            },
            {
                "edge_id": "x->y@E1",
                "source_label": "bybit:DOGEUSDT",
                "target_label": "binance_um:DOGEUSDT",
                "source_symbol": "DOGEUSDT",
                "target_symbol": "DOGEUSDT",
                "event_type": "E1",
                "bar_open_time": pd.Timestamp("2025-11-05", tz="UTC"),
                "severity": SEVERITY_INFORMATIONAL,
                "statistically_real": False,
                "tradeable_after_cost": False,
                "target_long_active": True,
                "lag_minutes": "5,15,30",
            },
        ]
    )
    agg = summarize_overlay(log)
    assert set(agg["month"]) == {"2025-10", "2025-11"}
    assert agg[agg["month"] == "2025-10"]["n_flags"].iloc[0] == 2
    assert agg[agg["month"] == "2025-11"]["n_flags"].iloc[0] == 1


def test_write_overlay_report_produces_all_artefacts(tmp_path: Path) -> None:
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    inputs = OverlayInputs(
        nodes=(_node(),),
        source_cvd_by_label={"bybit:DOGEUSDT": cvd},
        positions=None,
        overlay_cfg=PropagationOverlayConfig(
            cfg=PropagationConfig(rolling_window_bars=144)
        ),
    )
    log = run_overlay(inputs)
    summary = summarize_overlay(log)
    paths = write_overlay_report(log, summary, (_node(),), tmp_path)
    for key in ("propagation_risk_log_csv", "propagation_risk_monthly_csv", "summary_md"):
        assert paths[key].exists(), f"missing {key}"
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "Propagation-graph-node Risk Overlay" in md
    assert "informational" in md  # node is informational severity


# -------------------------------------------------------------------
# End-to-end disk driver
# -------------------------------------------------------------------


def test_run_overlay_from_disk_with_synthetic_cvd(tmp_path: Path) -> None:
    # Write a synthetic graph_nodes.json + a CVD parquet that resembles the
    # real one Phase 3 produces, then run the overlay end-to-end.
    nodes_path = tmp_path / "graph_nodes.json"
    nodes_path.write_text(json.dumps([_node()]), encoding="utf-8")
    cvd = _make_cvd_frame("DOGEUSDT", n_bars=600, cvd_pattern=_planted_event_cvd())
    bybit_root = tmp_path / "bybit"
    out_dir = bybit_root / "DOGEUSDT" / "5min"
    out_dir.mkdir(parents=True, exist_ok=True)
    bucketed = cvd.assign(month=cvd["bar_open_time"].dt.strftime("%Y-%m"))
    for month, chunk in bucketed.groupby("month"):
        chunk.drop(columns=["month"]).to_parquet(out_dir / f"{month}.parquet", index=False)

    paths = run_overlay_from_disk(
        graph_nodes_path=nodes_path,
        cvd_roots={"bybit": bybit_root},
        bar_size_minutes=5,
        report_root=tmp_path / "overlay",
        overlay_cfg=PropagationOverlayConfig(
            cfg=PropagationConfig(rolling_window_bars=144)
        ),
    )
    for key in ("propagation_risk_log_csv", "propagation_risk_monthly_csv", "summary_md"):
        assert paths[key].exists()
    log = pd.read_csv(paths["propagation_risk_log_csv"])
    assert not log.empty
    assert (log["severity"] == SEVERITY_INFORMATIONAL).all()


def test_run_overlay_from_disk_empty_when_node_file_missing(tmp_path: Path) -> None:
    paths = run_overlay_from_disk(
        graph_nodes_path=tmp_path / "missing.json",
        cvd_roots={},
        bar_size_minutes=5,
        report_root=tmp_path / "out",
    )
    log = pd.read_csv(paths["propagation_risk_log_csv"])
    assert log.empty
