"""Tests for the v9 retest entry predicate ported to graph_cpt_quant.

P1 contract: ``passes_v9_retest(low, close, level, ema20, retest_buffer)`` is
the **predicate-only** port of the v9_sprint inline retest expression at
``src/short_cpt_quant/v9_sprint.py:1989-1994``. It deliberately does not move
v9's setup_* / retest_* column production into graph — just the 4-scalar
gate. The v07a2 paper-live ledger then offers an additional
``entry_policy: v9_retest`` so CIC1/CIC2 candidates can AB it against the
existing ``reclaim`` policy.

Run:
    cd graph_cpt_quant && python -m pytest tests/test_v9_retest_predicate.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# -------------------------- Pure predicate --------------------------


class TestPassesV9RetestPredicate:
    """Lock equivalence with the v9_sprint inline expression."""

    def test_parity_with_v9_inline_expression(self):
        from pressure_graph.entry.retest import passes_v9_retest

        rng = np.random.default_rng(20260611)
        for _ in range(200):
            low = float(rng.uniform(50, 150))
            close = float(rng.uniform(50, 150))
            level = float(rng.uniform(50, 150))
            ema20 = float(rng.uniform(50, 150))
            buf = float(rng.uniform(0.001, 0.02))
            # Authoritative reference from v9_sprint.py:1989-1994 — copied
            # inline so the parity check survives even if v9 is unreachable.
            expected = (
                (low <= level * (1.0 + buf))
                and (close > level)
                and (close > ema20)
            )
            assert passes_v9_retest(low, close, level, ema20, buf) is expected

    def test_blocks_when_close_below_ema20(self):
        from pressure_graph.entry.retest import passes_v9_retest

        # close > level but close < ema20 -> blocked.
        assert passes_v9_retest(low=99.0, close=100.5, level=100.0, ema20=101.0) is False

    def test_blocks_when_close_at_or_below_level(self):
        from pressure_graph.entry.retest import passes_v9_retest

        # The strict > level requirement: close == level is NOT a reclaim.
        assert passes_v9_retest(low=99.0, close=100.0, level=100.0, ema20=99.0) is False

    def test_requires_touch_into_retest_band(self):
        from pressure_graph.entry.retest import passes_v9_retest

        # buffer 0.003 -> trigger 100.3. low=100.4 means we never dipped
        # into the band -> not a retest, would degenerate to next_open.
        assert (
            passes_v9_retest(low=100.4, close=101.0, level=100.0, ema20=99.5, retest_buffer=0.003)
            is False
        )

    def test_passes_full_setup(self):
        from pressure_graph.entry.retest import passes_v9_retest

        # low dips to retest band, close back above level AND above ema20.
        assert (
            passes_v9_retest(low=100.2, close=100.6, level=100.0, ema20=99.5, retest_buffer=0.003)
            is True
        )

    def test_default_retest_buffer(self):
        """Default buffer must match v9 retest_buffer (0.003 ~30bp)."""
        from pressure_graph.entry import retest as retest_mod

        # Pulled into a constant so the YAML default and the predicate default agree.
        assert retest_mod.DEFAULT_RETEST_BUFFER == 0.003


# -------------------------- Paper-live dispatch --------------------------


def _signal_then_retest_bars() -> pd.DataFrame:
    """3 bars: bar0 = signal, bar1 = dip touching level, bar2 = reclaim close > level.

    Entry happens on bar3 (next-open after the reclaim bar2).
    """
    base = pd.Timestamp("2026-06-01T00:00:00Z")
    bars = [
        # open, high, low, close
        (99.0, 100.5, 98.5, 100.0),  # bar0 — signal_close = 100
        (100.0, 100.5, 99.7, 100.2),  # bar1 — dip to 99.7 (within 0.30% buffer of 100)
        (100.2, 101.0, 100.1, 100.8),  # bar2 — close=100.8 > signal_close=100 AND > ema20
        (100.8, 101.5, 100.4, 101.2),  # bar3 — entry next_open = 100.8
        (101.2, 102.0, 101.0, 101.8),  # bar4
        (101.8, 105.5, 101.5, 105.0),  # bar5 — TP hits
    ]
    rows = []
    for idx, (open_, high, low, close) in enumerate(bars):
        bar_open = base + pd.Timedelta(minutes=15 * idx)
        rows.append(
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "bar_open_time": bar_open,
                "bar_close_time": bar_open + pd.Timedelta(minutes=15),
                "feature_time": bar_open + pd.Timedelta(minutes=15),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "ema20": 99.5,  # close > ema20 throughout
            }
        )
    return pd.DataFrame(rows)


class TestPaperLiveDispatchV9Retest:
    def test_v07a2_dispatch_returns_filled_entry(self):
        """_entry_for_signal must accept entry_policy='v9_retest' and fill on retest bar."""
        from pressure_graph.config.v07a2 import V07A2Candidate
        from pressure_graph.paper_live.v07a2 import _entry_for_signal

        bars = _signal_then_retest_bars()
        spec = V07A2Candidate(
            candidate="V9_RETEST_PROBE",
            role="shadow_v9_retest",
            universe_top_n=50,
            market_gate="c2_bucket_beta_extreme_overextended",
            event_col="bullish_volume_shock_event",
            entry_policy="v9_retest",
            pullback_pct=0.003,
            valid_bars=8,
            execution_rule="vol_regime_fast",
            rationale="P1 AB shadow",
        )
        result = _entry_for_signal(bars, signal_idx=0, spec=spec)
        assert result["status"] == "filled", result
        # v9_retest is a single-bar predicate. bar1 (low=99.7, close=100.2,
        # ema20=99.5) satisfies all three conditions on its own, so the
        # entry is the next bar's open (bar2, open=100.2).
        assert result["entry_idx"] == 2, result
        assert abs(float(result["entry_price"]) - 100.2) < 1e-9
        assert result["entry_reason"] == "v9_retest_next_open"

    def test_v07a2_dispatch_blocks_when_close_below_ema20(self):
        """If ema20 is above close on every retest bar -> no fill."""
        from pressure_graph.config.v07a2 import V07A2Candidate
        from pressure_graph.paper_live.v07a2 import _entry_for_signal

        bars = _signal_then_retest_bars().copy()
        bars["ema20"] = 200.0  # all close < ema20
        spec = V07A2Candidate(
            candidate="V9_RETEST_PROBE",
            role="shadow_v9_retest",
            universe_top_n=50,
            market_gate="c2_bucket_beta_extreme_overextended",
            event_col="bullish_volume_shock_event",
            entry_policy="v9_retest",
            pullback_pct=0.003,
            valid_bars=8,
            execution_rule="vol_regime_fast",
            rationale="P1 AB shadow",
        )
        result = _entry_for_signal(bars, signal_idx=0, spec=spec)
        assert result["status"] != "filled", result
