"""Tests for v10 S2 (long-failure -> short) paper-live shadow.

P0b additions to v10_short_mirror + v07d2:
- ``_funding_block_short(row)``: blocks short entries when funding is too
  negative (shorts crowded, paying longs).
- ``_vol_regime_rule_short(row)``: asymmetric TP/SL profile for shorts.
- ``simulate_short_candidate(..., rule_factory=..., funding_blocker=...)``:
  extended kwargs preserving backward compatibility.
- ``write_v07d2_s2_paper_live(prepared, signal_days, report_root)``:
  paper-live shadow wrapper writing candidate summary + current_status.md.

Run:
    cd graph_cpt_quant && python -m pytest tests/test_v10_s2_paper_live.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v10_short_mirror import (
    CANDIDATES,
    ShortCandidate,
    simulate_short_candidate,
)


def _s2_setup_then_break_low() -> pd.DataFrame:
    """Sequence: bar0 = bullish multi setup, bar2 breaks bar0.low -> S2 short.

    Designed for ``S2_FAIL_CIC2_BREAK_LOW`` candidate path:
    - gate_col = ``c2_beta_continuation`` (caller fills in True)
    - event_col = ``bullish_volume_shock_event`` (True only at bar0)
    - entry_kind = ``break_signal_low``
    The fixture price path drives a TP short exit after entry.
    """
    base = pd.Timestamp("2026-06-01T00:00:00Z")
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # bar0 — signal bar, signal_low = 99.0
        (100.0, 100.5, 99.5, 100.0),  # bar1 — no break
        (100.0, 100.5, 98.0, 98.5),   # bar2 — closes 98.5 <= signal_low 99 -> short triggers
        (98.0, 98.5, 96.0, 96.5),     # bar3 — entry next_open = 98.0
        (96.0, 96.5, 94.0, 94.5),     # bar4 — drift down
        (94.0, 94.5, 92.0, 92.5),     # bar5 — continues down (TP=95.55 at -2.5%)
        (92.0, 92.5, 90.0, 90.5),     # bar6 — far below TP
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
                "dynamic_all_rank": 1,
                "symbol_volatility_percentile": 20.0,
                "btc_market_state": "BTC_down",
                "funding_rate_settled": 0.0,
                "c2_beta_continuation": True,
                "bullish_volume_shock_event": True if idx == 0 else False,
            }
        )
    return pd.DataFrame(rows)


# -------------------------- Funding blocker --------------------------


class TestFundingBlockShort:
    def test_blocks_when_funding_too_negative(self):
        from pressure_graph.reports.v10_short_mirror import _funding_block_short

        assert _funding_block_short(pd.Series({"funding_rate_settled": -0.001})) is True

    def test_passes_when_funding_neutral(self):
        from pressure_graph.reports.v10_short_mirror import _funding_block_short

        assert _funding_block_short(pd.Series({"funding_rate_settled": 0.0})) is False

    def test_passes_when_funding_positive(self):
        """Positive funding = longs paying shorts = cheap short entry."""
        from pressure_graph.reports.v10_short_mirror import _funding_block_short

        assert _funding_block_short(pd.Series({"funding_rate_settled": 0.0005})) is False

    def test_passes_when_funding_missing(self):
        """Missing funding data must not silently block — fail-open here is right."""
        from pressure_graph.reports.v10_short_mirror import _funding_block_short

        assert _funding_block_short(pd.Series({})) is False


# -------------------------- Asymmetric SL/TP --------------------------


class TestAsymmetricShortRule:
    def test_low_vol_profile(self):
        from pressure_graph.reports.v10_short_mirror import _vol_regime_rule_short

        row = pd.Series({"symbol_volatility_percentile": 20.0})
        rule = _vol_regime_rule_short(row)
        assert abs(rule.tp - 0.025) < 1e-9
        assert abs(rule.sl - 0.015) < 1e-9

    def test_mid_vol_profile(self):
        from pressure_graph.reports.v10_short_mirror import _vol_regime_rule_short

        row = pd.Series({"symbol_volatility_percentile": 60.0})
        rule = _vol_regime_rule_short(row)
        assert abs(rule.tp - 0.035) < 1e-9
        assert abs(rule.sl - 0.020) < 1e-9

    def test_high_vol_profile(self):
        from pressure_graph.reports.v10_short_mirror import _vol_regime_rule_short

        row = pd.Series({"symbol_volatility_percentile": 90.0})
        rule = _vol_regime_rule_short(row)
        assert abs(rule.tp - 0.045) < 1e-9
        assert abs(rule.sl - 0.025) < 1e-9


# -------------------------- Backward compat --------------------------


class TestSimulateShortCandidateBackwardCompat:
    def test_existing_call_unchanged_no_new_kwargs(self):
        """Calling simulate_short_candidate with the old signature must keep working."""
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        bars = [(100, 101, 99, 100), (100, 102, 98.5, 98.8), (99.8, 100, 96.5, 97)]
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
                    "dynamic_all_rank": 1,
                    "symbol_volatility_percentile": 20.0,
                    "btc_market_state": "BTC_down",
                    "bear_gate": True,
                    "bear_event": True if idx == 0 else False,
                }
            )
        data = pd.DataFrame(rows)
        candidate = ShortCandidate(
            "S",
            "bearish_mirror",
            "bear_gate",
            "bear_event",
            "bounce_reject",
            strict_entry_gate=True,
        )
        trades = simulate_short_candidate(data, candidate)
        assert len(trades) == 1


# -------------------------- S2 with new kwargs --------------------------


class TestS2BreakLowKwargs:
    def test_funding_blocker_kwarg_blocks_s2_entry(self):
        from pressure_graph.reports.v10_short_mirror import _funding_block_short

        data = _s2_setup_then_break_low()
        data["funding_rate_settled"] = -0.001  # too-negative funding for shorts
        candidate = next(c for c in CANDIDATES if c.candidate == "S2_FAIL_CIC2_BREAK_LOW")
        trades = simulate_short_candidate(
            data, candidate, funding_blocker=_funding_block_short
        )
        assert trades.empty, "negative funding must block S2 short entry"

    def test_rule_factory_kwarg_applies_short_asymmetric_rule(self):
        from pressure_graph.reports.v10_short_mirror import _vol_regime_rule_short

        data = _s2_setup_then_break_low()
        candidate = next(c for c in CANDIDATES if c.candidate == "S2_FAIL_CIC2_BREAK_LOW")
        trades = simulate_short_candidate(
            data, candidate, rule_factory=_vol_regime_rule_short
        )
        assert not trades.empty, "fixture should trigger S2 short entry"
        trade = trades.iloc[0]
        assert trade["exit_reason"].startswith("tp"), trade["exit_reason"]


# -------------------------- Paper-live wrapper --------------------------


class TestS2PaperLiveWriter:
    def test_writes_summary_and_status(self, tmp_path: Path):
        from pressure_graph.paper_live.v07d2 import write_v07d2_s2_paper_live

        data = _s2_setup_then_break_low()
        # Wide signal_days window so our 7-bar fixture is in scope.
        outputs = write_v07d2_s2_paper_live(
            data,
            signal_days=365,
            report_root=tmp_path,
        )
        assert outputs["candidate_summary"].exists()
        assert outputs["current_status"].exists()
        status = outputs["current_status"].read_text(encoding="utf-8")
        assert "insufficient" in status, "with one trade, sample_status must be insufficient"
