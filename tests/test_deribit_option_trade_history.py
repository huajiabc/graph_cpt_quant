from __future__ import annotations

import math

import pandas as pd

from pressure_graph.deribit_option_trade_history import (
    build_daily_option_surface,
    candidate_strikes,
    implied_volatility,
    inverse_option_price,
    monthly_expiries,
    normalize_active_trade_bars,
    option_instrument,
    parse_option_instrument,
    quarterly_expiries,
)


def test_quarterly_expiry_and_instrument_round_trip() -> None:
    expiries = quarterly_expiries(
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"),
    )
    assert [value.isoformat() for value in expiries] == [
        "2021-03-26T08:00:00+00:00",
        "2021-06-25T08:00:00+00:00",
        "2021-09-24T08:00:00+00:00",
        "2021-12-31T08:00:00+00:00",
    ]
    name = option_instrument(expiries[1], 50_000, "c")
    assert name == "BTC-25JUN21-50000-C"
    parsed = parse_option_instrument(name)
    assert parsed["strike"] == 50_000
    assert parsed["option_type"] == "call"
    assert parsed["expiration_time"] == expiries[1]


def test_monthly_expiries_include_non_quarter_months() -> None:
    expiries = monthly_expiries(
        pd.Timestamp("2021-04-01", tz="UTC"),
        pd.Timestamp("2021-06-30 23:59", tz="UTC"),
    )
    assert [value.isoformat() for value in expiries] == [
        "2021-04-30T08:00:00+00:00",
        "2021-05-28T08:00:00+00:00",
        "2021-06-25T08:00:00+00:00",
    ]


def test_inverse_implied_volatility_round_trip() -> None:
    expected = 0.72
    price = inverse_option_price(40_000, 50_000, 30 / 365.25, expected, "call")
    actual = implied_volatility(price, 40_000, 50_000, 30 / 365.25, "call")
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)


def test_candidate_strikes_use_coarse_archival_lattice() -> None:
    assert candidate_strikes(32_000) == [
        20_000.0,
        25_000.0,
        30_000.0,
        35_000.0,
        40_000.0,
        45_000.0,
    ]
    assert candidate_strikes(104_000)[0] == 60_000.0
    assert candidate_strikes(104_000)[-1] == 150_000.0


def test_normalizer_drops_exchange_forward_fills() -> None:
    times = pd.date_range("2021-06-01", periods=2, freq="h", tz="UTC")
    option = pd.DataFrame(
        {
            "bar_open_time": times,
            "open": [0.02, 0.02],
            "high": [0.02, 0.02],
            "low": [0.02, 0.02],
            "close": [0.02, 0.02],
            "volume": [10.0, 0.0],
            "cost": [0.2, 0.0],
        }
    )
    underlying = pd.DataFrame(
        {
            "bar_open_time": times,
            "open": [40_000.0, 40_100.0],
            "close": [40_100.0, 40_200.0],
        }
    )
    normalized = normalize_active_trade_bars(
        option, underlying, "BTC-25JUN21-50000-C"
    )
    assert len(normalized) == 1
    assert normalized.iloc[0]["trade_vwap_btc"] == 0.02
    assert normalized.iloc[0]["volume"] == 10.0


def test_daily_surface_requires_actual_two_sided_wings() -> None:
    date = pd.Timestamp("2025-06-01", tz="UTC")
    expiry = pd.Timestamp("2025-06-27 08:00", tz="UTC")
    rows = []
    specs = [
        ("p80", 80_000, "put", 0.25, 0.75, -0.25, 30.0),
        ("p90", 90_000, "put", 0.35, 0.70, -0.35, 20.0),
        ("c100", 100_000, "call", 0.50, 0.65, 0.50, 40.0),
        ("c110", 110_000, "call", 0.25, 0.68, 0.25, 25.0),
    ]
    for name, strike, kind, abs_delta, iv, delta, volume in specs:
        rows.append(
            {
                "instrument_name": name,
                "bar_open_time": date + pd.Timedelta(hours=12),
                "surface_date": date,
                "expiration_time": expiry,
                "strike": strike,
                "option_type": kind,
                "volume": volume,
                "dte": 26.0,
                "log_moneyness": math.log(strike / 100_000),
                "implied_volatility": iv,
                "abs_delta": abs_delta,
            }
        )
    surface = build_daily_option_surface(pd.DataFrame(rows))
    assert len(surface) == 1
    row = surface.iloc[0]
    assert bool(row["quality_pass"])
    assert math.isclose(row["downside_risk_reversal"], 0.07, abs_tol=1e-12)
    assert row["feature_time"] == pd.Timestamp("2025-06-02", tz="UTC")
