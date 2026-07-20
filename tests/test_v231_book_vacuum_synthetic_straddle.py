import pandas as pd

from pressure_graph.reports.v231_book_vacuum_synthetic_straddle import (
    V231Config,
    price_v231_synthetic_straddles,
)


def test_v231_unchanged_spot_loses_theta_before_hurdle() -> None:
    entry = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    expiry = entry + pd.Timedelta(days=21)
    features = pd.DataFrame(
        [
            {
                "entry_time": entry,
                "entry_month": "2026-01",
                "entry_spot": 100_000.0,
                "causal_atm_iv": 0.50,
                "surface_expiration_time": expiry,
            }
        ]
    )
    times = pd.date_range(entry, periods=9, freq="h")
    prices = pd.DataFrame(
        {"symbol": "BTCUSDT", "feature_time": times, "close": 100_000.0}
    )
    outcome = price_v231_synthetic_straddles(features, prices, V231Config())
    assert outcome.iloc[0]["gross_premium_return_4h"] < 0
    assert outcome.iloc[0]["primary_net_premium_return_4h"] < 0
