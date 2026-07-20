import pandas as pd

from pressure_graph.reports.v146_v112_exact_temporal_extension import (
    audit_v146_parity,
    build_v146_extended_panel,
)


def test_v146_splice_preserves_canonical_rows() -> None:
    canonical = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ALTUSDT"],
            "feature_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]
            ),
            "ret_1h": [0.1, 0.2],
            "future_ret_4h": [0.3, 0.4],
            "month_start": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]
            ),
        }
    )
    index = pd.date_range("2025-12-31 23:00Z", periods=22, freq="15min")
    close = pd.DataFrame(
        {"BTCUSDT": range(22), "ALTUSDT": range(10, 32)}, index=index
    ).astype(float)
    extended = build_v146_extended_panel(canonical, close)
    original = extended[
        extended["feature_time"].eq(pd.Timestamp("2026-01-01T00:00:00Z"))
    ]
    assert len(original) == 2
    assert original.set_index("symbol").loc["ALTUSDT", "ret_1h"] == 0.2
    assert extended["feature_time"].max() > pd.Timestamp("2026-01-01T00:00:00Z")


def test_v146_parity_detects_return_drift() -> None:
    canonical = pd.DataFrame(
        {
            "candidate": ["A"],
            "feature_time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "horizon_hours": [4],
            "spread_gross": [0.01],
            "spread_net_20bp": [0.008],
            "spread_net_30bp": [0.007],
            "spread_net_50bp": [0.005],
        }
    )
    assert audit_v146_parity(canonical, canonical)["passed"]
    drifted = canonical.copy()
    drifted.loc[0, "spread_gross"] += 1e-5
    assert not audit_v146_parity(drifted, canonical)["passed"]
