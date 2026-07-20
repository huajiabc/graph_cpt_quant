import pandas as pd

from pressure_graph.reports.v221_sfi_fss3_overlay_feature_audit import (
    V221Config,
    build_v221_overlay_features,
)


def test_overlay_preserves_sign_and_half_notional() -> None:
    entry = pd.Timestamp("2025-09-08", tz="UTC")
    symbols = ["L1", "L2", "S1", "S2"]
    panel = pd.DataFrame(
        {
            "entry_time": [entry] * 4,
            "month_start": [pd.Timestamp("2025-09-01", tz="UTC")] * 4,
            "period": ["development"] * 4,
            "symbol": symbols,
            "score_7d": [-1.0, -0.5, 0.5, 1.0],
            "btc_beta": [1.0] * 4,
        }
    )
    source = entry - pd.Timedelta(hours=12)
    sfi = pd.DataFrame(
        {
            "feature_time": [source] * 4,
            "symbol": symbols,
            "spot_minus_perp_flow_z": [2.0, -1.0, 1.0, -2.0],
            "feature_eligible": [True] * 4,
        }
    )
    cfg = V221Config(minimum_sfi_coverage=4, minimum_side_breadth=2)
    result = build_v221_overlay_features(panel, sfi, cfg)
    assert result.loc[result["side"].eq("long"), "overlay_raw_weight"].sum() == 0.5
    assert result.loc[result["side"].eq("short"), "overlay_raw_weight"].sum() == -0.5
    assert (
        result.set_index("symbol").at["L1", "overlay_raw_weight"]
        > result.set_index("symbol").at["L2", "overlay_raw_weight"]
    )
    assert abs(result.set_index("symbol").at["S2", "overlay_raw_weight"]) > abs(
        result.set_index("symbol").at["S1", "overlay_raw_weight"]
    )
