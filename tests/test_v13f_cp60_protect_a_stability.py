from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v13d_cp60_context_protection import (
    _add_context_features,
    _add_high_flags,
    _thresholds_from_exits,
)
from pressure_graph.reports.v13f_cp60_protect_a_stability import (
    _leave_one_burst_out,
    _leave_one_month_out,
    _protected_exit_cap_summary,
)


def _row(idx: int, *, month: str, burst_id: str, symbol: str, net_keep: float) -> dict[str, object]:
    entry = pd.Timestamp(f"{month}-10T00:00:00Z") + pd.Timedelta(minutes=5 * idx)
    return {
        "exchange": "bybit",
        "symbol": symbol,
        "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
        "signal_id": f"sig-{idx}",
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=4),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "entry_price": 100.0,
        "checkpoint_gross_return": -0.001,
        "checkpoint_net_at_cost": -0.005,
        "checkpoint_mfe": 0.01,
        "checkpoint_mae": -0.01,
        "gross_return": net_keep + 0.004,
        "net_return_at_cost": net_keep,
        "volume_impulse_density": 0.4,
        "c2_beta_extension_score": 100.0,
        "cluster_impulse_density": 0.6,
        "volume_z_1h": 3.0,
        "burst_id": burst_id,
        "burst_count_so_far": idx + 1,
        "month": month,
    }


def _sample() -> pd.DataFrame:
    data = pd.DataFrame(
        [
            _row(0, month="2026-01", burst_id="b0", symbol="AUSDT", net_keep=0.020),
            _row(1, month="2026-01", burst_id="b0", symbol="BUSDT", net_keep=0.015),
            _row(2, month="2026-01", burst_id="b0", symbol="CUSDT", net_keep=-0.015),
            _row(3, month="2026-02", burst_id="b1", symbol="DUSDT", net_keep=0.018),
        ]
    )
    data = _add_context_features(data)
    return _add_high_flags(data, _thresholds_from_exits(data))


def test_protect_a_burst_month_stability_tables() -> None:
    sample = _sample()

    burst = _leave_one_burst_out(sample)
    month = _leave_one_month_out(sample)
    cap = _protected_exit_cap_summary(sample)

    assert not burst.empty
    assert {"removed_burst_id", "delta_vs_CP60_all", "delta_vs_S3_CP60_O6"}.issubset(burst.columns)
    assert set(burst["removed_burst_id"]) == {"b0", "b1"}

    assert not month.empty
    assert {"removed_month", "still_above_CP60_all"}.issubset(month.columns)
    assert set(month["removed_month"]) == {"2026-01", "2026-02"}

    assert not cap.empty
    assert {"Protect_A_cap1_per_burst", "Protect_A_cap2_per_burst", "Protect_A_uncapped"}.issubset(
        set(cap["rule"])
    )
    cap1 = cap[cap["rule"].eq("Protect_A_cap1_per_burst")].iloc[0]
    assert cap1["max_protected_exits_per_burst"] <= 1
