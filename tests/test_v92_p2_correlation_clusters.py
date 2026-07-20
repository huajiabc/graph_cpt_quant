from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v92_p2_correlation_clusters import (
    attach_asof_correlation_clusters,
    build_asof_correlation_membership,
)


def _frame(july_multiplier: float) -> pd.DataFrame:
    times = pd.date_range("2026-06-23", "2026-07-02", freq="15min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(92)
    factor_ab = rng.normal(0.0, 0.01, len(times))
    factor_cd = rng.normal(0.0, 0.01, len(times))
    rows: list[dict[str, object]] = []
    for idx, stamp in enumerate(times):
        current_multiplier = july_multiplier if stamp >= pd.Timestamp("2026-07-01", tz="UTC") else 1.0
        for symbol, value in [
            ("AAAUSDT", factor_ab[idx]),
            ("BBBUSDT", factor_ab[idx] + rng.normal(0.0, 0.0002)),
            ("CCCUSDT", factor_cd[idx] * current_multiplier),
            ("DDDUSDT", factor_cd[idx] * current_multiplier + rng.normal(0.0, 0.0002)),
        ]:
            rows.append({"feature_time": stamp, "symbol": symbol, "ret_1h": value})
    return pd.DataFrame(rows)


def test_monthly_correlation_membership_is_asof_and_attaches() -> None:
    first = _frame(1.0)
    altered = _frame(-20.0)
    membership, edges = build_asof_correlation_membership(first)
    altered_membership, _ = build_asof_correlation_membership(altered)
    july = membership[pd.to_datetime(membership["month_start"], utc=True).eq("2026-07-01")]
    altered_july = altered_membership[
        pd.to_datetime(altered_membership["month_start"], utc=True).eq("2026-07-01")
    ]
    cluster = july.set_index("symbol")["correlation_cluster_id"]
    assert cluster["AAAUSDT"] == cluster["BBBUSDT"]
    assert cluster["CCCUSDT"] == cluster["DDDUSDT"]
    assert cluster["AAAUSDT"] != cluster["CCCUSDT"]
    assert july["correlation_cluster_input_covered"].all()
    assert july["history_end_exclusive"].eq(pd.Timestamp("2026-07-01", tz="UTC")).all()
    pd.testing.assert_series_equal(
        cluster.sort_index(),
        altered_july.set_index("symbol")["correlation_cluster_id"].sort_index(),
    )
    assert len(edges[pd.to_datetime(edges["month_start"], utc=True).eq("2026-07-01")]) == 2

    attached = attach_asof_correlation_clusters(first, membership)
    july_rows = attached[pd.to_datetime(attached["feature_time"], utc=True).ge("2026-07-01")]
    assert july_rows["correlation_cluster_input_covered"].all()
    assert july_rows["correlation_cluster_id"].ne("").all()
