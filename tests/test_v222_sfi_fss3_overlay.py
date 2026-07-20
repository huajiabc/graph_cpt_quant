from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v222_sfi_fss3_overlay import (
    V222Config,
    baseline_reconstruction_errors,
    build_v222_nulls,
    build_v222_path,
    load_v222_inputs,
    overlay_raw_target,
)


def test_v222_observed_target_preserves_names_signs_and_half_sides() -> None:
    panel, features = load_v222_inputs()
    entry = features["entry_time"].min()
    local = panel[panel["entry_time"].eq(entry)]
    week = features[features["entry_time"].eq(entry)]
    raw = overlay_raw_target(local, week)
    eligible = local.dropna(
        subset=["score_7d", "price_return", "future_funding", "btc_beta"]
    )
    expected = set(eligible.loc[eligible["score_7d"].ne(0), "symbol"])
    assert set(raw) == expected
    assert np.isclose(sum(weight for weight in raw.values() if weight > 0), 0.5)
    assert np.isclose(sum(weight for weight in raw.values() if weight < 0), -0.5)


def test_v222_reversed_target_preserves_side_multiplier_distribution() -> None:
    panel, features = load_v222_inputs()
    entry = features["entry_time"].min()
    local = panel[panel["entry_time"].eq(entry)]
    week = features[features["entry_time"].eq(entry)]
    observed = overlay_raw_target(local, week, mode="observed")
    reversed_target = overlay_raw_target(local, week, mode="reversed")
    for sign in (1, -1):
        left = sorted(abs(value) for value in observed.values() if np.sign(value) == sign)
        right = sorted(
            abs(value) for value in reversed_target.values() if np.sign(value) == sign
        )
        assert np.allclose(left, right)


def test_v222_zero_tilt_reconstructs_saved_fss3() -> None:
    panel, features = load_v222_inputs()
    cfg = V222Config(null_iterations=2, bootstrap_iterations=10)
    baseline = build_v222_path(panel, features, cfg, mode="baseline")
    errors = baseline_reconstruction_errors(baseline, cfg)
    assert errors["maximum_absolute_error"].max() <= 1e-12


def test_v222_random_mode_is_reproducible() -> None:
    panel, features = load_v222_inputs()
    cfg = V222Config(null_iterations=2, bootstrap_iterations=10)
    left = build_v222_path(
        panel, features, cfg, mode="random", rng=np.random.default_rng(7)
    )
    right = build_v222_path(
        panel, features, cfg, mode="random", rng=np.random.default_rng(7)
    )
    pd.testing.assert_series_equal(
        left["primary_net_return"], right["primary_net_return"]
    )
    baseline = build_v222_path(panel, features, cfg, mode="baseline")
    null = build_v222_nulls(
        panel, features, baseline, V222Config(null_iterations=1, seed=6)
    )
    active = left["overlay_active"]
    expected = (
        left.loc[active, "primary_net_return"]
        - baseline.loc[active, "primary_net_return"]
    ).mean()
    assert np.isclose(null.iloc[0]["mean_active_primary_increment"], expected)
