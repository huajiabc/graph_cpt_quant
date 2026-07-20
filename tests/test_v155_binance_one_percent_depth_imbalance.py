from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    V155Config,
    beta_neutral_v155_weights,
    build_v155_portfolio,
    build_v155_random_controls,
    estimate_v155_betas,
    select_v155_sides,
)


def test_select_v155_sides_retains_inside_band() -> None:
    order = list(FROZEN_SYMBOLS)
    previous_longs = {order[4], order[5], order[6], order[7]}
    previous_shorts = {order[8], order[9], order[10], order[11]}
    longs, shorts = select_v155_sides(order, previous_longs, previous_shorts)
    assert set(longs) == previous_longs
    assert set(shorts) == previous_shorts


def test_beta_neutral_v155_weights_are_exact() -> None:
    longs = list(FROZEN_SYMBOLS[:4])
    shorts = list(FROZEN_SYMBOLS[-4:])
    betas = {symbol: 0.5 + index / 10 for index, symbol in enumerate(FROZEN_SYMBOLS)}
    weights = beta_neutral_v155_weights(longs, shorts, betas)
    residual = sum(weights[symbol] * betas[symbol] for symbol in longs + shorts)
    residual += weights[BTC]
    assert np.isclose(sum(abs(weight) for weight in weights.values()), 1.0)
    assert abs(residual) < 1e-12


def test_estimate_v155_betas_enforces_minimum_samples() -> None:
    rng = np.random.default_rng(7)
    btc = rng.normal(size=600)
    returns = pd.DataFrame(
        {
            BTC: btc,
            FROZEN_SYMBOLS[0]: 1.5 * btc,
            FROZEN_SYMBOLS[1]: np.r_[2.0 * btc[:499], np.full(101, np.nan)],
        }
    )
    betas = estimate_v155_betas(
        returns,
        symbols=FROZEN_SYMBOLS[:2],
        minimum_samples=500,
    )
    assert np.isclose(betas[FROZEN_SYMBOLS[0]], 1.5)
    assert FROZEN_SYMBOLS[1] not in betas


def test_v155_frozen_counts() -> None:
    cfg = V155Config()
    assert len(FROZEN_SYMBOLS) == 16
    assert cfg.long_count == cfg.short_count == 4
    assert cfg.retention_count == 8


def test_fast_random_control_matches_full_portfolio() -> None:
    rows = []
    for day_index, day in enumerate(pd.date_range("2026-01-01", periods=3, tz="UTC")):
        for symbol_index, symbol in enumerate(sorted(FROZEN_SYMBOLS)):
            rows.append(
                {
                    "decision_time": day,
                    "source_day": day - pd.Timedelta(days=1),
                    "period": "validation",
                    "symbol": symbol,
                    "feature_1pct": symbol_index,
                    "stale_feature_1pct": symbol_index,
                    "feature_5pct": symbol_index,
                    "btc_beta": 0.5 + symbol_index / 100,
                    "price_return": (symbol_index - 8) / 1000 + day_index / 10000,
                    "btc_return": day_index / 1000,
                }
            )
    panel = pd.DataFrame(rows)
    cfg = V155Config(random_iterations=1, bootstrap_iterations=10)
    seed = cfg.seed + 1
    slow = build_v155_portfolio(panel, cfg, random_seed=seed)
    fast = build_v155_random_controls(panel, cfg).iloc[0]
    assert np.isclose(fast["mean_primary_net_return"], slow["primary_net_return"].mean())
    assert np.isclose(fast["mean_turnover"], slow["realized_turnover"].mean())
