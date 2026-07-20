from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import FROZEN_SYMBOLS
from pressure_graph.reports.v159_hourly_cross_venue_depth_imbalance import (
    V159Config,
    rolling_v159_betas,
)


def test_rolling_v159_betas_are_causal_and_exact() -> None:
    rng = np.random.default_rng(11)
    btc = rng.normal(size=800)
    frame = pd.DataFrame({BTC: btc})
    for index, symbol in enumerate(FROZEN_SYMBOLS):
        frame[symbol] = (1.0 + index / 10) * btc
    cfg = V159Config(beta_window_hours=720, minimum_beta_samples=500)
    betas = rolling_v159_betas(frame, cfg)
    assert betas.iloc[498].isna().all()
    assert np.isclose(betas.iloc[499][FROZEN_SYMBOLS[0]], 1.0)
    assert np.isclose(betas.iloc[-1][FROZEN_SYMBOLS[-1]], 2.5)
