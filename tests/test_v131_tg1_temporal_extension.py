from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v125_cross_venue_perpetual_carry import V125Config


def test_temporal_extension_entry_is_frozen_to_first_complete_july_week() -> None:
    cfg = V125Config(first_entry=pd.Timestamp("2025-07-07", tz="UTC"))
    assert cfg.first_entry == pd.Timestamp("2025-07-07", tz="UTC")
