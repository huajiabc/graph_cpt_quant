from __future__ import annotations

import numpy as np

from pressure_graph.reports.v140_candidate_audit import (
    V140AuditConfig,
    _alternate_block_bootstrap,
)


def test_alternate_block_bootstrap_constant_series() -> None:
    cfg = V140AuditConfig(bootstrap_iterations=50, bootstrap_block_weeks=4)
    low, high = _alternate_block_bootstrap(np.full(12, 0.01), cfg)
    assert np.isclose(low, 0.01)
    assert np.isclose(high, 0.01)
