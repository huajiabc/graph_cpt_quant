from __future__ import annotations

import numpy as np

from pressure_graph.reports.v166_fixed_core_satellite_audit import (
    V166AuditConfig,
    _alternate_bootstrap,
)


def test_alternate_bootstrap_is_deterministic() -> None:
    values = np.linspace(-0.01, 0.03, 49)
    cfg = V166AuditConfig(bootstrap_iterations=100)
    first = _alternate_bootstrap(values, cfg)
    second = _alternate_bootstrap(values, cfg)
    assert np.allclose(first, second)
    assert first[0] < first[1]
