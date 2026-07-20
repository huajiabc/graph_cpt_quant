import numpy as np

from pressure_graph.reports.v150_fss3_independent_audit import (
    V150AuditConfig,
    _alternate_block_bootstrap,
    _array_capped_step,
    _array_neutralize,
)


def test_v150_alternate_bootstrap_constant_series() -> None:
    cfg = V150AuditConfig(bootstrap_iterations=50)
    low, high = _alternate_block_bootstrap(np.full(12, 0.01), cfg)
    assert np.isclose(low, 0.01)
    assert np.isclose(high, 0.01)


def test_v150_array_cap_preserves_beta_and_gross() -> None:
    beta = np.asarray([1.2, 0.6])
    previous_alt, previous_btc = _array_neutralize(
        np.asarray([0.5, -0.5]), beta
    )
    target_alt, target_btc = _array_neutralize(
        np.asarray([-0.5, 0.5]), beta
    )
    cfg = V150AuditConfig(transition_turnover_cap=0.30)
    alt, btc, turnover = _array_capped_step(
        previous_alt,
        previous_btc,
        target_alt,
        target_btc,
        beta,
        np.asarray([True, True]),
        cfg,
    )
    assert turnover <= 0.30 + 1e-12
    assert np.isclose(np.abs(alt).sum() + abs(btc), 1.0, atol=1e-12)
    assert np.isclose(np.dot(alt, beta) + btc, 0.0, atol=1e-12)
