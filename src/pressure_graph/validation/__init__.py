"""Project-wide candidate validation harness.

Per 77.docx P0 (vEval Global Gate5 / Concentration Validator), this
package owns the distribution / concentration / robustness gate
checks that every short, long, and hybrid candidate must surface in
its candidate_summary.csv.

Public API: see ``gate_checks``.
"""
from pressure_graph.validation.gate_checks import (
    GateNames,
    GateThresholds,
    CandidateVerdict,
    DistributionMetrics,
    BootstrapMetrics,
    WalkForwardMetrics,
    RandomBaselineMetrics,
    Verdict,
    compute_distribution_metrics,
    compute_bootstrap_ci,
    compute_walk_forward_metrics,
    compute_random_baseline_metrics,
    evaluate_candidate_verdict,
)

__all__ = [
    "GateNames",
    "GateThresholds",
    "CandidateVerdict",
    "DistributionMetrics",
    "BootstrapMetrics",
    "WalkForwardMetrics",
    "RandomBaselineMetrics",
    "Verdict",
    "compute_distribution_metrics",
    "compute_bootstrap_ci",
    "compute_walk_forward_metrics",
    "compute_random_baseline_metrics",
    "evaluate_candidate_verdict",
]
