from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v21h_state_discovery_synthesis import (
    V21HConfig,
    write_v21h_state_discovery_synthesis,
)


def _touch(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_state_discovery_synthesis_builds_handoff(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    required = {
        "v2_0_graph_motif_search": ["stable_candidates.csv", "top_paths.csv", "candidate_notes.md"],
        "v2_1a_holdout_autopsy": ["holdout_by_cic_type.csv", "candidate_notes.md"],
        "v2_1b_state_cluster_atlas": ["state_cluster_membership.csv", "state_cluster_summary.csv", "candidate_notes.md"],
        "v2_1c_state_transition_graph": ["state_transition_edges.csv", "holdout_transition_autopsy.csv", "candidate_notes.md"],
        "v2_1d_router_preflight": ["router_rule_summary.csv", "candidate_notes.md"],
        "v2_1f_state_drift_audit": ["state_feature_period_drift.csv", "walkforward_novelty_bucket_summary.csv", "candidate_notes.md"],
    }
    for dirname, files in required.items():
        for file in files:
            _touch(reports / dirname / file)
    (reports / "v2_1e_walkforward_router_stability").mkdir(parents=True, exist_ok=True)
    (reports / "v2_1g_meta_router_action_labels").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rule_id": "baseline_B4",
                "validation_delta_vs_baseline_net20": 0.0,
                "holdout_delta_vs_baseline_net20": 0.0,
                "delta_vs_baseline_net20": 0.0,
            },
            {
                "rule_id": "reduce_size_50_prior_bad_clusters",
                "validation_delta_vs_baseline_net20": -0.01,
                "holdout_delta_vs_baseline_net20": 0.001,
                "delta_vs_baseline_net20": -0.002,
            },
        ]
    ).to_csv(reports / "v2_1e_walkforward_router_stability" / "walkforward_router_summary.csv", index=False)
    _touch(reports / "v2_1e_walkforward_router_stability" / "candidate_notes.md")
    pd.DataFrame(
        [
            {"period": "holdout", "feature": "market_impulse_density", "psi_vs_preholdout": 5.0},
            {"period": "holdout", "feature": "burst_count_so_far", "psi_vs_preholdout": 4.0},
        ]
    ).to_csv(reports / "v2_1f_state_drift_audit" / "state_feature_period_drift.csv", index=False)
    pd.DataFrame(
        [
            {
                "trade_key": "t1",
                "pre_entry_action_label": "core_trade",
                "label_source": "realized_counterfactual_not_asof_feature",
            }
        ]
    ).to_csv(reports / "v2_1g_meta_router_action_labels" / "meta_router_event_labels.csv", index=False)
    pd.DataFrame([{"trade_key": "t1", "state_cluster_id": "SDG00"}]).to_csv(
        reports / "v2_1g_meta_router_action_labels" / "meta_router_feature_matrix.csv", index=False
    )
    pd.DataFrame([{"column": "label_source", "role": "metadata"}]).to_csv(
        reports / "v2_1g_meta_router_action_labels" / "router_dataset_schema.csv", index=False
    )
    _touch(reports / "v2_1g_meta_router_action_labels" / "candidate_notes.md")

    outputs = write_v21h_state_discovery_synthesis(
        V21HConfig(report_root=tmp_path / "out", reports_root=reports)
    )
    manifest = pd.read_csv(outputs["stage_manifest"])
    readiness = pd.read_csv(outputs["v22_readiness_checklist"])

    assert manifest["artifact_status"].eq("available").all()
    assert readiness["status"].eq("passed").all()
    assert outputs["candidate_notes"].exists()
