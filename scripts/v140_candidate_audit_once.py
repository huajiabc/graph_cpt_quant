from __future__ import annotations

from pressure_graph.reports.v140_candidate_audit import run_v140_candidate_audit


if __name__ == "__main__":
    for name, path in run_v140_candidate_audit().items():
        print(f"{name}: {path}")
