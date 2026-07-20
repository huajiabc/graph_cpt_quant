from pressure_graph.reports.v158_pair_shock_candidate_audit import (
    write_v158_pair_shock_candidate_audit,
)


if __name__ == "__main__":
    for name, path in write_v158_pair_shock_candidate_audit().items():
        print(f"{name}: {path}")
