from pressure_graph.reports.v2328_multisource_interaction_ridge_independent_audit import (
    write_v2328_multisource_interaction_ridge_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v2328_multisource_interaction_ridge_independent_audit().items():
        print(f"{name}: {path}")
