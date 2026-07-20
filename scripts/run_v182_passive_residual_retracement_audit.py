from pressure_graph.reports.v182_passive_residual_retracement_audit import (
    write_v182_audit,
)


if __name__ == "__main__":
    for name, path in write_v182_audit().items():
        print(f"{name}: {path}")
