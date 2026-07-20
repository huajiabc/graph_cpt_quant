from pressure_graph.reports.v181_residual_dispersion_long_horizon_execution_audit import (
    write_v181_audit,
)


if __name__ == "__main__":
    for name, path in write_v181_audit().items():
        print(f"{name}: {path}")
