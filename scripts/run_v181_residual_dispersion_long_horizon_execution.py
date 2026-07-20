from pressure_graph.reports.v181_residual_dispersion_long_horizon_execution import (
    write_v181_residual_dispersion_long_horizon_execution,
)


if __name__ == "__main__":
    for name, path in write_v181_residual_dispersion_long_horizon_execution().items():
        print(f"{name}: {path}")
