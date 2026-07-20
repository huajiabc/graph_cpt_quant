from pressure_graph.reports.v182_passive_residual_retracement import (
    write_v182_passive_residual_retracement,
)


if __name__ == "__main__":
    for name, path in write_v182_passive_residual_retracement().items():
        print(f"{name}: {path}")
