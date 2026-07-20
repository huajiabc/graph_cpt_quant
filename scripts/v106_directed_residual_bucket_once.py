from pressure_graph.reports.v106_directed_residual_bucket import (
    write_v106_directed_residual_bucket,
)


if __name__ == "__main__":
    for name, path in write_v106_directed_residual_bucket().items():
        print(f"{name}={path}")
