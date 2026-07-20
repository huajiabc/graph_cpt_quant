from pressure_graph.reports.v180_extreme_residual_dispersion_compression_audit import (
    write_v180_audit,
)


if __name__ == "__main__":
    for name, path in write_v180_audit().items():
        print(f"{name}: {path}")
