from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    write_v180_extreme_residual_dispersion_compression,
)


if __name__ == "__main__":
    for name, path in write_v180_extreme_residual_dispersion_compression().items():
        print(f"{name}: {path}")
