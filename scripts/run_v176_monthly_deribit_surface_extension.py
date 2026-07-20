from pressure_graph.reports.v176_monthly_deribit_surface_extension import (
    write_v176_monthly_deribit_surface_extension,
)


if __name__ == "__main__":
    for name, path in write_v176_monthly_deribit_surface_extension().items():
        print(f"{name}: {path}")
