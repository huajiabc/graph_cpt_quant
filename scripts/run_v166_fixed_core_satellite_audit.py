from pressure_graph.reports.v166_fixed_core_satellite_audit import (
    write_v166_fixed_core_satellite_audit,
)


if __name__ == "__main__":
    for name, path in write_v166_fixed_core_satellite_audit().items():
        print(f"{name}: {path}")
