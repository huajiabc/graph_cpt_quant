from pressure_graph.reports.v228_vacuum_pressure_cross_section_spread import (
    write_v228_vacuum_pressure_cross_section_spread,
)


if __name__ == "__main__":
    for name, path in write_v228_vacuum_pressure_cross_section_spread().items():
        print(f"{name}: {path}")
