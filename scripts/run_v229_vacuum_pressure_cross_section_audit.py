from pressure_graph.reports.v229_vacuum_pressure_cross_section_audit import (
    write_v229_vacuum_pressure_cross_section_audit,
)


if __name__ == "__main__":
    for name, path in write_v229_vacuum_pressure_cross_section_audit().items():
        print(f"{name}: {path}")
