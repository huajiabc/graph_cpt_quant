from pressure_graph.reports.v227_vacuum_pressure_cross_section_feature_audit import (
    write_v227_vacuum_pressure_cross_section_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v227_vacuum_pressure_cross_section_feature_audit().items():
        print(f"{name}: {path}")
