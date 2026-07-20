from pressure_graph.reports.v193_premium_pressure_shape_feature_audit import (
    write_v193_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v193_feature_audit().items():
        print(f"{name}: {path}")
