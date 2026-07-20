from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    write_v195_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v195_feature_audit().items():
        print(f"{name}: {path}")
