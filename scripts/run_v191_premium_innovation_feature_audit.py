from pressure_graph.reports.v191_premium_innovation_feature_audit import (
    write_v191_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v191_feature_audit().items():
        print(f"{name}: {path}")
