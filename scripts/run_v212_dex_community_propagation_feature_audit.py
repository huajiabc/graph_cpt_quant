from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    write_v212_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v212_feature_audit().items():
        print(f"{name}: {path}")
