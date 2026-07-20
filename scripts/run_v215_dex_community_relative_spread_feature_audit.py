from pressure_graph.reports.v215_dex_community_relative_spread_feature_audit import (
    write_v215_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v215_feature_audit().items():
        print(f"{name}: {path}")
