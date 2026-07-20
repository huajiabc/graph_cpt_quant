from pressure_graph.reports.v197_rich_premium_short_feature_audit import (
    write_v197_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v197_feature_audit().items():
        print(f"{name}: {path}")
