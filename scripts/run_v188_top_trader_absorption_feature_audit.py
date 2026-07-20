from pressure_graph.reports.v188_top_trader_absorption_feature_audit import (
    write_v188_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v188_feature_audit().items():
        print(f"{name}: {path}")
