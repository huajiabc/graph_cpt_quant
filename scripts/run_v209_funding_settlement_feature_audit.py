from pressure_graph.reports.v209_funding_settlement_feature_audit import (
    write_v209_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v209_feature_audit().items():
        print(f"{name}: {path}")
