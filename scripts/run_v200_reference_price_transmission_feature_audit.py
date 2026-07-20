from pressure_graph.reports.v200_reference_price_transmission_feature_audit import (
    write_v200_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v200_feature_audit().items():
        print(f"{name}: {path}")
