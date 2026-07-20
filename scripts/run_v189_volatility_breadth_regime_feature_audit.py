from pressure_graph.reports.v189_volatility_breadth_regime_feature_audit import (
    write_v189_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v189_feature_audit().items():
        print(f"{name}: {path}")
