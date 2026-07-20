from pressure_graph.reports.v2311_positive_q80_vacuum_breakout_feature_audit import (
    write_v2311_positive_q80_vacuum_breakout_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v2311_positive_q80_vacuum_breakout_feature_audit().items():
        print(f"{name}: {path}")
