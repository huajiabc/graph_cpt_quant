from pressure_graph.reports.v2316_positive_q85_vacuum_breakout_audit import (
    write_v2316_positive_q85_vacuum_breakout_audit,
)


if __name__ == "__main__":
    for name, path in write_v2316_positive_q85_vacuum_breakout_audit().items():
        print(f"{name}: {path}")
