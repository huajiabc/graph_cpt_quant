from pressure_graph.reports.v2313_positive_q80_vacuum_breakout_audit import (
    write_v2313_positive_q80_vacuum_breakout_audit,
)


if __name__ == "__main__":
    for name, path in write_v2313_positive_q80_vacuum_breakout_audit().items():
        print(f"{name}: {path}")
