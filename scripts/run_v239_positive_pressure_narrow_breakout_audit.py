from pressure_graph.reports.v239_positive_pressure_narrow_breakout_audit import (
    write_v239_positive_pressure_narrow_breakout_audit,
)


if __name__ == "__main__":
    for name, path in write_v239_positive_pressure_narrow_breakout_audit().items():
        print(f"{name}: {path}")
