from pressure_graph.reports.v2312_positive_q80_vacuum_breakout import (
    write_v2312_positive_q80_vacuum_breakout,
)


if __name__ == "__main__":
    for name, path in write_v2312_positive_q80_vacuum_breakout().items():
        print(f"{name}: {path}")
