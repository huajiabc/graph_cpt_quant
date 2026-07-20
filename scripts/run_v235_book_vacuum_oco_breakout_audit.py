from pressure_graph.reports.v235_book_vacuum_oco_breakout_audit import (
    write_v235_book_vacuum_oco_breakout_audit,
)


if __name__ == "__main__":
    for name, path in write_v235_book_vacuum_oco_breakout_audit().items():
        print(f"{name}: {path}")
