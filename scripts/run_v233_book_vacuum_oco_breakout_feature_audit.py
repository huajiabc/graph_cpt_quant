from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    write_v233_book_vacuum_oco_breakout_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v233_book_vacuum_oco_breakout_feature_audit().items():
        print(f"{name}: {path}")
