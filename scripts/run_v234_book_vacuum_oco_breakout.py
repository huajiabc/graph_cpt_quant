from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    write_v234_book_vacuum_oco_breakout,
)


if __name__ == "__main__":
    for name, path in write_v234_book_vacuum_oco_breakout().items():
        print(f"{name}: {path}")
