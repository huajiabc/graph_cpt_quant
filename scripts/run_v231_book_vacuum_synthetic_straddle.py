from pressure_graph.reports.v231_book_vacuum_synthetic_straddle import (
    write_v231_book_vacuum_synthetic_straddle,
)


if __name__ == "__main__":
    for name, path in write_v231_book_vacuum_synthetic_straddle().items():
        print(f"{name}: {path}")
