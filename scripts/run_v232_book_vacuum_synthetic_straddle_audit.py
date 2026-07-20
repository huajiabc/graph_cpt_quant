from pressure_graph.reports.v232_book_vacuum_synthetic_straddle_audit import (
    write_v232_book_vacuum_synthetic_straddle_audit,
)


if __name__ == "__main__":
    for name, path in write_v232_book_vacuum_synthetic_straddle_audit().items():
        print(f"{name}: {path}")
