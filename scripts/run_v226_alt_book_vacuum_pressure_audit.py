from pressure_graph.reports.v226_alt_book_vacuum_pressure_audit import (
    write_v226_alt_book_vacuum_pressure_audit,
)


if __name__ == "__main__":
    for name, path in write_v226_alt_book_vacuum_pressure_audit().items():
        print(f"{name}: {path}")
