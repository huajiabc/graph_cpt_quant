from pressure_graph.reports.v225_alt_book_vacuum_pressure_to_btc import (
    write_v225_alt_book_vacuum_pressure_to_btc,
)


if __name__ == "__main__":
    for name, path in write_v225_alt_book_vacuum_pressure_to_btc().items():
        print(f"{name}: {path}")
