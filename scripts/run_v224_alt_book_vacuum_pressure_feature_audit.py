from pressure_graph.reports.v224_alt_book_vacuum_pressure_feature_audit import (
    write_v224_alt_book_vacuum_pressure_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v224_alt_book_vacuum_pressure_feature_audit().items():
        print(f"{name}: {path}")
