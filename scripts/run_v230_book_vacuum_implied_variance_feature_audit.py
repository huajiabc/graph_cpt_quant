from pressure_graph.reports.v230_book_vacuum_implied_variance_feature_audit import (
    write_v230_book_vacuum_implied_variance_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v230_book_vacuum_implied_variance_feature_audit().items():
        print(f"{name}: {path}")
