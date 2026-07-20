from pressure_graph.reports.v2332_sparse_volatility_tail_selector import (
    write_v2332_sparse_volatility_tail_selector,
)


if __name__ == "__main__":
    for name, path in write_v2332_sparse_volatility_tail_selector().items():
        print(f"{name}: {path}")
