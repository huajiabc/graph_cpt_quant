from pressure_graph.reports.v2333_sparse_volatility_tail_independent_audit import (
    write_v2333_sparse_volatility_tail_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v2333_sparse_volatility_tail_independent_audit().items():
        print(f"{name}: {path}")
