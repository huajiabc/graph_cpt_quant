from pressure_graph.reports.v105_direct_graph_bucket_portfolio import (
    write_v105_direct_graph_bucket_portfolio,
)


if __name__ == "__main__":
    for name, path in write_v105_direct_graph_bucket_portfolio().items():
        print(f"{name}={path}")
