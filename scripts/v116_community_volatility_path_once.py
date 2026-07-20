from pressure_graph.reports.v116_community_volatility_path import (
    write_v116_community_volatility_path,
)


if __name__ == "__main__":
    for name, path in write_v116_community_volatility_path().items():
        print(f"{name}={path}")
