from pressure_graph.reports.v115_cross_community_volatility_front import (
    write_v115_cross_community_volatility_front,
)


if __name__ == "__main__":
    for name, path in write_v115_cross_community_volatility_front().items():
        print(f"{name}={path}")
