from pressure_graph.reports.v121_top_trader_community_rotation import (
    write_v121_top_trader_community_rotation,
)


if __name__ == "__main__":
    for name, path in write_v121_top_trader_community_rotation().items():
        print(f"{name}: {path}")
