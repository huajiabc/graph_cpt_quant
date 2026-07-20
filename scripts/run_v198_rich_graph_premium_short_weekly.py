from pressure_graph.reports.v198_rich_graph_premium_short_weekly import (
    write_v198_rich_graph_premium_short_weekly,
)


if __name__ == "__main__":
    for name, path in write_v198_rich_graph_premium_short_weekly().items():
        print(f"{name}: {path}")
