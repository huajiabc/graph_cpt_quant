from pressure_graph.reports.v196_graph_premium_relative_value_weekly import (
    write_v196_graph_premium_relative_value_weekly,
)


if __name__ == "__main__":
    for name, path in write_v196_graph_premium_relative_value_weekly().items():
        print(f"{name}: {path}")
