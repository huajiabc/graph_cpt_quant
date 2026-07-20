from pressure_graph.reports.v198_rich_graph_premium_short_weekly_audit import (
    write_v198_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v198_independent_audit().items():
        print(f"{name}: {path}")
