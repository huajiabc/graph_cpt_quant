from pressure_graph.reports.v196_graph_premium_relative_value_weekly_audit import (
    write_v196_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v196_independent_audit().items():
        print(f"{name}: {path}")
