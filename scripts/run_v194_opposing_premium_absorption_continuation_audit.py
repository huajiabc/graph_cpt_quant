from pressure_graph.reports.v194_opposing_premium_absorption_continuation_audit import (
    write_v194_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v194_independent_audit().items():
        print(f"{name}: {path}")
