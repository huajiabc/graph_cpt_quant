from pressure_graph.reports.v194_opposing_premium_absorption_continuation import (
    write_v194_opposing_premium_absorption_continuation,
)


if __name__ == "__main__":
    for name, path in write_v194_opposing_premium_absorption_continuation().items():
        print(f"{name}: {path}")
