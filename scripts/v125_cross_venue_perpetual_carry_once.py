from pressure_graph.reports.v125_cross_venue_perpetual_carry import (
    write_v125_cross_venue_perpetual_carry,
)


if __name__ == "__main__":
    for name, path in write_v125_cross_venue_perpetual_carry().items():
        print(f"{name}: {path}")
