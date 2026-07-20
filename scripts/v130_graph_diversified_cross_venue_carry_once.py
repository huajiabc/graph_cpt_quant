from pressure_graph.reports.v130_graph_diversified_cross_venue_carry import (
    write_v130_graph_diversified_cross_venue_carry,
)


if __name__ == "__main__":
    for name, path in write_v130_graph_diversified_cross_venue_carry().items():
        print(f"{name}: {path}")
