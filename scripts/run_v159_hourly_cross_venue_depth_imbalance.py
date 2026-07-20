from pressure_graph.reports.v159_hourly_cross_venue_depth_imbalance import (
    write_v159_hourly_cross_venue_depth_imbalance,
)


if __name__ == "__main__":
    for name, path in write_v159_hourly_cross_venue_depth_imbalance().items():
        print(f"{name}: {path}")
