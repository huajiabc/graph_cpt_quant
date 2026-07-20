from pressure_graph.reports.v127_broader_cross_venue_carry_bucket import (
    write_v127_broader_cross_venue_carry_bucket,
)


if __name__ == "__main__":
    for name, path in write_v127_broader_cross_venue_carry_bucket().items():
        print(f"{name}: {path}")
