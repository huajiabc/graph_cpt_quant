from pressure_graph.reports.v123_cross_sectional_funding_carry import (
    write_v123_cross_sectional_funding_carry,
)


if __name__ == "__main__":
    for name, path in write_v123_cross_sectional_funding_carry().items():
        print(f"{name}: {path}")
