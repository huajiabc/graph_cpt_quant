from pressure_graph.reports.v124_delta_neutral_spot_perp_carry import (
    write_v124_delta_neutral_spot_perp_carry,
)


if __name__ == "__main__":
    for name, path in write_v124_delta_neutral_spot_perp_carry().items():
        print(f"{name}: {path}")
