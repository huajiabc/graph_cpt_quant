from pressure_graph.reports.v157_pair_shock_fragile_receiver import (
    write_v157_pair_shock_fragile_receiver,
)


if __name__ == "__main__":
    for name, path in write_v157_pair_shock_fragile_receiver().items():
        print(f"{name}: {path}")
