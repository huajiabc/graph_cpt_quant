from pressure_graph.reports.v175_deribit_skew_receiver_insulator_spread import (
    write_v175_deribit_skew_receiver_insulator_spread,
)


if __name__ == "__main__":
    for name, path in write_v175_deribit_skew_receiver_insulator_spread().items():
        print(f"{name}: {path}")
