from pressure_graph.reports.v170_option_skew_directional_btc import (
    write_v170_option_skew_directional_btc,
)


if __name__ == "__main__":
    for name, path in write_v170_option_skew_directional_btc().items():
        print(f"{name}: {path}")
