from pressure_graph.reports.v171_option_skew_innovation_btc import (
    write_v171_option_skew_innovation_btc,
)


if __name__ == "__main__":
    for name, path in write_v171_option_skew_innovation_btc().items():
        print(f"{name}: {path}")
