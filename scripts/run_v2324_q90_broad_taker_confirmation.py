from pressure_graph.reports.v2324_q90_broad_taker_confirmation import (
    write_v2324_q90_broad_taker_confirmation,
)


if __name__ == "__main__":
    for name, path in write_v2324_q90_broad_taker_confirmation().items():
        print(f"{name}: {path}")
