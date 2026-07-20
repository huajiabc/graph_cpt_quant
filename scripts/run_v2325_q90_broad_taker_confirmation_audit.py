from pressure_graph.reports.v2325_q90_broad_taker_confirmation_audit import (
    write_v2325_q90_broad_taker_confirmation_audit,
)


if __name__ == "__main__":
    for name, path in write_v2325_q90_broad_taker_confirmation_audit().items():
        print(f"{name}: {path}")
