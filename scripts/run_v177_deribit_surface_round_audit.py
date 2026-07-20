from pressure_graph.reports.v177_deribit_surface_round_audit import (
    write_v177_deribit_surface_round_audit,
)


if __name__ == "__main__":
    for name, path in write_v177_deribit_surface_round_audit().items():
        print(f"{name}: {path}")
