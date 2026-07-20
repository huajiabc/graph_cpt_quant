from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    write_v2334_audit,
)


if __name__ == "__main__":
    for name, path in write_v2334_audit().items():
        print(f"{name}: {path}")
