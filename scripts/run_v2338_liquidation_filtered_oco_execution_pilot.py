from pressure_graph.reports.v2338_liquidation_filtered_oco_execution_pilot import (
    write_v2338,
)


if __name__ == "__main__":
    for name, path in write_v2338().items():
        print(f"{name}: {path}")
