from pressure_graph.reports.v118_crowding_unwind_transmission import (
    write_v118_crowding_unwind_transmission,
)


if __name__ == "__main__":
    for name, path in write_v118_crowding_unwind_transmission().items():
        print(f"{name}: {path}")
