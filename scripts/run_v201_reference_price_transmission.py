from pressure_graph.reports.v201_reference_price_transmission import (
    write_v201_reveal,
)


if __name__ == "__main__":
    for name, path in write_v201_reveal().items():
        print(f"{name}: {path}")
