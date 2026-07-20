from pressure_graph.reports.v219_spot_perp_flow_inventory import write_v219_reveal


if __name__ == "__main__":
    for name, path in write_v219_reveal().items():
        print(f"{name}: {path}")
