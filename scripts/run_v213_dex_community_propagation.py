from pressure_graph.reports.v213_dex_community_propagation import write_v213_reveal


if __name__ == "__main__":
    for name, path in write_v213_reveal().items():
        print(f"{name}: {path}")
