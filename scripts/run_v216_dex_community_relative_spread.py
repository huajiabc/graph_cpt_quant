from pressure_graph.reports.v216_dex_community_relative_spread import write_v216_reveal


if __name__ == "__main__":
    for name, path in write_v216_reveal().items():
        print(f"{name}: {path}")
