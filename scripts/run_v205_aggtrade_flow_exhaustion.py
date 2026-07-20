from pressure_graph.reports.v205_aggtrade_flow_exhaustion import write_v205_reveal


if __name__ == "__main__":
    for name, path in write_v205_reveal().items():
        print(f"{name}: {path}")
