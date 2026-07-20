from pressure_graph.reports.v210_funding_settlement_rebound import write_v210_reveal


if __name__ == "__main__":
    for name, path in write_v210_reveal().items():
        print(f"{name}: {path}")
