from pressure_graph.reports.v117_direct_dvol_carry import write_v117_direct_dvol_carry


if __name__ == "__main__":
    for name, path in write_v117_direct_dvol_carry().items():
        print(f"{name}: {path}")
