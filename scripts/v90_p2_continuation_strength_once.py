from pressure_graph.reports.v90_p2_continuation_strength import write_v90_p2_continuation_strength


if __name__ == "__main__":
    for name, path in write_v90_p2_continuation_strength().items():
        print(f"{name}: {path}")
