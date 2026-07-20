from pressure_graph.reports.v238_positive_pressure_narrow_breakout_robustness import (
    write_v238_positive_pressure_narrow_breakout_robustness,
)


if __name__ == "__main__":
    for name, path in write_v238_positive_pressure_narrow_breakout_robustness().items():
        print(f"{name}: {path}")
