from pressure_graph.reports.v2318_q90_breakout_cm2_overlay import (
    write_v2318_q90_breakout_cm2_overlay,
)


if __name__ == "__main__":
    for name, path in write_v2318_q90_breakout_cm2_overlay().items():
        print(f"{name}: {path}")
