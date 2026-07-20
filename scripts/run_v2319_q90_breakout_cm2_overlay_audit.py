from pressure_graph.reports.v2319_q90_breakout_cm2_overlay_audit import (
    write_v2319_q90_breakout_cm2_overlay_audit,
)


if __name__ == "__main__":
    for name, path in write_v2319_q90_breakout_cm2_overlay_audit().items():
        print(f"{name}: {path}")
