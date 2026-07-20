from pressure_graph.reports.v189_high_breadth_unwind_cascade_audit import (
    write_v189_audit,
)


if __name__ == "__main__":
    for name, path in write_v189_audit().items():
        print(f"{name}: {path}")
