from pressure_graph.reports.v189_high_breadth_unwind_cascade import (
    write_v189_high_breadth_unwind_cascade,
)


if __name__ == "__main__":
    for name, path in write_v189_high_breadth_unwind_cascade().items():
        print(f"{name}: {path}")
