from pressure_graph.reports.v122_positioning_absorption_impulse import (
    write_v122_positioning_absorption_impulse,
)


if __name__ == "__main__":
    for name, path in write_v122_positioning_absorption_impulse().items():
        print(f"{name}: {path}")
