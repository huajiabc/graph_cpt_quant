from pressure_graph.reports.v131_tg1_temporal_extension import (
    write_v131_tg1_temporal_extension,
)


if __name__ == "__main__":
    for name, path in write_v131_tg1_temporal_extension().items():
        print(f"{name}: {path}")
