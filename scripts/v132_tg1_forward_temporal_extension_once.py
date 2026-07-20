from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    write_v132_tg1_forward_temporal_extension,
)


if __name__ == "__main__":
    for name, path in write_v132_tg1_forward_temporal_extension().items():
        print(f"{name}: {path}")
