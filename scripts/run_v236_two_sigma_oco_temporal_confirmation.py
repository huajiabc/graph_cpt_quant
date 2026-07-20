from pressure_graph.reports.v236_two_sigma_oco_temporal_confirmation import (
    write_v236_two_sigma_oco_temporal_confirmation,
)


if __name__ == "__main__":
    for name, path in write_v236_two_sigma_oco_temporal_confirmation().items():
        print(f"{name}: {path}")
