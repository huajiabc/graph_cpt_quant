from pressure_graph.reports.v237_two_sigma_oco_temporal_confirmation_audit import (
    write_v237_two_sigma_oco_temporal_confirmation_audit,
)


if __name__ == "__main__":
    for name, path in write_v237_two_sigma_oco_temporal_confirmation_audit().items():
        print(f"{name}: {path}")
