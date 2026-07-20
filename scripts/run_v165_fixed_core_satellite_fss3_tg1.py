from pressure_graph.reports.v165_fixed_core_satellite_fss3_tg1 import (
    write_v165_fixed_core_satellite_fss3_tg1,
)


if __name__ == "__main__":
    for name, path in write_v165_fixed_core_satellite_fss3_tg1().items():
        print(f"{name}: {path}")
