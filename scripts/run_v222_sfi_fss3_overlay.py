from pressure_graph.reports.v222_sfi_fss3_overlay import (
    write_v222_sfi_fss3_overlay,
)


if __name__ == "__main__":
    for name, path in write_v222_sfi_fss3_overlay().items():
        print(f"{name}: {path}")
