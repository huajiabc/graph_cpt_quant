from pressure_graph.reports.v223_sfi_fss3_overlay_audit import (
    write_v223_sfi_fss3_overlay_audit,
)


if __name__ == "__main__":
    for name, path in write_v223_sfi_fss3_overlay_audit().items():
        print(f"{name}: {path}")
