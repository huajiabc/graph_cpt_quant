from pressure_graph.reports.v221_sfi_fss3_overlay_feature_audit import write_v221_feature_audit


if __name__ == "__main__":
    for name, path in write_v221_feature_audit().items():
        print(f"{name}: {path}")
