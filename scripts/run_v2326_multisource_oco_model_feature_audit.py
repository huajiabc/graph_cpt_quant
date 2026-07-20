from pressure_graph.reports.v2326_multisource_oco_model_feature_audit import (
    write_v2326_multisource_oco_model_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v2326_multisource_oco_model_feature_audit().items():
        print(f"{name}: {path}")
