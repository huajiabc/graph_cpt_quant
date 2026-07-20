from pressure_graph.reports.v2310_v238_forward_shadow_feature_update import (
    write_v2310_v238_forward_shadow_feature_update,
)


if __name__ == "__main__":
    for name, path in write_v2310_v238_forward_shadow_feature_update().items():
        print(f"{name}: {path}")
