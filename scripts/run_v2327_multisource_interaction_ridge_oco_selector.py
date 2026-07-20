from pressure_graph.reports.v2327_multisource_interaction_ridge_oco_selector import (
    write_v2327_multisource_interaction_ridge_oco_selector,
)


if __name__ == "__main__":
    for name, path in write_v2327_multisource_interaction_ridge_oco_selector().items():
        print(f"{name}: {path}")
