from pressure_graph.reports.v2335_liquidation_pre_event_feature_contract import (
    write_v2335_contract,
)


if __name__ == "__main__":
    for name, path in write_v2335_contract().items():
        print(f"{name}: {path}")
