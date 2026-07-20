from pressure_graph.reports.v2329_event_volatility_transmission_feature_audit import (
    write_v2329_event_volatility_transmission_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v2329_event_volatility_transmission_feature_audit().items():
        print(f"{name}: {path}")
