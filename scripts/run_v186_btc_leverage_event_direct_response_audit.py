from pressure_graph.reports.v186_btc_leverage_event_direct_response_audit import (
    write_v186_audit,
)


if __name__ == "__main__":
    for name, path in write_v186_audit().items():
        print(f"{name}: {path}")
