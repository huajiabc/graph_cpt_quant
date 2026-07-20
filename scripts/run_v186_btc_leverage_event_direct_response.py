from pressure_graph.reports.v186_btc_leverage_event_direct_response import (
    write_v186_btc_leverage_event_direct_response,
)


if __name__ == "__main__":
    for name, path in write_v186_btc_leverage_event_direct_response().items():
        print(f"{name}: {path}")
