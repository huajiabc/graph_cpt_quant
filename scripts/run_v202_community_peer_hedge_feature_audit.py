from pressure_graph.reports.v202_community_peer_hedge_feature_audit import (
    write_v202_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v202_feature_audit().items():
        print(f"{name}: {path}")
