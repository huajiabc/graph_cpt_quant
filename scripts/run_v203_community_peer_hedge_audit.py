from pressure_graph.reports.v203_community_peer_hedge_audit import (
    write_v203_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v203_independent_audit().items():
        print(f"{name}: {path}")
