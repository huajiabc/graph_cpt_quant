from pressure_graph.reports.v203_community_peer_hedge import write_v203_reveal


if __name__ == "__main__":
    for name, path in write_v203_reveal().items():
        print(f"{name}: {path}")
