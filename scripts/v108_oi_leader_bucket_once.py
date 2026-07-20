from pressure_graph.reports.v108_oi_leader_bucket import write_v108_oi_leader_bucket


if __name__ == "__main__":
    for name, path in write_v108_oi_leader_bucket().items():
        print(f"{name}={path}")
