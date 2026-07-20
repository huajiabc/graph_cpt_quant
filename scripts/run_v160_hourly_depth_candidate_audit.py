from pressure_graph.reports.v160_hourly_depth_candidate_audit import (
    write_v160_hourly_depth_candidate_audit,
)


if __name__ == "__main__":
    for name, path in write_v160_hourly_depth_candidate_audit().items():
        print(f"{name}: {path}")
