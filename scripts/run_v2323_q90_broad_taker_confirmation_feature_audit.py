from pressure_graph.reports.v2323_q90_broad_taker_confirmation_feature_audit import (
    write_v2323_q90_broad_taker_confirmation_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v2323_q90_broad_taker_confirmation_feature_audit().items():
        print(f"{name}: {path}")
