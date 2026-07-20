from pressure_graph.reports.v119_variance_risk_premium import (
    write_v119_variance_risk_premium,
)


if __name__ == "__main__":
    for name, path in write_v119_variance_risk_premium().items():
        print(f"{name}: {path}")
