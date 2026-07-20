from pressure_graph.reports.v114_semivariance_transmission import (
    write_v114_semivariance_transmission,
)


if __name__ == "__main__":
    for name, path in write_v114_semivariance_transmission().items():
        print(f"{name}={path}")
