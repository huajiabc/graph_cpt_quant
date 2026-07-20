from pressure_graph.reports.v164_weekly_liquidity_quality_audit import (
    write_v164_weekly_liquidity_quality_audit,
)


if __name__ == "__main__":
    for name, path in write_v164_weekly_liquidity_quality_audit().items():
        print(f"{name}: {path}")
