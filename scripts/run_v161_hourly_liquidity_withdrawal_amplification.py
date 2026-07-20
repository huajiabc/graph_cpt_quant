from pressure_graph.reports.v161_hourly_liquidity_withdrawal_amplification import (
    write_v161_hourly_liquidity_withdrawal_amplification,
)


if __name__ == "__main__":
    for name, path in write_v161_hourly_liquidity_withdrawal_amplification().items():
        print(f"{name}: {path}")
