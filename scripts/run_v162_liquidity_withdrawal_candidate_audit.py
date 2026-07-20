from pressure_graph.reports.v162_liquidity_withdrawal_candidate_audit import (
    write_v162_liquidity_withdrawal_candidate_audit,
)


if __name__ == "__main__":
    for name, path in write_v162_liquidity_withdrawal_candidate_audit().items():
        print(f"{name}: {path}")
