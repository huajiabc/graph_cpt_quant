from pressure_graph.reports.v192_premium_innovation_unwind_reversal_audit import (
    write_v192_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v192_independent_audit().items():
        print(f"{name}: {path}")
