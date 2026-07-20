from pressure_graph.reports.v192_premium_innovation_unwind_reversal import (
    write_v192_premium_innovation_unwind_reversal,
)


if __name__ == "__main__":
    for name, path in write_v192_premium_innovation_unwind_reversal().items():
        print(f"{name}: {path}")
