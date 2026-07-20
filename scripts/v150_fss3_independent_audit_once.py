from __future__ import annotations

from pressure_graph.reports.v150_fss3_independent_audit import (
    run_v150_fss3_independent_audit,
)


if __name__ == "__main__":
    for name, path in run_v150_fss3_independent_audit().items():
        print(f"{name}: {path}")
