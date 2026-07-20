from __future__ import annotations

from pressure_graph.reports.v153_crowded_source_exhaustion_reversal import (
    write_v153_crowded_source_exhaustion_reversal,
)


if __name__ == "__main__":
    for name, path in write_v153_crowded_source_exhaustion_reversal().items():
        print(f"{name}: {path}")
