from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v104_p2_graph_bucket_overlay import (  # noqa: E402
    write_v104_p2_graph_bucket_overlay,
)


if __name__ == "__main__":
    for name, path in write_v104_p2_graph_bucket_overlay().items():
        print(f"{name}={path}")
