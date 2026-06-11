#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest -q
python -m ruff check .

pressure-graph run-v03 --config configs/v0_3.yaml

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("reports/v0_3")
summary = pd.read_csv(base / "c2_all_eligible_summary.csv")
ex_may = pd.read_csv(base / "c2_ex_may_summary.csv")

print("\n## C2 all-eligible summary, 5bp")
print(
    summary[summary["cost_single_side_bps"].eq(5)]
    .sort_values("net_expectancy", ascending=False)
    .to_string(index=False)
)

print("\n## C2 ex-May, 5bp")
print(
    ex_may[
        ex_may["cost_single_side_bps"].eq(5)
        & ex_may["partition"].isin(["full_12m", "ex_2026_05", "only_2026_05"])
    ].to_string(index=False)
)
PY
