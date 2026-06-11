#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate quant
fi

days="${1:-7}"

python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .

pressure-graph run-v05-paper \
  --config configs/v0_3.yaml \
  --paper-config configs/v0_5_paper_live.yaml \
  --days "${days}"

printf "\n## v0.5 current status\n"
sed -n '1,120p' reports/v0_5_paper_live/current_status.md

printf "\n## v0.5 candidate status\n"
sed -n '1,160p' reports/v0_5_paper_live/candidate_status.md
