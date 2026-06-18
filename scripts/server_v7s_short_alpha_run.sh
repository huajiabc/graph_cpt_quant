#!/usr/bin/env bash
# v7S Short Alpha Exploration — A100 production runner.
#
# Direction E (strict CIC-failure-confirmed short) is the only direction
# currently wired. Other directions A/B/C/D will be added in follow-up
# commits and re-use the same runner with --enabled-directions.
#
# Inputs expected on the A100 box:
#   data/processed/v0_3/perp_pressure_features_all_eligible.parquet
#   reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet
#   data/orderflow_history/binance_um/cic_event_orderflow.parquet  (optional;
#     without it the sell_flow gate fails closed unless e_sell_flow_fail_open=true)
#
# Outputs land under:
#   reports/v7s_short_alpha/E_cic_failure_confirmed/
#
# Usage (on A100 over the jumphost chain documented in memory note
# `a100-ssh-access`):
#   ssh -L 2222:10.106.200.247:2222 root@10.115.7.6 -p 25711
#   ssh root@localhost -p 2222
#   cd /opt/data/private/Wangjb/graph_cpt_quant
#   bash scripts/server_v7s_short_alpha_run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  export PS1="${PS1:-}"
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate quant
elif command -v conda >/dev/null 2>&1; then
  export PS1="${PS1:-}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate quant
fi

log_dir="logs/v7s_short_alpha"
mkdir -p "${log_dir}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_file="${log_dir}/run_${started_at//:/-}.log"

{
  echo "===== v7S Short Alpha Exploration ${started_at} ====="
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "commit: $(git rev-parse --short HEAD)"
  echo "python: $(python --version 2>&1)"
  python -m pressure_graph.cli run-v7s-short-alpha \
    --config configs/v0_3.yaml
  echo "===== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} 2>&1 | tee -a "${log_file}"
