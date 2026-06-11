#!/usr/bin/env bash
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

top_n="${V085_TOP_N:-50}"
max_symbols="${V085_MAX_SYMBOLS:-10}"
depth_limit="${V085_DEPTH_LIMIT:-200}"
retain_days="${V085_RETAIN_DAYS:-7}"
orderbook_root="${V085_ORDERBOOK_ROOT:-data/orderbook/v0_8_5/bybit}"
demand_queue_path="${V085_DEMAND_QUEUE_PATH:-data/orderflow/demand_queue.parquet}"
sleep_seconds="${V085_SLEEP_SECONDS:-60}"
log_dir="logs/v0_8_5_orderbook"
mkdir -p "${log_dir}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

while true; do
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "===== v0.8.5 orderbook snapshot loop ${started_at} ====="
    python scripts/v085_orderbook_snapshot_once.py \
      --base-config configs/v0_3.yaml \
      --orderbook-root "${orderbook_root}" \
      --demand-queue-path "${demand_queue_path}" \
      --top-n "${top_n}" \
      --max-symbols "${max_symbols}" \
      --depth-limit "${depth_limit}" \
      --retain-days "${retain_days}"
  } >> "${log_dir}/orderbook_snapshot_loop.log" 2>&1 || true

  sleep "${sleep_seconds}"
done
