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

top_n="${V08_TOP_N:-50}"
max_symbols="${V08_MAX_SYMBOLS:-5}"
recent_trade_limit="${V08_RECENT_TRADE_LIMIT:-500}"
retain_days="${V08_RETAIN_DAYS:-7}"
report_lookback_days="${V08_REPORT_LOOKBACK_DAYS:-7}"
orderflow_root="${V08_ORDERFLOW_ROOT:-data/orderflow/v0_8_demand/bybit}"
demand_queue_path="${V08_DEMAND_QUEUE_PATH:-data/orderflow/demand_queue.parquet}"
sleep_seconds="${V08_SLEEP_SECONDS:-60}"
log_dir="logs/v0_8_orderflow"
mkdir -p "${log_dir}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

while true; do
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "===== v0.8 orderflow shadow loop ${started_at} ====="
    python scripts/v08_orderflow_shadow_once.py \
      --base-config configs/v0_3.yaml \
      --orderflow-root "${orderflow_root}" \
      --demand-queue-path "${demand_queue_path}" \
      --top-n "${top_n}" \
      --max-symbols "${max_symbols}" \
      --recent-trade-limit "${recent_trade_limit}" \
      --retain-days "${retain_days}" \
      --report-lookback-days "${report_lookback_days}"
  } >> "${log_dir}/orderflow_shadow_loop.log" 2>&1 || true

  sleep "${sleep_seconds}"
done
