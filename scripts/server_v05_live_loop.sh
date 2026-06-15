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

export PYTHONPATH="${PWD}/src:${PWD}/scripts:${PYTHONPATH:-}"

history_days="${V05_HISTORY_DAYS:-45}"
signal_days="${V05_SIGNAL_DAYS:-7}"
max_symbols="${V05_MAX_SYMBOLS:-0}"
sleep_seconds="${V05_SLEEP_SECONDS:-900}"
log_dir="logs/v0_5"
mkdir -p "${log_dir}"

while true; do
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "===== v0.5 live loop ${started_at} ====="
    python scripts/v05_live_once.py \
      --base-config configs/v0_3.yaml \
      --paper-config configs/v0_5_paper_live.yaml \
      --history-days "${history_days}" \
      --signal-days "${signal_days}" \
      --max-symbols "${max_symbols}"
  } >> "${log_dir}/live_loop.log" 2>&1 || true

  now_epoch="$(date -u +%s)"
  next_epoch="$(( ((now_epoch / sleep_seconds) + 1) * sleep_seconds + 20 ))"
  sleep_for="$(( next_epoch - now_epoch ))"
  if [ "${sleep_for}" -lt 30 ]; then
    sleep_for="${sleep_seconds}"
  fi
  sleep "${sleep_for}"
done
