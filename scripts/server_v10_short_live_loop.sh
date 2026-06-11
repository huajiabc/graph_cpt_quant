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

signal_days="${V10_SHORT_SIGNAL_DAYS:-7}"
sleep_seconds="${V10_SHORT_SLEEP_SECONDS:-900}"
log_dir="logs/v1_0_short"
mkdir -p "${log_dir}"

while true; do
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "===== v1.0 short mirror live shadow ${started_at} ====="
    python scripts/v10_short_live_once.py \
      --prepared-path data/live_v07d2/processed/v0_7d2_live_features.parquet \
      --signal-days "${signal_days}"
  } >> "${log_dir}/live_loop.log" 2>&1 || true

  now_epoch="$(date -u +%s)"
  next_epoch="$(( ((now_epoch / sleep_seconds) + 1) * sleep_seconds + 80 ))"
  sleep_for="$(( next_epoch - now_epoch ))"
  if [ "${sleep_for}" -lt 30 ]; then
    sleep_for="${sleep_seconds}"
  fi
  sleep "${sleep_for}"
done

