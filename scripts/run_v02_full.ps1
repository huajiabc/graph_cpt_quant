$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "logs\v0_2"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("full_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$StatusPath = Join-Path $LogDir "full_run_status.txt"

function Write-Status {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
    $line | Set-Content -Path $StatusPath
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Status "START $Name"
    & $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Status "FAILED $Name exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "DONE $Name"
}

Run-Step "collect remaining public trades" {
    pressure-graph collect-v01-1m --source public-trades --symbol-day-offset 485 --max-symbol-days 800 --public-trade-workers 8
}

Run-Step "run v0.1 1m execution comparison" {
    pressure-graph run-v01-1m --source public-trades
}

Run-Step "run v0.2 full tick execution and baselines" {
    pressure-graph run-v02 --source public-trades
}

Run-Step "summarize v0.2 outputs" {
    @'
from pathlib import Path
import pandas as pd

base = Path("reports/v0_2")
tick = pd.read_csv(base / "tick_execution_comparison.csv")
cols = [
    "candidate",
    "fill_policy",
    "cost_single_side_bps",
    "trades",
    "signals",
    "fill_rate",
    "net_expectancy",
    "tp_first_rate",
    "sl_first_rate",
    "timeout_rate",
]
print("## tick 5bp top")
print(
    tick[tick["cost_single_side_bps"].eq(5)][cols]
    .sort_values("net_expectancy", ascending=False)
    .head(20)
    .to_string(index=False)
)
for name in ["matched_random_baseline.csv", "entry_only_baseline.csv"]:
    path = base / name
    if path.exists() and path.stat().st_size:
        df = pd.read_csv(path)
        print(f"\n## {name}")
        print(
            df[df["cost_single_side_bps"].eq(5)][cols]
            .sort_values("net_expectancy", ascending=False)
            .head(20)
            .to_string(index=False)
        )
'@ | python -
}

Write-Status "COMPLETE v0.2 full run"
