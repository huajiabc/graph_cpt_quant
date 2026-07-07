param(
    [int]$SleepSeconds = 900,
    [int]$HistoryDays = 45,
    [int]$SignalDays = 7,
    [int]$ReferenceEveryCycles = 4
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = 'E:\graph_quant'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogRoot = Join-Path $ProjectRoot 'logs\paper_live_loop'
$LockPath = Join-Path $LogRoot 'graph_paper_live_loop.lock'

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\scripts"

if (Test-Path $LockPath) {
    $existing = Get-Content $LockPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing -and (Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue)) {
        Add-Content (Join-Path $LogRoot 'all_loop.log') "$(Get-Date -Format o) existing loop pid=$existing, exiting"
        exit 0
    }
}
$PID | Out-File -FilePath $LockPath -Encoding ascii -Force

function Invoke-Logged {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $log = Join-Path $LogRoot "$Name.log"
    Add-Content (Join-Path $LogRoot 'all_loop.log') "===== $stamp START $Name ====="
    & $Python @CommandArgs >> $log 2>&1
    $rc = $LASTEXITCODE
    $done = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Add-Content (Join-Path $LogRoot 'all_loop.log') "===== $done DONE $Name rc=$rc ====="
    return $rc
}

$cycle = 0
try {
    while ($true) {
        $cycle += 1
        Add-Content (Join-Path $LogRoot 'all_loop.log') "===== cycle=$cycle local=$(Get-Date -Format o) ====="

        Invoke-Logged 'v07d2' @(
            'scripts\v07d2_live_once.py',
            '--base-config', 'configs\v0_3.yaml',
            '--paper-config', 'configs\v0_7d2_cic_mir1_paper_live.yaml',
            '--history-days', "$HistoryDays",
            '--signal-days', "$SignalDays"
        ) | Out-Null

        Invoke-Logged 'health_v07d2' @(
            'scripts\live_health_check.py',
            '--base-config', 'configs\v0_3.yaml',
            '--live-root', 'data\live_v07d2',
            '--report-root', 'reports\v0_7d2_cic_mir1_paper_live',
            '--processed-path', 'data\live_v07d2\processed\v0_7d2_live_features.parquet',
            '--health-report', 'reports\v0_7d2_cic_mir1_paper_live\live_health_status.md'
        ) | Out-Null

        Invoke-Logged 'v08_orderflow' @(
            'scripts\v08_orderflow_shadow_once.py',
            '--base-config', 'configs\v0_3.yaml',
            '--source-report-root', 'reports\v0_7d2_cic_mir1_paper_live',
            '--live-feature-path', 'data\live_v07d2\processed\v0_7d2_live_features.parquet'
        ) | Out-Null

        Invoke-Logged 'v085_orderbook' @(
            'scripts\v085_orderbook_snapshot_once.py',
            '--base-config', 'configs\v0_3.yaml',
            '--live-feature-path', 'data\live_v07d2\processed\v0_7d2_live_features.parquet'
        ) | Out-Null

        if ($ReferenceEveryCycles -gt 0 -and (($cycle - 1) % $ReferenceEveryCycles -eq 0)) {
            Invoke-Logged 'v05' @(
                'scripts\v05_live_once.py',
                '--base-config', 'configs\v0_3.yaml',
                '--paper-config', 'configs\v0_5_paper_live.yaml',
                '--history-days', "$HistoryDays",
                '--signal-days', "$SignalDays"
            ) | Out-Null

            Invoke-Logged 'v06a3' @(
                'scripts\v06a3_live_once.py',
                '--base-config', 'configs\v0_3.yaml',
                '--paper-config', 'configs\v0_6a3_paper_live.yaml',
                '--history-days', "$HistoryDays",
                '--signal-days', "$SignalDays"
            ) | Out-Null

            Invoke-Logged 'v07a2' @(
                'scripts\v07a2_live_once.py',
                '--base-config', 'configs\v0_3.yaml',
                '--paper-config', 'configs\v0_7a2_mir1_paper_live.yaml',
                '--history-days', "$HistoryDays",
                '--signal-days', "$SignalDays"
            ) | Out-Null
        }

        Start-Sleep -Seconds $SleepSeconds
    }
}
finally {
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
