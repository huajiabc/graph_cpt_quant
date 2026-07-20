param(
    [int]$SleepSeconds = 900,
    [int]$HistoryDays = 45,
    [int]$SignalDays = 7,
    [int]$ReferenceEveryCycles = 4
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = 'E:\graph_quant'
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Python = if (Test-Path -LiteralPath $VenvPython) {
    $VenvPython
} else {
    $resolvedPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $resolvedPython) {
        throw "Python was not found at $VenvPython or on PATH."
    }
    $resolvedPython.Source
}
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

        if ($ReferenceEveryCycles -gt 0 -and (($cycle - 1) % $ReferenceEveryCycles -eq 0)) {
            if ($cycle -eq 1 -or ($cycle % 96 -eq 0)) {
                Invoke-Logged 'v93_token_mapping_refresh' @(
                    'scripts\refresh_dexpaprika_token_mapping_v93.py',
                    '--promote'
                ) | Out-Null
            }
            Invoke-Logged 'v65_token_attention_refresh' @(
                'scripts\backfill_dexpaprika_token_ohlcv_v65.py',
                '--lookback-days', '14'
            ) | Out-Null
        }

        $primaryRc = Invoke-Logged 'v07d2' @(
            'scripts\v07d2_live_once.py',
            '--base-config', 'configs\v0_3.yaml',
            '--paper-config', 'configs\v0_7d2_cic_mir1_paper_live.yaml',
            '--history-days', "$HistoryDays",
            '--signal-days', "$SignalDays"
        )

        if ($primaryRc -ne 0) {
            Add-Content (Join-Path $LogRoot 'all_loop.log') "$(Get-Date -Format o) primary refresh failed rc=$primaryRc; dependent jobs skipped"
            Invoke-Logged 'health_v07d2' @(
                'scripts\live_health_check.py',
                '--base-config', 'configs\v0_3.yaml',
                '--live-root', 'data\live_v07d2',
                '--report-root', 'reports\v0_7d2_cic_mir1_paper_live',
                '--processed-path', 'data\live_v07d2\processed\v0_7d2_live_features.parquet',
                '--health-report', 'reports\v0_7d2_cic_mir1_paper_live\live_health_status.md'
            ) | Out-Null
            Start-Sleep -Seconds $SleepSeconds
            continue
        }

        # Frozen short candidates run as paper-live shadows only. They consume
        # the as-of feature set refreshed by v07d2 and never place real orders.
        Invoke-Logged 'v10_short_s1_diagnostic' @(
            'scripts\v10_short_live_once.py',
            '--prepared-path', 'data\live_v07d2\processed\v0_7d2_live_features.parquet',
            '--signal-days', "$SignalDays",
            '--family', 's1'
        ) | Out-Null

        # v11.2 is a paper-live shadow observer with an isolated 73-symbol
        # kline cache. It writes virtual signals only and has no order route.
        Invoke-Logged 'v112_topology_shadow' @(
            'scripts\v112_topology_live_once.py',
            '--config', 'configs\v11_2_high_vol_topology_paper_live.yaml'
        ) | Out-Null

        # Frozen v14.9 FSS3 runs as a stateful weekly record-only shadow.
        # The runner refreshes its isolated cache only when a decision is due
        # or its six-hour observation interval has elapsed. It has no order route.
        Invoke-Logged 'fss3_forward_shadow' @(
            'scripts\fss3_forward_shadow_once.py',
            '--config', 'configs\v14_9_fss3_forward_shadow.yaml'
        ) | Out-Null

        # CM2 is the exact fixed 80% FSS3 / 20% TG1 portfolio shadow.
        # TG1 is an internal reference sleeve only; either missing sleeve
        # fails the combined week closed. No cross-sleeve netting is claimed.
        Invoke-Logged 'cm2_forward_shadow' @(
            'scripts\cm2_forward_shadow_once.py',
            '--config', 'configs\v16_5_cm2_forward_shadow.yaml'
        ) | Out-Null

        # The post-selected q90 breakout remains CANDIDATE_WATCH. Official
        # Binance daily archives are delayed, so this is untouched forward
        # research evidence only and can never count as execution-timely.
        Invoke-Logged 'q90_forward_shadow' @(
            'scripts\q90_forward_shadow_once.py',
            '--config', 'configs\v23_8_q90_forward_shadow.yaml'
        ) | Out-Null

        # Liquidation graph features are a LIVE_DIAGNOSTIC volatility-state
        # factor, not a directional strategy. The remote causal clock starts
        # at the first successful collection batch and outcomes stay unloaded.
        Invoke-Logged 'liquidation_graph_diagnostic' @(
            'scripts\liquidation_graph_forward_once.py',
            '--config', 'configs\v23_42_liquidation_graph_live_diagnostic.yaml'
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
