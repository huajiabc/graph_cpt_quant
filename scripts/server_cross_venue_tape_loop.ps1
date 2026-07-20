param(
    [int]$RestartDelaySeconds = 10,
    [int]$MaxSymbols = 20
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = 'E:\graph_quant'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogRoot = Join-Path $ProjectRoot 'logs\cross_venue_tape'
$LockPath = Join-Path $LogRoot 'cross_venue_tape_loop.lock'

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\scripts"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python was not found at $Python"
}
if (Test-Path -LiteralPath $LockPath) {
    $existing = Get-Content $LockPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing -and (Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue)) {
        Add-Content (Join-Path $LogRoot 'loop.log') "$(Get-Date -Format o) existing loop pid=$existing, exiting"
        exit 0
    }
}
$PID | Out-File -FilePath $LockPath -Encoding ascii -Force

try {
    while ($true) {
        Add-Content (Join-Path $LogRoot 'loop.log') "$(Get-Date -Format o) recorder start"
        & $Python 'scripts\cross_venue_tape_recorder.py' '--max-symbols' "$MaxSymbols" >> (Join-Path $LogRoot 'recorder.log') 2>&1
        $rc = $LASTEXITCODE
        Add-Content (Join-Path $LogRoot 'loop.log') "$(Get-Date -Format o) recorder exit rc=$rc"
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
finally {
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
