param(
    [string]$PingUrl = "",
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"

function Pass([string]$msg) { Write-Host "[PASS] $msg" -ForegroundColor Green }
function Fail([string]$msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }
function Info([string]$msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }

Set-Location $RepoDir

Write-Host ""
Write-Host "========================================"
Write-Host "  SmartCharge Pre-Validation (PowerShell)"
Write-Host "========================================"
Info "Repo: $(Get-Location)"

if ($PingUrl -ne "") {
    Info "Step 1/4: Pinging Space /reset"
    try {
        $body = @{ mode = "easy"; seed = 42 } | ConvertTo-Json
        $resp = Invoke-WebRequest -Uri ($PingUrl.TrimEnd('/') + "/reset") -Method POST -ContentType "application/json" -Body $body
        if ($resp.StatusCode -eq 200) { Pass "Space /reset returned HTTP 200" } else { Fail "Space /reset status=$($resp.StatusCode)" }
    } catch {
        Fail "Space ping failed: $($_.Exception.Message)"
    }
} else {
    Info "Step 1/4: Skipping Space ping (no URL provided)"
}

Info "Step 2/4: Checking required files"
foreach ($f in @("openenv.yaml", "inference.py", "Dockerfile", "README.md")) {
    if (-not (Test-Path $f)) { Fail "Missing $f" }
}
Pass "Required files exist"

$taskCount = (Select-String -Path "openenv.yaml" -Pattern "^\s*-\s+id:" | Measure-Object).Count
if ($taskCount -lt 3) { Fail "openenv.yaml has fewer than 3 tasks ($taskCount)" }
Pass "openenv.yaml defines >= 3 tasks ($taskCount)"

Info "Step 3/4: Validating inference output format for easy/medium/hard"
$tasks = @("easy", "medium", "hard")
foreach ($task in $tasks) {
    $outFile = Join-Path $env:TEMP "openenv_$task.log"

    $env:API_BASE_URL = if ($env:API_BASE_URL) { $env:API_BASE_URL } else { "https://router.huggingface.co/v1" }
    $env:MODEL_NAME = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "baseline" }
    $env:TASK_NAME = $task
    $env:MAX_STEPS_PER_TASK = if ($env:MAX_STEPS_PER_TASK) { $env:MAX_STEPS_PER_TASK } else { "20" }

    try {
        python .\inference.py *> $outFile
    } catch {
        Get-Content $outFile -ErrorAction SilentlyContinue
        Fail "inference failed for task=$task"
    }

    $lines = Get-Content $outFile | Where-Object { $_.Trim().Length -gt 0 }
    if ($lines.Count -eq 0) { Fail "no output for task=$task" }

    if ($lines[0] -notmatch '^\[START\] task=\S+ env=\S+ model=\S+$') { Fail "bad [START] for task=${task}: $($lines[0])" }
    if ($lines[-1] -notmatch '^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=.*$') { Fail "bad [END] for task=${task}: $($lines[-1])" }
    if (-not ($lines | Where-Object { $_.StartsWith("[STEP] ") })) { Fail "missing [STEP] lines for task=${task}" }

    foreach ($line in $lines) {
        if ($line.StartsWith("[STEP] ")) {
            if ($line -notmatch '^\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=.+$') {
                Fail "bad [STEP] for task=${task}: $line"
            }
            $rewardStr = [regex]::Match($line, 'reward=(\d+\.\d{2})').Groups[1].Value
            $reward = [double]$rewardStr
            if ($reward -lt 0.0 -or $reward -gt 1.0) { Fail "reward out of bounds for task=${task}: $reward" }
        }
    }
    Pass "inference format OK for task=$task"
}

Info "Step 4/4: API import check"
python -c "from service.api import app; assert app is not None; print('api import ok')" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Fail "API import failed (install dependencies with: pip install -r requirements.txt)"
}
Pass "API app import check passed"

Write-Host ""
Write-Host "ALL PRE-VALIDATION CHECKS PASSED" -ForegroundColor Green
