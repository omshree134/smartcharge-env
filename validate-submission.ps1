param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"

function Log([string]$msg) { Write-Host "[$((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))] $msg" }
function Pass([string]$msg) { Log "PASSED -- $msg" }
function Fail([string]$msg) { Log "FAILED -- $msg"; exit 1 }
function Hint([string]$msg) { Write-Host "  Hint: $msg" -ForegroundColor Yellow }

if (-not (Test-Path $RepoDir)) {
    Fail "directory '$RepoDir' not found"
}
$repoPath = (Resolve-Path $RepoDir).Path
$PingUrl = $PingUrl.TrimEnd("/")

Write-Host ""
Write-Host "========================================"
Write-Host "  OpenEnv Submission Validator"
Write-Host "========================================"
Log "Repo:     $repoPath"
Log "Ping URL: $PingUrl"
Write-Host ""

Log "Step 1/3: Pinging HF Space ($PingUrl/reset) ..."
try {
    $response = Invoke-WebRequest `
        -Uri "$PingUrl/reset" `
        -Method POST `
        -ContentType "application/json" `
        -Body "{}" `
        -TimeoutSec 30

    if ($response.StatusCode -eq 200) {
        Pass "HF Space is live and responds to /reset"
    } else {
        Fail "HF Space /reset returned HTTP $($response.StatusCode) (expected 200)"
    }
} catch {
    Fail "HF Space not reachable (connection failed or timed out)"
    Hint "Check your network connection and that the Space is running."
}

Log "Step 2/3: Running docker build ..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "docker command not found"
    Hint "Install Docker: https://docs.docker.com/get-docker/"
}

$dockerContext = $null
if (Test-Path (Join-Path $repoPath "Dockerfile")) {
    $dockerContext = $repoPath
} elseif (Test-Path (Join-Path $repoPath "server\Dockerfile")) {
    $dockerContext = Join-Path $repoPath "server"
} else {
    Fail "No Dockerfile found in repo root or server/ directory"
}

Log "  Found Dockerfile in $dockerContext"

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$dockerOutput = & docker build $dockerContext 2>&1
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction
if ($dockerExit -eq 0) {
    Pass "Docker build succeeded"
} else {
    Write-Host ($dockerOutput | Select-Object -Last 20)
    Fail "Docker build failed"
}

Log "Step 3/3: Running openenv validate ..."
if (-not (Get-Command openenv -ErrorAction SilentlyContinue)) {
    Fail "openenv command not found"
    Hint "Install it: pip install openenv-core"
}

Push-Location $repoPath
try {
    $validateOutput = & openenv validate 2>&1
    if ($LASTEXITCODE -eq 0) {
        Pass "openenv validate passed"
        if ($validateOutput) { Log "  $validateOutput" }
    } else {
        Write-Host $validateOutput
        Fail "openenv validate failed"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================"
Write-Host "  All 3/3 checks passed!"
Write-Host "  Your submission is ready to submit."
Write-Host "========================================"
Write-Host ""
