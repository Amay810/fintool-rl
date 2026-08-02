param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [string]$ApiKey = "",

    [ValidateSet("train", "dev", "test", "challenge")]
    [string]$Split = "test",

    [int]$Limit = 0,

    [int]$MaxSteps = 8,

    [string]$Db = "data\sec_snapshot_15.sqlite",

    [string]$Tasks = "data\generated_sec_15_tasks.jsonl",

    [string]$OutDir = "logs"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$safeName = ($ModelName -replace "[^A-Za-z0-9_.-]", "_")
$store = Join-Path $OutDir "baseline_${safeName}_${Split}.sqlite"
$report = Join-Path $OutDir "baseline_${safeName}_${Split}.report.json"
$failures = Join-Path $OutDir "baseline_${safeName}_${Split}.failures.jsonl"

$env:FINTOOL_LLM_BASE_URL = $BaseUrl
$env:FINTOOL_LLM_MODEL = $ModelName
$env:FINTOOL_LLM_API_KEY = $ApiKey

Write-Host "Running baseline"
Write-Host "  model : $ModelName"
Write-Host "  split : $Split"
Write-Host "  store : $store"
Write-Host "  report: $report"

$args = @(
    "baseline",
    "--db", $Db,
    "--tasks", $Tasks,
    "--split", $Split,
    "--store", $store,
    "--report", $report,
    "--failure-table", $failures,
    "--skip-existing",
    "--max-steps", "$MaxSteps"
)
if ($Limit -gt 0) {
    $args += @("--limit", "$Limit")
}

& fintool-rl @args
if ($LASTEXITCODE -ne 0) {
    throw "baseline failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Re-analyze later with:"
Write-Host "  fintool-rl analyze-baseline --db $Db --tasks $Tasks --split $Split --store $store --report $report --failure-table $failures"
