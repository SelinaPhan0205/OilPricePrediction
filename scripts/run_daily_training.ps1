param(
    [string]$Dataset = "data\processed\dataset_final_noleak_ext_market_regime_v1.csv",
    [string]$OutDir = "ml\classification\results_daily_ext_market_regime_v1_run",
    [string]$Python = "C:\Users\SelinaPhan\AppData\Local\Programs\Python\Python313\python.exe",
    [string[]]$Steps = @(
        "ml\classification\step1_train_baseline.py",
        "ml\classification\step2_finetune_ensemble.py",
        "ml\classification\step5_smart_selection.py"
    )
)

$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $RepoRoot

if (![System.IO.Path]::IsPathRooted($Dataset)) {
    $Dataset = Join-Path $RepoRoot $Dataset
}
if (![System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $RepoRoot $OutDir
}

if (!(Test-Path $Python)) {
    throw "Python not found: $Python"
}
if (!(Test-Path $Dataset)) {
    throw "Dataset not found: $Dataset"
}

$OutDir = [System.IO.Path]::GetFullPath($OutDir)
$LogDir = Join-Path $OutDir "logs"
if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$env:CLASSIFICATION_DATA_PATH = $Dataset
$env:CLASSIFICATION_OUT_DIR = $OutDir
$env:CLASSIFICATION_PRICE_SOURCE_PATH = Join-Path $RepoRoot "data\processed\dataset_step4_noleak.csv"

if (!$env:SEARCH_N_JOBS) { $env:SEARCH_N_JOBS = "4" }
if (!$env:MODEL_N_JOBS) { $env:MODEL_N_JOBS = "4" }
if (!$env:STEP2_N_ITER) { $env:STEP2_N_ITER = "20" }
if (!$env:STEP5_N_ITER) { $env:STEP5_N_ITER = "10" }
if (!$env:STEP5_PERM_REPEATS) { $env:STEP5_PERM_REPEATS = "3" }
if (!$env:STEP5_TOPN_LIST) { $env:STEP5_TOPN_LIST = "10,15,20,25" }

Write-Host "Dataset: $Dataset"
Write-Host "Output : $OutDir"
Write-Host "Logs   : $LogDir"

foreach ($Step in $Steps) {
    if (!(Test-Path $Step)) {
        throw "Step not found: $Step"
    }

    $Name = [System.IO.Path]::GetFileNameWithoutExtension($Step)
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Running $Step"
    Write-Host "Console log is not tee'd to file in sandboxed PowerShell."
    Write-Host "========================================================================"

    & $Python $Step
    if ($LASTEXITCODE -ne 0) {
        throw "Training step failed: $Step"
    }
}

& $Python scripts\summarize_daily_training.py --out-dir $OutDir
if ($LASTEXITCODE -ne 0) {
    throw "Summary failed"
}
