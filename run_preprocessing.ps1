param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if (-not $PythonPath) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
    else { $PythonPath = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location -LiteralPath $taskRoot
try {
    & $PythonPath -X utf8 -u (Join-Path $taskRoot 'preprocess_d.py')
    if ($LASTEXITCODE -ne 0) { throw 'Preprocessing failed.' }
    & $PythonPath -X utf8 -u (Join-Path $taskRoot 'verify_preprocessing.py')
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
} finally { Pop-Location }
