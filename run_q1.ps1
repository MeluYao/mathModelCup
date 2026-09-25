param([string]$PythonPath='')
$ErrorActionPreference='Stop'
if (-not $PythonPath) {
    $taskPython=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $taskPython) { $PythonPath=$taskPython }
    else { $PythonPath=(Get-Command python -ErrorAction Stop).Source }
}
Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($taskScript in @('solve_q1.py','verify_q1.py','visualize_q1.py')) {
        & $PythonPath -X utf8 -u (Join-Path $PSScriptRoot $taskScript)
        if ($LASTEXITCODE -ne 0) { throw "$taskScript failed." }
    }
} finally { Pop-Location }
