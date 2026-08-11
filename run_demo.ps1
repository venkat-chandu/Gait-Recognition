# Runs the full synthetic demonstration and opens the Streamlit UI in Chrome.
$ErrorActionPreference = 'Stop'
$runtimeRoot = Join-Path $env:TEMP 'gait-demo-runtime'
$runtimePython = Join-Path $runtimeRoot 'Scripts\python.exe'
$runtimeStreamlit = Join-Path $runtimeRoot 'Scripts\streamlit.exe'
$logFile = Join-Path $PSScriptRoot 'outputs\demo_run.log'
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot 'outputs') | Out-Null

if (-not (Test-Path $runtimePython)) {
    python -m venv $runtimeRoot
}

& $runtimePython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt') 2>&1 | Tee-Object -FilePath $logFile -Append
& $runtimePython -m scripts.generate_synthetic_data 2>&1 | Tee-Object -FilePath $logFile -Append
& $runtimePython -m training.train --epochs 35 2>&1 | Tee-Object -FilePath $logFile -Append
& $runtimePython -m evaluation.evaluate 2>&1 | Tee-Object -FilePath $logFile -Append

Start-Process -FilePath $runtimeStreamlit -ArgumentList 'run','app.py','--server.headless=true' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
Start-Sleep -Seconds 8
$chromeCandidates = @(
    (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
    (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe')
)
$chrome = $chromeCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if ($chrome) { Start-Process -FilePath $chrome -ArgumentList 'http://localhost:8501' }
else { Start-Process 'http://localhost:8501' }
