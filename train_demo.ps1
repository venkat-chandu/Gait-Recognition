# Generate the synthetic dataset, train the saved model, and evaluate it.
$ErrorActionPreference = 'Stop'
$runtimePython = Join-Path $env:TEMP 'gait-demo-runtime\Scripts\python.exe'
$logFile = Join-Path $PSScriptRoot 'outputs\training_run.log'
if (-not (Test-Path $runtimePython)) { throw 'The demo Python environment is not ready.' }
& $runtimePython -m scripts.generate_synthetic_data 2>&1 | Tee-Object -FilePath $logFile
& $runtimePython -m training.train --epochs 35 2>&1 | Tee-Object -FilePath $logFile -Append
& $runtimePython -m evaluation.evaluate 2>&1 | Tee-Object -FilePath $logFile -Append
