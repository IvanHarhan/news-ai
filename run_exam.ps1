# Chain: wait for download_fnspid.ps1 to finish -> parse CSV -> prices -> exam.
# Logs in A:\news_ai_data\fnspid\{parse,exam}.log, final line "CHAIN DONE"/"CHAIN FAILED".
$d = "A:\news_ai_data\fnspid"
$dl = Join-Path $d "download.log"
$chain = Join-Path $d "chain.log"
Set-Location $PSScriptRoot
while (-not (Select-String -Path $dl -Pattern "DONE|GAVE UP" -Quiet)) { Start-Sleep -Seconds 60 }
if (Select-String -Path $dl -Pattern "GAVE UP" -Quiet) { Add-Content $chain "CHAIN FAILED: download"; exit 1 }
Add-Content $chain ("{0} parse start" -f (Get-Date -Format "HH:mm"))
& .venv\Scripts\python.exe -u fnspid.py *> (Join-Path $d "parse.log")
if ($LASTEXITCODE -ne 0) { Add-Content $chain "CHAIN FAILED: parse"; exit 1 }
Add-Content $chain ("{0} exam start" -f (Get-Date -Format "HH:mm"))
& .venv\Scripts\python.exe -u exam.py *> (Join-Path $d "exam.log")
if ($LASTEXITCODE -ne 0) { Add-Content $chain "CHAIN FAILED: exam"; exit 1 }
Add-Content $chain ("{0} CHAIN DONE" -f (Get-Date -Format "HH:mm"))
