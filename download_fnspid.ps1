# Download FNSPID nasdaq_exteral_data.csv (23.2 GB) with resume, retrying until complete.
# Runs detached; progress in A:\news_ai_data\fnspid\download.log
$dir = "A:\news_ai_data\fnspid"
$file = Join-Path $dir "nasdaq_exteral_data.csv"
$log = Join-Path $dir "download.log"
$url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
$expected = 23232979597
New-Item -ItemType Directory -Force $dir | Out-Null
for ($i = 1; $i -le 200; $i++) {
    $have = 0
    if (Test-Path $file) { $have = (Get-Item $file).Length }
    if ($have -ge $expected) { Add-Content $log "DONE $have"; exit 0 }
    Add-Content $log ("{0} attempt {1}, have {2:N0} bytes" -f (Get-Date -Format "HH:mm:ss"), $i, $have)
    & curl.exe -L -sS -C - --retry 5 --retry-delay 10 --speed-limit 50000 --speed-time 120 -o $file $url
    Start-Sleep -Seconds 15
}
Add-Content $log "GAVE UP"
exit 1
