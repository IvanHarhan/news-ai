@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem UI: local server on 8790 in a separate window + browser, then training.
start "news_ai UI" /min .venv\Scripts\python.exe -m http.server 8790 --bind 127.0.0.1 --directory ui
timeout /t 1 >nul
start "" http://127.0.0.1:8790/
.venv\Scripts\python.exe train.py
pause
