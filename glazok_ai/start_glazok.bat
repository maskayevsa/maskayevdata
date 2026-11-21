@echo off
chcp 65001 > nul
title Glazok AI Server
cd /d "%~dp0"

echo ==================================================
echo   Запуск сервера Видеоглазка (Supervisor + AI)
echo   Web-интерфейс: http://127.0.0.1:8081
echo   Не закрывайте это окно для работы системы
echo ==================================================
echo.

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -u "app\supervisor.py"
) else (
    python -u "app\supervisor.py"
)

echo.
echo [!] Супервизор завершил работу.
pause
