@echo off
echo ========================================================
echo  Lancement du Tableau de Bord EEG (Detection d'Absence)
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    echo [OK] Utilisation de l'environnement virtuel .venv
    ".venv\Scripts\python.exe" web_dashboard\server.py
) else (
    echo [INFO] Utilisation de Python système...
    python web_dashboard\server.py
)
pause
