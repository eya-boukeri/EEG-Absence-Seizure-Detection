Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Lancement du Tableau de Bord EEG (Detection d'Absence) " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "[OK] Utilisation de l'environnement virtuel .venv" -ForegroundColor Green
    & $venvPython (Join-Path $PSScriptRoot "web_dashboard/server.py")
} else {
    Write-Host "[INFO] Utilisation de python systeme..." -ForegroundColor Yellow
    python (Join-Path $PSScriptRoot "web_dashboard/server.py")
}
