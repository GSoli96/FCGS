@echo off
echo ==========================================
echo  FCGS — Sync codice su tutti i PC
echo ==========================================
echo.

:: ── 1. Commit + push locale ───────────────────────────────────────────────
echo [Locale] Staging modifiche...
cd /d %~dp0..
git add -u
git add consolidate_results.py dataset_manager.py stop_fcgs.py fix_hf_filenames.py upload_datasets_to_hf.py requirements_windows.txt setup.txt tasks.md README.md geosciences_stress_test\ grid_search_config_MSI.json grid_search_config_Siando.json grid_search_config_MSIDomino11.json grid_search_config_DAIS-RTX-5000-5.json 2>nul

git diff --cached --quiet
if errorlevel 1 (
    echo [Locale] Committing...
    git commit -m "Update automatico da bat"
    echo [Locale] Pushing su GitHub...
    git push
) else (
    echo [Locale] Nessuna modifica da committare. Verifico push...
    git push
)
echo.

:: ── 2. Clone o pull sui PC remoti ─────────────────────────────────────────
set REPO=https://github.com/GSoli96/FCGS.git

echo [MSI] Clone o pull...
ssh msi "if exist C:\Users\giand\Desktop\prog\FCGS\.git (cd /d C:\Users\giand\Desktop\prog\FCGS && git pull) else (mkdir C:\Users\giand\Desktop\prog && cd /d C:\Users\giand\Desktop\prog && git clone %REPO% FCGS)"
echo.

echo [Domino11] Clone o pull...
ssh domino11 "if exist C:\Users\domin\Desktop\prog\FCGS\.git (cd /d C:\Users\domin\Desktop\prog\FCGS && git pull) else (mkdir C:\Users\domin\Desktop\prog && cd /d C:\Users\domin\Desktop\prog && git clone %REPO% FCGS)"
echo.

echo [Domino12] Clone o pull...
ssh domino12 "if exist C:\Users\dastl\Desktop\Giando\FCGS\.git (cd /d C:\Users\dastl\Desktop\Giando\FCGS && git pull) else (mkdir C:\Users\dastl\Desktop\Giando && cd /d C:\Users\dastl\Desktop\Giando && git clone %REPO% FCGS)"
echo.

echo ==========================================
echo  Fatto.
echo ==========================================
pause
