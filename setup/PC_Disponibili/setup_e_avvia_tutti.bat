@echo off
echo ==========================================
echo  FCGS — Setup completo e avvio su tutti i PC
echo ==========================================
echo.

set REPO=https://github.com/GSoli96/FCGS.git

:: ── MSI ───────────────────────────────────────────────────────────────────
echo [MSI] 1/5 Clone o pull...
ssh msi "if exist C:\Users\giand\Desktop\prog\FCGS\.git (cd /d C:\Users\giand\Desktop\prog\FCGS && git pull) else (if not exist C:\Users\giand\Desktop\prog mkdir C:\Users\giand\Desktop\prog && cd /d C:\Users\giand\Desktop\prog && git clone %REPO% FCGS)"
echo [MSI] 2/5 Creazione venv (se non esiste)...
ssh msi "if not exist C:\Users\giand\Desktop\prog\FCGS\venv\Scripts\python.exe (cd /d C:\Users\giand\Desktop\prog\FCGS && python -m venv venv && venv\Scripts\python -m pip install --upgrade pip)"
echo [MSI] 3/5 Installazione requirements...
ssh msi "cd /d C:\Users\giand\Desktop\prog\FCGS && venv\Scripts\pip install -r requirements_windows.txt"
echo [MSI] 4/5 Installazione PyTorch CUDA 12.6...
ssh msi "cd /d C:\Users\giand\Desktop\prog\FCGS && venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
echo [MSI] 5/5 Avvio FCGS...
ssh msi "cd /d C:\Users\giand\Desktop\prog\FCGS && start \"\" venv\Scripts\python federated_grid_search.py"
echo [MSI] Fatto.
echo.

:: ── Domino11 ──────────────────────────────────────────────────────────────
echo [Domino11] 1/5 Clone o pull...
ssh domino11 "if exist C:\Users\domin\Desktop\prog\FCGS\.git (cd /d C:\Users\domin\Desktop\prog\FCGS && git pull) else (if not exist C:\Users\domin\Desktop\prog mkdir C:\Users\domin\Desktop\prog && cd /d C:\Users\domin\Desktop\prog && git clone %REPO% FCGS)"
echo [Domino11] 2/5 Creazione venv (se non esiste)...
ssh domino11 "if not exist C:\Users\domin\Desktop\prog\FCGS\venv\Scripts\python.exe (cd /d C:\Users\domin\Desktop\prog\FCGS && python -m venv venv && venv\Scripts\python -m pip install --upgrade pip)"
echo [Domino11] 3/5 Installazione requirements...
ssh domino11 "cd /d C:\Users\domin\Desktop\prog\FCGS && venv\Scripts\pip install -r requirements_windows.txt"
echo [Domino11] 4/5 Installazione PyTorch CUDA 12.6...
ssh domino11 "cd /d C:\Users\domin\Desktop\prog\FCGS && venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
echo [Domino11] 5/5 Avvio FCGS...
ssh domino11 "cd /d C:\Users\domin\Desktop\prog\FCGS && start \"\" venv\Scripts\python federated_grid_search.py"
echo [Domino11] Fatto.
echo.

:: ── Domino12 ──────────────────────────────────────────────────────────────
echo [Domino12] 1/5 Clone o pull...
ssh domino12 "if exist C:\Users\dastl\Desktop\Giando\FCGS\.git (cd /d C:\Users\dastl\Desktop\Giando\FCGS && git pull) else (if not exist C:\Users\dastl\Desktop\Giando mkdir C:\Users\dastl\Desktop\Giando && cd /d C:\Users\dastl\Desktop\Giando && git clone %REPO% FCGS)"
echo [Domino12] 2/5 Creazione venv (se non esiste)...
ssh domino12 "if not exist C:\Users\dastl\Desktop\Giando\FCGS\venv\Scripts\python.exe (cd /d C:\Users\dastl\Desktop\Giando\FCGS && python -m venv venv && venv\Scripts\python -m pip install --upgrade pip)"
echo [Domino12] 3/5 Installazione requirements...
ssh domino12 "cd /d C:\Users\dastl\Desktop\Giando\FCGS && venv\Scripts\pip install -r requirements_windows.txt"
echo [Domino12] 4/5 Installazione PyTorch CUDA 12.6...
ssh domino12 "cd /d C:\Users\dastl\Desktop\Giando\FCGS && venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
echo [Domino12] 5/5 Avvio FCGS...
ssh domino12 "cd /d C:\Users\dastl\Desktop\Giando\FCGS && start \"\" venv\Scripts\python federated_grid_search.py"
echo [Domino12] Fatto.
echo.

echo ==========================================
echo  Setup e avvio completati su tutti i PC.
echo ==========================================
pause
