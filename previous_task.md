# FCGS — Riepilogo sessioni precedenti

## Stato al 2026-06-01 (sera)

---

## Comandi PC

```
# Connessione SSH
ssh giand@msi
ssh dastl@domino12
ssh domin@domino11
ssh daisl@alienware
ssh mldl@geosciences

# Percorsi FCGS per PC
Siando   (locale)  : C:\Users\giand\Desktop\prog\FCGS\
MSI                : C:\Users\giand\Desktop\prog\FCGS\
MSIDomino12        : C:\Users\dastl\Desktop\Giando\FCGS\
MSIDomino11        : C:\Users\domin\Desktop\prog\FCGS\
Alienware          : C:\Users\daisl\Desktop\Giando\FCGS\  (hostname: DAIS-RTX-5000-5)
geosciences        : /home/mldl/itadata/FCGS  (Linux)

# Siando IP (per SCP da altri PC verso Siando)
172.28.1.52  (rete universitaria)
100.125.156.12  (Tailscale VPN)

# Comandi completi → PC_Disponibili\Comandi_avvio.txt e comandi_pc.txt
```

---

## Stato dataset al 2026-06-01 sera

| PC | dataset/ | dbluigi/ | Stato |
|---|---|---|---|
| **Siando** | 83.800 file | 31.590 file | Sorgente |
| **MSI** | copiato da Siando via SCP | copiato | ✓ completo |
| **Domino12** | in copia via SCP (da Siando) | in copia | in corso |
| **Domino11** | in copia via SCP (da Siando) | in copia | in corso |
| **Alienware** | in copia via SCP (da Siando, PID 921) | in copia | in corso |
| **geosciences** | 22/22 dataset al 100% | 22/22 OK | ✓ completo |

### Verifica dataset geosciences (download_datasets.py --check)
```
Tutti i 22 dataset OK al 100% — nessun dataset da scaricare
```

---

## Stato esperimenti al 2026-06-01 sera

| PC | Stato | Note |
|---|---|---|
| **MSI** | **FCGS IN ESECUZIONE** ✓ | Dataset copiati da Siando, avviata manualmente |
| **MSIDomino12** | In attesa completamento copia dataset | Poi ricreare venv + avviare |
| **MSIDomino11** | In attesa completamento copia dataset | Poi ricreare venv + avviare |
| **Alienware** | In attesa completamento copia dataset | Venv Python 3.11 già installato |
| **geosciences** | Dataset 100% OK, pronta per avvio | Venv da ricreare senza tenseal |
| **Siando** | Non monitorato | |

---

## Fix applicati in questa sessione (2026-06-01)

### Fix dataset_manager.py
1. **`_adaptive_startup_timeout()`** — timeout avvio server adattivo
2. **GPU cleanup** — `torch.cuda.empty_cache()` + `gc.collect()` dopo ogni run
3. **Backoff su fallimento** — `sleep(10)` extra dopo run_summary None
4. **Bootstrap logging** — `log/log_{PC}/bootstrap.log` prima del logger Python
5. **Download timeout** — `_DOWNLOAD_TIMEOUT_S=1800s` con ThreadPoolExecutor
6. **Fail reason dettagliato** — 3 categorie di errore Telegram
7. **Download sequenziale su Windows** — forza `parallel=False` per evitare errno 22 HF
8. **Logging traceback errno 22** — hint MAX_PATH + traceback completo
9. **Soglia 90% in `_download_folder`** — verifica completezza dopo download (non solo > 0)

### Nuovo script: download_datasets.py
- Progress bar unico per dataset (tqdm, disabilita HF per-file bars)
- `--token` salva in `.hf_token` per usi futuri
- `--parallel N` per N dataset simultanei
- `--check` per solo verifica stato
- `--force` per riscarica forzata
- Backoff esponenziale su rate limit 429

---

## Token HuggingFace (READ, salvati in .hf_token su ogni PC)

| PC | Token salvato |
|---|---|
| MSI | ✓ `hf_rhGJzYCW...` |
| Domino11 | ✓ `hf_UwkfUcMI...` |
| Domino12 | ✓ `hf_REFmIKmH...` |
| Alienware | ✓ `hf_xJnztApW...` |
| geosciences | ✓ `hf_ZxXuuOeB...` |

---

## Nuovi PC aggiunti

### Alienware (DAIS-RTX-5000-5)
- **GPU**: RTX 5090 (31.5 GB VRAM)
- **RAM**: 64 GB — **CPU**: Intel Ultra 9 285K 24 core
- **Config**: `grid_search_config_DAIS-RTX-5000-5.json`
- **Homomorphic**: SÌ — **Worker**: 4 — **Modelli**: tutti
- **PyTorch**: cu128
- **Git**: via GitHub Desktop (`C:\Users\daisl\AppData\Local\GitHubDesktop\app-3.5.11\resources\app\git\cmd\git.exe`)

---

## Comandi avvio (da dentro il PC, sessione SSH)

```cmd
:: Windows — avvio FCGS
powershell Start-Process -FilePath venv\Scripts\python.exe -ArgumentList federated_grid_search.py -WorkingDirectory %CD% -WindowStyle Hidden

:: Windows — download dataset
venv\Scripts\python.exe download_datasets.py --parallel 2

:: Windows — verifica dataset
venv\Scripts\python.exe download_datasets.py --check

:: Windows — venv da zero (Python 3.11, cu126 per RTX 3070, cu128 per RTX 5090)
rmdir /S /Q venv
py -3.11 -m venv venv
venv\Scripts\pip install -r requirements_windows.txt
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```bash
# Linux (geosciences) — avvio FCGS
mkdir -p log/log_geosciences
nohup python -u federated_grid_search.py > log/log_geosciences/main.log 2>&1 &

# Linux — verifica dataset
python download_datasets.py --check

# Linux — venv da zero (senza tenseal, cu121)
rm -rf venv && python3 -m venv venv && source venv/bin/activate
grep -v "^tenseal" requirements_linux.txt > /tmp/req_geo.txt
pip install -r /tmp/req_geo.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Cosa fare la prossima sessione

1. Verificare completamento copia dataset su Domino12, Domino11, Alienware
2. Ricreare venv su Domino12 e Domino11 (Python 3.11, cu126)
3. Avviare FCGS su Domino12, Domino11, Alienware, geosciences
4. Monitorare Telegram heartbeat — verificare che MSI stia girando correttamente
5. Consolidare CSV quando più PC completano configurazioni
