# FCGS — Riepilogo sessioni precedenti

## Stato al 2026-06-01 (aggiornato sera)

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

# Comandi completi → PC_Disponibili\Comandi_avvio.txt e comandi_pc.txt
```

---

## Stato esperimenti al 2026-06-01

| PC | Stato | CSV | Note |
|---|---|---|---|
| **MSI** | FERMO — da riavviare | 460 CSV preservati | Killato per HF rate limit. Pronto al riavvio dopo Domino11 |
| **MSIDomino12** | FERMO — da riavviare | 0 | Killato per HF rate limit. Pronto al riavvio dopo Domino11 |
| **MSIDomino11** | **DOWNLOADING** (in corso) | 0 | Download sequenziale di 14 dataset. Avviato alle 12:29 |
| **Alienware** | PRONTO — non avviato | 0 | Venv Python 3.11 installato, repo clonato |
| **geosciences** | Stress test in corso (PID 1197781) | — | Avviato alle 15:09, nohup -u, logs in stress_20260601_150937/ |
| **Siando** | Non monitorato | N/D | |

---

## Fix applicati in questa sessione (2026-06-01) — tutti committati e pushati

### Fix 1 — `_adaptive_startup_timeout()` in `federated_grid_search.py`
- Timeout avvio server adattivo per complessità esperimento

### Fix 2 — Backoff su fallimento
- `sleep(10)` extra dopo run_summary None + `sleep(5)` fine task

### Fix 3 — GPU cleanup tra le run
- `torch.cuda.empty_cache()` + `gc.collect()` dopo ogni run

### Fix 4 — Bootstrap logging
- `log/log_{PC}/bootstrap.log` scritto prima del logger Python in `main()`

### Fix 5 — Timeout per-download
- `_DOWNLOAD_TIMEOUT_S = 1800s` con `concurrent.futures` timeout

### Fix 6 — Messaggi errore dettagliati
- 3 categorie: server_ready_timeout / exitcode non-zero / NaN metriche

### Fix 7 — Download sequenziale su Windows (nuovo)
- `dataset_manager.py`: Windows forza `parallel=False` per evitare
  errno 22 su lock file HuggingFace con download paralleli

### Fix 8 — Logging traceback errno 22 (nuovo)
- `dataset_manager.py`: traceback completo + hint per MAX_PATH su errno 22

---

## Nuovi PC aggiunti

### Alienware (DAIS-RTX-5000-5)
- **GPU**: RTX 5090 (31.5 GB VRAM)
- **RAM**: 64 GB
- **Config**: `grid_search_config_DAIS-RTX-5000-5.json`
- **Homomorphic**: SÌ (Python 3.11 installato)
- **Modelli**: tutti (AlexNet, ConvNet, ResNet18/34/50/101)
- **Worker**: 4
- **Venv**: creato, tenseal + torch cu128 installati

### geosciences (aggiornato)
- **Config**: `grid_search_config_geosciences.json` — rimossa homomorphic
  (Python 3.12 non compatibile con tenseal 0.3.16)
- **Stress test**: `run_progressive_test.py --max-workers 8`
  in esecuzione (PID 1197781, logs in `geosciences_stress_test/`)

---

## Problema HuggingFace rate limit
- Causa: MSI (14 dataset) + Domino12 (5 dataset) scaricavano in parallelo
  → superato limite 1000 req/5min
- Fix applicato: download sequenziale su Windows (Fix 7)
- Soluzione operativa: avviare i PC **uno alla volta**, aspettando che
  i download finiscano prima di avviare il successivo

---

## Comandi avvio corretti (da usare su tutti i PC Windows)

```powershell
# Avvio (da dentro la sessione SSH del PC)
powershell Start-Process -FilePath venv\Scripts\python.exe -ArgumentList federated_grid_search.py -WorkingDirectory <PATH_FCGS> -WindowStyle Hidden

# PID
powershell Get-Content log\log_<PCNAME>\bootstrap.log

# Kill FCGS solo
powershell "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*federated_grid_search*'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}"
```

---

## Cosa fare la prossima sessione

### Quando Domino11 finisce i download
1. Avviare MSI (460 CSV già presenti, riparte da config 461+)
2. Avviare Domino12 (riparte da zero)
3. Avviare Alienware (primo avvio, scaricherà tutti i dataset)
4. Avviare uno alla volta per non fare rate limit HF

### Stress test geosciences
5. Analizzare risultati quando PID 1197781 termina:
   - Report: `geosciences_stress_test/reports/report_*.txt`
   - Log run: `geosciences_stress_test/logs/stress_20260601_150937/run_w*.log`
   - Copiare su Siando con SCP

### Monitoraggio
6. Telegram heartbeat ogni 4h per verificare che i fix funzionino
7. Verificare che Non ci siano più blocchi a 525/4224 su MSI

---

## Note tecniche

- **wmic rimosso su Windows 11 recenti** → usare `Start-Process` per avvio,
  `Get-CimInstance` per kill
- **git su Alienware** via GitHub Desktop:
  `C:\Users\daisl\AppData\Local\GitHubDesktop\app-3.5.11\resources\app\git\cmd\git.exe`
- **Bootstrap log** salvato in `log/log_{PC}/bootstrap.log` con PID del processo
- **geosciences nohup**: usare sempre `python -u` per output non bufferizzato
