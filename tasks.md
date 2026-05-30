# FCGS — Lista problemi da risolvere

## Stato: [x] = completato, [-] = in corso, [ ] = da fare

---

## PRIMA COSA — Log, CSV, Telegram

- [x] **1.1** logs_geosciences analizzati: 320 warning training_size=0 (init, atteso), 12 watchdog, 24 namespace errors. Nessun errore GPU/porta.
- [x] **1.2** "Total training size is 0": warning esplicito in `data_splitter.py` quando `n_train==0`. Root cause: cascading da min_class_count < num_clients.
- [x] **1.3** Watchdog startup timeout fixato: `compute_effective_clients()` in `data_splitter.py` calcola i client effettivi PRIMA di avviare il server → `MIN_NUM_WORKERS` corretto. `run_multiple_clients.py` lancia solo thread con directory esistenti.
- [x] **1.4** Namespace error (`/ is not a connected namespace`): aggiunto try/except + check `sio.connected` in `_on_stop_and_eval` (`federated_client.py:319`). Emit `client_eval` protetta come `client_update`.
- [x] **1.5** `consolidate_results.py`: deduplica per chiave hyperparameter, mantiene best_acc. 1904 righe uniche da 2642 totali (738 duplicati rimossi).
- [x] **1.6** `send_telegram()` scrive ogni messaggio anche nel log (strip HTML).
- [x] **1.7** `_setup_main_logger()` aggiunto in `federated_grid_search.py`.

## SECONDA COSA — Parallelizzazione per PC e struttura directory

- [x] **2.1** `grid_search_config_geosciences.json`: GoogLeNet/AlexNet/ResNet18-50-101, num_clients 2-32, global_epoch 10-30, num_parallel_executions 8
- [x] **2.2** ~~`grid_search_config_MSIDomino11.json`~~: ricreato (2026-05-29) con `num_parallel_executions 1` (batteria).
- [x] **2.3** `grid_search_config_MSIDomino12.json`: ConvNet/AlexNet/ResNet18/ResNet34, num_clients 2-8, num_parallel_executions 3
- [x] **2.4** `grid_search_config_MSI.json`: ConvNet/AlexNet/ResNet18, num_clients 2-4, local_epoch [2], num_parallel_executions 2
- [x] **2.5** `grid_search_config_Siando.json`: ConvNet/AlexNet/ResNet18/ResNet34, num_clients 2-8, num_parallel_executions 2
- [x] **2.10** `grid_search_config_DAIS-RTX-5000-5.json`: **PC non disponibile** (2026-05-29).
- [x] **2.6** Struttura log: `log/log_{PC}/` implementata nel codice
- [x] **2.7** Struttura csv: `csv/csv_{PC}/` implementata nel codice
- [x] **2.8** `base_log_path: "log"` in tutti i JSON, percorsi aggiornati in `federated_grid_search.py`
- [x] **2.9** Glob pattern aggiornato con backward-compatibility (nuova + legacy + all_results.csv)

## TERZA COSA — Path cross-platform, dataset naming e download

- [x] **3.1** `_PROJECT_ROOT = Path(__file__).parent` aggiunto, tutti i path ancorati
- [x] **3.2** `trusted_authority.py`: solo console logger, nessun path relativo — verificato OK
- [x] **3.3** Verificato naming HF: 25.330 file con virgola (`EDSS1,5.png`) — gestiti da `data_splitter.py` (`,` → `_` in split copy).
- [x] **3.4** HF dataset rinominato: 25.330 file `EDSS*,*.png` → `EDSS*_*.png` su HuggingFace.
- [x] **3.5** `dataset_manager.py`: scarica solo i dataset delle configurazioni pending (filtro per PC)
- [x] **3.6** `dataset_manager.py`: parallelizzare con `ThreadPoolExecutor(max_workers=4)` + `check_and_download_datasets_parallel`

## QUARTA COSA — Worker scheduling, zombie cleanup e test

- [x] **4.1** `HEAVY_CLIENT_THRESHOLD = 8` in codice, `heavy_client_threshold` configurabile nel JSON
- [x] **4.2** Meccanismo heavy/normal implementato in `run_grid_search_worker()` con `heavy_lock`, `heavy_running`, `active_normal_workers`
- [x] **4.3** `_cleanup_after_task()` aggiunto in `federated_grid_search.py`: killa zombie server + children (psutil), verifica porta liberata, forza kill se necessario. Chiamata dopo ogni task.
- [x] **4.4** Port retry loop anti-TOCTOU in `run_grid_search_worker`: dopo `find_free_port()`, ri-verifica la porta libera con retry (5 tentativi con backoff).
- [ ] **4.5** Test progressivo su **geosciences** (via SSH): eseguire `geosciences_stress_test/run_progressive_test.py --max-workers 8`. Invia il report generato.
- [x] **4.6** `test_fcgs.py` aggiornato a 83 test (+ 1 skipped su Windows): sezioni A-L + test_pre_split_min_num_workers_reflects_actual_io

## QUINTA COSA — Cleanup repo (sessione 2026-05-28)

- [x] **5.1** Rimossi file runtime: `fcgs.pid`, `monitor.pid`, `.monitor_last_telegram`
- [x] **5.2** Rimosso `geosciences_stress_test.py` (radice) — sostituito da `geosciences_stress_test/`
- [x] **5.3** `.gitignore` aggiornato: `geosciences_stress_test/` escluso dal repo (solo locale su geosciences)
- [x] **5.4** `geosciences_stress_test/` creato localmente con: `run_progressive_test.py`, `README.md`, `grid_search_config_geosciences_stress.json`

---

## FIX SESSIONE 2026-05-29 — Bug critici in esecuzione Siando

- [x] **6.1** `UnicodeEncodeError` in `send_telegram` (`federated_grid_search.py`): `_safe_print()` con fallback su `stdout.buffer.write(utf-8)`.
- [x] **6.2** `FileNotFoundError` in `model_manager.py`: iteratore esplicito + try/except su `next()` in `_run_training_epoch` e `validate()`.
- [x] **6.3** `_copy_images` in `data_splitter.py`: retry fino a 3 volte con backoff se `shutil.copy` fallisce con `OSError`.
- [x] **6.4** Verifica post-split in `data_splitter.py`: conta solo client con `client_i/train/` non vuota (`valid_clients`).
- [x] **6.5** Startup zombie check in `federated_grid_search.py` `main()`.
- [x] **6.6** `FederatedClient.__init__`: rimosso `self.connect_to_server()` da `__init__`.
- [x] **6.7** `run_multiple_clients.py` refactored: crea tutti i `FederatedClient` PRIMA dei thread.
- [x] **6.8** Verifica client post-run in `run_grid_search_worker`.
- [x] **6.9** Test suite aggiornata a 82 test.

---

## SETTIMA COSA — Fix MIN_NUM_WORKERS race condition (sessione 2026-05-29)

- [x] **7.1** `federated_grid_search.py`: rimosso `compute_effective_clients()` (dry-run). Aggiunto `DatasetSplitter.split_dataset()` reale dopo `makedirs` e PRIMA dell'avvio server. Check `os.path.isdir` su dataset prima dello split — skip con warning se mancante.
- [x] **7.2** `run_multiple_clients.py`: aggiunto parametro `already_split=False` a `main()`. Se `True`: salta split, usa `config['MIN_NUM_WORKERS']`.
- [x] **7.3** `test_fcgs.py`: aggiunto `test_pre_split_min_num_workers_reflects_actual_io` in sezione I.

---

## Note diagnostiche dai log (637 file, ~263k righe)

| Tipo errore | Frequenza | PC | Probabile causa | Stato |
|---|---|---|---|---|
| Watchdog startup timeout 600s | 31x | Siando, MSI, geosciences | Client con directory vuote (dataset piccolo vs num_clients) | FIXATO (1.3 + 7.1) |
| Total training size = 0 | 313x | Tutti | min_class_count < num_clients → primo round senza dati aggregati (NORMALE al round 0) | WARNING aggiunto |
| No clients remaining round X | Media | Siando, MSIDomino12 | Conseguenza di training_size=0 | FIXATO indirettamente |
| Namespace connection error | 49x | Siando, geosciences | emit su socket già disconnesso in `_on_stop_and_eval` | FIXATO (1.4) |
| Missing dataset files | Risolto | MSIDomino11 | Path HF erano sbagliati | RISOLTO |

---

## PIANO OPERATIVO — Avvio esperimenti su tutti i PC (sessione 2026-05-31)

### Stato generale

| PC | Stato | CSV prodotti | Note |
|---|---|---|---|
| Siando | ✅ IN ESECUZIONE | 201 (ultimo 31/05 00:33) | 120/16848 al heartbeat 23:31. Lento, ETA 557h |
| MSI | ❌ BLOCCATO | 0 | 2 istanze in conflitto (PID 12160 + 22876). Da killare e riavviare |
| MSIDomino12 | ⏳ DA AVVIARE | — | In attesa che MSI funzioni |
| MSIDomino11 | ⏳ DA AVVIARE | — | In attesa che Domino12 sia avviato |

---

### Fase 1 — Download dataset su Siando ✅ COMPLETATA

- [x] `dataset_slices_sorted_augmented` scaricato
- [x] `dataset_multy_sorted_T1_augmented` scaricato
- [x] `dataset_multy_sorted_T2_augmented` scaricato
- [x] `dataset_multy_sorted_augmented` scaricato
- [x] `dataset_ISIC2` scaricato (con 1 retry per errore tqdm)

---

### Fase 2 — Verifica strutture locali [ ] DA FARE

Per ogni dataset nella config Siando (22 totali), verificare:
- Path esiste e contiene immagini
- Numero sottocartelle == num_classes
- Nessuna cartella anomala (train/, test/, README/)

Dataset con attenzione particolare:
- `dataset_multy_sorted_augmented` (3 classi) — scaricare era incompleto, verificare ora
- `dataset_multy_sorted_T1_augmented` (3 classi) — mancava classe_2
- `dataset_multy_sorted_T2_augmented` (3 classi) — mancava classe_2
- `dataset_ISIC2` (2 classi Benign/Malignant)
- `dataset_slices_sorted_T2` (2 classi, class_1/T2/ già fixato localmente)

**Regola**: se sottocartelle != num_classes → STOP + Telegram + attendi risposta utente.

---

### Fase 3 — Test dataset splitting [ ] DA FARE

Per ogni dataset verificato OK, test split di prova con num_clients=2:
```python
from data_splitter import compute_effective_clients
effective = compute_effective_clients(source_images_dir=dataset_abs_path, num_clients=2)
# Se effective >= 2 → OK; se effective == 0 → PROBLEMA
```

---

### Fase 4 — Fix strutture (se necessario) [PARZIALE]

- [x] `dataset_slices_sorted_T2/class_1/T2/*.png` → `class_1/` (già fatto localmente)
- [ ] Upload fix `dataset_slices_sorted_T2` su HuggingFace (token write da `.hf_token_write`)

---

### Fase 5 — Avvio Siando ✅ COMPLETATA

- [x] Processo avviato (30/05 19:31)
- [x] Telegram "FCGS avviata — Siando" ricevuto
- [x] 201 CSV prodotti, heartbeat ricevuto

---

### Fase 6 — Fix e riavvio MSI ❌ DA FARE (priorità alta)

Problema attuale: 2 istanze Python in conflitto, 0 CSV, `infrastruttura.json` vuoto su MSI.

- [ ] **6a** — Killare entrambe le istanze su MSI:
  ```
  ssh msi "taskkill /F /PID 12160 /PID 22876"
  ```
- [ ] **6b** — Verificare che `infrastruttura.json` su MSI sia valido (attualmente 0 byte → copiarlo da qui)
- [ ] **6c** — Riavviare con una sola istanza venv:
  ```
  ssh msi "powershell -Command Start-Process -FilePath 'C:\Users\giand\Desktop\prog\FCGS\venv\Scripts\python.exe' -ArgumentList 'federated_grid_search.py' -WorkingDirectory 'C:\Users\giand\Desktop\prog\FCGS' -WindowStyle Hidden"
  ```
- [ ] **6d** — Verificare che il log `log/log_MSI/main_*.log` venga creato (entro 5 min)
- [ ] **6e** — Verificare primo heartbeat Telegram da MSI (dopo 4h)

---

### Fase 7 — Avvio Domino12 ⏳ DA FARE (dopo MSI OK)

Prerequisito: MSI ha iniziato gli esperimenti (log mostra avvio worker).

```
ssh domino12 "powershell -Command Start-Process -FilePath 'C:\Users\dastl\Desktop\Giando\FCGS\venv\Scripts\python.exe' -ArgumentList 'federated_grid_search.py' -WorkingDirectory 'C:\Users\dastl\Desktop\Giando\FCGS' -WindowStyle Hidden"
```

- [ ] Verificare log `log/log_MSIDomino12/main_*.log` creato
- [ ] Verificare heartbeat Telegram da Domino12

---

### Fase 8 — Avvio Domino11 ⏳ DA FARE (dopo Domino12 OK)

```
ssh domino11 "powershell -Command Start-Process -FilePath 'C:\Users\domin\Desktop\prog\FCGS\venv\Scripts\python.exe' -ArgumentList 'federated_grid_search.py' -WorkingDirectory 'C:\Users\domin\Desktop\prog\FCGS' -WindowStyle Hidden"
```

- [ ] Verificare log `log/log_MSIDomino11/main_*.log` creato
- [ ] Verificare heartbeat Telegram da Domino11

---

### Regole di sicurezza

- Dataset classi diverse da attese → STOP + Telegram + attendi risposta
- SSH fallisce → STOP + Telegram + attendi risposta
- Split effective=0 → STOP + Telegram + attendi risposta
- Processo crasha → Telegram alert ma continua con altri PC
- Ogni azione rilevante → log console + Telegram

---

### Verifica end-to-end

1. Log `main_*.log` di ogni PC mostra "FCGS avviata" + worker attivi
2. Nessun errore FATAL nei log
3. Telegram heartbeat ogni 4h ricevuti
4. CSV in `csv/csv_<PC>/` vengono popolati
