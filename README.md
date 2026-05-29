# Geosciences Stress Test

Script per il test progressivo del sistema FCGS su `mldl@geosciences` con 32+ client e worker crescenti.

## Obiettivo

Determinare il numero massimo di worker paralleli sostenibili con `num_clients=32` sulla macchina geosciences (4 GPU: 3× A800 40GB + RTX A1000 8GB, GPU1 in blacklist).

## Setup

1. Copia questa directory su geosciences:
   ```bash
   scp -r geosciences_stress_test/ mldl@geosciences:/path/to/FCGS/
   ```

2. Assicurati che il venv sia attivato e il progetto FCGS installato:
   ```bash
   cd /path/to/FCGS
   source venv/bin/activate
   pip install psutil  # se non già presente
   ```

3. Verifica che i dataset siano presenti (o scarica con `python federated_grid_search.py` in test):
   ```bash
   ls dataset/dataset_slices_sorted/
   ls dataset/dataset_slices_sorted_Flair/
   ```

## Esecuzione

Test standard (1 → 6 worker):
```bash
cd /path/to/FCGS
python geosciences_stress_test/run_progressive_test.py
```

Test fino a 8 worker:
```bash
python geosciences_stress_test/run_progressive_test.py --max-workers 8
```

Anteprima senza eseguire (dry-run):
```bash
python geosciences_stress_test/run_progressive_test.py --dry-run
```

Riparti da N worker (se i precedenti sono già OK):
```bash
python geosciences_stress_test/run_progressive_test.py --start-from 3 --max-workers 8
```

## Output

- **Report**: `geosciences_stress_test/reports/report_<timestamp>.txt`
- **Log run**: `geosciences_stress_test/logs/stress_<timestamp>/run_w<N>_c32.log`

## Interpretazione risultati

Il report finale indica:
- Il numero massimo di worker stabili con 32 client
- La probabile causa del primo fallimento:
  - **CUDA OOM**: troppa GPU memory → ridurre `batch_size` o limitare i worker
  - **Porta non disponibile**: race condition sulle porte → già fixata nel codice
  - **Watchdog timeout**: lentezza rete/avvio → aumentare `server_startup_timeout`
  - **Crash/Exception**: bug non gestito → analizzare il log specifico

## Configurazione dello stress test

File: `grid_search_config_geosciences_stress.json`

Contiene solo 2 dataset piccoli (`dataset_slices_sorted` e `dataset_slices_sorted_Flair`)
con `num_clients=32`, `global_epoch=5`, `AlexNet` — sufficiente per il test di stabilità senza sprecare ore su configurazioni pesanti.

Per modificare il numero di client o aggiungere dataset, edita direttamente il JSON.
