"""
consolidate_results.py — Consolida e deduplicazione i CSV di tutti i PC in all_results.csv.

Per ogni gruppo di configurazioni identiche (stesso dataset, modello, hyperparametri),
mantiene solo la riga con il best_acc più alto.

Uso:
    python consolidate_results.py
    python consolidate_results.py --dry-run   # solo report, non scrive
"""
import csv
import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet

_PROJECT_ROOT = Path(__file__).parent

DEDUP_KEYS = [
    'dataset_name', 'model_name', 'num_clients', 'aggregation_algorithm',
    'global_epoch', 'local_epoch', 'batch_size', 'image_size', 'encryption_mode',
    'learning_rate', 'models_percentage', 'fedprox_mu', 'he_backend',
    'convnet_hidden1', 'convnet_hidden2', 'num_custom_layers',
]

METRIC_KEY = 'best_acc'


def _row_key(row: Dict) -> FrozenSet[Tuple[str, str]]:
    items = []
    for k in DEDUP_KEYS:
        v = row.get(k, '')
        if v is None:
            v = ''
        items.append((k, str(v).strip()))
    return frozenset(items)


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def load_csv(path: str) -> List[Dict]:
    rows = []
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        print(f"  [WARN] Impossibile leggere {path}: {e}")
    return rows


def find_all_csvs() -> List[str]:
    found = []
    # Nuova struttura: csv/csv_{PC}/
    for p in glob.glob(str(_PROJECT_ROOT / 'csv' / 'csv_*' / 'federated_grid_search_results_*.csv')):
        found.append(p)
    # Legacy: csv_{PC}/{PC}/
    for p in glob.glob(str(_PROJECT_ROOT / 'csv_*' / '*' / 'federated_grid_search_results_*.csv')):
        if p not in found:
            found.append(p)
    # all_results.csv corrente
    ar = str(_PROJECT_ROOT / 'all_results.csv')
    if os.path.exists(ar) and ar not in found:
        found.append(ar)
    return found


def consolidate(dry_run: bool = False) -> None:
    csv_files = find_all_csvs()
    if not csv_files:
        print("Nessun CSV trovato.")
        return

    print(f"CSV trovati ({len(csv_files)}):")
    all_rows: List[Dict] = []
    for f in csv_files:
        rows = load_csv(f)
        print(f"  {f}: {len(rows)} righe")
        all_rows.extend(rows)

    print(f"\nTotale righe caricate: {len(all_rows)}")

    # Deduplicazione: per ogni chiave, mantieni la riga con best_acc più alto
    best: Dict[FrozenSet, Dict] = {}
    for row in all_rows:
        key = _row_key(row)
        existing = best.get(key)
        if existing is None or _safe_float(row.get(METRIC_KEY)) > _safe_float(existing.get(METRIC_KEY)):
            best[key] = row

    unique_rows = list(best.values())
    print(f"Righe uniche dopo deduplicazione: {len(unique_rows)}")
    print(f"Duplicati rimossi: {len(all_rows) - len(unique_rows)}")

    if dry_run:
        print("\n[dry-run] Nessun file scritto.")
        return

    # Determina tutte le colonne
    all_keys: set = set()
    for row in unique_rows:
        all_keys.update(k for k in row.keys() if k is not None)
    fieldnames = sorted(all_keys, key=lambda x: x or '')

    out_path = str(_PROJECT_ROOT / 'all_results.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

    print(f"\nall_results.csv aggiornato: {len(unique_rows)} righe -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolida i CSV di tutti i PC in all_results.csv')
    parser.add_argument('--dry-run', action='store_true', help='Solo report, non scrive')
    args = parser.parse_args()
    consolidate(dry_run=args.dry_run)
