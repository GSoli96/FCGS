"""
consolidate_results.py — Consolida e deduplica i CSV di tutti i PC in all_results.csv.

Per ogni gruppo di configurazioni identiche (stesso dataset, modello, hyperparametri),
mantiene solo la riga con il best_acc più alto. Scarta automaticamente i risultati
inattendibili (best_round <= 0 o total_duration < 60s) e la cartella result_DELL/csv_MSI
(data leakage rilevato: 51% dei run a 100% accuracy).

Uso:
    python consolidate_results.py              # aggrega e scrive all_results.csv
    python consolidate_results.py --dry-run    # solo report, non scrive
    python consolidate_results.py --cleanup    # dopo aver scritto, elimina le cartelle per-PC in results/
"""
import csv
import os
import glob
import shutil
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

# Cartelle da escludere perché i risultati non sono attendibili.
# result_DELL/csv_MSI: 51% dei run a 100% accuracy su tutti i modelli/dataset → data leakage.
EXCLUDE_PATHS = [
    'result_DELL/csv/csv_MSI',
]


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


def _is_reliable(row: Dict) -> bool:
    """Scarta righe dove il training non è effettivamente avvenuto."""
    try:
        best_round = int(float(row.get('best_round', -1) or -1))
    except (ValueError, TypeError):
        best_round = -1
    try:
        duration = float(row.get('total_duration', 0) or 0)
    except (ValueError, TypeError):
        duration = 0
    return best_round > 0 and duration >= 60


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
    normalized_excludes = [e.replace('/', os.sep) for e in EXCLUDE_PATHS]

    # Tutti i federated_grid_search_results_*.csv in qualsiasi sottocartella di results/
    for p in glob.glob(str(_PROJECT_ROOT / 'results' / '**' / 'federated_grid_search_results_*.csv'),
                       recursive=True):
        if not any(excl in p for excl in normalized_excludes):
            found.append(p)

    # Tutti gli all_results*.csv dentro results/ (aggregazioni intermedie per-PC)
    for p in glob.glob(str(_PROJECT_ROOT / 'results' / '**' / 'all_results*.csv'), recursive=True):
        if p not in found:
            found.append(p)

    # all_results.csv nella root (storico consolidato precedente)
    root_ar = str(_PROJECT_ROOT / 'all_results.csv')
    if os.path.exists(root_ar) and root_ar not in found:
        found.append(root_ar)

    return found


def consolidate(dry_run: bool = False, cleanup: bool = False) -> None:
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

    # Filtro affidabilità
    before_filter = len(all_rows)
    all_rows = [r for r in all_rows if _is_reliable(r)]
    print(f"Righe scartate (best_round<=0 o duration<60s): {before_filter - len(all_rows)}")
    print(f"Righe dopo filtro affidabilità: {len(all_rows)}")

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

    if cleanup:
        import stat

        def _force_remove(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        results_dir = _PROJECT_ROOT / 'results'
        if results_dir.is_dir():
            print(f"\nElimino le sottocartelle per-PC in {results_dir}...")
            for item in results_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, onerror=_force_remove)
                    print(f"  Eliminata: {item}")
            print("Cleanup completato.")
        else:
            print("Cartella results/ non trovata, nulla da eliminare.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Consolida i CSV di tutti i PC in all_results.csv')
    parser.add_argument('--dry-run', action='store_true', help='Solo report, non scrive')
    parser.add_argument('--cleanup', action='store_true',
                        help='Elimina le cartelle per-PC in results/ dopo la consolidazione')
    args = parser.parse_args()

    if args.cleanup and args.dry_run:
        print("[WARN] --cleanup ignorato in modalità --dry-run.")

    consolidate(dry_run=args.dry_run, cleanup=args.cleanup and not args.dry_run)
