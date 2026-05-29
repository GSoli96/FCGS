"""
dataset_manager.py — Verifica e scarica automaticamente i dataset mancanti da HuggingFace.

Viene chiamato da federated_grid_search.py prima di avviare la grid search.
Se tutti i dataset sono presenti localmente, non fa nulla.
Se mancano, li scarica dal repository HF privato (in parallelo).

Token read:  variabile d'ambiente HF_TOKEN oppure file .hf_token
Token write: variabile d'ambiente HF_TOKEN_WRITE oppure file .hf_token_write
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).parent
_TOKEN_FILE = _SCRIPT_DIR / '.hf_token'
_TOKEN_WRITE_FILE = _SCRIPT_DIR / '.hf_token_write'
DEFAULT_REPO_ID = 'Siando/fcgs-datasets'


def _get_hf_token() -> Optional[str]:
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text().strip()
    return None


def _get_hf_token_write() -> Optional[str]:
    token = os.environ.get('HF_TOKEN_WRITE', '').strip()
    if token:
        return token
    if _TOKEN_WRITE_FILE.exists():
        return _TOKEN_WRITE_FILE.read_text().strip()
    return _get_hf_token()


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def _dataset_exists(path: str) -> bool:
    """Returns True only if the dataset directory contains actual image files.

    Checks one level deep (dataset_root/class_name/image.jpg) to catch the case
    where a partial HF download created class subdirectories but left them empty.
    """
    p = Path(path) if Path(path).is_absolute() else _SCRIPT_DIR / path
    if not p.exists():
        return False
    try:
        for item in p.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS:
                        return True
            elif item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS:
                return True
        return False
    except PermissionError:
        return False


def _find_missing(datasets: List[Dict]) -> List[Dict]:
    return [ds for ds in datasets if not _dataset_exists(ds['path'])]


def _download_folder(repo_id: str, token: str, hf_path: str) -> bool:
    """Scarica una singola cartella del dataset dal repo HF."""
    from huggingface_hub import snapshot_download
    hf_path = hf_path.replace('\\', '/')
    patterns = [f"{hf_path}/*", f"{hf_path}/**/*"]
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
            local_dir=str(_SCRIPT_DIR),
            allow_patterns=patterns,
            ignore_patterns=['*.gitattributes'],
            max_workers=2,
        )
        return _dataset_exists(hf_path)
    except Exception as e:
        print(f"  [DatasetManager] ERRORE download '{hf_path}': {e}")
        return False


def _download_common(config: Dict, parallel: bool = True, max_parallel: int = 4) -> bool:
    """Logica comune: trova i dataset mancanti e li scarica (sequenziale o parallelo)."""
    datasets = config.get('datasets', [])
    if not datasets:
        return True

    missing = _find_missing(datasets)
    if not missing:
        print(f"[DatasetManager] Tutti i {len(datasets)} dataset presenti localmente. Nessun download necessario.")
        return True

    print(f"[DatasetManager] Dataset mancanti ({len(missing)}/{len(datasets)}):")
    for ds in missing:
        print(f"  - {ds['name']}  ({ds['path']})")

    token = _get_hf_token()
    if not token:
        print("[DatasetManager] ATTENZIONE: token HuggingFace non trovato.")
        print(f"  Imposta HF_TOKEN o crea il file {_TOKEN_FILE.name}")
        print("  I dataset mancanti non verranno scaricati — la grid search potrebbe fallire.")
        return False

    repo_id = config.get('hf_repo_id', DEFAULT_REPO_ID)
    failed = []

    if parallel and len(missing) > 1:
        workers = min(max_parallel, len(missing))
        print(f"[DatasetManager] Scarico {len(missing)} dataset da {repo_id} ({workers} in parallelo)...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download_folder, repo_id, token, ds['path']): ds for ds in missing}
            for future in as_completed(futures):
                ds = futures[future]
                try:
                    ok = future.result()
                except Exception as exc:
                    ok = False
                    print(f"    [FAIL] {ds['name']} exception: {exc}")
                if ok:
                    print(f"    [OK] {ds['name']} scaricato.")
                else:
                    print(f"    [FAIL] {ds['name']} FALLITO.")
                    failed.append(ds['name'])
    else:
        print(f"[DatasetManager] Scarico da {repo_id} ...")
        for ds in missing:
            print(f"  -> {ds['name']} ...")
            ok = _download_folder(repo_id, token, ds['path'])
            if ok:
                print(f"    [OK] {ds['name']} scaricato.")
            else:
                print(f"    [FAIL] {ds['name']} FALLITO.")
                failed.append(ds['name'])

    if failed:
        print(f"[DatasetManager] Download fallito per: {failed}")
        print("  La grid search continuerà ma salterà i dataset mancanti.")
        return False

    print(f"[DatasetManager] Tutti i dataset scaricati correttamente.")
    return True


def check_and_download_datasets(config: Dict) -> bool:
    """Controlla e scarica i dataset mancanti (sequenziale, retrocompatibile)."""
    return _download_common(config, parallel=False)


def check_and_download_datasets_parallel(config: Dict, max_parallel: int = 4) -> bool:
    """Controlla e scarica i dataset mancanti in parallelo (max_parallel dataset contemporanei)."""
    return _download_common(config, parallel=True, max_parallel=max_parallel)
