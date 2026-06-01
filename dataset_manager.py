"""
dataset_manager.py — Verifica e scarica automaticamente i dataset mancanti da HuggingFace.

Viene chiamato da federated_grid_search.py prima di avviare la grid search.
Se tutti i dataset sono presenti localmente, non fa nulla.
Se mancano, li scarica dal repository HF privato (in parallelo).

Token read:  variabile d'ambiente HF_TOKEN oppure file .hf_token
Token write: variabile d'ambiente HF_TOKEN_WRITE oppure file .hf_token_write
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).parent
_TOKEN_FILE = _SCRIPT_DIR / '.hf_token'
_TOKEN_WRITE_FILE = _SCRIPT_DIR / '.hf_token_write'
DEFAULT_REPO_ID = 'Siando/fcgs-datasets'

_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


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


def _log(msg: str, logger: Optional[logging.Logger] = None, level: str = 'info') -> None:
    print(msg)
    if logger:
        getattr(logger, level)(msg)


def _count_images(path: str) -> int:
    """Conta le immagini in un dataset (un livello di profondità: root/classe/img)."""
    p = Path(path) if Path(path).is_absolute() else _SCRIPT_DIR / path
    if not p.exists():
        return 0
    count = 0
    try:
        for item in p.iterdir():
            if item.is_dir():
                count += sum(1 for f in item.iterdir()
                             if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS)
            elif item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS:
                count += 1
    except PermissionError:
        pass
    return count


def _dataset_exists(path: str) -> bool:
    """Returns True only if the dataset directory contains actual image files."""
    return _count_images(path) > 0


def _find_missing(datasets: List[Dict], logger: Optional[logging.Logger] = None) -> List[Dict]:
    """Trova dataset mancanti o incompleti (< 90% del num_images atteso)."""
    missing = []
    for ds in datasets:
        actual = _count_images(ds['path'])
        expected = ds.get('num_images', 0)
        if actual == 0:
            _log(f"  [DatasetManager] {ds['name']}: ASSENTE (0 immagini trovate)", logger)
            missing.append(ds)
        elif expected > 0 and actual < expected * 0.9:
            _log(f"  [DatasetManager] {ds['name']}: INCOMPLETO ({actual}/{expected} immagini, "
                 f"{actual/expected*100:.1f}%) — da ri-scaricare", logger, 'warning')
            missing.append(ds)
        else:
            _log(f"  [DatasetManager] {ds['name']}: OK ({actual} immagini)", logger)
    return missing


_DOWNLOAD_TIMEOUT_S = 1800  # 30 min per singolo dataset — previene hang indefiniti su connessioni stalled


def _download_folder(repo_id: str, token: str, hf_path: str,
                     max_retries: int = 3, logger: Optional[logging.Logger] = None) -> bool:
    """Scarica una singola cartella del dataset dal repo HF, con retry e timeout per-download."""
    import errno
    import time
    import traceback
    import concurrent.futures
    from huggingface_hub import snapshot_download
    hf_path = hf_path.replace('\\', '/')
    patterns = [f"{hf_path}/*", f"{hf_path}/**/*"]

    def _do_download():
        snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
            local_dir=str(_SCRIPT_DIR),
            allow_patterns=patterns,
            ignore_patterns=['*.gitattributes'],
            max_workers=1,
        )

    for attempt in range(1, max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(_do_download)
                try:
                    _fut.result(timeout=_DOWNLOAD_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    _log(f"  [DatasetManager] TIMEOUT download '{hf_path}' dopo {_DOWNLOAD_TIMEOUT_S}s "
                         f"(tentativo {attempt}/{max_retries}).", logger, 'error')
                    if attempt < max_retries:
                        wait = 30 * attempt
                        _log(f"  [DatasetManager] Attendo {wait}s prima di riprovare...", logger)
                        time.sleep(wait)
                    continue
            if _dataset_exists(hf_path):
                return True
            _log(f"  [DatasetManager] '{hf_path}': download completato ma nessuna immagine "
                 f"(tentativo {attempt}/{max_retries}).", logger, 'warning')
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"  [DatasetManager] ERRORE download '{hf_path}' "
                 f"(tentativo {attempt}/{max_retries}): {e}", logger, 'error')
            # errno 22 su Windows = path troppo lungo (MAX_PATH 260) o caratteri non validi nel nome file.
            # Hint utile per debug: abilitare i long paths con:
            #   reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
            if isinstance(e, OSError) and e.errno == errno.EINVAL:
                _log(f"  [DatasetManager] HINT [Errno 22]: probabile path troppo lungo (MAX_PATH Windows) "
                     f"o carattere non valido nel nome file del dataset.\n"
                     f"  Traceback completo:\n{tb}", logger, 'error')
            else:
                _log(f"  [DatasetManager] Traceback:\n{tb}", logger, 'error')
        if attempt < max_retries:
            wait = 30 * attempt
            _log(f"  [DatasetManager] Attendo {wait}s prima di riprovare...", logger)
            time.sleep(wait)
    return False


def _download_common(config: Dict, parallel: bool = True, max_parallel: int = 4,
                     logger: Optional[logging.Logger] = None) -> bool:
    """Logica comune: trova i dataset mancanti/incompleti e li scarica."""
    import platform
    datasets = config.get('datasets', [])
    # On Windows, concurrent HF snapshot_download calls collide on lock/cache files
    # producing errno 22 (Invalid argument). Force sequential to avoid this.
    if platform.system() == 'Windows' and parallel and max_parallel > 1:
        _log("[DatasetManager] Windows rilevato: download sequenziale forzato (parallel=1 per evitare errno 22 su lock HF).", logger, 'warning')
        parallel = False
    if not datasets:
        return True

    _log("[DatasetManager] Verifica dataset locali...", logger)
    missing = _find_missing(datasets, logger)
    if not missing:
        _log(f"[DatasetManager] Tutti i {len(datasets)} dataset presenti e completi.", logger)
        return True

    _log(f"[DatasetManager] Dataset da scaricare ({len(missing)}/{len(datasets)}):", logger, 'warning')
    for ds in missing:
        _log(f"  - {ds['name']} ({ds['path']})", logger, 'warning')

    token = _get_hf_token()
    if not token:
        _log("[DatasetManager] ATTENZIONE: token HuggingFace non trovato. "
             f"Imposta HF_TOKEN o crea {_TOKEN_FILE.name}", logger, 'error')
        return False

    repo_id = config.get('hf_repo_id', DEFAULT_REPO_ID)
    failed = []

    if parallel and len(missing) > 1:
        workers = min(max_parallel, len(missing))
        _log(f"[DatasetManager] Scarico {len(missing)} dataset da {repo_id} "
             f"({workers} in parallelo)...", logger)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download_folder, repo_id, token, ds['path'], 3, logger): ds
                       for ds in missing}
            for future in as_completed(futures):
                ds = futures[future]
                try:
                    ok = future.result()
                except Exception as exc:
                    ok = False
                    _log(f"    [FAIL] {ds['name']} exception: {exc}", logger, 'error')
                if ok:
                    _log(f"    [OK] {ds['name']} scaricato.", logger)
                else:
                    _log(f"    [FAIL] {ds['name']} FALLITO.", logger, 'error')
                    failed.append(ds['name'])
    else:
        _log(f"[DatasetManager] Scarico da {repo_id} (sequenziale)...", logger)
        for ds in missing:
            _log(f"  -> {ds['name']} ...", logger)
            ok = _download_folder(repo_id, token, ds['path'], 3, logger)
            if ok:
                _log(f"    [OK] {ds['name']} scaricato.", logger)
            else:
                _log(f"    [FAIL] {ds['name']} FALLITO.", logger, 'error')
                failed.append(ds['name'])

    if failed:
        _log(f"[DatasetManager] Download fallito per: {failed}", logger, 'error')
        return False

    _log("[DatasetManager] Tutti i dataset scaricati correttamente.", logger)
    return True


def check_and_download_datasets(config: Dict, logger: Optional[logging.Logger] = None) -> bool:
    """Controlla e scarica i dataset mancanti (sequenziale, retrocompatibile)."""
    return _download_common(config, parallel=False, logger=logger)


def check_and_download_datasets_parallel(config: Dict, max_parallel: int = 4,
                                         logger: Optional[logging.Logger] = None) -> bool:
    """Controlla e scarica i dataset mancanti/incompleti in parallelo."""
    return _download_common(config, parallel=True, max_parallel=max_parallel, logger=logger)
