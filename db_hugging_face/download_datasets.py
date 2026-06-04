"""
download_datasets.py — Scarica dataset da HuggingFace in parallelo, rispettando i rate limit.

Strategia anti-rate-limit:
  - list_repo_tree() UNA volta per dataset  → poche chiamate Hub API (~5-10 per dataset)
  - Download via URL CDN diretti (/resolve/) → bucket Resolver, molto più generoso
  - Rate limiter interno a 10 req/sec        → sotto il limite Free (16.7 req/sec)
  - Thread pool con N worker paralleli       → default 4

Rate limit HuggingFace (piano Free, finestre da 5 minuti):
  - Hub API (list, commit, search): 1.000 req/5min  → ~3.3 req/sec
  - Resolver (CDN /resolve/):       5.000 req/5min  → ~16.7 req/sec

Uso:
    python download_datasets.py                    # config automatica per PC corrente
    python download_datasets.py --config FILE      # config esplicita
    python download_datasets.py --check            # solo verifica stato locale
    python download_datasets.py --force            # riscarica anche file già presenti
    python download_datasets.py --workers N        # worker paralleli (default: 4)
    python download_datasets.py --token hf_xxx     # token HuggingFace
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

_SCRIPT_DIR = Path(__file__).parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from tqdm import tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False


def _get_hf_token() -> str:
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    for tf in (_SCRIPT_DIR / 'setup' / '.hf_token', _SCRIPT_DIR / '.hf_token'):
        if tf.exists():
            return tf.read_text().strip()
    return ''


def _find_config() -> Path:
    import PCNAME
    pc = PCNAME.name
    candidates = [
        _SCRIPT_DIR / 'configs' / f'grid_search_config_{pc}.json',
        _SCRIPT_DIR / 'configs' / 'grid_search_config.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Nessun config trovato per PC '{pc}'")


def _count_images(path: str) -> int:
    from dataset_manager import _count_images as _ci
    return _ci(path)


class _RateLimiter:
    """Limita le richieste a max `rps` al secondo (condiviso tra tutti i thread)."""
    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._lock = Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# Rate limiter globale: 10 req/sec → 3.000 req/5min (sotto il limite Free di 5.000)
_RESOLVER_LIMITER = _RateLimiter(rps=10)


def _list_remote_files(api, repo_id: str, path_in_repo: str, token: str) -> list:
    """
    Lista tutti i file (RepoFile) sotto path_in_repo nel repo HuggingFace.
    Fa poche chiamate API paginate — non dipende dal numero di file.
    """
    from huggingface_hub.hf_api import RepoFile
    files = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type='dataset',
        path_in_repo=path_in_repo,
        recursive=True,
        token=token,
    ):
        if isinstance(item, RepoFile):
            files.append(item)
    return files


def _download_file(session, url: str, local_path: Path, max_retries: int = 5) -> bool:
    """
    Scarica un file via CDN diretto. Rispetta il rate limiter globale (10 req/sec).
    Usa il bucket Resolver di HF (5.000 req/5min per Free).
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + '.tmp')

    for attempt in range(1, max_retries + 1):
        try:
            _RESOLVER_LIMITER.acquire()
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    wait = 60 * attempt
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=131072):
                        f.write(chunk)
            tmp_path.rename(local_path)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < max_retries:
                time.sleep(5 * attempt)

    return False


def _download_dataset(
    api, repo_id: str, ds: dict, token: str,
    workers: int, force: bool, print_lock: Lock,
) -> tuple[bool, int, int]:
    """
    Scarica un dataset completo.
    Ritorna (successo, file_scaricati, file_totali_remoti).
    """
    import requests

    hf_path = ds['path'].replace('\\', '/')
    name = ds['name']

    with print_lock:
        print(f"  [{name}] listing file remoti...", flush=True)

    try:
        remote_files = _list_remote_files(api, repo_id, hf_path, token)
    except Exception as e:
        with print_lock:
            print(f"  [{name}] ERRORE listing: {e}", flush=True)
        return False, 0, 0

    # Calcola quali file mancano o sono corrotti (size 0)
    to_download = []
    for rf in remote_files:
        local_path = _SCRIPT_DIR / rf.path
        needs_dl = force or not local_path.exists() or local_path.stat().st_size == 0
        if needs_dl:
            to_download.append(rf)

    already = len(remote_files) - len(to_download)
    with print_lock:
        print(
            f"  [{name}] {len(remote_files)} remoti | {already} già presenti | "
            f"{len(to_download)} da scaricare",
            flush=True,
        )

    if not to_download:
        return True, 0, len(remote_files)

    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    ok_count = 0
    fail_count = 0
    bar = tqdm(total=len(to_download), desc=f"  {name[:35]}", unit='file', leave=True) if _TQDM_OK else None
    bar_lock = Lock()

    def _dl(rf):
        url = f"{base_url}/{rf.path}"
        local_path = _SCRIPT_DIR / rf.path
        return _download_file(session, url, local_path), rf.path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_dl, rf): rf for rf in to_download}
        for fut in as_completed(futures):
            ok, path = fut.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                with print_lock:
                    print(f"\n  FAIL: {path}", flush=True)
            if bar:
                with bar_lock:
                    bar.update(1)

    if bar:
        bar.close()

    return fail_count == 0, ok_count, len(remote_files)


def main():
    parser = argparse.ArgumentParser(description='Scarica dataset FCGS da HuggingFace senza rate limit')
    parser.add_argument('--config', type=str, help='Path config JSON (default: auto per PC)')
    parser.add_argument('--force', action='store_true', help='Riscarica anche file già presenti')
    parser.add_argument('--check', action='store_true', help='Solo verifica stato locale, non scarica')
    parser.add_argument('--workers', type=int, default=4, help='Worker paralleli per file (default: 4, Free plan: max ~10 req/sec)')
    parser.add_argument('--token', type=str, help='Token HuggingFace Read. Salvato in setup/.hf_token.')
    args = parser.parse_args()

    if args.token:
        token_file = _SCRIPT_DIR / 'setup' / '.hf_token'
        token_file.write_text(args.token.strip(), encoding='utf-8')
        print(f"Token salvato in {token_file}")

    config_path = Path(args.config) if args.config else _find_config()
    print(f"Config: {config_path.name}")

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    datasets = config.get('datasets', [])
    repo_id = config.get('hf_repo_id', 'Siando/fcgs-datasets')

    if not datasets:
        print("Nessun dataset nel config.")
        return

    token = _get_hf_token()
    if not token:
        print("ERRORE: token HuggingFace non trovato.")
        print("  Usa: python download_datasets.py --token hf_xxxxxxxxxxxx")
        print("  Token Read: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"Token: {token[:8]}...{token[-4:]}")
    print(f"Repo:  {repo_id}")
    print(f"Dataset nel config: {len(datasets)}\n")

    # Verifica stato locale
    print("Verifica stato dataset locali...")
    results = []
    ds_bar = tqdm(datasets, unit='dataset') if _TQDM_OK else datasets
    for ds in ds_bar:
        actual = _count_images(ds['path'])
        expected = ds.get('num_images', 0)
        pct = actual / expected * 100 if expected > 0 else (100.0 if actual > 0 else 0.0)
        ok = (actual >= expected * 0.9) if expected > 0 else (actual > 0)
        results.append({'ds': ds, 'actual': actual, 'expected': expected, 'ok': ok, 'pct': pct})

    print()
    ok_n = sum(1 for r in results if r['ok'])
    print(f"{'Dataset':<45} {'Stato':<10} {'Locali':>8} {'Attese':>8} {'%':>6}")
    print('-' * 80)
    for r in results:
        status = 'OK' if r['ok'] else 'INCOMPLETO'
        print(f"{r['ds']['name']:<45} {status:<10} {r['actual']:>8} {r['expected']:>8} {r['pct']:>5.1f}%")
    print('-' * 80)
    print(f"Totale: {ok_n} OK, {len(results) - ok_n} da scaricare\n")

    if args.check:
        return

    to_dl = [r for r in results if not r['ok'] or args.force]
    if not to_dl:
        print("Tutti i dataset sono già completi.")
        return

    print(f"Download {len(to_dl)} dataset con {args.workers} worker paralleli per file...")
    print("(list_repo_tree = poche chiamate API; download via CDN = nessun rate limit)\n")

    from huggingface_hub import HfApi
    api = HfApi()

    print_lock = Lock()
    total_ok = 0
    failed_names = []

    for i, r in enumerate(to_dl, 1):
        ds = r['ds']
        print(f"[{i}/{len(to_dl)}] {ds['name']}", flush=True)

        ok, downloaded, total_remote = _download_dataset(
            api=api, repo_id=repo_id, ds=ds,
            token=token, workers=args.workers,
            force=args.force, print_lock=print_lock,
        )

        actual_now = _count_images(ds['path'])
        pct = actual_now / ds.get('num_images', 1) * 100

        tag = 'OK  ' if ok else 'FAIL'
        print(f"  {tag} {ds['name']}: {actual_now}/{ds.get('num_images',0)} immagini ({pct:.1f}%)\n", flush=True)

        if ok:
            total_ok += 1
        else:
            failed_names.append(ds['name'])

    print('=' * 60)
    print(f"Download completato: {total_ok}/{len(to_dl)} OK")
    if failed_names:
        print(f"Falliti ({len(failed_names)}): {', '.join(failed_names)}")
        print("\nSuggerimenti:")
        print("  - Riprova con: python download_datasets.py")
        print("  - Aumenta i worker: --workers 16")
        print("  - Verifica accesso al repo con il token")
    print('=' * 60)


if __name__ == '__main__':
    main()
