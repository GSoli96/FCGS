"""
download_datasets.py — Scarica tutti i dataset mancanti/incompleti con progress bar.

Uso:
    python download_datasets.py                          # usa config del PC corrente
    python download_datasets.py --config <file.json>     # config esplicita
    python download_datasets.py --force                  # riscarica anche dataset già OK
    python download_datasets.py --check                  # solo verifica, non scarica

Il download è sempre SEQUENZIALE per evitare rate limit HuggingFace (1000 req/5min per token).
Ogni PC dovrebbe avere il proprio token HF per avere quota indipendente:
  - Crea token su https://huggingface.co/settings/tokens (tipo: Read)
  - Salvalo in .hf_token nella cartella FCGS oppure come variabile HF_TOKEN
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.parent  # project root
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


def _count_images(path: str) -> int:
    from dataset_manager import _count_images as _ci
    return _ci(path)


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


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


def _download_one(repo_id: str, token: str, ds: dict, attempt: int = 1, max_retries: int = 5,
                  timeout_s: int = 1800) -> tuple[bool, int]:
    """
    Scarica un singolo dataset con progress bar sui file.
    Ritorna (successo, immagini_finali).
    """
    import concurrent.futures
    from huggingface_hub import snapshot_download

    hf_path = ds['path'].replace('\\', '/')
    expected = ds.get('num_images', 0)
    patterns = [f"{hf_path}/*", f"{hf_path}/**/*"]

    for att in range(1, max_retries + 1):
        try:
            def _do():
                # Disabilita i progress bar interni di HuggingFace (uno per file).
                # Usiamo il nostro progress bar unico per dataset.
                try:
                    import huggingface_hub.utils as _hfu
                    _hfu.disable_progress_bars()
                except Exception:
                    pass
                snapshot_download(
                    repo_id=repo_id,
                    repo_type='dataset',
                    token=token,
                    local_dir=str(_SCRIPT_DIR),
                    allow_patterns=patterns,
                    ignore_patterns=['*.gitattributes'],
                    max_workers=4,
                )

            # Progress bar che mostra immagini scaricate durante il download
            start_count = _count_images(hf_path)
            bar = None
            stop_monitor = threading.Event()

            def _monitor():
                nonlocal bar
                if _TQDM_OK and expected > 0:
                    bar = tqdm(total=expected, initial=start_count,
                               desc=f"  {hf_path.split('/')[-1][:30]}",
                               unit='img', leave=False, dynamic_ncols=True)
                    last = start_count
                    while not stop_monitor.is_set():
                        cur = _count_images(hf_path)
                        if cur > last:
                            bar.update(cur - last)
                            last = cur
                        stop_monitor.wait(timeout=2)
                    cur = _count_images(hf_path)
                    if cur > last:
                        bar.update(cur - last)
                    bar.close()

            mon = threading.Thread(target=_monitor, daemon=True)
            mon.start()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do)
                try:
                    fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    stop_monitor.set()
                    mon.join(timeout=3)
                    print(f"    TIMEOUT dopo {timeout_s}s (tentativo {att}/{max_retries})")
                    if att < max_retries:
                        time.sleep(30 * att)
                    continue

            stop_monitor.set()
            mon.join(timeout=3)

            actual = _count_images(hf_path)
            if expected > 0 and actual < expected:
                print(f"    Incompleto: {actual}/{expected} ({actual/expected*100:.1f}%) "
                      f"— tentativo {att}/{max_retries}")
                if att < max_retries:
                    time.sleep(30 * att)
                continue
            return True, actual

        except Exception as e:
            err_str = str(e)
            if '429' in err_str:
                # Rate limit HF: attesa esponenziale — 60s, 120s, 180s, ...
                wait = 60 * att
                print(f"    Rate limit HF (429) — attendo {wait}s (tentativo {att}/{max_retries})")
                print(f"    SUGGERIMENTO: usa un token HF diverso per questo PC")
                time.sleep(wait)
            else:
                print(f"    ERRORE: {e} (tentativo {att}/{max_retries})")
                if att < max_retries:
                    time.sleep(30 * att)

    actual = _count_images(hf_path)
    return False, actual


def main():
    parser = argparse.ArgumentParser(description='Scarica dataset FCGS con progress bar')
    parser.add_argument('--config', type=str, help='Path al config JSON (default: auto)')
    parser.add_argument('--force', action='store_true', help='Riscarica anche dataset già OK')
    parser.add_argument('--check', action='store_true', help='Solo verifica stato, non scarica')
    parser.add_argument('--retries', type=int, default=5, help='Tentativi per dataset (default: 5)')
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout per download in secondi (default: 1800)')
    parser.add_argument('--token', type=str, help='Token HuggingFace (Read). Verrà salvato in .hf_token per usi futuri.')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                        help='Dataset da scaricare in parallelo (default: 1). Con token dedicato per PC puoi usare 2-3.')
    args = parser.parse_args()

    # Salva il token in .hf_token se passato via --token
    if args.token:
        token_file = _SCRIPT_DIR / 'setup' / '.hf_token'
        token_file.write_text(args.token.strip(), encoding='utf-8')
        print(f"Token salvato in {token_file}")

    config_path = Path(args.config) if args.config else _find_config()
    print(f"Config: {config_path.name}")

    config = _load_config(config_path)
    datasets = config.get('datasets', [])
    repo_id = config.get('hf_repo_id', 'Siando/fcgs-datasets')

    if not datasets:
        print("Nessun dataset nel config.")
        return

    token = _get_hf_token()
    if not token:
        print("ERRORE: token HuggingFace non trovato.")
        print("  Usa: python download_datasets.py --token hf_xxxxxxxxxxxx")
        print("  Oppure crea .hf_token nella cartella FCGS")
        print("  Token: https://huggingface.co/settings/tokens (tipo: Read)")
        sys.exit(1)
    print(f"Token HF: {token[:8]}...{token[-4:]}")
    print(f"Repo: {repo_id}")
    print(f"Totale dataset nel config: {len(datasets)}\n")

    # Verifica stato attuale
    results = []
    print("Verifica stato dataset locali...")
    bar = tqdm(datasets, unit='dataset') if _TQDM_OK else datasets
    for ds in bar:
        actual = _count_images(ds['path'])
        expected = ds.get('num_images', 0)
        if expected > 0:
            pct = actual / expected * 100
            ok = actual >= expected * 0.9
        else:
            pct = 100.0 if actual > 0 else 0.0
            ok = actual > 0
        results.append({'ds': ds, 'actual': actual, 'expected': expected, 'ok': ok, 'pct': pct})

    print()
    ok_count = sum(1 for r in results if r['ok'])
    fail_count = len(results) - ok_count
    print(f"{'Dataset':<45} {'Stato':<8} {'Trovate':>8} {'Attese':>8} {'%':>6}")
    print('-' * 80)
    for r in results:
        status = 'OK' if r['ok'] else 'INCOMPLETO'
        print(f"{r['ds']['name']:<45} {status:<8} {r['actual']:>8} {r['expected']:>8} {r['pct']:>5.1f}%")
    print('-' * 80)
    print(f"Totale: {ok_count} OK, {fail_count} da scaricare\n")

    if args.check:
        return

    to_download = [r for r in results if not r['ok'] or args.force]
    if not to_download:
        print("Tutti i dataset sono già completi.")
        return

    n_parallel = args.parallel
    mode = f"{n_parallel} in parallelo" if n_parallel > 1 else "sequenziale"
    print(f"Download di {len(to_download)} dataset ({mode}, hf_xet chunk paralleli per file)...\n")

    success = 0
    failed_names = []
    print_lock = threading.Lock()

    def _download_and_report(r, idx):
        ds = r['ds']
        name = ds['name']
        expected = r['expected']
        with print_lock:
            print(f"  [{idx}/{len(to_download)}] Avvio: {name}")
        ok, actual = _download_one(
            repo_id=repo_id, token=token, ds=ds,
            max_retries=args.retries, timeout_s=args.timeout,
        )
        pct = actual / expected * 100 if expected > 0 else (100.0 if actual > 0 else 0.0)
        with print_lock:
            tag = 'OK  ' if ok else 'FAIL'
            print(f"  {tag} {name}: {actual}/{expected} ({pct:.1f}%)")
        return ok, name

    if n_parallel <= 1:
        for idx, r in enumerate(to_download, 1):
            ok, name = _download_and_report(r, idx)
            if ok:
                success += 1
            else:
                failed_names.append(name)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futures = {ex.submit(_download_and_report, r, idx): r
                       for idx, r in enumerate(to_download, 1)}
            bar = tqdm(as_completed(futures), total=len(futures), unit='dataset') if _TQDM_OK else as_completed(futures)
            for fut in bar:
                ok, name = fut.result()
                if ok:
                    success += 1
                else:
                    failed_names.append(name)

    print(f"\n{'='*60}")
    print(f"Download completato: {success}/{len(to_download)} OK")
    if failed_names:
        print(f"Falliti ({len(failed_names)}): {', '.join(failed_names)}")
        print("\nSuggerimenti:")
        print("  - Usa un token HF diverso per questo PC (rate limit per token)")
        print("  - Aspetta qualche minuto e riprova con: python download_datasets.py")
        print("  - Controlla che il token abbia accesso al repo Siando/fcgs-datasets")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
