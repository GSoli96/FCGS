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
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent

try:
    from tqdm import tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False


def _get_hf_token() -> str:
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    tf = _SCRIPT_DIR / '.hf_token'
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
        _SCRIPT_DIR / f'grid_search_config_{pc}.json',
        _SCRIPT_DIR / 'grid_search_config.json',
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
                # hf_xet è usato automaticamente da snapshot_download quando installato.
                # Xet scarica i file in chunk paralleli → molto più veloce di HTTPS standard.
                # max_workers=4 per i chunk Xet interni (non download paralleli di file diversi).
                snapshot_download(
                    repo_id=repo_id,
                    repo_type='dataset',
                    token=token,
                    local_dir=str(_SCRIPT_DIR),
                    allow_patterns=patterns,
                    ignore_patterns=['*.gitattributes'],
                    max_workers=4,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do)
                try:
                    fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    print(f"    TIMEOUT dopo {timeout_s}s (tentativo {att}/{max_retries})")
                    if att < max_retries:
                        time.sleep(30 * att)
                    continue

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
    args = parser.parse_args()

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
        print("  Crea .hf_token nella cartella FCGS oppure imposta HF_TOKEN=...")
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

    print(f"Download di {len(to_download)} dataset (sequenziale per evitare rate limit HF)...\n")

    success = 0
    failed_names = []

    outer_bar = tqdm(to_download, unit='dataset', desc='Totale') if _TQDM_OK else to_download
    for r in outer_bar:
        ds = r['ds']
        name = ds['name']
        expected = r['expected']

        if _TQDM_OK:
            outer_bar.set_description(f"{name[:35]}")
        else:
            print(f"\n[{to_download.index(r)+1}/{len(to_download)}] {name} ({r['actual']}/{expected})")

        ok, actual = _download_one(
            repo_id=repo_id,
            token=token,
            ds=ds,
            max_retries=args.retries,
            timeout_s=args.timeout,
        )

        if ok:
            success += 1
            pct = actual / expected * 100 if expected > 0 else 100.0
            print(f"  OK  {name}: {actual}/{expected} ({pct:.1f}%)")
        else:
            failed_names.append(name)
            pct = actual / expected * 100 if expected > 0 else 0.0
            print(f"  FAIL {name}: {actual}/{expected} ({pct:.1f}%) — INCOMPLETO dopo {args.retries} tentativi")

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
