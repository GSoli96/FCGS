"""
upload_datasets_to_hf.py — Carica dataset/ e dbluigi/ su HuggingFace (riprendibile).

Usa upload_large_folder: parallelizza i worker, e' riprendibile (se interrompi e rilanci
riprende da dove si era fermato). Ideale per dataset da GB.

Uso:
    python upload_datasets_to_hf.py                    # carica tutto
    python upload_datasets_to_hf.py --only dataset     # solo dataset/
    python upload_datasets_to_hf.py --only dbluigi     # solo dbluigi/
    python upload_datasets_to_hf.py --workers 4        # worker paralleli

Token: variabile HF_TOKEN oppure file .hf_token (deve avere permessi write)

Rate limit HuggingFace (piano Free, finestre da 5 minuti):
  - Hub API (commit, list): 1.000 req/5min  → max ~3 req/sec
  - Resolver (CDN download): 5.000 req/5min → max ~17 req/sec
  Con DEFAULT_WORKERS = 4 si restano ampiamente sotto i limiti.
"""
import os
import sys
import argparse
import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.parent  # project root
_TOKEN_FILE = _SCRIPT_DIR / 'setup' / '.hf_token'
DEFAULT_REPO_ID = 'Siando/fcgs-datasets'
UPLOAD_FOLDERS = ['dataset', 'dbluigi', 'dbaltri']
DEFAULT_WORKERS = 4


def get_token(cli_token: str) -> str:
    token = cli_token or os.environ.get('HF_TOKEN', '').strip()
    if not token and _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text().strip()
    if not token:
        print("ERRORE: nessun token HF trovato.")
        print(f"  Usa --token <token>, oppure HF_TOKEN, oppure il file {_TOKEN_FILE.name}")
        sys.exit(1)
    return token


def ensure_repo(api, repo_id: str, token: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type='dataset', token=token)
        print(f"Repository '{repo_id}' presente.")
    except Exception:
        print(f"Creazione repository privato '{repo_id}' ...")
        api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, token=token)
        print(f"Repository creato.")


def count_folder(folder_path: Path):
    files = [f for f in folder_path.rglob('*') if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    return len(files), total_mb


def upload_with_large_folder(api, folders: list, repo_id: str, token: str, num_workers: int) -> None:
    """
    Carica le cartelle specificate caricando dalla root del progetto con allow_patterns.
    upload_large_folder non supporta path_in_repo, quindi partiamo dalla root
    e filtriamo con allow_patterns — cosi' i path nel repo sono identici ai path locali.
    """
    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    patterns = [f"{folder}/**" for folder in folders]

    total_files = sum(count_folder(_SCRIPT_DIR / f)[0] for f in folders if (_SCRIPT_DIR / f).exists())
    total_mb = sum(count_folder(_SCRIPT_DIR / f)[1] for f in folders if (_SCRIPT_DIR / f).exists())

    print(f"Upload: {', '.join(folders)} — {total_files:,} file, {total_mb/1024:.1f} GB")
    print(f"Workers: {num_workers} | Avvio: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"(Se interrompi, rilancia lo stesso comando per riprendere)")
    print("=" * 60)

    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(_SCRIPT_DIR),
        repo_type='dataset',
        allow_patterns=patterns,
        num_workers=num_workers,
        print_report=True,
        print_report_every=300,
    )


def main():
    parser = argparse.ArgumentParser(description='Carica i dataset su HuggingFace.')
    parser.add_argument('--repo', default=DEFAULT_REPO_ID,
                        help=f'ID repository HF (default: {DEFAULT_REPO_ID})')
    parser.add_argument('--token', default=None,
                        help='Token HuggingFace (write)')
    parser.add_argument('--only', default=None, choices=UPLOAD_FOLDERS,
                        help='Carica solo questa cartella (dataset, dbluigi o dbaltri)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Worker paralleli (default: {DEFAULT_WORKERS})')
    args = parser.parse_args()

    token = get_token(args.token)

    from huggingface_hub import HfApi
    # Passa il token all'istanza: upload_large_folder non accetta token come argomento
    api = HfApi(token=token)

    me = api.whoami()
    print(f"Autenticato come: {me['name']}")

    ensure_repo(api, args.repo, token)

    folders = [args.only] if args.only else UPLOAD_FOLDERS

    # Verifica che almeno una cartella esista
    existing = [f for f in folders if (_SCRIPT_DIR / f).exists()]
    if not existing:
        print(f"ERRORE: nessuna delle cartelle {folders} trovata in {_SCRIPT_DIR}")
        sys.exit(1)
    missing = [f for f in folders if f not in existing]
    if missing:
        print(f"ATTENZIONE: cartelle non trovate (saltate): {missing}")

    print(f"\n{'='*60}")
    print(f"Upload in: {args.repo}")
    print(f"Cartelle: {existing}")

    start = datetime.datetime.now()
    upload_with_large_folder(api, existing, args.repo, token, args.workers)

    elapsed = datetime.datetime.now() - start
    print(f"\nUpload completato in {elapsed}.")
    print(f"Repository: https://huggingface.co/datasets/{args.repo}")


if __name__ == '__main__':
    main()
