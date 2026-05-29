"""
fix_hf_filenames.py — Rinomina i file con virgole nel nome (es. EDSS1,5.png -> EDSS1_5.png)
sia in locale che sul repository HuggingFace.

Uso tipico:
    python fix_hf_filenames.py --dry-run         # solo anteprima
    python fix_hf_filenames.py --local-only      # solo rename locale (senza HF)
    python fix_hf_filenames.py                   # rename locale + fix HF

Se il rename locale e' gia' stato fatto (file locali gia' senza virgola) ma HF
non e' stato aggiornato (token errato la prima volta):
    python fix_hf_filenames.py --hf-only         # solo fix HF, salta rename locale

Il token write deve stare in .hf_token_write (o env HF_TOKEN_WRITE).
Crealo su: https://huggingface.co/settings/tokens  (tipo: Write)
"""
import os
import sys
import argparse
from pathlib import Path
import re

_SCRIPT_DIR = Path(__file__).parent
_TOKEN_WRITE_FILE = _SCRIPT_DIR / '.hf_token_write'
_TOKEN_FILE = _SCRIPT_DIR / '.hf_token'
DEFAULT_REPO_ID = 'Siando/fcgs-datasets'
BATCH_SIZE = 250  # ~102 commit totali per 25330 file — ben sotto il rate limit HF di 128/ora


def get_write_token() -> str:
    tok = os.environ.get('HF_TOKEN_WRITE', '').strip()
    if not tok and _TOKEN_WRITE_FILE.exists():
        tok = _TOKEN_WRITE_FILE.read_text().strip()
    if not tok:
        print("ERRORE: token HF write non trovato.")
        print("  1. Vai su https://huggingface.co/settings/tokens")
        print("  2. Crea un token con tipo 'Write'")
        print("  3. Salvalo in .hf_token_write")
        sys.exit(1)
    # Verifica che non sia lo stesso del read token (probabilmente read-only)
    if _TOKEN_FILE.exists():
        read_tok = _TOKEN_FILE.read_text().strip()
        if tok == read_tok:
            print("ATTENZIONE: .hf_token_write contiene lo stesso token di .hf_token.")
            print("  Probabilmente e' un token read-only. Potrebbe dare 403.")
    return tok


def find_local_comma_files(base_dirs=None) -> list:
    """Trova tutti i file con virgola nel nome. Ritorna lista di (old_path, new_path)."""
    if base_dirs is None:
        base_dirs = [_SCRIPT_DIR / 'dataset', _SCRIPT_DIR / 'dbluigi', _SCRIPT_DIR / 'dbaltri']
    files = []
    for base in base_dirs:
        if not Path(base).exists():
            continue
        for f in Path(base).rglob('*'):
            if f.is_file() and ',' in f.name:
                new_name = re.sub(r',', '_', f.name)
                new_path = f.parent / new_name
                files.append((f, new_path))
    return files


def find_hf_comma_files(repo_id: str, token: str) -> list:
    """
    Lista i file con virgola nel nome sul repo HF.
    Ritorna lista di (hf_old_path, hf_new_path, local_new_path).
    local_new_path e' None se il file locale rinominato non esiste.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print("Listing file su HF (potrebbe richiedere 30-60s)...")
    all_files = list(api.list_repo_files(repo_id, repo_type='dataset', token=token))
    comma_files = []
    for f in all_files:
        p = Path(f)
        if ',' in p.name:
            new_name = re.sub(r',', '_', p.name)
            hf_new = str(p.parent / new_name).replace('\\', '/')
            local_new = _SCRIPT_DIR / str(p.parent / new_name)
            comma_files.append((f, hf_new, local_new if local_new.exists() else None))
    return comma_files


def rename_local_files(pairs: list) -> list:
    """Rinomina i file locali. Ritorna lista di (new_path, old_path) per i riusciti."""
    renamed = []
    errors = 0
    for old_path, new_path in pairs:
        try:
            old_path.rename(new_path)
            renamed.append((new_path, old_path))
        except OSError as e:
            print(f"  [ERRORE rename] {old_path.name}: {e}")
            errors += 1
    if errors:
        print(f"  {errors} file non rinominati.")
    return renamed


def fix_hf_in_batches(pairs: list, repo_id: str, token: str) -> bool:
    """
    Carica i file rinominati su HF e cancella i vecchi, in batch.
    pairs: lista di (new_local_path, old_local_path)
           oppure (new_local_path_or_None, old_hf_path_str, new_hf_path_str)
    Ritorna True se tutti i batch riescono.
    """
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

    api = HfApi(token=token)

    # Normalizza: accetta sia (new_local, old_local) che (new_local, old_hf, new_hf)
    operations_add = []
    operations_del = []
    skipped = 0
    for item in pairs:
        if len(item) == 2:
            new_local, old_local = item
            old_hf = str(Path(old_local).relative_to(_SCRIPT_DIR)).replace('\\', '/')
            new_hf = str(Path(new_local).relative_to(_SCRIPT_DIR)).replace('\\', '/')
        else:
            new_local, old_hf, new_hf = item
            if new_local is None:
                print(f"  [SKIP] file locale non trovato per: {old_hf}")
                skipped += 1
                continue

        if not Path(new_local).exists():
            print(f"  [SKIP] file locale mancante: {new_local}")
            skipped += 1
            continue

        operations_add.append(CommitOperationAdd(
            path_in_repo=new_hf,
            path_or_fileobj=str(new_local)
        ))
        operations_del.append(CommitOperationDelete(path_in_repo=old_hf))

    total = len(operations_add)
    if skipped:
        print(f"  {skipped} file saltati (mancanti in locale).")
    if total == 0:
        print("Nessuna operazione HF da eseguire.")
        return True

    print(f"Operazioni HF: {total} upload + {total} delete, batch da {BATCH_SIZE}...")
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    failed_batches = []

    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_add = operations_add[i:i+BATCH_SIZE]
        batch_del = operations_del[i:i+BATCH_SIZE]
        msg = f"Fix: rename comma->underscore in filenames (batch {batch_num}/{n_batches})"
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=batch_add + batch_del,
                commit_message=msg,
            )
            print(f"  Batch {batch_num}/{n_batches} OK ({len(batch_add)} file)")
        except Exception as e:
            print(f"  [ERRORE] batch {batch_num}/{n_batches}: {e}")
            failed_batches.append(batch_num)

    if failed_batches:
        print(f"\nBatch falliti: {failed_batches}")
        print("Rilancia il comando per riprendere (i batch gia' completati su HF non vengono ripetuti).")
        return False

    print(f"\nTutti i {n_batches} batch completati su HF.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix virgole nei filename (locale + HF)')
    parser.add_argument('--dry-run', action='store_true', help='Solo anteprima, non rinomina')
    parser.add_argument('--local-only', action='store_true', help='Solo rename locale, non tocca HF')
    parser.add_argument('--hf-only', action='store_true',
                        help='Solo fix HF (file locali gia\' rinominati)')
    parser.add_argument('--repo', default=DEFAULT_REPO_ID)
    parser.add_argument('--dirs', nargs='+', default=None, help='Cartelle da scannerizzare')
    args = parser.parse_args()

    if args.hf_only:
        # --- Modalita' HF-only: file locali gia' rinominati ---
        print("Modalita' --hf-only: cerco file con virgola direttamente su HF...")
        token = get_write_token()
        hf_pairs = find_hf_comma_files(args.repo, token)
        print(f"File con virgola su HF: {len(hf_pairs)}")
        if not hf_pairs:
            print("Nessun fix necessario su HF.")
            return
        if args.dry_run:
            for old_hf, new_hf, local in hf_pairs[:10]:
                status = "OK locale" if local else "MANCANTE"
                print(f"  {old_hf}  ->  {Path(new_hf).name}  [{status}]")
            if len(hf_pairs) > 10:
                print(f"  ... e altri {len(hf_pairs)-10}")
            return
        ok = fix_hf_in_batches(
            [(local, old_hf, new_hf) for old_hf, new_hf, local in hf_pairs],
            args.repo, token
        )
        sys.exit(0 if ok else 1)
        return

    # --- Modalita' normale: rename locale + fix HF ---
    base_dirs = [_SCRIPT_DIR / d for d in args.dirs] if args.dirs else None
    pairs = find_local_comma_files(base_dirs)
    print(f"File con virgola in locale: {len(pairs)}")

    if not pairs:
        print("Nessun file locale con virgola trovato.")
        if not args.local_only:
            print("Se HF non e' aggiornato, usa: python fix_hf_filenames.py --hf-only")
        return

    if args.dry_run:
        print("Esempi (prime 10):")
        for old, new in pairs[:10]:
            print(f"  {old.relative_to(_SCRIPT_DIR)}  ->  {new.name}")
        if len(pairs) > 10:
            print(f"  ... e altri {len(pairs)-10}")
        return

    print(f"Rinomina locale di {len(pairs)} file...")
    renamed = rename_local_files(pairs)
    print(f"Rinominati {len(renamed)} file in locale.")

    if args.local_only:
        print("--local-only: HF non modificato.")
        print("Done.")
        return

    token = get_write_token()
    ok = fix_hf_in_batches(renamed, args.repo, token)
    if ok:
        print("Fix HF completato.")
    else:
        print("Fix HF parziale. Rilancia per completare.")
    print("Done.")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
