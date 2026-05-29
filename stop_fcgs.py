"""
stop_fcgs.py — Ferma la grid search in modo pulito uccidendo l'intero albero di processi.

Uso:
    python stop_fcgs.py          # ferma il processo che ha scritto fcgs.pid
    python stop_fcgs.py <PID>    # ferma un PID specifico

Perche' serve questo script:
    Stop-Process -Force (SIGKILL) uccide solo il main process.
    I worker (multiprocessing.Process) diventano orfani perche' non sono daemon
    (non possono esserlo: creano a loro volta subprocessi per i server Flask).
    taskkill /F /T uccide il processo E tutto il suo albero di figli.
"""
import sys
import subprocess
import os

PID_FILE = 'fcgs.pid'


def kill_tree(pid: int) -> None:
    if sys.platform == 'win32':
        result = subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True, text=True
        )
        print(result.stdout.strip() or result.stderr.strip())
    else:
        # Linux/macOS: uccidi il process group
        try:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"Inviato SIGTERM al process group di PID {pid}.")
        except Exception as e:
            print(f"Errore: {e}")


def main():
    if len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
        except ValueError:
            print(f"PID non valido: {sys.argv[1]}")
            sys.exit(1)
    elif os.path.exists(PID_FILE):
        pid = int(open(PID_FILE).read().strip())
        print(f"PID letto da {PID_FILE}: {pid}")
    else:
        print(f"Nessun PID fornito e {PID_FILE} non trovato.")
        print("Uso: python stop_fcgs.py <PID>")
        sys.exit(1)

    print(f"Terminazione albero processi PID {pid}...")
    kill_tree(pid)
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


if __name__ == '__main__':
    main()
