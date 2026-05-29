"""
run_progressive_test.py — Stress test progressivo per geosciences.

Esegue configurazioni con 32 client aumentando progressivamente il numero di worker
paralleli (1 -> 2 -> 3 -> ...) fino al primo errore. Monitora CPU, RAM, GPU e porte
per diagnosticare la causa del crash.

Uso:
    python run_progressive_test.py                    # default: fino a 6 worker
    python run_progressive_test.py --max-workers 8    # testa fino a 8 worker
    python run_progressive_test.py --clients 32       # num_clients (default: 32)
    python run_progressive_test.py --dry-run          # mostra il piano senza eseguire

Report finale salvato in: geosciences_stress_test/report_<timestamp>.txt
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CONFIG_STRESS = _SCRIPT_DIR / "grid_search_config_geosciences_stress.json"
_LOG_DIR = _SCRIPT_DIR / "logs"
_REPORT_DIR = _SCRIPT_DIR / "reports"


def _try_import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        print("[WARNING] psutil non disponibile. Installa con: pip install psutil")
        return None


def _gpu_stats():
    """Restituisce statistiche GPU via nvidia-smi (formato CSV)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": parts[0], "name": parts[1],
                        "mem_used_mb": parts[2], "mem_total_mb": parts[3],
                        "util_pct": parts[4], "temp_c": parts[5]
                    })
            return gpus
    except Exception:
        pass
    return []


def _system_snapshot(psutil_mod):
    """Snapshot CPU/RAM/GPU del sistema."""
    snap = {"ts": datetime.datetime.now().isoformat()}
    if psutil_mod:
        snap["cpu_pct"] = psutil_mod.cpu_percent(interval=0.5)
        vm = psutil_mod.virtual_memory()
        snap["ram_used_gb"] = round(vm.used / 1024**3, 2)
        snap["ram_total_gb"] = round(vm.total / 1024**3, 2)
        snap["ram_pct"] = vm.percent

        # Porte in ascolto
        try:
            conns = psutil_mod.net_connections(kind="tcp")
            listening = [c.laddr.port for c in conns if c.status == "LISTEN"]
            snap["listening_ports_count"] = len(listening)
        except Exception:
            snap["listening_ports_count"] = -1

    gpus = _gpu_stats()
    snap["gpus"] = gpus
    return snap


def _format_snapshot(snap):
    lines = [f"  Timestamp: {snap.get('ts', '?')}"]
    if "cpu_pct" in snap:
        lines.append(f"  CPU: {snap['cpu_pct']}%  |  "
                     f"RAM: {snap['ram_used_gb']}/{snap['ram_total_gb']} GB ({snap['ram_pct']}%)")
    if "listening_ports_count" in snap:
        lines.append(f"  Porte in ascolto (TCP): {snap['listening_ports_count']}")
    for g in snap.get("gpus", []):
        lines.append(f"  GPU {g['index']} ({g['name']}): "
                     f"{g['mem_used_mb']}/{g['mem_total_mb']} MB mem, "
                     f"{g['util_pct']}% util, {g['temp_c']} gradiC")
    return "\n".join(lines)


def _monitor_loop(psutil_mod, stop_event, snapshots, interval=10):
    """Thread che campiona le risorse ogni `interval` secondi."""
    while not stop_event.is_set():
        snap = _system_snapshot(psutil_mod)
        snapshots.append(snap)
        stop_event.wait(interval)


def _run_single_worker_test(num_workers, num_clients, log_dir, dry_run=False):
    """
    Esegue federated_grid_search.py con num_parallel_executions=num_workers
    e num_clients=num_clients. Ritorna (success, duration_s, error_msg).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / f"run_w{num_workers}_c{num_clients}.log"

    # Crea una config temporanea con il numero di worker voluto
    with open(_CONFIG_STRESS) as f:
        config = json.load(f)

    config["num_parallel_executions"] = num_workers
    config["common_search_space"]["num_clients"] = [num_clients]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     dir=_PROJECT_ROOT, delete=False) as tf:
        json.dump(config, tf, indent=2)
        tmp_config_path = tf.name

    cmd = [sys.executable, str(_PROJECT_ROOT / "federated_grid_search.py"), tmp_config_path]

    if dry_run:
        print(f"  [DRY-RUN] Eseguirei: {' '.join(cmd)}")
        os.unlink(tmp_config_path)
        return True, 0.0, None

    print(f"  Avvio: {' '.join(cmd)}")
    print(f"  Log: {run_log}")

    start = time.time()
    try:
        with open(run_log, "w") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=7200,  # 2h max per run
                cwd=str(_PROJECT_ROOT)
            )
        duration = time.time() - start
        if proc.returncode != 0:
            return False, duration, f"Exit code {proc.returncode} (vedi {run_log})"
        return True, duration, None
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, duration, f"TIMEOUT dopo {duration:.0f}s"
    except Exception as e:
        duration = time.time() - start
        return False, duration, str(e)
    finally:
        try:
            os.unlink(tmp_config_path)
        except Exception:
            pass


def _analyze_log_for_errors(log_dir, num_workers, num_clients):
    """Analizza il log di una run e classifica gli errori trovati."""
    run_log = log_dir / f"run_w{num_workers}_c{num_clients}.log"
    if not run_log.exists():
        return {"error": "log not found"}

    errors = {"watchdog": 0, "namespace": 0, "port": 0, "oom": 0,
              "crash": 0, "timeout": 0, "other": []}
    try:
        content = run_log.read_text(encoding="utf-8", errors="replace")
        errors["watchdog"] = content.count("Watchdog:")
        errors["namespace"] = content.count("is not a connected namespace")
        errors["port"] = content.count("still occupied") + content.count("Address already in use")
        errors["oom"] = content.count("CUDA out of memory") + content.count("OutOfMemoryError")
        errors["crash"] = content.count("EXCEPTION") + content.count("Traceback (most recent call last)")
        errors["timeout"] = content.count("TimeoutExpired") + content.count("timeout")
        for line in content.splitlines():
            if "ERROR" in line and not any(k in line for k in ["namespace", "Watchdog", "Port"]):
                errors["other"].append(line.strip()[:200])
        errors["other"] = errors["other"][:10]  # max 10 esempi
    except Exception as e:
        errors["error"] = str(e)
    return errors


def main():
    parser = argparse.ArgumentParser(description="Stress test progressivo su geosciences")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Numero massimo di worker paralleli da testare (default: 6)")
    parser.add_argument("--clients", type=int, default=32,
                        help="Numero di client per ogni run (default: 32)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra il piano senza eseguire")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Parti da N worker (default: 1)")
    args = parser.parse_args()

    psutil_mod = _try_import_psutil()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_log_dir = _LOG_DIR / f"stress_{ts}"

    report_path = _REPORT_DIR / f"report_{ts}.txt"
    report_lines = [
        f"FCGS Geosciences Stress Test — {ts}",
        f"Obiettivo: trovare il massimo num_workers con num_clients={args.clients}",
        f"Range testato: {args.start_from} -> {args.max_workers} worker",
        "=" * 60,
        ""
    ]

    print("\n" + "=" * 60)
    print(f"FCGS Geosciences Stress Test")
    print(f"num_clients={args.clients}, range worker: {args.start_from}->{args.max_workers}")
    print("=" * 60 + "\n")

    results = {}

    for nw in range(args.start_from, args.max_workers + 1):
        print(f"\n{'-' * 60}")
        print(f"TEST: {nw} worker × {args.clients} client")
        print(f"{'-' * 60}")

        # Snapshot pre-run
        pre_snap = _system_snapshot(psutil_mod)
        print("Risorse pre-run:")
        print(_format_snapshot(pre_snap))

        # Monitor in background
        snapshots = []
        stop_evt = threading.Event()
        monitor_t = threading.Thread(
            target=_monitor_loop, args=(psutil_mod, stop_evt, snapshots, 15),
            daemon=True
        )
        monitor_t.start()

        success, duration, error_msg = _run_single_worker_test(
            nw, args.clients, run_log_dir, dry_run=args.dry_run
        )

        stop_evt.set()
        monitor_t.join(timeout=2)

        # Snapshot post-run
        post_snap = _system_snapshot(psutil_mod)
        print("Risorse post-run:")
        print(_format_snapshot(post_snap))

        # Analisi log
        errors = _analyze_log_for_errors(run_log_dir, nw, args.clients)

        results[nw] = {
            "success": success,
            "duration_s": round(duration, 1),
            "error_msg": error_msg,
            "errors": errors,
            "peak_snapshot": max(snapshots, key=lambda s: s.get("cpu_pct", 0)) if snapshots else post_snap
        }

        status = "OK" if success else "FAIL"
        print(f"\n-> Risultato: {status}  |  Durata: {duration:.0f}s")
        if error_msg:
            print(f"  Errore: {error_msg}")
        if errors.get("watchdog"):
            print(f"  Watchdog timeout: {errors['watchdog']}")
        if errors.get("oom"):
            print(f"  CUDA OOM: {errors['oom']}")
        if errors.get("port"):
            print(f"  Problemi porta: {errors['port']}")
        if errors.get("crash"):
            print(f"  Crash/Exception: {errors['crash']}")

        report_lines += [
            f"WORKER {nw} × {args.clients} CLIENT",
            f"  Stato: {status}  |  Durata: {duration:.0f}s",
            f"  Errore: {error_msg or 'nessuno'}",
            f"  Errori nel log: watchdog={errors.get('watchdog',0)}, "
            f"OOM={errors.get('oom',0)}, port={errors.get('port',0)}, "
            f"crash={errors.get('crash',0)}, namespace={errors.get('namespace',0)}",
            f"  Risorse peak: {_format_snapshot(results[nw]['peak_snapshot'])}",
            ""
        ]

        if not success:
            print(f"\n! Primo fallimento a {nw} worker. Interruzione test.")
            report_lines.append(
                f"PRIMO FALLIMENTO: {nw} worker × {args.clients} client — "
                f"max stabile = {nw - 1} worker"
            )
            break

    # Riepilogo finale
    max_stable = max((nw for nw, r in results.items() if r["success"]), default=0)
    print("\n" + "=" * 60)
    print("RIEPILOGO FINALE")
    print("=" * 60)
    for nw, r in sorted(results.items()):
        tag = "OK" if r["success"] else "FAIL"
        print(f"  {nw} worker × {args.clients} client: {tag}  ({r['duration_s']:.0f}s)")
    print(f"\nMax worker stabili con {args.clients} client: {max_stable}")
    if max_stable == 0:
        print("ATTENZIONE: anche 1 worker ha fallito. Controllare i log.")
    elif max_stable < args.max_workers:
        first_fail = max_stable + 1
        fail_errors = results.get(first_fail, {}).get("errors", {})
        causes = []
        if fail_errors.get("oom"):
            causes.append("CUDA OOM (GPU memoria)")
        if fail_errors.get("port"):
            causes.append("porta non disponibile")
        if fail_errors.get("watchdog"):
            causes.append("watchdog timeout (lentezza/rete)")
        if fail_errors.get("crash"):
            causes.append("eccezione non gestita")
        if causes:
            print(f"Probabile causa del fallimento a {first_fail} worker: {', '.join(causes)}")
    print("=" * 60)

    report_lines += [
        "",
        "=" * 60,
        "RIEPILOGO",
        f"Max worker stabili: {max_stable}",
        f"Report completo: {report_path}",
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport salvato: {report_path}")


if __name__ == "__main__":
    main()
