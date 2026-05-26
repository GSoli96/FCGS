import json
import sys
import threading
import time
import itertools
import traceback
import socket as _socket
import shutil
import stat
import subprocess
import signal as _signal
import requests
from requests.exceptions import ConnectionError
import socketio
import argparse

import PCNAME
import csv
import os
from typing import Dict, List, Any, Generator, Set, FrozenSet, Tuple
import multiprocessing

from trusted_authority import TrustedAuthority
import federated_server
import run_multiple_clients

NUM_PARALLEL_EXECUTIONS = 20
GRID_SEARCH_CONFIG_PATH = f'grid_search_config_{PCNAME.name}.json'
VERBOSE_DUPLICATE_CHECK = False


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def send_telegram(token: str, chat_id: str, message: str) -> None:
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"[Notifier] Telegram error: {e}")

def get_config_fingerprint(config: Dict, keys: Set[str]) -> FrozenSet[Tuple[str, str]]:
    fingerprint_items = []
    for key in sorted(list(keys)):
        if key in config and config[key] is not None and config[key] != '':
            fingerprint_items.append((key, str(config[key])))
    return frozenset(fingerprint_items)

def _safe_rmtree(path: str) -> None:
    def _onerror(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass
    for _ in range(5):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except OSError:
            time.sleep(2)


def _get_pid_on_port(port: int) -> int:
    """Returns PID listening on the given TCP port, or 0 if not found. Cross-platform."""
    if sys.platform == 'win32':
        try:
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5 and parts[3] == 'LISTENING':
                        local_addr = parts[1]
                        if local_addr.endswith(f':{port}') or f':{port}' in local_addr:
                            try:
                                return int(parts[-1])
                            except ValueError:
                                pass
        except Exception:
            pass
    else:
        # Linux/macOS: prova ss prima, poi lsof come fallback
        try:
            import re
            result = subprocess.run(
                ['ss', '-tlnp', f'sport = :{port}'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if f':{port}' in line:
                    m = re.search(r'pid=(\d+)', line)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True, text=True, timeout=10
            )
            first = result.stdout.strip().split('\n')[0]
            if first.isdigit():
                return int(first)
        except Exception:
            pass
    return 0


def _is_python_process(pid: int) -> bool:
    """Returns True if the given PID is a Python interpreter process. Cross-platform."""
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                capture_output=True, text=True, timeout=10
            )
            return 'python' in result.stdout.lower()
        except Exception:
            return False
    else:
        # Linux: leggi /proc/<pid>/cmdline
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().decode('utf-8', errors='replace').replace('\x00', ' ')
            return 'python' in cmdline.lower()
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'comm='],
                capture_output=True, text=True, timeout=10
            )
            return 'python' in result.stdout.lower()
        except Exception:
            return False


def _kill_process(pid: int) -> None:
    """Force-kills a process by PID. Cross-platform."""
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True, timeout=10)
    else:
        os.kill(pid, _signal.SIGKILL)


def find_free_port(host: str, preferred_port: int, max_scan: int = 200) -> int:
    """Returns the first free TCP port starting from preferred_port.
    Tries up to max_scan consecutive ports. Falls back to the OS-assigned port if none found."""
    for candidate in range(preferred_port, preferred_port + max_scan):
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                s.bind((host, candidate))
                return candidate
        except OSError:
            continue
    # Last resort: let the OS pick a free port
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_port_free(host: str, port: int, timeout: int = 120) -> bool:
    # Bind-based check: unico modo affidabile (TIME_WAIT non blocca connect ma blocca bind)
    # After 30s of waiting, actively kill the Python process holding the port.
    deadline = time.time() + timeout
    kill_after = time.time() + 30
    kill_attempted = False
    while time.time() < deadline:
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError:
            if not kill_attempted and time.time() >= kill_after:
                kill_attempted = True
                pid = _get_pid_on_port(port)
                if pid and pid != os.getpid() and _is_python_process(pid):
                    print(f"  [PortFree] Port {port} held by Python PID {pid}. Force-terminating...")
                    try:
                        _kill_process(pid)
                        print(f"  [PortFree] PID {pid} killed. Waiting for port to release...")
                        time.sleep(5)
                        continue
                    except Exception as e:
                        print(f"  [PortFree] Could not terminate PID {pid}: {e}")
            time.sleep(3)
    return False


def pre_download_model_weights(model_names: set) -> None:
    """
    Downloads all pretrained weights to the local torchvision disk cache before workers start.
    Worker processes (spawned fresh) then load from cache instantly — no download delay.
    """
    try:
        import torchvision.models as tv_models
    except ImportError:
        print("[PreDownload] torchvision not available, skipping pre-download.")
        return

    weight_map = {
        'GoogLeNet': (tv_models.googlenet, tv_models.GoogLeNet_Weights.IMAGENET1K_V1),
        'AlexNet':   (tv_models.alexnet,    tv_models.AlexNet_Weights.IMAGENET1K_V1),
        'ResNet18':  (tv_models.resnet18,   tv_models.ResNet18_Weights.IMAGENET1K_V1),
        'ResNet34':  (tv_models.resnet34,   tv_models.ResNet34_Weights.IMAGENET1K_V1),
        'ResNet50':  (tv_models.resnet50,   tv_models.ResNet50_Weights.IMAGENET1K_V1),
        'ResNet101': (tv_models.resnet101,  tv_models.ResNet101_Weights.IMAGENET1K_V1),
    }

    print("\n=== Pre-downloading model weights (CPU cache check) ===")
    for name in sorted(model_names):
        if name not in weight_map:
            print(f"[PreDownload] {name}: no pretrained weights needed (skipping).")
            continue
        fn, weights_enum = weight_map[name]
        print(f"[PreDownload] Checking {name}...", flush=True)
        try:
            model = fn(weights=weights_enum)
            del model
            print(f"[PreDownload] {name} weights ready.")
        except Exception as e:
            print(f"[PreDownload] WARNING — could not pre-load {name}: {e}")
    print("=== Pre-download complete ===\n")


def wait_for_server_ready(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200: return True
        except ConnectionError:
            time.sleep(1)
    return False


def load_json(filename: str) -> Dict:
    with open(filename) as f: return json.load(f)


def append_results_to_csv(csv_path: str, run_summary: Dict, lock: multiprocessing.Lock):
    lock.acquire()
    try:
        all_keys = set(run_summary.keys())
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                all_keys.update(header)

        file_is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if file_is_empty:
                writer.writeheader()
            writer.writerow(run_summary)
    finally:
        lock.release()


def generate_configurations(base_config: Dict, search_space: Dict) -> Generator[Dict[str, Any], None, None]:
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def _server_process_target(config: Dict, result_queue: multiprocessing.Queue) -> None:
    """Gira il FederatedServer in un subprocess separato e mette run_summary nella queue."""
    try:
        server = federated_server.FederatedServer(config)
        try:
            server.run()
        except SystemExit:
            # _shutdown_server usa ctypes per sollevare SystemExit nel thread main
            # del subprocess. Lo catturiamo per poter ancora salvare i risultati.
            pass
        run_summary = server.aggregator.get_run_summary()
        result_queue.put(run_summary)
    except Exception as e:
        print(f"[ServerProc W{config.get('worker_id','?')}] Error: {e}\n{traceback.format_exc()}")
        result_queue.put(None)


def start_ta_thread(config: Dict, ta_instance_ref: List) -> None:
    worker_id = config.get('worker_id', 'N/A')
    try:
        ta = TrustedAuthority(
            host=config['ip_address'],
            port=config['ta_port'],
            he_backend=config.get('he_backend', 'paillier')
        )
        ta_instance_ref.append(ta)
        ta.run()
    except Exception as e:
        print(f"[W{worker_id}] TA CRASH: {e}\n{traceback.format_exc()}")


def _ta_process_target(config: Dict) -> None:
    """Runs the Trusted Authority in a separate subprocess."""
    worker_id = config.get('worker_id', 'N/A')
    try:
        from trusted_authority import TrustedAuthority
        ta = TrustedAuthority(
            host=config['ip_address'],
            port=config['ta_port'],
            he_backend=config.get('he_backend', 'paillier')
        )
        ta.run()
    except Exception as e:
        import traceback
        print(f"[TAProc W{worker_id}] Error: {e}\n{traceback.format_exc()}")


def _heartbeat_loop(
        telegram_token: str,
        telegram_chat_id: str,
        completed_count: multiprocessing.Value,
        total_configs: int,
        start_time: float,
        pc_name: str,
        interval_seconds: int,
        stop_event: threading.Event,
) -> None:
    while not stop_event.wait(interval_seconds):
        done = completed_count.value
        elapsed = time.time() - start_time
        remaining = total_configs - done
        pct = done / total_configs * 100 if total_configs > 0 else 0
        eta_str = _format_duration((elapsed / done) * remaining) if done > 0 else "N/A"
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS Heartbeat — {pc_name}</b>\n"
                      f"Completate: <b>{done}/{total_configs}</b> ({pct:.1f}%)\n"
                      f"Rimanenti: {remaining}\n"
                      f"Tempo trascorso: {_format_duration(elapsed)}\n"
                      f"ETA: {eta_str}")


def run_grid_search_worker(
        worker_id: int,
        task_queue: multiprocessing.JoinableQueue,
        base_port: int,
        csv_lock: multiprocessing.Lock,
        completed_count: multiprocessing.Value,
        last_notified: multiprocessing.Value,
        total_configs: int,
        start_time: float,
        notify_lock: multiprocessing.Lock,
        notification_interval: int,
        telegram_token: str,
        telegram_chat_id: str,
) -> None:
    print(f"--- Starting Grid Search Worker {worker_id} ---")
    pc_name = PCNAME.name
    while True:
        config = task_queue.get()
        if config is None:
            print(f"--- Worker {worker_id} shutting down. ---")
            task_queue.task_done()
            break

        dataset_name = config.get('dataset_name', 'N/A')
        model_name = config.get('model_name', 'N/A')
        print(f"[W{worker_id}] START {dataset_name} | {model_name}")

        worker_splitting_dir = None
        try:
            run_identifier = f"{dataset_name}_run_{worker_id}_{model_name}"
            worker_splitting_dir = os.path.join(config['base_split_data_path'] + f'_{pc_name}', run_identifier)
            worker_log_dir = os.path.join(config['base_log_path'] + f'_{pc_name}', run_identifier)
            worker_plot_dir = os.path.join(config['base_plot_path'] + f'_{pc_name}', run_identifier)
            worker_metrics_dir = os.path.join(config['base_csv_path'] + f'_{pc_name}', "runs")
            # worker_splitting_dir viene creato DOPO i port check per non lasciare
            # directory orfane quando la porta è occupata e il worker fa continue
            os.makedirs(worker_log_dir, exist_ok=True)
            os.makedirs(worker_plot_dir, exist_ok=True)
            os.makedirs(worker_metrics_dir, exist_ok=True)

            config['worker_id'] = worker_id
            preferred_port = base_port + worker_id * 2
            config['port'] = find_free_port(config['ip_address'], preferred_port)
            if config['port'] != preferred_port:
                print(f"[W{worker_id}] Port {preferred_port} busy, using {config['port']} instead.")
            config['splitting_dir'] = worker_splitting_dir
            config['log_dir'] = worker_log_dir
            config['plot_dir'] = worker_plot_dir
            config['run_metrics_output_path'] = worker_metrics_dir
            config["MIN_NUM_WORKERS"] = int(config['num_clients'] * config['models_percentage'])

            if not wait_for_port_free(config['ip_address'], config['port'], timeout=180):
                msg = f"[W{worker_id}] Port {config['port']} still occupied after 180s. Skipping {dataset_name}|{model_name}."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Port busy — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Porta {config['port']} occupata dopo 180s. Skipping.")
                continue

            # Porta libera: ora è sicuro creare la directory di split
            os.makedirs(worker_splitting_dir, exist_ok=True)

            # Il server gira in un subprocess separato così, se si blocca,
            # possiamo killarlo con SIGKILL e liberare la porta immediatamente.
            result_queue = multiprocessing.Queue()
            server_process = multiprocessing.Process(
                target=_server_process_target,
                args=(config, result_queue),
                daemon=True
            )
            server_process.start()
            server_url = f"http://{config['ip_address']}:{config['port']}"
            server_ready_timeout = config.get('server_ready_timeout', 60)
            server_join_timeout = config.get('server_join_timeout', 120)
            if wait_for_server_ready(server_url, timeout=server_ready_timeout):
                run_multiple_clients.main(config)
            else:
                msg = f"[W{worker_id}] Server non avviato per {dataset_name}|{model_name}. Skipping."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Server timeout — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Il server non ha risposto entro {server_ready_timeout}s.")
            server_process.join(timeout=server_join_timeout)
            if server_process.is_alive():
                msg = f"[W{worker_id}] Server process ancora vivo dopo {server_join_timeout}s per {dataset_name}|{model_name}. SIGKILL."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Server hung — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Server non terminato dopo {server_join_timeout}s. Forzato SIGKILL.")
                server_process.kill()
                server_process.join(timeout=10)
            try:
                run_summary = result_queue.get(timeout=5)
            except Exception:
                run_summary = None
            if run_summary:
                append_results_to_csv(config['shared_csv_path'], run_summary, csv_lock)
            else:
                msg = f"[W{worker_id}] run_summary None per {dataset_name}|{model_name} (crash o kill server)."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS run_summary None — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Nessun risultato: server crashato o killato.")
            if os.path.exists(worker_splitting_dir):
                _safe_rmtree(worker_splitting_dir)
                print(f"[W{worker_id}] Cleaned up split dir: {worker_splitting_dir}")
            time.sleep(1)

            should_notify = False
            with notify_lock:
                with completed_count.get_lock():
                    completed_count.value += 1
                    done = completed_count.value
                    if done - last_notified.value >= notification_interval:
                        last_notified.value = done
                        should_notify = True

            print(f"[W{worker_id}] DONE  {dataset_name} | {model_name} ({done}/{total_configs})")

            if should_notify and telegram_token and telegram_chat_id:
                elapsed = time.time() - start_time
                remaining = total_configs - done
                eta_str = _format_duration((elapsed / done) * remaining) if done > 0 else "N/A"
                pct = done / total_configs * 100 if total_configs > 0 else 0
                send_telegram(telegram_token, telegram_chat_id,
                              f"<b>FCGS Progress — {pc_name}</b>\n"
                              f"Completate: <b>{done}/{total_configs}</b> ({pct:.1f}%)\n"
                              f"Rimanenti: {remaining}\n"
                              f"Tempo trascorso: {_format_duration(elapsed)}\n"
                              f"ETA: {eta_str}\n"
                              f"Ultimo: {dataset_name} | {model_name}")
                print(f"[Notifier] Progress Telegram inviato ({done}/{total_configs})")

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"[W{worker_id}] EXCEPTION on {dataset_name}|{model_name}:\n{tb_str}")
            if telegram_token and telegram_chat_id:
                send_telegram(telegram_token, telegram_chat_id,
                              f"<b>⚠️ FCGS ERRORE — Worker {worker_id} ({pc_name})</b>\n"
                              f"Config: {dataset_name} | {model_name}\n"
                              f"<code>{str(e)[:500]}</code>")
            if worker_splitting_dir is not None and os.path.exists(worker_splitting_dir):
                _safe_rmtree(worker_splitting_dir)
        finally:
            task_queue.task_done()


def _compute_num_workers(config: dict) -> int:
    if 'num_parallel_executions' in config:
        return int(config['num_parallel_executions'])
    try:
        import torch
        if torch.cuda.is_available():
            blacklist = config.get('gpu_blacklist', [])
            usable = [i for i in range(torch.cuda.device_count()) if i not in blacklist]
            return max(NUM_PARALLEL_EXECUTIONS, len(usable) * 4)
    except Exception:
        pass
    return NUM_PARALLEL_EXECUTIONS


def main():
    if not os.path.exists(GRID_SEARCH_CONFIG_PATH):
        print(f"ERRORE: nessun file di configurazione trovato per la macchina '{PCNAME.name}'.")
        print(f"        Atteso: {GRID_SEARCH_CONFIG_PATH}")
        print(f"        Crea il file oppure passalo come argomento: python federated_grid_search.py <config.json>")
        sys.exit(1)
    print(f"=== Configurazione: {GRID_SEARCH_CONFIG_PATH} ===")
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()
    pc_name = PCNAME.name

    # Non cancelliamo le directory di split all'avvio: ogni worker le ripulisce
    # internamente (tramite _clear_existing_split) all'inizio di ogni iterazione.
    # Il cleanup globale causa FileNotFoundError se un'altra istanza è ancora attiva.

    base_csv_path = base_grid_config['base_csv_path']+f'_{pc_name}'
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(base_grid_config.get('base_log_path', 'logs')+f'_{pc_name}', exist_ok=True)
    os.makedirs(base_grid_config.get('base_plot_path', 'static/plots')+f'_{pc_name}', exist_ok=True)
    os.makedirs(base_grid_config.get('base_split_data_path', f'run_dataset')+f'_{pc_name}', exist_ok=True)
    path_risultati = os.path.join(pc_name, f'federated_grid_search_results_{pc_name}.csv')
    shared_csv_path = os.path.join(base_csv_path, path_risultati)

    directory_path = os.path.dirname(shared_csv_path)

    os.makedirs(directory_path, exist_ok=True)

    executed_fingerprints: Set[FrozenSet[Tuple[str, str]]] = set()
    common_search_space = base_grid_config['common_search_space']
    model_specific_search_space = base_grid_config['model_specific_search_space']

    all_possible_search_keys = set(common_search_space.keys())
    for model_params in model_specific_search_space.values():
        all_possible_search_keys.update(model_params.keys())
    all_possible_search_keys.update(['dataset_name', 'model_name'])

    if os.path.exists(shared_csv_path) and os.path.getsize(shared_csv_path) > 0:
        with open(shared_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalizza la riga del CSV nello stesso modo in cui normalizzeremo le nuove config
                model_name_from_row = row.get('model_name') or ''
                if row.get('aggregation_algorithm') != "FedProx": row['fedprox_mu'] = '0.0'
                if row.get('encryption_mode') == 'no_encryption': row['he_backend'] = 'N/A'
                if 'ResNet' in model_name_from_row or 'GoogLeNet' in model_name_from_row or 'AlexNet' in model_name_from_row:
                    row.setdefault('image_size', '224')
                    row.setdefault('convnet_hidden1', '-1')
                    row.setdefault('convnet_hidden2', '-1')
                elif 'ConvNet' in model_name_from_row:
                    row.setdefault('num_custom_layers', '-1')

                # Usa il set di chiavi globale per creare il fingerprint
                fingerprint = get_config_fingerprint(row, all_possible_search_keys)
                executed_fingerprints.add(fingerprint)
    print(f"Loaded {len(executed_fingerprints)} fingerprints of previously executed configurations.")

    fixed_params = {k: v for k, v in base_grid_config.items() if
                    k not in ['datasets', 'common_search_space', 'model_specific_search_space']}
    fixed_params['shared_csv_path'] = shared_csv_path
    configs_to_run_count = 0
    total_generated_configs = 0

    for dataset_info in base_grid_config['datasets']:
        # Itera sui modelli *attualmente attivi* nel file di configurazione
        for model_name, specific_params in model_specific_search_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({'dataset_name': dataset_info['name'], 'dataset_path': dataset_info['path'],
                                      'num_classes': dataset_info['num_classes'], 'model_name': model_name})
            current_search_space = {**common_search_space, **specific_params}

            for hyper_config in generate_configurations(model_base_config, current_search_space):
                total_generated_configs += 1
                if VERBOSE_DUPLICATE_CHECK: print("-" * 50,
                                                  f"\n[{total_generated_configs}] Generated Raw Config:\n{json.dumps(hyper_config, indent=2)}")

                if hyper_config.get('aggregation_algorithm') != "FedProx": hyper_config['fedprox_mu'] = 0.0
                if hyper_config.get('encryption_mode') == 'no_encryption': hyper_config['he_backend'] = 'N/A'
                if 'ResNet' in model_name or 'GoogLeNet' in model_name or 'AlexNet' in model_name:
                    hyper_config['image_size'] = 224
                    hyper_config.setdefault('convnet_hidden1', -1)
                    hyper_config.setdefault('convnet_hidden2', -1)
                elif 'ConvNet' in model_name:
                    hyper_config.setdefault('num_custom_layers', -1)
                if VERBOSE_DUPLICATE_CHECK: print(f" -> Normalized Config:\n{json.dumps(hyper_config, indent=2)}")

                fingerprint = get_config_fingerprint(hyper_config, all_possible_search_keys)

                if VERBOSE_DUPLICATE_CHECK: print(f" -> Fingerprint: {fingerprint}")

                if fingerprint in executed_fingerprints:
                    if VERBOSE_DUPLICATE_CHECK: print(" -> STATUS: DUPLICATE - Skipping.")
                    continue

                if VERBOSE_DUPLICATE_CHECK: print(" -> STATUS: NEW - Adding to queue.")
                executed_fingerprints.add(fingerprint)
                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Generated {total_generated_configs} total configurations.")
    print(f"Skipped {total_generated_configs - configs_to_run_count} already executed or redundant configurations.")
    print(f"Adding {configs_to_run_count} new unique configurations to the execution queue.")
    print("=" * 80)

    if configs_to_run_count == 0:
        print("=== No new configurations to run. Exiting. ===")
        return

    telegram_token = base_grid_config.get('telegram_bot_token', '')
    telegram_chat_id = base_grid_config.get('telegram_chat_id', '')
    notification_interval = int(base_grid_config.get('notification_interval', 10))

    completed_count = multiprocessing.Value('i', 0)
    last_notified = multiprocessing.Value('i', 0)
    notify_lock = multiprocessing.Lock()
    start_time = time.time()

    num_workers = _compute_num_workers(base_grid_config)

    if telegram_token and telegram_chat_id:
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS avviata — {pc_name}</b>\n"
                      f"Config da eseguire: <b>{configs_to_run_count}</b>\n"
                      f"Workers paralleli: {num_workers}\n"
                      f"Notifica ogni {notification_interval} completamenti.")
    # Pre-download weights for all models in the search space before spawning workers
    pre_download_model_weights(set(model_specific_search_space.keys()))

    processes = []
    print(f"\n=== Starting {num_workers} Grid Search workers ===")
    base_port = base_grid_config['port']
    worker_args = (task_queue, base_port, csv_lock, completed_count, last_notified,
                   configs_to_run_count, start_time, notify_lock, notification_interval,
                   telegram_token, telegram_chat_id)
    for i in range(num_workers):
        process = multiprocessing.Process(target=run_grid_search_worker, args=(i, *worker_args))
        processes.append(process)
        process.start()

    def _shutdown_all(signum=None, frame=None):
        print("\n[FCGS] Signal received. Terminating all worker processes...")
        for p in processes:
            if p.is_alive():
                p.kill()
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _shutdown_all)
    _signal.signal(_signal.SIGINT, _shutdown_all)

    heartbeat_stop = threading.Event()
    heartbeat_interval = int(base_grid_config.get('heartbeat_interval_hours', 4)) * 3600
    if telegram_token and telegram_chat_id:
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(telegram_token, telegram_chat_id, completed_count, configs_to_run_count,
                  start_time, pc_name, heartbeat_interval, heartbeat_stop),
            daemon=True
        )
        heartbeat_thread.start()

    task_queue.join()
    heartbeat_stop.set()
    for _ in range(num_workers):
        task_queue.put(None)
    for process in processes:
        process.join()

    elapsed = time.time() - start_time
    print("\n=== All parallel executions have finished. ===")
    if telegram_token and telegram_chat_id:
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS completata — {pc_name}</b>\n"
                      f"Config eseguite: {completed_count.value}/{configs_to_run_count}\n"
                      f"Tempo totale: {_format_duration(elapsed)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', type=str, default=GRID_SEARCH_CONFIG_PATH)
    args = parser.parse_args()
    GRID_SEARCH_CONFIG_PATH = args.config
    multiprocessing.set_start_method('spawn', force=True)
    main()