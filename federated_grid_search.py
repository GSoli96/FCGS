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
import re
import logging
import requests
from requests.exceptions import ConnectionError
import socketio
import argparse
from pathlib import Path

import PCNAME
import csv
import os
from typing import Dict, List, Any, Generator, Set, FrozenSet, Tuple
import multiprocessing

from trusted_authority import TrustedAuthority
import federated_server
import run_multiple_clients

NUM_PARALLEL_EXECUTIONS = 20
# Configs con num_clients > questo valore vengono eseguiti in isolamento (heavy mode)
HEAVY_CLIENT_THRESHOLD = 8

_PROJECT_ROOT = Path(__file__).parent
_pc_specific_config = f'grid_search_config_{PCNAME.name}.json'
GRID_SEARCH_CONFIG_PATH = _pc_specific_config if os.path.exists(_pc_specific_config) else 'grid_search_config.json'
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


def _safe_print(text: str) -> None:
    """Print handling UnicodeEncodeError on Windows (e.g. emoji in cp1252 stdout)."""
    try:
        print(text)
    except UnicodeEncodeError:
        import sys as _sys
        try:
            _sys.stdout.buffer.write((text + '\n').encode('utf-8'))
            _sys.stdout.buffer.flush()
        except Exception:
            print(text.encode('ascii', errors='replace').decode('ascii'))


def send_telegram(token: str, chat_id: str, message: str, logger=None) -> None:
    clean = re.sub(r'<[^>]+>', '', message).strip()
    log_line = f"[Telegram] {clean}"
    if logger:
        logger.info(log_line)
    else:
        _safe_print(log_line)
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        err = f"[Notifier] Telegram error: {e}"
        if logger:
            logger.error(err)
        else:
            _safe_print(err)


def _setup_main_logger(pc_name: str, base_log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"FCGS-Main-{pc_name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_dir = os.path.join(_PROJECT_ROOT, base_log_path, f'log_{pc_name}')
    os.makedirs(log_dir, exist_ok=True)
    timestr = time.strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'main_{timestr}.log'), encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

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


def _port_is_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    """Returns True only if a connection is actively accepted.
    Any connection failure (refused, timeout, error) is treated as 'free'."""
    try:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return True
    except OSError:
        return False


def find_free_port(host: str, preferred_port: int, max_scan: int = 200) -> int:
    """Returns the first TCP port (starting from preferred_port) that nothing is actively listening on."""
    for candidate in range(preferred_port, preferred_port + max_scan):
        if not _port_is_listening(host, candidate):
            return candidate
    # Last resort: let the OS pick a free port
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_port_free(host: str, port: int, timeout: int = 120) -> bool:
    """Waits until no process is actively listening on the port (netstat-based check), or timeout expires.
    After 30s, attempts to kill the Python process holding the port."""
    deadline = time.time() + timeout
    kill_after = time.time() + 30
    kill_attempted = False
    while time.time() < deadline:
        if _get_pid_on_port(port) == 0:
            return True
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


_MODEL_COST = {
    'ConvNet':   1.0,
    'AlexNet':   2.0,
    'ResNet18':  2.5,
    'GoogLeNet': 3.0,
    'ResNet34':  3.5,
    'ResNet50':  4.5,
    'ResNet101': 6.0,
}

def compute_complexity_score(cfg: Dict) -> float:
    """
    Stima il costo computazionale di una configurazione FL.
    Usato per ordinare le task dalla più semplice alla più complessa.

    Fattori considerati:
      - num_clients × global_epoch × local_epoch  (tempo lineare)
      - image_size²                               (costo quadratico per immagine)
      - num_images                                (dimensione dataset, normalizzata su 5000)
      - peso del modello                          (parametri e forward pass)
      - cifratura omomorfica                      (×10 rispetto a no_encryption)
      - num_custom_layers                         (layer extra FC)
    """
    score = (cfg.get('num_clients', 2)
             * cfg.get('global_epoch', 5)
             * cfg.get('local_epoch', 2))

    image_size = cfg.get('image_size', 128)
    score *= (image_size / 128.0) ** 2

    num_images = cfg.get('num_images', 5000)
    score *= (num_images / 5000.0)

    model = cfg.get('model_name', 'ConvNet')
    score *= _MODEL_COST.get(model, 2.0)

    if cfg.get('encryption_mode') == 'homomorphic':
        score *= 10.0

    layers = cfg.get('num_custom_layers', 2)
    if layers and layers > 0:
        score *= (layers / 2.0)

    return score


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
        error_count: multiprocessing.Value = None,
        fail_count: multiprocessing.Value = None,
        logger=None,
) -> None:
    while not stop_event.wait(interval_seconds):
        done = completed_count.value
        elapsed = time.time() - start_time
        remaining = total_configs - done
        pct = done / total_configs * 100 if total_configs > 0 else 0
        eta_str = _format_duration((elapsed / done) * remaining) if done > 0 else "N/A"
        errors = error_count.value if error_count is not None else 0
        fails = fail_count.value if fail_count is not None else 0
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS Heartbeat — {pc_name}</b>\n"
                      f"Completate: <b>{done}/{total_configs}</b> ({pct:.1f}%)\n"
                      f"Rimanenti: {remaining}\n"
                      f"Errori worker: {errors} | Fail (no risultati): {fails}\n"
                      f"Tempo trascorso: {_format_duration(elapsed)}\n"
                      f"ETA: {eta_str}",
                      logger=logger)


def _cleanup_after_task(
        server_process: multiprocessing.Process,
        port: int,
        host: str,
        worker_id: int,
        max_port_wait: int = 30
) -> bool:
    """
    Post-task cleanup: kill zombie server process (+ children) and verify the port is freed.
    Returns True if the port is confirmed free, False if it could not be freed.
    """
    # Kill server + children (psutil tracks child processes created by Flask/Werkzeug)
    if server_process.is_alive():
        try:
            import psutil
            parent = psutil.Process(server_process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception:
            pass
        server_process.kill()
        server_process.join(timeout=10)

    # Wait for port to be released
    deadline = time.time() + max_port_wait
    while time.time() < deadline:
        if not _port_is_listening(host, port):
            return True
        time.sleep(1)

    # Force-kill whatever is still holding the port
    pid = _get_pid_on_port(port)
    if pid and pid != os.getpid():
        print(f"[W{worker_id}] Port {port} still held by PID {pid} after task. Force-killing.")
        try:
            _kill_process(pid)
            time.sleep(3)
        except Exception as e:
            print(f"[W{worker_id}] Could not kill PID {pid}: {e}")

    if _port_is_listening(host, port):
        print(f"[W{worker_id}] WARNING: port {port} could not be freed after task cleanup.")
        return False
    return True


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
        gpu_blacklist: list = None,
        error_count: multiprocessing.Value = None,
        fail_count: multiprocessing.Value = None,
        heavy_lock: multiprocessing.Lock = None,
        heavy_running: multiprocessing.Value = None,
        active_normal_workers: multiprocessing.Value = None,
        heavy_threshold: int = HEAVY_CLIENT_THRESHOLD,
) -> None:
    print(f"--- Starting Grid Search Worker {worker_id} ---")
    pc_name = PCNAME.name
    logger = logging.getLogger(f"FCGS-Worker-{worker_id}-{pc_name}")

    # Distribuisce i worker sui GPU disponibili (escludendo gpu_blacklist).
    # Ogni worker process ha il proprio spazio di ambiente, quindi impostare
    # CUDA_VISIBLE_DEVICES qui non interferisce con gli altri worker.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _blacklist = gpu_blacklist or []
            _n = _torch.cuda.device_count()
            _usable = [i for i in range(_n) if i not in _blacklist]
            if _usable:
                _gpu = _usable[worker_id % len(_usable)]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu)
                print(f"[W{worker_id}] Assigned to GPU {_gpu} "
                      f"(usable: {_usable}, total: {_n})")
    except Exception:
        pass

    while True:
        config = task_queue.get()
        if config is None:
            print(f"--- Worker {worker_id} shutting down. ---")
            task_queue.task_done()
            break

        dataset_name = config.get('dataset_name', 'N/A')
        model_name = config.get('model_name', 'N/A')
        is_heavy = config.get('num_clients', 2) > heavy_threshold
        print(f"[W{worker_id}] START {dataset_name} | {model_name} ({'heavy' if is_heavy else 'normal'})")

        worker_splitting_dir = None
        _heavy_acquired = False
        _normal_counted = False
        try:
            # --- Heavy/Normal scheduling ---
            if is_heavy and heavy_lock is not None:
                print(f"[W{worker_id}] Heavy config ({config['num_clients']} clients > {heavy_threshold}): attendendo heavy_lock...")
                heavy_lock.acquire()
                _heavy_acquired = True
                if heavy_running is not None:
                    with heavy_running.get_lock():
                        heavy_running.value = 1
                # Aspetta che tutti i worker normali finiscano il task corrente
                deadline = time.time() + 3600
                while active_normal_workers is not None and time.time() < deadline:
                    with active_normal_workers.get_lock():
                        if active_normal_workers.value == 0:
                            break
                    time.sleep(2)
                print(f"[W{worker_id}] Heavy config: via libera, avvio esecuzione.")
            elif not is_heavy and heavy_running is not None:
                # Worker normale: aspetta se c'è un heavy in corso
                while True:
                    with heavy_running.get_lock():
                        if heavy_running.value == 0:
                            break
                    time.sleep(2)
                if active_normal_workers is not None:
                    with active_normal_workers.get_lock():
                        active_normal_workers.value += 1
                    _normal_counted = True

            run_identifier = f"{dataset_name}_run_{worker_id}_{model_name}"
            worker_splitting_dir = os.path.join(_PROJECT_ROOT, config['base_split_data_path'], f'split_{pc_name}', run_identifier)
            worker_log_dir = os.path.join(_PROJECT_ROOT, config['base_log_path'], f'log_{pc_name}', run_identifier)
            worker_plot_dir = os.path.join(_PROJECT_ROOT, config['base_plot_path'], f'plots_{pc_name}', run_identifier)
            worker_metrics_dir = os.path.join(_PROJECT_ROOT, config['base_csv_path'], f'csv_{pc_name}', "runs")
            # worker_splitting_dir viene creato DOPO i port check per non lasciare
            # directory orfane quando la porta è occupata e il worker fa continue
            os.makedirs(worker_log_dir, exist_ok=True)
            os.makedirs(worker_plot_dir, exist_ok=True)
            os.makedirs(worker_metrics_dir, exist_ok=True)

            config['worker_id'] = worker_id
            preferred_port = base_port + worker_id * 2
            # Port selection with retry to mitigate TOCTOU race condition:
            # another worker could grab the port between find_free_port() and the server bind.
            for _port_retry in range(5):
                config['port'] = find_free_port(config['ip_address'], preferred_port)
                time.sleep(0.1 * (_port_retry + 1))
                if not _port_is_listening(config['ip_address'], config['port']):
                    break
            if config['port'] != preferred_port:
                print(f"[W{worker_id}] Port {preferred_port} busy, using {config['port']} instead.")
            config['splitting_dir'] = worker_splitting_dir
            config['log_dir'] = worker_log_dir
            config['plot_dir'] = worker_plot_dir
            config['run_metrics_output_path'] = worker_metrics_dir

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

            # Esegui lo split REALE prima di avviare il server — così MIN_NUM_WORKERS
            # riflette i client effettivi dopo I/O (non una stima dry-run che ignora
            # fallimenti di copia su Windows sotto carico).
            _dataset_abs = os.path.join(str(_PROJECT_ROOT), config['dataset_path'])
            if not os.path.isdir(_dataset_abs):
                logger.warning(f"[W{worker_id}] Dataset non trovato: {_dataset_abs}. "
                               f"Skipping {dataset_name}|{model_name}.")
                _safe_rmtree(worker_splitting_dir)
                continue
            from data_splitter import DatasetSplitter
            _splitter = DatasetSplitter(
                output_base_dir=worker_splitting_dir,
                source_images_dir=_dataset_abs,
                num_clients=config['num_clients'],
            )
            actual_clients = _splitter.split_dataset()
            if actual_clients == 0:
                logger.warning(f"[W{worker_id}] split_dataset=0 client per "
                               f"{dataset_name}|{model_name}. Skipping.")
                _safe_rmtree(worker_splitting_dir)
                continue
            if actual_clients < config['num_clients']:
                logger.info(f"[W{worker_id}] Split I/O: {actual_clients}/"
                            f"{config['num_clients']} client validi dopo copia file.")
            config["MIN_NUM_WORKERS"] = actual_clients

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
            client_instances = []
            if wait_for_server_ready(server_url, timeout=server_ready_timeout):
                client_instances = run_multiple_clients.main(config, already_split=True)
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
            # Kill zombie server process + children and verify port is freed
            _cleanup_after_task(server_process, config['port'], config['ip_address'], worker_id)

            # Verify all client socket connections are closed before the next run.
            # run_multiple_clients already did join+force-disconnect with timeout;
            # this is a final safety check with references we hold here.
            still_connected = [c for c in client_instances if c.sio.connected]
            if still_connected:
                print(f"[W{worker_id}] {len(still_connected)} client socket(s) still connected after cleanup. "
                      f"Force-disconnecting...")
                for c in still_connected:
                    try:
                        c.sio.disconnect()
                    except Exception as _e:
                        print(f"[W{worker_id}] Error disconnecting client: {_e}")

            try:
                run_summary = result_queue.get(timeout=5)
            except Exception:
                run_summary = None
            if run_summary:
                append_results_to_csv(config['shared_csv_path'], run_summary, csv_lock)
            else:
                if fail_count is not None:
                    with fail_count.get_lock():
                        fail_count.value += 1
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
            if error_count is not None:
                with error_count.get_lock():
                    error_count.value += 1
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
            if _heavy_acquired and heavy_lock is not None:
                if heavy_running is not None:
                    with heavy_running.get_lock():
                        heavy_running.value = 0
                heavy_lock.release()
            if _normal_counted and active_normal_workers is not None:
                with active_normal_workers.get_lock():
                    active_normal_workers.value = max(0, active_normal_workers.value - 1)
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
        print(f"ERRORE: nessun file di configurazione trovato.")
        print(f"        Cercato: {_pc_specific_config}, poi: grid_search_config.json")
        print(f"        Crea il file oppure passalo come argomento: python federated_grid_search.py <config.json>")
        sys.exit(1)
    print(f"=== Configurazione: {GRID_SEARCH_CONFIG_PATH} ===")
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)

    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()
    pc_name = PCNAME.name

    # Nuova struttura: csv/csv_{PC}/ e log/log_{PC}/
    _base_log = base_grid_config.get('base_log_path', 'log')
    _base_csv = base_grid_config.get('base_csv_path', 'csv')
    _base_plot = base_grid_config.get('base_plot_path', 'static/plots')
    _base_split = base_grid_config.get('base_split_data_path', 'run_dataset')

    base_csv_path = os.path.join(_PROJECT_ROOT, _base_csv, f'csv_{pc_name}')
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(os.path.join(_PROJECT_ROOT, _base_log, f'log_{pc_name}'), exist_ok=True)
    os.makedirs(os.path.join(_PROJECT_ROOT, _base_plot, f'plots_{pc_name}'), exist_ok=True)
    os.makedirs(os.path.join(_PROJECT_ROOT, _base_split, f'split_{pc_name}'), exist_ok=True)

    shared_csv_path = os.path.join(base_csv_path, f'federated_grid_search_results_{pc_name}.csv')

    executed_fingerprints: Set[FrozenSet[Tuple[str, str]]] = set()
    common_search_space = base_grid_config['common_search_space']
    model_specific_search_space = base_grid_config['model_specific_search_space']

    all_possible_search_keys = set(common_search_space.keys())
    for model_params in model_specific_search_space.values():
        all_possible_search_keys.update(model_params.keys())
    all_possible_search_keys.update(['dataset_name', 'model_name'])

    # Carica fingerprint da TUTTI i CSV di risultati presenti (tutti i PC).
    # Struttura nuova: csv/csv_{PC}/federated_grid_search_results_{PC}.csv
    # Struttura legacy: csv_{PC}/{PC}/federated_grid_search_results_{PC}.csv
    import glob as _glob
    all_result_csvs = _glob.glob(
        os.path.join(_PROJECT_ROOT, _base_csv, 'csv_*', 'federated_grid_search_results_*.csv')
    )
    # Fallback legacy: vecchia struttura prima del refactoring
    for _lp in _glob.glob(os.path.join(_PROJECT_ROOT, f'{_base_csv}_*', '*', 'federated_grid_search_results_*.csv')):
        if _lp not in all_result_csvs:
            all_result_csvs.append(_lp)
    # Include anche all_results.csv se presente (storico consolidato)
    _all_results_path = os.path.join(_PROJECT_ROOT, 'all_results.csv')
    if os.path.exists(_all_results_path) and _all_results_path not in all_result_csvs:
        all_result_csvs.append(_all_results_path)
    if shared_csv_path not in all_result_csvs and os.path.exists(shared_csv_path):
        all_result_csvs.append(shared_csv_path)

    total_loaded = 0
    for csv_path in all_result_csvs:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            continue
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name_from_row = row.get('model_name') or ''
                if row.get('aggregation_algorithm') != "FedProx": row['fedprox_mu'] = '0.0'
                if row.get('encryption_mode') == 'no_encryption': row['he_backend'] = 'N/A'
                if 'ResNet' in model_name_from_row or 'GoogLeNet' in model_name_from_row or 'AlexNet' in model_name_from_row:
                    row.setdefault('image_size', '224')
                    row.setdefault('convnet_hidden1', '-1')
                    row.setdefault('convnet_hidden2', '-1')
                elif 'ConvNet' in model_name_from_row:
                    row.setdefault('num_custom_layers', '-1')
                fingerprint = get_config_fingerprint(row, all_possible_search_keys)
                executed_fingerprints.add(fingerprint)
        count = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1
        print(f"  Loaded {count} fingerprints from {csv_path}")
        total_loaded += count
    print(f"Loaded {len(executed_fingerprints)} unique fingerprints from {len(all_result_csvs)} CSV(s) ({total_loaded} rows total).")

    fixed_params = {k: v for k, v in base_grid_config.items() if
                    k not in ['datasets', 'common_search_space', 'model_specific_search_space']}
    fixed_params['shared_csv_path'] = shared_csv_path
    configs_to_run_count = 0
    total_generated_configs = 0

    # Raccoglie tutte le configurazioni nuove in una lista prima di metterle in coda,
    # in modo da poterle ordinare per complessità crescente (le più veloci prima).
    pending_configs: List[Dict] = []

    for dataset_info in base_grid_config['datasets']:
        # Itera sui modelli *attualmente attivi* nel file di configurazione
        for model_name, specific_params in model_specific_search_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({'dataset_name': dataset_info['name'], 'dataset_path': dataset_info['path'],
                                      'num_classes': dataset_info['num_classes'], 'model_name': model_name,
                                      'num_images': dataset_info.get('num_images', 5000)})
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
                pending_configs.append(hyper_config)
                configs_to_run_count += 1

    # Ordina dalla configurazione più semplice alla più complessa.
    pending_configs.sort(key=compute_complexity_score)

    # Scarica solo i dataset effettivamente usati dalle configurazioni da eseguire.
    # Fatto qui (dopo aver costruito pending_configs) per non scaricare dataset
    # che questo PC non utilizzerà mai.
    if pending_configs:
        required_paths = {c['dataset_path'] for c in pending_configs}
        datasets_to_download = [ds for ds in base_grid_config.get('datasets', [])
                                 if ds['path'] in required_paths]
        download_config = {**base_grid_config, 'datasets': datasets_to_download}
        from dataset_manager import check_and_download_datasets_parallel
        check_and_download_datasets_parallel(download_config, max_parallel=4)

    for hyper_config in pending_configs:
        task_queue.put(hyper_config)

    if pending_configs:
        scores = [compute_complexity_score(c) for c in pending_configs]
        print(f"Complexity score range: {scores[0]:.1f} (min) -> {scores[-1]:.1f} (max)")

    print("=" * 80)
    print(f"Generated {total_generated_configs} total configurations.")
    print(f"Skipped {total_generated_configs - configs_to_run_count} already executed or redundant configurations.")
    print(f"Adding {configs_to_run_count} new unique configurations to the execution queue (sorted by complexity).")
    print("=" * 80)

    if configs_to_run_count == 0:
        print("=== No new configurations to run. Exiting. ===")
        return

    # Kill any leftover FCGS processes from previous runs before writing the new PID.
    # This prevents zombie workers that survive a partial stop from competing for GPU/CSV.
    _pid_file = os.path.join(_PROJECT_ROOT, 'fcgs.pid')
    if os.path.exists(_pid_file):
        try:
            old_pid = int(open(_pid_file).read().strip())
            if old_pid != os.getpid():
                print(f"[Startup] Found stale fcgs.pid ({old_pid}). Killing old process tree...")
                try:
                    import psutil as _psutil
                    try:
                        old_proc = _psutil.Process(old_pid)
                        for child in old_proc.children(recursive=True):
                            try:
                                child.kill()
                            except _psutil.NoSuchProcess:
                                pass
                        old_proc.kill()
                    except _psutil.NoSuchProcess:
                        pass
                    print(f"[Startup] Old process tree killed.")
                except ImportError:
                    # fallback: platform kill
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(old_pid)],
                                       capture_output=True)
                    else:
                        try:
                            os.kill(old_pid, _signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        except Exception as _e:
            print(f"[Startup] Could not kill old process from fcgs.pid: {_e}")

    with open(_pid_file, 'w') as _f:
        _f.write(str(os.getpid()))

    telegram_token = base_grid_config.get('telegram_bot_token', '')
    telegram_chat_id = base_grid_config.get('telegram_chat_id', '')
    notification_interval = int(base_grid_config.get('notification_interval', 10))
    heavy_threshold = int(base_grid_config.get('heavy_client_threshold', HEAVY_CLIENT_THRESHOLD))

    completed_count = multiprocessing.Value('i', 0)
    last_notified = multiprocessing.Value('i', 0)
    error_count = multiprocessing.Value('i', 0)
    fail_count = multiprocessing.Value('i', 0)
    notify_lock = multiprocessing.Lock()
    heavy_lock = multiprocessing.Lock()
    heavy_running = multiprocessing.Value('i', 0)
    active_normal_workers = multiprocessing.Value('i', 0)
    start_time = time.time()

    _main_logger = _setup_main_logger(pc_name, _base_log)

    num_workers = _compute_num_workers(base_grid_config)

    if telegram_token and telegram_chat_id:
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS avviata — {pc_name}</b>\n"
                      f"Config da eseguire: <b>{configs_to_run_count}</b>\n"
                      f"Workers paralleli: {num_workers}\n"
                      f"Notifica ogni {notification_interval} completamenti.",
                      logger=_main_logger)
    # Pre-download weights for all models in the search space before spawning workers
    pre_download_model_weights(set(model_specific_search_space.keys()))

    processes = []
    print(f"\n=== Starting {num_workers} Grid Search workers ===")
    base_port = base_grid_config['port']
    gpu_blacklist = base_grid_config.get('gpu_blacklist', [])
    worker_args = (task_queue, base_port, csv_lock, completed_count, last_notified,
                   configs_to_run_count, start_time, notify_lock, notification_interval,
                   telegram_token, telegram_chat_id, gpu_blacklist, error_count, fail_count,
                   heavy_lock, heavy_running, active_normal_workers, heavy_threshold)
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
                  start_time, pc_name, heartbeat_interval, heartbeat_stop,
                  error_count, fail_count, _main_logger),
            daemon=True
        )
        heartbeat_thread.start()

    task_queue.join()
    heartbeat_stop.set()
    for _ in range(num_workers):
        task_queue.put(None)
    for process in processes:
        process.join()

    if os.path.exists(_pid_file):
        os.remove(_pid_file)

    elapsed = time.time() - start_time
    print("\n=== All parallel executions have finished. ===")
    if telegram_token and telegram_chat_id:
        send_telegram(telegram_token, telegram_chat_id,
                      f"<b>FCGS completata — {pc_name}</b>\n"
                      f"Config eseguite: {completed_count.value}/{configs_to_run_count}\n"
                      f"Tempo totale: {_format_duration(elapsed)}",
                      logger=_main_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', type=str, default=GRID_SEARCH_CONFIG_PATH)
    args = parser.parse_args()
    GRID_SEARCH_CONFIG_PATH = args.config
    multiprocessing.set_start_method('spawn', force=True)
    main()