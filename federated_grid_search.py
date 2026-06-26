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
import argparse
from pathlib import Path

import csv
import os
from typing import Dict, List, Any, Generator, Set, FrozenSet, Tuple, Optional
import multiprocessing
import multiprocessing.pool

import federated_server
import run_multiple_clients


# Pool con worker non-daemon: i Pool standard hanno worker daemon che non possono
# creare processi figli. Questa subclass aggira il limite mantenendo l'intera
# interfaccia di Pool (imap_unordered, initializer, ecc.).
class _NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        proc.__class__ = _NoDaemonProcess
        return proc

_PROJECT_ROOT = Path(__file__).parent
_CONFIGS_DIR = _PROJECT_ROOT / 'configs'
GRID_SEARCH_CONFIG_PATH = str(_CONFIGS_DIR / 'grid_search_config.json')


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"



def _setup_main_logger(base_log_path: str) -> logging.Logger:
    logger = logging.getLogger("FCGS-Main")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_dir = os.path.join(str(_PROJECT_ROOT), base_log_path)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'main.log'), mode='w', encoding='utf-8')
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
    items = []
    for key in sorted(list(keys)):
        if key in config and config[key] is not None and config[key] != '':
            items.append((key, str(config[key])))
    return frozenset(items)


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
        try:
            result = subprocess.run(
                ['ss', '-tlnp', f'sport = :{port}'], capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if f':{port}' in line:
                    m = re.search(r'pid=(\d+)', line)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'], capture_output=True, text=True, timeout=10)
            first = result.stdout.strip().split('\n')[0]
            if first.isdigit():
                return int(first)
        except Exception:
            pass
    return 0


def _is_python_process(pid: int) -> bool:
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                capture_output=True, text=True, timeout=10)
            return 'python' in result.stdout.lower()
        except Exception:
            return False
    else:
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().decode('utf-8', errors='replace').replace('\x00', ' ')
            return 'python' in cmdline.lower()
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'comm='], capture_output=True, text=True, timeout=10)
            return 'python' in result.stdout.lower()
        except Exception:
            return False


def _kill_process(pid: int) -> None:
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True, timeout=10)
    else:
        os.kill(pid, _signal.SIGKILL)


def _port_is_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return True
    except OSError:
        return False


def find_free_port(host: str, preferred_port: int, max_scan: int = 200) -> int:
    for candidate in range(preferred_port, preferred_port + max_scan):
        if not _port_is_listening(host, candidate):
            return candidate
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_port_free(host: str, port: int, timeout: int = 120) -> bool:
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
                    time.sleep(5)
                    continue
                except Exception as e:
                    print(f"  [PortFree] Could not terminate PID {pid}: {e}")
        time.sleep(3)
    return False


def pre_download_model_weights(model_names: set) -> None:
    try:
        import torchvision.models as tv_models
    except ImportError:
        print("[PreDownload] torchvision not available, skipping.")
        return
    weight_map = {
        'GoogLeNet': (tv_models.googlenet, tv_models.GoogLeNet_Weights.IMAGENET1K_V1),
        'AlexNet':   (tv_models.alexnet,   tv_models.AlexNet_Weights.IMAGENET1K_V1),
        'ResNet18':  (tv_models.resnet18,  tv_models.ResNet18_Weights.IMAGENET1K_V1),
        'ResNet34':  (tv_models.resnet34,  tv_models.ResNet34_Weights.IMAGENET1K_V1),
        'ResNet50':  (tv_models.resnet50,  tv_models.ResNet50_Weights.IMAGENET1K_V1),
        'ResNet101': (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V1),
    }
    print("\n=== Pre-downloading model weights ===")
    for name in sorted(model_names):
        if name not in weight_map:
            print(f"[PreDownload] {name}: no pretrained weights (skipping).")
            continue
        fn, weights_enum = weight_map[name]
        print(f"[PreDownload] Checking {name}...", flush=True)
        try:
            _m = fn(weights=weights_enum)
            del _m
            print(f"[PreDownload] {name} ready.")
        except Exception as e:
            print(f"[PreDownload] WARNING: could not pre-load {name}: {e}")
    print("=== Pre-download complete ===\n")


def wait_for_server_ready(url: str, timeout: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url).status_code == 200:
                return True
        except ConnectionError:
            time.sleep(1)
    return False


def load_json(filename: str) -> Dict:
    with open(filename) as f:
        return json.load(f)


def append_results_to_csv(csv_path: str, run_summary: Dict) -> None:
    all_keys = set(run_summary.keys())
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            all_keys.update(header)
    file_is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = sorted(list(all_keys))
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if file_is_empty:
            writer.writeheader()
        writer.writerow(run_summary)


_MODEL_COST = {
    'ConvNet':   1.0,
    'AlexNet':   2.0,
    'ResNet18':  2.5,
    'GoogLeNet': 3.0,
    'ResNet34':  3.5,
    'ResNet50':  4.5,
    'ResNet101': 6.0,
}


def _adaptive_startup_timeout(config: dict, base: int = 600) -> int:
    t = base
    images = config.get('num_images', 1000)
    t += max(0, int((images - 1000) / 500) * 30)
    model_extra = {
        'AlexNet': 60, 'GoogLeNet': 120,
        'ResNet18': 120, 'ResNet34': 180, 'ResNet50': 240, 'ResNet101': 360,
    }
    t += model_extra.get(config.get('model_name', ''), 0)
    if config.get('encryption_mode') == 'homomorphic':
        t += 300
    return min(t, 3600)


def compute_complexity_score(cfg: Dict) -> float:
    score = (cfg.get('num_clients', 2)
             * cfg.get('global_epoch', 5)
             * cfg.get('local_epoch', 2))
    image_size = cfg.get('image_size', 128)
    score *= (image_size / 128.0) ** 2
    num_images = cfg.get('num_images', 5000)
    score *= (num_images / 5000.0)
    score *= _MODEL_COST.get(cfg.get('model_name', 'ConvNet'), 2.0)
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


# ---------------------------------------------------------------------------
# Server / TA subprocess targets
# ---------------------------------------------------------------------------

def _server_process_target(config: Dict, result_queue: multiprocessing.Queue) -> None:
    _log_dir_base = config.get('log_dir', 'logs')
    _crash_log = os.path.join(_log_dir_base, time.strftime('%d%m'),
                              'FL-Server-LOG', f'{time.strftime("%m%d%H%M")}_crash.log')
    try:
        server = federated_server.FederatedServer(config)
        try:
            server.run()
        except SystemExit:
            pass
        if server.aggregator.run_summary is None and server.aggregator.metrics_history:
            try:
                server.aggregator.save_results()
            except Exception:
                pass
        result_queue.put(server.aggregator.get_run_summary())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ServerProc] Error: {e}\n{tb}")
        try:
            os.makedirs(os.path.dirname(_crash_log), exist_ok=True)
            with open(_crash_log, 'w', encoding='utf-8') as cf:
                cf.write(f"SERVER CRASH — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"Dataset: {config.get('dataset_name')} | Model: {config.get('model_name')}\n\n{tb}")
        except Exception:
            pass
        result_queue.put(None)


def _ta_process_target(config: Dict) -> None:
    try:
        from trusted_authority import TrustedAuthority
        ta = TrustedAuthority(
            host=config['ip_address'],
            port=config['ta_port'],
            he_backend=config.get('he_backend', 'paillier')
        )
        ta.run()
    except Exception as e:
        print(f"[TAProc] Error: {e}\n{traceback.format_exc()}")


def _cleanup_after_task(server_process: multiprocessing.Process, port: int,
                         host: str, worker_id: int, max_port_wait: int = 30) -> bool:
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

    deadline = time.time() + max_port_wait
    while time.time() < deadline:
        if not _port_is_listening(host, port):
            return True
        time.sleep(1)

    pid = _get_pid_on_port(port)
    if pid and pid != os.getpid():
        print(f"[W{worker_id}] Port {port} still held by PID {pid}. Force-killing.")
        try:
            _kill_process(pid)
            time.sleep(3)
        except Exception as e:
            print(f"[W{worker_id}] Could not kill PID {pid}: {e}")

    if _port_is_listening(host, port):
        print(f"[W{worker_id}] WARNING: port {port} could not be freed.")
        return False
    return True


# ---------------------------------------------------------------------------
# Pool worker initializer — assegna GPU in base al PID
# ---------------------------------------------------------------------------

def _worker_init(gpu_blacklist: list) -> None:
    try:
        import torch as _t
        if _t.cuda.is_available():
            usable = [i for i in range(_t.cuda.device_count()) if i not in (gpu_blacklist or [])]
            if usable:
                gpu_idx = os.getpid() % len(usable)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(usable[gpu_idx])
                print(f"[Worker PID {os.getpid()}] GPU {usable[gpu_idx]} assigned.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Singolo esperimento FL — eseguito dai worker del Pool
# ---------------------------------------------------------------------------

def run_single_experiment(config: Dict) -> Optional[Dict]:
    """Esegue un singolo esperimento FL. Chiamata come worker di multiprocessing.Pool."""
    import gc

    dataset_name = config.get('dataset_name', 'N/A')
    model_name = config.get('model_name', 'N/A')

    run_id = f"{dataset_name}__{model_name}__{os.getpid()}__{int(time.time() * 1000) % 100000}"
    splitting_dir = os.path.join(str(_PROJECT_ROOT), config['base_split_data_path'], run_id)
    log_dir       = os.path.join(str(_PROJECT_ROOT), config['base_log_path'], run_id)
    plot_dir      = os.path.join(str(_PROJECT_ROOT), config['base_plot_path'], run_id)
    metrics_dir   = os.path.join(str(_PROJECT_ROOT), config['base_csv_path'], 'runs')

    os.makedirs(log_dir,     exist_ok=True)
    os.makedirs(plot_dir,    exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    config = {**config,
              'splitting_dir':          splitting_dir,
              'log_dir':                log_dir,
              'plot_dir':               plot_dir,
              'run_metrics_output_path': metrics_dir,
              'worker_id':              os.getpid() % 100}

    try:
        print(f"[{dataset_name}|{model_name}] START")

        # Port selection
        for _retry in range(5):
            config['port'] = find_free_port(config['ip_address'], config['port'])
            time.sleep(0.1 * (_retry + 1))
            if not _port_is_listening(config['ip_address'], config['port']):
                break

        if not wait_for_port_free(config['ip_address'], config['port'], timeout=180):
            print(f"[{dataset_name}|{model_name}] Port {config['port']} busy after 180s. Skipping.")
            return None

        # Dataset check
        _dataset_abs = os.path.join(str(_PROJECT_ROOT), config['dataset_path'])
        from dataset_manager import _dataset_exists as _has_images
        if not os.path.isdir(_dataset_abs) or not _has_images(_dataset_abs):
            print(f"[{dataset_name}|{model_name}] Dataset not found: {_dataset_abs}. Skipping.")
            return None

        # Dataset split
        os.makedirs(splitting_dir, exist_ok=True)
        from data_splitter import DatasetSplitter
        actual_clients = DatasetSplitter(splitting_dir, _dataset_abs,
                                         config['num_clients']).split_dataset()
        if actual_clients == 0:
            print(f"[{dataset_name}|{model_name}] No valid clients after split. Skipping.")
            _safe_rmtree(splitting_dir)
            return None
        config['MIN_NUM_WORKERS'] = actual_clients

        # Adaptive startup timeout
        config['server_startup_timeout'] = _adaptive_startup_timeout(
            config, config.get('server_startup_timeout', 600))

        # Server subprocess
        result_queue = multiprocessing.Queue()
        server_process = multiprocessing.Process(
            target=_server_process_target,
            args=(config, result_queue),
        )
        server_process.start()

        server_url = f"http://{config['ip_address']}:{config['port']}"
        if wait_for_server_ready(server_url, timeout=config.get('server_ready_timeout', 60)):
            run_multiple_clients.main(config, already_split=True)
        else:
            print(f"[{dataset_name}|{model_name}] Server not ready in time.")

        server_process.join(timeout=config.get('server_join_timeout', 120))
        _cleanup_after_task(server_process, config['port'], config['ip_address'], os.getpid())

        # GPU + memory cleanup
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        try:
            run_summary = result_queue.get(timeout=5)
        except Exception:
            run_summary = None

        if run_summary is None:
            print(f"[{dataset_name}|{model_name}] No result (server crash or NaN loss).")
            time.sleep(10)

        return run_summary

    except Exception as e:
        print(f"[{dataset_name}|{model_name}] ERROR: {e}\n{traceback.format_exc()}")
        return None
    finally:
        if os.path.exists(splitting_dir):
            _safe_rmtree(splitting_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FCGS Federated Grid Search')
    parser.add_argument(
        'config', nargs='?', default=GRID_SEARCH_CONFIG_PATH,
        help=f'File di configurazione JSON (default: {GRID_SEARCH_CONFIG_PATH})'
    )
    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"ERRORE: config non trovata: {config_path}")
        sys.exit(1)

    print(f"=== Configurazione: {config_path} ===")
    base_config = load_json(config_path)

    pc_name = _socket.gethostname()
    _pc_results_dir = f"results/results_{pc_name}"
    base_config['base_csv_path']        = f"{_pc_results_dir}/csv"
    base_config['base_log_path']        = f"{_pc_results_dir}/log"
    base_config['base_plot_path']       = f"{_pc_results_dir}/plots"
    base_config['base_split_data_path'] = f"{_pc_results_dir}/run_dataset"
    base_config['pc_name']              = pc_name

    _base_csv   = base_config['base_csv_path']
    _base_log   = base_config['base_log_path']
    _base_plot  = base_config['base_plot_path']
    _base_split = base_config['base_split_data_path']

    base_csv_path = os.path.join(str(_PROJECT_ROOT), _base_csv)
    os.makedirs(base_csv_path,                                              exist_ok=True)
    os.makedirs(os.path.join(str(_PROJECT_ROOT), _base_log),               exist_ok=True)
    os.makedirs(os.path.join(str(_PROJECT_ROOT), _base_plot),              exist_ok=True)
    os.makedirs(os.path.join(str(_PROJECT_ROOT), _base_split),             exist_ok=True)

    logger = _setup_main_logger(_base_log)
    logger.info(f"PC: {pc_name}  →  results in: {_pc_results_dir}")

    shared_csv_path = os.path.join(base_csv_path, f'federated_grid_search_results_{pc_name}.csv')
    base_config['shared_csv_path'] = shared_csv_path

    # ── Fingerprint dedup ─────────────────────────────────────────────────
    common_search_space        = base_config['common_search_space']
    model_specific_search_space = base_config['model_specific_search_space']

    all_possible_search_keys = set(common_search_space.keys())
    for model_params in model_specific_search_space.values():
        all_possible_search_keys.update(model_params.keys())
    all_possible_search_keys.update(['dataset_name', 'model_name'])

    executed_fingerprints: Set[FrozenSet[Tuple[str, str]]] = set()
    dedup_sources = [
        str(_PROJECT_ROOT / 'all_results.csv'),  # storico consolidato
        shared_csv_path,                          # run corrente
    ]
    total_loaded = 0
    for csv_path in dedup_sources:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            continue
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name_from_row = row.get('model_name') or ''
                if row.get('aggregation_algorithm') != 'FedProx':
                    row['fedprox_mu'] = '0.0'
                if row.get('encryption_mode') == 'no_encryption':
                    row['he_backend'] = 'N/A'
                if any(m in model_name_from_row for m in ('ResNet', 'GoogLeNet', 'AlexNet')):
                    row.setdefault('image_size', '224')
                    row.setdefault('convnet_hidden1', '-1')
                    row.setdefault('convnet_hidden2', '-1')
                elif 'ConvNet' in model_name_from_row:
                    row.setdefault('num_custom_layers', '-1')
                executed_fingerprints.add(get_config_fingerprint(row, all_possible_search_keys))
        count = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1
        logger.info(f"Loaded {count} fingerprints from {csv_path}")
        total_loaded += count
    logger.info(f"Total unique fingerprints: {len(executed_fingerprints)} (from {total_loaded} rows)")

    # ── Config generation ─────────────────────────────────────────────────
    fixed_params = {k: v for k, v in base_config.items()
                    if k not in ('datasets', 'common_search_space', 'model_specific_search_space')}
    fixed_params['shared_csv_path'] = shared_csv_path

    pending_configs: List[Dict] = []
    total_generated = 0

    for dataset_info in base_config['datasets']:
        for model_name, specific_params in model_specific_search_space.items():
            model_base = fixed_params.copy()
            model_base.update({
                'dataset_name': dataset_info['name'],
                'dataset_path': dataset_info['path'],
                'num_classes':  dataset_info['num_classes'],
                'model_name':   model_name,
                'num_images':   dataset_info.get('num_images', 5000),
            })
            current_search_space = {**common_search_space, **specific_params}

            for hyper_config in generate_configurations(model_base, current_search_space):
                total_generated += 1
                if hyper_config.get('aggregation_algorithm') != 'FedProx':
                    hyper_config['fedprox_mu'] = 0.0
                if hyper_config.get('encryption_mode') == 'no_encryption':
                    hyper_config['he_backend'] = 'N/A'
                if any(m in model_name for m in ('ResNet', 'GoogLeNet', 'AlexNet')):
                    hyper_config['image_size'] = 224
                    hyper_config.setdefault('convnet_hidden1', -1)
                    hyper_config.setdefault('convnet_hidden2', -1)
                elif 'ConvNet' in model_name:
                    hyper_config.setdefault('num_custom_layers', -1)

                fingerprint = get_config_fingerprint(hyper_config, all_possible_search_keys)
                if fingerprint in executed_fingerprints:
                    continue

                executed_fingerprints.add(fingerprint)
                pending_configs.append(hyper_config)

    pending_configs.sort(key=compute_complexity_score)

    if pending_configs:
        scores = [compute_complexity_score(c) for c in pending_configs]
        print(f"Complexity score range: {scores[0]:.1f} (min) -> {scores[-1]:.1f} (max)")

    print("=" * 80)
    print(f"Generated {total_generated} total configurations.")
    print(f"Skipped   {total_generated - len(pending_configs)} already executed.")
    print(f"Running   {len(pending_configs)} new configurations (sorted by complexity).")
    print("=" * 80)

    if not pending_configs:
        print("=== No new configurations. Exiting. ===")
        return

    # ── Dataset download ──────────────────────────────────────────────────
    required_paths = {c['dataset_path'] for c in pending_configs}
    datasets_to_download = [ds for ds in base_config.get('datasets', [])
                             if ds['path'] in required_paths]
    download_config = {**base_config, 'datasets': datasets_to_download}
    from dataset_manager import check_and_download_datasets_parallel
    download_ok = check_and_download_datasets_parallel(
        download_config, max_parallel=2, logger=logger)
    if not download_ok:
        logger.warning("Download incompleto al primo tentativo. Riprovo tra 60s...")
        time.sleep(60)
        download_ok = check_and_download_datasets_parallel(
            download_config, max_parallel=2, logger=logger)
    if not download_ok:
        from dataset_manager import _dataset_exists as _has_imgs
        failed_paths = {ds['path'] for ds in datasets_to_download
                        if not _has_imgs(os.path.join(str(_PROJECT_ROOT), ds['path']))}
        if failed_paths:
            logger.warning(f"Dataset non disponibili: {failed_paths}. Rimosse le config corrispondenti.")
            before = len(pending_configs)
            pending_configs = [c for c in pending_configs if c['dataset_path'] not in failed_paths]
            logger.warning(f"Config rimosse: {before - len(pending_configs)}. Rimangono: {len(pending_configs)}.")
        if not pending_configs:
            logger.error("Nessuna configurazione rimasta dopo rimozione dataset mancanti.")
            sys.exit(1)

    pre_download_model_weights(set(model_specific_search_space.keys()))

    # ── Kill old FCGS instance ────────────────────────────────────────────
    pid_file = str(_PROJECT_ROOT / 'fcgs.pid')
    if os.path.exists(pid_file):
        try:
            old_pid = int(open(pid_file).read().strip())
            if old_pid != os.getpid():
                print(f"[Startup] Stale fcgs.pid ({old_pid}). Killing old process tree...")
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
                except ImportError:
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(old_pid)],
                                       capture_output=True)
                    else:
                        try:
                            os.kill(old_pid, _signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        except Exception as _e:
            print(f"[Startup] Could not kill old process: {_e}")

    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    # ── Run ───────────────────────────────────────────────────────────────
    gpu_blacklist = base_config.get('gpu_blacklist', [])
    n_workers     = int(base_config.get('num_parallel_executions', 3))
    start_time    = time.time()

    logger.info(f"Avvio grid search: {len(pending_configs)} config, {n_workers} worker(s)")
    completed = errors = 0

    pool = _NoDaemonPool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(gpu_blacklist,),
    )

    def _shutdown(signum=None, frame=None):
        logger.info("Shutdown signal received. Terminating pool...")
        pool.terminate()
        if os.path.exists(pid_file):
            os.remove(pid_file)
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _shutdown)
    _signal.signal(_signal.SIGINT,  _shutdown)

    try:
        for run_summary in pool.imap_unordered(run_single_experiment, pending_configs):
            if run_summary:
                append_results_to_csv(shared_csv_path, run_summary)
            else:
                errors += 1
            completed += 1
            logger.info(f"[{completed}/{len(pending_configs)}] Done. Errors so far: {errors}")
    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    logger.info(f"=== Grid search completata: {completed}/{len(pending_configs)} config, "
                f"{errors} errori, tempo: {_format_duration(elapsed)} ===")

    if os.path.exists(pid_file):
        os.remove(pid_file)

    logger.info("=== FCGS Grid Search Complete ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
