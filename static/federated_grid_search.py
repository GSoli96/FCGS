import json
import threading
import time
import itertools
import traceback
import socket as _socket
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

NUM_PARALLEL_EXECUTIONS = 12
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'
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

def wait_for_port_free(host: str, port: int, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) != 0:
                    return True
        except Exception:
            return True
        time.sleep(3)
    return False


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


def start_server_thread(config: Dict, server_instance_ref: List) -> None:
    worker_id = config.get('worker_id', 'N/A')
    port = config['port']
    try:
        server = federated_server.FederatedServer(config)
        server_instance_ref.append(server)
        server.run()
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"[W{worker_id}] SERVER CRASH on port {port}: {e}\n{tb_str}")
        send_telegram(
            config.get('telegram_bot_token', ''),
            config.get('telegram_chat_id', ''),
            f"<b>⚠️ FCGS SERVER CRASH — Worker {worker_id} ({PCNAME.name})</b>\n"
            f"Port: {port}\n<code>{str(e)[:400]}</code>"
        )


def start_ta_thread(config: Dict, ta_instance_ref: List) -> None:
    worker_id = config.get('worker_id', 'N/A')
    try:
        ta = TrustedAuthority(host=config['ip_address'], port=config['ta_port'])
        ta_instance_ref.append(ta)
        ta.run()
    except Exception as e:
        print(f"[W{worker_id}] TA CRASH: {e}\n{traceback.format_exc()}")


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

        try:
            run_identifier = f"{dataset_name}_run_{worker_id}_{model_name}"
            worker_splitting_dir = os.path.join(config['base_split_data_path'] + f'_{pc_name}', run_identifier)
            worker_log_dir = os.path.join(config['base_log_path'] + f'_{pc_name}', run_identifier)
            worker_plot_dir = os.path.join(config['base_plot_path'] + f'_{pc_name}', run_identifier)
            worker_metrics_dir = os.path.join(config['base_csv_path'] + f'_{pc_name}', "runs")
            os.makedirs(worker_splitting_dir, exist_ok=True)
            os.makedirs(worker_log_dir, exist_ok=True)
            os.makedirs(worker_plot_dir, exist_ok=True)
            os.makedirs(worker_metrics_dir, exist_ok=True)

            config['worker_id'] = worker_id
            config['port'] = base_port + worker_id * 2
            config['ta_port'] = base_port + worker_id * 2 + 1
            config['splitting_dir'] = worker_splitting_dir
            config['log_dir'] = worker_log_dir
            config['plot_dir'] = worker_plot_dir
            config['run_metrics_output_path'] = worker_metrics_dir
            config["MIN_NUM_WORKERS"] = int(config['num_clients'] * config['models_percentage'])

            ta_instance_ref = []
            server_instance_ref = []
            use_encryption = config.get('encryption_mode', 'no_encryption') != 'no_encryption'
            ta_thread = None
            if use_encryption:
                ta_thread = threading.Thread(target=start_ta_thread, args=(config, ta_instance_ref))
                ta_thread.start()
                time.sleep(3)

            if not wait_for_port_free(config['ip_address'], config['port'], timeout=180):
                msg = f"[W{worker_id}] Port {config['port']} still occupied after 180s. Skipping {dataset_name}|{model_name}."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Port busy — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Porta {config['port']} occupata dopo 180s. Skipping.")
                continue

            server_thread = threading.Thread(target=start_server_thread, args=(config, server_instance_ref))
            server_thread.start()
            server_url = f"http://{config['ip_address']}:{config['port']}"
            if wait_for_server_ready(server_url):
                run_multiple_clients.main(config)
            else:
                msg = f"[W{worker_id}] Server non avviato per {dataset_name}|{model_name}. Skipping."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Server timeout — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Il server non ha risposto entro 60s.")
            server_thread.join(timeout=120)
            if server_thread.is_alive():
                msg = f"[W{worker_id}] Server thread ancora vivo dopo 120s per {dataset_name}|{model_name}."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Server hung — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Server thread non terminato dopo 120s.")
                wait_for_port_free(config['ip_address'], config['port'], timeout=120)
            if use_encryption and ta_thread is not None:
                try:
                    sio_client = socketio.Client(reconnection=False)
                    sio_client.connect(f"http://{config['ip_address']}:{config['ta_port']}", transports=['websocket'])
                    sio_client.emit('shutdown_ta')
                    time.sleep(1)
                    sio_client.disconnect()
                except Exception as e:
                    print(f"[W{worker_id}] Could not shutdown TA: {e}")
                ta_thread.join(timeout=15)
            if server_instance_ref:
                run_summary = server_instance_ref[0].aggregator.get_run_summary()
                if run_summary:
                    append_results_to_csv(config['shared_csv_path'], run_summary, csv_lock)
                else:
                    msg = f"[W{worker_id}] run_summary None per {dataset_name}|{model_name}."
                    print(msg)
                    if telegram_token and telegram_chat_id:
                        send_telegram(telegram_token, telegram_chat_id,
                                      f"<b>⚠️ FCGS run_summary None — Worker {worker_id} ({pc_name})</b>\n"
                                      f"Config: {dataset_name} | {model_name}\n"
                                      f"Training completato ma nessun risultato salvato.")
            else:
                msg = f"[W{worker_id}] server_instance_ref vuoto per {dataset_name}|{model_name} (crash avvio server)."
                print(msg)
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id,
                                  f"<b>⚠️ FCGS Server crash — Worker {worker_id} ({pc_name})</b>\n"
                                  f"Config: {dataset_name} | {model_name}\n"
                                  f"Server crashato prima di inizializzarsi.")
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
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()
    pc_name = PCNAME.name
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

    if task_queue.empty():
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