import os
import time
import threading
import json
from typing import Dict, List, Tuple

from data_splitter import DatasetSplitter
from federated_client import FederatedClient


def load_json(filename: str) -> Dict:
    with open(filename) as f:
        return json.load(f)


def main(config: Dict, already_split: bool = False) -> List[FederatedClient]:
    """
    Split dataset (unless already_split=True), launch client threads, wait for completion.

    Returns the list of FederatedClient instances so the caller can verify
    all clients are dead before moving to the next run.

    Each client thread calls client.connect_to_server() which blocks until
    the server sends 'shutdown' and the socket disconnects.  If a thread
    does not finish within the computed deadline we force-disconnect its
    socket and let the daemon thread die on its own.

    already_split=True: lo split è già stato eseguito dal chiamante prima dell'avvio
    del server; config['MIN_NUM_WORKERS'] contiene già il numero reale di client validi.
    """
    num_clients  = config['num_clients']
    splitting_dir = config['splitting_dir']
    source_images_dir = config['dataset_path']

    # Max time a single client can legitimately run
    global_epoch  = int(config.get('global_epoch', 30))
    round_timeout = int(config.get('round_timeout', 1800))
    startup_buf   = int(config.get('server_startup_timeout', 600))
    client_max_wait = global_epoch * round_timeout + startup_buf  # generous upper bound

    if not already_split:
        splitter = DatasetSplitter(
            output_base_dir=splitting_dir,
            source_images_dir=source_images_dir,
            num_clients=num_clients
        )
        actual_clients = splitter.split_dataset()
        if actual_clients < num_clients:
            print(f"[run_multiple_clients] Avvio {actual_clients} client (richiesti {num_clients}, "
                  f"limitati dal dataset). Il server accettera' {actual_clients} client.")
            config['MIN_NUM_WORKERS'] = actual_clients
    else:
        # Split già eseguito da federated_grid_search.py prima del server —
        # MIN_NUM_WORKERS è già il conteggio reale post-I/O.
        actual_clients = config.get('MIN_NUM_WORKERS', num_clients)

    # ── Create all FederatedClient instances BEFORE starting threads ──────────
    # This gives us a handle on each client's .sio so we can force-disconnect
    # any thread that is still alive after the run should have ended.
    client_instances: List[FederatedClient] = []
    threads: List[threading.Thread] = []

    for i in range(actual_clients):
        client_dataset_path = os.path.join(splitting_dir, f'client_{i}')
        try:
            client = FederatedClient(config, client_dataset_path)
            client_instances.append(client)
            thread = threading.Thread(
                target=client.connect_to_server,
                name=f"FCClient-{i}",
                daemon=True   # daemon: dies with the worker process even if join times out
            )
            thread.start()
            threads.append(thread)
            time.sleep(2)
        except Exception as e:
            print(f"[run_multiple_clients] Error creating client_{i}: {e}")
            import traceback
            traceback.print_exc()

    # ── Join threads with timeout, force-disconnect survivors ─────────────────
    for i, (thread, client) in enumerate(zip(threads, client_instances)):
        thread.join(timeout=client_max_wait)
        if thread.is_alive():
            print(f"[run_multiple_clients] client_{i} still alive after {client_max_wait}s. "
                  f"Force-disconnecting (sio.connected={client.sio.connected})...")
            try:
                if client.sio.connected:
                    client.sio.disconnect()
            except Exception as disc_err:
                print(f"[run_multiple_clients] disconnect error for client_{i}: {disc_err}")
            thread.join(timeout=30)
            if thread.is_alive():
                print(f"[run_multiple_clients] WARNING: client_{i} still alive after "
                      f"forced disconnect. Thread is daemon — will die with the process.")

    return client_instances


if __name__ == '__main__':
    print("This script is intended to be run by 'federated_grid_search.py'.")
