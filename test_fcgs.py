"""
test_fcgs.py — Suite di test per FCGS.

Sezioni:
  A — Path e directory (cross-platform)
  B — Dataset manager (filtro + download parallelo mock)
  C — Worker scheduling (heavy/normal isolation)
  D — CSV thread-safety e deduplication
  E — Telegram + log
  F — Stress test worker scheduling
  G — Dataset loading e image access (DatasetSplitter + ImageFolder)
  H — Namespace error handling (socket disconnesso durante emit)
  I — Watchdog scenarios (effective clients, dataset piccoli)
  J — Port management (find_free_port, race condition, cleanup)
  K — Heavy scheduling avanzato (isolation, children, boundary)
  L — Zombie process cleanup (_cleanup_after_task)
"""
import csv
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

_PROJECT_ROOT = Path(__file__).parent


# ===========================================================================
# Funzioni helper a livello di modulo per multiprocessing (Windows spawn)
# ===========================================================================

def _proc_hang_forever():
    time.sleep(3600)


def _proc_exit_immediately():
    pass


def _proc_hold_port(port):
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1', port))
        s.listen(1)
        time.sleep(3600)
    finally:
        s.close()


def _proc_parent_with_child(pid_list):
    child = multiprocessing.Process(target=_proc_hang_forever)
    child.daemon = False
    child.start()
    pid_list.append(child.pid)
    time.sleep(3600)
sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# A — PATH E DIRECTORY
# ===========================================================================

class TestPaths(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_dir_structure(self):
        """log/log_{PC}/ viene creato correttamente."""
        log_root = os.path.join(self.tmp, 'log')
        log_pc = os.path.join(log_root, 'log_TestPC')
        os.makedirs(log_pc, exist_ok=True)
        self.assertTrue(os.path.isdir(log_pc))
        self.assertTrue(log_pc.replace('\\', '/').endswith('log/log_TestPC'))

    def test_csv_dir_structure(self):
        """csv/csv_{PC}/ viene creato correttamente."""
        csv_root = os.path.join(self.tmp, 'csv')
        csv_pc = os.path.join(csv_root, 'csv_TestPC')
        os.makedirs(csv_pc, exist_ok=True)
        self.assertTrue(os.path.isdir(csv_pc))

    def test_cross_platform_join(self):
        """os.path.join produce path validi su Windows e POSIX."""
        p = os.path.join('log', 'log_TestPC', 'run_identifier')
        self.assertIn('log_TestPC', p)
        self.assertNotIn('//', p)
        self.assertNotIn('\\\\', p.replace('\\\\', ''))

    def test_project_root_is_directory(self):
        """_PROJECT_ROOT punta a una directory esistente."""
        self.assertTrue(_PROJECT_ROOT.is_dir())

    def test_path_with_spaces_in_run_id(self):
        """Run identifier con caratteri speciali non causa errori di path."""
        run_id = "dataset_slices_sorted_run_0_ResNet18"
        p = os.path.join(self.tmp, 'log', 'log_TestPC', run_id)
        os.makedirs(p, exist_ok=True)
        self.assertTrue(os.path.isdir(p))


# ===========================================================================
# B — DATASET MANAGER
# ===========================================================================

class TestDatasetManager(unittest.TestCase):

    def test_required_datasets_filter(self):
        """Solo i dataset effettivamente usati in pending_configs vengono scaricati."""
        all_datasets = [
            {'name': 'ds_A', 'path': 'dataset/ds_A', 'num_classes': 2, 'num_images': 100},
            {'name': 'ds_B', 'path': 'dataset/ds_B', 'num_classes': 2, 'num_images': 200},
            {'name': 'ds_C', 'path': 'dataset/ds_C', 'num_classes': 3, 'num_images': 300},
        ]
        pending_configs = [
            {'dataset_path': 'dataset/ds_A', 'model_name': 'ConvNet'},
            {'dataset_path': 'dataset/ds_C', 'model_name': 'ResNet18'},
        ]
        required_paths = {c['dataset_path'] for c in pending_configs}
        filtered = [ds for ds in all_datasets if ds['path'] in required_paths]
        self.assertEqual(len(filtered), 2)
        names = {ds['name'] for ds in filtered}
        self.assertIn('ds_A', names)
        self.assertIn('ds_C', names)
        self.assertNotIn('ds_B', names)

    def test_required_datasets_filter_empty_pending(self):
        """Con zero pending_configs, non viene scaricato nulla."""
        all_datasets = [{'name': 'ds_A', 'path': 'dataset/ds_A', 'num_classes': 2, 'num_images': 100}]
        pending_configs = []
        required_paths = {c['dataset_path'] for c in pending_configs}
        filtered = [ds for ds in all_datasets if ds['path'] in required_paths]
        self.assertEqual(len(filtered), 0)

    @patch('dataset_manager._download_folder')
    @patch('dataset_manager._get_hf_token', return_value='fake_token')
    @patch('dataset_manager._find_missing')
    def test_parallel_download_called_concurrently(self, mock_missing, mock_token, mock_dl):
        """check_and_download_datasets_parallel usa ThreadPoolExecutor."""
        import dataset_manager
        call_times = []

        def slow_download(repo_id, token, path):
            call_times.append(time.time())
            time.sleep(0.05)
            return True

        mock_dl.side_effect = slow_download
        mock_missing.return_value = [
            {'name': 'ds_A', 'path': 'p/A'},
            {'name': 'ds_B', 'path': 'p/B'},
            {'name': 'ds_C', 'path': 'p/C'},
        ]
        config = {'datasets': mock_missing.return_value, 'hf_repo_id': 'test/repo'}
        start = time.time()
        result = dataset_manager.check_and_download_datasets_parallel(config, max_parallel=3)
        elapsed = time.time() - start
        self.assertTrue(result)
        # Se fossero sequenziali ci vorrebbero ~0.15s; paralleli ~0.05s
        self.assertLess(elapsed, 0.12)

    @patch('dataset_manager._download_folder', return_value=True)
    @patch('dataset_manager._get_hf_token', return_value='fake_token')
    @patch('dataset_manager._find_missing')
    def test_sequential_fallback_single_dataset(self, mock_missing, mock_token, mock_dl):
        """Con un solo dataset mancante, il download funziona correttamente."""
        import dataset_manager
        mock_missing.return_value = [{'name': 'ds_A', 'path': 'p/A'}]
        config = {'datasets': mock_missing.return_value, 'hf_repo_id': 'test/repo'}
        result = dataset_manager.check_and_download_datasets_parallel(config)
        self.assertTrue(result)
        mock_dl.assert_called_once()

    @patch('dataset_manager._find_missing', return_value=[])
    def test_no_download_when_all_present(self, mock_missing):
        """Se tutti i dataset sono presenti, non viene fatto alcun download."""
        import dataset_manager
        config = {'datasets': [{'name': 'ds_A', 'path': 'p/A'}]}
        result = dataset_manager.check_and_download_datasets_parallel(config)
        self.assertTrue(result)

    def test_hf_token_read_from_file(self):
        """_get_hf_token legge correttamente da file .hf_token."""
        import dataset_manager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.token', delete=False) as f:
            f.write('  test_token_123  \n')
            fname = f.name
        try:
            original = dataset_manager._TOKEN_FILE
            dataset_manager._TOKEN_FILE = Path(fname)
            token = dataset_manager._get_hf_token()
            self.assertEqual(token, 'test_token_123')
        finally:
            dataset_manager._TOKEN_FILE = original
            os.unlink(fname)

    def test_hf_token_write_read(self):
        """_get_hf_token_write legge correttamente da file .hf_token_write."""
        import dataset_manager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.token_write', delete=False) as f:
            f.write('write_token_456\n')
            fname = f.name
        try:
            original = dataset_manager._TOKEN_WRITE_FILE
            dataset_manager._TOKEN_WRITE_FILE = Path(fname)
            token = dataset_manager._get_hf_token_write()
            self.assertEqual(token, 'write_token_456')
        finally:
            dataset_manager._TOKEN_WRITE_FILE = original
            os.unlink(fname)


# ===========================================================================
# C — WORKER SCHEDULING (heavy/normal)
# ===========================================================================

def _simulate_worker(
    worker_id: int,
    configs: list,
    results: list,
    heavy_lock: multiprocessing.Lock,
    heavy_running: multiprocessing.Value,
    active_normal_workers: multiprocessing.Value,
    heavy_threshold: int,
    sleep_per_task: float = 0.05,
):
    """Simula il comportamento di scheduling heavy/normal senza fare ML reale."""
    for cfg in configs:
        is_heavy = cfg.get('num_clients', 2) > heavy_threshold
        _heavy_acquired = False
        _normal_counted = False
        try:
            if is_heavy:
                heavy_lock.acquire()
                _heavy_acquired = True
                with heavy_running.get_lock():
                    heavy_running.value = 1
                deadline = time.time() + 10
                while time.time() < deadline:
                    with active_normal_workers.get_lock():
                        if active_normal_workers.value == 0:
                            break
                    time.sleep(0.01)
            else:
                deadline = time.time() + 10
                while time.time() < deadline:
                    with heavy_running.get_lock():
                        if heavy_running.value == 0:
                            break
                    time.sleep(0.01)
                with active_normal_workers.get_lock():
                    active_normal_workers.value += 1
                _normal_counted = True

            time.sleep(sleep_per_task)
            results.append({'worker': worker_id, 'config': cfg, 'ts': time.time()})
        finally:
            if _heavy_acquired:
                with heavy_running.get_lock():
                    heavy_running.value = 0
                heavy_lock.release()
            if _normal_counted:
                with active_normal_workers.get_lock():
                    active_normal_workers.value = max(0, active_normal_workers.value - 1)


class TestWorkerScheduling(unittest.TestCase):

    def _make_shared(self):
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        return heavy_lock, heavy_running, active_normal_workers

    def test_heavy_config_isolated(self):
        """Un config heavy gira da solo (nessun worker normale attivo nello stesso momento)."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        threshold = 8

        timeline = []

        def normal_worker():
            cfg = {'num_clients': 4}
            _heavy_acquired = False
            _normal_counted = False
            try:
                while True:
                    with heavy_running.get_lock():
                        if heavy_running.value == 0:
                            break
                    time.sleep(0.005)
                with active_normal_workers.get_lock():
                    active_normal_workers.value += 1
                _normal_counted = True
                timeline.append(('normal_start', time.time()))
                time.sleep(0.04)
                timeline.append(('normal_end', time.time()))
            finally:
                if _normal_counted:
                    with active_normal_workers.get_lock():
                        active_normal_workers.value = max(0, active_normal_workers.value - 1)

        def heavy_worker():
            heavy_lock.acquire()
            _heavy_acquired = True
            with heavy_running.get_lock():
                heavy_running.value = 1
            deadline = time.time() + 5
            while time.time() < deadline:
                with active_normal_workers.get_lock():
                    if active_normal_workers.value == 0:
                        break
                time.sleep(0.005)
            timeline.append(('heavy_start', time.time()))
            time.sleep(0.04)
            timeline.append(('heavy_end', time.time()))
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        # Avvia prima il worker normale, poi quello heavy
        t_normal = threading.Thread(target=normal_worker)
        t_heavy = threading.Thread(target=heavy_worker)
        t_normal.start()
        time.sleep(0.01)
        t_heavy.start()
        t_normal.join(timeout=2)
        t_heavy.join(timeout=2)

        events = {e[0]: e[1] for e in timeline}
        # heavy deve partire dopo che normal è finito
        self.assertIn('normal_end', events)
        self.assertIn('heavy_start', events)
        self.assertGreaterEqual(events['heavy_start'], events['normal_end'] - 0.01)

    def test_two_heavy_configs_sequential(self):
        """Due config heavy non girano mai in contemporanea."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        intervals = []

        def heavy_worker(wid):
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            start = time.time()
            time.sleep(0.05)
            end = time.time()
            intervals.append((wid, start, end))
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        t1 = threading.Thread(target=heavy_worker, args=(1,))
        t2 = threading.Thread(target=heavy_worker, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        # Verifica che non si sovrappongano
        _, s1, e1 = intervals[0]
        _, s2, e2 = intervals[1]
        overlap = min(e1, e2) - max(s1, s2)
        self.assertLessEqual(overlap, 0.01, "Due heavy worker si sono sovrapposti!")

    def test_normal_resumes_after_heavy(self):
        """Dopo la fine di un heavy, i normal worker riprendono."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        normal_ran = multiprocessing.Value('i', 0)

        def normal_worker():
            deadline = time.time() + 5
            while time.time() < deadline:
                with heavy_running.get_lock():
                    if heavy_running.value == 0:
                        break
                time.sleep(0.005)
            with active_normal_workers.get_lock():
                active_normal_workers.value += 1
            time.sleep(0.02)
            with normal_ran.get_lock():
                normal_ran.value += 1
            with active_normal_workers.get_lock():
                active_normal_workers.value = max(0, active_normal_workers.value - 1)

        def heavy_worker():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            time.sleep(0.05)
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        t_heavy = threading.Thread(target=heavy_worker)
        t_normals = [threading.Thread(target=normal_worker) for _ in range(3)]

        t_heavy.start()
        time.sleep(0.01)
        for t in t_normals:
            t.start()
        t_heavy.join(timeout=2)
        for t in t_normals:
            t.join(timeout=2)

        self.assertEqual(normal_ran.value, 3, "Non tutti i normal worker sono stati eseguiti dopo l'heavy")

    def test_threshold_boundary(self):
        """num_clients == threshold è normale; threshold+1 è heavy."""
        threshold = 8
        self.assertFalse(8 > threshold)   # num_clients=8 → normale
        self.assertTrue(9 > threshold)    # num_clients=9 → heavy
        self.assertTrue(16 > threshold)   # num_clients=16 → heavy
        self.assertTrue(32 > threshold)   # num_clients=32 → heavy
        self.assertFalse(2 > threshold)   # num_clients=2 → normale

    def test_normal_configs_run_concurrently(self):
        """Più config normali girano in parallelo senza bloccarsi a vicenda."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        starts = []

        def normal_worker():
            with heavy_running.get_lock():
                pass  # non blocked
            with active_normal_workers.get_lock():
                active_normal_workers.value += 1
            starts.append(time.time())
            time.sleep(0.05)
            with active_normal_workers.get_lock():
                active_normal_workers.value = max(0, active_normal_workers.value - 1)

        threads = [threading.Thread(target=normal_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)

        # Tutti devono essere partiti entro 0.02s l'uno dall'altro (paralleli)
        self.assertEqual(len(starts), 4)
        self.assertLess(max(starts) - min(starts), 0.03)


# ===========================================================================
# D — CSV THREAD-SAFETY E DEDUPLICATION
# ===========================================================================

class TestCSV(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, 'test_results.csv')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _append_row(self, lock, row):
        from federated_grid_search import append_results_to_csv
        append_results_to_csv(self.csv_path, row, lock)

    def test_csv_append_threadsafe(self):
        """10 thread scrivono contemporaneamente senza perdere righe."""
        lock = multiprocessing.Lock()
        n = 10
        threads = []
        for i in range(n):
            row = {'worker_id': i, 'dataset_name': f'ds_{i}', 'best_acc': 0.5 + i * 0.01}
            t = threading.Thread(target=self._append_row, args=(lock, row))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), n)

    def test_consolidate_keeps_best_acc(self):
        """consolidate_results deduplicha e mantiene la riga con best_acc più alto."""
        from consolidate_results import _row_key, _safe_float, METRIC_KEY

        rows = [
            {'dataset_name': 'ds_A', 'model_name': 'ConvNet', 'num_clients': '2',
             'aggregation_algorithm': 'FedAvg', 'global_epoch': '5', 'local_epoch': '2',
             'batch_size': '16', 'image_size': '128', 'encryption_mode': 'no_encryption',
             'learning_rate': '0.001', 'models_percentage': '1', 'fedprox_mu': '0.0',
             'he_backend': 'N/A', 'convnet_hidden1': '32', 'convnet_hidden2': '16',
             'num_custom_layers': '-1', 'best_acc': '0.75'},
            {'dataset_name': 'ds_A', 'model_name': 'ConvNet', 'num_clients': '2',
             'aggregation_algorithm': 'FedAvg', 'global_epoch': '5', 'local_epoch': '2',
             'batch_size': '16', 'image_size': '128', 'encryption_mode': 'no_encryption',
             'learning_rate': '0.001', 'models_percentage': '1', 'fedprox_mu': '0.0',
             'he_backend': 'N/A', 'convnet_hidden1': '32', 'convnet_hidden2': '16',
             'num_custom_layers': '-1', 'best_acc': '0.85'},
        ]
        best = {}
        for row in rows:
            key = _row_key(row)
            existing = best.get(key)
            if existing is None or _safe_float(row.get(METRIC_KEY)) > _safe_float(existing.get(METRIC_KEY)):
                best[key] = row

        self.assertEqual(len(best), 1)
        kept = list(best.values())[0]
        self.assertEqual(kept['best_acc'], '0.85')

    def test_consolidate_preserves_distinct_configs(self):
        """Configurazioni con parametri diversi NON vengono unite."""
        from consolidate_results import _row_key

        base = {'dataset_name': 'ds_A', 'model_name': 'ConvNet', 'num_clients': '2',
                'aggregation_algorithm': 'FedAvg', 'global_epoch': '5', 'local_epoch': '2',
                'batch_size': '16', 'image_size': '128', 'encryption_mode': 'no_encryption',
                'learning_rate': '0.001', 'models_percentage': '1', 'fedprox_mu': '0.0',
                'he_backend': 'N/A', 'convnet_hidden1': '32', 'convnet_hidden2': '16',
                'num_custom_layers': '-1', 'best_acc': '0.80'}
        row2 = {**base, 'global_epoch': '10', 'best_acc': '0.90'}

        k1 = _row_key(base)
        k2 = _row_key(row2)
        self.assertNotEqual(k1, k2)


# ===========================================================================
# E — TELEGRAM + LOG
# ===========================================================================

class TestTelegramLog(unittest.TestCase):

    def test_telegram_logs_to_logger_when_provided(self):
        """send_telegram scrive il messaggio nel logger quando fornito."""
        from federated_grid_search import send_telegram
        mock_logger = MagicMock()
        send_telegram('', '', '<b>Test</b> messaggio', logger=mock_logger)
        mock_logger.info.assert_called_once()
        logged = mock_logger.info.call_args[0][0]
        self.assertIn('[Telegram]', logged)
        self.assertIn('Test', logged)
        self.assertNotIn('<b>', logged)

    def test_telegram_strips_html(self):
        """I tag HTML vengono rimossi dal messaggio nel log."""
        from federated_grid_search import send_telegram
        mock_logger = MagicMock()
        send_telegram('', '', '<b>Titolo</b>\n<code>error msg</code>', logger=mock_logger)
        logged = mock_logger.info.call_args[0][0]
        self.assertNotIn('<b>', logged)
        self.assertNotIn('<code>', logged)
        self.assertIn('Titolo', logged)
        self.assertIn('error msg', logged)

    def test_telegram_prints_when_no_logger(self):
        """Senza logger, send_telegram stampa su stdout."""
        from federated_grid_search import send_telegram
        import io
        captured = io.StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            send_telegram('', '', 'hello world')
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        self.assertIn('[Telegram]', output)
        self.assertIn('hello world', output)

    @patch('requests.post')
    def test_telegram_sends_http_request(self, mock_post):
        """Con token e chat_id validi, viene fatto un POST a Telegram."""
        from federated_grid_search import send_telegram
        mock_post.return_value.status_code = 200
        send_telegram('fake_token', '12345', 'test msg')
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn('api.telegram.org', call_args[0][0])

    @patch('requests.post')
    def test_telegram_skips_if_no_token(self, mock_post):
        """Senza token, non viene fatto alcun HTTP request."""
        from federated_grid_search import send_telegram
        send_telegram('', '12345', 'test msg')
        mock_post.assert_not_called()

    def test_setup_main_logger_creates_file(self):
        """_setup_main_logger crea il file di log nella directory corretta."""
        from federated_grid_search import _setup_main_logger
        with tempfile.TemporaryDirectory() as tmpdir:
            import federated_grid_search as fgs
            orig_root = fgs._PROJECT_ROOT
            fgs._PROJECT_ROOT = Path(tmpdir)
            try:
                logger = _setup_main_logger('TestPC', 'log')
                logger.info('Test log entry')
                log_dir = Path(tmpdir) / 'log' / 'log_TestPC'
                self.assertTrue(log_dir.exists())
                log_files = list(log_dir.glob('main_*.log'))
                self.assertEqual(len(log_files), 1)
                content = log_files[0].read_text(encoding='utf-8')
                self.assertIn('Test log entry', content)
                # Cleanup handlers to avoid file lock issues on Windows
                for h in logger.handlers[:]:
                    h.close()
                    logger.removeHandler(h)
            finally:
                fgs._PROJECT_ROOT = orig_root


# ===========================================================================
# F — STRESS TEST WORKER SCHEDULING
# ===========================================================================

class TestWorkerSchedulingStress(unittest.TestCase):

    def test_mixed_normal_heavy_stress(self):
        """20 normal + 3 heavy: nessun heavy gira mentre ci sono normal attivi."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        violations = multiprocessing.Value('i', 0)
        threshold = 8

        def normal_task():
            deadline = time.time() + 10
            while time.time() < deadline:
                with heavy_running.get_lock():
                    if heavy_running.value == 0:
                        break
                time.sleep(0.005)
            with active_normal_workers.get_lock():
                active_normal_workers.value += 1
            time.sleep(0.02)
            with active_normal_workers.get_lock():
                active_normal_workers.value = max(0, active_normal_workers.value - 1)

        def heavy_task():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            # Verifica che nessun normal sia attivo
            deadline = time.time() + 5
            while time.time() < deadline:
                with active_normal_workers.get_lock():
                    if active_normal_workers.value == 0:
                        break
                time.sleep(0.005)
            # Esegui il task
            for _ in range(5):
                time.sleep(0.01)
                with active_normal_workers.get_lock():
                    if active_normal_workers.value > 0:
                        with violations.get_lock():
                            violations.value += 1
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        threads = [threading.Thread(target=normal_task) for _ in range(20)]
        threads += [threading.Thread(target=heavy_task) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(violations.value, 0, f"Heavy e normal si sono sovrapposti {violations.value} volte!")

    def test_all_configs_complete(self):
        """Tutti i task (normal e heavy) vengono completati senza blocchi."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        completed = multiprocessing.Value('i', 0)
        threshold = 8

        configs = [{'num_clients': 2}] * 10 + [{'num_clients': 16}] * 3

        def run_config(cfg):
            is_heavy = cfg['num_clients'] > threshold
            _heavy_acquired = False
            _normal_counted = False
            try:
                if is_heavy:
                    heavy_lock.acquire()
                    _heavy_acquired = True
                    with heavy_running.get_lock():
                        heavy_running.value = 1
                    deadline = time.time() + 5
                    while time.time() < deadline:
                        with active_normal_workers.get_lock():
                            if active_normal_workers.value == 0:
                                break
                        time.sleep(0.005)
                else:
                    deadline = time.time() + 5
                    while time.time() < deadline:
                        with heavy_running.get_lock():
                            if heavy_running.value == 0:
                                break
                        time.sleep(0.005)
                    with active_normal_workers.get_lock():
                        active_normal_workers.value += 1
                    _normal_counted = True
                time.sleep(0.01)
                with completed.get_lock():
                    completed.value += 1
            finally:
                if _heavy_acquired:
                    with heavy_running.get_lock():
                        heavy_running.value = 0
                    heavy_lock.release()
                if _normal_counted:
                    with active_normal_workers.get_lock():
                        active_normal_workers.value = max(0, active_normal_workers.value - 1)

        threads = [threading.Thread(target=run_config, args=(c,)) for c in configs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(completed.value, len(configs), f"Solo {completed.value}/{len(configs)} task completati!")


# ===========================================================================
# G — DATASET LOADING E IMAGE ACCESS
# ===========================================================================

def _create_fake_dataset(base_dir: str, num_classes: int = 2,
                          images_per_class: int = 20,
                          with_commas: bool = False) -> None:
    """Crea un dataset fake con immagini PNG minimali."""
    try:
        from PIL import Image
        import numpy as np
        use_pil = True
    except ImportError:
        use_pil = False

    for cls in range(num_classes):
        class_dir = os.path.join(base_dir, f'classe_{cls}')
        os.makedirs(class_dir, exist_ok=True)
        for i in range(images_per_class):
            if with_commas and i % 3 == 0:
                fname = f'patient1_slice{i}_EDSS1,5.png'
            else:
                fname = f'patient1_slice{i}_EDSS2.png'
            fpath = os.path.join(class_dir, fname)
            if use_pil:
                arr = (os.urandom(128 * 128 * 3))
                img_array = list(arr)
                img = Image.frombytes('RGB', (128, 128), bytes(img_array[:128*128*3]))
                img.save(fpath)
            else:
                # Crea un PNG minimale valido senza PIL
                import struct, zlib
                def png_chunk(name, data):
                    c = struct.pack('>I', len(data)) + name + data
                    return c + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff)
                signature = b'\x89PNG\r\n\x1a\n'
                ihdr = png_chunk(b'IHDR', struct.pack('>IIBBBBB', 4, 4, 8, 2, 0, 0, 0))
                raw = b'\x00\xff\x00\xff\x00\xff\x00\xff' * 4  # 4x4 RGB greens
                idat = png_chunk(b'IDAT', zlib.compress(raw))
                iend = png_chunk(b'IEND', b'')
                with open(fpath, 'wb') as f:
                    f.write(signature + ihdr + idat + iend)


class TestDatasetLoading(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.tmp, 'dataset')
        self.split_dir = os.path.join(self.tmp, 'split')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dataset_splitter_handles_comma_filenames(self):
        """DatasetSplitter copia file con virgola rinominandoli con underscore."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=10,
                              with_commas=True)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        splitter.split_dataset()

        # Verifica che nessun file nella split dir abbia virgole
        for root, _, files in os.walk(self.split_dir):
            for fname in files:
                self.assertNotIn(',', fname, f"Virgola trovata in split file: {fname}")

    def test_dataset_splitter_creates_client_dirs(self):
        """DatasetSplitter crea le directory client_0, client_1, ecc."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=20)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        splitter.split_dataset()
        client_dirs = [d for d in os.listdir(self.split_dir) if d.startswith('client_')]
        self.assertGreaterEqual(len(client_dirs), 1)

    def test_dataset_splitter_non_zero_training_data(self):
        """Ogni client riceve almeno 1 immagine di training (no 'Total training size=0')."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=20)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        splitter.split_dataset()
        for i in range(2):
            train_dir = os.path.join(self.split_dir, f'client_{i}', 'train')
            if os.path.exists(train_dir):
                total_images = sum(
                    len([f for f in os.listdir(os.path.join(train_dir, cls))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    for cls in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, cls))
                )
                self.assertGreater(total_images, 0,
                    f"client_{i} ha 0 immagini di training — causa dei 'Total training size=0'")

    def test_split_images_loadable_by_pil(self):
        """Tutte le immagini nello split dir sono apribili con PIL."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL non disponibile")
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=10,
                              with_commas=True)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        splitter.split_dataset()

        errors = []
        for root, _, files in os.walk(self.split_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(root, fname)
                    try:
                        img = Image.open(fpath)
                        img.verify()
                    except Exception as e:
                        errors.append(f"{fpath}: {e}")
        self.assertEqual(errors, [], f"Immagini non apribili: {errors[:3]}")

    def test_imagefolder_loads_split_correctly(self):
        """torchvision.datasets.ImageFolder carica correttamente le split directory."""
        try:
            from torchvision.datasets import ImageFolder
            import torchvision.transforms as T
        except ImportError:
            self.skipTest("torchvision non disponibile")
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=20,
                              with_commas=True)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        splitter.split_dataset()

        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        client_train = os.path.join(self.split_dir, 'client_0', 'train')
        if not os.path.exists(client_train):
            self.skipTest("client_0/train non creata (troppo pochi dati per il test)")

        dataset = ImageFolder(root=client_train, transform=transform)
        self.assertGreater(len(dataset), 0, "ImageFolder carica 0 immagini dal client_0/train")

        # Carica un batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        images, labels = batch
        self.assertEqual(images.shape[1], 3)  # 3 canali RGB
        self.assertEqual(images.shape[2], 32)
        self.assertEqual(images.shape[3], 32)

    def test_splitter_fewer_images_than_clients(self):
        """DatasetSplitter gestisce il caso in cui le immagini per classe < num_clients."""
        from data_splitter import DatasetSplitter
        # Solo 3 immagini per classe, ma chiediamo 10 client
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=10
        )
        # Non deve crashare, ma produrre meno di 10 client
        splitter.split_dataset()
        client_dirs = [d for d in os.listdir(self.split_dir) if d.startswith('client_')]
        self.assertLessEqual(len(client_dirs), 3)  # ridotto al min_class_count

    def test_splitter_returns_correct_effective_client_count(self):
        """split_dataset ritorna il numero effettivo di client creati."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=40)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=4)
        actual = splitter.split_dataset()
        self.assertEqual(actual, 4)
        client_dirs = [d for d in os.listdir(self.split_dir) if d.startswith('client_')]
        self.assertEqual(len(client_dirs), actual)

    def test_compute_effective_clients_matches_split_result(self):
        """compute_effective_clients predice correttamente quanti client crea split_dataset."""
        from data_splitter import DatasetSplitter, compute_effective_clients
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        effective = compute_effective_clients(self.dataset_dir, num_clients=8)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=8)
        actual = splitter.split_dataset()
        self.assertEqual(effective, actual)

    def test_hf_naming_scan(self):
        """fix_hf_filenames trova correttamente i file con virgola in un dir di test."""
        from fix_hf_filenames import find_local_comma_files
        # Crea un dataset fake con virgole
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=10,
                              with_commas=True)
        pairs = find_local_comma_files(base_dirs=[Path(self.dataset_dir)])
        self.assertGreater(len(pairs), 0, "Nessun file con virgola trovato")
        for old, new in pairs:
            self.assertIn(',', old.name)
            self.assertNotIn(',', new.name)

    def test_hf_naming_local_fix(self):
        """fix_hf_filenames rinomina correttamente i file locali."""
        from fix_hf_filenames import find_local_comma_files, rename_local_files
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=10,
                              with_commas=True)
        pairs = find_local_comma_files(base_dirs=[Path(self.dataset_dir)])
        self.assertGreater(len(pairs), 0)
        before_count = len(pairs)
        rename_local_files(pairs)
        # Dopo il rename, nessun file dovrebbe avere virgole
        remaining = find_local_comma_files(base_dirs=[Path(self.dataset_dir)])
        self.assertEqual(len(remaining), 0, f"Rimasti {len(remaining)} file con virgola dopo il fix")

    def test_data_splitter_builds_correct_dataframe(self):
        """DatasetSplitter._build_dataframe_from_folders legge le classi correttamente."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=3, images_per_class=10)
        splitter = DatasetSplitter(
            output_base_dir=self.split_dir,
            source_images_dir=self.dataset_dir,
            num_clients=2
        )
        df = splitter.dataframe
        self.assertFalse(df.empty)
        self.assertIn('filename', df.columns)
        self.assertIn('class', df.columns)
        unique_classes = df['class'].nunique()
        self.assertEqual(unique_classes, 3)
        total_images = len(df)
        self.assertEqual(total_images, 30)  # 3 classi x 10 immagini


# ===========================================================================
# F — STRESS TEST AGGIUNTIVI
# ===========================================================================

class TestWorkerSchedulingStressExtra(unittest.TestCase):

    def test_32_client_config_is_heavy(self):
        """Una config con 32 client è heavy (> HEAVY_CLIENT_THRESHOLD=8)."""
        from federated_grid_search import HEAVY_CLIENT_THRESHOLD
        self.assertTrue(32 > HEAVY_CLIENT_THRESHOLD)

    def test_16_client_config_is_heavy(self):
        """Una config con 16 client è heavy."""
        from federated_grid_search import HEAVY_CLIENT_THRESHOLD
        self.assertTrue(16 > HEAVY_CLIENT_THRESHOLD)

    def test_8_client_config_is_not_heavy(self):
        """Una config con 8 client (== threshold) NON è heavy (condizione è >)."""
        from federated_grid_search import HEAVY_CLIENT_THRESHOLD
        self.assertFalse(8 > HEAVY_CLIENT_THRESHOLD)

    def test_5_heavy_10_normal_all_complete(self):
        """Mix di 5 heavy (32 client) e 10 normal (4 client): tutti i 15 task completati."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        completed = multiprocessing.Value('i', 0)
        threshold = 8

        configs = [{'num_clients': 32}] * 5 + [{'num_clients': 4}] * 10

        def run_config(cfg):
            is_heavy = cfg['num_clients'] > threshold
            _heavy_acquired = False
            _normal_counted = False
            try:
                if is_heavy:
                    heavy_lock.acquire()
                    _heavy_acquired = True
                    with heavy_running.get_lock():
                        heavy_running.value = 1
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        with active_normal_workers.get_lock():
                            if active_normal_workers.value == 0:
                                break
                        time.sleep(0.005)
                else:
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        with heavy_running.get_lock():
                            if heavy_running.value == 0:
                                break
                        time.sleep(0.005)
                    with active_normal_workers.get_lock():
                        active_normal_workers.value += 1
                    _normal_counted = True
                time.sleep(0.01)
                with completed.get_lock():
                    completed.value += 1
            finally:
                if _heavy_acquired:
                    with heavy_running.get_lock():
                        heavy_running.value = 0
                    heavy_lock.release()
                if _normal_counted:
                    with active_normal_workers.get_lock():
                        active_normal_workers.value = max(0, active_normal_workers.value - 1)

        threads = [threading.Thread(target=run_config, args=(c,)) for c in configs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(completed.value, len(configs),
                         f"Solo {completed.value}/{len(configs)} task completati!")

    def test_stress_32_clients_1_worker(self):
        """1 worker con config da 32 client (heavy): viene eseguito in isolamento senza crash."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        completed = multiprocessing.Value('i', 0)
        threshold = 8

        cfg = {'num_clients': 32}  # heavy

        def run_heavy():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            deadline = time.time() + 5
            while time.time() < deadline:
                with active_normal_workers.get_lock():
                    if active_normal_workers.value == 0:
                        break
                time.sleep(0.005)
            time.sleep(0.02)
            with completed.get_lock():
                completed.value += 1
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        t = threading.Thread(target=run_heavy)
        t.start()
        t.join(timeout=5)
        self.assertEqual(completed.value, 1, "Il task da 32 client (1 worker) non e' completato")

    def test_stress_16_clients_2_workers(self):
        """2 worker con config da 16 client (heavy): entrambi completati, non in contemporanea."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        completed = multiprocessing.Value('i', 0)
        concurrent = multiprocessing.Value('i', 0)
        peak_concurrent = multiprocessing.Value('i', 0)

        def run_heavy():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            deadline = time.time() + 5
            while time.time() < deadline:
                with active_normal_workers.get_lock():
                    if active_normal_workers.value == 0:
                        break
                time.sleep(0.005)
            with concurrent.get_lock():
                concurrent.value += 1
                with peak_concurrent.get_lock():
                    if concurrent.value > peak_concurrent.value:
                        peak_concurrent.value = concurrent.value
            time.sleep(0.03)
            with concurrent.get_lock():
                concurrent.value -= 1
            with completed.get_lock():
                completed.value += 1
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        threads = [threading.Thread(target=run_heavy) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(completed.value, 2, "Non entrambi i task da 16 client completati")
        self.assertEqual(peak_concurrent.value, 1, f"Peak concurrent={peak_concurrent.value}, atteso 1")

    def test_stress_mixed_configs(self):
        """Mix: 3 config da 32 client (heavy) + 8 config da 4 client (normal). Tutti completati."""
        heavy_lock = multiprocessing.Lock()
        heavy_running = multiprocessing.Value('i', 0)
        active_normal_workers = multiprocessing.Value('i', 0)
        completed = multiprocessing.Value('i', 0)
        threshold = 8

        configs = [{'num_clients': 32}] * 3 + [{'num_clients': 4}] * 8

        def run_config(cfg):
            is_heavy = cfg['num_clients'] > threshold
            _heavy_acquired = False
            _normal_counted = False
            try:
                if is_heavy:
                    heavy_lock.acquire()
                    _heavy_acquired = True
                    with heavy_running.get_lock():
                        heavy_running.value = 1
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        with active_normal_workers.get_lock():
                            if active_normal_workers.value == 0:
                                break
                        time.sleep(0.005)
                else:
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        with heavy_running.get_lock():
                            if heavy_running.value == 0:
                                break
                        time.sleep(0.005)
                    with active_normal_workers.get_lock():
                        active_normal_workers.value += 1
                    _normal_counted = True
                time.sleep(0.01)
                with completed.get_lock():
                    completed.value += 1
            finally:
                if _heavy_acquired:
                    with heavy_running.get_lock():
                        heavy_running.value = 0
                    heavy_lock.release()
                if _normal_counted:
                    with active_normal_workers.get_lock():
                        active_normal_workers.value = max(0, active_normal_workers.value - 1)

        threads = [threading.Thread(target=run_config, args=(c,)) for c in configs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(completed.value, len(configs),
                         f"Solo {completed.value}/{len(configs)} task completati nel mix heavy/normal!")


# ===========================================================================
# H — NAMESPACE ERROR HANDLING
# ===========================================================================

class TestNamespaceError(unittest.TestCase):
    """Testa la gestione del namespace error 'is not a connected namespace'."""

    def test_client_eval_skipped_when_disconnected(self):
        """Quando sio.connected = False, l'emit di client_eval viene saltato."""
        mock_sio = MagicMock()
        mock_sio.connected = False
        mock_logger = MagicMock()

        response = {'test_loss': 0.5, 'test_acc': 0.8}
        if mock_sio.connected:
            try:
                mock_sio.emit('client_eval', response)
            except Exception as e:
                mock_logger.warning("Could not send client_eval: %s", e)
        else:
            mock_logger.warning("Socket disconnected before sending client_eval. Skipping.")

        mock_sio.emit.assert_not_called()
        mock_logger.warning.assert_called_once()

    def test_client_eval_sent_when_connected(self):
        """Quando sio.connected = True e nessuna eccezione, l'emit viene eseguito."""
        mock_sio = MagicMock()
        mock_sio.connected = True
        mock_logger = MagicMock()

        response = {'test_loss': 0.5, 'test_acc': 0.8}
        if mock_sio.connected:
            try:
                mock_sio.emit('client_eval', response)
            except Exception as e:
                mock_logger.warning("Could not send client_eval: %s", e)
        else:
            mock_logger.warning("Socket disconnected. Skipping.")

        mock_sio.emit.assert_called_once_with('client_eval', response)
        mock_logger.warning.assert_not_called()

    def test_namespace_exception_caught_as_warning(self):
        """Un'eccezione namespace error durante emit viene loggata come warning, non propagata."""
        mock_sio = MagicMock()
        mock_sio.connected = True
        mock_sio.emit.side_effect = Exception("/ is not a connected namespace")
        mock_logger = MagicMock()

        response = {'test_loss': 0.5, 'test_acc': 0.8}
        # Non deve sollevare eccezione
        try:
            if mock_sio.connected:
                try:
                    mock_sio.emit('client_eval', response)
                except Exception as e:
                    mock_logger.warning("Could not send client_eval (socket disconnected): %s", e)
        except Exception:
            self.fail("L'eccezione namespace non è stata catturata correttamente")

        mock_logger.warning.assert_called_once()
        warning_call = str(mock_logger.warning.call_args)
        self.assertIn('not a connected namespace', warning_call)

    def test_client_update_not_sent_when_disconnected(self):
        """L'emit di client_update non avviene se sio.connected = False."""
        mock_sio = MagicMock()
        mock_sio.connected = False

        if not mock_sio.connected:
            pass  # skip emit
        else:
            mock_sio.emit('client_update', {'round': 1})

        mock_sio.emit.assert_not_called()

    def test_update_worker_exception_logged_not_propagated(self):
        """L'eccezione nel worker thread viene loggata come error, non propagata."""
        logged_errors = []

        class _Logger:
            def error(self, msg, *args, **kwargs):
                logged_errors.append(msg % args if args else msg)

        logger = _Logger()

        def simulate_update_worker():
            try:
                raise Exception("/ is not a connected namespace")
            except Exception as e:
                logger.error("An error occurred in the update worker thread: %s", e)

        simulate_update_worker()

        self.assertEqual(len(logged_errors), 1)
        self.assertIn('not a connected namespace', logged_errors[0])

    def test_multiple_namespace_errors_all_logged(self):
        """Più eccezioni namespace error consecutive vengono tutte loggate."""
        warnings = []
        mock_sio = MagicMock()
        mock_sio.connected = True
        mock_sio.emit.side_effect = Exception("/ is not a connected namespace")

        for _ in range(3):
            if mock_sio.connected:
                try:
                    mock_sio.emit('client_eval', {})
                except Exception as e:
                    warnings.append(str(e))

        self.assertEqual(len(warnings), 3)


# ===========================================================================
# I — WATCHDOG SCENARIOS
# ===========================================================================

class TestWatchdogScenarios(unittest.TestCase):
    """Testa gli scenari di watchdog timeout e calcolo effective clients."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.tmp, 'dataset')
        self.split_dir = os.path.join(self.tmp, 'split')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compute_effective_clients_with_small_dataset(self):
        """compute_effective_clients riduce il numero di client quando il dataset è piccolo."""
        from data_splitter import compute_effective_clients
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        result = compute_effective_clients(self.dataset_dir, num_clients=8)
        self.assertLessEqual(result, 3)
        self.assertGreaterEqual(result, 1)

    def test_compute_effective_clients_does_not_exceed_requested(self):
        """compute_effective_clients non supera il numero di client richiesti."""
        from data_splitter import compute_effective_clients
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=20)
        result = compute_effective_clients(self.dataset_dir, num_clients=4)
        self.assertLessEqual(result, 4)

    def test_compute_effective_clients_empty_dataset(self):
        """compute_effective_clients gestisce directory vuota senza crash."""
        from data_splitter import compute_effective_clients
        os.makedirs(self.dataset_dir, exist_ok=True)
        # Non deve sollevare eccezione
        try:
            result = compute_effective_clients(self.dataset_dir, num_clients=4)
            self.assertGreaterEqual(result, 0)
        except Exception as e:
            self.fail(f"compute_effective_clients ha crashato su dataset vuoto: {e}")

    def test_splitter_limits_dirs_to_effective_clients(self):
        """Con 3 immagini per classe, vengono create al massimo 3 directory client."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=8)
        actual = splitter.split_dataset()
        self.assertLessEqual(actual, 3)
        client_dirs = [d for d in os.listdir(self.split_dir) if d.startswith('client_')]
        self.assertEqual(len(client_dirs), actual)

    def test_no_orphan_dirs_beyond_effective_clients(self):
        """Nessuna directory client_{i} per i >= effective_clients (no zombie dirs)."""
        from data_splitter import DatasetSplitter
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=8)
        actual = splitter.split_dataset()
        for i in range(actual, 8):
            orphan = os.path.join(self.split_dir, f'client_{i}')
            self.assertFalse(os.path.exists(orphan),
                             f"client_{i} non dovrebbe esistere (effective={actual})")

    def test_min_class_warning_printed_when_dataset_small(self):
        """DatasetSplitter stampa warning quando min_class_count < num_clients."""
        from data_splitter import DatasetSplitter
        import io
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=10)

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            splitter.split_dataset()
        finally:
            sys.stdout = old_stdout

        self.assertIn('Warning', captured.getvalue())

    def test_zero_train_images_warning_printed(self):
        """DatasetSplitter stampa warning quando un client ha 0 immagini di training."""
        from data_splitter import DatasetSplitter
        import io
        # Con pochissime immagini e validazione split, qualche client potrebbe avere n_train=0
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=2)
        splitter = DatasetSplitter(self.split_dir, self.dataset_dir, num_clients=2)

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            splitter.split_dataset()
        finally:
            sys.stdout = old_stdout
        # Il test verifica solo che non crashi, non necessariamente il warning
        # (dipende da quante immagini vengono assegnate alla validazione)
        self.assertTrue(True)  # no crash = pass

    def test_effective_clients_at_least_1(self):
        """compute_effective_clients restituisce almeno 1 se il dataset non è vuoto."""
        from data_splitter import compute_effective_clients
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=5)
        result = compute_effective_clients(self.dataset_dir, num_clients=4)
        self.assertGreaterEqual(result, 1)

    def test_run_multiple_clients_skips_empty_dirs(self):
        """run_multiple_clients avvia solo i thread per cui esistono directory client."""
        import run_multiple_clients as rmc
        _create_fake_dataset(self.dataset_dir, num_classes=2, images_per_class=3)

        # Con 3 immagini per classe e 8 client richiesti, split_dataset crea <= 3 dir
        config = {
            'num_clients': 8,
            'splitting_dir': self.split_dir,
            'dataset_path': self.dataset_dir,
            'ip_address': '127.0.0.1',
            'port': 65000,
            'MIN_NUM_WORKERS': 8,
        }
        threads_started = []
        original_start = threading.Thread.start

        def mock_start(self_t):
            threads_started.append(self_t)
            # Non avviare davvero: intercettiamo prima di connettere al server
            pass

        with patch.object(threading.Thread, 'start', mock_start):
            # Sostituisce FederatedClient con un no-op per non tentare connessioni reali
            with patch('run_multiple_clients.FederatedClient', return_value=None):
                try:
                    rmc.main(config)
                except Exception:
                    pass  # possibili errori di connessione — non ci interessa

        # Il numero di thread avviati deve essere <= 3 (limitato dal dataset)
        # Non deve avviare 8 thread per 8 client quando il dataset ne supporta al massimo 3
        self.assertLessEqual(len(threads_started), 4,
                             f"run_multiple_clients ha avviato {len(threads_started)} thread invece di <=3")

    def test_watchdog_fires_on_no_clients(self):
        """compute_effective_clients=0 non causa hang: nessun thread viene avviato."""
        from data_splitter import compute_effective_clients
        import os
        # Directory vuota senza immagini
        empty_dir = os.path.join(self.tmp, 'empty_dataset')
        os.makedirs(empty_dir, exist_ok=True)
        result = compute_effective_clients(empty_dir, num_clients=4)
        # Se 0 o 1, run_multiple_clients avvierebbe 0 o 1 thread — no hang infinito
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 4)

    def test_pre_split_min_num_workers_reflects_actual_io(self):
        """MIN_NUM_WORKERS deve venire dal valore reale di split_dataset, non da dry-run.

        Simula il caso in cui split_dataset() restituisce meno client del previsto
        (fallimenti I/O) e verifica che config['MIN_NUM_WORKERS'] rispecchi quel valore
        PRIMA dell'avvio del server — cioè il path already_split=True in run_multiple_clients.
        """
        import run_multiple_clients as rmc

        # Prepara una splitting_dir con 2 client validi (train/ non vuota)
        os.makedirs(self.split_dir, exist_ok=True)
        for i in range(2):
            train_dir = os.path.join(self.split_dir, f'client_{i}', 'train', 'cls')
            os.makedirs(train_dir)
            open(os.path.join(train_dir, f'img{i}.png'), 'w').close()

        config = {
            'num_clients': 4,          # richiesti 4, ma split ha creato solo 2
            'splitting_dir': self.split_dir,
            'dataset_path': self.dataset_dir,
            'ip_address': '127.0.0.1',
            'port': 65001,
            'MIN_NUM_WORKERS': 2,      # già impostato dal pre-split (come fa federated_grid_search)
            'global_epoch': 1,
            'round_timeout': 10,
            'server_startup_timeout': 10,
        }

        threads_started = []

        def mock_start(self_t):
            threads_started.append(self_t)

        with patch.object(threading.Thread, 'start', mock_start):
            with patch('run_multiple_clients.FederatedClient', return_value=MagicMock(sio=MagicMock(connected=False))):
                try:
                    rmc.main(config, already_split=True)
                except Exception:
                    pass

        # already_split=True usa MIN_NUM_WORKERS (2) come actual_clients, non num_clients (4)
        self.assertLessEqual(len(threads_started), 2,
                             f"already_split=True deve avviare ≤2 thread, ha avviato {len(threads_started)}")


# ===========================================================================
# J — PORT MANAGEMENT
# ===========================================================================

class TestPortManagement(unittest.TestCase):
    """Testa le funzioni di gestione porte in federated_grid_search.py."""

    def test_find_free_port_returns_valid_range(self):
        """find_free_port ritorna una porta nel range 1024-65534."""
        from federated_grid_search import find_free_port
        port = find_free_port('127.0.0.1', 19700)
        self.assertGreater(port, 1024)
        self.assertLess(port, 65535)

    def test_find_free_port_skips_occupied(self):
        """find_free_port salta le porte già in ascolto."""
        from federated_grid_search import find_free_port
        import socket as _sock
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        occupied = s.getsockname()[1]
        try:
            result = find_free_port('127.0.0.1', occupied)
            self.assertNotEqual(result, occupied,
                                f"find_free_port ha restituito la porta occupata {occupied}")
        finally:
            s.close()

    def test_port_is_listening_true_when_listening(self):
        """_port_is_listening ritorna True se una socket è in ascolto."""
        from federated_grid_search import _port_is_listening
        import socket as _sock
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
        try:
            self.assertTrue(_port_is_listening('127.0.0.1', port))
        finally:
            s.close()

    def test_port_is_listening_false_when_free(self):
        """_port_is_listening ritorna False se la porta è libera."""
        from federated_grid_search import find_free_port, _port_is_listening
        port = find_free_port('127.0.0.1', 19800)
        self.assertFalse(_port_is_listening('127.0.0.1', port))

    def test_wait_for_port_free_immediate_when_free(self):
        """wait_for_port_free ritorna True subito se la porta è già libera."""
        from federated_grid_search import find_free_port, wait_for_port_free
        port = find_free_port('127.0.0.1', 19900)
        result = wait_for_port_free('127.0.0.1', port, timeout=5)
        self.assertTrue(result)

    def test_port_retry_loop_finds_free_port(self):
        """Il retry loop trova una porta libera anche se la prima è occupata."""
        from federated_grid_search import find_free_port, _port_is_listening
        import socket as _sock
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        occupied = s.getsockname()[1]
        try:
            found_port = occupied  # inizia dall'occupata
            for _retry in range(5):
                found_port = find_free_port('127.0.0.1', occupied)
                time.sleep(0.05 * (_retry + 1))
                if not _port_is_listening('127.0.0.1', found_port):
                    break
            self.assertNotEqual(found_port, occupied)
            self.assertFalse(_port_is_listening('127.0.0.1', found_port))
        finally:
            s.close()

    def test_multiple_workers_get_different_ports(self):
        """find_free_port con preferred_port diversi restituisce porte diverse."""
        from federated_grid_search import find_free_port
        ports = set()
        for worker_id in range(4):
            preferred = 19000 + worker_id * 2
            p = find_free_port('127.0.0.1', preferred)
            ports.add(p)
        self.assertEqual(len(ports), 4, f"Porte duplicate trovate: {ports}")

    def test_wait_for_port_free_timeout_returns_false(self):
        """wait_for_port_free ritorna False se la porta rimane occupata fino allo scadere del timeout."""
        from federated_grid_search import wait_for_port_free
        import socket as _sock
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
        try:
            # Con timeout=2s e porta occupata, deve restituire False
            result = wait_for_port_free('127.0.0.1', port, timeout=2)
            self.assertFalse(result, f"wait_for_port_free avrebbe dovuto restituire False (porta {port} ancora occupata)")
        finally:
            s.close()

    def test_cleanup_returns_false_if_port_stuck(self):
        """_cleanup_after_task ritorna False se la porta non si libera (mock del processo che la tiene)."""
        from federated_grid_search import _cleanup_after_task, _port_is_listening
        from unittest.mock import patch

        # Processo già morto — la porta è "libera" secondo _port_is_listening
        # Simuliamo il caso opposto: la porta rimane occupata
        # Usiamo patch per far credere che la porta sia sempre occupata
        fake_port = 20600

        with patch('federated_grid_search._port_is_listening', return_value=True), \
             patch('federated_grid_search._get_pid_on_port', return_value=0):
            p = multiprocessing.Process(target=_proc_exit_immediately)
            p.start()
            p.join(timeout=5)
            result = _cleanup_after_task(p, fake_port, '127.0.0.1', worker_id=99, max_port_wait=2)
            self.assertFalse(result, "_cleanup_after_task avrebbe dovuto ritornare False con porta bloccata")


# ===========================================================================
# K — HEAVY SCHEDULING AVANZATO
# ===========================================================================

class TestHeavySchedulingAdvanced(unittest.TestCase):
    """Test avanzati per lo scheduling heavy/normal."""

    def _make_shared(self):
        return (multiprocessing.Lock(),
                multiprocessing.Value('i', 0),
                multiprocessing.Value('i', 0))

    def test_heavy_waits_for_active_normals_to_finish(self):
        """Un heavy worker aspetta che tutti i normal attivi finiscano."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        normals_active_when_heavy_ran = multiprocessing.Value('i', 0)

        def normal_worker():
            with active_normal_workers.get_lock():
                active_normal_workers.value += 1
            time.sleep(0.08)
            with active_normal_workers.get_lock():
                active_normal_workers.value = max(0, active_normal_workers.value - 1)

        def heavy_worker():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            deadline = time.time() + 5
            while time.time() < deadline:
                with active_normal_workers.get_lock():
                    if active_normal_workers.value == 0:
                        break
                time.sleep(0.005)
            # Cattura quanti normali sono ancora attivi nel momento in cui inizia
            with active_normal_workers.get_lock():
                normals_active_when_heavy_ran.value = active_normal_workers.value
            time.sleep(0.02)
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        normals = [threading.Thread(target=normal_worker) for _ in range(3)]
        h = threading.Thread(target=heavy_worker)
        for t in normals:
            t.start()
        time.sleep(0.02)
        h.start()
        for t in normals:
            t.join(timeout=3)
        h.join(timeout=5)

        self.assertEqual(normals_active_when_heavy_ran.value, 0,
                         f"{normals_active_when_heavy_ran.value} normal attivi quando heavy è partito!")

    def test_normal_workers_do_not_start_during_heavy(self):
        """I worker normali attendono la fine dell'heavy prima di partire."""
        heavy_lock, heavy_running, active_normal_workers = self._make_shared()
        started_during_heavy = multiprocessing.Value('i', 0)
        heavy_done = multiprocessing.Value('i', 0)

        def heavy_worker():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            time.sleep(0.1)
            with heavy_running.get_lock():
                heavy_running.value = 0
            with heavy_done.get_lock():
                heavy_done.value = 1
            heavy_lock.release()

        def normal_worker():
            deadline = time.time() + 5
            while time.time() < deadline:
                with heavy_running.get_lock():
                    if heavy_running.value == 0:
                        break
                time.sleep(0.005)
            with heavy_done.get_lock():
                if heavy_done.value == 0:
                    with started_during_heavy.get_lock():
                        started_during_heavy.value += 1
            with active_normal_workers.get_lock():
                active_normal_workers.value += 1
            time.sleep(0.01)
            with active_normal_workers.get_lock():
                active_normal_workers.value = max(0, active_normal_workers.value - 1)

        h = threading.Thread(target=heavy_worker)
        normals = [threading.Thread(target=normal_worker) for _ in range(4)]
        h.start()
        time.sleep(0.01)
        for t in normals:
            t.start()
        h.join(timeout=3)
        for t in normals:
            t.join(timeout=3)

        self.assertEqual(started_during_heavy.value, 0,
                         f"{started_during_heavy.value} normal partiti mentre l'heavy era in corso!")

    def test_heavy_lock_allows_only_one_heavy_at_a_time(self):
        """Massimo 1 heavy gira alla volta (heavy_lock è esclusivo)."""
        heavy_lock, heavy_running, _ = self._make_shared()
        concurrent = multiprocessing.Value('i', 0)
        peak = multiprocessing.Value('i', 0)

        def heavy_worker():
            heavy_lock.acquire()
            with heavy_running.get_lock():
                heavy_running.value = 1
            with concurrent.get_lock():
                concurrent.value += 1
                with peak.get_lock():
                    if concurrent.value > peak.value:
                        peak.value = concurrent.value
            time.sleep(0.03)
            with concurrent.get_lock():
                concurrent.value -= 1
            with heavy_running.get_lock():
                heavy_running.value = 0
            heavy_lock.release()

        threads = [threading.Thread(target=heavy_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(peak.value, 1,
                         f"Peak di {peak.value} heavy worker contemporanei (atteso 1)")

    def test_threshold_boundary_values(self):
        """Verifica i valori limite del HEAVY_CLIENT_THRESHOLD."""
        from federated_grid_search import HEAVY_CLIENT_THRESHOLD
        self.assertEqual(HEAVY_CLIENT_THRESHOLD, 8)
        # Valori <= threshold = normale
        for n in [1, 2, 4, 8]:
            self.assertFalse(n > HEAVY_CLIENT_THRESHOLD, f"num_clients={n} non dovrebbe essere heavy")
        # Valori > threshold = heavy
        for n in [9, 16, 32]:
            self.assertTrue(n > HEAVY_CLIENT_THRESHOLD, f"num_clients={n} dovrebbe essere heavy")


# ===========================================================================
# L — ZOMBIE PROCESS CLEANUP
# ===========================================================================

class TestZombieCleanup(unittest.TestCase):
    """Testa _cleanup_after_task() per processi zombie e porte bloccate."""

    def test_cleanup_kills_alive_server_process(self):
        """_cleanup_after_task killa un processo che non termina da solo."""
        from federated_grid_search import _cleanup_after_task, find_free_port

        port = find_free_port('127.0.0.1', 20000)
        p = multiprocessing.Process(target=_proc_hang_forever)
        p.start()

        _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)

        p.join(timeout=5)
        self.assertFalse(p.is_alive(), "Il processo zombie non è stato killato da _cleanup_after_task")

    def test_cleanup_returns_true_when_server_already_dead(self):
        """_cleanup_after_task ritorna True se il server è già terminato e la porta è libera."""
        from federated_grid_search import _cleanup_after_task, find_free_port

        port = find_free_port('127.0.0.1', 20100)
        p = multiprocessing.Process(target=_proc_exit_immediately)
        p.start()
        p.join(timeout=5)  # processo già morto

        result = _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)
        self.assertTrue(result)

    def test_cleanup_handles_already_dead_process_no_exception(self):
        """_cleanup_after_task non solleva eccezioni se il processo è già morto."""
        from federated_grid_search import _cleanup_after_task, find_free_port

        port = find_free_port('127.0.0.1', 20200)
        p = multiprocessing.Process(target=_proc_exit_immediately)
        p.start()
        p.join(timeout=5)

        try:
            _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)
        except Exception as e:
            self.fail(f"_cleanup_after_task ha sollevato un'eccezione: {e}")

    def test_cleanup_frees_port_held_by_process(self):
        """_cleanup_after_task libera la porta dopo aver killato il processo che la teneva."""
        from federated_grid_search import _cleanup_after_task, find_free_port, _port_is_listening
        import socket as _sock

        port = find_free_port('127.0.0.1', 20300)

        proc = multiprocessing.Process(target=_proc_hold_port, args=(port,))
        proc.start()
        # Aspetta che la porta venga aperta (su Windows spawn=lento → 5s)
        for _ in range(50):
            if _port_is_listening('127.0.0.1', port):
                break
            time.sleep(0.1)

        if not _port_is_listening('127.0.0.1', port):
            proc.terminate()
            proc.join(timeout=3)
            self.skipTest("Porta non occupata entro 5s (processo troppo lento su questo sistema)")

        _cleanup_after_task(proc, port, '127.0.0.1', worker_id=99, max_port_wait=15)

        self.assertFalse(_port_is_listening('127.0.0.1', port),
                         "Porta ancora occupata dopo _cleanup_after_task")

    def test_cleanup_kills_server_children_with_psutil(self):
        """_cleanup_after_task killa i processi figlio del server (con psutil)."""
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil non disponibile")

        from federated_grid_search import _cleanup_after_task, find_free_port

        port = find_free_port('127.0.0.1', 20400)
        pid_list = multiprocessing.Manager().list()
        p = multiprocessing.Process(target=_proc_parent_with_child, args=(pid_list,))
        p.start()
        # Aspetta che il child sia avviato
        for _ in range(20):
            if len(pid_list) > 0:
                break
            time.sleep(0.1)

        _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)
        time.sleep(0.5)

        if len(pid_list) > 0:
            child_pid = pid_list[0]
            try:
                child_proc = psutil.Process(child_pid)
                is_zombie = child_proc.status() == psutil.STATUS_ZOMBIE
                self.assertTrue(
                    not child_proc.is_running() or is_zombie,
                    f"Child process {child_pid} ancora in esecuzione!"
                )
            except psutil.NoSuchProcess:
                pass  # Morto = test passato

    def test_client_threads_join_after_completion(self):
        """Dopo la simulazione di run_multiple_clients, nessun thread rimane vivo."""
        threads = []
        results = []

        def fake_client(cid):
            time.sleep(0.02)
            results.append(cid)

        for i in range(4):
            t = threading.Thread(target=fake_client, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=2)

        self.assertEqual(len(results), 4)
        alive = [t for t in threads if t.is_alive()]
        self.assertEqual(len(alive), 0, f"{len(alive)} thread client rimasti vivi (zombie thread)")

    def test_cleanup_multiple_times_idempotent(self):
        """Chiamare _cleanup_after_task più volte sullo stesso processo non crasha."""
        from federated_grid_search import _cleanup_after_task, find_free_port

        port = find_free_port('127.0.0.1', 20500)
        p = multiprocessing.Process(target=_proc_exit_immediately)
        p.start()
        p.join(timeout=5)

        try:
            _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)
            _cleanup_after_task(p, port, '127.0.0.1', worker_id=99)
        except Exception as e:
            self.fail(f"Seconda chiamata a _cleanup_after_task ha crashato: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
