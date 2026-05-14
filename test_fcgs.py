"""
Test suite per FCGS (Federated Computing Grid Search).
Copre: configurazioni JSON, logica di deduplicazione/fingerprinting,
       aggregazione pesi, utils, watchdog, assegnamento porte.
Eseguire dalla root del progetto con il venv attivo:
    python -m pytest test_fcgs.py -v
oppure:
    python test_fcgs.py
"""
import sys
import os
import json
import logging
import itertools
import unittest
from typing import Dict, FrozenSet, Set, Tuple
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Funzioni helper copiate da federated_grid_search.py (testate isolatamente)
# ---------------------------------------------------------------------------

def _get_config_fingerprint(config: Dict, keys: Set[str]) -> FrozenSet[Tuple[str, str]]:
    items = []
    for key in sorted(list(keys)):
        if key in config and config[key] is not None and config[key] != '':
            items.append((key, str(config[key])))
    return frozenset(items)


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _generate_configurations(base_config: Dict, search_space: Dict):
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def _normalize_config(hyper_config: Dict, model_name: str) -> Dict:
    """Applica le stesse normalizzazioni di federated_grid_search.py."""
    c = hyper_config.copy()
    if c.get('aggregation_algorithm') != 'FedProx':
        c['fedprox_mu'] = 0.0
    if c.get('encryption_mode') == 'no_encryption':
        c['he_backend'] = 'N/A'
    if 'ResNet' in model_name or 'GoogLeNet' in model_name or 'AlexNet' in model_name:
        c['image_size'] = 224
        c.setdefault('convnet_hidden1', -1)
        c.setdefault('convnet_hidden2', -1)
    elif 'ConvNet' in model_name:
        c.setdefault('num_custom_layers', -1)
    return c


def _count_unique_configs(gs_config: Dict) -> int:
    """Conta le configurazioni uniche eseguibili per un dato config JSON."""
    common_ss = gs_config['common_search_space']
    model_ss = gs_config['model_specific_search_space']

    all_keys: Set[str] = set(common_ss.keys())
    for mp in model_ss.values():
        all_keys.update(mp.keys())
    all_keys.update(['dataset_name', 'model_name'])

    seen: Set[FrozenSet] = set()

    for dataset_info in gs_config['datasets']:
        for model_name, specific_params in model_ss.items():
            base = {'dataset_name': dataset_info['name'], 'model_name': model_name}
            search_space = {**common_ss, **specific_params}
            for raw in _generate_configurations(base, search_space):
                norm = _normalize_config(raw, model_name)
                fp = _get_config_fingerprint(norm, all_keys)
                seen.add(fp)

    return len(seen)


# ---------------------------------------------------------------------------
# 1. VALIDAZIONE CONFIGURAZIONI JSON
# ---------------------------------------------------------------------------

REQUIRED_GRID_KEYS = {
    'ip_address', 'port', 'num_parallel_executions',
    'base_csv_path', 'base_log_path', 'base_split_data_path',
    'early_stop_patience', 'weighted_aggregation',
    'datasets', 'common_search_space', 'model_specific_search_space',
}

REQUIRED_COMMON_KEYS = {
    'encryption_mode', 'he_backend', 'device',
    'global_epoch', 'local_epoch', 'num_clients',
    'learning_rate', 'batch_size', 'aggregation_algorithm', 'fedprox_mu',
}

CONFIG_FILES = {
    'geosciences':  'grid_search_config.json',
    'MSIDomino11':  'grid_search_config_MSIDomino11.json',
    'MSIDomino12':  'grid_search_config_MSIDomino12.json',
}

EXPECTED_WORKERS = {
    'geosciences': 12,
    'MSIDomino11': 3,
    'MSIDomino12': 3,
}

EXPECTED_DATASET_COUNTS = {
    'geosciences': 8,
    'MSIDomino11': 9,
    'MSIDomino12': 5,
}


class TestConfigValidation(unittest.TestCase):

    def _load(self, fname: str) -> Dict:
        path = os.path.join(PROJECT_ROOT, fname)
        self.assertTrue(os.path.exists(path), f"Config file not found: {fname}")
        with open(path) as f:
            return json.load(f)

    def test_all_configs_are_valid_json(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                self.assertIsInstance(cfg, dict)

    def test_required_top_level_keys(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                for key in REQUIRED_GRID_KEYS:
                    self.assertIn(key, cfg, f"'{key}' mancante in {fname}")

    def test_required_common_search_space_keys(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                cs = cfg['common_search_space']
                for key in REQUIRED_COMMON_KEYS:
                    self.assertIn(key, cs, f"'{key}' mancante in common_search_space di {fname}")

    def test_num_parallel_executions(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                self.assertEqual(
                    cfg['num_parallel_executions'], EXPECTED_WORKERS[machine],
                    f"num_parallel_executions errato per {machine}"
                )

    def test_dataset_counts(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                self.assertEqual(
                    len(cfg['datasets']), EXPECTED_DATASET_COUNTS[machine],
                    f"Numero dataset errato per {machine}"
                )

    def test_datasets_have_required_fields(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                for ds in cfg['datasets']:
                    self.assertIn('name', ds)
                    self.assertIn('path', ds)
                    self.assertIn('num_classes', ds)
                    self.assertIsInstance(ds['num_classes'], int)
                    self.assertGreater(ds['num_classes'], 0)

    def test_no_dataset_overlap_between_machines(self):
        sets = {}
        for machine, fname in CONFIG_FILES.items():
            cfg = self._load(fname)
            sets[machine] = {ds['name'] for ds in cfg['datasets']}

        machines = list(sets.keys())
        for i in range(len(machines)):
            for j in range(i + 1, len(machines)):
                m1, m2 = machines[i], machines[j]
                overlap = sets[m1] & sets[m2]
                self.assertEqual(
                    len(overlap), 0,
                    f"Dataset overlap tra {m1} e {m2}: {overlap}"
                )

    def test_all_22_datasets_covered(self):
        all_datasets = set()
        for fname in CONFIG_FILES.values():
            cfg = self._load(fname)
            for ds in cfg['datasets']:
                all_datasets.add(ds['name'])
        self.assertEqual(len(all_datasets), 22, f"Totale dataset: {len(all_datasets)}, attesi 22")

    def test_geosciences_has_only_slices_datasets(self):
        cfg = self._load(CONFIG_FILES['geosciences'])
        for ds in cfg['datasets']:
            self.assertTrue(
                ds['name'].startswith('dataset_slices_sorted'),
                f"Dataset non-slices in geosciences: {ds['name']}"
            )

    def test_domino11_has_multy_and_brain(self):
        cfg = self._load(CONFIG_FILES['MSIDomino11'])
        names = {ds['name'] for ds in cfg['datasets']}
        self.assertIn('brain_tumor', names)
        for n in names:
            if n != 'brain_tumor':
                self.assertTrue(n.startswith('dataset_multy_sorted'), f"Dataset inatteso in Domino11: {n}")

    def test_domino12_has_skin_datasets(self):
        cfg = self._load(CONFIG_FILES['MSIDomino12'])
        names = {ds['name'] for ds in cfg['datasets']}
        expected = {'dataset_HAM1000', 'dataset_ISIC', 'dataset_ISIC2', 'dataset_RD_colored', 'dataset_RD_kdr299'}
        self.assertEqual(names, expected)

    def test_port_is_integer(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                self.assertIsInstance(cfg['port'], int)

    def test_search_space_lists_non_empty(self):
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                cfg = self._load(fname)
                for key, val in cfg['common_search_space'].items():
                    self.assertIsInstance(val, list, f"'{key}' non è lista in {fname}")
                    self.assertGreater(len(val), 0, f"'{key}' è lista vuota in {fname}")


# ---------------------------------------------------------------------------
# 2. LOGICA DI GENERAZIONE CONFIGURAZIONI E FINGERPRINTING
# ---------------------------------------------------------------------------

class TestFingerprintNormalization(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(PROJECT_ROOT, 'grid_search_config.json')) as f:
            self.cfg = json.load(f)
        self.all_keys: Set[str] = set(self.cfg['common_search_space'].keys())
        for mp in self.cfg['model_specific_search_space'].values():
            self.all_keys.update(mp.keys())
        self.all_keys.update(['dataset_name', 'model_name'])

    def test_fedavg_forces_fedprox_mu_to_zero(self):
        c = {'aggregation_algorithm': 'FedAvg', 'fedprox_mu': 0.05, 'encryption_mode': 'no_encryption',
             'model_name': 'ConvNet'}
        norm = _normalize_config(c, 'ConvNet')
        self.assertEqual(norm['fedprox_mu'], 0.0)

    def test_fedprox_keeps_mu(self):
        c = {'aggregation_algorithm': 'FedProx', 'fedprox_mu': 0.05, 'encryption_mode': 'no_encryption',
             'model_name': 'ConvNet'}
        norm = _normalize_config(c, 'ConvNet')
        self.assertEqual(norm['fedprox_mu'], 0.05)

    def test_he_backend_normalized_to_na_for_no_encryption(self):
        """Con no_encryption he_backend viene normalizzato a N/A (schema irrilevante)."""
        c = {'aggregation_algorithm': 'FedAvg', 'fedprox_mu': 0.05,
             'encryption_mode': 'no_encryption', 'he_backend': 'ckks', 'model_name': 'ConvNet'}
        norm = _normalize_config(c, 'ConvNet')
        self.assertEqual(norm['he_backend'], 'N/A')

    def test_he_backend_kept_for_homomorphic(self):
        """Con cifratura omomorfica he_backend rimane ckks (schema usato a runtime)."""
        c = {'aggregation_algorithm': 'FedAvg', 'fedprox_mu': 0.05,
             'encryption_mode': 'homomorphic', 'he_backend': 'ckks', 'model_name': 'ConvNet'}
        norm = _normalize_config(c, 'ConvNet')
        self.assertEqual(norm['he_backend'], 'ckks')

    def test_googlenet_forces_image_size_224(self):
        for model in ('GoogLeNet', 'AlexNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'):
            with self.subTest(model=model):
                c = {'image_size': 128, 'encryption_mode': 'no_encryption', 'aggregation_algorithm': 'FedAvg',
                     'fedprox_mu': 0.05}
                norm = _normalize_config(c, model)
                self.assertEqual(norm['image_size'], 224, f"{model} deve avere image_size=224")

    def test_convnet_does_not_force_image_size(self):
        c = {'image_size': 128, 'encryption_mode': 'no_encryption', 'aggregation_algorithm': 'FedAvg',
             'fedprox_mu': 0.05}
        norm = _normalize_config(c, 'ConvNet')
        self.assertEqual(norm['image_size'], 128)

    def test_he_backend_is_ckks_only(self):
        """Per ora he_backend usa solo ckks."""
        for machine, fname in CONFIG_FILES.items():
            with self.subTest(machine=machine):
                with open(os.path.join(PROJECT_ROOT, fname)) as f:
                    cfg = json.load(f)
                backends = cfg['common_search_space']['he_backend']
                self.assertEqual(backends, ['ckks'], f"he_backend deve essere ['ckks'] in {fname}")

    def test_image_size_128_224_deduplicates_for_resnet(self):
        base = {'dataset_name': 'ds', 'model_name': 'ResNet18',
                'aggregation_algorithm': 'FedAvg', 'fedprox_mu': 0.05,
                'encryption_mode': 'no_encryption', 'he_backend': 'ckks',
                'num_custom_layers': 2, 'global_epoch': 5}
        c128 = {**base, 'image_size': 128}
        c224 = {**base, 'image_size': 224}
        norm128 = _normalize_config(c128, 'ResNet18')
        norm224 = _normalize_config(c224, 'ResNet18')
        fp128 = _get_config_fingerprint(norm128, self.all_keys)
        fp224 = _get_config_fingerprint(norm224, self.all_keys)
        self.assertEqual(fp128, fp224, "image_size 128 e 224 devono deduplicarsi per ResNet")


class TestConfigCounting(unittest.TestCase):

    def _load(self, fname: str) -> Dict:
        with open(os.path.join(PROJECT_ROOT, fname)) as f:
            return json.load(f)

    def test_geosciences_unique_config_count(self):
        cfg = self._load('grid_search_config.json')
        count = _count_unique_configs(cfg)
        # 2 combo cifratura: no_enc(N/A) + homomorphic+ckks
        self.assertEqual(count, 201_600,
                         f"geosciences: attese 201600 config uniche, trovate {count}")

    def test_domino11_unique_config_count(self):
        cfg = self._load('grid_search_config_MSIDomino11.json')
        count = _count_unique_configs(cfg)
        self.assertEqual(count, 226_800,
                         f"MSIDomino11: attese 226800 config uniche, trovate {count}")

    def test_domino12_unique_config_count(self):
        cfg = self._load('grid_search_config_MSIDomino12.json')
        count = _count_unique_configs(cfg)
        self.assertEqual(count, 126_000,
                         f"MSIDomino12: attese 126000 config uniche, trovate {count}")

    def test_total_config_count(self):
        total = 0
        for fname in CONFIG_FILES.values():
            cfg = self._load(fname)
            total += _count_unique_configs(cfg)
        self.assertEqual(total, 554_400,
                         f"Totale config uniche: attese 554400, trovate {total}")


# ---------------------------------------------------------------------------
# 3. UTILS: SERIALIZZAZIONE PICKLE
# ---------------------------------------------------------------------------

class TestUtils(unittest.TestCase):

    def setUp(self):
        from utils import object_to_pickle_string, pickle_string_to_object
        self.serialize = object_to_pickle_string
        self.deserialize = pickle_string_to_object

    def test_roundtrip_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        s = self.serialize(arr)
        result = self.deserialize(s)
        np.testing.assert_array_almost_equal(arr, result)

    def test_roundtrip_list_of_numpy(self):
        data = [np.ones((3, 4)), np.zeros((5,))]
        s = self.serialize(data)
        result = self.deserialize(s)
        for orig, restored in zip(data, result):
            np.testing.assert_array_almost_equal(orig, restored)

    def test_roundtrip_dict(self):
        d = {'a': 1, 'b': [1, 2, 3], 'c': 'hello'}
        s = self.serialize(d)
        result = self.deserialize(s)
        self.assertEqual(d, result)

    def test_roundtrip_scalar(self):
        val = 42.5
        s = self.serialize(val)
        result = self.deserialize(s)
        self.assertAlmostEqual(val, result)

    def test_serialized_is_string(self):
        obj = {'key': 'value'}
        s = self.serialize(obj)
        self.assertIsInstance(s, str)

    def test_deterministic_content(self):
        arr = np.array([1.0, 2.0, 3.0])
        s1 = self.serialize(arr)
        s2 = self.serialize(arr)
        self.assertEqual(s1, s2)


# ---------------------------------------------------------------------------
# 4. AGGREGATORE (con ModelManager mockato)
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile (installarlo con: pip install torch)")
class TestAggregatorLogic(unittest.TestCase):

    def _make_aggregator(self, mock_mm_class, early_stop_patience=3):
        mock_instance = MagicMock()
        mock_instance.get_weights.return_value = [np.zeros((2, 2)), np.zeros(2)]
        mock_mm_class.return_value = mock_instance

        logger = logging.getLogger('test_aggregator')
        logger.addHandler(logging.NullHandler())

        config = {
            'model_name': 'ResNet18',
            'num_classes': 2,
            'early_stop_patience': early_stop_patience,
            'image_size': 224,
            'num_custom_layers': 2,
            'run_metrics_output_path': '/tmp/test_metrics',
            'worker_id': 0,
            'dataset_name': 'test_ds',
            'plot_dir': '/tmp/test_plots',
        }
        from aggregator import Aggregator
        return Aggregator(config, logger)

    @patch('aggregator.ModelManager')
    def test_aggregate_weights_fedavg(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        w1 = [np.array([1.0, 2.0]), np.array([3.0])]
        w2 = [np.array([3.0, 4.0]), np.array([1.0])]
        updates = [
            {'weights': w1, 'train_size': 10},
            {'weights': w2, 'train_size': 10},
        ]
        result = agg.aggregate_weights(updates, 'FedAvg')
        self.assertTrue(result)
        # Somma pesata: w1*10 + w2*10
        np.testing.assert_array_almost_equal(agg.current_weights[0], np.array([40.0, 60.0]))
        np.testing.assert_array_almost_equal(agg.current_weights[1], np.array([40.0]))

    @patch('aggregator.ModelManager')
    def test_aggregate_weights_fedprox(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        w1 = [np.array([2.0])]
        w2 = [np.array([4.0])]
        updates = [
            {'weights': w1, 'train_size': 5},
            {'weights': w2, 'train_size': 15},
        ]
        result = agg.aggregate_weights(updates, 'FedProx')
        self.assertTrue(result)
        np.testing.assert_array_almost_equal(agg.current_weights[0], np.array([2.0*5 + 4.0*15]))

    @patch('aggregator.ModelManager')
    def test_aggregate_weights_empty_updates(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        result = agg.aggregate_weights([], 'FedAvg')
        self.assertFalse(result)

    @patch('aggregator.ModelManager')
    def test_aggregate_weights_zero_total_size(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        updates = [{'weights': [np.array([1.0])], 'train_size': 0}]
        result = agg.aggregate_weights(updates, 'FedAvg')
        self.assertFalse(result)

    @patch('aggregator.ModelManager')
    def test_aggregate_weights_unsupported_algorithm(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        updates = [{'weights': [np.array([1.0])], 'train_size': 10}]
        with self.assertRaises(ValueError):
            agg.aggregate_weights(updates, 'UnknownAlgo')

    @patch('aggregator.ModelManager')
    def test_aggregate_train_loss_weighted(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        agg.metrics_history[0] = {}
        agg.aggregate_train_loss_weighted([0.5, 0.3], [10, 10], current_round=0)
        self.assertAlmostEqual(agg.metrics_history[0]['train_loss'], 0.4, places=5)

    @patch('aggregator.ModelManager')
    def test_aggregate_train_loss_weighted_unequal(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        agg.metrics_history[0] = {}
        agg.aggregate_train_loss_weighted([1.0, 0.0], [1, 3], current_round=0)
        expected = (1.0 * 1 + 0.0 * 3) / 4
        self.assertAlmostEqual(agg.metrics_history[0]['train_loss'], expected, places=5)

    @patch('aggregator.ModelManager')
    def test_aggregate_evaluation_updates_best_f1(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        agg.metrics_history[0] = {}
        eval_updates = [
            {'test_loss': 0.3, 'test_f1': 0.8, 'test_acc': 0.85,
             'test_prec': 0.82, 'test_recall': 0.78, 'test_size': 50},
            {'test_loss': 0.4, 'test_f1': 0.7, 'test_acc': 0.75,
             'test_prec': 0.72, 'test_recall': 0.68, 'test_size': 50},
        ]
        agg.aggregate_evaluation_results(eval_updates, 0)
        self.assertAlmostEqual(agg.best_f1_score, 0.75, places=5)

    @patch('aggregator.ModelManager')
    def test_early_stopping_triggered(self, mock_mm):
        agg = self._make_aggregator(mock_mm, early_stop_patience=3)

        def make_eval(loss, f1):
            return [{'test_loss': loss, 'test_f1': f1, 'test_acc': 0.8,
                     'test_prec': 0.8, 'test_recall': 0.8, 'test_size': 100}]

        agg.metrics_history = {0: {}, 1: {}, 2: {}, 3: {}}
        agg.aggregate_evaluation_results(make_eval(0.3, 0.8), 0)  # prev=None → no increment
        agg.aggregate_evaluation_results(make_eval(0.4, 0.7), 1)  # peggioramento → counter=1
        agg.aggregate_evaluation_results(make_eval(0.5, 0.6), 2)  # peggioramento → counter=2
        stop = agg.aggregate_evaluation_results(make_eval(0.6, 0.5), 3)  # counter=3 → stop
        self.assertTrue(stop, "Early stopping deve scattare dopo 3 peggioramenti consecutivi")

    @patch('aggregator.ModelManager')
    def test_early_stopping_reset_on_improvement(self, mock_mm):
        agg = self._make_aggregator(mock_mm, early_stop_patience=3)

        def make_eval(loss, f1):
            return [{'test_loss': loss, 'test_f1': f1, 'test_acc': 0.8,
                     'test_prec': 0.8, 'test_recall': 0.8, 'test_size': 100}]

        agg.metrics_history = {i: {} for i in range(5)}
        agg.aggregate_evaluation_results(make_eval(0.3, 0.8), 0)
        agg.aggregate_evaluation_results(make_eval(0.4, 0.7), 1)  # counter=1
        agg.aggregate_evaluation_results(make_eval(0.5, 0.6), 2)  # counter=2
        agg.aggregate_evaluation_results(make_eval(0.2, 0.9), 3)  # miglioramento → reset=0
        stop = agg.aggregate_evaluation_results(make_eval(0.3, 0.8), 4)  # counter=1 → no stop
        self.assertFalse(stop, "Early stopping non deve scattare dopo reset")

    @patch('aggregator.ModelManager')
    def test_get_stats_returns_dataframe(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        import pandas as pd
        agg.metrics_history[0] = {'train_loss': 0.5, 'test_loss': 0.4, 'test_f1': 0.7,
                                   'test_acc': 0.75, 'test_prec': 0.72, 'test_recall': 0.68}
        df = agg.get_stats()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('train_loss', df.columns)

    @patch('aggregator.ModelManager')
    def test_get_run_summary_none_before_save(self, mock_mm):
        agg = self._make_aggregator(mock_mm)
        self.assertIsNone(agg.get_run_summary())


# ---------------------------------------------------------------------------
# 5. PORT ASSIGNMENT: nessuna collisione tra worker
# ---------------------------------------------------------------------------

class TestPortAssignment(unittest.TestCase):

    def _get_ports(self, base_port: int, num_workers: int):
        ports = set()
        for i in range(num_workers):
            server_port = base_port + i * 2
            ta_port = base_port + i * 2 + 1
            ports.add(server_port)
            ports.add(ta_port)
        return ports

    def test_no_port_collision_geosciences(self):
        """12 worker, base 5001 → porta max 5001+11*2+1=5024. No overlap."""
        ports = self._get_ports(5001, 12)
        self.assertEqual(len(ports), 24)  # 12 worker × 2 porte ciascuno

    def test_no_port_collision_domino(self):
        """3 worker, base 5001 → 6 porte distinte."""
        ports = self._get_ports(5001, 3)
        self.assertEqual(len(ports), 6)

    def test_geosciences_port_range(self):
        ports = self._get_ports(5001, 12)
        self.assertEqual(min(ports), 5001)
        self.assertEqual(max(ports), 5001 + 11 * 2 + 1)  # 5024

    def test_server_and_ta_ports_alternate(self):
        """Verifica la formula: server=base+i*2, ta=base+i*2+1."""
        for i in range(12):
            server = 5001 + i * 2
            ta = 5001 + i * 2 + 1
            self.assertEqual(ta - server, 1)


# ---------------------------------------------------------------------------
# 6. FORMAT DURATION
# ---------------------------------------------------------------------------

class TestFormatDuration(unittest.TestCase):

    def test_seconds_only(self):
        self.assertEqual(_format_duration(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(_format_duration(125), "2m 5s")

    def test_hours_minutes_seconds(self):
        self.assertEqual(_format_duration(3723), "1h 2m 3s")

    def test_zero(self):
        self.assertEqual(_format_duration(0), "0s")

    def test_exactly_one_hour(self):
        self.assertEqual(_format_duration(3600), "1h 0m 0s")

    def test_exactly_one_minute(self):
        self.assertEqual(_format_duration(60), "1m 0s")


# ---------------------------------------------------------------------------
# 7. FEDERATED SERVER: variabili watchdog e dropout presenti
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
class TestFederatedServerWatchdog(unittest.TestCase):

    @patch('aggregator.ModelManager')
    @patch('federated_server.ModelManager', create=True)
    def test_server_has_watchdog_state(self, mock_mm_server, mock_mm_agg):
        """Verifica che FederatedServer abbia tutte le variabili del watchdog."""
        mock_instance = MagicMock()
        mock_instance.get_weights.return_value = [np.zeros((2, 2))]
        mock_mm_agg.return_value = mock_instance

        # Patch anche aggregator.ModelManager usato da Aggregator
        with patch('aggregator.ModelManager') as mock_agg_mm:
            mock_agg_mm.return_value = mock_instance

            config = {
                'ip_address': '127.0.0.1',
                'port': 59999,
                'ta_port': 60000,
                'model_name': 'ResNet18',
                'num_classes': 2,
                'early_stop_patience': 5,
                'image_size': 224,
                'num_custom_layers': 2,
                'MIN_NUM_WORKERS': 2,
                'encryption_mode': 'no_encryption',
                'round_timeout': 150,
                'run_metrics_output_path': '/tmp',
                'plot_dir': '/tmp',
                'dataset_name': 'test',
                'worker_id': 0,
            }

            from federated_server import FederatedServer
            server = FederatedServer(config)

            # Variabili watchdog
            self.assertTrue(hasattr(server, 'round_phase'))
            self.assertTrue(hasattr(server, '_last_activity'))
            self.assertTrue(hasattr(server, '_round_timeout'))
            self.assertTrue(hasattr(server, 'clients_selected_for_round'))
            self.assertTrue(hasattr(server, 'clients_submitted_update'))
            self.assertTrue(hasattr(server, 'clients_submitted_eval'))

            # Valori iniziali
            self.assertEqual(server.round_phase, 'idle')
            self.assertEqual(server._round_timeout, 150)
            self.assertIsInstance(server.clients_selected_for_round, set)
            self.assertIsInstance(server.clients_submitted_update, set)
            self.assertIsInstance(server.clients_submitted_eval, set)

    def test_watchdog_round_timeout_default(self):
        """Verifica il valore di default del round_timeout."""
        with patch('aggregator.ModelManager') as mock_agg_mm:
            mock_instance = MagicMock()
            mock_instance.get_weights.return_value = [np.zeros((2, 2))]
            mock_agg_mm.return_value = mock_instance
            config = {
                'ip_address': '127.0.0.1', 'port': 59998, 'ta_port': 59999,
                'model_name': 'ResNet18', 'num_classes': 2, 'early_stop_patience': 5,
                'image_size': 224, 'num_custom_layers': 2, 'MIN_NUM_WORKERS': 2,
                'encryption_mode': 'no_encryption',
                'run_metrics_output_path': '/tmp', 'plot_dir': '/tmp',
                'dataset_name': 'test', 'worker_id': 0,
            }
            from federated_server import FederatedServer
            server = FederatedServer(config)
            self.assertEqual(server._round_timeout, 150,
                             "round_timeout default deve essere 150s")


# ---------------------------------------------------------------------------
# 7b. SOURCE CHECK: static/federated_server.py contiene codice watchdog
# ---------------------------------------------------------------------------

class TestStaticServerSource(unittest.TestCase):
    """Test su file sorgente — non richiede torch."""

    def test_static_server_has_watchdog_symbols(self):
        static_path = os.path.join(PROJECT_ROOT, 'static', 'federated_server.py')
        self.assertTrue(os.path.exists(static_path), "static/federated_server.py non trovato")
        with open(static_path) as f:
            source = f.read()
        for symbol in ('round_phase', '_last_activity', '_round_timeout',
                        'clients_selected_for_round', 'clients_submitted_update',
                        '_watchdog_loop', 'round_timeout'):
            self.assertIn(symbol, source,
                          f"'{symbol}' non trovato in static/federated_server.py")

    def test_root_server_has_watchdog_symbols(self):
        root_path = os.path.join(PROJECT_ROOT, 'federated_server.py')
        with open(root_path) as f:
            source = f.read()
        for symbol in ('round_phase', '_last_activity', '_round_timeout',
                        'clients_selected_for_round', 'clients_submitted_update',
                        '_watchdog_loop'):
            self.assertIn(symbol, source,
                          f"'{symbol}' non trovato in federated_server.py")

    def test_static_server_has_disconnect_handling(self):
        static_path = os.path.join(PROJECT_ROOT, 'static', 'federated_server.py')
        with open(static_path) as f:
            source = f.read()
        self.assertIn('_on_disconnect', source)
        self.assertIn('num_clients_per_round', source)
        self.assertIn('expected_eval_count', source)

    def test_static_server_not_identical_to_root(self):
        """Static e root NON devono essere identici (la static ha differenze)."""
        with open(os.path.join(PROJECT_ROOT, 'federated_server.py')) as f:
            root = f.read()
        with open(os.path.join(PROJECT_ROOT, 'static', 'federated_server.py')) as f:
            static = f.read()
        self.assertNotEqual(root, static,
                            "static e root federated_server.py non devono essere identici")


# ---------------------------------------------------------------------------
# 8. IMPORT CHECK: tutti i moduli principali importabili
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):

    def test_import_utils(self):
        import utils
        self.assertTrue(hasattr(utils, 'object_to_pickle_string'))
        self.assertTrue(hasattr(utils, 'pickle_string_to_object'))

    def test_import_PCNAME(self):
        import PCNAME
        self.assertTrue(hasattr(PCNAME, 'name'))
        self.assertIsInstance(PCNAME.name, str)
        self.assertGreater(len(PCNAME.name), 0)

    def test_import_data_splitter(self):
        import data_splitter
        self.assertTrue(hasattr(data_splitter, 'DatasetSplitter'))

    def test_import_trusted_authority(self):
        import trusted_authority
        self.assertTrue(hasattr(trusted_authority, 'TrustedAuthority'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_model_manager(self):
        import model_manager
        self.assertTrue(hasattr(model_manager, 'ModelManager'))
        self.assertTrue(hasattr(model_manager, 'ConvNet'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_aggregator(self):
        import aggregator
        self.assertTrue(hasattr(aggregator, 'Aggregator'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_federated_grid_search(self):
        import federated_grid_search
        self.assertTrue(hasattr(federated_grid_search, 'main'))
        self.assertTrue(hasattr(federated_grid_search, 'get_config_fingerprint'))
        self.assertTrue(hasattr(federated_grid_search, 'generate_configurations'))
        self.assertTrue(hasattr(federated_grid_search, 'send_telegram'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_federated_server(self):
        import federated_server
        self.assertTrue(hasattr(federated_server, 'FederatedServer'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_federated_client(self):
        import federated_client
        self.assertTrue(hasattr(federated_client, 'FederatedClient'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch non disponibile")
    def test_import_run_multiple_clients(self):
        import run_multiple_clients
        self.assertTrue(hasattr(run_multiple_clients, 'main'))


# ---------------------------------------------------------------------------
# 9. GRID SEARCH CONFIG LOADING (come lo fa federated_grid_search.main)
# ---------------------------------------------------------------------------

class TestGridSearchConfigLoading(unittest.TestCase):

    def test_geosciences_config_loads_correctly(self):
        with open(os.path.join(PROJECT_ROOT, 'grid_search_config.json')) as f:
            cfg = json.load(f)
        self.assertIsInstance(cfg['datasets'], list)
        self.assertIsInstance(cfg['common_search_space'], dict)
        self.assertIsInstance(cfg['model_specific_search_space'], dict)

    def test_config_all_models_present(self):
        expected_models = {'GoogLeNet', 'AlexNet', 'ConvNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'}
        for fname in CONFIG_FILES.values():
            with open(os.path.join(PROJECT_ROOT, fname)) as f:
                cfg = json.load(f)
            models = set(cfg['model_specific_search_space'].keys())
            self.assertEqual(models, expected_models, f"Modelli mancanti/extra in {fname}: {models ^ expected_models}")

    def test_encryption_modes_present(self):
        for fname in CONFIG_FILES.values():
            with open(os.path.join(PROJECT_ROOT, fname)) as f:
                cfg = json.load(f)
            modes = set(cfg['common_search_space']['encryption_mode'])
            self.assertIn('no_encryption', modes, f"no_encryption mancante in {fname}")
            self.assertIn('homomorphic', modes, f"homomorphic mancante in {fname}")

    def test_aggregation_algorithms_are_fedavg_fedprox(self):
        for fname in CONFIG_FILES.values():
            with open(os.path.join(PROJECT_ROOT, fname)) as f:
                cfg = json.load(f)
            algos = set(cfg['common_search_space']['aggregation_algorithm'])
            self.assertEqual(algos, {'FedAvg', 'FedProx'}, f"Algoritmi errati in {fname}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
