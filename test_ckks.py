"""
Test CKKS encryption end-to-end:
  1. Round-trip encrypt -> decrypt (correttezza numerica)
  2. Aggregazione omomorfica FedAvg (somma pesata + divisione lato client)
  3. Avvio TA con backend CKKS e distribuzione chiavi via SocketIO
"""
import codecs
import threading
import time
import sys
import socket as _socket
import numpy as np
import socketio as sio_lib

import tenseal as ts
from utils import (encrypt_weights_ckks, decrypt_weights_ckks,
                   sum_encrypted_weights_ckks, multiply_encrypted_weights_ckks_by_scalar,
                   encrypt_weights_bfv, decrypt_weights_bfv,
                   sum_encrypted_weights_bfv, multiply_encrypted_weights_bfv_by_scalar)
from trusted_authority import TrustedAuthority

TOLERANCE = 1e-3   # CKKS e' approssimato, errore atteso <0.1%

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx


def fake_weights():
    """Simula i pesi di una rete piccola: 3 layer di dimensioni diverse."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((64, 3, 3, 3)).astype(np.float32),   # conv  ~1728 params
        rng.standard_normal((128,)).astype(np.float32),           # bias  ~128  params
        rng.standard_normal((512, 128)).astype(np.float32),       # fc    ~65536 params (>4096 slot)
    ]


def max_abs_err(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def wait_for_port(host, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# TEST 1: round-trip encrypt -> decrypt
# ---------------------------------------------------------------------------

def test_roundtrip():
    print("\n[TEST 1] Round-trip encrypt -> decrypt")
    ctx = make_context()
    weights = fake_weights()

    enc = encrypt_weights_ckks(ctx, weights)
    dec = decrypt_weights_ckks(ctx, enc)

    assert len(dec) == len(weights), "Numero di layer diverso dopo decryption"
    for i, (orig, rec) in enumerate(zip(weights, dec)):
        assert orig.shape == rec.shape, "Layer %d: shape mismatch %s != %s" % (i, orig.shape, rec.shape)
        err = float(np.max(np.abs(orig - rec)))
        assert err < TOLERANCE, "Layer %d: errore %.6f supera tolleranza %.4f" % (i, err, TOLERANCE)
        print("  Layer %d: shape=%s  max_err=%.2e  OK" % (i, orig.shape, err))
    print("[TEST 1] PASSED")


# ---------------------------------------------------------------------------
# TEST 2: aggregazione omomorfica FedAvg (2 client)
# ---------------------------------------------------------------------------

def test_fedavg_aggregation():
    print("\n[TEST 2] Aggregazione omomorfica FedAvg (2 client)")
    full_ctx = make_context()
    full_ctx_bytes = full_ctx.serialize(save_secret_key=True)

    full_ctx.make_context_public()
    pub_ctx_bytes = full_ctx.serialize(save_secret_key=False)
    pub_ctx  = ts.context_from(pub_ctx_bytes)
    full_ctx = ts.context_from(full_ctx_bytes)   # ripristina con secret key

    w1 = fake_weights()
    w2 = [w * 2 for w in fake_weights()]   # pesi diversi per i due client
    s1, s2 = 100, 200
    total = s1 + s2

    # lato client: cifra con full context
    enc1 = encrypt_weights_ckks(full_ctx, w1)
    enc2 = encrypt_weights_ckks(full_ctx, w2)

    # lato server: aggrega con public context (somma pesata)
    weighted1 = multiply_encrypted_weights_ckks_by_scalar(pub_ctx, enc1, s1)
    weighted2 = multiply_encrypted_weights_ckks_by_scalar(pub_ctx, enc2, s2)
    aggregated = sum_encrypted_weights_ckks(pub_ctx, weighted1, weighted2)

    # lato client: decifra e divide per total (FedAvg)
    dec_sum = decrypt_weights_ckks(full_ctx, aggregated)
    result  = [layer / total for layer in dec_sum]

    # verifica vs calcolo in chiaro
    expected = [(w1[i] * s1 + w2[i] * s2) / total for i in range(len(w1))]
    err = max_abs_err(result, expected)
    assert err < TOLERANCE, "Errore aggregazione %.6f supera tolleranza %.4f" % (err, TOLERANCE)
    print("  Max errore su tutti i layer: %.2e  OK" % err)
    print("[TEST 2] PASSED")


# ---------------------------------------------------------------------------
# TEST 3: TA con backend CKKS -- avvio e distribuzione chiavi
# ---------------------------------------------------------------------------

def test_ta_ckks():
    print("\n[TEST 3] Trusted Authority CKKS -- avvio e distribuzione chiavi")
    TA_PORT = 15099
    keys_received = threading.Event()
    result = {}

    ta = TrustedAuthority(host='127.0.0.1', port=TA_PORT, he_backend='ckks')
    ta_thread = threading.Thread(target=ta.run, daemon=True)
    ta_thread.start()

    assert wait_for_port('127.0.0.1', TA_PORT, timeout=20), "TA non ha aperto la porta entro 20s"

    client = sio_lib.Client(reconnection=False)

    @client.on('distribute_keys')
    def on_keys(data):
        result['data'] = data
        keys_received.set()
        client.disconnect()

    try:
        client.connect('http://127.0.0.1:%d' % TA_PORT, transports=['websocket'])
        client.emit('request_keys')
        assert keys_received.wait(timeout=15), "Timeout: chiavi CKKS non ricevute dalla TA"
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    data = result['data']
    assert data.get('backend') == 'ckks', "Backend atteso 'ckks', ricevuto '%s'" % data.get('backend')
    assert 'full_context' in data and 'public_context' in data, "Campi mancanti nella risposta TA"

    full_bytes = codecs.decode(data['full_context'].encode(), 'base64')
    pub_bytes  = codecs.decode(data['public_context'].encode(), 'base64')
    full_ctx = ts.context_from(full_bytes)
    pub_ctx  = ts.context_from(pub_bytes)

    assert full_ctx.is_private(), "Il full context dovrebbe avere la secret key"
    assert not pub_ctx.is_private(), "Il public context NON dovrebbe avere la secret key"
    print("  TA avviata, chiavi ricevute, full_ctx privato, pub_ctx pubblico  OK")

    # Shutdown TA
    try:
        sc = sio_lib.Client(reconnection=False)
        sc.connect('http://127.0.0.1:%d' % TA_PORT, transports=['websocket'])
        sc.emit('shutdown_ta')
        time.sleep(1)
        sc.disconnect()
    except Exception:
        pass

    print("[TEST 3] PASSED")


# ---------------------------------------------------------------------------
# TEST 4: BFV round-trip encrypt -> decrypt
# ---------------------------------------------------------------------------

def make_bfv_context():
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=7340033)
    return ctx


def test_bfv_roundtrip():
    print("\n[TEST 4] BFV round-trip encrypt -> decrypt")
    BFV_SCALE = 1000
    BFV_TOLERANCE = 1.0 / BFV_SCALE   # errore max atteso: meta' del passo di quantizzazione

    ctx = make_bfv_context()
    weights = fake_weights()

    enc = encrypt_weights_bfv(ctx, weights, scale=BFV_SCALE)
    dec = decrypt_weights_bfv(ctx, enc)

    assert len(dec) == len(weights), "Numero di layer diverso dopo decryption"
    for i, (orig, rec) in enumerate(zip(weights, dec)):
        assert orig.shape == rec.shape, "Layer %d: shape mismatch" % i
        err = float(np.max(np.abs(orig - rec)))
        assert err < BFV_TOLERANCE, "Layer %d: errore %.6f supera tolleranza %.6f" % (i, err, BFV_TOLERANCE)
        print("  Layer %d: shape=%s  max_err=%.2e  OK" % (i, orig.shape, err))
    print("[TEST 4] PASSED")


# ---------------------------------------------------------------------------
# TEST 5: BFV aggregazione omomorfica FedAvg (2 client)
# ---------------------------------------------------------------------------

def test_bfv_fedavg_aggregation():
    print("\n[TEST 5] BFV aggregazione omomorfica FedAvg (2 client)")
    BFV_SCALE = 1000
    BFV_TOLERANCE = 1.0 / BFV_SCALE

    full_ctx = make_bfv_context()
    full_ctx_bytes = full_ctx.serialize(save_secret_key=True)
    full_ctx.make_context_public()
    pub_ctx_bytes = full_ctx.serialize(save_secret_key=False)
    pub_ctx  = ts.context_from(pub_ctx_bytes)
    full_ctx = ts.context_from(full_ctx_bytes)

    w1 = fake_weights()
    w2 = [w * 2 for w in fake_weights()]
    s1, s2 = 100, 200
    total = s1 + s2

    enc1 = encrypt_weights_bfv(full_ctx, w1, scale=BFV_SCALE)
    enc2 = encrypt_weights_bfv(full_ctx, w2, scale=BFV_SCALE)

    weighted1 = multiply_encrypted_weights_bfv_by_scalar(pub_ctx, enc1, s1)
    weighted2 = multiply_encrypted_weights_bfv_by_scalar(pub_ctx, enc2, s2)
    aggregated = sum_encrypted_weights_bfv(pub_ctx, weighted1, weighted2)

    dec_sum = decrypt_weights_bfv(full_ctx, aggregated)
    result  = [layer / total for layer in dec_sum]

    expected = [(w1[i] * s1 + w2[i] * s2) / total for i in range(len(w1))]
    err = max_abs_err(result, expected)
    assert err < BFV_TOLERANCE, "Errore aggregazione %.6f supera tolleranza %.6f" % (err, BFV_TOLERANCE)
    print("  Max errore su tutti i layer: %.2e  OK" % err)
    print("[TEST 5] PASSED")


# ---------------------------------------------------------------------------
# TEST 6: TA con backend BFV -- avvio e distribuzione chiavi
# ---------------------------------------------------------------------------

def test_ta_bfv():
    print("\n[TEST 6] Trusted Authority BFV -- avvio e distribuzione chiavi")
    TA_PORT = 15100
    keys_received = threading.Event()
    result = {}

    ta = TrustedAuthority(host='127.0.0.1', port=TA_PORT, he_backend='bfv')
    ta_thread = threading.Thread(target=ta.run, daemon=True)
    ta_thread.start()

    assert wait_for_port('127.0.0.1', TA_PORT, timeout=20), "TA non ha aperto la porta entro 20s"

    client = sio_lib.Client(reconnection=False)

    @client.on('distribute_keys')
    def on_keys(data):
        result['data'] = data
        keys_received.set()
        client.disconnect()

    try:
        client.connect('http://127.0.0.1:%d' % TA_PORT, transports=['websocket'])
        client.emit('request_keys')
        assert keys_received.wait(timeout=15), "Timeout: chiavi BFV non ricevute dalla TA"
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    data = result['data']
    assert data.get('backend') == 'bfv', "Backend atteso 'bfv', ricevuto '%s'" % data.get('backend')
    assert 'full_context' in data and 'public_context' in data, "Campi mancanti nella risposta TA"

    full_bytes = codecs.decode(data['full_context'].encode(), 'base64')
    pub_bytes  = codecs.decode(data['public_context'].encode(), 'base64')
    full_ctx = ts.context_from(full_bytes)
    pub_ctx  = ts.context_from(pub_bytes)

    assert full_ctx.is_private(), "Il full context dovrebbe avere la secret key"
    assert not pub_ctx.is_private(), "Il public context NON dovrebbe avere la secret key"
    print("  TA avviata, chiavi BFV ricevute, full_ctx privato, pub_ctx pubblico  OK")

    try:
        sc = sio_lib.Client(reconnection=False)
        sc.connect('http://127.0.0.1:%d' % TA_PORT, transports=['websocket'])
        sc.emit('shutdown_ta')
        time.sleep(1)
        sc.disconnect()
    except Exception:
        pass

    print("[TEST 6] PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    passed = 0
    failed = 0
    for test_fn in [test_roundtrip, test_fedavg_aggregation, test_ta_ckks,
                    test_bfv_roundtrip, test_bfv_fedavg_aggregation, test_ta_bfv]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print("\n[FAILED] %s: %s" % (test_fn.__name__, e))
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "=" * 40)
    print("Risultato: %d passed, %d failed" % (passed, failed))
    sys.exit(0 if failed == 0 else 1)
