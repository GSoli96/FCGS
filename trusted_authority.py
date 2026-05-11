import codecs
import ctypes
import logging
import pickle
import threading
import time
from threading import Thread
from flask import Flask, request
from flask_socketio import SocketIO, emit
from phe import paillier
from utils import object_to_pickle_string

class TrustedAuthority:
    """
    Implements a Trusted Authority (TA) for distributing cryptographic keys.

    This server generates a single Paillier keypair upon startup and distributes
    it to any client that connects and requests it. It's designed to be run
    alongside a federated learning server to manage key distribution securely.
    """

    def __init__(self, host: str, port: int, key_length: int = 128, he_backend: str = 'paillier'):
        self.host = host
        self.port = port
        self.he_backend = he_backend
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.logger = self._setup_logger()

        if he_backend == 'ckks':
            self.logger.info("Generating TenSEAL CKKS context (poly_modulus_degree=8192)...")
            import tenseal as ts
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            context.generate_galois_keys()
            context.global_scale = 2 ** 40
            self.ckks_full_context_bytes = context.serialize(save_secret_key=True)
            context.make_context_public()
            self.ckks_public_context_bytes = context.serialize(save_secret_key=False)
            self.logger.info("CKKS context generated successfully.")
        else:
            self.logger.info("Generating Paillier keypair with n_length=%d...", key_length)
            self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=key_length)
            self.pickled_pubkey = object_to_pickle_string(self.pubkey)
            self.pickled_privkey = object_to_pickle_string(self.privkey)
            self.logger.info("Keypair generated successfully.")

        self._register_handlers()

    def _setup_logger(self) -> logging.Logger:
        """Configures and returns a logger for the TA."""
        logger = logging.getLogger(f"TrustedAuthority-{self.port}")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _register_handlers(self) -> None:
        """Registers SocketIO event handlers."""
        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('request_keys')(self._on_request_keys)
        self.socketio.on('shutdown_ta')(self._on_shutdown_ta)

    def run(self) -> None:
        """Starts the Flask-SocketIO server for the TA."""
        self.logger.info("Trusted Authority starting on http://%s:%d", self.host, self.port)
        self._run_thread_ident = threading.current_thread().ident
        self.socketio.run(self.app, host=self.host, port=self.port)

    def _on_connect(self):
        self.logger.info("Client connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Client disconnected: %s", request.sid)

    def _on_shutdown_ta(self):
        """Handles a shutdown request from the orchestrator."""
        self.logger.info("Shutdown request received. Stopping the Trusted Authority server.")
        thread_ident = getattr(self, '_run_thread_ident', None)
        def _do_stop():
            time.sleep(0.5)
            if thread_ident:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(thread_ident),
                    ctypes.py_object(SystemExit)
                )
        Thread(target=_do_stop, daemon=True).start()

    def _on_request_keys(self):
        """Handles a client's request for keys and sends them back."""
        self.logger.info("Received key request from client %s. Distributing keys.", request.sid)
        if self.he_backend == 'ckks':
            emit('distribute_keys', {
                'backend': 'ckks',
                'full_context': codecs.encode(self.ckks_full_context_bytes, 'base64').decode(),
                'public_context': codecs.encode(self.ckks_public_context_bytes, 'base64').decode(),
            })
        else:
            emit('distribute_keys', {
                'backend': 'paillier',
                'pubkey': self.pickled_pubkey,
                'privkey': self.pickled_privkey,
            })

if __name__ == '__main__':
    # Esempio di come avviare la TA stand-alone per test
    ta = TrustedAuthority(host='127.0.0.1', port=5001)
    ta.run()