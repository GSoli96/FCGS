import pickle
import codecs
import logging
from typing import List, Dict, Any, Union
import numpy as np
import time

# Configure a default logger for when none is provided
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def object_to_pickle_string(obj: Any, file_path: str = None, save_to_file: bool = False) -> str:
    """
    Serializes a Python object into a base64 encoded string or saves it to a file.

    Args:
        obj (Any): The Python object to serialize.
        file_path (str, optional): The file path to save the object to. Defaults to None.
        save_to_file (bool): If True and file_path is provided, saves to a file
                             and returns the path. Otherwise, returns a string.

    Returns:
        str: A base64 encoded string of the pickled object or the file path if saved.
    """
    if save_to_file and file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        return file_path

    return codecs.encode(pickle.dumps(obj), "base64").decode()


def pickle_string_to_object(s: str) -> Any:
    """
    Deserializes a base64 encoded string or a pickle file path into a Python object.

    Note:
        This function handles two cases for backward compatibility:
        1. If 's' is a base64 encoded string.
        2. If 's' is a file path ending in '.pkl'.
        This dual functionality should be used with care.

    Args:
        s (str): The base64 encoded string or the file path to a pickle file.

    Returns:
        Any: The deserialized Python object.
    """
    if ".pkl" in s:
        log.debug("Deserializing object from file path: %s", s)
        with open(s, "rb") as f:
            return pickle.load(f)
    else:
        log.debug("Deserializing object from base64 string.")
        return pickle.loads(codecs.decode(s.encode(), "base64"))


def encrypt_weights(
        public_key: Any,
        weights: List[np.ndarray],
        encryption_mode: str = 'direct_encrypted_update',
        logger: logging.Logger = log
) -> List[Dict[str, Any]]:
    """
    Encrypts a list of NumPy array weights using a Paillier public key.

    Args:
        public_key (Any): The Paillier public key.
        weights (List[np.ndarray]): A list of model weights as NumPy arrays.
        encryption_mode (str): The encryption mode. If 'simulation', encryption is
                               skipped for debugging purposes.
        logger (logging.Logger): The logger to use for progress updates.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a weight tensor and contains its original
                              shape and a list of encrypted values.
    """
    if encryption_mode == 'simulation':
        logger.info("SIMULATION MODE: Skipping encryption (pass-through).")
        return [{'shape': w.shape, 'values': w.flatten().tolist()} for w in weights]

    encrypted_weights = []
    num_layers = len(weights)
    logger.info("Starting encryption of %d layers...", num_layers)

    for i, w_layer in enumerate(weights):
        flat_weights = w_layer.flatten()
        encrypted_values = [public_key.encrypt(float(val)) for val in flat_weights]
        encrypted_weights.append({'shape': w_layer.shape, 'values': encrypted_values})

        progress = (i + 1) / num_layers * 100
        logger.info("Encryption progress: %.2f%% (Layer %d/%d encrypted)", progress, i + 1, num_layers)
        time.sleep(0)

    logger.info("Encryption complete.")
    return encrypted_weights


def decrypt_weights(
        private_key: Any,
        encrypted_weights: List[Dict[str, Any]],
        logger: logging.Logger = log
) -> List[np.ndarray]:
    """
    Decrypts a list of Paillier-encrypted weights.

    Args:
        private_key (Any): The Paillier private key.
        encrypted_weights (List[Dict[str, Any]]): The list of encrypted weight dictionaries.
        logger (logging.Logger): The logger to use for progress updates.

    Returns:
        List[np.ndarray]: The decrypted weights as a list of NumPy arrays.
    """
    decrypted_weights = []
    num_layers = len(encrypted_weights)
    logger.info("Starting decryption of %d layers...", num_layers)

    for i, item in enumerate(encrypted_weights):
        shape = item['shape']
        encrypted_values = item['values']

        flat_weights = np.array([private_key.decrypt(val) for val in encrypted_values])
        decrypted_weights.append(flat_weights.reshape(shape))

        progress = (i + 1) / num_layers * 100
        logger.info("Decryption progress: %.2f%% (Layer %d/%d decrypted)", progress, i + 1, num_layers)

    logger.info("Decryption complete.")
    return decrypted_weights


def sum_encrypted_weights(
        weights_a: List[Dict[str, Any]],
        weights_b: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Homomorphically adds two lists of encrypted weights.

    Args:
        weights_a (List[Dict[str, Any]]): The first list of encrypted weights.
        weights_b (List[Dict[str, Any]]): The second list of encrypted weights.

    Returns:
        List[Dict[str, Any]]: The homomorphically summed encrypted weights.
    """
    summed_weights = []
    for i in range(len(weights_a)):
        sum_values = [a + b for a, b in zip(weights_a[i]['values'], weights_b[i]['values'])]
        summed_weights.append({'shape': weights_a[i]['shape'], 'values': sum_values})
    return summed_weights


def multiply_encrypted_weights_by_scalar(
        weights: List[Dict[str, Any]],
        scalar: Union[int, float]
) -> List[Dict[str, Any]]:
    """
    Homomorphically multiplies a list of encrypted weights by a plaintext scalar.

    Args:
        weights (List[Dict[str, Any]]): The list of encrypted weights.
        scalar (Union[int, float]): The plaintext scalar to multiply by.

    Returns:
        List[Dict[str, Any]]: The resulting encrypted weights.
    """
    multiplied_weights = []
    for i in range(len(weights)):
        mul_values = [w * scalar for w in weights[i]['values']]
        multiplied_weights.append({'shape': weights[i]['shape'], 'values': mul_values})
    return multiplied_weights


# --- CKKS (TenSEAL) functions ---

def encrypt_weights_ckks(
        context: Any,
        weights: List[np.ndarray],
        logger: logging.Logger = log
) -> List[Dict[str, Any]]:
    """
    Encrypts model weights using TenSEAL CKKS batched vectors.
    Each layer is split into chunks of at most poly_modulus_degree//2 values.
    """
    import tenseal as ts
    # poly_modulus_degree=8192 → 4096 slot per vettore (TenSEAL non espone il valore dal context)
    max_slots = 4096
    encrypted = []
    num_layers = len(weights)
    logger.info("Starting CKKS encryption of %d layers (max_slots=%d)...", num_layers, max_slots)
    for i, layer in enumerate(weights):
        flat = layer.flatten().tolist()
        chunks = [flat[j:j + max_slots] for j in range(0, len(flat), max_slots)]
        enc_chunks = [ts.ckks_vector(context, chunk).serialize() for chunk in chunks]
        encrypted.append({'shape': layer.shape, 'chunks': enc_chunks, 'size': len(flat)})
        logger.info("CKKS encryption: layer %d/%d done", i + 1, num_layers)
    logger.info("CKKS encryption complete.")
    return encrypted


def decrypt_weights_ckks(
        context: Any,
        encrypted_weights: List[Dict[str, Any]],
        logger: logging.Logger = log
) -> List[np.ndarray]:
    """Decrypts a list of CKKS-encrypted weight layers."""
    import tenseal as ts
    decrypted = []
    num_layers = len(encrypted_weights)
    logger.info("Starting CKKS decryption of %d layers...", num_layers)
    for i, item in enumerate(encrypted_weights):
        flat = []
        for chunk_bytes in item['chunks']:
            flat.extend(ts.ckks_vector_from(context, chunk_bytes).decrypt())
        decrypted.append(np.array(flat[:item['size']], dtype=np.float32).reshape(item['shape']))
        logger.info("CKKS decryption: layer %d/%d done", i + 1, num_layers)
    logger.info("CKKS decryption complete.")
    return decrypted


def sum_encrypted_weights_ckks(
        context: Any,
        weights_a: List[Dict[str, Any]],
        weights_b: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Homomorphically adds two lists of CKKS-encrypted weights."""
    import tenseal as ts
    result = []
    for a, b in zip(weights_a, weights_b):
        chunks = []
        for ca, cb in zip(a['chunks'], b['chunks']):
            va = ts.ckks_vector_from(context, ca)
            vb = ts.ckks_vector_from(context, cb)
            chunks.append((va + vb).serialize())
        result.append({'shape': a['shape'], 'chunks': chunks, 'size': a['size']})
    return result


def multiply_encrypted_weights_ckks_by_scalar(
        context: Any,
        weights: List[Dict[str, Any]],
        scalar: Union[int, float]
) -> List[Dict[str, Any]]:
    """Homomorphically multiplies CKKS-encrypted weights by a plaintext scalar."""
    import tenseal as ts
    result = []
    for item in weights:
        chunks = [(ts.ckks_vector_from(context, c) * scalar).serialize() for c in item['chunks']]
        result.append({'shape': item['shape'], 'chunks': chunks, 'size': item['size']})
    return result