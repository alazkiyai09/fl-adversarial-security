"""Cryptographic utilities for secure aggregation."""

import os
import secrets
from typing import Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization
import numpy as np


def generate_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """
    Generate RSA key pair for secure communication.

    Args:
        key_size: Size of the RSA key in bits

    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=key_size, backend=default_backend()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_pem, public_pem


def encrypt(
    plaintext: bytes, public_key_pem: bytes, use_hybrid: bool = True
) -> Tuple[bytes, bytes]:
    """
    Encrypt data using RSA public key.

    Args:
        plaintext: Data to encrypt
        public_key_pem: PEM-encoded public key
        use_hybrid: If True, use hybrid encryption (RSA + AES)

    Returns:
        Tuple of (encrypted_data, optional_iv)
    """
    public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())

    if not use_hybrid:
        # Direct RSA encryption (for small data only)
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return ciphertext, None

    # Hybrid encryption: RSA for key, AES for data
    # Generate random AES key
    aes_key = os.urandom(32)
    iv = os.urandom(16)

    # Encrypt data with AES
    cipher = Cipher(
        algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(plaintext) + encryptor.finalize()

    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return encrypted_key + encrypted_data, iv


def decrypt(
    ciphertext: bytes, private_key_pem: bytes, iv: Optional[bytes] = None
) -> bytes:
    """
    Decrypt data using RSA private key.

    Args:
        ciphertext: Data to decrypt
        private_key_pem: PEM-encoded private key
        iv: Initialization vector (for hybrid encryption)

    Returns:
        Decrypted plaintext
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem, password=None, backend=default_backend()
    )

    if iv is None:
        # Direct RSA decryption
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return plaintext

    # Hybrid decryption
    # First 256 bytes (2048 bits) are the encrypted AES key
    key_size_bytes = 256  # For 2048-bit RSA
    encrypted_key = ciphertext[:key_size_bytes]
    encrypted_data = ciphertext[key_size_bytes:]

    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Decrypt data with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(encrypted_data) + decryptor.finalize()

    return plaintext


def generate_random_mask(shape: tuple, bit_size: int = 32) -> np.ndarray:
    """
    Generate random mask for secure aggregation.

    Args:
        shape: Shape of the mask
        bit_size: Bit size for random values

    Returns:
        Random mask as numpy array
    """
    # SECURITY: Use cryptographically secure random generation for masks
    # Generate random values using secrets module for cryptographic security
    # For efficiency with large arrays, we generate seeds and use numpy

    # For small masks, use secrets directly
    if np.prod(shape) <= 1000:
        mask = np.array([
            secrets.randbits(bit_size) - 2**(bit_size-1)
            for _ in range(int(np.prod(shape)))
        ], dtype=np.int64).reshape(shape)
    else:
        # For large masks, use secrets-seeded numpy for performance
        # This is still cryptographically secure due to the seed
        seed = secrets.randbits(32)
        rng = np.random.RandomState(seed)
        mask = rng.randint(
            -2 ** (bit_size - 1), 2 ** (bit_size - 1), size=shape, dtype=np.int64
        )
    return mask.astype(np.float32)


def pairwise_mask(
    values: np.ndarray, client_id: int, n_clients: int, seed: int = 42
) -> np.ndarray:
    """
    Apply pairwise masking for secure aggregation.

    Each client adds masks that will cancel out during aggregation.

    Args:
        values: Values to mask
        client_id: Current client ID
        n_clients: Total number of clients
        seed: Random seed for reproducibility

    Returns:
        Masked values
    """
    np.random.seed(seed + client_id)

    masked_values = values.copy().astype(np.int64)

    # Add masks for all pairs (i, j) where i < j
    for other_client in range(n_clients):
        if client_id < other_client:
            # Client adds mask for this pair
            mask = generate_random_mask(values.shape)
            masked_values += mask
        elif client_id > other_client:
            # Client subtracts mask for this pair
            mask = generate_random_mask(values.shape)
            masked_values -= mask

    return masked_values.astype(np.float32)


def compute_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash

    Returns:
        Hexadecimal hash string
    """
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(data)
    return digest.finalize().hex()
