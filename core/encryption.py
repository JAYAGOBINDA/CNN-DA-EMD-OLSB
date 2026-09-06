"""
Security & Encryption Module for Steganography Payload.
Implements AES-256-GCM authenticated encryption and PBKDF2HMAC key derivation.
"""

import os
from typing import Tuple
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def derive_key(password: str, salt: bytes, iterations: int = 100_000) -> bytes:
    """
    Derives a 256-bit key from user password using PBKDF2 with SHA-256.
    """
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')
    else:
        password_bytes = password

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password_bytes)


def encrypt_payload(data: bytes, password: str, associated_data: bytes = None) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypts arbitrary byte payload using AES-256-GCM.

    Args:
        data: Plaintext bytes to encrypt.
        password: User password for key derivation.
        associated_data: Optional Additional Authenticated Data (AAD).
            If provided, this data is authenticated by the GCM tag but NOT
            encrypted. Used to bind recovery side information to the ciphertext
            so that tampering with either is detected.

    Returns:
        salt (16 bytes), nonce (12 bytes), ciphertext (includes GCM authentication tag)
    """
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = derive_key(password, salt)
    
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, associated_data=associated_data)
    
    return salt, nonce, ciphertext


def decrypt_payload(ciphertext: bytes, password: str, salt: bytes, nonce: bytes,
                    associated_data: bytes = None) -> bytes:
    """
    Decrypts AES-256-GCM ciphertext. Raises InvalidTag if password is incorrect,
    data tampered, or associated_data does not match what was used during encryption.
    """
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=associated_data)
