"""Optional certificate-based authentication for SignGuard.

This module provides optional X.509 certificate support for enhanced
authentication in production deployments.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


class CertificateAuthority:
    """Simple certificate authority for SignGuard."""

    def __init__(
        self,
        ca_key: Optional[ec.EllipticCurvePrivateKey] = None,
        ca_cert: Optional[x509.Certificate] = None,
    ):
        """Initialize CA.

        Args:
            ca_key: CA private key (generates new if None)
            ca_cert: CA certificate (generates new if None)
        """
        self.backend = default_backend()
        
        if ca_key is None or ca_cert is None:
            ca_key, ca_cert = self._generate_ca()
        
        self.ca_key = ca_key
        self.ca_cert = ca_cert

    def _generate_ca(self) -> Tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
        """Generate self-signed CA certificate."""
        # Generate CA key
        ca_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        
        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "SignGuard"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SignGuard CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "SignGuard Root CA"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))  # 10 years
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ), critical=True)
            .sign(ca_key, hashes.SHA256(), self.backend)
        )
        
        return ca_key, cert

    def issue_client_certificate(
        self,
        client_id: str,
        client_public_key: ec.EllipticCurvePublicKey,
        validity_days: int = 365,
    ) -> x509.Certificate:
        """Issue certificate for a client.

        Args:
            client_id: Client identifier
            client_public_key: Client's public key
            validity_days: Certificate validity period in days

        Returns:
            Client certificate
        """
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SignGuard"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"client_{client_id}"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.ca_cert.subject)
            .public_key(client_public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=False,
                crl_sign=False,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(self.ca_key, hashes.SHA256(), self.backend)
        )
        
        return cert

    def verify_certificate(self, cert: x509.Certificate) -> bool:
        """Verify a certificate against the CA.

        Args:
            cert: Certificate to verify

        Returns:
            True if certificate is valid
        """
        try:
            # Check if expired
            if datetime.utcnow() < cert.not_valid_before or datetime.utcnow() > cert.not_valid_after:
                return False
            
            # Verify signature
            cert.public_key().verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                ec.ECDSA(hashes.SHA256())
            )
            
            return True
        except Exception:
            return False

    def save_ca(self, ca_dir: Path | str) -> None:
        """Save CA certificate and key.

        Args:
            ca_dir: Directory to save CA files
        """
        ca_dir = Path(ca_dir)
        ca_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CA certificate
        cert_path = ca_dir / "ca_cert.pem"
        with open(cert_path, "wb") as f:
            f.write(self.ca_cert.public_bytes(serialization.Encoding.PEM))
        
        # Save CA key
        key_path = ca_dir / "ca_key.pem"
        with open(key_path, "wb") as f:
            f.write(self.ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

    @classmethod
    def load_ca(cls, ca_dir: Path | str) -> "CertificateAuthority":
        """Load CA from directory.

        Args:
            ca_dir: Directory containing CA files

        Returns:
            CertificateAuthority instance
        """
        ca_dir = Path(ca_dir)
        
        # Load CA certificate
        with open(ca_dir / "ca_cert.pem", "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        # Load CA key
        with open(ca_dir / "ca_key.pem", "rb") as f:
            ca_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        
        return cls(ca_key=ca_key, ca_cert=ca_cert)
