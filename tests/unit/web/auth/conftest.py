"""Shared fixtures for auth provider tests.

Provides RSA keypair generation, JWKS response building, and JWT
signing for both OIDC and Entra test modules.
"""

from __future__ import annotations

import json

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey


@pytest.fixture
def rsa_keypair() -> tuple[RSAPrivateKey, RSAPublicKey]:
    """Generate an RSA key pair for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


def build_rsa_jwk(public_key: RSAPublicKey, *, alg: str | None = "RS256") -> dict[str, object]:
    """Build a JWKS response dict from the test RSA public key."""
    # Use PyJWT's RSAAlgorithm to export the public key as a JWK dict
    jwk_json = pyjwt.algorithms.RSAAlgorithm.to_jwk(public_key)
    key_dict = json.loads(jwk_json)
    key_dict["kid"] = "test-key-1"
    key_dict["use"] = "sig"
    if alg is None:
        key_dict.pop("alg", None)
    else:
        key_dict["alg"] = alg
    return {"keys": [key_dict]}


@pytest.fixture
def jwks_response(rsa_keypair):
    """Build a JWKS response dict from the test RSA public key."""
    _, public_key = rsa_keypair
    return build_rsa_jwk(public_key)


def make_rsa_token(private_key, claims: dict[str, object], *, algorithm: str = "RS256") -> str:
    """Sign a JWT with an RSA private key using the requested algorithm.

    Not a fixture — a plain helper function imported explicitly by
    test modules that need it.
    """
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(
        claims,
        priv_pem.decode(),
        algorithm=algorithm,
        headers={"kid": "test-key-1"},
    )


def make_rs256_token(private_key, claims: dict[str, object]) -> str:
    """Backward-compatible helper for the common RS256 case."""
    return make_rsa_token(private_key, claims, algorithm="RS256")
