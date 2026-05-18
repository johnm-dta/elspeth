"""Shareable-review token signing primitive.

Wraps the standard-library HMAC primitive (``hmac.new(..., hashlib.sha256)``) —
the same primitive backing ``core/landscape/exporter.py::_sign_record`` — but
NOT a reuse of the exporter's API. The exporter signs Tier-1 audit records
(dict-keyed payloads with a ``signature`` key inserted alongside the data);
this signer produces a URL-safe self-contained capability token.

The signed envelope encodes ``(version, session_id, state_id, expires_at,
nonce_hex, payload_digest, created_by_user_id)`` so verification is
schema-free: signature math alone is authoritative. ``payload_digest`` is the
content-address of the snapshot blob in the payload store — tampering with
either the token fields OR the blob is detected on resolve.

Threat-model discipline:

* ``verify`` always uses ``hmac.compare_digest`` for constant-time signature
  comparison. New discipline at the boundary; the exporter's ``.hexdigest()``
  pattern is NOT a precedent here because the exporter does not verify at
  attacker-controlled boundaries. Existing precedents for ``compare_digest``
  at boundary verifiers in this project: ``core/payload_store.py`` and
  ``web/blobs/service.py`` (both compare caller-supplied content hashes
  against stored hashes — the same pattern this signer follows).
* The 32-byte signing-key floor is enforced both at the WebSettings field
  validator AND at the signer's constructor — defense in depth. Direct unit
  construction (e.g. ``ShareTokenSigner(b"short")``) must fail fast.
* Tokens carry their own expiry; expired tokens fail verify before any
  downstream lookup runs.
* The payload version is a closed constant; unknown versions reject (no
  silent forward-compat).

Rotating the signing key invalidates EVERY outstanding token — there is no
dual-key acceptance window in v1. Documented behaviour; recovery is "re-issue
the affected links."

Layer: L3 (web application).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from uuid import UUID

# Closed enum: the v1 envelope schema. Bumping this requires landing a new
# signer version side-by-side; do NOT alter v1 verify semantics in place.
# Adjacent forward-compat concern: key rotation tooling is out of scope —
# rotating the signing key invalidates EVERY outstanding token.
_PAYLOAD_VERSION: Final[int] = 1

# HMAC-SHA256 emits a 32-byte digest. The envelope appends the raw bytes.
_SIGNATURE_BYTES: Final[int] = 32

# Existing precedents in the project for `hmac.compare_digest` at boundary
# verifiers — both compare a caller-supplied content_hash against a stored
# hash, the same pattern this signer follows:
#   * core/payload_store.py (content-hash verification on retrieve)
#   * web/blobs/service.py  (content-hash verification on share-link consume)

# Length-prefix byte count (4 BE bytes ⇒ up to 4 GiB payload). The payload
# is canonical-JSON of a small dict, so a 4-byte prefix is comfortable
# overhead while leaving the wire format self-describing.
_LENGTH_PREFIX_BYTES: Final[int] = 4

# Signing-key floor: matches WebSettings.shareable_link_signing_key validator.
# 32 bytes is HMAC-SHA256's digest (output) size — the natural entropy floor
# for a tag produced by this hash. (NB: HMAC-SHA256's block size is 64 bytes;
# the floor here is the digest size, not the block size.) Matches the
# documented operator recipe `openssl rand -base64 32`. Defense in depth.
_MIN_SIGNING_KEY_BYTES: Final[int] = 32


class InvalidToken(Exception):
    """Raised when a shareable-review token fails verification.

    The exception message is intentionally generic ("malformed", "expired",
    "signature mismatch") so a probing attacker cannot distinguish failure
    modes via the error string. Logging the specific reason at the call site
    is fine; reflecting it to the wire is not.
    """


@dataclass(frozen=True, slots=True)
class ShareTokenPayload:
    """The signed envelope of a shareable-review capability token.

    Fields are deliberately flat and serialisable (UUIDs as canonical strings,
    datetimes as ISO-8601 with timezone). The canonical JSON used for signing
    is alphabetical-key order with no whitespace — see ``to_canonical_json``.

    Field semantics:

    * ``version`` — closed enum; only ``1`` is accepted in v1.
    * ``session_id`` / ``state_id`` — composition session and state the
      reviewer can inspect via the shared route.
    * ``created_at`` — when the token was minted (wall-clock at issuance).
      Carried in the envelope rather than in the snapshot blob so two
      re-mints over an unchanged composition still produce the same blob
      digest (the blob is content-addressed; only stable content goes in).
    * ``expires_at`` — absolute UTC datetime; the signer's verify() rejects
      tokens past this point. Lifetime is stamped at issue time using
      ``WebSettings.shareable_link_lifetime_seconds``.
    * ``nonce_hex`` — opaque random hex string. Distinguishes two tokens
      minted at the same instant for the same (session, state, digest); not
      used for replay defence (the token itself is the capability).
    * ``payload_digest`` — content-address of the snapshot blob in the
      payload store. Format is ``"sha256:" + hex``. Tamper-detection: the
      blob is read by this digest on resolve.
    * ``created_by_user_id`` — opaque user id of the original signer.
      Carried in the envelope so the inspect route can surface attribution
      to the reviewer.
    """

    version: int
    session_id: UUID
    state_id: UUID
    created_at: datetime
    expires_at: datetime
    nonce_hex: str
    payload_digest: str
    created_by_user_id: str

    def to_canonical_json(self) -> bytes:
        """Return the canonical-JSON byte encoding used for signing.

        Discipline: sorted keys, ASCII-only separators with no whitespace,
        utf-8 byte encoding. Mirrors the canonical-encoding posture used
        elsewhere in the project (``rfc8785`` for L1 audit records); this
        signer keeps the encoding inline because the payload is tiny and
        the field set is closed.
        """
        d = {
            "version": self.version,
            "session_id": str(self.session_id),
            "state_id": str(self.state_id),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "nonce_hex": self.nonce_hex,
            "payload_digest": self.payload_digest,
            "created_by_user_id": self.created_by_user_id,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ShareTokenSigner:
    """HMAC-SHA256 signer/verifier for shareable-review capability tokens.

    Wire format::

        urlsafe_b64encode(
            len(payload_json).to_bytes(4, "big")  # length prefix
            + payload_json                         # canonical JSON bytes
            + hmac_signature                       # 32 raw HMAC-SHA256 bytes
        )

    The length prefix lets the decoder split payload-bytes from signature-
    bytes without parsing JSON first, which keeps signature verification as
    a pure byte-level operation (constant-time, no parser state).
    """

    def __init__(self, signing_key: bytes) -> None:
        if len(signing_key) < _MIN_SIGNING_KEY_BYTES:
            raise ValueError(f"signing_key must be at least {_MIN_SIGNING_KEY_BYTES} bytes (got {len(signing_key)})")
        self._key = signing_key

    def sign(self, payload: ShareTokenPayload) -> str:
        """Return a URL-safe base64 token encoding payload + HMAC signature."""
        body = payload.to_canonical_json()
        signature = hmac.new(self._key, body, hashlib.sha256).digest()
        blob = len(body).to_bytes(_LENGTH_PREFIX_BYTES, "big") + body + signature
        return base64.urlsafe_b64encode(blob).decode("ascii")

    def verify(self, token: str) -> ShareTokenPayload:
        """Verify the signature, decode, and return the payload.

        Raises ``InvalidToken`` on:

        * malformed base64,
        * truncated envelope (shorter than ``LENGTH_PREFIX + SIGNATURE``),
        * length-prefix / total-length mismatch,
        * signature mismatch (constant-time compared via
          ``hmac.compare_digest``),
        * JSON decode failure on the payload body,
        * payload-shape mismatch (missing keys, type errors),
        * unsupported version (forward-compat closed),
        * expired ``expires_at``.

        Verify is the only path that reads attacker-controlled bytes. The
        function deliberately raises with generic messages — see ``InvalidToken``.
        """
        try:
            blob = base64.urlsafe_b64decode(token.encode("ascii"))
        except (binascii.Error, ValueError) as exc:
            raise InvalidToken("malformed token") from exc
        if len(blob) < _LENGTH_PREFIX_BYTES + _SIGNATURE_BYTES:
            raise InvalidToken("truncated token")
        body_len = int.from_bytes(blob[:_LENGTH_PREFIX_BYTES], "big")
        if len(blob) != _LENGTH_PREFIX_BYTES + body_len + _SIGNATURE_BYTES:
            raise InvalidToken("token length mismatch")
        body = blob[_LENGTH_PREFIX_BYTES : _LENGTH_PREFIX_BYTES + body_len]
        signature = blob[_LENGTH_PREFIX_BYTES + body_len :]
        expected = hmac.new(self._key, body, hashlib.sha256).digest()
        # MUST be the module-qualified call: the unit test monkeypatches
        # `hmac.compare_digest` and would not observe a locally-imported
        # alias. See test_compare_digest_used.
        if not hmac.compare_digest(signature, expected):
            raise InvalidToken("signature mismatch")
        try:
            payload_dict = json.loads(body)
        except json.JSONDecodeError as exc:
            raise InvalidToken("payload decode failed") from exc
        try:
            payload = ShareTokenPayload(
                version=payload_dict["version"],
                session_id=UUID(payload_dict["session_id"]),
                state_id=UUID(payload_dict["state_id"]),
                created_at=datetime.fromisoformat(payload_dict["created_at"]),
                expires_at=datetime.fromisoformat(payload_dict["expires_at"]),
                nonce_hex=payload_dict["nonce_hex"],
                payload_digest=payload_dict["payload_digest"],
                created_by_user_id=payload_dict["created_by_user_id"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvalidToken("payload shape mismatch") from exc
        if payload.version != _PAYLOAD_VERSION:
            raise InvalidToken(f"unsupported version {payload.version}")
        if payload.expires_at < datetime.now(UTC):
            raise InvalidToken("token expired")
        return payload
