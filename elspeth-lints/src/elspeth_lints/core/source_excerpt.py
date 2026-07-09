"""Outbound-content sanitiser for the judge prompt's source-code excerpt.

This module is the single trust boundary between local source bytes and
the OpenRouter LLM call. Two security gates:

1. **Path containment.** The attacker-controllable input is the
   ``file_path`` segment of an allowlist entry key (any user with
   write access to the allowlist YAML can forge one). A forged
   ``../../../etc/passwd`` key would otherwise resolve to the host
   filesystem and ship ±15 lines of arbitrary content to a third-party
   LLM. ``resolve_safe_excerpt_path`` rejects anything that resolves
   outside the operator-supplied ``--root``.

2. **Secrets scrubbing.** Even a legitimate path may contain inline
   secrets (hardcoded API keys, JWT signatures, PEM blocks, ``# DO NOT
   COMMIT`` strings). ``scrub_secrets`` redacts a curated pattern set
   and emits a structured ``RedactionRecord`` for each match so the
   audit trail captures *that redaction occurred* — per the project's
   auditability standard, an unrecorded transformation is fraud.

Why a single helper module: every code path that builds the judge prompt
MUST go through this scrubber surface. Filesystem-derived excerpts use
``extract_safe_excerpt`` for path containment + scrubbing. Already-inline
labelled corpus snippets use ``scrub_secrets`` directly because there is
no filesystem path to contain, but the outbound prompt text still must
be secret-redacted before it reaches OpenRouter.

Failure shape:

* Path containment fails -> ``SourceExcerptPathOutsideRootError``,
  subclass of ``ValueError``. Deliberately NOT a ``RuntimeError`` so
  ``reaudit._reaudit_one_entry``'s T6b transport-failure net does not
  conflate a security signal (forged path) with a transient network
  blip. ``cli._run_justify`` exits non-zero immediately; reaudit
  classifies the entry as ``SOURCE_EXCERPT_REJECTED`` and continues.

* Path resolves but file missing -> ``FileNotFoundError`` (raised by
  ``Path.resolve(strict=True)``). The honest Tier-1 behaviour for "the
  allowlist references a file that no longer exists" is the same as
  ``ENTRY_OBSOLETE`` at the caller's discretion.

* Secrets present -> the excerpt text is redacted in-place; the call
  proceeds with a sanitised excerpt and a populated
  ``redactions`` tuple. The scrubber never raises on a hit; redaction
  is preferable to fail-closed because a fail-closed sweep on a single
  planted PEM block would deny the operator visibility into the
  surrounding (legitimate) suppression debt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


class SourceExcerptPathOutsideRootError(ValueError):
    """The resolved source-excerpt path escapes the operator-supplied root.

    Raised by ``resolve_safe_excerpt_path`` when the
    attacker-controllable ``file_path`` segment of an allowlist entry
    key resolves (after symlink resolution via ``Path.resolve``) to a
    location that is not relative to the project root. This is a
    security signal: the only way to reach this branch is a forged or
    tampered allowlist entry. The message carries both the resolved
    target and the resolved root so the operator can correlate the
    attacker's intent against the allowlist diff.

    Subclasses ``ValueError`` (not ``RuntimeError``) so reaudit's
    transport-failure handler — which catches ``RuntimeError`` and
    classifies the entry as ``JUDGE_CALL_FAILED`` (operator reads as
    "rerun later") — does NOT swallow this signal. The reaudit driver
    catches this exception explicitly and emits
    ``ReauditDivergence.SOURCE_EXCERPT_REJECTED`` instead, which the
    operator reads as "investigate the YAML for tampering".
    """


@dataclass(frozen=True, slots=True)
class RedactionRecord:
    """One scrubber-applied redaction within a source-excerpt window.

    ``pattern_name`` is the human-readable identifier of the regex that
    matched (e.g. ``"aws_access_key"``, ``"pem_private_key"``). It is
    the audit-record vocabulary; operators sort/filter the report by
    pattern name. Mirrors the project's "record what we got" stance —
    if the scrubber matched, we record what kind of match it was.

    ``byte_count`` is the length of the redacted substring in bytes
    (UTF-8). When ``_Pattern.redact_group`` is 0 this is the full
    matched substring; when a capture-group is configured (e.g. the
    ``authorization_bearer`` token-only redaction) the byte count
    measures only the redacted token, not the surrounding label. The
    operator can correlate the deficit against the original excerpt's
    size to estimate how much of the window was redacted.

    ``redacted_hash`` is the first 16 hex chars of SHA-256 of the
    redacted substring, salted with the file's SHA-256 fingerprint when
    the redaction was performed via ``extract_safe_excerpt``. Sufficient
    for audit replay correlation (an operator with access to the
    unredacted source can re-derive the salted hash and confirm what
    was redacted) without persisting the secret itself. Per the
    project's hash-survives-payload-deletion principle: the hash is the
    durable handle to a value the audit trail deliberately does NOT
    carry.

    Direct callers of ``scrub_secrets`` who pass no salt produce an
    unsalted hash — preserved for the grep-equality cross-excerpt
    correlation use case (same secret across two files redacted in
    isolation produces the same token). The two regimes are kept
    distinct: the salted regime is the production path through
    ``extract_safe_excerpt`` and the only one that reaches the audit
    YAML/JSONL; the unsalted regime supports diagnostic use only.

    Threat model (operator-visible, T8c-corrected):
        The salt scopes brute-force cost to a per-file dimension rather
        than across the whole audit corpus: an attacker who exfiltrates
        the YAML/JSONL audit trail and tries to recover redacted values
        with a precomputed rainbow table over common low-entropy
        secrets (``hunter2``, ``admin1234``, ``Password!``) is forced
        to recompute the table per ``file_fingerprint`` rather than
        once across all files. That is the property the salt actually
        delivers.

        It does NOT defeat per-file targeted brute force. The
        ``file_fingerprint`` is itself stored in the audit trail
        (intentionally — the C8-3 binding requires it). An attacker who
        has both the ``redacted_hash`` AND the ``file_fingerprint`` can
        try candidate strings + this file's fingerprint in seconds.
        For low-entropy secrets recovered through this channel the salt
        provides no resistance.

        Mitigating per-file targeted brute force requires a SECRET salt
        — an HMAC key held outside the audit trail (env var, KMS,
        hardware token) and combined with the file fingerprint to
        produce the per-record salt. That is a separate design
        decision (key management, rotation, recovery semantics) and is
        explicitly out of scope here. The current scheme is honest
        about defeating cross-corpus rainbow tables only.
    """

    pattern_name: str
    byte_count: int
    redacted_hash: str


@dataclass(frozen=True, slots=True)
class SafeExcerpt:
    """The output of ``extract_safe_excerpt``: scrubbed text + audit record.

    ``text`` is the redaction-applied excerpt; this is the string that
    enters ``JudgeRequest.surrounding_code`` and therefore the
    OpenRouter prompt. Callers MUST NOT re-read the file or otherwise
    reconstruct the unredacted bytes downstream of this dataclass —
    the whole purpose of the boundary is that the unredacted bytes
    never escape this module.

    ``redactions`` is the sequence of redactions applied to produce
    ``text`` (empty tuple when the excerpt was clean). Caller persists
    this in its own audit record: ``cli._run_justify`` writes a
    ``judge_excerpt_redactions:`` block into the YAML entry;
    ``reaudit._reaudit_one_entry`` stores it on the
    ``ReauditOutcome`` and the sidecar writer serialises it into the
    JSONL trail.

    ``file_fingerprint`` is the SHA-256 hex digest of the source file's
    bytes. The judge-write path in ``cli._run_justify`` compares it to the
    scanner finding's file digest before calling the judge, proving the
    scanner finding and prompt excerpt came from the same source snapshot. It
    is also the salt the scrubber used for ``RedactionRecord.redacted_hash``.
    """

    text: str
    redactions: tuple[RedactionRecord, ...]
    file_fingerprint: str


# =========================================================================
# Path containment
# =========================================================================


def resolve_safe_excerpt_path(*, root: Path, target_file: Path) -> Path:
    """Resolve ``target_file`` and prove it lives inside ``root``.

    Both inputs are resolved via ``Path.resolve(strict=True)``, which:

    * makes the path absolute,
    * canonicalises any ``..`` segments,
    * follows symlinks,
    * raises ``FileNotFoundError`` if the path does not exist.

    The strict mode is load-bearing: a non-existent path that *would*
    resolve outside root if it existed should still be rejected here,
    but a non-existent path inside root is a legitimate
    ``ENTRY_OBSOLETE`` signal — letting ``FileNotFoundError`` propagate
    lets the caller distinguish "file missing" (operational) from
    "path escapes root" (security).

    After resolution, ``Path.is_relative_to`` is the canonical
    containment check. It uses the resolved absolute paths, so symlink
    games can't fool it.

    Raises:
        SourceExcerptPathOutsideRootError: ``target_file`` resolves
            outside ``root``. This is the security branch; the message
            includes both resolved paths.
        FileNotFoundError: ``target_file`` (or ``root``) does not
            exist. The operational branch.
    """
    resolved_root = root.resolve(strict=True)
    resolved_target = target_file.resolve(strict=True)
    if not resolved_target.is_relative_to(resolved_root):
        raise SourceExcerptPathOutsideRootError(
            f"Source-excerpt target {resolved_target!s} resolves outside "
            f"the project root {resolved_root!s}. The only way to reach "
            "this branch is a forged or tampered allowlist entry key — "
            "refusing to read the file or call the judge. Inspect the "
            "allowlist YAML for the offending entry and treat as a "
            "potential exfiltration attempt."
        )
    return resolved_target


# =========================================================================
# Secrets scrubbing
# =========================================================================
#
# Pattern set is curated, not exhaustive. Each entry has a unique
# ``pattern_name`` that becomes the audit vocabulary. Patterns are
# ordered most-specific-first because ``scrub_secrets`` walks them in
# order and applies redactions cumulatively — a generic catch-all
# fires only on text the specific patterns missed.
#
# Two pattern shapes are supported:
#
# * Whole-match redaction (``redact_group=0``, the default). The full
#   regex match is replaced with the ``[REDACTED-SECRET-<hash>]``
#   token. Used for patterns where the match itself is the secret
#   (Stripe ``sk_live_…``, AWS ``AKIA…``, OAuth-style tokens).
#
# * Group-bounded redaction (``redact_group=N``). Only the Nth capture
#   group is replaced; surrounding text is preserved. Used for HTTP-
#   header / URL-suffix shapes where a fixed prefix carries no
#   sensitive content but the trailing token does — currently only
#   ``Authorization: Bearer <token>``. The byte_count and redacted_hash
#   measure ONLY the group, not the whole match.
#
# Whole-match redactions normally collapse the match to one token.
# Patterns that may span source lines set ``preserve_line_count=True`` so
# ``extract_safe_excerpt`` can still render faithful line-number prefixes
# after scrubbing.
#
# A separate ``path_hint_required`` field gates patterns that are too
# permissive to run unconditionally but unambiguous given filesystem
# context — currently the bare-base64 SSH-key body, which without the
# ``.pem`` / ``.key`` / ``id_rsa`` filename hint would false-positive
# on every long base64 chunk (sourcemaps, embedded crypto material,
# minified JS). The hint is a substring match against the lowercased
# file path supplied at ``extract_safe_excerpt`` time; if no hint is
# supplied (direct ``scrub_secrets`` callers) these patterns are
# skipped.
#
# The 32+ char generic-key pattern is proximity-gated to label words
# (``key``, ``token``, ``secret``, ``password``, ``passwd``, ``auth``,
# ``bearer``, ``api_key``, ``access_key``, ``private_key``) so it does
# not flag every SHA-256 hex constant in the codebase (test fixtures,
# rfc8785 vectors, hashlib output) and lose the operator's trust in
# the scrubber. Without that gate the legitimate-suppression workflow
# would drown in false positives and operators would learn to ignore
# the scrubber report — which is exactly the failure mode "audit by
# habit" produces. Note that the simpler ``dotenv_secret_assignment``
# pattern (label = quoted-value) can FP on prose-like values: a line
# ``secret = "this is a quite long description"`` matches because the
# value floor is 8 chars and the regex does not constrain
# entropy. This is the conscious trade-off: under-redaction is the
# security failure, over-redaction is the LLM-context failure. The
# scrubber biases toward over-redaction inside label-proximity zones
# and toward under-redaction outside them.


@dataclass(frozen=True, slots=True)
class _Pattern:
    name: str
    regex: re.Pattern[str]
    # When non-zero, the regex's Nth capture group is the only segment
    # redacted; the rest of the match passes through verbatim. The
    # ``authorization_bearer`` pattern uses this to keep the ``Authorization:
    # Bearer`` label visible in the prompt while redacting just the
    # token — operators reading the LLM transcript see the structural
    # shape of what was carried without seeing the credential.
    redact_group: int = 0
    # Pattern is run only when ANY substring in this tuple appears
    # (case-insensitively) in the path-hint string supplied to
    # ``scrub_secrets``. Empty tuple = run unconditionally. Used for
    # the ``ssh_private_key_body`` pattern, which is too permissive to
    # run on arbitrary source but unambiguous on filesystem paths
    # matching ``*.pem`` / ``*.key`` / ``id_rsa`` / ``id_ed25519``.
    path_hint_required: tuple[str, ...] = ()
    # Preserve the number of ``\n`` delimiters in a whole-match redaction.
    # Used for multi-line PEM blocks so source-excerpt rendering keeps a
    # stable one-rendered-line-per-source-line contract.
    preserve_line_count: bool = False


_PEM_PRIVATE_KEY_BLOCK_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----")
_SPLITLINES_BOUNDARY_RE = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")


_SECRET_PATTERNS: tuple[_Pattern, ...] = (
    # PEM-wrapped private keys. A complete block must redact as a single
    # audit event before the standalone delimiter patterns fire; otherwise
    # ordinary body lines embedded in source survive because the generic
    # high-entropy matcher is label-proximity gated. The replacement
    # preserves newline count so ``extract_safe_excerpt`` can keep source
    # line numbers faithful after the scrub.
    _Pattern(
        name="pem_private_key_block",
        regex=_PEM_PRIVATE_KEY_BLOCK_RE,
        preserve_line_count=True,
    ),
    # Standalone delimiter fallbacks for truncated excerpts whose window
    # contains only one end of the PEM block.
    _Pattern(
        name="pem_private_key_header",
        regex=re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ),
    _Pattern(
        name="pem_private_key_footer",
        regex=re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    ),
    # AWS access key ids. Six legitimate prefixes per AWS documentation
    # (AKIA = user access key, ASIA = STS, AIDA/AGPA/ANPA/ANVA/AROA =
    # account/user/group/role principals). 16 base32-uppercase chars
    # follow.
    _Pattern(
        name="aws_access_key",
        regex=re.compile(r"\b(?:AKIA|ASIA|AIDA|AGPA|ANPA|ANVA|AROA)[0-9A-Z]{16}\b"),
    ),
    # GitHub personal access tokens (classic + fine-grained).
    _Pattern(
        name="github_pat_classic",
        regex=re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    ),
    _Pattern(
        name="github_pat_fine_grained",
        regex=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    ),
    # GitLab personal access tokens and OAuth application secrets.
    # Format reference: GitLab docs — ``glpat-`` user PATs and
    # ``gloas-`` OAuth application secrets carry ≥20 url-safe chars.
    _Pattern(
        name="gitlab_pat",
        regex=re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    ),
    _Pattern(
        name="gitlab_oauth_secret",
        regex=re.compile(r"\bgloas-[A-Za-z0-9_-]{20,}\b"),
    ),
    # Slack tokens (bot / user / app / etc.).
    _Pattern(
        name="slack_token",
        regex=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+"),
    ),
    # Stripe API keys. Six prefix variants per Stripe key-format docs:
    # ``sk_live_`` / ``sk_test_`` (secret keys), ``pk_live_`` /
    # ``pk_test_`` (publishable — still tied to the account, redact),
    # ``rk_live_`` (restricted), and ``whsec_`` (webhook signing
    # secret). Body floor is 24 chars (Stripe's documented minimum)
    # for ``sk_/pk_/rk_`` shapes and 32 for ``whsec_`` (longer because
    # webhook secrets carry HMAC-suitable entropy).
    _Pattern(
        name="stripe_secret_key",
        regex=re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b"),
    ),
    _Pattern(
        name="stripe_test_secret_key",
        regex=re.compile(r"\bsk_test_[A-Za-z0-9]{24,}\b"),
    ),
    _Pattern(
        name="stripe_publishable_key",
        regex=re.compile(r"\bpk_live_[A-Za-z0-9]{24,}\b"),
    ),
    _Pattern(
        name="stripe_test_publishable_key",
        regex=re.compile(r"\bpk_test_[A-Za-z0-9]{24,}\b"),
    ),
    _Pattern(
        name="stripe_restricted_key",
        regex=re.compile(r"\brk_live_[A-Za-z0-9]{24,}\b"),
    ),
    _Pattern(
        name="stripe_webhook_secret",
        regex=re.compile(r"\bwhsec_[A-Za-z0-9]{32,}\b"),
    ),
    # OpenAI / OpenRouter API keys. Three shapes:
    # * Legacy ``sk-<48 base62>`` (the classic API key form).
    # * Project ``sk-proj-<40+ url-safe>`` (the 2024+ project-scoped
    #   form, variable length).
    # * Session token ``sess-<40+ base62>`` (used by the dashboard).
    # ``sk-ant-`` (Anthropic) MUST match before generic ``sk-`` so the
    # vendor-specific pattern wins the pattern_name tag.
    _Pattern(
        name="anthropic_api_key",
        regex=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}\b"),
    ),
    _Pattern(
        name="openai_project_key",
        regex=re.compile(r"\bsk-proj-[A-Za-z0-9_-]{40,}\b"),
    ),
    _Pattern(
        name="openai_session_key",
        regex=re.compile(r"\bsess-[A-Za-z0-9]{40,}\b"),
    ),
    # Trailing anchor: see ``google_api_key`` below for the
    # ``(?!\w)`` vs ``\b`` rationale. For ``openai_api_key`` the body
    # alphabet ``[A-Za-z0-9]{48}`` contains only word chars so the
    # change is empirically a no-op against current input; it is
    # kept here for consistency with ``google_api_key`` and to keep
    # the two patterns aligned should a future ``openai_api_key``
    # format admit non-word chars in the body.
    _Pattern(
        name="openai_api_key",
        regex=re.compile(r"\bsk-[A-Za-z0-9]{48}(?!\w)"),
    ),
    # HuggingFace user access tokens. Format per HF token docs:
    # ``hf_`` prefix + 34+ base62 chars.
    _Pattern(
        name="huggingface_token",
        regex=re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
    ),
    # Google API keys. Format per Google Cloud docs: literal ``AIza``
    # prefix + exactly 35 url-safe chars. The fixed length is the
    # distinguishing signal (other AI*-prefixed strings exist but
    # don't hit the 39-char total).
    # Trailing anchor is ``(?!\w)`` rather than ``\b`` because the
    # body alphabet ``[A-Za-z0-9_-]{35}`` includes ``-`` (non-word).
    # If a real Google API key ends in ``-`` and is followed in source
    # by another non-word char (``,`` / ``"`` / ``)`` / end-of-string),
    # ``\b`` requires a word/non-word TRANSITION and finds none
    # (``-`` is non-word, ``,`` is non-word), so the literal slips
    # through unredacted. ``(?!\w)`` simply asserts "no word char
    # follows" and succeeds in this case. Also honest at end-of-string
    # (``\b`` requires a neighbouring word char on at least one side;
    # lookahead does not). See T8c regression test
    # ``test_scrub_secrets_google_key_with_non_word_terminator``.
    _Pattern(
        name="google_api_key",
        regex=re.compile(r"\bAIza[A-Za-z0-9_-]{35}(?!\w)"),
    ),
    # JWT shape: three base64url segments joined by '.'. The middle
    # segment carries the claims; the third carries the signature.
    _Pattern(
        name="jwt",
        regex=re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    # GCP service-account JSON marker. Whole-file shape is JSON; even
    # the inline marker is enough signal to redact the surrounding
    # line.
    _Pattern(
        name="gcp_service_account_marker",
        regex=re.compile(r'"type"\s*:\s*"service_account"'),
    ),
    # OpenSSH private-key bodies (ssh-rsa / ssh-ed25519 / ssh-dss /
    # ecdsa-sha2-* prefixes on the public-key line, but the body lines
    # are caught by the high-entropy generic below).
    _Pattern(
        name="ssh_public_key",
        regex=re.compile(r"\b(?:ssh-rsa|ssh-ed25519|ssh-dss|ecdsa-sha2-\S+)\s+[A-Za-z0-9+/=]{60,}\b"),
    ),
    # Discord and Slack incoming-webhook URLs. The URL itself IS the
    # secret — anyone with the URL can post to the channel. Discord
    # format reference: ``/api/webhooks/<channel_id>/<token>`` where
    # token is 40+ url-safe chars. Slack format: ``hooks.slack.com/
    # services/T.../B.../<token>``. NOTE (operator-visible): legitimate
    # documentation OF these URL shapes — README examples, format
    # specs cited in comments — will be redacted by these patterns. The
    # trade-off is intentional: we cannot distinguish a real webhook
    # from a docstring example by regex alone, and the cost of
    # over-redacting documentation in an LLM prompt (small loss of
    # context) is lower than the cost of leaking an active webhook.
    _Pattern(
        name="discord_webhook_url",
        regex=re.compile(r"https://discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]{40,}"),
    ),
    _Pattern(
        name="slack_webhook_url",
        regex=re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]{20,}"),
    ),
    # HTTP Authorization header carrying a Bearer token. The
    # ``Authorization:`` label and the literal ``Bearer`` keyword carry
    # no sensitive content; redacting them would obscure the structural
    # shape of what the source contains. ``redact_group=1`` confines
    # the redaction to the token body. Body charset matches RFC 6750
    # (base64 / base64url / a few delimiters used by JWT-shaped
    # bearers). Floor at 20 chars to avoid matching the placeholder
    # ``Bearer <token>`` examples in comments / docs.
    _Pattern(
        name="authorization_bearer",
        regex=re.compile(r"(?i)Authorization:\s*Bearer\s+([A-Za-z0-9._/+=-]{20,})"),
        redact_group=1,
    ),
    # Azure storage account connection-string component. The full
    # connection string is the secret; the ``AccountKey=`` segment
    # carries the credential. ``AccountKey`` is the load-bearing
    # label — other components (``DefaultEndpointsProtocol``, ``AccountName``,
    # ``EndpointSuffix``) are not sensitive on their own. We redact
    # the whole ``AccountKey=<base64>`` match so the label is visible
    # but the credential is gone. Body floor at 40 base64 chars (Azure
    # keys are 64 chars base64-decoded).
    _Pattern(
        name="azure_storage_account_key",
        regex=re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}"),
    ),
    # .env-style label = quoted-value assignments. Catches the common
    # ``SECRET=...``, ``API_KEY=...`` shapes that show up in test
    # fixtures and inline config. Case-insensitive label match, value
    # length floor of 8 to avoid flagging ``password=foo`` in unit
    # tests that demonstrably aren't secrets. KNOWN FALSE-POSITIVE:
    # prose-like values match here — e.g.
    # ``secret = "this is a quite long description"``. We accept this
    # over-redaction because the alternative (entropy-gating the
    # value, or requiring a non-space character class) is fragile and
    # the under-redaction failure mode is worse than the
    # over-redaction one.
    _Pattern(
        name="dotenv_secret_assignment",
        regex=re.compile(
            r"(?i)(?:secret|password|passwd|token|api[_-]?key|access[_-]?key|private[_-]?key|auth|bearer)\s*=\s*['\"]?[^'\"\s]{8,}",
        ),
    ),
    # Generic high-entropy 32+ char value proximate to a label word.
    # The label-proximity gate (within 32 chars on the same line) is
    # what distinguishes a real key from a SHA-256 hex constant in a
    # test fixture. We match ``label[:= ]+VALUE`` within one line; the
    # value floor is 32 chars of base64-alphabet to avoid matching
    # short hex hashes.
    _Pattern(
        name="labelled_high_entropy_value",
        regex=re.compile(
            r"(?i)(?:secret|password|passwd|token|api[_-]?key|access[_-]?key|private[_-]?key|auth|bearer)"
            r"[^\n]{0,32}?"
            r"['\"]([A-Za-z0-9+/=_\-]{32,})['\"]",
        ),
    ),
    # Bare SSH / PEM private-key body lines. ONLY runs when the
    # file-path hint indicates the source is a key file
    # (``*.pem`` / ``*.key`` / ``id_rsa`` / ``id_ed25519`` / ``id_ecdsa``).
    # Without the hint this regex would catch every long base64 chunk
    # (sourcemaps, embedded binaries, minified JS). With the hint, the
    # file is structurally a key file and every long base64 line in it
    # is part of the body — the over-redaction failure mode is bounded
    # to "we don't ship the contents of a key file to the LLM", which
    # is the correct behaviour. ``^...$`` with ``re.MULTILINE`` matches
    # a full raw line; this requires ``extract_safe_excerpt`` to scrub
    # the raw window BEFORE applying the line-number prefix renderer
    # (T8c). The earlier render-then-scrub ordering made ``^`` match
    # the marker prefix ``>>   123  `` rather than the body content,
    # silently dropping every body redaction in production while the
    # unit test (which called ``scrub_secrets`` on un-prefixed text)
    # continued to pass — see T8c regression test
    # ``test_extract_safe_excerpt_redacts_ssh_private_key_body``.
    _Pattern(
        name="ssh_private_key_body",
        regex=re.compile(r"^[A-Za-z0-9+/=]{60,}$", re.MULTILINE),
        path_hint_required=(".pem", ".key", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"),
    ),
)


def _redaction_digest(redacted_segment: str, salt_bytes: bytes) -> str:
    return hashlib.sha256(redacted_segment.encode("utf-8") + salt_bytes).hexdigest()[:16]


def _splitlines_line_number(text: str, offset: int) -> int:
    """Return the 1-based line number at ``offset`` under ``str.splitlines`` rules."""
    return sum(1 for _ in _SPLITLINES_BOUNDARY_RE.finditer(text, 0, offset)) + 1


def scrub_secrets(
    text: str,
    *,
    salt: str | None = None,
    path_hint: str | None = None,
) -> SafeExcerpt:
    """Apply the curated secret-pattern set to ``text``.

    Returns a ``SafeExcerpt`` whose ``text`` is the redaction-applied
    excerpt and whose ``redactions`` tuple captures each match in source
    order. Patterns are applied in declaration order; later patterns
    operate on already-redacted text, so a match that overlaps a prior
    redaction is silently absorbed (the redacted token does not itself
    match any pattern). The returned ``file_fingerprint`` is the salt
    string when supplied, else the empty string — direct ``scrub_secrets``
    callers that pass no salt are diagnostic-only and the
    ``SafeExcerpt.file_fingerprint`` field is correspondingly empty.

    Each match is replaced with ``[REDACTED-SECRET-<hash16>]`` where
    ``<hash16>`` is the first 16 hex chars of SHA-256 over the
    matched bytes plus the ``salt`` string (UTF-8 encoded). The salt
    scopes brute-force cost per-file-fingerprint when the production
    path through ``extract_safe_excerpt`` supplies it. Without the
    salt the hash is stable across runs, so the same secret redacted
    in two different excerpts produces the same token — operators can
    grep the unsalted-mode audit trail for repeated leakage of the
    same value. Both regimes are deterministic; the salted regime
    breaks cross-file grep equality on purpose (the security property
    the salt provides).

    ``path_hint`` is a file-path string used to gate
    ``path_hint_required`` patterns (currently ``ssh_private_key_body``).
    Hint matching is a case-insensitive substring check; supply
    ``str(target_file)`` from ``extract_safe_excerpt`` to enable. Without
    a hint, path-gated patterns are skipped (the bare ``scrub_secrets``
    surface is diagnostic-only and the gated patterns would otherwise
    false-positive on arbitrary source).

    The empty-text case returns an empty ``SafeExcerpt`` with an empty
    redactions tuple; callers can rely on the dataclass shape being
    non-None regardless of input.
    """
    records: list[RedactionRecord] = []
    out = text
    salt_bytes = salt.encode("utf-8") if salt is not None else b""
    lowered_hint = path_hint.lower() if path_hint is not None else ""
    for pattern in _SECRET_PATTERNS:
        # Skip path-gated patterns unless the supplied path hint
        # contains one of the required substrings. Empty/absent hint
        # means "diagnostic call without filesystem context" and the
        # pattern is skipped — see ``_Pattern.path_hint_required``.
        if pattern.path_hint_required and not any(hint in lowered_hint for hint in pattern.path_hint_required):
            continue

        # ``re.sub`` with a callback gives us the matched substring so
        # we can hash it and emit a record. Walking each pattern across
        # the (potentially already-redacted) accumulator means later
        # patterns can't re-match a token we already replaced — the
        # replacement token doesn't contain a base64 secret shape.
        def _replace(
            match: re.Match[str],
            _name: str = pattern.name,
            _redact_group: int = pattern.redact_group,
            _preserve_line_count: bool = pattern.preserve_line_count,
        ) -> str:
            # ``redact_group=0`` is the default: redact the whole match.
            # Otherwise we redact only the named capture group and rebuild
            # the output with the surrounding text intact.
            redacted_segment = match.group(_redact_group)
            digest = _redaction_digest(redacted_segment, salt_bytes)
            records.append(
                RedactionRecord(
                    pattern_name=_name,
                    byte_count=len(redacted_segment.encode("utf-8")),
                    redacted_hash=digest,
                )
            )
            token = f"[REDACTED-SECRET-{digest}]"
            if _redact_group == 0:
                if _preserve_line_count:
                    return "\n".join(token for _ in redacted_segment.split("\n"))
                return token
            # Group-bounded redaction: splice the token in place of the
            # group within the whole match. ``match.start/end`` give
            # absolute offsets in the underlying string; we want them
            # relative to ``match.group(0)``.
            whole = match.group(0)
            group_start = match.start(_redact_group) - match.start(0)
            group_end = match.end(_redact_group) - match.start(0)
            return whole[:group_start] + token + whole[group_end:]

        out = pattern.regex.sub(_replace, out)
    return SafeExcerpt(
        text=out,
        redactions=tuple(records),
        file_fingerprint=salt if salt is not None else "",
    )


def _redact_pem_private_key_blocks_in_window(
    *,
    full_text: str,
    window_lines: list[str],
    start_line: int,
    end_line: int,
    salt: str,
) -> tuple[str, tuple[RedactionRecord, ...]]:
    """Redact any visible line that intersects a full-file PEM private-key block.

    ``extract_safe_excerpt`` renders only a bounded window, but real PEM
    blocks can be larger than that window. A raw-window-only regex misses
    body-only windows whose BEGIN/END delimiters sit outside the excerpt.
    This helper uses full-file spans to identify private-key blocks, then
    masks the intersecting window lines while preserving the window's line
    count for later source-line rendering. The redaction record hashes and
    counts the exact visible window lines replaced, not the hidden rest of
    the full-file block.
    """
    redacted_lines = list(window_lines)
    records: list[RedactionRecord] = []
    salt_bytes = salt.encode("utf-8")
    for match in _PEM_PRIVATE_KEY_BLOCK_RE.finditer(full_text):
        block_start_line = _splitlines_line_number(full_text, match.start())
        block_end_line = _splitlines_line_number(full_text, match.end())
        if block_start_line > end_line or block_end_line < start_line:
            continue

        visible_start_line = max(block_start_line, start_line)
        visible_end_line = min(block_end_line, end_line)
        redacted_segment = "\n".join(window_lines[visible_start_line - start_line : visible_end_line - start_line + 1])
        digest = _redaction_digest(redacted_segment, salt_bytes)
        records.append(
            RedactionRecord(
                pattern_name="pem_private_key_block",
                byte_count=len(redacted_segment.encode("utf-8")),
                redacted_hash=digest,
            )
        )
        token = f"[REDACTED-SECRET-{digest}]"
        for source_line in range(visible_start_line, visible_end_line + 1):
            redacted_lines[source_line - start_line] = token

    return "\n".join(redacted_lines), tuple(records)


# =========================================================================
# Composed helper — the only surface callers should use
# =========================================================================


def extract_safe_excerpt(
    *,
    root: Path,
    target_file: Path,
    line: int,
    context_lines: int,
) -> SafeExcerpt:
    """Path-contained, scrubber-gated source excerpt for the judge prompt.

    This is the single chokepoint every code path that builds the
    judge's ``surrounding_code`` MUST funnel through. Composition:

    1. ``resolve_safe_excerpt_path`` proves ``target_file`` lives
       inside ``root`` (else raises
       ``SourceExcerptPathOutsideRootError``). Uses ``strict=True``
       resolution so a missing file raises ``FileNotFoundError``.
    2. The resolved file is read once as bytes and hashed (SHA-256)
       to produce the ``file_fingerprint``. The bytes are then decoded
       as UTF-8 for the excerpt window. The single read + single hash
       is the source of truth for the justify snapshot check and the
       scrubber's per-file hash salt.
    3. Excerpt window is ``[line - context_lines, line + context_lines]``
       clamped to valid line indices (1-based), mirroring the original
       ``_extract_surrounding_code`` shape so the judge sees the same
       ±N-line window with the same ``>>``-prefixed marker line.
    4. The RAW window text (just the body, no line-number prefix) is first
       masked for any full-file PEM private-key block that intersects the
       window. This catches header-only, footer-only, and body-only windows
       around large embedded keys. The result is then fed through
       ``scrub_secrets`` with the file fingerprint as salt and the resolved
       target's path-string as the path hint. The salt scopes brute-force
       cost per file; the hint enables the ``ssh_private_key_body`` pattern
       only on structurally-key files.
    5. The scrubbed body is then prefixed line-by-line with the
       ``>>``/``  `` + line-number marker. Scrubber patterns that span
       newlines must preserve the newline count; ``pem_private_key_block``
       does this deliberately so line numbers retain fidelity with the
       source file.

    Why scrub BEFORE render (T8c, supersedes the earlier
    render-then-scrub ordering):

    The ``ssh_private_key_body`` pattern uses ``^[A-Za-z0-9+/=]{60,}$``
    with ``re.MULTILINE``. ``^`` matches the start of a logical line.
    Render-then-scrub put the line-number prefix ``>>   123  `` at the
    start of every line, so ``^`` matched the prefix's leading marker
    character (``>``) rather than the base64 body — the pattern
    silently dropped every body match in production. The unit test
    (which called ``scrub_secrets`` on un-prefixed text) continued to
    pass and hid the gap.

    Scrub-before-render makes the test surface
    (``scrub_secrets(raw_text)``) match the production surface (raw
    text -> scrub -> render). It also leaves
    ``RedactionRecord.byte_count`` and ``RedactionRecord.redacted_hash``
    naturally measured against the RAW secret bytes — under the old
    ordering the line-number prefix would have been excluded from
    those measurements only because no pattern matched across the
    prefix boundary, an implicit dependency we no longer rely on.

    Operator-visible consequence: SSH key files (``*.pem`` /
    ``id_rsa`` / etc.) now have their body lines redacted in the
    judge prompt. Previously they slipped through. Audit JSONL written
    before T8c may contain unredacted SSH bodies for entries justified
    against such files.
    """
    resolved = resolve_safe_excerpt_path(root=root, target_file=target_file)
    raw_bytes = resolved.read_bytes()
    file_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines()
    start = max(1, line - context_lines)
    end = min(len(lines), line + context_lines)
    # Empty-window early return (closes elspeth-9bbb9df9a5 / T8d, restores
    # pre-T8c behaviour). Two distinct shapes reach this branch:
    #
    # 1. ``len(lines) == 0`` — the source file is genuinely empty
    #    (e.g. a zero-byte ``__init__.py``). ``end = min(0, …) = 0``,
    #    ``start = max(1, …) >= 1``, so ``start > end``.
    # 2. ``line > len(lines)`` — the requested line is past current EOF.
    #    A stale allowlist entry pointing at a line number that no
    #    longer exists is the realistic trigger. ``end = len(lines)``,
    #    ``start = max(1, line - context_lines) > len(lines) = end``.
    #
    # Both cases share a single semantic: there is no window to render.
    # The honest output is an empty ``SafeExcerpt`` carrying the (still
    # meaningful) file fingerprint and no redactions — the caller can
    # then decide whether an empty excerpt is an ENTRY_OBSOLETE signal,
    # a no-op justify, or an operator-actionable refusal. We deliberately
    # do NOT route this through ``scrub_secrets("")`` and the
    # downstream invariant: the invariant compares the rendered line
    # count (``"".split("\n") == [""]``, length 1) to the expected
    # window line count (0 here), would misfire, and would raise a
    # misleading "scrubber altered newline count" RuntimeError. The
    # invariant is intended to surface a real cross-newline pattern
    # match — an empty window is not that condition.
    if start > end:
        return SafeExcerpt(
            text="",
            redactions=(),
            file_fingerprint=file_fingerprint,
        )
    window_lines = lines[start - 1 : end]
    raw_window, pem_block_redactions = _redact_pem_private_key_blocks_in_window(
        full_text=text,
        window_lines=window_lines,
        start_line=start,
        end_line=end,
        salt=file_fingerprint,
    )
    scrubbed = scrub_secrets(raw_window, salt=file_fingerprint, path_hint=str(resolved))
    scrubbed_lines = scrubbed.text.split("\n")
    # ``split("\n")`` produces ``end - start + 1`` segments when the
    # raw window had that many lines. Multi-line patterns must preserve
    # the newline count; this assertion surfaces drift loudly, per
    # Tier-1 doctrine.
    expected_lines = end - start + 1
    if len(scrubbed_lines) != expected_lines:
        raise RuntimeError(
            f"extract_safe_excerpt: scrubber altered newline count "
            f"(raw window had {expected_lines} lines, scrubbed has "
            f"{len(scrubbed_lines)}); a secret pattern must have "
            f"changed the newline count. Refusing to render a line-number "
            f"prefix against a mismatched body — file={resolved}."
        )
    rendered_lines: list[str] = []
    for offset, body_line in enumerate(scrubbed_lines):
        line_num = start + offset
        marker = ">>" if line_num == line else "  "
        rendered_lines.append(f"{marker} {line_num:5d}  {body_line}")
    rendered = "\n".join(rendered_lines)
    return SafeExcerpt(
        text=rendered,
        redactions=pem_block_redactions + scrubbed.redactions,
        file_fingerprint=file_fingerprint,
    )
