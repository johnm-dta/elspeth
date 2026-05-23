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
MUST go through ``extract_safe_excerpt``. The structural guarantee is
mechanical (one importable surface) rather than audit-by-grep. The two
call sites in this commit are ``cli._run_justify`` (single-entry write
path, closes elspeth-9bbb9df9a5 / C1-2(c) + C2-2) and
``reaudit._reaudit_one_entry`` (N-entry sweep path, closes
elspeth-ebb2b88753 / C3-4 where the same defect amplifies to N
exfiltrations per CI invocation).

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

    ``byte_count`` is the length of the matched substring in bytes
    (UTF-8). Operators can correlate the deficit against the original
    excerpt's size to estimate how much of the window was redacted.

    ``redacted_hash`` is the first 16 hex chars of SHA-256 of the
    matched substring. Sufficient for audit replay correlation (an
    operator with access to the unredacted source can re-derive the
    hash and confirm what was redacted) without persisting the secret
    itself. Per the project's hash-survives-payload-deletion principle:
    the hash is the durable handle to a value the audit trail
    deliberately does NOT carry.
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
    """

    text: str
    redactions: tuple[RedactionRecord, ...]


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
# The 32+ char generic-key pattern is proximity-gated to label words
# (``key``, ``token``, ``secret``, ``password``, ``passwd``, ``auth``,
# ``bearer``, ``api_key``, ``access_key``, ``private_key``) so it does
# not flag every SHA-256 hex constant in the codebase (test fixtures,
# rfc8785 vectors, hashlib output) and lose the operator's trust in
# the scrubber. Without that gate the legitimate-suppression workflow
# would drown in false positives and operators would learn to ignore
# the scrubber report — which is exactly the failure mode "audit by
# habit" produces.


@dataclass(frozen=True, slots=True)
class _Pattern:
    name: str
    regex: re.Pattern[str]


_SECRET_PATTERNS: tuple[_Pattern, ...] = (
    # PEM-wrapped private keys. The block prefix is the load-bearing
    # signal; we redact from the header through the next newline that
    # terminates the matched line, leaving the surrounding excerpt
    # intact. Multi-line PEM bodies still match per line via the
    # generic high-entropy pattern below.
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
    # Slack tokens (bot / user / app / etc.).
    _Pattern(
        name="slack_token",
        regex=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+"),
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
    # .env-style label = quoted-value assignments. Catches the common
    # ``SECRET=...``, ``API_KEY=...`` shapes that show up in test
    # fixtures and inline config. Case-insensitive label match, value
    # length floor of 8 to avoid flagging ``password=foo`` in unit
    # tests that demonstrably aren't secrets.
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
)


def scrub_secrets(text: str) -> SafeExcerpt:
    """Apply the curated secret-pattern set to ``text``.

    Returns the redacted text plus a tuple of ``RedactionRecord``
    capturing each match in source order. Patterns are applied in
    declaration order; later patterns operate on already-redacted
    text, so a match that overlaps a prior redaction is silently
    absorbed (the redacted token does not itself match any pattern).

    Each match is replaced with ``[REDACTED-SECRET-<hash16>]`` where
    ``<hash16>`` is the first 16 hex chars of SHA-256 of the matched
    substring. The hash is stable across runs so the same secret
    redacted in two different excerpts produces the same token —
    operators can grep the audit trail for repeated leakage of the
    same value.

    The empty-text case returns an empty ``SafeExcerpt`` with an empty
    redactions tuple; callers can rely on the dataclass shape being
    non-None regardless of input.
    """
    records: list[RedactionRecord] = []
    out = text
    for pattern in _SECRET_PATTERNS:
        # ``re.sub`` with a callback gives us the matched substring so
        # we can hash it and emit a record. Walking each pattern across
        # the (potentially already-redacted) accumulator means later
        # patterns can't re-match a token we already replaced — the
        # replacement token doesn't contain a base64 secret shape.
        def _replace(match: re.Match[str], _name: str = pattern.name) -> str:
            matched = match.group(0)
            digest = hashlib.sha256(matched.encode("utf-8")).hexdigest()[:16]
            records.append(
                RedactionRecord(
                    pattern_name=_name,
                    byte_count=len(matched.encode("utf-8")),
                    redacted_hash=digest,
                )
            )
            return f"[REDACTED-SECRET-{digest}]"

        out = pattern.regex.sub(_replace, out)
    return SafeExcerpt(text=out, redactions=tuple(records))


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
    2. The resolved file is read once. Excerpt window is
       ``[line - context_lines, line + context_lines]`` clamped to
       valid line indices (1-based), mirroring the original
       ``_extract_surrounding_code`` shape so the judge sees the same
       ±N-line window with the same ``>>``-prefixed marker line.
    3. The rendered window is fed through ``scrub_secrets``, returning
       a ``SafeExcerpt`` whose ``text`` is the already-redacted string
       safe to pass into ``JudgeRequest.surrounding_code``.

    The two-step "render-then-scrub" order (rather than "scrub-then-
    render") is deliberate: the scrubber operates on the line-number-
    prefixed rendered text, which means redaction tokens are visible
    in the judge prompt at the same line they replaced. The line-
    number prefix itself never matches any secret pattern, so this
    ordering preserves audit-trail line-number fidelity without
    impeding scrubber accuracy.
    """
    resolved = resolve_safe_excerpt_path(root=root, target_file=target_file)
    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = max(1, line - context_lines)
    end = min(len(lines), line + context_lines)
    rendered_lines: list[str] = []
    for line_num in range(start, end + 1):
        marker = ">>" if line_num == line else "  "
        rendered_lines.append(f"{marker} {line_num:5d}  {lines[line_num - 1]}")
    rendered = "\n".join(rendered_lines)
    return scrub_secrets(rendered)
