"""Skill packs for the LLM pipeline composer.

Skill packs are markdown files loaded into the system prompt to teach
the LLM how to use the composition tools effectively.

Two layers:
- **Core skills** ship with the package (this directory).
- **Deployment skills** live in the data directory (``data/skills/``)
  and are optional.  They let operators inject company-specific
  knowledge (provider mappings, custom patterns, domain vocabulary)
  without editing the core skill pack.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent


def load_skill(name: str) -> str:
    """Load a core skill pack by name (without extension).

    Args:
        name: Skill filename without .md extension (e.g. 'pipeline_composer').

    Returns:
        The skill content as a string.

    Raises:
        FileNotFoundError: If the skill file does not exist.
    """
    path = _SKILLS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def load_skill_with_hash(name: str) -> tuple[str, str]:
    """Load a core skill pack and its SHA-256 hex digest atomically (F-5a).

    Returns ``(text, sha256_hex)``. Both values are derived from the same
    in-memory read; the hash and text are atomically consistent — the hash
    is computed over exactly the text the LLM sees, not over a fresh re-read
    of disk that could have changed between reads.

    The result is ``@lru_cache``'d: production wiring resolves both the
    composer prompt and the audit-row ``composer_skill_hash`` from the same
    cached tuple. To detect mid-process hot-reload of the on-disk file (a
    partial-state hazard that would cause the audit row to disagree with the
    prompt actually sent to the LLM), the compose loop re-asserts at startup
    that the on-disk SHA-256 still equals the cached hash; mismatch raises
    an operator-actionable error rather than silently shipping stale text.

    Spec reference: docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md
    "Skill hash atomicity (F-5a)" (~lines 2374-2395).
    """
    path = _SKILLS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    sha256_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, sha256_hex


def assert_skill_hash_unchanged_on_disk(name: str, expected_sha256: str) -> None:
    """Re-read the skill file from disk and assert its SHA-256 still matches.

    F-5a partial-state guard: detects the case where the in-memory cached
    text was loaded before a hot-reload changed the on-disk file. If a
    mismatch is found, raise ``RuntimeError`` with an operator-actionable
    message — the audit row's ``composer_skill_hash`` would no longer
    correspond to the file currently on disk, breaking the cross-version
    diff path. Callers (the compose loop init) should crash; the operator
    restarts the service to reload the LRU cache.
    """
    path = _SKILLS_DIR / f"{name}.md"
    on_disk_text = path.read_text(encoding="utf-8")
    on_disk_hash = hashlib.sha256(on_disk_text.encode("utf-8")).hexdigest()
    if on_disk_hash != expected_sha256:
        raise RuntimeError(
            f"Composer skill hash mismatch for {name!r}: in-memory cached hash "
            f"({expected_sha256}) differs from on-disk file hash ({on_disk_hash}). "
            f"The on-disk skill markdown was modified after the LRU cache populated; "
            f"the LLM is being prompted with the cached (older) text. Restart "
            f"elspeth-web.service to reload the skill cache."
        )


# Deployment skills beyond this size are rejected to prevent context
# window exhaustion.  The core skill is ~24KB; 64KB allows generous
# deployment content without risking prompt bloat.
MAX_DEPLOYMENT_SKILL_BYTES = 64 * 1024


def load_deployment_skill(name: str, data_dir: str | Path | None = None) -> str:
    """Load an optional deployment-specific skill overlay.

    Looks for ``{data_dir}/skills/{name}.md``.  Returns an empty string
    if the file does not exist or *data_dir* is ``None``.

    Raises ``ValueError`` if the file exceeds ``MAX_DEPLOYMENT_SKILL_BYTES``
    — this prevents accidental context window exhaustion from oversized
    deployment skills.

    Args:
        name: Skill filename without .md extension.
        data_dir: Root data directory (e.g. ``data/``).  When ``None``
            the function returns ``""`` immediately.

    Returns:
        The deployment skill content, or ``""`` if absent.

    Raises:
        ValueError: If the deployment skill file exceeds the size limit.
        OSError: If the file exists but cannot be read (e.g.
            ``PermissionError``, ``IsADirectoryError``).  These indicate
            operator misconfiguration and must not be silenced.
    """
    if data_dir is None:
        return ""
    path = Path(data_dir) / "skills" / f"{name}.md"
    # Bounded read: cap at MAX+1 bytes so oversized files never materialize
    # fully in memory. Decoding only happens after the size check passes,
    # and the raw bytes are reused for the final decode so we never encode
    # the content a second time.
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_DEPLOYMENT_SKILL_BYTES + 1)
    except FileNotFoundError:
        # File does not exist — no deployment skill configured.
        return ""
    if len(raw) > MAX_DEPLOYMENT_SKILL_BYTES:
        raise ValueError(
            f"Deployment skill at {path} exceeds the {MAX_DEPLOYMENT_SKILL_BYTES} byte limit. "
            f"Reduce the file size or increase MAX_DEPLOYMENT_SKILL_BYTES."
        )
    return raw.decode("utf-8")
