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
