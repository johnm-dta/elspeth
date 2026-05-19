"""Contracts for synthesised (cache-replay) Landscape audit writes.

Used by ``elspeth.core.landscape.write_repository`` (producer of the
synthesised audit rows) and ``elspeth.web.composer.tutorial_service``
(builder of the spec list from cached tutorial YAML). Lives in
``contracts/`` per the L0 layer policy so both sites import the same
type without crossing layers in an upward direction.
"""

from __future__ import annotations

from dataclasses import dataclass

from elspeth.contracts.enums import NodeType


@dataclass(frozen=True, slots=True)
class SynthesisedNodeSpec:
    """One synthetic node to record in the Landscape ``nodes`` table.

    Carries the YAML-declared role (source / transform / sink) so the writer
    records authoritative Tier-1 topology rather than inferring role from list
    position. Plugin reuse (e.g. two ``llm`` transforms, csv source plus csv
    sink) requires one spec per occurrence; the audit row count must equal the
    YAML occurrence count.

    Fields are scalars (``NodeType`` is a ``StrEnum``) so ``frozen=True`` alone
    suffices — no ``deep_freeze`` guard required.
    """

    node_type: NodeType
    plugin_name: str
    plugin_version: str
