"""Expected boundary partition for the audit-readiness panel.

ELSPETH's Three-Tier Trust Model (CLAUDE.md) treats external data as
crossing Tier-3 at sources, at sinks, and at transforms that make
external calls (HTTP, LLM, blob store, downstream service). The
audit-readiness panel uses this partition to highlight which catalog
entries an auditor must trace to source on every run.

Runtime classification lives in
``elspeth.web.audit_readiness.service._build_plugin_trust_row`` via the
predicate:

    kind in ("source", "sink") or
    plugin_cls.determinism in _AUDIT_FLAGGED_DETERMINISMS

i.e. every Source and every Sink is uniformly boundary, and a Transform
is boundary iff its declared ``Determinism`` is one of the audit-flagged
classes (currently ``EXTERNAL_CALL`` and ``NON_DETERMINISTIC``). This
module does NOT participate in that classification.

WHY THIS MODULE EXISTS — AUDIT DISCOVERABILITY
==============================================

The frozensets below are catalog-state pins, not allowlists. They are
consumed exclusively by the parity tests in
``tests/unit/web/audit_readiness/test_boundary_predicate_parity.py`` to
assert that the live builtin catalog's boundary partition matches the
documented expectation.

A future PR adding a new boundary plugin — a new Source, a new Sink, or
a Transform declaring ``Determinism.EXTERNAL_CALL`` or ``NON_DETERMINISTIC``
— will cause the parity test to fail. The author MUST update the relevant
frozenset here in the same commit. That update appears as a *production-
code* diff in the PR, so review sees an audit-relevant change in
production code rather than only a test-file edit.

This is the explicit gesture the deleted ``trust.py`` provided before
Phase 7A's No-Legacy discharge: a new Tier-3 crossing must surface as a
production-code change. Runtime classification has been (correctly)
relocated onto the plugin classes via ``Determinism``; the *audit-PR
discoverability gesture* now lives here. The two concerns are now
decoupled: a misclassified plugin fails one test, an undeclared catalog
addition fails this test, and adding a Tier-3 crossing is impossible
without a production-code diff that audit reviewers will see.

TWO-TIER RATIONALE FOR BOUNDARY TRANSFORMS
==========================================

(Preserved verbatim from the deleted ``trust.py`` so the historical
distinction does not have to be rediscovered.)

  - EXTERNAL_CALL determinism (automated): ``web_scrape``,
    ``rag_retrieval``, ``azure_content_safety``, ``azure_prompt_shield``.
    These plugins declare ``Determinism.EXTERNAL_CALL`` and the runtime
    predicate picks them up automatically.
  - Manual curation (LLM-class, NON_DETERMINISTIC): ``llm``. The plugin
    declares ``Determinism.NON_DETERMINISTIC`` because LLM outputs are
    not reproducible. It crosses an LLM API boundary and must be visible
    to auditors as BOUNDARY. Any future plugin added to this tier should
    appear in the transforms set with an explicit ``# NON_DETERMINISTIC,
    manual curation: <surface>`` comment.

ADDING A NEW BOUNDARY PLUGIN
============================

  1. Register the plugin in the relevant builtin manager registration
     (``elspeth.plugins.infrastructure.manager``).
  2. For a Transform, declare ``determinism = Determinism.EXTERNAL_CALL``
     on the class (or ``NON_DETERMINISTIC`` for an LLM-class boundary).
  3. Add the plugin's ``.name`` value to the relevant frozenset in this
     module, IN THE SAME COMMIT. The parity test will fail otherwise.
  4. A code reviewer seeing the diff to this file now has the explicit
     signal: "this PR adds a new Tier-3 crossing; route through
     audit-architecture review."

Sources and sinks are uniformly boundary by architecture: there is no
such thing as an "internal source" or "internal sink" in ELSPETH (a
Source reads external data into the pipeline; a Sink writes pipeline
data out — both cross Tier-3 by definition, regardless of whether the
destination is local or remote).

Layer: L3 (application).
"""

from __future__ import annotations

# Every registered Source is a boundary plugin by architecture: a Source
# reads external data into the pipeline. There is no such thing as an
# "internal source" in ELSPETH.
EXPECTED_BOUNDARY_SOURCES: frozenset[str] = frozenset(
    {
        "azure_blob",
        "csv",
        "dataverse",
        "json",
        "null",
        "text",
    },
)

# Every registered Sink is a boundary plugin by architecture: writing
# data out of the pipeline crosses an external trust boundary regardless
# of whether the destination is a local file or a remote service.
EXPECTED_BOUNDARY_SINKS: frozenset[str] = frozenset(
    {
        "azure_blob",
        "chroma_sink",
        "csv",
        "database",
        "dataverse",
        "json",
    },
)

# Boundary Transforms — those whose declared determinism is in
# ``_AUDIT_FLAGGED_DETERMINISMS`` (currently EXTERNAL_CALL and
# NON_DETERMINISTIC). Every other Transform is internal-only. See the
# module docstring's TWO-TIER RATIONALE for the automated-vs-manual
# distinction.
EXPECTED_BOUNDARY_TRANSFORMS: frozenset[str] = frozenset(
    {
        "azure_content_safety",  # Determinism.EXTERNAL_CALL
        "azure_prompt_shield",  # Determinism.EXTERNAL_CALL
        "llm",  # Determinism.NON_DETERMINISTIC — manual curation (LLM API boundary)
        "rag_retrieval",  # Determinism.EXTERNAL_CALL
        "web_scrape",  # Determinism.EXTERNAL_CALL
    },
)
