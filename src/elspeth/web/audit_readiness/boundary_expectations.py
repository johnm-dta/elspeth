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
classes (currently ``IO_READ``, ``EXTERNAL_CALL``, and
``NON_DETERMINISTIC``). This module does NOT participate in that
classification.

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

TRANSFORM BOUNDARY RATIONALE
==========================================

(Preserved verbatim from the deleted ``trust.py`` so the historical
distinction does not have to be rediscovered.)

  - IO_READ determinism (automated): payload/file parser transforms such
    as ``blob_csv_expand``. These do not call the network themselves, but
    they read blob-backed external bytes and cross the parser trust
    boundary into row data.
  - EXTERNAL_CALL determinism (automated): ``blob_fetch``, ``web_scrape``,
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

The deliberate widening of sink classification (csv/json sinks are now
boundary, where the deleted ``trust.py`` excluded them) is recorded in
ADR-021 (``docs/architecture/adr/021-sources-and-sinks-uniformly-boundary.md``).
Read that ADR for the rationale, the three rejected alternatives, and
the follow-up UX nit about detail-row wording for local-filesystem
sinks.

Layer: L3 (application).
"""

from __future__ import annotations

from elspeth.contracts.enums import Determinism

# Per-plugin determinism expectations. The previous frozenset-of-names
# form was strengthened to a per-name map after a parity-test review
# observed that set-equality could theoretically cancel under offsetting
# drift (rename + re-add, or determinism change on a boundary-classified
# plugin where the kind short-circuits the predicate). Per-plugin
# determinism pinning catches both:
#
#   - A new builtin plugin not added here fails the parity test (name set
#     drift).
#   - A determinism declaration that changes value on a registered plugin
#     fails the parity test (value drift), even when the plugin's kind
#     would short-circuit the boundary predicate. A Source that flips
#     from IO_READ to NON_DETERMINISTIC remains boundary by kind, but the
#     declaration drift is itself audit-relevant — auditors care about
#     the declared determinism, not only the boundary outcome.
#
# Sources and sinks are uniformly boundary by architecture regardless of
# declared determinism (ADR-021). The maps below pin the declared
# determinism per builtin, not the boundary classification (which is
# derivable from kind + the predicate in
# ``_build_plugin_trust_row``). Transforms classify as boundary iff
# their declared determinism is in ``_AUDIT_FLAGGED_DETERMINISMS``.

EXPECTED_SOURCE_DETERMINISMS: dict[str, Determinism] = {
    "azure_blob": Determinism.IO_READ,
    "csv": Determinism.IO_READ,
    "dataverse": Determinism.EXTERNAL_CALL,
    "json": Determinism.IO_READ,
    "null": Determinism.DETERMINISTIC,
    "text": Determinism.IO_READ,
}

EXPECTED_SINK_DETERMINISMS: dict[str, Determinism] = {
    "azure_blob": Determinism.IO_WRITE,
    "chroma_sink": Determinism.IO_WRITE,
    "csv": Determinism.IO_WRITE,
    "database": Determinism.IO_WRITE,
    "dataverse": Determinism.EXTERNAL_CALL,
    "json": Determinism.IO_WRITE,
}

EXPECTED_TRANSFORM_DETERMINISMS: dict[str, Determinism] = {
    "azure_content_safety": Determinism.EXTERNAL_CALL,
    "azure_document_intelligence": Determinism.EXTERNAL_CALL,
    "azure_prompt_shield": Determinism.EXTERNAL_CALL,
    "blob_csv_expand": Determinism.IO_READ,
    "blob_fetch": Determinism.EXTERNAL_CALL,
    "batch_classifier_metrics": Determinism.DETERMINISTIC,
    "batch_data_quality_report": Determinism.DETERMINISTIC,
    "batch_distribution_profile": Determinism.DETERMINISTIC,
    "batch_drift_compare": Determinism.DETERMINISTIC,
    "batch_effect_size": Determinism.DETERMINISTIC,
    "batch_experiment_compare": Determinism.DETERMINISTIC,
    "batch_outlier_annotator": Determinism.DETERMINISTIC,
    "batch_paired_preference": Determinism.DETERMINISTIC,
    "batch_replicate": Determinism.DETERMINISTIC,
    "batch_stats": Determinism.DETERMINISTIC,
    "batch_threshold_summary": Determinism.DETERMINISTIC,
    "batch_top_k": Determinism.DETERMINISTIC,
    "field_mapper": Determinism.DETERMINISTIC,
    "json_explode": Determinism.DETERMINISTIC,
    "keyword_filter": Determinism.DETERMINISTIC,
    "line_explode": Determinism.DETERMINISTIC,
    "llm": Determinism.NON_DETERMINISTIC,  # manual curation (LLM API boundary)
    "passthrough": Determinism.DETERMINISTIC,
    "rag_retrieval": Determinism.EXTERNAL_CALL,
    "report_assemble": Determinism.DETERMINISTIC,
    "truncate": Determinism.DETERMINISTIC,
    "type_coerce": Determinism.DETERMINISTIC,
    "value_transform": Determinism.DETERMINISTIC,
    "web_scrape": Determinism.EXTERNAL_CALL,
}

# Derived name-only sets preserved for the unchanged "every Source/Sink
# is boundary by architecture" assertions. These are computed from the
# determinism maps above so a single declaration site governs both.
EXPECTED_BOUNDARY_SOURCES: frozenset[str] = frozenset(EXPECTED_SOURCE_DETERMINISMS)
EXPECTED_BOUNDARY_SINKS: frozenset[str] = frozenset(EXPECTED_SINK_DETERMINISMS)
EXPECTED_BOUNDARY_TRANSFORMS: frozenset[str] = frozenset(
    {
        name
        for name, det in EXPECTED_TRANSFORM_DETERMINISMS.items()
        if det in {Determinism.IO_READ, Determinism.EXTERNAL_CALL, Determinism.NON_DETERMINISTIC}
    },
)
