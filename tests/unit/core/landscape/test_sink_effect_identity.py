"""Bounded lineage and deterministic sink-effect identity contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    AuditExportSignedManifestInput,
    AuditExportSigningMode,
    AuditExportSnapshotChunkInput,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectInputKind,
    SinkEffectMemberCandidate,
    SinkEffectRole,
    _create_restricted_audit_export_snapshot_reader,
)
from elspeth.core.landscape.execution.sink_effect_identity import (
    MAX_LINEAGE_DEPTH,
    MAX_LINEAGE_EVIDENCE_BYTES,
    MAX_LINEAGE_NODES_PER_MEMBER,
    MAX_LINEAGE_PARENTS,
    compute_audit_export_effect_identity,
    compute_pipeline_effect_identity,
    resolve_sink_effect_members,
)


@dataclass(frozen=True)
class _Token:
    token_id: str
    row_id: str
    run_id: str
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None


@dataclass(frozen=True)
class _Row:
    row_id: str
    run_id: str
    ingest_sequence: int


@dataclass(frozen=True)
class _Parent:
    token_id: str
    parent_token_id: str
    ordinal: int


class _LineageSource:
    def __init__(
        self,
        *,
        tokens: tuple[_Token, ...],
        rows: tuple[_Row, ...],
        parents: tuple[_Parent, ...],
    ) -> None:
        self.query = self
        self.tokens = {token.token_id: token for token in tokens}
        self.rows = {row.row_id: row for row in rows}
        self.parents: dict[str, list[_Parent]] = {}
        for parent in parents:
            self.parents.setdefault(parent.token_id, []).append(parent)

    def get_token(self, token_id: str) -> _Token | None:
        return self.tokens.get(token_id)

    def get_tokens_by_ids(self, token_ids: tuple[str, ...]) -> list[_Token]:
        return [self.tokens[token_id] for token_id in token_ids if token_id in self.tokens]

    def get_token_parents(self, token_id: str) -> list[_Parent]:
        return list(self.parents.get(token_id, ()))

    def get_row(self, row_id: str) -> _Row | None:
        return self.rows.get(row_id)


def _candidate(token_id: str, value: int = 1) -> SinkEffectMemberCandidate:
    return SinkEffectMemberCandidate(token_id=token_id, row={"value": value})


def _logical_graph() -> tuple[_LineageSource, tuple[SinkEffectMemberCandidate, ...]]:
    tokens = tuple(
        _Token(token_id, row_id, "run-1")
        for token_id, row_id in (
            ("root", "row-0"),
            ("fork-0", "row-0"),
            ("fork-1", "row-0"),
            ("joined", "row-0"),
            ("next-root", "row-1"),
        )
    )
    rows = (_Row("row-0", "run-1", 0), _Row("row-1", "run-1", 1))
    parents = (
        _Parent("fork-0", "root", 0),
        _Parent("fork-1", "root", 1),
        _Parent("joined", "fork-0", 0),
        _Parent("joined", "fork-1", 1),
    )
    candidates = tuple(_candidate(token_id, index) for index, token_id in enumerate(("next-root", "joined", "fork-1", "fork-0")))
    return _LineageSource(tokens=tokens, rows=rows, parents=parents), candidates


def test_member_order_uses_ingest_then_recursive_parent_ordinals() -> None:
    source, candidates = _logical_graph()
    ordered = resolve_sink_effect_members(source, reversed(candidates))
    # Recursive tuple ordering is structural: the join begins with its
    # ordinal-zero parent's path and therefore sorts before the ordinal-one
    # fork sibling for the same ingest sequence.
    assert [member.token_id for member in ordered] == ["fork-0", "joined", "fork-1", "next-root"]
    assert [member.ordinal for member in ordered] == list(range(len(candidates)))
    assert ordered[0].lineage_json == "[[0,[]]]"
    assert ordered[1].lineage_json == "[[0,[[0,[]]]],[1,[[1,[]]]]]"
    assert ordered[2].lineage_json == "[[1,[]]]"


@pytest.mark.parametrize(
    "corruption", ["cycle", "missing_parent", "duplicate_ordinal", "non_dense_ordinal", "cross_run", "repeated_parent"]
)
def test_lineage_corruption_fails_closed(corruption: str) -> None:
    tokens = [_Token("child", "row-0", "run-1"), _Token("parent", "row-0", "run-1")]
    parents = [_Parent("child", "parent", 0)]
    if corruption == "cycle":
        parents.append(_Parent("parent", "child", 0))
    elif corruption == "missing_parent":
        parents[0] = _Parent("child", "absent", 0)
    elif corruption == "duplicate_ordinal":
        tokens.append(_Token("other", "row-0", "run-1"))
        parents.append(_Parent("child", "other", 0))
    elif corruption == "non_dense_ordinal":
        tokens.append(_Token("other", "row-0", "run-1"))
        parents.append(_Parent("child", "other", 2))
    elif corruption == "cross_run":
        tokens[1] = _Token("parent", "row-x", "run-2")
    else:
        parents.append(_Parent("child", "parent", 1))
    source = _LineageSource(
        tokens=tuple(tokens),
        rows=(_Row("row-0", "run-1", 0), _Row("row-x", "run-2", 0)),
        parents=tuple(parents),
    )
    with pytest.raises(AuditIntegrityError):
        resolve_sink_effect_members(source, (_candidate("child"),))


@pytest.mark.parametrize(
    ("limit_name", "source", "candidate"),
    [
        (
            "depth",
            _LineageSource(
                tokens=tuple(_Token(f"t{i}", "row-0", "run-1") for i in range(MAX_LINEAGE_DEPTH + 2)),
                rows=(_Row("row-0", "run-1", 0),),
                parents=tuple(_Parent(f"t{i}", f"t{i - 1}", 0) for i in range(1, MAX_LINEAGE_DEPTH + 2)),
            ),
            _candidate(f"t{MAX_LINEAGE_DEPTH + 1}"),
        ),
        (
            "fan-in",
            _LineageSource(
                tokens=tuple(
                    [_Token("child", "row-0", "run-1")] + [_Token(f"p{i}", "row-0", "run-1") for i in range(MAX_LINEAGE_PARENTS + 1)]
                ),
                rows=(_Row("row-0", "run-1", 0),),
                parents=tuple(_Parent("child", f"p{i}", i) for i in range(MAX_LINEAGE_PARENTS + 1)),
            ),
            _candidate("child"),
        ),
    ],
)
def test_lineage_resource_limits_fail_closed(limit_name: str, source: _LineageSource, candidate: SinkEffectMemberCandidate) -> None:
    with pytest.raises(AuditIntegrityError, match=limit_name):
        resolve_sink_effect_members(source, (candidate,))


def test_lineage_contract_limits_are_exact() -> None:
    assert (MAX_LINEAGE_DEPTH, MAX_LINEAGE_NODES_PER_MEMBER, MAX_LINEAGE_PARENTS, MAX_LINEAGE_EVIDENCE_BYTES) == (
        256,
        4096,
        1024,
        64 * 1024,
    )


def test_relation_child_mismatch_and_claimed_lineage_without_parent_fail_closed() -> None:
    row = _Row("row-0", "run-1", 0)
    mismatched = _LineageSource(
        tokens=(_Token("child", "row-0", "run-1"), _Token("parent", "row-0", "run-1")),
        rows=(row,),
        parents=(_Parent("different-child", "parent", 0),),
    )
    # Route the corrupt row through the child's lookup bucket, as a damaged
    # loader/query could do despite its embedded child ID.
    mismatched.parents["child"] = list(mismatched.parents.pop("different-child"))
    with pytest.raises(AuditIntegrityError, match="relation child"):
        resolve_sink_effect_members(mismatched, (_candidate("child"),))

    missing_relation = _LineageSource(
        tokens=(_Token("child", "row-0", "run-1", fork_group_id="fork-1"),),
        rows=(row,),
        parents=(),
    )
    with pytest.raises(AuditIntegrityError, match="claims lineage"):
        resolve_sink_effect_members(missing_relation, (_candidate("child"),))


def test_parent_row_ownership_corruption_fails_closed() -> None:
    source = _LineageSource(
        tokens=(_Token("child", "row-0", "run-1"), _Token("parent", "row-x", "run-1")),
        rows=(_Row("row-0", "run-1", 0), _Row("row-x", "run-1", 1)),
        parents=(_Parent("child", "parent", 0),),
    )
    with pytest.raises(AuditIntegrityError, match="row"):
        resolve_sink_effect_members(source, (_candidate("child"),))


def test_lineage_node_and_evidence_limits_fail_closed() -> None:
    # 1 root + 1,024 parents + 4 unique grandparents each exceeds 4,096
    # visited nodes without exceeding either depth or per-token fan-in.
    node_tokens = [_Token("root", "row-0", "run-1")]
    node_parents: list[_Parent] = []
    for parent_index in range(MAX_LINEAGE_PARENTS):
        parent_id = f"p{parent_index}"
        node_tokens.append(_Token(parent_id, "row-0", "run-1"))
        node_parents.append(_Parent("root", parent_id, parent_index))
        for child_index in range(4):
            grandparent_id = f"g{parent_index}-{child_index}"
            node_tokens.append(_Token(grandparent_id, "row-0", "run-1"))
            node_parents.append(_Parent(parent_id, grandparent_id, child_index))
    node_source = _LineageSource(
        tokens=tuple(node_tokens),
        rows=(_Row("row-0", "run-1", 0),),
        parents=tuple(node_parents),
    )
    with pytest.raises(AuditIntegrityError, match="node count"):
        resolve_sink_effect_members(node_source, (_candidate("root"),))

    # A bounded shared ancestor chain is repeated structurally beneath 1,000
    # distinct root parents. Unique visited nodes remain below 4,096, while
    # the canonical evidence necessarily exceeds 64 KiB.
    evidence_tokens = [_Token("root", "row-0", "run-1")]
    evidence_parents: list[_Parent] = []
    for depth in range(101):
        evidence_tokens.append(_Token(f"chain-{depth}", "row-0", "run-1"))
        if depth:
            evidence_parents.append(_Parent(f"chain-{depth}", f"chain-{depth - 1}", 0))
    for parent_index in range(1_000):
        parent_id = f"repeat-{parent_index}"
        evidence_tokens.append(_Token(parent_id, "row-0", "run-1"))
        evidence_parents.append(_Parent("root", parent_id, parent_index))
        evidence_parents.append(_Parent(parent_id, "chain-100", parent_index))
    evidence_source = _LineageSource(
        tokens=tuple(evidence_tokens),
        rows=(_Row("row-0", "run-1", 0),),
        parents=tuple(evidence_parents),
    )
    with pytest.raises(AuditIntegrityError, match="lineage evidence"):
        resolve_sink_effect_members(evidence_source, (_candidate("root"),))


def test_state_attempt_ids_do_not_change_effect_identity() -> None:
    source, candidates = _logical_graph()
    members = resolve_sink_effect_members(source, candidates)
    first = compute_pipeline_effect_identity(
        run_id="run-1",
        sink_node_id="sink-1",
        role=SinkEffectRole.PRIMARY,
        sink_config={"format": "json"},
        target_config={"path": "safe/output.json"},
        members=members,
    )
    first_resolution = SimpleNamespace(members=members, current_state_ids=("state-attempt-0",))
    second_resolution = SimpleNamespace(members=members, current_state_ids=("state-attempt-1",))
    second = compute_pipeline_effect_identity(
        run_id="run-1",
        sink_node_id="sink-1",
        role=SinkEffectRole.PRIMARY,
        sink_config={"format": "json"},
        target_config={"path": "safe/output.json"},
        members=second_resolution.members,
    )
    assert first_resolution.current_state_ids != second_resolution.current_state_ids
    assert first == second
    assert first.member_ids == tuple(member.member_effect_id for member in first.members)
    assert all(first.member_ids)
    for identifier in (
        first.effect_id,
        first.artifact_id,
        first.artifact_idempotency_key,
        first.stream_id,
        *first.member_ids,
    ):
        assert len(identifier) == 64
        int(identifier, 16)


def test_failsink_identity_binds_each_members_primary_effect_id() -> None:
    source, candidates = _logical_graph()
    members = resolve_sink_effect_members(source, candidates)
    first_members = tuple(replace(member, primary_effect_id="a" * 64) for member in members)
    second_members = tuple(replace(member, primary_effect_id="b" * 64) for member in members)

    def identity(bound_members):
        return compute_pipeline_effect_identity(
            run_id="run-1",
            sink_node_id="failsink-1",
            role=SinkEffectRole.FAILSINK,
            sink_config={"format": "json"},
            target_config={"path": "safe/failsink.json"},
            members=bound_members,
        )

    first = identity(first_members)
    second = identity(second_members)

    assert first.membership_or_manifest_hash != second.membership_or_manifest_hash
    assert first.effect_id != second.effect_id


def test_pipeline_identity_binds_ordered_payload_and_safe_target_without_leaking_values() -> None:
    source, candidates = _logical_graph()
    members = resolve_sink_effect_members(source, candidates)
    baseline = compute_pipeline_effect_identity(
        run_id="run-1",
        sink_node_id="sink-1",
        role=SinkEffectRole.PRIMARY,
        sink_config={"format": "json"},
        target_config={"bucket": "public-output", "prefix": "daily"},
        members=members,
    )
    changed = compute_pipeline_effect_identity(
        run_id="run-1",
        sink_node_id="sink-1",
        role=SinkEffectRole.PRIMARY,
        sink_config={"format": "json"},
        target_config={"bucket": "public-output", "prefix": "weekly"},
        members=members,
    )
    assert baseline.effect_id != changed.effect_id
    assert baseline.stream_id != changed.stream_id
    rendered = repr(baseline)
    assert "public-output" not in rendered and "daily" not in rendered
    with pytest.raises(ValueError, match="credential-free"):
        compute_pipeline_effect_identity(
            run_id="run-1",
            sink_node_id="sink-1",
            role=SinkEffectRole.PRIMARY,
            sink_config={"format": "json"},
            target_config={"password": "not-safe"},
            members=members,
        )


def _export_input() -> SinkEffectAuditExportSnapshotInput:
    chunk_bytes = b'{"record":1}\n'
    chunk_hash = sha256(chunk_bytes).hexdigest()
    chunk = AuditExportSnapshotChunkInput(0, f"sha256:{chunk_hash}", chunk_hash, len(chunk_bytes), 1)
    manifest_bytes = (
        b'{"chunk_count":1,"derivation_version":"audit-export-derivation-v1","export_format":"json",'
        b'"exported_at":"2026-07-16T01:02:03.456789Z","final_hash":"' + b"6" * 64 + b'",'
        b'"hash_algorithm":"sha256","last_chunk_seal_hash":"' + b"7" * 64 + b'","manifest_hash":"' + b"3" * 64 + b'",'
        b'"record_chain_algorithm":"sha256_concat_record_sha256_v1","record_count":1,"record_type":"manifest",'
        b'"registry_key_hash":"' + b"2" * 64 + b'","run_id":"source-run-1","schema":"elspeth.audit-export-manifest.v2",'
        b'"signature":null,"signature_algorithm":"unsigned","signature_key_id":"UNSIGNED","snapshot_hash":"' + b"4" * 64 + b'",'
        b'"snapshot_id":"' + b"1" * 64 + b'","snapshot_seal_hash":"' + b"8" * 64 + b'",'
        b'"source_completed_at":"2026-07-16T01:02:03.456789Z","source_status":"completed","total_bytes":13}'
    )
    manifest_hash = sha256(manifest_bytes).hexdigest()
    descriptor = AuditExportSignedManifestInput(
        content_ref=f"sha256:{manifest_hash}",
        content_hash=manifest_hash,
        size_bytes=len(manifest_bytes),
        manifest_schema="elspeth.audit-export-manifest.v2",
        derivation_version="audit-export-derivation-v1",
        signature_algorithm=AuditExportSigningMode.UNSIGNED,
        signature_key_id="UNSIGNED",
        record_chain_algorithm="sha256_concat_record_sha256_v1",
        final_hash="6" * 64,
        signature=None,
    )
    objects = {chunk.content_ref: chunk_bytes, descriptor.content_ref: manifest_bytes}
    reader = _create_restricted_audit_export_snapshot_reader(
        snapshot_id="1" * 64,
        source_run_id="source-run-1",
        registry_key_hash="2" * 64,
        manifest_hash="3" * 64,
        snapshot_hash="4" * 64,
        export_format=AuditExportFormat.JSON,
        signing_mode=AuditExportSigningMode.UNSIGNED,
        signer_key_id="UNSIGNED",
        record_count=1,
        total_bytes=len(chunk_bytes),
        serialization_version="audit-export-v2",
        exported_at="2026-07-16T01:02:03.456789Z",
        source_completed_at="2026-07-16T01:02:03.456789Z",
        source_status="completed",
        last_chunk_seal_hash="7" * 64,
        snapshot_seal_hash="8" * 64,
        chunks=(chunk,),
        signed_manifest=descriptor,
        store_resolver=objects.__getitem__,
        record_counter=lambda content: content.count(b"\n"),
        signed_manifest_verifier=lambda _content, _descriptor: None,
    )
    return SinkEffectAuditExportSnapshotInput(
        snapshot_id="1" * 64,
        source_run_id="source-run-1",
        registry_key_hash="2" * 64,
        manifest_hash="3" * 64,
        snapshot_hash="4" * 64,
        serialization_version="audit-export-v2",
        export_format=AuditExportFormat.JSON,
        signing_mode=AuditExportSigningMode.UNSIGNED,
        signer_key_id="UNSIGNED",
        record_count=1,
        total_bytes=len(chunk_bytes),
        chunk_count=1,
        chunks=(chunk,),
        signed_manifest=descriptor,
        reader=reader,
    )


def test_audit_export_identity_binds_manifest_without_synthetic_members() -> None:
    snapshot = _export_input()
    identity = compute_audit_export_effect_identity(
        snapshot,
        {"path": "safe/export.json"},
        sink_node_id="audit-export",
        role=SinkEffectRole.PRIMARY,
    )
    assert identity.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT
    assert identity.member_ids == ()
    assert identity.members == ()
    assert identity.snapshot_hash == snapshot.snapshot_hash


def test_exact_final_manifest_descriptor_converges_across_reader_instances() -> None:
    first = _export_input()
    second = _export_input()
    assert first.reader is not second.reader
    assert compute_audit_export_effect_identity(
        first, {"path": "safe/export.json"}, sink_node_id="audit-export", role=SinkEffectRole.PRIMARY
    ) == compute_audit_export_effect_identity(
        second, {"path": "safe/export.json"}, sink_node_id="audit-export", role=SinkEffectRole.PRIMARY
    )


def test_export_cross_mapping_fails_before_hashing() -> None:
    snapshot = _export_input()
    forged = SimpleNamespace(**{name: getattr(snapshot, name) for name in snapshot.__dataclass_fields__})
    forged.signer_key_id = "different-key"
    with pytest.raises((TypeError, ValueError), match=r"signer|snapshot"):
        compute_audit_export_effect_identity(forged, {"path": "safe/export.json"}, sink_node_id="audit-export", role=SinkEffectRole.PRIMARY)
