"""Deterministic unit tests for the composer parity live-acceptance oracle.

Plan 05 Task 5. Every test is offline: no provider network, no server, no live
key. A valid sanitized evidence document is built in memory, written to a
tmp evidence directory, and verified through the oracle's real load + verify
path. Each negative control mutates exactly one aspect of that valid document
and asserts the oracle REJECTS it with the intended stable code — including the
critical fake-provider control, whose evidence *lies* in its manifest
(``provider.mode == "live"``) yet is rejected on intrinsic per-call properties.

The oracle lives in ``evals/composer-parity/live_acceptance.py``; the hyphenated
directory is not an importable package, so it is loaded from its file path.
"""

from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.unit.evals.composer_parity._live_acceptance_fixtures import (
    REVISION,
    build_valid_evidence,
    la,
    write_evidence,
)

_MODULE_PATH = Path(la.__file__)


def _verify(tmp_path: Path, document: dict[str, Any], *, revision: str = REVISION) -> Any:
    write_evidence(tmp_path, revision, document)
    return la.verify_evidence_dir(tmp_path, revision)


# --------------------------------------------------------------------------- #
# Positive acceptance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("surface", ["freeform", "guided_full", "guided_staged"])
def test_valid_evidence_accepted(tmp_path: Path, surface: str) -> None:
    report = _verify(tmp_path, build_valid_evidence(surface=surface))
    assert report.surface == surface
    assert report.revision == REVISION
    assert "live_provider" in report.checks
    assert len(report.checks) == 9


def test_accept_single_repair(tmp_path: Path) -> None:
    """One automatic structured repair is within budget (boundary of `> 1`)."""
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["repair_count"] = 1
    assert _verify(tmp_path, doc).checks


def test_accept_inclusive_range_edges(tmp_path: Path) -> None:
    """Amounts at 0/100 and confidences at 0.0/1.0 are inside the closed ranges."""
    doc = build_valid_evidence()
    row = doc[la.BUSINESS_OUTPUT_FILE]["success"][0]
    row["blue_amount"] = 0
    row["red_amount"] = 100
    row["blue_confidence"] = 0.0
    row["red_confidence"] = 1.0
    assert _verify(tmp_path, doc).checks


# --------------------------------------------------------------------------- #
# Content negative controls (plan Task 5 list)
# --------------------------------------------------------------------------- #


def _expect_reject(tmp_path: Path, document: dict[str, Any], code: str, *, revision: str = REVISION) -> None:
    with pytest.raises(la.AcceptanceError) as excinfo:
        _verify(tmp_path, document, revision=revision)
    assert excinfo.value.code == code, f"expected code {code!r}, got {excinfo.value.code!r}: {excinfo.value}"


def test_reject_one_llm(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.GRAPH_FILE]["nodes"] = [n for n in doc[la.GRAPH_FILE]["nodes"] if n["id"] != "red_llm"]
    _expect_reject(tmp_path, doc, "two_distinct_llm_nodes")


def test_reject_missing_branch_row(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    calls = doc[la.RUN_LLM_CALLS_FILE]
    # Drop one blue-branch assessment: blue_llm now covers only nine rows.
    for i, call in enumerate(calls):
        if call["branch_node_id"] == "blue_llm":
            del calls[i]
            break
    _expect_reject(tmp_path, doc, "branch_missing_rows")


def test_reject_wrong_merge(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.GRAPH_FILE]["merge"]["semantics"] = "override"
    _expect_reject(tmp_path, doc, "wrong_merge")


def test_reject_wrong_merge_mode(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.GRAPH_FILE]["merge"]["mode"] = "first"
    _expect_reject(tmp_path, doc, "wrong_merge")


def test_reject_unrouted_failure(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.GRAPH_FILE]["edges"] = [e for e in doc[la.GRAPH_FILE]["edges"] if e["role"] != "error"]
    _expect_reject(tmp_path, doc, "unrouted_failure")


def test_reject_extra_field(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["saturation"] = 42
    _expect_reject(tmp_path, doc, "field_contract_violation")


def test_reject_missing_field(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    del doc[la.BUSINESS_OUTPUT_FILE]["success"][3]["red_reason"]
    _expect_reject(tmp_path, doc, "field_contract_violation")


def test_reject_leaked_metadata(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    # A raw-response metadata field that survived cleanup into the business row.
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_reason_model"] = "anthropic/claude"
    _expect_reject(tmp_path, doc, "leaked_metadata")


def test_reject_jsonl_root(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    rows = doc[la.BUSINESS_OUTPUT_FILE]["success"]
    # JSONL / object root: an object keyed by index rather than a JSON array.
    doc[la.BUSINESS_OUTPUT_FILE]["success"] = {str(i): row for i, row in enumerate(rows)}
    _expect_reject(tmp_path, doc, "jsonl_root")


def test_reject_duplicate_identity(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.INPUT_IDENTITIES_FILE][9] = dict(doc[la.INPUT_IDENTITIES_FILE][0])
    _expect_reject(tmp_path, doc, "duplicate_identity")


def test_reject_duplicate_business_identity(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    rows = doc[la.BUSINESS_OUTPUT_FILE]["success"]
    rows[9]["color_name"] = rows[0]["color_name"]
    rows[9]["hex"] = rows[0]["hex"]
    _expect_reject(tmp_path, doc, "duplicate_identity")


def test_reject_bool_as_number(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_amount"] = True
    _expect_reject(tmp_path, doc, "amount_not_integer")


def test_reject_amount_out_of_range(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["red_amount"] = 250
    _expect_reject(tmp_path, doc, "amount_out_of_range")


def test_reject_nan_confidence(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_confidence"] = float("nan")
    _expect_reject(tmp_path, doc, "confidence_not_finite")


def test_reject_confidence_out_of_range(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["red_confidence"] = 1.5
    _expect_reject(tmp_path, doc, "confidence_out_of_range")


def test_reject_empty_reason(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_reason"] = "   "
    _expect_reject(tmp_path, doc, "reason_empty")


def test_reject_nonterminal_accounting_open(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.RUN_ACCOUNTING_FILE]["closed"] = False
    _expect_reject(tmp_path, doc, "nonterminal_accounting")


def test_reject_nonterminal_accounting_pending(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.RUN_ACCOUNTING_FILE]["pending_tokens"] = 3
    _expect_reject(tmp_path, doc, "nonterminal_accounting")


def test_reject_excess_repair(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["repair_count"] = 2
    _expect_reject(tmp_path, doc, "excess_repair")


def test_reject_operator_topology_correction(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["operator_corrections"] = [{"kind": "topology", "detail": "operator added a coalesce"}]
    _expect_reject(tmp_path, doc, "operator_topology_correction")


def test_reject_mixed_revision(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.GRAPH_FILE]["revision"] = "some-other-revision"
    _expect_reject(tmp_path, doc, "mixed_revision")


# --------------------------------------------------------------------------- #
# Fake-provider control: the manifest LIES (mode == "live") but intrinsic
# per-call evidence gives it away. This is the whole oracle's acceptance bar.
# --------------------------------------------------------------------------- #


def test_reject_fake_provider_sentinel_model(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    assert doc[la.MANIFEST_FILE]["provider"]["mode"] == "live"  # the lie
    doc[la.RUN_LLM_CALLS_FILE][0]["model_returned"] = "mock-echo-model"
    _expect_reject(tmp_path, doc, "fake_provider")


def test_reject_fake_provider_no_usage(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.RUN_LLM_CALLS_FILE][5]["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _expect_reject(tmp_path, doc, "fake_provider")


def test_reject_fake_provider_forged_request_id(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.RUN_LLM_CALLS_FILE][2]["provider_request_id"] = "replay-fixture-0001"
    _expect_reject(tmp_path, doc, "fake_provider")


def test_reject_fake_provider_missing_request_id(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.RUN_LLM_CALLS_FILE][7]["provider_request_id"] = ""
    _expect_reject(tmp_path, doc, "fake_provider")


def test_reject_honest_fake_provider_mode(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["mode"] = "fake"
    _expect_reject(tmp_path, doc, "provider_not_live")


# --------------------------------------------------------------------------- #
# Evidence-directory hygiene + redaction
# --------------------------------------------------------------------------- #


def test_reject_lax_permissions(tmp_path: Path) -> None:
    revision_dir = write_evidence(tmp_path, REVISION, build_valid_evidence())
    revision_dir.chmod(0o777)
    with pytest.raises(la.EvidenceHygieneError) as excinfo:
        la.verify_evidence_dir(tmp_path, REVISION)
    assert excinfo.value.code == "lax_permissions"


def test_reject_symlink_in_tree(tmp_path: Path) -> None:
    revision_dir = write_evidence(tmp_path, REVISION, build_valid_evidence())
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (revision_dir / "link.json").symlink_to(outside)
    with pytest.raises(la.EvidenceHygieneError) as excinfo:
        la.verify_evidence_dir(tmp_path, REVISION)
    assert excinfo.value.code == "symlink_rejected"


def test_reject_revision_dir_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real_rev"
    write_evidence(tmp_path, "real_rev", build_valid_evidence(revision="linked"))
    link_base = tmp_path / "evidence"
    link_base.mkdir()
    (link_base / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(la.EvidenceHygieneError) as excinfo:
        la.verify_evidence_dir(link_base, "linked")
    assert excinfo.value.code == "symlink_rejected"


def test_reject_bad_revision_token(tmp_path: Path) -> None:
    write_evidence(tmp_path, REVISION, build_valid_evidence())
    with pytest.raises(la.EvidenceHygieneError) as excinfo:
        la.verify_evidence_dir(tmp_path, "../escape")
    assert excinfo.value.code == "bad_revision_token"


def test_reject_missing_evidence_file(tmp_path: Path) -> None:
    revision_dir = write_evidence(tmp_path, REVISION, build_valid_evidence())
    (revision_dir / la.RUN_ACCOUNTING_FILE).unlink()
    with pytest.raises(la.EvidenceHygieneError) as excinfo:
        la.verify_evidence_dir(tmp_path, REVISION)
    assert excinfo.value.code == "missing_evidence_file"


@pytest.mark.parametrize(
    "path, key, value",
    [
        ([la.MANIFEST_FILE, "provider"], "authorization", "Bearer sk-secret-value"),
        ([la.RUN_LLM_CALLS_FILE, 0], "raw_response", "the full provider body"),
        ([la.MANIFEST_FILE], "session_cookie", "abc"),
        ([la.RUN_LLM_CALLS_FILE, 1], "api_key", "sk-live-123"),
    ],
)
def test_reject_sensitive_content_retained(tmp_path: Path, path: list[Any], key: str, value: str) -> None:
    doc = build_valid_evidence()
    target: Any = doc
    for step in path:
        target = target[step]
    target[key] = value
    _expect_reject(tmp_path, doc, "sensitive_content_retained")


def test_reject_sensitive_bearer_value_under_benign_key(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["note"] = "call used Bearer sk-abcdef0123456789"
    _expect_reject(tmp_path, doc, "sensitive_content_retained")


def test_redact_evidence_strips_secrets_then_accepts(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["authorization"] = "Bearer sk-should-be-stripped"
    doc[la.RUN_LLM_CALLS_FILE][0]["set-cookie"] = "session=nope"
    redacted = {name: la.redact_evidence(payload) for name, payload in doc.items()}
    assert "authorization" not in redacted[la.MANIFEST_FILE]["provider"]
    assert "set-cookie" not in redacted[la.RUN_LLM_CALLS_FILE][0]
    report = _verify(tmp_path, redacted)
    assert report.checks


# --------------------------------------------------------------------------- #
# The `run` machinery is offline-testable via an injected collector; the live
# collector default declines rather than fabricate provider evidence.
# --------------------------------------------------------------------------- #


def test_run_live_with_injected_collector_writes_redacted_and_verifies(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def collector(*, base_url: str, api_key: str, surface: str, fixture: Path, intent: Path, revision: str) -> dict[str, Any]:
        captured["base_url"] = base_url
        doc = build_valid_evidence(surface=surface, revision=revision)
        # A collector that failed to sanitize: the export path must redact it.
        doc[la.MANIFEST_FILE]["provider"]["authorization"] = "Bearer sk-leaked"
        return doc

    report = la.run_live(
        base_url="https://staging.example",
        api_key="dummy-not-a-real-key",
        surface="freeform",
        fixture=tmp_path / "colours.csv",
        intent=tmp_path / "request.txt",
        revision=REVISION,
        evidence_dir=tmp_path / "evidence",
        collector=collector,
    )
    assert report.surface == "freeform"
    assert captured["base_url"] == "https://staging.example"
    # The written manifest was redacted on export.
    written = json.loads((tmp_path / "evidence" / REVISION / la.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert "authorization" not in written["provider"]
    # Files are written with restrictive permissions.
    mode = stat.S_IMODE((tmp_path / "evidence" / REVISION / la.MANIFEST_FILE).lstat().st_mode)
    assert mode & (stat.S_IWGRP | stat.S_IWOTH) == 0


def test_run_live_rejects_fake_collector_evidence(tmp_path: Path) -> None:
    def collector(*, base_url: str, api_key: str, surface: str, fixture: Path, intent: Path, revision: str) -> dict[str, Any]:
        doc = build_valid_evidence(surface=surface, revision=revision)
        doc[la.RUN_LLM_CALLS_FILE][0]["provider_request_id"] = "mock-0"
        return doc

    with pytest.raises(la.AcceptanceError) as excinfo:
        la.run_live(
            base_url="https://staging.example",
            api_key="dummy",
            surface="freeform",
            fixture=tmp_path / "c.csv",
            intent=tmp_path / "r.txt",
            revision=REVISION,
            evidence_dir=tmp_path / "evidence",
            collector=collector,
        )
    assert excinfo.value.code == "fake_provider"


def test_default_live_collector_declines(tmp_path: Path) -> None:
    with pytest.raises(la.LiveCollectionUnavailable):
        la.run_live(
            base_url="https://staging.example",
            api_key="dummy",
            surface="freeform",
            fixture=tmp_path / "c.csv",
            intent=tmp_path / "r.txt",
            revision=REVISION,
            evidence_dir=tmp_path / "evidence",
        )


def test_cli_run_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ELSPETH_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("ELSPETH_EVAL_BASE_URL", raising=False)
    argv = [
        "run",
        "--surface",
        "freeform",
        "--base-url",
        "https://staging.example",
        "--fixture",
        str(tmp_path / "c.csv"),
        "--intent",
        str(tmp_path / "r.txt"),
        "--revision",
        REVISION,
        "--evidence-dir",
        str(tmp_path / "evidence"),
    ]
    with pytest.raises(SystemExit):
        la.main(argv)


def test_deepcopy_isolation_between_negatives() -> None:
    """Sanity: build_valid_evidence returns fresh mutable state each call."""
    a = build_valid_evidence()
    b = build_valid_evidence()
    a[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_amount"] = -1
    assert b[la.BUSINESS_OUTPUT_FILE]["success"][0]["blue_amount"] != -1
    assert copy.deepcopy(a) is not a


def test_module_path_is_tracked() -> None:
    """The oracle file exists at the expected path for the git-tracking check."""
    assert _MODULE_PATH.is_file()
    assert _MODULE_PATH.name == "live_acceptance.py"
    assert not os.path.islink(_MODULE_PATH)
