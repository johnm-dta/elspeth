"""Tracer-bullet: set_source end-to-end through manifest + walker (spec §11).

These tests pin the integration shape established in Task 4 of the Phase 2
redaction plan.  Tasks 13/14/15 replicate the same shape for other tools,
so the assertions here are load-bearing for the bulk-promotion wave.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

import pytest
from pydantic import BaseModel, ValidationError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.redaction import (
    MANIFEST,
    REDACTED_BLOB_SOURCE_PATH,
    Sensitive,
    SetSourceArgumentsModel,
    _redact_via_schema,
    _summarize_set_source_options,
    redact_guided_snapshot_storage_paths,
    redact_source_storage_path,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


def test_set_source_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["set_source"]
    assert entry.argument_model is SetSourceArgumentsModel
    assert entry.policy is None


def test_set_source_argument_model_validates_real_llm_shape() -> None:
    llm_args = {
        "plugin": "csv",
        "options": {"path": "/tmp/data.csv", "header": True},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    validated = SetSourceArgumentsModel.model_validate(llm_args)
    assert validated.plugin == "csv"
    assert validated.options == {"path": "/tmp/data.csv", "header": True}
    assert validated.on_success == "rows"
    assert validated.on_validation_failure == "discard"


def test_set_source_argument_model_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate({})


def test_set_source_argument_model_rejects_wrong_type() -> None:
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate(
            {
                "plugin": 42,
                "options": {},
                "on_success": "rows",
                "on_validation_failure": "discard",
            }
        )


def test_set_source_argument_model_rejects_extra_fields() -> None:
    """rev-2 M.1: extra='forbid' prevents argument_canonical/walker drift.

    Without this, a stray ``inline_blob`` or ``label`` field would be
    silently accepted by Pydantic but unrecorded in the walker schema —
    breaking the manifest/canonical-arguments parity invariant the
    adequacy guard relies on.
    """
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate(
            {
                "plugin": "csv",
                "options": {"path": "/tmp/x.csv"},
                "on_success": "rows",
                "on_validation_failure": "discard",
                "inline_blob": {"foo": "bar"},  # not a set_source field
            }
        )


def test_redact_substitutes_options_via_summarizer() -> None:
    """Sensitive[options] is replaced by the summarizer string at the top level.

    The summarizer returns canonical JSON of the options shape with scalar
    values redacted.
    Because Sensitive() substitutes the ENTIRE marked value, the top-level
    ``options`` slot in the redacted output is a string (the summarizer
    return), not a dict.  This is the load-bearing shape contract: the
    persistence boundary receives a scalar where a dict would otherwise
    sit.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "plugin": "csv",
        "options": {"path": "/internal/blob/path.csv", "blob_ref": "abc"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
    assert redacted["plugin"] == "csv"
    assert redacted["on_success"] == "rows"
    assert redacted["on_validation_failure"] == "discard"
    # Sensitive substitution: options is now the summarizer's str output.
    assert isinstance(redacted["options"], str)
    assert json.loads(redacted["options"]) == {
        "blob_ref": "<redacted-option-value>",
        "path": "<redacted-option-value>",
    }
    # The original internal path MUST NOT appear in the summary.
    assert "/internal/blob/path.csv" not in redacted["options"]
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "set_source", "shape": "type_driven"}]


def test_redact_source_options_summary_hides_paths_without_blob_ref() -> None:
    """Source option summaries must not preserve raw paths without blob_ref."""
    tel = NoopRedactionTelemetry()
    args = {
        "plugin": "csv",
        "options": {"path": "/tmp/data.csv"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
    assert isinstance(redacted["options"], str)
    assert "/tmp/data.csv" not in redacted["options"]


def test_redact_source_options_summary_hides_credential_values() -> None:
    """Credential-bearing source plugin option values must not survive."""
    tel = NoopRedactionTelemetry()
    raw_connection_string = "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=KEYVALUE;EndpointSuffix=core.windows.net"
    raw_sas_token = "sig=abcdefghijklmnopqrstuvwxyz1234567890"
    raw_client_secret = "client-secret-value"
    raw_path = "container/private/customer.csv"
    args = {
        "plugin": "azure_blob",
        "options": {
            "connection_string": raw_connection_string,
            "sas_token": raw_sas_token,
            "client_secret": raw_client_secret,
            "container": "private-container",
            "blob_path": raw_path,
        },
        "on_success": "rows",
        "on_validation_failure": "discard",
    }

    redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
    serialized = json.dumps(redacted, sort_keys=True)

    assert isinstance(redacted["options"], str)
    assert raw_connection_string not in serialized
    assert raw_sas_token not in serialized
    assert raw_client_secret not in serialized
    assert raw_path not in serialized


def test_redact_source_storage_path_masks_file_shape_when_blob_ref_present() -> None:
    """The ``file`` option is an equivalent blob storage-path carrier to ``path``.

    Blob ownership and fork code (blobs/service.py, sessions fork) treat both
    ``path`` and ``file`` as internal storage-path carriers, so a state with
    ``options={"blob_ref": ..., "file": <internal storage_path>}`` must have its
    ``file`` masked too — otherwise the internal blob path leaks through the
    redaction surface (elspeth-a7aa07b7ce).
    """
    state = {
        "source": {
            "plugin": "csv",
            "options": {"file": "/internal/blob/secret-storage.csv", "blob_ref": "abc"},
        }
    }
    redacted = redact_source_storage_path(state)
    assert redacted["source"]["options"]["file"] == REDACTED_BLOB_SOURCE_PATH
    assert "/internal/blob/secret-storage.csv" not in str(redacted)
    # Input is not mutated.
    assert state["source"]["options"]["file"] == "/internal/blob/secret-storage.csv"


def test_redact_source_storage_path_masks_path_shape_when_blob_ref_present() -> None:
    """Regression: the ``path`` shape stays redacted (elspeth-a7aa07b7ce)."""
    state = {"source": {"options": {"path": "/internal/blob/p.csv", "blob_ref": "abc"}}}
    redacted = redact_source_storage_path(state)
    assert redacted["source"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH


def test_redact_source_storage_path_leaves_manual_file_without_blob_ref() -> None:
    """A manual ``file`` path without ``blob_ref`` is not a blob carrier — unchanged."""
    state = {"source": {"options": {"file": "/tmp/user-data.csv"}}}
    redacted = redact_source_storage_path(state)
    assert redacted["source"]["options"]["file"] == "/tmp/user-data.csv"
    assert REDACTED_BLOB_SOURCE_PATH not in str(redacted)


def test_redact_guided_snapshot_masks_both_channels() -> None:
    """A guided blob source is committed via manual set_source (blob_ref stripped),
    so the committed source carries the real storage_path with NO blob_ref and the
    source-keyed redaction misses it. The co-located schema-8 reviewed source retained
    blob_ref; the helper uses it (no DB lookup) to mask BOTH the committed source
    path (channel 2) and the reviewed snapshot path (channel 3)."""
    real_path = "/home/u/elspeth/data/blobs/sess/abc_data.csv"
    sources = {"source": {"plugin": "csv", "options": {"path": real_path, "schema": {"mode": "observed"}}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "source",
                    "plugin": "csv",
                    "options": {
                        "path": real_path,
                        "blob_ref": "11111111-1111-4111-8111-111111111111",
                        "schema": {"mode": "observed"},
                    },
                }
            },
            "pending_source_intents": {},
        }
    }
    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)
    assert sources_out["source"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    reviewed = meta_out["guided_session"]["reviewed_sources"]["11111111-1111-4111-8111-111111111111"]
    assert reviewed["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    # blob_ref is retained — it is the redaction SIGNAL, not a sensitive value.
    assert reviewed["options"]["blob_ref"] == "11111111-1111-4111-8111-111111111111"
    assert real_path not in str(sources_out)
    assert real_path not in str(meta_out)
    # inputs are not mutated.
    assert sources["source"]["options"]["path"] == real_path
    original = composer_meta["guided_session"]["reviewed_sources"]["11111111-1111-4111-8111-111111111111"]
    assert original["options"]["path"] == real_path


def test_redact_guided_snapshot_leaves_operator_typed_source() -> None:
    """No blob_ref on the snapshot => the path is operator-typed, NOT a blob
    carrier => nothing is redacted on either channel."""
    sources = {"source": {"options": {"path": "/tmp/user.csv"}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "source",
                    "options": {"path": "/tmp/user.csv", "schema": {"mode": "observed"}},
                }
            },
            "pending_source_intents": {},
        }
    }
    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)
    assert sources_out["source"]["options"]["path"] == "/tmp/user.csv"
    reviewed = meta_out["guided_session"]["reviewed_sources"]["11111111-1111-4111-8111-111111111111"]
    assert reviewed["options"]["path"] == "/tmp/user.csv"
    assert REDACTED_BLOB_SOURCE_PATH not in str((sources_out, meta_out))


def test_redact_guided_snapshot_noop_for_freeform_state() -> None:
    """Freeform state (composer_meta is None, or has no guided_session snapshot) is
    returned unchanged — the helper only acts on a guided blob-backed snapshot."""
    sources = {"source": {"options": {"path": "/some/path.csv", "blob_ref": "x"}}}
    s1, m1 = redact_guided_snapshot_storage_paths(sources, None)
    assert s1 == sources
    assert m1 is None
    s2, m2 = redact_guided_snapshot_storage_paths(sources, {"repair_turns_used": 0})
    assert s2["source"]["options"]["path"] == "/some/path.csv"
    assert m2 == {"repair_turns_used": 0}


def test_redact_guided_snapshot_rejects_malformed_present_guided_session() -> None:
    sources = {"source": {"options": {"path": "/some/path.csv"}}}
    with pytest.raises(ValueError, match="guided_session must be a dict"):
        redact_guided_snapshot_storage_paths(sources, {"guided_session": "not-a-session"})


def test_redact_guided_snapshot_rejects_malformed_present_snapshot_options() -> None:
    sources = {"source": {"options": {"path": "/some/path.csv"}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {"11111111-1111-4111-8111-111111111111": {"name": "source", "options": "not-options"}},
            "pending_source_intents": {},
        }
    }
    with pytest.raises(ValueError, match=r"reviewed_sources.*options must be a dict"):
        redact_guided_snapshot_storage_paths(sources, composer_meta)


def test_redact_guided_snapshot_requires_schema8_pending_source_intents() -> None:
    with pytest.raises(KeyError, match="pending_source_intents"):
        redact_guided_snapshot_storage_paths(
            {},
            {"guided_session": {"reviewed_sources": {}}},
        )


def test_redact_guided_snapshot_requires_exact_pending_intent_options() -> None:
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {},
            "pending_source_intents": {
                "11111111-1111-4111-8111-111111111111": {"name": "incoming"},
            },
        }
    }
    with pytest.raises(KeyError, match="options"):
        redact_guided_snapshot_storage_paths({}, composer_meta)


def test_redact_guided_snapshot_rejects_malformed_source_when_blob_redaction_active() -> None:
    real_path = "/home/u/elspeth/data/blobs/sess/abc_data.csv"
    sources = {"source": {"options": "not-options"}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "source",
                    "options": {"path": real_path, "blob_ref": "11111111-1111-4111-8111-111111111111"},
                }
            },
            "pending_source_intents": {},
        }
    }
    with pytest.raises(ValueError, match=r"source\.options must be a dict"):
        redact_guided_snapshot_storage_paths(sources, composer_meta)


def test_redact_guided_snapshot_masks_file_carrier() -> None:
    """``file`` is an equivalent storage-path carrier to ``path`` (elspeth-a7aa07b7ce);
    the guided snapshot helper masks it on both channels too."""
    real = "/internal/blobs/sess/zzz_data.csv"
    sources = {"source": {"options": {"file": real}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "source",
                    "options": {"file": real, "blob_ref": "11111111-1111-4111-8111-111111111111"},
                }
            },
            "pending_source_intents": {},
        }
    }
    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)
    assert sources_out["source"]["options"]["file"] == REDACTED_BLOB_SOURCE_PATH
    reviewed = meta_out["guided_session"]["reviewed_sources"]["11111111-1111-4111-8111-111111111111"]
    assert reviewed["options"]["file"] == REDACTED_BLOB_SOURCE_PATH
    assert real not in str((sources_out, meta_out))


def test_redact_guided_snapshot_handles_plural_reviewed_sources_by_name() -> None:
    first_path = "/internal/blobs/first.csv"
    second_path = "/internal/blobs/second.csv"
    sources = {
        "first": {"options": {"path": first_path}},
        "second": {"options": {"path": second_path}},
    }
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "first",
                    "options": {"path": first_path, "blob_ref": "11111111-1111-4111-8111-111111111111"},
                },
                "22222222-2222-4222-8222-222222222222": {
                    "name": "second",
                    "options": {"path": second_path, "blob_ref": "22222222-2222-4222-8222-222222222222"},
                },
            },
            "pending_source_intents": {},
        }
    }

    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources_out["first"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    assert sources_out["second"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    reviewed = meta_out["guided_session"]["reviewed_sources"]
    assert reviewed["11111111-1111-4111-8111-111111111111"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    assert reviewed["22222222-2222-4222-8222-222222222222"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    assert first_path not in str((sources_out, meta_out))
    assert second_path not in str((sources_out, meta_out))


def test_redact_guided_snapshot_allows_two_reviewed_names_to_share_one_blob_path() -> None:
    shared_path = "/internal/blobs/shared.csv"
    sources = {
        "first": {"options": {"path": shared_path}},
        "second": {"options": {"path": shared_path}},
    }
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                stable_id: {
                    "name": name,
                    "options": {"path": shared_path, "blob_ref": "abc12300-0000-4000-8000-000000000000"},
                }
                for stable_id, name in (
                    ("11111111-1111-4111-8111-111111111111", "first"),
                    ("22222222-2222-4222-8222-222222222222", "second"),
                )
            },
            "pending_source_intents": {},
        }
    }

    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources_out is not None
    assert sources_out["first"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    assert sources_out["second"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
    assert shared_path not in str((sources_out, meta_out))


@pytest.mark.parametrize(
    "invalid_blob_ref",
    [None, "", 123, "98B1357D-5AAB-4FB3-85B4-5AD643912E84"],
    ids=["none", "empty", "wrong_type", "noncanonical_uuid"],
)
def test_redact_guided_snapshot_rejects_present_invalid_reviewed_blob_ref(invalid_blob_ref: object) -> None:
    stable_id = "11111111-1111-4111-8111-111111111111"
    live_path = "/internal/blobs/foreign.csv"
    sources = {"source": {"options": {"path": live_path}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                stable_id: {
                    "name": "source",
                    "options": {"path": live_path, "blob_ref": invalid_blob_ref},
                }
            },
            "pending_source_intents": {},
        }
    }

    with pytest.raises(AuditIntegrityError, match="canonical UUID"):
        redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources["source"]["options"]["path"] == live_path
    assert composer_meta["guided_session"]["reviewed_sources"][stable_id]["options"] == {
        "path": live_path,
        "blob_ref": invalid_blob_ref,
    }


def test_redact_guided_snapshot_rejects_reviewed_blob_ref_without_string_path_carrier() -> None:
    """A reviewed blob binding without its path cannot be mapped safely."""
    stable_id = "11111111-1111-4111-8111-111111111111"
    live_path = "/internal/blobs/foreign.csv"
    sources = {"source": {"options": {"path": live_path}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                stable_id: {
                    "name": "source",
                    "options": {"blob_ref": stable_id},
                }
            },
            "pending_source_intents": {},
        }
    }

    with pytest.raises(AuditIntegrityError, match="string path carrier"):
        redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources["source"]["options"]["path"] == live_path
    assert composer_meta["guided_session"]["reviewed_sources"][stable_id]["options"] == {"blob_ref": stable_id}


def test_redact_guided_snapshot_fails_closed_when_name_drift_hides_same_blob_path() -> None:
    real_path = "/internal/blobs/renamed.csv"
    sources = {"renamed": {"options": {"path": real_path}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {
                "11111111-1111-4111-8111-111111111111": {
                    "name": "original",
                    "options": {"path": real_path, "blob_ref": "11111111-1111-4111-8111-111111111111"},
                }
            },
            "pending_source_intents": {},
        }
    }

    with pytest.raises(AuditIntegrityError, match="guided blob source mapping"):
        redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources["renamed"]["options"]["path"] == real_path
    snapshot = composer_meta["guided_session"]["reviewed_sources"]["11111111-1111-4111-8111-111111111111"]
    assert snapshot["options"]["path"] == real_path


@pytest.mark.parametrize("carrier", ["path", "file"])
def test_redact_guided_pending_source_intent_blob_path_without_mutation(carrier: str) -> None:
    real_path = f"/internal/blobs/pending-{carrier}.csv"
    pending_id = "11111111-1111-4111-8111-111111111111"
    sources = {"current": {"options": {"path": "/operator/current.csv"}}}
    composer_meta = {
        "guided_session": {
            "reviewed_sources": {},
            "pending_source_intents": {
                pending_id: {
                    "name": "incoming",
                    "phase": "inspection_review",
                    "plugin": "csv",
                    "options": {carrier: real_path, "blob_ref": "11111111-1111-4111-8111-111111111111"},
                    "inspection_facts": None,
                    "observed_columns": [],
                    "sample_rows": [],
                }
            },
        }
    }

    sources_out, meta_out = redact_guided_snapshot_storage_paths(sources, composer_meta)

    assert sources_out == sources
    assert meta_out is not None
    pending = meta_out["guided_session"]["pending_source_intents"][pending_id]
    assert pending["options"][carrier] == REDACTED_BLOB_SOURCE_PATH
    assert pending["options"]["blob_ref"] == "11111111-1111-4111-8111-111111111111"
    assert real_path not in str(meta_out)
    assert composer_meta["guided_session"]["pending_source_intents"][pending_id]["options"][carrier] == real_path


def test_summarize_set_source_options_accepts_coerced_datetime() -> None:
    """Pin rev-3 A7: summarizer MUST NOT raise on reachable input values.

    Spec §9 RSK-03 requires the summarizer not raise on any reachable
    input value.  Pydantic 2.x can coerce string-like inputs to
    :class:`datetime` when the field accepts ``Any``; :func:`json.dumps`
    raises :class:`TypeError` on ``datetime`` unless ``default=str`` is
    supplied.  This test pins the ``default=str`` argument so a future
    refactor that removes it fails loudly here rather than silently
    violating RSK-03.
    """
    options = {"since": datetime(2026, 1, 1, tzinfo=UTC), "key": "v"}
    result = _summarize_set_source_options(options)
    assert isinstance(result, str)


_CANARY = "CANARY-SENSITIVE-PATH-DO-NOT-LEAK"


def test_serialization_boundary_canary_not_in_json_output() -> None:
    """Pin the Phase 3 cross-boundary integration contract (rev-2 BLOCKER_A).

    Phase 3 passes the result of :func:`redact_tool_call_arguments` through
    :func:`json.dumps` before writing to ``chat_messages.tool_calls``.  This
    test verifies the canary never survives that serialization — even
    though :func:`json.dumps` would otherwise re-emit the canary if it
    appeared anywhere in the dict. The source-option summarizer substitutes
    scalar option values before serialization, independent of blob_ref.
    """
    args = {
        "plugin": "csv",
        "options": {"path": _CANARY, "blob_ref": "abc123"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    result = redact_tool_call_arguments("set_source", args, telemetry=NoopRedactionTelemetry())
    serialized = json.dumps(result, sort_keys=True)
    assert _CANARY not in serialized, (
        "Sensitive canary value appeared in serialized output. "
        "Redaction did not remove it from the persistence path. "
        f"Serialized: {serialized!r}"
    )
    assert "options" in serialized  # key preserved, value redacted


# ---------------------------------------------------------------------------
# Task-7 boundary tests: no-summarizer → sentinel; nested-path → NotImplementedError
# ---------------------------------------------------------------------------


def test_redact_via_schema_substitutes_sentinel_for_sensitive_field_without_summarizer() -> None:
    """Task-7: Sensitive field with no summarizer receives REDACTED_SENSITIVE_NO_SUMMARIZER.

    The Task-4 tracer-bullet raised ``NotImplementedError`` here to force
    Task 8 to define the policy.  Task 7 defines the policy: substitute the
    no-summarizer sentinel rather than preserving the raw value.  Task 8 will
    generalise nested-path handling; this test pins the top-level case.
    """
    from elspeth.web.composer.redaction import REDACTED_SENSITIVE_NO_SUMMARIZER

    class _StubModel(BaseModel):
        secret: Annotated[str, Sensitive()]  # no summarizer

    validated = _StubModel.model_validate({"secret": "CANARY"})
    tel = NoopRedactionTelemetry()
    result = _redact_via_schema("stub_tool", validated, _StubModel, telemetry=tel)
    assert result["secret"] == REDACTED_SENSITIVE_NO_SUMMARIZER
    assert "CANARY" not in str(result.values())


def test_redact_via_schema_substitutes_nested_sensitive_path() -> None:
    """Task-8 generalisation: nested-path Sensitive field is substituted in-place.

    Task 4's tracer-bullet raised ``NotImplementedError`` for any path
    containing ``.``, ``[``, or ``{``; Task 8 supersedes that boundary by
    implementing the per-path substitute closure on ``TraversalNode``. The
    inner field's summarizer output replaces the value at the nested location
    while the surrounding structure is preserved.
    """

    class _InnerModel(BaseModel):
        inner_secret: Annotated[str, Sensitive(summarizer=lambda v: "<fixed-sum>")]
        public_field: str

    class _OuterModel(BaseModel):
        payload: _InnerModel

    validated = _OuterModel.model_validate({"payload": {"inner_secret": "RAW_SECRET", "public_field": "shown"}})
    tel = NoopRedactionTelemetry()
    result = _redact_via_schema("stub_tool", validated, _OuterModel, telemetry=tel)
    assert result["payload"]["inner_secret"] == "<fixed-sum>"
    assert result["payload"]["public_field"] == "shown"
    assert "RAW_SECRET" not in str(result)
