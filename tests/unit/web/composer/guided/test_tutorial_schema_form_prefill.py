"""Tutorial-only schema_form knob prefill (passive guided walk).

After the dynamic source-from-chat commits an inline ``json`` source, the
wizard re-renders the source ``plugin_options`` schema_form
(:func:`build_step_1_schema_form_turn_from_resolved`). That form declares
``on_validation_failure`` REQUIRED with no default (``SourceDataConfig``), but
the chat-committed source never persisted it, so the field surfaces
required-empty and the passive learner's "Continue" submit stays disabled. For
a TUTORIAL session the server injects the honest worked-example value so the
form enables without typing.

These tests pin the pure helper directly (inject-if-absent + required-no-default
guard), then the emitter integration: a tutorial turn gains
``on_validation_failure="discard"``; a non-tutorial turn is BYTE-UNCHANGED (the
load-bearing non-regression invariant).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from elspeth.web.catalog.knob_schema import SchemaFormPayload
from elspeth.web.composer.guided.emitters import build_step_1_schema_form_turn_from_resolved
from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.tutorial_schema_form_prefill import (
    prefill_tutorial_schema_form_knobs,
)

_REQUIRED_OVF_FIELD: dict[str, Any] = {
    "name": "on_validation_failure",
    "label": "On validation failure",
    "kind": "text",
    "required": True,
    "nullable": False,
}


def _plugin_options_payload(*, fields: list[dict[str, Any]], prefilled: dict[str, Any]) -> SchemaFormPayload:
    payload: dict[str, Any] = {
        "mode": "plugin_options",
        "plugin": "json",
        "knobs": {"fields": fields},
        "prefilled": dict(prefilled),
    }
    return cast(SchemaFormPayload, payload)


class TestPrefillHelper:
    def test_injects_discard_when_required_and_absent(self) -> None:
        payload = _plugin_options_payload(
            fields=[_REQUIRED_OVF_FIELD],
            prefilled={"schema": {"mode": "observed"}, "path": "x", "blob_ref": "b"},
        )
        prefill_tutorial_schema_form_knobs(payload, tutorial=True)
        assert payload["prefilled"]["on_validation_failure"] == "discard"

    def test_noop_when_not_tutorial(self) -> None:
        before = {"schema": {"mode": "observed"}, "path": "x"}
        payload = _plugin_options_payload(fields=[_REQUIRED_OVF_FIELD], prefilled=before)
        prefill_tutorial_schema_form_knobs(payload, tutorial=False)
        assert payload["prefilled"] == before  # byte-unchanged

    def test_noop_when_already_prefilled(self) -> None:
        # A committed source that DID persist on_validation_failure must not be
        # overwritten — inject-if-absent only.
        payload = _plugin_options_payload(
            fields=[_REQUIRED_OVF_FIELD],
            prefilled={"on_validation_failure": "quarantine_sink"},
        )
        prefill_tutorial_schema_form_knobs(payload, tutorial=True)
        assert payload["prefilled"]["on_validation_failure"] == "quarantine_sink"

    def test_noop_when_knob_has_default(self) -> None:
        # If the plugin declares on_validation_failure with a default (not
        # required-empty), the form is not blocked, so the tutorial does NOT
        # inject — it only unblocks genuinely required-empty knobs.
        optional_field = {**_REQUIRED_OVF_FIELD, "required": False, "default": "discard"}
        payload = _plugin_options_payload(fields=[optional_field], prefilled={})
        prefill_tutorial_schema_form_knobs(payload, tutorial=True)
        assert "on_validation_failure" not in payload["prefilled"]

    def test_noop_when_knob_absent_from_schema(self) -> None:
        # A plugin whose schema has no on_validation_failure knob (e.g. a sink)
        # is untouched — we never invent a knob the plugin does not declare.
        path_only = {"name": "path", "label": "Path", "kind": "text", "required": True, "nullable": False}
        payload = _plugin_options_payload(fields=[path_only], prefilled={})
        prefill_tutorial_schema_form_knobs(payload, tutorial=True)
        assert payload["prefilled"] == {}

    def test_noop_for_recipe_decision_mode(self) -> None:
        payload: dict[str, Any] = {
            "mode": "recipe_decision",
            "knobs": {"fields": [_REQUIRED_OVF_FIELD]},
            "prefilled": {},
            "recipe_context": {"recipe_name": "r", "description": "d", "alternatives": []},
        }
        prefill_tutorial_schema_form_knobs(cast(SchemaFormPayload, payload), tutorial=True)
        assert payload["prefilled"] == {}


class _SourceCatalog:
    """Catalog stub whose json source schema declares the three required-no-default
    knobs the live json source lowers: ``schema_config`` (json-value), ``path``,
    and ``on_validation_failure`` (mirrors ``SourceDataConfig`` / ``DataPluginConfig``)."""

    def get_schema(self, plugin_type: str, plugin_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            json_schema={"properties": {}},
            knob_schema={
                "fields": [
                    {
                        "name": "schema_config",
                        "label": "Schema",
                        "kind": "json-value",
                        "required": True,
                        "nullable": False,
                    },
                    {"name": "path", "label": "Path", "kind": "text", "required": True, "nullable": False},
                    {
                        "name": "on_validation_failure",
                        "label": "On validation failure",
                        "kind": "text",
                        "required": True,
                        "nullable": False,
                    },
                ]
            },
        )


def _committed_json_source() -> SourceResolved:
    # Mirrors the LIVE chat-committed source (run8 ground truth): options carry
    # schema/path/blob_ref but NOT on_validation_failure (never persisted at the
    # chat-apply commit), so the re-rendered form is required-empty for it.
    return SourceResolved(
        plugin="json",
        options={"path": "inline", "blob_ref": "blob-1"},
        observed_columns=("url",),
        sample_rows=({"url": "http://127.0.0.1/tutorial-site/p1.html"},),
    )


class TestEmitterIntegration:
    def test_tutorial_turn_injects_discard(self) -> None:
        turn = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog(), tutorial=True)
        assert turn["payload"]["prefilled"]["on_validation_failure"] == "discard"

    def test_tutorial_satisfies_every_required_no_default_knob(self) -> None:
        # The unblock invariant: after the tutorial prefill, EVERY required-no-default
        # knob the json source declares (schema_config, path, on_validation_failure)
        # is present in prefilled — so the frontend's canSubmit() enables Continue
        # without the passive learner typing. schema_config rides the same canonical
        # observed-mode schema the emitter hardcodes.
        turn = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog(), tutorial=True)
        payload = turn["payload"]
        prefilled = payload["prefilled"]
        assert prefilled["schema_config"] == {"mode": "observed"}
        assert prefilled["on_validation_failure"] == "discard"
        required_no_default = {f["name"] for f in payload["knobs"]["fields"] if f["required"] and "default" not in f}
        assert required_no_default <= set(prefilled), f"unsatisfied required-no-default knobs: {required_no_default - set(prefilled)}"

    def test_non_tutorial_turn_is_byte_unchanged(self) -> None:
        # The default tutorial=False MUST leave the non-tutorial form identical
        # to the pre-change emitter output. Pin the exact prefilled dict.
        default_turn = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog())
        explicit_turn = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog(), tutorial=False)
        assert default_turn["payload"]["prefilled"] == {
            "schema": {"mode": "observed"},
            "path": "inline",
            "blob_ref": "blob-1",
        }
        assert default_turn == explicit_turn

    def test_injected_dict_is_isolated_from_shared_default(self) -> None:
        # A dict worked-example value (schema_config) must be deep-copied on inject,
        # so mutating one turn's prefilled cannot corrupt the shared module constant
        # (or a sibling turn). Pins the deepcopy guard.
        first = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog(), tutorial=True)
        first_schema = first["payload"]["prefilled"]["schema_config"]
        assert isinstance(first_schema, dict)
        first_schema["mode"] = "MUTATED"
        second = build_step_1_schema_form_turn_from_resolved(_committed_json_source(), _SourceCatalog(), tutorial=True)
        assert second["payload"]["prefilled"]["schema_config"] == {"mode": "observed"}
