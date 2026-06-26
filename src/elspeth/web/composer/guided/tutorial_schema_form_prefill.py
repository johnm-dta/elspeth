"""Tutorial-only ``plugin_options`` schema_form knob prefill (passive walk).

The tutorial is the NORMAL guided flow surfaced as a *prefilled, locked worked
example*: the learner presses the buttons and types nothing. After the dynamic
source-from-chat commits an inline ``json`` source, the wizard re-renders the
source ``plugin_options`` schema_form
(:func:`build_step_1_schema_form_turn_from_resolved`). That form declares
``on_validation_failure`` REQUIRED with no default (``SourceDataConfig``), but
the chat-committed source never persisted it, so the field surfaces
required-empty and the passive learner's "Continue" submit stays disabled. For a
TUTORIAL session we inject the honest worked-example value so the form enables
without typing.

Discipline
----------
* TUTORIAL-gated by the CALLER (the emitter's ``tutorial`` flag, set from
  ``guided.profile == TUTORIAL_PROFILE`` at each route site). A non-tutorial
  schema_form is BYTE-UNCHANGED: ``tutorial=False`` short-circuits before any
  mutation.
* MINIMAL + honest: only knobs in :data:`_TUTORIAL_SCHEMA_FORM_DEFAULTS` are
  ever filled, and only when the plugin declares them required-with-no-default
  AND they are absent from ``prefilled`` (inject-if-absent — a committed source
  that DID persist the value rides through untouched). A required knob with no
  obvious safe worked-example value is deliberately NOT here — it must surface
  as a real blocker rather than be silently fabricated.
* ``on_validation_failure="discard"`` is the always-valid explicit-drop
  sentinel (``engine/orchestrator/validation.py``): the synthetic sample pages
  are valid-by-construction, so a row never fails validation and this route is
  never exercised at runtime. A real *production* pipeline routes
  non-conformant rows to a quarantine sink instead (which needs a second sink in
  the pipeline) — surfaced to the learner via the tutorial teaching copy.
* Mutates ``payload['prefilled']`` IN PLACE, BEFORE the caller hashes the turn,
  so the audit hash reflects exactly what the learner saw and resubmits (the
  Turn dict itself is not persisted — only its hash enters the audit trail).

Trust tier: L3 web layer. Pure function — no I/O, no clock, no uuid.
"""

from __future__ import annotations

from elspeth.web.catalog.knob_schema import SchemaFormPayload

# The required-no-default plugin knobs a TUTORIAL schema_form may safely prefill,
# and the honest worked-example value for each. Keep this MINIMAL: a required
# knob with no obvious safe worked-example value is NOT here — it must surface as
# a real blocker rather than be silently fabricated.
#
# The source's ``schema`` knob is deliberately NOT here: the emitters already
# hardcode ``prefilled["schema"] = {"mode": "observed"}`` and (since the
# knob-lowering alias fix) the knob lowers under that same ``schema`` name, so
# it is satisfied for every user without tutorial-specific help.
_TUTORIAL_SCHEMA_FORM_DEFAULTS: dict[str, object] = {
    "on_validation_failure": "discard",
}


def prefill_tutorial_schema_form_knobs(
    payload: SchemaFormPayload,
    *,
    tutorial: bool,
) -> None:
    """Inject worked-example defaults into a TUTORIAL ``plugin_options`` form.

    Mutates ``payload['prefilled']`` IN PLACE. For each knob in
    :data:`_TUTORIAL_SCHEMA_FORM_DEFAULTS` that the plugin declares
    required-with-no-default and that is absent from ``prefilled``, sets the
    honest worked-example value — so the passive learner's submit enables
    without typing.

    No-op (``prefilled`` byte-unchanged) when ``tutorial`` is False, the payload
    is not ``plugin_options``, the knob is already prefilled, or the plugin does
    not declare the knob required-with-no-default. The CALLER gates ``tutorial``
    on the tutorial profile; this function does not read the profile.
    """
    if not tutorial:
        return
    if payload.get("mode") != "plugin_options":
        return
    required_no_default = {field["name"] for field in payload["knobs"]["fields"] if field["required"] and "default" not in field}
    prefilled = payload["prefilled"]
    for name, value in _TUTORIAL_SCHEMA_FORM_DEFAULTS.items():
        if name in required_no_default and name not in prefilled:
            prefilled[name] = value
