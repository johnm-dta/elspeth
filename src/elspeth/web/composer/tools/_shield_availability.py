"""Deployment-availability of the authorized prompt-injection shield (B-vs-C signal)."""

from __future__ import annotations

from elspeth.web.catalog.schemas import PluginSecretRequirement
from elspeth.web.composer.tools._availability import secret_unavailable_message
from elspeth.web.composer.tools._common import ToolContext

# The authorized prompt-injection shield and the secret ref that gates its use.
# Mirrors azure_prompt_shield's discovery_secret_requirements
# ({"api_key": ("AZURE_CONTENT_SAFETY_KEY",)}); the catalog promotes that to a
# PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)).
_AZURE_PROMPT_SHIELD_NAME = "azure_prompt_shield"
_AZURE_PROMPT_SHIELD_REQUIREMENT = PluginSecretRequirement(
    field="api_key",
    candidates=("AZURE_CONTENT_SAFETY_KEY",),
)


def azure_prompt_shield_available(context: ToolContext) -> bool:
    """Return True iff an authorized prompt-injection shield is usable here.

    Reuses the existing discovery surface
    (:func:`elspeth.web.composer.tools._availability.secret_unavailable_message`)
    keyed on the shield-specific ``AZURE_CONTENT_SAFETY_KEY`` candidate (NOT a
    coarse any-secret check). Returns the FAIL-SAFE ``False`` (State C) when
    availability is undeterminable — no secret service or no user. Because the
    requirement carries non-empty ``candidates``, ``_requirement_satisfied``
    takes the candidate branch (``has_ref(user_id, "AZURE_CONTENT_SAFETY_KEY")``).
    """

    if context.secret_service is None or context.user_id is None:
        return False
    message = secret_unavailable_message(
        plugin_type="transform",
        plugin_name=_AZURE_PROMPT_SHIELD_NAME,
        requirements=(_AZURE_PROMPT_SHIELD_REQUIREMENT,),
        context=context,
    )
    return message is None
