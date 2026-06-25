"""Tutorial-only recipe-slot prefill for the passive guided walk (p4 Task 8b).

The tutorial is the NORMAL guided flow surfaced as a *prefilled, locked worked
example*: the learner presses the buttons and types nothing. At STEP_2.5 the
deterministic matcher (:func:`match_recipe`) returns a :class:`RecipeMatch`
whose ``unsatisfied_slots`` (``model``, ``api_key_secret``, ``abuse_contact``,
``scraping_reason`` for the web-scrape recipe) are the slots a normal operator
fills via the ``recipe_offer`` form. A passive learner can't type them, so for a
TUTORIAL session we MOVE those slots out of ``unsatisfied_slots`` into ``slots``
with honest worked-example values. The emitter then projects them as
``prefilled`` (read-only) with an empty ``knobs`` set, so the frontend's
``recipe_decision`` widget enables "Apply recipe" (nothing required-empty) and
resubmits the prefilled values verbatim — satisfying the accept-seam binding
check (``_prefilled_recipe_slot_mismatches``).

Discipline
----------
* TUTORIAL-gated by the CALLER (``guided.profile == TUTORIAL_PROFILE``). A
  non-tutorial recipe_offer is UNCHANGED: the four slots still surface as
  ``unsatisfied`` for the operator to fill.
* Honest values, sourced from config — never fabricated:
    - ``model`` rides ``WebSettings.composer_model`` (the deployment's
      configured composer LLM). Note ``tutorial_model_id`` is a compound cache
      KEY, not a model id — it is deliberately NOT used here.
    - ``api_key_secret`` is the deployment's configured LLM secret-REF NAME
      (a name like ``OPENROUTER_API_KEY``, never a credential value), paired to
      the recipe's provider via the same ``_PROVIDER_REQUIRED_ENV_KEYS`` map the
      composer uses, and verified present in
      ``WebSettings.server_secret_allowlist``.
    - ``abuse_contact`` / ``scraping_reason`` are fixed demo values — honest
      because the tutorial "scrapes" our own app-served LOCAL synthetic pages,
      not a third party.
* Fail-closed: if the model or the LLM secret-ref cannot be sourced from config,
  raise :class:`InvariantError` (HTTP 500). The tutorial cannot run without an
  LLM; we never fabricate a model or secret.
* The web_scrape ``allowed_hosts`` SSRF allowlist is NOT handled here. It is set
  server-side at the STEP_2.5 *accept* seam (Task 8a), never a learner-visible
  prefilled slot.
"""

from __future__ import annotations

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.recipes import get_recipe
from elspeth.web.config import WebSettings

# Operator-approved fixed worked-example values (honest: the tutorial scrapes our
# own app-served synthetic sample pages, not a third party).
TUTORIAL_ABUSE_CONTACT = "noreply@demo.com"
TUTORIAL_SCRAPING_REASON = "ELSPETH guided-creation tutorial — scraping app-served synthetic sample pages (demo)."

# The slot names this prefill can fill, and where each value comes from. Only the
# names present in a match's ``unsatisfied_slots`` are actually prefilled, so a
# recipe that does not surface ``abuse_contact``/``scraping_reason`` (e.g. a
# non-web-scrape LLM recipe) simply gets ``model``/``api_key_secret``.
_TUTORIAL_FILLABLE_SLOTS = frozenset({"model", "api_key_secret", "abuse_contact", "scraping_reason"})


def _require_composer_model(settings: WebSettings) -> str:
    """The deployment's configured composer model, or fail closed."""
    model = settings.composer_model
    if not model:
        raise InvariantError(
            "tutorial recipe prefill: WebSettings.composer_model is empty; "
            "the tutorial worked example cannot name an LLM model (fail-closed)."
        )
    return model


def _tutorial_recipe_provider(recipe_match: RecipeMatch) -> str:
    """The LLM provider the recipe wires (its ``provider`` slot default).

    The web-scrape recipe leaves ``provider`` optional with a registry default of
    ``"openrouter"``; the matcher never resolves it, so it is read from the
    recipe spec rather than from ``recipe_match.slots``.
    """
    recipe = get_recipe(recipe_match.recipe_name)
    if recipe is None:
        raise InvariantError(f"tutorial recipe prefill: recipe {recipe_match.recipe_name!r} is not registered (fail-closed).")
    provider_spec = recipe.slots.get("provider")
    if provider_spec is None or not provider_spec.default:
        raise InvariantError(
            f"tutorial recipe prefill: recipe {recipe_match.recipe_name!r} has no provider default; "
            "cannot derive the LLM secret-ref (fail-closed)."
        )
    return str(provider_spec.default)


def _resolve_api_key_secret(*, provider: str, settings: WebSettings) -> str:
    """The deployment's configured LLM secret-REF NAME for ``provider``.

    Pairs the provider to its credential-env name via the same
    ``_PROVIDER_REQUIRED_ENV_KEYS`` map the composer uses for provider-auth
    readiness (imported lazily to keep this module free of the heavy service
    import at load time), then verifies the deployment has allowlisted that ref.
    Returns the ref NAME only — never a credential value.
    """
    # Single source of truth for provider -> credential-env-name. Lazy import:
    # service.py pulls the provider stack; importing it at module load would
    # widen this helper's import surface for no benefit.
    from elspeth.web.composer.service import _PROVIDER_REQUIRED_ENV_KEYS

    required = _PROVIDER_REQUIRED_ENV_KEYS.get(provider)
    if not required:
        raise InvariantError(f"tutorial recipe prefill: no known LLM secret-ref for provider {provider!r} (fail-closed).")
    secret_ref = required[0]
    if secret_ref not in settings.server_secret_allowlist:
        raise InvariantError(
            f"tutorial recipe prefill: LLM secret-ref {secret_ref!r} (provider {provider!r}) is not in "
            "WebSettings.server_secret_allowlist; the tutorial cannot wire an LLM credential (fail-closed)."
        )
    return secret_ref


def prefill_tutorial_recipe_slots(
    *,
    recipe_match: RecipeMatch,
    settings: WebSettings,
) -> RecipeMatch:
    """Return a TUTORIAL-prefilled copy of ``recipe_match``.

    Moves every tutorial-fillable slot present in ``recipe_match.unsatisfied_slots``
    into ``slots`` with its honest worked-example value, leaving any other
    unsatisfied slot untouched. The caller MUST gate this on the tutorial
    profile; this function does not check the profile.

    Returns ``recipe_match`` unchanged when no tutorial-fillable slot is
    unsatisfied (nothing to do). Raises :class:`InvariantError` if a needed
    value cannot be sourced from config (fail-closed).
    """
    to_fill = [name for name in recipe_match.unsatisfied_slots if name in _TUTORIAL_FILLABLE_SLOTS]
    if not to_fill:
        return recipe_match

    moved: dict[str, object] = {}
    for name in to_fill:
        if name == "model":
            moved[name] = _require_composer_model(settings)
        elif name == "api_key_secret":
            provider = _tutorial_recipe_provider(recipe_match)
            moved[name] = _resolve_api_key_secret(provider=provider, settings=settings)
        elif name == "abuse_contact":
            moved[name] = TUTORIAL_ABUSE_CONTACT
        elif name == "scraping_reason":
            moved[name] = TUTORIAL_SCRAPING_REASON

    new_slots = {**dict(recipe_match.slots), **moved}
    new_unsatisfied = {name: spec for name, spec in recipe_match.unsatisfied_slots.items() if name not in moved}
    return RecipeMatch(
        recipe_name=recipe_match.recipe_name,
        slots=new_slots,
        unsatisfied_slots=new_unsatisfied,
    )
