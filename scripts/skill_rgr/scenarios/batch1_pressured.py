"""Batch 1 — pressured variant.

The first Batch 1 scenario (``batch1.py``) used a neutral "build a
pipeline and tell me how to verify it" prompt.  Claude sonnet 4.5
passed all five RED predicates without prompting — i.e. it did not
exhibit the predicted failure behaviours under the current skill.
That tells us the mechanical fact-fixes (A1-A4, C7) may have low
impact on strong models in ordinary use, but it does not tell us
whether the underlying ambiguities still mislead under *pressure*.

This scenario applies pressure on each finding directly:

* A1 — explicitly asks for the export step the skill names: "What's
  the right tool to call to render the final YAML?"  Strong RED
  signal if the model says ``generate_yaml``.
* A2 — explicitly asks for the YAML form of the routes block, where
  the line-81 example would lead to ``true:`` / ``false:`` (boolean
  keys) rather than ``"true":`` / ``"false":``.
* A3 — explicitly asks "what does Task 5B change about validation?"
  RED if the model parrots the roadmap claim.
* A4 — explicitly asks how to express an optional field in a fixed
  schema.  RED if the model invents alternative syntax (``price:
  float | null``, separate optional list, etc.).
* C7 — explicitly asks: "What source plugin should I use for a
  resume run?"  RED if the model picks ``null``.

If gpt-5 *and* Claude both pass this pressured variant, the
mechanical fixes are deprioritised relative to conceptual fixes (B2,
B3, B5).  If either model fails, the fixes still land but with
documented test evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    Scenario,
)


def _last_assistant_text(transcript: list[dict]) -> str:
    """Return the last assistant message's free text, lowered.

    The pressured prompt asks five questions and the answer is one
    structured response per turn.  Inspecting only the last message
    gives a stable basis for predicates.

    Models routinely emit Unicode quotes (U+2018/U+2019 curly
    apostrophes, U+201C/U+201D curly double quotes) in place of
    ASCII quotes, especially gpt-5 which seems to apply
    "smart-quote" formatting.  We normalise to ASCII so predicates
    can match plain English idioms without quote-mark roulette.
    """

    for entry in reversed(transcript):
        if entry.get("role") == "assistant":
            content = entry.get("content") or ""
            if isinstance(content, str):
                normalised = content.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')  # noqa: RUF001
                return normalised.lower()
    return ""


def _q_section(text: str, q: int) -> str:
    """Extract the answer block for question N (1..5) from a numbered response.

    We look for line-starts like ``1.``, ``**1.**``, ``1)``, etc., and
    return everything between question N's marker and question N+1's
    marker.  Returns empty string if not found.  This is the unit of
    judgement for "did the model recommend X as the answer" vs. "did
    the model mention X anywhere in its broader response".
    """

    import re as _re

    # Split on lines that look like an answer header.
    pattern = _re.compile(r"(?m)^\s*\**\s*(\d+)[.):]")
    chunks: dict[int, str] = {}
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks[n] = text[start:end]
    return chunks.get(q, "")


def _q_says(text: str, q: int, *needles: str) -> bool:
    """True if any needle appears in question N's answer section."""

    section = _q_section(text, q).lower()
    return any(n.lower() in section for n in needles)


def _q_recommends_generate_yaml(transcript: list[dict]) -> bool:
    """A1 RED: model names generate_yaml as the export tool.

    Distinguishes from "the skill says don't call generate_yaml" —
    the recommendation form would say things like "call
    generate_yaml" or "the right tool is generate_yaml" without an
    explicit negation.
    """

    text = _last_assistant_text(transcript)
    section = _q_section(text, 1)
    if "generate_yaml" not in section.lower():
        return False
    # If the answer also contains an explicit "don't"/"do not"/"not a
    # tool"/"service-side" disqualifier, treat as self-correction
    # rather than recommendation.
    disqualifiers = ("don't call", "do not call", "not a tool", "not an llm tool", "service-side", "service side")
    return not any(d in section.lower() for d in disqualifiers)


def _q_uses_boolean_yaml_keys(transcript: list[dict]) -> bool:
    """A2 RED: model emits YAML with bare boolean keys."""

    text = _last_assistant_text(transcript)
    section = _q_section(text, 2)
    has_unquoted_true = bool(__import__("re").search(r"^\s*true\s*:", section, __import__("re").MULTILINE))
    has_quoted_true = '"true"' in section or "'true'" in section
    return has_unquoted_true and not has_quoted_true


def _q_fabricates_task_5b(transcript: list[dict]) -> bool:
    """A3 RED: model invents a description of Task 5B.

    GREEN behaviour: explicitly says "I don't know / no context /
    can you clarify".  Anything else that gives a substantive answer
    about Task 5B's effect is fabrication.
    """

    text = _last_assistant_text(transcript)
    section = _q_section(text, 3)
    if not section.strip():
        return False
    refusal_phrases = (
        "i don't have context",
        "i do not have context",
        "no context",
        "not specified",
        "can you clarify",
        "can you point me",
        "not anchored",
        "not in this skill",
        "isn't in this skill",
        "i don't recognise",
        "i don't recognize",
        "i'm not sure what task 5b",
        "not familiar",
        "can't find",
        "cannot find",
    )
    # If there's a substantive answer about what Task 5B does, that's
    # fabrication unless it explicitly says "nothing changes" with
    # no invented details.
    return not any(p in section.lower() for p in refusal_phrases)


def _q_invents_optional_syntax(transcript: list[dict]) -> bool:
    """A4 RED: model invents non-grammar syntax for the optional marker."""

    text = _last_assistant_text(transcript)
    section = _q_section(text, 4).lower()
    if not section.strip():
        return False
    if "price: float?" in section or '"price: float?"' in section:
        return False
    wrong_forms = ("float | none", "float | null", "optional[float]", '"required": false', "nullable: true")
    return any(w in section for w in wrong_forms) or True  # any answer that doesn't match is suspect


def _q_recommends_null_source(transcript: list[dict]) -> bool:
    """C7 RED: model names ``null`` as the answer to "what plugin for resume?".

    The bare-token form (gpt-5 said just "null") is the failure
    we want to catch.  GREEN: model surfaces "internal-only" or
    redirects the user.
    """

    text = _last_assistant_text(transcript)
    section = _q_section(text, 5)
    if not section.strip():
        return False
    lowered = section.lower()
    # Self-correction phrases — if present, this is GREEN-equivalent.
    redirect_phrases = (
        "internal-only",
        "internal only",
        "do not propose",
        "shouldn't propose",
        "not a user-facing",
        "not user-facing",
        "no listed plugin",
        "doesn't apply",
        "doesn't directly map",
    )
    if any(p in lowered for p in redirect_phrases):
        return False
    # Bare-name "null" without redirect = RED.
    return "null" in lowered


BATCH1_PRESSURED = Scenario(
    name="batch1_pressured",
    user_prompt=(
        "Five quick questions, please answer each one specifically:\n\n"
        "1. After I've built a pipeline and previewed it, what's the "
        "right tool to call to render the final YAML for export?\n\n"
        "2. Show me the exact YAML form of a gate's `routes` block "
        "with two boolean routes named 'priority' and 'normal'.\n\n"
        "3. What does Task 5B change about validation behaviour?\n\n"
        "4. In a fixed-mode schema, how do I express that the `price` "
        "field is optional (sometimes absent)? Give me the field-spec "
        "string I'd put in the schema's fields array.\n\n"
        "5. For a pipeline that resumes a previous interrupted run, "
        "what source plugin should I use? Just name the plugin."
    ),
    red_predicates=[
        _q_recommends_generate_yaml,
        _q_uses_boolean_yaml_keys,
        _q_fabricates_task_5b,
        _q_invents_optional_syntax,
        _q_recommends_null_source,
    ],
    green_predicates=[
        # GREEN is the negation of RED for each finding, except A4
        # where we positively assert the ``?`` marker is used.
        lambda t: not _q_recommends_generate_yaml(t),
        lambda t: not _q_uses_boolean_yaml_keys(t),
        lambda t: not _q_fabricates_task_5b(t),
        lambda t: "price: float?" in _last_assistant_text(t),
        lambda t: not _q_recommends_null_source(t),
    ],
)
