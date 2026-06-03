"""Draft a persona-flavoured opening prompt for a panel-cohort scenario.

Given an example's settings.yaml and a persona spec, calls Haiku 4.5 to
produce the natural-language opening message that persona would type to
ask the composer to build the workflow this YAML describes.

Why this exists (not just hand-author every prompt):
  * 33 examples x 2-4 personas = 66-132 opening prompts. Hand-authoring is
    16-33 hours of writing work the operator confirmed is too costly.
  * LLM drafting from settings.yaml + persona spec produces in-character
    asks at ~$0.001 per draft. Operator spot-checks smoke cohort,
    samples broader cohort.

Why a separate module from `judge_persona.sh`:
  * The drafter runs once per (example, persona) cell at scenario-build
    time. The judge runs once per scenario at scoring time. Different
    lifecycle, different inputs, different prompts.
  * Reuses the API call plumbing pattern (provider auto-detect from env,
    JSON parsing with fence-stripping, retry logic) but on string output
    not JSON output — the drafter returns plain text.

Usage as a module:

    from evals.lib.prompt_drafter import draft_opening_prompt
    msg = draft_opening_prompt(persona_spec_text, settings_yaml_text)

CLI for spot-checking a single cell:

    python3 -m evals.lib.prompt_drafter \\
        evals/composer-harness/personas/p1_compliance.md \\
        examples/boolean_routing/settings.yaml
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are drafting an OPENING MESSAGE that a specific user persona \
would type to a pipeline-composer chat assistant to ask for a workflow. You \
have two inputs:

1. PERSONA SPEC — the bio, cognitive style, linguistic constraints, \
competence ceiling, and concession rule for the persona. Internalise this. \
The opening message must sound exactly like this person.

2. PIPELINE SETTINGS YAML — the technical specification of the workflow they \
want. The persona should describe the OUTCOME this YAML achieves, not the \
plumbing.

Rules:

- Output ONLY the opening message text. No preamble, no JSON, no quotes \
around it, no "here is the message:" framing. Just the message verbatim, \
ready to be pasted into a chat box.

- Stay in the persona's voice. Use the persona's vocabulary, sentence \
length, and tone. Use the persona's stack-name shibboleths if applicable \
(Marcus says HubSpot/Zapier; Sarah names her field of research; Linda \
uses compliance-domain nouns; Dev names ELSPETH primitives).

- Describe what the persona WANTS, not what the YAML CONTAINS. The persona \
does not read settings.yaml; they describe the business / research / \
compliance goal that makes the workflow useful.

- Respect the persona's competence ceiling. An amateur persona MUST NOT \
say "schema", "transform", "sink", "type_coerce", "route_to_sink", or any \
other architectural terminology — even if the YAML literally contains \
those plugin kinds. The persona describes outcomes; the composer's job is \
to translate to plumbing.

- A length appropriate to the persona. Linda is verbose and hedge-laden \
(2-4 paragraphs). Sarah is narrative (2-4 paragraphs framed as a research \
question). Marcus is short and imperative (1-3 short paragraphs). Dev is \
terse and prescriptive (1-3 sentences, may include backticked field names \
and primitive names — for Dev only).

- Reference the input data shape only as the persona would. They might \
say "I have a CSV from HubSpot with leads", "we have 80,000 survey \
responses in a spreadsheet", "the audit log dump is in Excel", etc. They \
do NOT describe the schema as "id: int, message: str, approved: str" \
(that's plumbing).

- If the YAML uses a specific input file path, the persona may reference \
the data informally (their HubSpot export, their survey CSV, etc.) but \
SHOULD NOT echo the literal filesystem path.

- Avoid putting words in the composer's mouth. The persona is asking for \
something; they don't pre-specify the answer.
"""


def _build_user_prompt(persona_spec: str, settings_yaml: str, example_name: str) -> str:
    return f"""PERSONA SPEC
=============
{persona_spec}

PIPELINE SETTINGS YAML (for `{example_name}` example)
=============
```yaml
{settings_yaml}
```

Now draft the opening message this persona would type to ask the composer \
to build this workflow. Output the message text only — no preamble, no \
quotes, no JSON wrapper."""


# ---------------------------------------------------------------------------
# API call (mirrors judge_persona.sh plumbing — kept in sync intentionally)
# ---------------------------------------------------------------------------


def _call_openrouter(api_key: str, system: str, user: str) -> str:
    body = {
        "model": "anthropic/claude-haiku-4.5",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,  # higher than judge — drafting wants persona variation
        "max_tokens": 1500,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://elspeth.foundryside.dev",
            "X-Title": "elspeth-panel-cohort-drafter",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def _call_anthropic(api_key: str, system: str, user: str) -> str:
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1500,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    blocks = d.get("content", [])
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts)


def _detect_provider() -> tuple[str, str] | None:
    """Return (provider, api_key) tuple, or None if no key is set."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter", os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    return None


def _strip_chrome(text: str) -> str:
    """Remove markdown fences, quotes, and "Here is the message:" preambles.

    Haiku is well-behaved at temperature=0.7 with a strict instruction but
    occasionally adds a leading fence or a quoted-line pattern. This is a
    last-line defense; if the model ignored the instruction, this gives us
    a clean string.
    """
    text = text.strip()
    # Markdown fences
    fence_match = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Wrapping double-quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("“") and text.endswith("”")):
        text = text[1:-1].strip()
    # Common preamble patterns
    for prefix in ("Here is the message:", "Opening message:", "Message:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DrafterError(Exception):
    """Raised when the LLM draft cannot be obtained."""


def draft_opening_prompt(
    persona_spec: str,
    settings_yaml: str,
    example_name: str,
    *,
    max_tries: int = 3,
) -> str:
    """Single-shot draft an opening message in persona voice.

    Raises:
        DrafterError: no API key in env, or API failed after retries.
    """
    detected = _detect_provider()
    if detected is None:
        raise DrafterError("no OPENROUTER_API_KEY or ANTHROPIC_API_KEY in env — cannot draft opening prompt without an LLM call")
    provider, api_key = detected
    system = SYSTEM_PROMPT
    user = _build_user_prompt(persona_spec, settings_yaml, example_name)

    last_err: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            if provider == "openrouter":
                raw = _call_openrouter(api_key, system, user)
            else:
                raw = _call_anthropic(api_key, system, user)
            return _strip_chrome(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            sys.stderr.write(f"prompt_drafter: API attempt {attempt} failed: {exc}\n")
            if attempt < max_tries:
                time.sleep(2**attempt)
    raise DrafterError(f"API failed after {max_tries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] in {"-h", "--help"}:
        sys.stderr.write("usage: python -m evals.lib.prompt_drafter <persona_spec_md> <settings_yaml>\n")
        return 64
    persona_path = pathlib.Path(argv[1])
    settings_path = pathlib.Path(argv[2])
    if not persona_path.is_file():
        sys.stderr.write(f"persona spec not found: {persona_path}\n")
        return 67
    if not settings_path.is_file():
        sys.stderr.write(f"settings yaml not found: {settings_path}\n")
        return 67
    persona_spec = persona_path.read_text()
    settings_yaml = settings_path.read_text()
    example_name = settings_path.parent.name
    try:
        msg = draft_opening_prompt(persona_spec, settings_yaml, example_name)
    except DrafterError as exc:
        sys.stderr.write(f"drafter error: {exc}\n")
        return 71
    sys.stdout.write(f"{msg}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
