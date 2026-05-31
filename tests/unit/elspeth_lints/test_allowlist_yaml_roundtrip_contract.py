"""Write→load round-trip contract for the allowlist YAML writer (C2-5 / C2-6).

The justify command writes Tier-1 audit data to per-module YAML files via
``_yaml_inline_scalar`` (for scalar fields) and a hand-rolled emitter (for
block-scalar fields). The loader reads those same files through
``yaml.safe_load`` and constructs ``AllowlistEntry`` instances. The contract
between writer and loader is that *every* string the operator supplies must
round-trip back as the same Python string — no silent type coercion, no
truncation, no smuggled ``None``.

The C2-5 fire was that PyYAML's YAML 1.1 implicit resolvers coerce a list of
sentinel scalars (``yes``, ``no``, ``on``, ``off``, ``null``, ``~``, plus
their three case variants, plus number-like and timestamp-like strings) from
``str`` to ``bool`` / ``None`` / ``int`` / ``float`` / ``date`` on reload. The
writer emitted those values unquoted, so an audit row written with
``--owner yes`` was unrecoverable as the string ``"yes"`` — it either
crashed the loader (its type guard rejects non-str) or, worse on a different
loader, silently became ``True``.

C2-6 is the structural gap: no test ever ran writer-output through loader
input. This file IS that test surface — every property here exercises the
full write→load contract, not just one half.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth_lints.core.allowlist import AuditReviewVerdict, JudgeVerdict, load_allowlist
from elspeth_lints.core.cli import (
    _YAML11_IMPLICIT_NON_STR_RESOLVERS,
    _build_audit_review_text,
    _build_yaml_entry_text,
    _value_resolves_to_non_str,
    _yaml_inline_scalar,
    main,
)
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL

# ---------- helpers ----------


_SYNTHETIC_SOURCE = '''\
"""Synthetic module used in YAML round-trip tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        return payload.get("name", "anonymous")
'''


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
    return root, target


def _build_allowlist_dir(tmp_path: Path) -> Path:
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    return allowlist_dir


def _mock_openrouter_completion(*, verdict: str, rationale: str) -> MagicMock:
    import json as _json

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message = MagicMock()
    completion.choices[0].message.content = _json.dumps(
        {"verdict": verdict, "rationale": rationale, "confidence": 0.91, "should_use_decorator": None}
    )
    completion.model = DEFAULT_JUDGE_MODEL
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 4000
    completion.usage.completion_tokens = 50
    completion.usage.total_tokens = 4050
    completion.usage.prompt_tokens_details = MagicMock()
    completion.usage.prompt_tokens_details.cached_tokens = 0
    return completion


@contextmanager
def _mock_judge_call(*, verdict: str, rationale: str) -> Iterator[MagicMock]:
    fake_completion = _mock_openrouter_completion(verdict=verdict, rationale=rationale)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test-key",
                "ELSPETH_JUDGE_METADATA_HMAC_KEY": "test-judge-metadata-hmac-key-2026-05-24",
            },
            clear=False,
        ),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


def _emit_and_reload_scalar(value: str) -> Any:
    """Round-trip ``value`` through the writer's inline scalar emitter and
    PyYAML's safe_load. Returns whatever Python value the loader produces.

    Equivalence test: ``_emit_and_reload_scalar(s) == s`` iff the writer's
    quoting decision was correct for ``s``.
    """
    emitted = _yaml_inline_scalar(value)
    reloaded = yaml.safe_load(io.StringIO(f"x: {emitted}\n"))
    assert isinstance(reloaded, dict) and set(reloaded.keys()) == {"x"}
    return reloaded["x"]


# ---------- _value_resolves_to_non_str primitive ----------


# The full PyYAML YAML 1.1 sentinel set (bool + null), lifted verbatim.
# Pinned here so a regression in the resolver constants is loud.
_BOOL_SENTINELS = (
    "yes",
    "Yes",
    "YES",
    "no",
    "No",
    "NO",
    "true",
    "True",
    "TRUE",
    "false",
    "False",
    "FALSE",
    "on",
    "On",
    "ON",
    "off",
    "Off",
    "OFF",
)
_NULL_SENTINELS = ("~", "null", "Null", "NULL", "")
# Numeric sentinels that PyYAML's actual regexes match (case-sensitive,
# strict on form). Empirically verified against ``yaml.safe_load`` rather
# than what one might naively expect:
#   - octal must be ``0[0-7_]+`` (no ``0o`` prefix; ``0o17`` reloads as str)
#   - hex must be ``0x`` lowercase (``0X1a`` reloads as str)
#   - float exponents require an explicit sign: ``1e10`` is str, ``1e+10`` is str,
#     ``1.5e-3`` is float (mantissa has ``.`` so plain-int-with-exponent path
#     doesn't apply); ``1.5e10`` is also str
_NUMERIC_SENTINELS = (
    "0",
    "42",
    "-1",
    "+7",
    "0b101",
    "017",
    "0x1A",
    "1_000",
    "3.14",
    "-2.5",
    ".5",
    "5.",
    "1.5e-3",
    ".inf",
    "-.inf",
    ".Inf",
    ".INF",
    ".nan",
    ".NaN",
    ".NAN",
)
_TIMESTAMP_SENTINELS = ("2024-01-01",)
_MERGE_VALUE_SENTINELS = ("<<", "=")
_PLAIN_SCALAR_SYNTAX_SENTINELS = ("-",)


@pytest.mark.parametrize("value", _BOOL_SENTINELS + _NULL_SENTINELS + _NUMERIC_SENTINELS + _TIMESTAMP_SENTINELS + _MERGE_VALUE_SENTINELS)
def test_value_resolves_to_non_str_detects_pyyaml_sentinels(value: str) -> None:
    """Every PyYAML 1.1 implicit-resolver sentinel must be flagged.

    The resolver-coverage matrix (bool / null / int / float / timestamp /
    merge / value) is the authoritative source: this test pins each tag
    family to its public predicate so a future PyYAML update or a typo in
    our regex constants is visible.
    """
    assert _value_resolves_to_non_str(value), (
        f"_value_resolves_to_non_str({value!r}) returned False but PyYAML safe_load would coerce this scalar to a non-str type"
    )


@pytest.mark.parametrize(
    "value",
    [
        "agent",
        "binding-test-agent",
        DEFAULT_JUDGE_MODEL,
        "Yes please",
        "yEs",
        "tRue",
        "nULl",
        "YeS",
        "42x",
        "v1.2.3",
        "abc123",
        "a",
        "A",
    ],
)
def test_value_resolves_to_non_str_passes_audit_safe_strings(value: str) -> None:
    """Strings that are NOT in the implicit-resolver match set must pass.

    Mixed-case-three-letter forms like ``yEs`` / ``tRue`` deliberately fall
    outside PyYAML's case-sensitive regex (which only accepts the three
    canonical variants lower / Title / UPPER per word). Over-quoting them
    would bloat audit YAML for no integrity gain.
    """
    assert not _value_resolves_to_non_str(value), (
        f"_value_resolves_to_non_str({value!r}) returned True but PyYAML safe_load preserves this scalar as a string"
    )


# ---------- writer-level quoting ----------


@pytest.mark.parametrize("sentinel", _BOOL_SENTINELS + _NULL_SENTINELS[:-1])  # exclude "" which has its own test
def test_writer_quotes_sentinel_so_loader_preserves_string(sentinel: str) -> None:
    """C2-5 historical bug: bare-emitting these strings reloads as bool/None.

    This is the regression pin for the original incident. If
    ``_yaml_inline_scalar`` ever stops quoting one of these, the operator-
    supplied audit field silently changes type on reload.
    """
    assert _emit_and_reload_scalar(sentinel) == sentinel


@pytest.mark.parametrize("number_like", _NUMERIC_SENTINELS + _TIMESTAMP_SENTINELS + _MERGE_VALUE_SENTINELS)
def test_writer_quotes_non_bool_implicit_resolvers(number_like: str) -> None:
    """Same contract for non-bool/null implicit resolvers.

    A ``--reason 42`` reload must produce the string ``"42"``, not the
    integer ``42`` — the audit field's type is part of the legal record.
    """
    assert _emit_and_reload_scalar(number_like) == number_like


@pytest.mark.parametrize("indicator", _PLAIN_SCALAR_SYNTAX_SENTINELS)
def test_writer_quotes_plain_scalar_syntax_indicators(indicator: str) -> None:
    """C2-6 regression: a single bare hyphen is YAML syntax, not data.

    ``x: -`` raises ``ScannerError`` because PyYAML reads the hyphen as a
    block-sequence indicator. The writer must quote the operator-supplied
    string before it reaches the loader.
    """
    assert _yaml_inline_scalar(indicator) == f"'{indicator}'"
    assert _emit_and_reload_scalar(indicator) == indicator


def test_writer_emits_empty_string_safely() -> None:
    """The empty string is matched by PyYAML's null resolver (regex alt ``| ``).

    An unquoted ``key:`` reloads as ``None``. The writer must emit ``''``.
    """
    assert _yaml_inline_scalar("") == "''"
    assert _emit_and_reload_scalar("") == ""


# ---------- property-based round-trip contract ----------


# The writer's accept set is narrower than PyYAML's reader-accept set: it
# additionally rejects LF/CR/NEL because single-quoted flow scalars fold
# those to a single space at load time. The Hypothesis blacklist below is
# therefore the union of "PyYAML reader rejects" and "single-quoted flow
# folds" — every character the writer refuses.
_WRITER_REJECTED_CHARS = [
    chr(c)
    for c in range(0x110000)
    if not (chr(c) == "\t" or 0x20 <= c <= 0x7E or 0xA0 <= c <= 0xD7FF or 0xE000 <= c <= 0xFFFD or 0x10000 <= c <= 0x10FFFF)
]


# Hypothesis strategy: any string PyYAML can represent in a single-quoted
# scalar AND that our writer accepts. The blacklist below precisely matches
# the writer's rejection set so the property tests round-trip identity,
# not writer-side ValueErrors.
_yaml_safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=["Cs"],
        blacklist_characters="".join(_WRITER_REJECTED_CHARS),
    ),
    min_size=0,
    max_size=120,
)


@given(value=_yaml_safe_text)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_property_inline_scalar_round_trips_through_safe_load(value: str) -> None:
    """For any audit-safe string, write→load is identity.

    This is the C2-6 structural fix: the property catches the whole class
    of writer/loader contract drift, not just the C2-5 incident. New
    PyYAML versions adding implicit resolvers, new operator-supplied edge
    cases (e.g. an owner name with embedded ``:``), or a future refactor
    of ``_yaml_inline_scalar`` that loses a quoting case will all be
    caught here.
    """
    reloaded = _emit_and_reload_scalar(value)
    assert reloaded == value, f"round-trip drift: input {value!r} emitted as {_yaml_inline_scalar(value)!r} reloaded as {reloaded!r}"


# Strategy that biases toward sentinel-adjacent strings: PyYAML's resolvers
# are case-sensitive on a small set, so mutations of that set are the
# highest-value adversarial inputs.
_sentinel_adjacent = st.sampled_from(_BOOL_SENTINELS + _NULL_SENTINELS + _NUMERIC_SENTINELS)


@given(value=_sentinel_adjacent)
@settings(max_examples=100)
def test_property_sentinel_set_round_trips(value: str) -> None:
    """Tighter property on the sentinel set itself — exhaustive in practice."""
    assert _emit_and_reload_scalar(value) == value


# ---------- negative: writer refuses C0 control characters ----------


@pytest.mark.parametrize(
    "bad_char",
    # Sample across PyYAML-reader-rejected: C0 controls (except TAB/LF/CR),
    # DEL, C1 controls (except NEL \x85), noncharacters \xFFFE/\xFFFF.
    [chr(c) for c in (0x00, 0x01, 0x07, 0x08, 0x0B, 0x0C, 0x0E, 0x1F, 0x7F, 0x80, 0x84, 0x86, 0x9F, 0xFFFE, 0xFFFF)],
)
def test_writer_refuses_reader_unprintable_chars(bad_char: str) -> None:
    """A YAML scalar cannot contain characters outside PyYAML's reader-
    accepted set. Rather than silently producing un-loadable YAML (which
    surfaces as a load-time ReaderError with a stack trace that doesn't
    point at the bug), the writer crashes at write time with a message
    that names the offending codepoint.

    This is the deliberate failure mode for pathological inputs in a
    Tier-1 field — sanitisation is the caller's responsibility, not the
    audit writer's.
    """
    with pytest.raises(ValueError, match="non-printable character"):
        _yaml_inline_scalar(f"prefix{bad_char}suffix")


@pytest.mark.parametrize("bad_char", ["\n", "\r", "\x85"])
def test_writer_refuses_line_break_chars_in_inline_scalar(bad_char: str) -> None:
    """LF, CR, and NEL are accepted by PyYAML's reader but folded to a
    single space inside single-quoted flow scalars (YAML 1.1 §6.3.2).
    The bytes survive read but the value does not round-trip; emitting
    such a value would corrupt the audit record on reload.

    The multi-paragraph audit fields (``reason``, ``judge_rationale``,
    ``safety``) are emitted as block scalars and don't reach this helper.
    """
    with pytest.raises(ValueError, match="line-break character"):
        _yaml_inline_scalar(f"prefix{bad_char}suffix")


@pytest.mark.parametrize("ok_char", ["\t", "\xa0", "é", "中", "𝄞"])
def test_writer_accepts_inline_safe_chars(ok_char: str) -> None:
    """The complement of the rejection set: TAB, NBSP, and arbitrary
    Unicode in the BMP+SMP ranges PyYAML accepts and preserves through
    a single-quoted flow scalar. Over-rejection is its own audit hazard
    (operator wants ``--owner "agent\tname"`` and gets a hard refusal).
    """
    out = _yaml_inline_scalar(f"prefix{ok_char}suffix")
    reloaded = yaml.safe_load(io.StringIO(f"x: {out}\n"))
    assert reloaded == {"x": f"prefix{ok_char}suffix"}


# ---------- end-to-end: CLI write path → loader round-trip ----------


@pytest.mark.parametrize(
    "sentinel",
    ["yes", "no", "true", "false", "null", "on", "off", "Yes", "YES", "True", "NULL", "42", "2024-01-01", ""],
)
def test_justify_cli_with_sentinel_owner_round_trips(tmp_path: Path, sentinel: str) -> None:
    """The original C2-5 fire path: operator passes a sentinel as ``--owner``.

    Pre-fix: the writer emitted ``owner: yes`` and the loader either
    crashed or (on a more permissive loader) silently produced
    ``owner=True``. Post-fix: the writer single-quotes the value and the
    loader recovers the original string.

    Punctuation-only sentinels such as ``~`` and ``-`` remain covered by
    helper-level YAML quoting tests, but the CLI must now reject them as
    non-substantive audit anchors before writing an unloadable allowlist row.
    The empty-string case exercises the upstream argparse guard; we skip the
    CLI end-to-end for that case but the bare-writer round-trip is still
    pinned above.
    """
    if sentinel == "":
        pytest.skip("argparse --owner type=_non_empty_string rejects empty input upstream")

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        sentinel,
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        assert main(argv) == 0

    target_yaml = allowlist_dir / "plugins.yaml"
    with patch.dict(os.environ, {"ELSPETH_JUDGE_METADATA_HMAC_KEY": "test-judge-metadata-hmac-key-2026-05-24"}, clear=False):
        loaded = load_allowlist(target_yaml, valid_rule_ids={"R1"}, source_root=root)
    assert len(loaded.entries) == 1
    entry = loaded.entries[0]
    assert entry.owner == sentinel
    assert isinstance(entry.owner, str)


# ---------- _build_yaml_entry_text: all helper-routed fields are safe ----------


@pytest.mark.parametrize("sentinel", ["yes", "null", "42", "on", "False", "~", "-"])
def test_build_yaml_entry_text_quotes_all_helper_routed_fields(sentinel: str) -> None:
    """Every field that flows through ``_yaml_inline_scalar`` must round-trip.

    Today the operator/source-derived helper-routed fields are
    ``owner``, ``judge_model``, ``judge_policy_hash``, ``scope_fingerprint``,
    and ``ast_path``. The generated ``judge_metadata_signature`` also routes through the
    helper and is asserted separately. This test sets each operator/
    source-derived field to the same sentinel and asserts the full entry
    round-trips — pinning the fix at the call-site level, not just the
    helper.
    """
    with patch.dict(os.environ, {"ELSPETH_JUDGE_METADATA_HMAC_KEY": "test-judge-metadata-hmac-key-2026-05-24"}, clear=False):
        text = _build_yaml_entry_text(
            key="plugins/widget.py:R1:Widget:lookup:fp=" + sentinel,
            owner=sentinel,
            reason="tier-3 boundary",
            verdict=JudgeVerdict.ACCEPTED,
            recorded_at=datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC),
            model_id=sentinel,
            policy_hash=sentinel,
            judge_rationale="judge agrees",
            judge_confidence=0.5,
            scope_fingerprint=sentinel,
            ast_path=sentinel,
        )
    full_yaml = "allow_hits:\n" + text
    loaded = yaml.safe_load(io.StringIO(full_yaml))
    assert isinstance(loaded, dict)
    entries = loaded["allow_hits"]
    assert len(entries) == 1
    e = entries[0]
    assert e["owner"] == sentinel and isinstance(e["owner"], str)
    assert e["judge_model"] == sentinel and isinstance(e["judge_model"], str)
    assert e["judge_policy_hash"] == sentinel and isinstance(e["judge_policy_hash"], str)
    assert e["scope_fingerprint"] == sentinel and isinstance(e["scope_fingerprint"], str)
    assert e["ast_path"] == sentinel and isinstance(e["ast_path"], str)
    assert isinstance(e["judge_metadata_signature"], str)
    assert e["judge_metadata_signature"].startswith("hmac-sha256:v2:")


def test_build_audit_review_text_quotes_all_helper_routed_fields() -> None:
    """Nested audit-review scalar fields must also survive YAML reload."""
    reviewed_at = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)
    text = _build_audit_review_text(
        verdict=AuditReviewVerdict.JUDGE_ACCEPTED_WRONG,
        reviewer="yes",
        reviewed_at=reviewed_at,
        rationale="Later reproduction showed the accepted suppression hid a real tier leak.",
    )

    full_yaml = f"allow_hits:\n- key: plugins/widget.py:R1:Widget:lookup:fp=abc\n  owner: operator\n  reason: tier-3 boundary\n{text}"
    loaded = yaml.safe_load(io.StringIO(full_yaml))
    assert isinstance(loaded, dict)
    review = loaded["allow_hits"][0]["audit_review"]
    assert review["reviewer"] == "yes" and isinstance(review["reviewer"], str)
    assert review["reviewed_at"] == reviewed_at.isoformat()
    assert isinstance(review["reviewed_at"], str)


# ---------- inventory: future-proofing the helper-call surface ----------


def test_yaml_inline_scalar_call_sites_are_inventoried() -> None:
    """Pin the set of fields that route through ``_yaml_inline_scalar``.

    This is the C2-6 "test the contract, not the symptom" insurance: if
    a future contributor adds a new field to ``_build_yaml_entry_text``
    that bypasses ``_yaml_inline_scalar`` (and therefore the sentinel
    quoting), this test fails and forces the contributor to either route
    it through the helper or add a written rationale.

    The check is byte-grep over ``cli.py`` for the helper's name. The
    expected count is 10:
      6 entry-level scalars (owner, judge_model, judge_policy_hash,
        scope_fingerprint, ast_path, judge_metadata_signature)
      2 redaction-record fields per nested ``judge_excerpt_redactions``
        entry (pattern, redacted_hash) — these route through the same
        helper because a maliciously-crafted source file could in
        principle produce a hash collision against a YAML 1.1 sentinel
        token, and ``pattern_name`` is a fixed vocabulary today but a
        future addition could trigger the same.
      2 nested ``audit_review`` scalars (reviewer, reviewed_at)
    Update this number deliberately when the inventory legitimately
    grows.
    """
    cli_path = Path(__file__).resolve().parents[3] / "elspeth-lints" / "src" / "elspeth_lints" / "core" / "cli.py"
    text = cli_path.read_text(encoding="utf-8")
    # ``def _yaml_inline_scalar(`` matches the definition line; everything
    # else is a call site.
    total_token_matches = text.count("_yaml_inline_scalar(")
    definition_count = text.count("def _yaml_inline_scalar(")
    call_count = total_token_matches - definition_count
    assert call_count == 10, (
        f"expected 10 _yaml_inline_scalar(...) call sites in cli.py, found "
        f"{call_count}; if a new field was added, also extend "
        "the helper-routed field round-trip tests above."
    )


def test_pyyaml_resolver_set_is_exhaustive() -> None:
    """The implicit-resolver constants in cli.py must cover every PyYAML
    safe_load implicit resolver registration.

    PyYAML registers seven resolvers in ``yaml/resolver.py``: bool, float,
    int, merge, null, timestamp, value. (The eighth — ``yaml`` — is
    documentation-only and unreachable.) Our ``_YAML11_IMPLICIT_NON_STR_RESOLVERS``
    tuple must have the same count; a future PyYAML release adding a
    resolver would silently widen the coercion surface, and this test is
    the canary.
    """
    from yaml.resolver import Resolver

    # Resolver's implicit-resolver registry is keyed by first-char; flatten
    # to a set of unique tags actually registered.
    tags: set[str] = set()
    for entries in Resolver.yaml_implicit_resolvers.values():
        for tag, _regex in entries:
            tags.add(tag)
    # Exclude the documentation-only ``tag:yaml.org,2002:yaml`` (the
    # registration's first-chars are ``!&*`` which cannot start a plain
    # scalar, so it never fires).
    coercing_tags = {t for t in tags if t != "tag:yaml.org,2002:yaml"}
    assert len(coercing_tags) == len(_YAML11_IMPLICIT_NON_STR_RESOLVERS), (
        f"PyYAML registers {len(coercing_tags)} coercing implicit resolvers "
        f"({sorted(coercing_tags)}); _YAML11_IMPLICIT_NON_STR_RESOLVERS has "
        f"{len(_YAML11_IMPLICIT_NON_STR_RESOLVERS)} regexes. If PyYAML added "
        f"a new resolver, lift its regex into cli.py and extend the tuple."
    )
