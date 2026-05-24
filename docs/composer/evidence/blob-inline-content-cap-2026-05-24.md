# Blob Inline Content Cap Evidence

**Date:** 2026-05-24
**Scope:** `elspeth-fdebcaa79a` P1 evidence for widened `blob_ref` / `inline_content`
**Branch:** RC5.2

## Summary

The widened `blob_ref` inline-content feature needs numeric caps that are tight
enough to bound runtime preflight and broad enough for legitimate long-form
public configuration content. The measured local data supports:

- **Per-ref upper cap:** `256 KiB`
- **Aggregate per-config cap:** `1 MiB`
- **Soft lower threshold:** warn on composer-authored refs below `256 B`; do
  not hard-reject direct YAML or API references solely for being small

The upper cap is intentionally larger than the current persisted composition
state, because the existing composer prompt corpus already contains
prompt-like artifacts up to `203,370 B`. A `64 KiB` cap would reject those
current internal artifacts if operators used the blob store for the same class
of content.

## Data Sources

Local SQLite state:

| Source | Count |
| --- | ---: |
| `data/sessions.db` `composition_states` | 44 |
| `data/sessions.db` `blobs` | 10 |
| `data/sessions.db` `chat_messages` | 363 |
| `data/runs/audit.db` `calls` | 178 |
| `data/runs/audit.db` `operations` | 47 |
| `data/runs/audit.db` `runs` | 17 |

Persisted composition string fields were sampled from `source`, `nodes`, and
`outputs` JSON using keys that represent likely inline-content candidates:
`system_prompt`, `prompt_template`, `query`, `sql`, `regex`, `pattern`,
`template`, `allowlist`, `denylist`, `public_cert`, `certificate`, `content`,
`scraping_reason`, and `abuse_contact`.

| Field | Samples | Min bytes | Max bytes | Avg bytes |
| --- | ---: | ---: | ---: | ---: |
| `prompt_template` | 42 | 242 | 649 | 394.0 |
| `scraping_reason` | 37 | 19 | 79 | 55.1 |
| `system_prompt` | 4 | 42 | 42 | 42.0 |
| `abuse_contact` | 37 | 14 | 25 | 18.8 |
| `content` | 7 | 4 | 7 | 4.9 |

Overall persisted composition candidates:

| Samples | Min bytes | Max bytes | Avg bytes | P50 bytes | P90 bytes | P99 bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 127 | 4 | 649 | 153.4 | 58 | 412 | 649 |

Current ready text/data blobs are toy-scale and do not justify a production
cap by themselves:

| MIME type | Status | Samples | Min bytes | Max bytes | Avg bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| `text/csv` | `ready` | 3 | 81 | 117 | 104.7 |
| `text/plain` | `ready` | 7 | 5 | 80 | 66.9 |

Overall ready text/data blob candidates:

| Samples | Min bytes | Max bytes | Avg bytes | P50 bytes | P90 bytes | P99 bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5 | 117 | 78.2 | 77 | 116 | 117 |

Examples and test YAML contain short inline prompt/template fields:

| Samples | Min bytes | Max bytes | Avg bytes | P50 bytes | P90 bytes | P99 bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 23 | 571 | 187.7 | 186 | 262 | 571 |

Largest sampled YAML fields:

| Bytes | Key | File |
| ---: | --- | --- |
| 571 | `prompt_template` | `examples/chroma_rag_qa/settings.yaml` |
| 422 | `system_prompt` | `examples/openrouter_multi_query_assessment/settings_overflow.yaml` |
| 262 | `prompt_template` | `examples/openrouter_sentiment/settings_pooled.yaml` |
| 262 | `prompt_template` | `examples/openrouter_sentiment/settings.yaml` |
| 262 | `prompt_template` | `examples/chaosllm_sentiment/settings.yaml` |

Prompt-like repository artifacts were sampled from:

- `src/elspeth/web/composer/skills`
- `docs/composer/evidence`
- `scripts/skill_rgr/candidates`

| Samples | Min bytes | Max bytes | Avg bytes | P50 bytes | P90 bytes | P99 bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | 1,142 | 203,370 | 27,878.4 | 16,081 | 71,593 | 203,370 |

Largest prompt-like artifacts:

| Bytes | File |
| ---: | --- |
| 203,370 | `src/elspeth/web/composer/skills/pipeline_composer.md` |
| 86,145 | `docs/composer/evidence/composer-briefing-2026-05-03.md` |
| 71,593 | `scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md` |
| 40,720 | `docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md` |
| 40,147 | `docs/composer/evidence/composer-progress-1a-rc5-2-merge-analysis-2026-05-12.md` |

## Decision Support

### Upper Cap

`256 KiB` per inline-content reference covers the largest sampled internal
prompt-like artifact (`203,370 B`) with about 29 percent headroom while keeping
each resolved value bounded. It is materially safer than an unbounded config
content read and less arbitrary than `64 KiB`, which would fail current
prompt-like artifacts.

### Aggregate Cap

`1 MiB` aggregate per config allows four refs near the per-ref cap, or more
typical smaller refs, while bounding preflight bytes under the existing
worker-to-event-loop bridge budget. Runtime implementation should enforce both
the per-ref and aggregate caps before bytes flow into plugin instantiation.

### Lower Threshold

The measured local state contains many legitimate short strings. A hard lower
bound would make direct-YAML use brittle for short regexes, allowlist fragments,
public contact strings, or public certificate snippets. The composer tool should
warn on refs below `256 B` to discourage "everything is a blob" usage, but the
validator should not reject solely because the content is small.

## Verification Inputs

This evidence used read-only SQLite, filesystem size, and YAML parse queries.
The local blob corpus is intentionally treated as weak cap evidence because it
is small and toy-scale; the repo prompt corpus is the decisive signal for the
upper bound.
