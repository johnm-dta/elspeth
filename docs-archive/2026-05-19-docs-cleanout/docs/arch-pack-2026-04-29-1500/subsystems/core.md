# `core/` — L1 Foundation

**Layer:** L1 — outbound subset of `{contracts}` only.
**Size:** 49 files, 20,791 LOC.
**Composite:** triggered by all three heuristics (≥4 sub-pkgs, ≥10k LOC, ≥20 files).
**Quality score:** **4 / 5**.

---

## §1 Responsibility

L1 foundation primitives:

- **Landscape audit DB** — recorder + four named repositories.
- **DAG construction & validation** — `ExecutionGraph` and helpers.
- **Configuration** — Dynaconf + Pydantic v2 settings.
- **Canonical-JSON hashing** — RFC 8785 / JCS for hash-stable serialisation.
- **Payload store** — separates large blobs from audit tables.
- **Retention** — purge policies for old payload data.
- **Rate limiting** — `RateLimiter` and `NoOpLimiter` protocol pair.
- **Security** — secret loaders, configuration secrets.
- **Expression parser** — for triggers, commencement gates, depends-on
  expressions.

---

## §2 Internal sub-areas

| Sub-area | Purpose |
|----------|---------|
| `landscape/` | Audit DB recorder + 4 repositories (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`); 20 schema tables |
| `dag/` | DAG construction, validation, `ExecutionGraph` |
| `checkpoint/` | Checkpoint manager + compatibility |
| `rate_limit/` | Rate-limiter protocol + implementations |
| `retention/` | Payload retention policies |
| `security/` | Secret loaders, config secrets |
| Top-level modules | `config.py`, `expression_parser.py`, canonical JSON, payload store, templates |

---

## §3 Dependencies

| Direction | Edges |
|-----------|-------|
| **Outbound** | `{contracts}` only (per layer model + clean enforcer) |
| **Inbound** | `{engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` |

---

## §4 Findings

### C1 — `core/config.py` cohesion is unverified · **Medium**

Single-file Pydantic settings (2,227 LOC) holding 12+ child dataclasses
across checkpoint, concurrency, database, landscape, payload-store,
rate-limit, retry, secrets, sinks, sources, transforms.

Pydantic settings concentrate for cross-validation reasons, so the LOC
is partially explained by the framework. Whether 2,227 LOC is
**appropriately concentrated** or **accreted by addition** is open.

**Recommendation:** [R5](../07-improvement-roadmap.md#r5) — per-file
deep-dive paired with an architecture-pack proposal (split or keep).

### C2 — `core/dag/graph.py` blast radius · **Medium**

`ExecutionGraph` (1,968 LOC) is consumed by:

- Every executor in `engine/`.
- `web/composer/_semantic_validator.py`.
- `web/execution/validation.py`.
- `core/checkpoint/manager.py` and `core/checkpoint/compatibility.py`.
- Indirectly by every plugin via the schema-contract validation flow.

Any change to `ExecutionGraph` semantics is system-wide. The test files
exist (`test_graph.py`, `test_graph_validation.py`) but their assertion
density relative to the file's behavioural surface is unverified.

**Recommendation:** L3 deep-dive on the public-contract test surface;
consider a pinned snapshot of `ExecutionGraph` semantic invariants.

### C3 — Audit table count divergence · **Medium**

`core/landscape/` contains **20** schema tables; the institutional
documentation records **21**. Documentation correctness, not
architecture; `ARCHITECTURE.md` is one major iteration behind on the
Landscape schema.

**Recommendation:** [R10](../07-improvement-roadmap.md#r10).

### C4 — `core/secrets.py` placement · **Low**

`core/secrets.py` (124 LOC, runtime resolver) lives at the `core/`
root, while `core/security/{secret_loader,config_secrets}.py` (529 LOC
combined) live in the sub-package. Responsibility-cut question between
L0 contracts and L1 core post-ADR-006.

**Recommendation:** consider relocating to `core/security/` for
namespace consistency, or document the rationale for the split inline.

---

## §5 Strengths

### The Landscape facade pattern is real, not aspirational

`landscape/__init__.py` re-exports exactly `RecorderFactory` and the
four named repositories:

- `DataFlowRepository`
- `ExecutionRepository`
- `QueryRepository`
- `RunLifecycleRepository`

Repositories are **not** re-exported through `core/__init__.py`.
Callers can only reach the audit DB through `RecorderFactory`. The
encapsulation is mechanically enforceable — a violator would have to
import the repository directly via its full path, which CI review
would catch.

### Protocol-based no-op parity is a deliberate offensive-programming discipline

Pairs of protocols and no-op implementations:

- `EventBus` / `NullEventBus`
- `RateLimiter` / `NoOpLimiter`

These ensure callers **never branch on `is None`**; absent functionality
is represented by an active no-op object that satisfies the protocol.
This is the right pattern for L1 primitives — the cluster pass
identified it as a deliberate idiom rather than incidental duplication.

---

## §6 Cross-cluster handshakes

| Partner | Direction | Shape |
|---------|-----------|-------|
| `contracts/` (L0) | core → contracts | 50+ identifiers imported (errors, payload protocols, freeze primitives, schema/schema_contract/secrets/security types, audit DTOs, checkpoint family, enums) |
| `engine/tokens.py:19` (L2) | engine → core | TokenManager façade delegates persistence to `DataFlowRepository` |
| `engine/triggers.py:24`, `engine/commencement.py:12`, `engine/dependency_resolver.py:14` (L2) | engine → core | Three sites consume `core/expression_parser.py` |
| `web/secrets/` (L3) | web → core | Runtime secret-ref resolver (`core/secrets.py`, 124 LOC) consumed when the web composer threads `{"secret_ref": ...}` references through resolved configs |
