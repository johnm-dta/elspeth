# Reproducible State Engine Assessment Program

This is the executable runbook for a future agent. Run commands from the
repository root unless a step says otherwise.

## Quick start

1. Read this page, the [criteria](completeness-criteria.md), the
   [architecture](architecture.md), and the [proof catalog](proof-catalog/README.md).
2. Run the full initializer below, or materialize a delta from its parent.
3. Capture the baseline before executing evidence.
4. Execute exact command vectors and retain honest limitations.
5. Have independent architecture, evidence, and future-agent readers review the
   package.
6. Update the hub only after material review findings are resolved.

## 1. Choose mode and assessment ID

Use local Canberra time in `YYYY-MM-DD-HHMM` form:

```bash
STATE_ASSESSMENT_ID="$(TZ=Australia/Canberra date '+%Y-%m-%d-%H%M')"
STATE_ASSESSMENT_DIR="docs/architecture/state_engine/assessments/${STATE_ASSESSMENT_ID}"
mkdir -p "${STATE_ASSESSMENT_DIR}"
printf '%s\n' "${STATE_ASSESSMENT_ID}" "${STATE_ASSESSMENT_DIR}"
```

Use `full` when the catalog, architecture, state vocabulary, transaction
boundaries, support profiles, or global verdict may change. Use `delta` only
for named legs/cases.

For a full assessment, initialize every catalog leg and hard gate directly:

```bash
.venv/bin/python - "${STATE_ASSESSMENT_ID}" "${STATE_ASSESSMENT_DIR}/assessment.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

assessment_id, output_name = sys.argv[1:]
catalog_path = Path('docs/architecture/state_engine/proof-catalog/v1/catalog.json')
catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
document = {
    'schema_version': 1,
    'assessment_id': assessment_id,
    'mode': 'full',
    'parent_assessment': None,
    'changed_tuples': [],
    'changed_gate_ids': [],
    'catalog': {
        'path': catalog_path.as_posix(),
        'catalog_id': catalog['catalog_id'],
        'schema_version': catalog['schema_version'],
        'sha256_at_evidence_capture': catalog_digest,
    },
    'baseline': {},
    'environment': {},
    'structure_snapshot': {},
    'tracker_snapshot': {},
    'evidence': [],
    'legs': [
        {
            'id': leg['id'],
            'derived_verdict': 'unknown',
            'default_status': 'unknown',
            'reason': 'No current evidence is attached.',
            'owner_issue': None,
            'exit_gate': 'Execute and attach every required catalog case.',
        }
        for leg in catalog['legs']
    ],
    'hard_gates': [
        {
            'id': gate['id'],
            'status': 'unknown',
            'support': [],
            'affected_leg_ids': [],
            'reason': 'Not yet evaluated.',
        }
        for gate in catalog['hard_gates']
    ],
    'derived': {
        'family_counts': {},
        'total': {'confirmed': 0, 'gap': 0, 'unknown': len(catalog['legs'])},
        'overall_verdict': 'insufficient_evidence',
        'reason': 'Assessment initialization only.',
    },
    'limitations': [],
    'review_record': f'docs/architecture/state_engine/assessments/{assessment_id}/review.md',
}
Path(output_name).write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
print(output_name)
PY
cp docs/architecture/state_engine/templates/assessment-readme.md "${STATE_ASSESSMENT_DIR}/README.md"
cp docs/architecture/state_engine/templates/verification-run.md "${STATE_ASSESSMENT_DIR}/evidence.md"
cp docs/architecture/state_engine/templates/review-record.md "${STATE_ASSESSMENT_DIR}/review.md"
```

A delta is a fully materialized 68-leg manifest, never a sparse patch. It names
`parent_assessment` as `{"path": "relative/path/assessment.json", "sha256":
"..."}`, lists exact changed proof cells as `{leg_id, dimension_id, case_id}`
objects, and lists changed gates in `changed_gate_ids`. Copy the parent, change
the assessment/baseline identity, and rerun or remove evidence for every
affected cell. The validator compares normalized cell status plus referenced
evidence content and gate records to the parent; undeclared changes fail.
Unchanged gap reasons and ownership metadata may be carried forward. A delta
cannot change the catalog or declare the global engine complete.

## 2. Capture Git identity

Prefer a clean dedicated worktree at the exact commit. Capture:

```bash
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git branch --show-current
git remote get-url origin
git status --porcelain=v2 --branch --untracked-files=all
git worktree list --porcelain
git submodule status --recursive
```

If relevant uncommitted changes exist, stop and either create a clean worktree
or retain the overlay explicitly outside the worktree before creating package
files:

```bash
STATE_CAPTURE_DIR="$(mktemp -d)"
git status --porcelain=v2 --branch --untracked-files=all \
  >"${STATE_CAPTURE_DIR}/worktree-status.txt"
git diff --binary >"${STATE_CAPTURE_DIR}/unstaged.patch"
git diff --binary --cached >"${STATE_CAPTURE_DIR}/staged.patch"
git ls-files --others --exclude-standard -z \
  >"${STATE_CAPTURE_DIR}/untracked.paths.z"
tar --null --verbatim-files-from -czf "${STATE_CAPTURE_DIR}/untracked.tar.gz" \
  -T "${STATE_CAPTURE_DIR}/untracked.paths.z"
sha256sum "${STATE_CAPTURE_DIR}"/*
```

Retain the patches, path list, and archive when an untracked or modified file
can affect behavior. A hash-only list cannot reconstruct untracked content.

Never claim a dirty run is reproducible from a commit alone. Never copy or
symlink another worktree's `.venv`; create and sync a worktree-local one.

## 3. Capture environment identity

```bash
.venv/bin/python --version
.venv/bin/python -m pytest --version
uv --version
git --version
sha256sum pyproject.toml uv.lock
uv pip freeze --python .venv/bin/python
uname -a
locale
```

Record Python executable/build, OS/kernel/architecture, locale, timezone,
`PYTHONHASHSEED`, Hypothesis profile, multiprocessing start method, database
dialect and server/image digest, SQLite/SQLAlchemy versions, and feature flags.

Record environment variable names and safe semantic values only. Do not store
credentials, tokens, connection secrets, or raw `.env` contents. State whether
`.env` was absent, loaded, or explicitly disabled. For evidence that does not
need credentials, pass `-p no:dotenv`; record its expected `env_files` warning.

## 4. Capture structural and tracker state

Use Loomweave's `project_status_get` and `index_diff_get`. Record the analysis
run ID, analyzed commit, plugin/entity/edge counts, staleness, truncation,
unresolved/external counts, and briefing blocks. Retain raw request/response
JSON plus hashes only when a structural or unreachability claim depends on it;
otherwise the exact summarized fields and limitations are sufficient.

Fail closed on `stale_worktree`. For ordinary stale state, refresh the index or
record a mechanically checkable changed-path-to-scope disjointness proof.

Capture Filigree with exact non-truncated queries. At minimum retain:

```bash
filigree --version
filigree search '[state engine]' --json
filigree ready --json
filigree blocked --json
```

Prefer a full JSONL export when tracker identity is load-bearing. Record capture
time, command/query, atomicity limitations, byte size, and SHA-256. A truncated
session-context screen is orientation, not canonical evidence.

## 5. Validate catalog identity

The current catalog is `proof-catalog/v1/catalog.json`. Record its schema
version and digest:

```bash
sha256sum docs/architecture/state_engine/proof-catalog/v1/catalog.json
.venv/bin/python -m json.tool \
  docs/architecture/state_engine/proof-catalog/v1/catalog.json >/dev/null
```

Reject duplicate JSON keys; normal `json.load` silently overwrites them. Use
this direct check for the catalog and every assessment manifest:

```bash
.venv/bin/python - "${STATE_ASSESSMENT_DIR}/assessment.json" <<'PY'
import json
import sys
from pathlib import Path

paths = [
    Path('docs/architecture/state_engine/proof-catalog/v1/catalog.json'),
    Path(sys.argv[1]),
]

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result

for path in paths:
    with path.open(encoding='utf-8') as handle:
        json.load(handle, object_pairs_hook=unique_object)
    print(f'valid JSON with unique keys: {path}')
PY
```

## 6. Execute evidence

Run production-boundary evidence first, then direct repository detail. Use
argument vectors in `assessment.json`; the Markdown command is a readable
rendering, not the authority.

For pytest evidence, emit a JUnit artifact and record exact node collection:

```bash
STATE_EVIDENCE_ARTIFACTS="${STATE_ASSESSMENT_DIR}/artifacts"
STATE_EVIDENCE_NODES="${STATE_ASSESSMENT_DIR}/nodes"
mkdir -p "${STATE_EVIDENCE_ARTIFACTS}" "${STATE_EVIDENCE_NODES}"
.venv/bin/python -m pytest --collect-only -q -n 0 \
  tests/path/test_file.py::test_exact_node \
  >"${STATE_EVIDENCE_ARTIFACTS}/EV-001.collect.stdout" \
  2>"${STATE_EVIDENCE_ARTIFACTS}/EV-001.collect.stderr"
awk 'index($0, "::") && $0 !~ /^ / {print}' \
  "${STATE_EVIDENCE_ARTIFACTS}/EV-001.collect.stdout" \
  >"${STATE_EVIDENCE_NODES}/EV-001.txt"
.venv/bin/python -m pytest -q -n 0 \
  --junitxml="${STATE_EVIDENCE_ARTIFACTS}/EV-001.junit.xml" \
  tests/path/test_file.py::test_exact_node \
  >"${STATE_EVIDENCE_ARTIFACTS}/EV-001.stdout" \
  2>"${STATE_EVIDENCE_ARTIFACTS}/EV-001.stderr"
sha256sum \
  "${STATE_EVIDENCE_NODES}/EV-001.txt" \
  "${STATE_EVIDENCE_ARTIFACTS}"/EV-001.*
```

Record exit code even when the command fails. Do not hide skips, xfails,
warnings, service absence, or nondeterministic timing.

## 7. Attach evidence to catalog cases

Evidence may cover only the exact leg/dimension/case tuples its assertions
prove. Each record says what it establishes and does not establish.

Use these minimum standards:

- production entry: real caller, not helper-only construction;
- rollback: before/after images include scheduler, events, coordination,
  outcomes, branch loss, effects, attempts, and external visibility as
  applicable;
- concurrency: independent connections/processes with a bounded completion
  oracle;
- crash/restart: fresh objects/process against the same durable store;
- boundary composition: representative supported plugin or orchestration
  boundary, not a mock-only/helper-only seam;
- read model: positive and negative truth-table arms, exact expiry boundary,
  owner/run scoping, deduplication, and ordering.

## 8. Classify gaps and tracker ownership

Every unresolved case records a reason, observable exit gate, and
`owner_issue`. Use `null` when genuinely unowned. Create or update a Filigree
issue for a coherent confirmed defect or actionable remediation theme; do not
create one issue per unknown proof cell. A closed evidence-package issue is
historical context, not the owner of broader residual work.

Filigree status is live and mutable. Store the issue ID and captured snapshot;
do not copy current assignment or priority into evergreen architecture prose.

## 9. Review and iterate

Dispatch independent readers for the three lenses defined in the framework.
Use `templates/review-record.md`. For each material finding:

1. reproduce or inspect the cited evidence;
2. accept, reject, or narrow it with a concrete reason;
3. change the catalog, assessment, evidence, or prose as required;
4. rerun affected commands;
5. request re-review from a fresh reader.

Do not treat reviewer names, approvals, or signatures as correctness evidence.
The final review record must contain the exact line `Review outcome: complete`;
the direct validator rejects a pending record.

## 10. Direct package validation

Before updating the hub:

```bash
STATE_ASSESSMENT_PATH="${STATE_ASSESSMENT_DIR}/assessment.json"
.venv/bin/python -m json.tool \
  docs/architecture/state_engine/proof-catalog/v1/catalog.json >/dev/null
.venv/bin/python -m json.tool \
  "${STATE_ASSESSMENT_PATH}" >/dev/null
if rg -n 'TBD|TODO|FIXME|<timestamp>|<full SHA>' \
  docs/architecture/state_engine/README.md \
  docs/architecture/state_engine/architecture.md \
  docs/architecture/state_engine/proof-matrix.md \
  docs/architecture/state_engine/completeness-criteria.md \
  docs/architecture/state_engine/assessment-framework.md \
  "${STATE_ASSESSMENT_DIR}/README.md" \
  "${STATE_ASSESSMENT_DIR}/evidence.md" \
  "${STATE_ASSESSMENT_DIR}/review.md"; then
  printf '%s\n' 'unresolved placeholder found' >&2
  exit 1
fi
rg -n '^Review outcome: complete$' "${STATE_ASSESSMENT_DIR}/review.md"
git diff --check
```

Run this contract check literally. It rejects duplicate keys, namespace or
profile drift, omitted legs, unsupported evidence promotion/N/A, dangling
coverage, invalid gate mappings, dishonest derived verdicts/counts, altered
node indexes, and a human proof matrix that contradicts the manifest:

The package-bearing checkout may include later documentation-only commits.
The validator resolves the recorded baseline commit/tree and refuses any
committed or uncommitted difference outside `docs/` before it runs live plugin
discovery. Historical behavioral divergence uses the detached rerun path below.

```bash
PYTHONOPTIMIZE=0 .venv/bin/python - "${STATE_ASSESSMENT_PATH}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

if not __debug__:
    raise SystemExit('assessment validation requires Python assertions enabled')

catalog_path = Path('docs/architecture/state_engine/proof-catalog/v1/catalog.json')
assessment_path = Path(sys.argv[1])

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result

def load(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle, object_pairs_hook=unique_object)

catalog = load(catalog_path)
assessment = load(assessment_path)
dimensions = [
    'production_entry', 'precondition_image', 'success_effects',
    'guard_refusal', 'zero_mutation_rollback', 'concurrency',
    'crash_restart', 'boundary_composition', 'read_model_truth_table',
    'maintenance',
]
expected_legs = (
    [f'TS-{n:02d}' for n in range(19)]
    + [f'AUX-{n:02d}' for n in range(1, 8)]
    + [f'RC-{n:02d}' for n in range(1, 8)]
    + [f'PB-{n:02d}' for n in range(1, 10)]
    + [f'RM-{n:02d}' for n in range(1, 14)]
    + [f'F-{n:02d}' for n in range(1, 14)]
)
assert catalog['dimensions'] == dimensions
assert catalog['required_leg_ids'] == expected_legs
catalog_legs = catalog['legs']
assert [leg['id'] for leg in catalog_legs] == expected_legs
assert len({leg['id'] for leg in catalog_legs}) == 68
families = {leg['family'] for leg in catalog_legs}
assert set(catalog['family_dimension_acceptance']) == families
for acceptance in catalog['family_dimension_acceptance'].values():
    assert list(acceptance) == dimensions
for leg in catalog_legs:
    profile = catalog['applicability_profiles'][leg['applicability_profile']]
    assert set(profile) == {'default_case_id', *dimensions}
    assert set(profile[dimension] for dimension in dimensions) <= {'required', 'not_applicable'}

legs = assessment['legs']
assert [leg['id'] for leg in legs] == expected_legs
assert len({leg['id'] for leg in legs}) == 68
assert assessment['mode'] in {'full', 'delta'}
if assessment['mode'] == 'full':
    assert assessment.get('parent_assessment') is None
    assert not assessment.get('changed_tuples')
else:
    assert assessment.get('parent_assessment')
    assert assessment.get('changed_tuples')
    assert assessment['derived']['overall_verdict'] != 'complete'
assert assessment['catalog']['catalog_id'] == catalog['catalog_id']
assert assessment['catalog']['schema_version'] == catalog['schema_version']
assert assessment['catalog']['sha256_at_evidence_capture'] == hashlib.sha256(
    catalog_path.read_bytes()
).hexdigest()

baseline = assessment['baseline']
for field in (
    'repository_root', 'remote', 'branch', 'commit', 'tree',
    'behavioral_overlay', 'worktree_status_at_evidence_capture',
    'submodules', 'worktrees_at_capture',
):
    assert field in baseline, ('baseline', field)
assert Path(baseline['repository_root']).is_absolute()
assert baseline['remote']
assert re.fullmatch(r'[0-9a-f]{40}', baseline['commit'])
assert re.fullmatch(r'[0-9a-f]{40}', baseline['tree'])
assert isinstance(baseline['worktree_status_at_evidence_capture'], list)
assert baseline['behavioral_overlay'] is None or isinstance(baseline['behavioral_overlay'], dict)
assert subprocess.check_output(
    ['git', 'rev-parse', f"{baseline['commit']}^{{tree}}"], text=True
).strip() == baseline['tree']
non_document_diff = subprocess.run(
    [
        'git', 'diff', '--quiet', f"{baseline['commit']}..HEAD", '--', '.',
        ':(exclude)docs/**',
    ],
    check=False,
).returncode
assert non_document_diff == 0, 'execution checkout differs from baseline outside docs/'
non_document_status = subprocess.check_output(
    [
        'git', 'status', '--porcelain', '--untracked-files=all', '--',
        '.', ':(exclude)docs/**',
    ],
    text=True,
).strip()
assert not non_document_status, f'non-document overlay is not clean: {non_document_status}'

environment = assessment['environment']
for field in (
    'captured_at', 'timezone', 'locale', 'kernel', 'python',
    'python_executable', 'python_build', 'pytest', 'uv', 'git', 'sqlite',
    'sqlalchemy', 'multiprocessing_start_method_before_tests',
    'pythonhashseed', 'dotenv', 'pyproject_sha256', 'uv_lock_sha256',
    'database_profile', 'sensitive_environment_captured',
):
    assert field in environment, ('environment', field)
assert re.fullmatch(r'[0-9a-f]{64}', environment['pyproject_sha256'])
assert re.fullmatch(r'[0-9a-f]{64}', environment['uv_lock_sha256'])
assert environment['sensitive_environment_captured'] is False
for snapshot_name in ('structure_snapshot', 'tracker_snapshot'):
    snapshot = assessment[snapshot_name]
    for field in ('provider', 'captured_at', 'limitation'):
        assert field in snapshot, (snapshot_name, field)

from elspeth.plugins.infrastructure.discovery import discover_all_plugins
live_plugins = {
    kind: sorted(plugin.name for plugin in plugin_classes)
    for kind, plugin_classes in discover_all_plugins().items()
}
profile_plugins = catalog['execution_profiles']['first_party_plugins']
for kind in ('sources', 'transforms', 'sinks'):
    assert profile_plugins[kind] == sorted(set(profile_plugins[kind]))
    assert profile_plugins[kind] == live_plugins[kind]

review_path = Path(assessment['review_record'])
assert not review_path.is_absolute() and review_path.is_file()
assert review_path.resolve() == (assessment_path.parent / 'review.md').resolve()
review_text = review_path.read_text(encoding='utf-8')
if len(re.findall(r'^Review outcome: complete$', review_text, re.MULTILINE)) != 1:
    raise AssertionError(f'review is not complete: {review_path}')
assert not re.search(r'(?i)\boutcome:\s*pending\b', review_text)
evidence_ids = [record['id'] for record in assessment['evidence']]
assert len(evidence_ids) == len(set(evidence_ids))
evidence_ids = set(evidence_ids)
catalog_by_id = {leg['id']: leg for leg in catalog_legs}
statuses = {'pass', 'partial', 'fail', 'unknown', 'not_applicable'}
verdicts = {'confirmed', 'gap', 'unknown'}
coverage_by_evidence = {}
evidence_by_id = {record['id']: record for record in assessment['evidence']}

changed_tuples = assessment.get('changed_tuples', [])
changed_keys = {
    (item['leg_id'], item['dimension_id'], item['case_id'])
    for item in changed_tuples
}
assert len(changed_keys) == len(changed_tuples)
for leg_id, dimension_id, case_id in changed_keys:
    assert leg_id in catalog_by_id
    assert dimension_id in dimensions
    allowed = set(catalog_by_id[leg_id].get('required_cases', ['leg-contract']))
    assert case_id in allowed
if assessment['mode'] == 'delta':
    parent = assessment['parent_assessment']
    assert set(parent) == {'path', 'sha256'}
    parent_path = Path(parent['path'])
    assert not parent_path.is_absolute() and parent_path.is_file()
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent['sha256']

changed_gate_ids = assessment.get('changed_gate_ids', [])
assert len(changed_gate_ids) == len(set(changed_gate_ids))
assert set(changed_gate_ids) <= {gate['id'] for gate in catalog['hard_gates']}
if assessment['mode'] == 'full':
    assert not changed_gate_ids

for record in assessment['evidence']:
    for field in (
        'kind', 'reproducibility_class', 'argv', 'cwd_relative',
        'timeout_seconds', 'safe_environment', 'resources', 'started_at',
        'ended_at', 'duration_seconds', 'exit_code', 'result_counts',
        'coverage', 'retained_artifacts', 'establishes', 'does_not_establish',
    ):
        assert field in record, (record['id'], field)
    assert record['reproducibility_class'] in {
        'deterministic', 'semantic_comparison', 'external_observation'
    }
    assert not Path(record['cwd_relative']).is_absolute()
    assert isinstance(record['safe_environment'], dict)
    assert all(
        re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)
        and (value is None or isinstance(value, str))
        for name, value in record['safe_environment'].items()
    )
    coverage = {
        (item['leg_id'], item['dimension_id'], item['case_id'])
        for item in record['coverage']
    }
    assert len(coverage) == len(record['coverage'])
    coverage_by_evidence[record['id']] = coverage
    for leg_id, dimension_id, case_id in coverage:
        assert leg_id in catalog_by_id
        assert dimension_id in dimensions
        allowed = set(catalog_by_id[leg_id].get('required_cases', ['leg-contract']))
        assert case_id in allowed
    if record['kind'] == 'pytest':
        index = record['collected_node_index']
        node_path = Path(index['path'])
        assert not node_path.is_absolute()
        assert node_path.is_file()
        assert hashlib.sha256(node_path.read_bytes()).hexdigest() == index['sha256']
        assert len(node_path.read_text(encoding='utf-8').splitlines()) == record['collected_nodes']
        artifact_names = {Path(item['path']).name for item in record['retained_artifacts']}
        assert any(name.endswith('.junit.xml') for name in artifact_names)
        assert any(name.endswith('.stdout') for name in artifact_names)
        assert any(name.endswith('.stderr') for name in artifact_names)
        junit_paths = [
            Path(item['path']) for item in record['retained_artifacts']
            if item['path'].endswith('.junit.xml')
        ]
        assert len(junit_paths) == 1
        junit_root = ET.parse(junit_paths[0]).getroot()
        suites = [junit_root] if junit_root.tag.endswith('testsuite') else [
            child for child in junit_root if child.tag.endswith('testsuite')
        ]
        junit_counts = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ('tests', 'failures', 'errors', 'skipped')
        }
        counts = record['result_counts']
        assert junit_counts['tests'] == sum(
            counts.get(key, 0)
            for key in ('passed', 'failed', 'errors', 'skipped', 'xfailed', 'xpassed')
        )
        assert junit_counts['tests'] == record['collected_nodes']
        assert junit_counts['failures'] == counts.get('failed', 0)
        assert junit_counts['errors'] == counts.get('errors', 0)
        assert junit_counts['skipped'] == counts.get('skipped', 0) + counts.get('xfailed', 0)
    for artifact in record['retained_artifacts']:
        path = Path(artifact['path'])
        assert not path.is_absolute()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']

gate_ids = [gate['id'] for gate in catalog['hard_gates']]
assert [gate['id'] for gate in assessment['hard_gates']] == gate_ids
open_affected = set()
for gate in assessment['hard_gates']:
    assert gate['status'] in {'open', 'closed', 'unknown'}
    assert gate.get('reason')
    assert set(gate['support']) <= evidence_ids
    assert len(gate['affected_leg_ids']) == len(set(gate['affected_leg_ids']))
    assert set(gate['affected_leg_ids']) <= set(expected_legs)
    if gate['status'] == 'open':
        open_affected.update(gate['affected_leg_ids'])

derived = {}
has_unresolved_cell = False

for leg in legs:
    assert leg['derived_verdict'] in verdicts
    assert leg['default_status'] == 'unknown'
    for key in ('reason', 'owner_issue', 'exit_gate'):
        assert key in leg
    allowed_cases = set(catalog_by_id[leg['id']].get('required_cases', ['leg-contract']))
    profile = catalog['applicability_profiles'][catalog_by_id[leg['id']]['applicability_profile']]
    cell_status = {
        (dimension, case): (
            leg['default_status'] if profile[dimension] == 'required' else 'not_applicable'
        )
        for dimension in dimensions
        for case in allowed_cases
    }
    seen = set()
    for override in leg.get('overrides', []):
        key = (override['dimension'], override['case'])
        assert key not in seen
        seen.add(key)
        assert override['dimension'] in dimensions
        assert override['case'] in allowed_cases
        assert override['status'] in statuses
        if override['status'] in {'pass', 'partial', 'fail'}:
            assert override.get('evidence')
        if override['status'] in {'partial', 'fail', 'unknown'}:
            for field in ('reason', 'owner_issue', 'exit_gate'):
                assert field in override
        if override['status'] == 'not_applicable':
            assert profile[override['dimension']] == 'not_applicable'
            assert not override.get('evidence')
        assert set(override.get('evidence', [])) <= evidence_ids
        for evidence_id in override.get('evidence', []):
            assert (leg['id'], override['dimension'], override['case']) in coverage_by_evidence[evidence_id]
            if override['status'] in {'pass', 'partial'}:
                record = evidence_by_id[evidence_id]
                assert record['exit_code'] == 0
                assert not record['result_counts'].get('failed', 0)
                assert not record['result_counts'].get('errors', 0)
        cell_status[key] = override['status']
    has_unresolved_cell = has_unresolved_cell or bool(
        {'unknown', 'partial'} & set(cell_status.values())
    )
    if 'fail' in cell_status.values() or leg['id'] in open_affected:
        derived[leg['id']] = 'gap'
    elif set(cell_status.values()) <= {'pass', 'not_applicable'}:
        derived[leg['id']] = 'confirmed'
    else:
        derived[leg['id']] = 'unknown'
    assert leg['derived_verdict'] == derived[leg['id']], leg['id']

hg09 = next(
    gate for gate in assessment['hard_gates']
    if gate['id'] == 'HG-09-mandatory-leg-unresolved'
)
assert hg09['status'] == ('open' if has_unresolved_cell else 'closed')

def normalized_cells(document):
    record_digests = {
        record['id']: hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        for record in document['evidence']
    }
    document_legs = {leg['id']: leg for leg in document['legs']}
    result = {}
    for catalog_leg in catalog_legs:
        leg = document_legs[catalog_leg['id']]
        cases = set(catalog_leg.get('required_cases', ['leg-contract']))
        profile = catalog['applicability_profiles'][catalog_leg['applicability_profile']]
        for dimension in dimensions:
            for case in cases:
                status = leg['default_status'] if profile[dimension] == 'required' else 'not_applicable'
                result[(leg['id'], dimension, case)] = (status, ())
        for override in leg.get('overrides', []):
            evidence_payload = tuple(
                (evidence_id, record_digests[evidence_id])
                for evidence_id in override.get('evidence', [])
            )
            result[(leg['id'], override['dimension'], override['case'])] = (
                override['status'], evidence_payload
            )
    return result

if assessment['mode'] == 'delta':
    parent_assessment = load(parent_path)
    assert parent_assessment['catalog']['catalog_id'] == assessment['catalog']['catalog_id']
    assert parent_assessment['catalog']['sha256_at_evidence_capture'] == assessment['catalog']['sha256_at_evidence_capture']
    assert [leg['id'] for leg in parent_assessment['legs']] == expected_legs
    parent_cells = normalized_cells(parent_assessment)
    current_cells = normalized_cells(assessment)
    actual_changed_keys = {
        key for key in current_cells if current_cells[key] != parent_cells[key]
    }
    assert actual_changed_keys == changed_keys
    parent_gates = {gate['id']: gate for gate in parent_assessment['hard_gates']}
    current_gates = {gate['id']: gate for gate in assessment['hard_gates']}
    actual_changed_gates = {
        gate_id for gate_id in current_gates
        if current_gates[gate_id] != parent_gates[gate_id]
    }
    assert actual_changed_gates == set(changed_gate_ids)

family_for = {leg['id']: leg['family'] for leg in catalog_legs}
actual = defaultdict(Counter)
for leg in legs:
    actual[family_for[leg['id']]][derived[leg['id']]] += 1
assert set(assessment['derived']['family_counts']) == families
for family, expected in assessment['derived']['family_counts'].items():
    assert set(expected) == verdicts
    assert {key: actual[family][key] for key in verdicts} == expected
total = Counter(derived.values())
assert set(assessment['derived']['total']) == verdicts
assert {key: total[key] for key in verdicts} == assessment['derived']['total']
gate_statuses = {gate['status'] for gate in assessment['hard_gates']}
if total.get('gap') or 'open' in gate_statuses:
    overall = 'not_complete'
elif total.get('unknown') or 'unknown' in gate_statuses:
    overall = 'insufficient_evidence'
else:
    overall = 'complete'
assert assessment['derived']['overall_verdict'] == overall

labels = {
    'transition': 'Token transitions',
    'auxiliary': 'Auxiliary state',
    'run_coordination': 'Run coordination',
    'production_boundary': 'Production boundaries',
    'read_model': 'Read models',
    'forbidden': 'Forbidden paths',
}
matrix = Path('docs/architecture/state_engine/proof-matrix.md').read_text(encoding='utf-8')
for family, label in labels.items():
    match = re.search(
        rf'^\| {re.escape(label)} \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|',
        matrix,
        re.MULTILINE,
    )
    assert match, label
    stated = tuple(map(int, match.groups()))
    expected = (
        sum(actual[family].values()),
        actual[family]['confirmed'],
        actual[family]['gap'],
        actual[family]['unknown'],
    )
    assert stated == expected, (label, stated, expected)
total_match = re.search(
    r'^\| \*\*Total\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \|',
    matrix,
    re.MULTILINE,
)
assert total_match
assert tuple(map(int, total_match.groups())) == (
    len(legs), total['confirmed'], total['gap'], total['unknown']
)
print(f'state-engine assessment contract: valid ({len(legs)} legs, {overall})')
PY
```

Batch-collect every pytest selector and compare the exact node list with the
retained index:

```bash
.venv/bin/python - "${STATE_ASSESSMENT_PATH}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

assessment = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for record in assessment['evidence']:
    if record['kind'] != 'pytest':
        continue
    argv = []
    source_argv = iter(record['argv'])
    for item in source_argv:
        if item == '--junitxml':
            next(source_argv)
        elif not item.startswith('--junitxml='):
            argv.append(item)
    argv.insert(argv.index('pytest') + 1, '--collect-only')
    result = subprocess.run(
        argv,
        cwd=record['cwd_relative'],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    collected = [line for line in result.stdout.splitlines() if '::' in line and not line.startswith(' ')]
    expected = Path(record['collected_node_index']['path']).read_text(encoding='utf-8').splitlines()
    assert collected == expected, record['id']
    print(f"{record['id']}: {len(collected)} exact nodes")
PY
```

Then check repository-relative Markdown links:

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path

roots = [Path('docs/architecture/state_engine'), Path('docs/README.md')]
files = []
for root in roots:
    files.extend(root.rglob('*.md') if root.is_dir() else [root])
missing = []
for source in files:
    text = source.read_text(encoding='utf-8')
    for target in re.findall(r'(?<!!)\[[^]]+\]\(([^)]+)\)', text):
        target = target.strip().split()[0].strip('<>')
        if target.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        path = target.split('#', 1)[0]
        if path and not (source.parent / path).resolve().exists():
            missing.append(f'{source}: {target}')
if missing:
    raise SystemExit('\n'.join(missing))
print(f'markdown links: valid across {len(files)} files')
PY
```

These are direct assessment operations, not unit tests for the document
package.

## Historical rerun

### Strict v1 rerun

Use this path when the assessment has a v1 manifest and recorded hashes:

1. Verify the catalog digest, node indexes, retained overlays, and evidence
   artifact hashes that actually exist in the manifest. Keep this
   package-bearing checkout separate from the baseline execution worktree:

   ```bash
   STATE_PACKAGE_ROOT="$(git rev-parse --show-toplevel)"
   STATE_PACKAGE_ASSESSMENT_DIR="${STATE_PACKAGE_ROOT}/docs/architecture/state_engine/assessments/<assessment-id>"
   test -f "${STATE_PACKAGE_ASSESSMENT_DIR}/assessment.json"
   STATE_RERUN_ROOT="$(mktemp -d)"
   STATE_EXECUTION_ROOT="${STATE_RERUN_ROOT}/repo"
   git -C "${STATE_PACKAGE_ROOT}" worktree add --detach \
     "${STATE_EXECUTION_ROOT}" <full-commit>
   cd "${STATE_EXECUTION_ROOT}"
   uv sync --frozen --all-extras
   STATE_RERUN_ID="$(TZ=Australia/Canberra date '+%Y-%m-%d-%H%M')"
   STATE_RERUN_OUTPUT="${STATE_PACKAGE_ASSESSMENT_DIR}/reruns/${STATE_RERUN_ID}"
   mkdir -p "${STATE_RERUN_OUTPUT}"
   ```

2. Reconstruct a recorded behavioral overlay, if any, in the execution
   worktree and verify its digest.
3. Compare environment facts before executing recorded argument vectors.
   Resolve manifest-relative node and original-artifact paths against
   `STATE_PACKAGE_ROOT`. Run behavioral commands in `STATE_EXECUTION_ROOT`.
   Preserve every behavioral selector and option, but rewrite output-only
   `--junitxml=...` arguments and stdout/stderr transport to the new rerun
   directory; the original dated package does not exist in an older baseline
   worktree. This is the only permitted argv rewrite:

   ```bash
   .venv/bin/python - \
     "${STATE_PACKAGE_ASSESSMENT_DIR}/assessment.json" \
     "${STATE_RERUN_OUTPUT}" <<'PY'
   import json
   import os
   import subprocess
   import sys
   from pathlib import Path

   manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
   output = Path(sys.argv[2]).resolve()
   executions = []
   for record in manifest['evidence']:
       junit_output = output / (record['id'] + '.junit.xml')
       argv = []
       source_argv = iter(record['argv'])
       for item in source_argv:
           if item == '--junitxml':
               next(source_argv)
               argv.extend([item, str(junit_output)])
           elif item.startswith('--junitxml='):
               argv.append(f'--junitxml={junit_output}')
           else:
               argv.append(item)
       environment = os.environ.copy()
       for name, value in record['safe_environment'].items():
           if value is None:
               environment.pop(name, None)
           else:
               environment[name] = str(value)
       timed_out = False
       with (
           (output / (record['id'] + '.stdout')).open('wb') as stdout,
           (output / (record['id'] + '.stderr')).open('wb') as stderr,
       ):
           try:
               result = subprocess.run(
                   argv,
                   cwd=record['cwd_relative'],
                   env=environment,
                   timeout=record['timeout_seconds'],
                   stdout=stdout,
                   stderr=stderr,
                   check=False,
               )
               exit_code = result.returncode
           except subprocess.TimeoutExpired:
               timed_out = True
               exit_code = 124
               stderr.write(b'\nassessment rerun timed out\n')
       executions.append({
           'id': record['id'],
           'original_argv': record['argv'],
           'executed_argv': argv,
           'cwd_relative': record['cwd_relative'],
           'timeout_seconds': record['timeout_seconds'],
           'safe_environment': record['safe_environment'],
           'timed_out': timed_out,
           'exit_code': exit_code,
       })
       print(f"{record['id']}: exit {exit_code}, timed_out={timed_out}")
   (output / 'execution-vectors.json').write_text(
       json.dumps(executions, indent=2) + '\n',
       encoding='utf-8',
   )
   PY
   ```
4. Treat historical tracker/index captures as evidence and new live state as a
   separate observation.
5. Write all new output to `STATE_RERUN_OUTPUT` in the package-bearing
   checkout; compare deterministic artifacts by hash and
   logs/JUnit by declared semantic fields. Elapsed time is informational.
6. Write a divergence report without modifying the original assessment or its
   retained artifacts.

### Legacy best-effort reconstruction

Pre-v1 packages may have no catalog, manifest, node index, overlay, or artifact
hashes. Do not pretend those fields exist. Recover the named Git document with
`git show <commit>:<path>`, create the detached worktree, execute only literal
commands preserved in the package, and mark every missing identity or artifact
`unreproducible`. Store the rerun and divergence report separately; never
rewrite the legacy result into v1 form.

## Failure handling

- `SCHEMA_MISMATCH`: stop and surface the Filigree upgrade guidance.
- stale Loomweave: refresh before structural claims.
- missing external credentials: mark affected cases unknown; do not infer pass.
- dirty unrecorded worktree: stop or capture the complete overlay.
- interrupted command: retain partial output as failed evidence and rerun under
  a new evidence ID.
- contradictory source/docs/tracker: open `HG-10`; do not choose the convenient
  surface silently.
