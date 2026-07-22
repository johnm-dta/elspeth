# State Engine Proof Catalog

The [v1 catalog](v1/catalog.json) is the finite proof universe for state-engine
assessments. It closes 68 stable legs, ten dimensions, named sink-effect cases,
and ten hard gates. Dated `assessment.json` files bind results to one exact code
baseline.

The stable namespaces are `TS-00..18`, `AUX-01..07`, `RC-01..07`,
`PB-01..09`, `RM-01..13`, and `F-01..13`. Run-coordination legs are explicit;
leader-seat and worker-registry changes are not hidden inside a generic fence.

## Catalog rules

- Leg IDs are literal and ordered; range expressions are explanatory only.
- Every leg uses an applicability profile that accounts for all ten dimensions;
  `family_dimension_acceptance` makes each dimension's proof obligation concrete.
- `execution_profiles` closes the state-store, deployment, lifecycle, and
  first-party plugin inventory. A changed inventory requires a catalog revision.
- The v1 profile treats every dimension as required. Narrowing a dimension to
  N/A requires a catalog revision with a precise reason; an assessor cannot do
  it ad hoc.
- Unless a leg declares dimension-specific cases, its required case is
  `leg-contract` for every dimension.
- A narrow arm is a case beneath its stable leg, never a new pseudo-leg.
- PB-06 and PB-07 declare the complete sink-effect lifecycle and restart seams
  explicitly because a broad “sink durability” cell would hide material gaps.
- `maintenance` is always required.

## Assessment overlay

Each dated assessment must contain all 68 leg IDs. To keep unresolved
assessments readable, a leg may declare `default_status: unknown`; the default
expands to every required dimension/case not named by an override. The derived
verdict remains unresolved until every default is replaced by `pass`, `fail`,
`partial`, or catalog-approved `not_applicable` evidence.

An omitted leg is invalid. An omitted override is not silently passed—it is
explicitly unknown through the default.

## Promotion rules

- `pass` requires executable evidence at the assessment baseline.
- `partial` and `fail` require executable evidence, an exact reason, an
  observable exit gate, and an explicit `owner_issue` key.
- `unknown` requires a reason, exit gate, and explicit owner key; `null` means
  visibly unowned.
- Documentary, decision, source-inspection, and tracker evidence may support a
  result but cannot independently produce behavioral `pass`.
- Evidence coverage names exact `(leg_id, dimension_id, case_id)` tuples.
- The assessment records both `establishes` and `does_not_establish`.

## Direct validation

Follow [the assessment program](../assessment-program.md). It provides
duplicate-key-safe JSON parsing, exact catalog/manifest checks, executable-node
collection, relative-link checks, and review requirements. These are direct
assessment operations, not unit tests for the document package.
