# ELSPETH release PDFs

This directory builds seven release PDFs covering the project across
the exec-briefing, engineering-evidence, design-reference, assurance,
and trust-primer tiers:

**Briefing tier (under-review, DRAFT-banner-bearing):**

- `out/elspeth-executive-summary.pdf` — exec-tier briefing, with a
  loud DRAFT banner. **Audience:** public-service executives.

**Engineering-evidence tier (reporting on shipped state):**

- `out/elspeth-progress-rc1-to-rc5.pdf` — cumulative engineering
  progress, RC-1 to RC-5.2. **Audience:** engineering reviewers.
- `out/elspeth-velocity-rc1-to-rc5.pdf` — per-day commit cadence.
  **Audience:** engineering reviewers.

**Design-reference tier (no DRAFT banner; describes committed design):**

- `out/elspeth-architecture.pdf` — architectural commitments (SDA
  model, four-layer dependency rules, Landscape audit substrate,
  three-tier trust model, ADR index). Compresses
  `docs/architecture/{overview,landscape,subsystems,…}.md` plus
  `docs/architecture/adr/`. **Audience:** engineering reviewers,
  auditors, integrators.
- `out/elspeth-composer.pdf` — Composer reference (personas,
  default-mode policy, IA decisions, implementation roadmap, chat
  protocol, audit surfaces). Compresses
  `docs/composer/ux-redesign-2026-05/`. **Audience:** engineering
  reviewers, UX reviewers, operational stakeholders.

**Assurance tier (no DRAFT banner; describes release-blocking promises):**

- `out/elspeth-guarantees.pdf` — audit, execution, data, external-call,
  recovery, configuration, authentication, secret-reference, session,
  and Composer guarantees. Compresses `docs/release/guarantees.md`.
  **Audience:** users, integrators, auditors, assurance staff.

**Trust-primer tier (no DRAFT banner; explains the data-handling model):**

- `out/elspeth-data-trust.pdf` — three-tier trust model and the
  practical error-handling consequences for readers evaluating how
  ELSPETH treats source data, pipeline data, external responses, and
  audit records. Compresses
  `docs/guides/data-trust-and-error-handling.md`. **Audience:**
  technical users, reviewers, integrators, assurance staff.

All seven are PDF/UA-1 conformant (Typst export-time check) and
tagged. They are **unofficial documents** — the AGDS-inspired visual
idiom does not imply Commonwealth endorsement; every page carries a
footer disclaimer to that effect.

Non-draft release PDFs use the cover distribution label **Public
evaluation release copy**. The executive summary remains a draft
internal brief until its approval banner is removed.

## Build

```bash
./build.sh             # default build
./build.sh --verify    # default build + veraPDF PDF/UA-1 strict check
```

The script preflights `typst` (required) and `pandoc` (optional —
not used in the default build), enforces a minimum version floor on
each (build fails below the floor), then checks each tool's version
against `versions.lock` (advisory — a drift from the locked version
prints a WARNING but does not fail the build), then compiles each
Typst source with `--pdf-standard ua-1`. Outputs land in `out/`.

Diagrams are drawn natively in Typst (cetz 0.4.2) — see
`theme.typ` `diagram-sda-flow()` and `diagram-trust-tiers()`. The
earlier Mermaid pipeline (mmdc / puppeteer / headless chromium) was
retired because vector output rendered crisper at any zoom,
reproduces deterministically, and removed the brittlest dependency
in the toolchain.

With `--verify`, the script additionally invokes veraPDF (local
binary if present, otherwise `verapdf/cli:v1.30.1` via Docker) and
fails the build on any PDF/UA-1 strict violation. Typst's internal
`--pdf-standard ua-1` check is best-effort at export time; veraPDF
is the authoritative validator and is recommended before any
external sign-off.

### Tool versions

`versions.lock` pins the exact `typst`, `pandoc`, and Typst Universe
package versions used to produce the bundled outputs in `out/`. The
PDFs typically build cleanly on any reasonably modern Typst, but
visual reproduction is byte-stable only at the locked versions. Bump
the lockfile in the same commit as a tool upgrade so the lockfile
and the bundled outputs stay aligned.

The currently locked versions are:

- Typst 0.14.2 — tagged-PDF by default; PDF/UA-1 export.
  Minimum required: 0.14.0 (build fails below this floor).
- cetz 0.4.2 — Typst Universe drawing library. Pin lives in
  `theme.typ` `#import "@preview/cetz:0.4.2"` and is mirrored here.
- cetz-plot 0.1.3 — Typst Universe chart library (used by velocity
  hero bar chart and progress milestone line chart).
- Pandoc 3.9 — reserved for a future markdown reflow path; not
  used in the default build. Optional — build proceeds without it.

### Incremental rebuilds (Makefile)

`build.sh` always re-renders the full PDF set. For tight edit-rebuild
loops, a Makefile is provided that recompiles only the changed
document(s):

```bash
make velocity        # rebuild only the velocity PDF
make guarantees      # rebuild only the guarantees PDF
make exec progress   # rebuild selected PDFs
make verify          # delegates to ./build.sh --verify (authoritative)
```

The Makefile deliberately skips the toolchain preflight and the
`versions.lock` advisory check that `build.sh` performs — use
`build.sh` when entering the project or before any sign-off pass.
The Makefile is a convenience for iterative authoring only.

### Clean build

```bash
make clean && make           # rebuild PDFs from scratch
```

Stale PDFs are otherwise overwritten in place; the explicit clean is
only useful when investigating a partial-build state.

## Regenerating after editing the source markdown

The Typst sources are not regenerated from `docs/release/*.md`,
`docs/architecture/*.md`, or `docs/composer/ux-redesign-2026-05/*.md`
automatically. If a source `.md` changes:

1. **Numbers, dates, and headline metrics** — update `data.typ`. This
   is the single source of truth for all the load-bearing figures
   (cumulative commits, daily series, peak days, phase totals, weekly
   totals). The progress, velocity, and (for `doc-date`)
   architecture / composer documents import from here.
2. **Prose** — update the corresponding `.typ` file
   (`executive-summary.typ`, `progress.typ`, `velocity.typ`,
   `architecture.typ`, `composer.typ`, `guarantees.typ`,
   `data-trust.typ`).
3. **Diagrams** — edit the cetz drawing functions in `theme.typ`
   (`diagram-sda-flow()`, `diagram-trust-tiers()`) and re-run
   `build.sh`. No external diagram pipeline.

### Source-of-truth for the design-reference PDFs

The architecture and composer PDFs are release-formatted reading
copies of working markdown sets, not original documents. When a
substantive fact changes (a new ADR, an amended layer rule, a phase
ships, a new persona is documented), update the markdown set FIRST,
then carry the change into the corresponding `.typ`:

| Markdown source | Release PDF | Notes |
|----|----|----|
| `docs/architecture/overview.md` and siblings | `architecture.typ` | The overview is the canonical compression source; the ADR index in `architecture.typ` mirrors the ADR set at `docs/architecture/adr/`. |
| `docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md` and siblings | `composer.typ` | The roadmap is the canonical phase tracker; phase-status claims in `composer.typ` trace to the roadmap and to operator-confirmed merge gates. |
| `docs/release/guarantees.md` | `guarantees.typ` | The markdown is the canonical assurance appendix; the PDF is a shorter reading copy for release reviewers. |
| `docs/guides/data-trust-and-error-handling.md` | `data-trust.typ` | The markdown remains the implementation guide; the PDF removes internal coding-instruction framing and explains the trust model for interested users and reviewers. |

## Files

| File | Purpose |
|------|---------|
| `build.sh` | Reproducible build script — preflights tools, compiles each PDF under PDF/UA-1, optionally runs veraPDF strict validation under `--verify`. |
| `tokens.typ` | Design tokens: AGDS-inspired palette, Public Sans / Lato type stack, spacing scale, page geometry. WCAG contrast ratios documented inline. |
| `theme.typ` | Reusable Typst components: page master with DRAFT banner support, cover-page, metric cards, callouts, status pills, accessible chart wrapper, branded table helper, native-Typst diagrams (`diagram-sda-flow`, `diagram-trust-tiers`). |
| `data.typ` | Single source of truth for load-bearing figures (release milestones, daily commits, phase totals, weekly summary, capability accretion) plus the `fmt-1dec()` one-decimal display helper. |
| `executive-summary.typ` | Exec summary source. DRAFT banner enabled. |
| `progress.typ` | Progress report source. |
| `velocity.typ` | Velocity report source. The hero chart (123 daily bars) and the seven-column phase table are rendered on landscape pages. |
| `architecture.typ` | Architecture reference source. Reading copy of `docs/architecture/*.md` and the ADR index. No DRAFT banner. |
| `composer.typ` | Composer reference source. Reading copy of `docs/composer/ux-redesign-2026-05/*.md` (personas, default mode, IA, phase roadmap, chat protocol, audit surfaces). No DRAFT banner. |
| `guarantees.typ` | Assurance appendix source. Reading copy of `docs/release/guarantees.md`. No DRAFT banner. |
| `data-trust.typ` | Data trust primer source. Reading copy of `docs/guides/data-trust-and-error-handling.md`. No DRAFT banner. |
| `fonts/public-sans/fonts/ttf/` | Public Sans font files (SIL Open Font License, bundled from `uswds/public-sans` v2.001). |
| `out/*.pdf` | Built PDFs. |

## Font note

**Public Sans** is the AGDS-recommended face. The font files are
bundled in `fonts/public-sans/` and loaded via `--font-path` in
`build.sh`. If the bundled font directory is removed, the fallback
chain in `tokens.typ` falls through to **Lato** (preinstalled on most
modern Linux distributions), then Liberation Sans, then DejaVu Sans.
The visual character of the documents shifts slightly toward Lato's
slightly heavier letterforms when the substitution happens.

`build.sh` consults `typst fonts --font-path …` at preflight and
emits a WARNING if `Public Sans` is not discoverable under the bundled
path — so a partial install (missing weight, broken symlink, or empty
directory) is caught at build time rather than producing silent
per-weight substitution.

### H1 page-break behaviour

`document-frame` accepts an `h1-pagebreak` parameter (default `true`).
When `true`, every level-1 heading starts a fresh page — a deliberate
slide-deck rhythm appropriate for executive briefings. When `false`,
sections flow continuously; short sections no longer trail half-empty
pages. The executive summary keeps the default; the progress and
velocity documents pass `false` so the linear engineering read isn't
inflated by page padding.

## Palette note

The colour values in `tokens.typ` were chosen to be visually
consistent with AGDS. They are **not extracted from a live AGDS
token feed**; if the published AGDS design-token JSON becomes
available, substitute the values in `tokens.typ` directly. Every
body-text-on-white combination has been computed against WCAG 2.2
AA (`>= 4.5:1`) and the ratios are recorded inline in `tokens.typ`.

## Accessibility

- Typst 0.14.2 emits tagged PDF by default.
- `build.sh` compiles each PDF with `--pdf-standard ua-1`; the build
  fails on detectable accessibility violations (Typst's internal
  export-time check — best-effort, not authoritative).
- `build.sh --verify` additionally runs veraPDF and fails on any
  PDF/UA-1 strict violation. veraPDF is the authoritative validator
  and should be the gate before any external sign-off.
- Every Mermaid diagram and every cetz chart has an alt-text or
  description attached.
- Heading hierarchy is sequential (H1 → H2 → H3); no skipped levels.
- Status pills use both colour and a glyph (`■` / `▣` / `□` — filled
  square / partial square / open square) — no information is conveyed
  by colour alone. The earlier disk-fill set (`● / ◐ / ○`) was rejected
  because the `◐` half-circle at 9pt on a coloured pill background
  reads as a chevron rather than a half-fill; the square set has
  unambiguously distinct geometry at any size on any background. See
  `theme.typ` `status-pill()` for the rationale.
- The DRAFT banner is present on the cover and on every interior
  page of the executive summary, in a high-contrast amber-on-black
  treatment (~10:1).

For full external sign-off, run `./build.sh --verify` and test the
built PDFs with a real screen reader (NVDA / VoiceOver). PDF
Accessibility Checker (PAC, Windows-only GUI) is a useful third
opinion alongside veraPDF.

The veraPDF invocation `./build.sh --verify` wraps is, for reference:

```bash
docker run --rm -v "$PWD/out:/work" \
  verapdf/cli:v1.30.1 \
  --flavour ua1 --format text /work/elspeth-executive-summary.pdf
```

Failures in PDF/UA-1 strict should be treated as release-blocking;
warnings are advisory and should be reviewed but do not automatically
block.
