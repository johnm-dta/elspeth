// theme.typ — page master, headings, components.
//
// One file owns the visual contract for the release PDF set. Imports
// pull in tokens.typ.

#import "tokens.typ": *
#import "@preview/cetz:0.4.2"

// ---------------------------------------------------------------------------
// Page master
// ---------------------------------------------------------------------------
//
// Every page in every document gets the unofficial-document disclaimer
// in its footer, and (optionally) a DRAFT banner across the top.
// Cover pages call cover-page() which suppresses the header/footer
// and renders its own banner; interior pages get the standard frame.

// PSPF classification marking. Applied at top AND bottom of every
// page (PSPF Policy 8 / Annex E convention) — bold uppercase, plain
// black on white. No background colour: OFFICIAL is the lowest
// classification tier and adding an amber/orange band would visually
// read as the higher "OFFICIAL: Sensitive" tier.
//
// Change this one line to relabel (e.g. "OFFICIAL: Sensitive",
// "PROTECTED"); the marking propagates to every band AND to the
// default value of the cover-page's `classification:` bibliographic
// parameter, so the cover metadata row stays in sync with the band.
//
// Design caveat (per-document override):
//   This is a module-level constant. The documents in this set
//   currently share the OFFICIAL marking and that is consistent with
//   the unofficial-document framing. If a future document in this set
//   needs a different classification (e.g. an exec summary marked
//   OFFICIAL: Sensitive while progress / velocity stay OFFICIAL),
//   either:
//     (a) parameterise `document-frame` to accept a `classification:`
//         argument that overrides this default; or
//     (b) duplicate this module per-classification.
//   Today we do neither — uniformity across the set is the contract.
//
// PSPF rendering note:
//   `classification-band(fill-colour: c-ink)` uses c-ink (#373b3f,
//   ~10.5:1 AAA against white) by default rather than pure black
//   (#000000). For OFFICIAL this is benign; for higher tiers, strict
//   PSPF template guidance typically specifies pure black. If you
//   relabel above to a higher tier, also re-evaluate the band's fill.
#let default-classification = "OFFICIAL"

// Handling caveat — appears in the footer alongside the classification.
#let handling-caveat = [
  Draft Only — Digital Transformation Agency.
]

#let classification-band(fill-colour: c-ink) = align(center,
  text(font: font-body, size: 11pt, weight: "bold", fill: fill-colour,
    tracking: 3pt, upper(default-classification)))

#let document-frame(
  title: "ELSPETH",
  subtitle: none,
  draft: false,
  // ISO-style date string to stamp inside the DRAFT banner of every
  // interior page header when `draft: true`. None means "render the
  // banner without an explicit date." Documents that want the date
  // stamped pass `draft-date: data.draft-date` so the value stays in
  // sync with the data SoT.
  draft-date: none,
  flipped: false,
  // Whether every level-1 heading should start a fresh page. The
  // executive summary uses page-per-section as a deliberate slide-deck
  // rhythm and keeps the default `true`; the progress and velocity
  // documents are linear engineering reads and pass `false` so short
  // sections flow continuously and don't trail half-empty pages.
  h1-pagebreak: true,
  body,
) = {
  set document(
    title: title,
    author: "John Morrissey",
    description: if subtitle != none { subtitle } else { title },
  )

  set text(font: font-body, size: size-body, lang: "en", fill: c-ink)
  set par(leading: 0.55em, spacing: 1.0em, justify: false)

  let margins = if flipped {
    (top: page-margin-landscape-top, bottom: page-margin-landscape-bottom,
     x: page-margin-landscape-x)
  } else {
    (top: page-margin-top, bottom: page-margin-bottom,
     inside: page-margin-inner, outside: page-margin-outer)
  }

  set page(
    paper: "a4",
    flipped: flipped,
    margin: margins,
    header: context {
      let page-num = counter(page).get().first()
      if page-num == 1 { return }
      classification-band()
      v(sp-1)
      grid(
        columns: (1fr, auto),
        column-gutter: sp-3,
        align(left + horizon, text(size: size-eyebrow, fill: c-muted,
          weight: "medium", upper(title))),
        align(right + horizon, text(size: size-eyebrow, fill: c-muted,
          if subtitle != none [ #subtitle ] else [])),
      )
      v(-sp-2)
      line(length: 100%, stroke: 0.5pt + c-rule)
      if draft {
        v(sp-1)
        block(
          width: 100%,
          inset: (x: sp-3, y: sp-1),
          fill: c-draft-bg,
          radius: 2pt,
        )[
          #set align(center)
          #text(font: font-body, size: 9pt, weight: "bold",
            fill: c-draft-fg, tracking: 2pt)[
            #if draft-date != none [
              DRAFT — AWAITING REVIEW (#draft-date)
            ] else [
              DRAFT — AWAITING REVIEW
            ]
          ]
        ]
      }
    },
    footer: context {
      let page-num = counter(page).get().first()
      if page-num == 1 { return }
      line(length: 100%, stroke: 0.5pt + c-rule)
      v(sp-1)
      grid(
        columns: (1fr, auto),
        column-gutter: sp-3,
        text(size: size-footer, fill: c-muted, handling-caveat),
        text(size: size-footer, fill: c-muted, weight: "medium",
          [Page #counter(page).display() ]),
      )
      v(sp-1)
      classification-band()
    },
  )

  // ---------------- Heading rules ------------------
  set heading(numbering: none)

  show heading.where(level: 1): it => {
    if h1-pagebreak { pagebreak(weak: true) }
    else { v(sp-6) }
    v(sp-4)
    block[
      #grid(
        columns: (3pt, 1fr),
        column-gutter: sp-3,
        rect(width: 3pt, height: 0.9em, fill: c-action, stroke: none),
        text(font: font-body, size: size-h1, weight: "bold",
          fill: c-navy, it.body),
      )
    ]
    v(sp-3)
  }

  show heading.where(level: 2): it => {
    v(sp-4)
    block[
      #text(font: font-body, size: size-eyebrow, weight: "bold",
        fill: c-action, tracking: 1.5pt,
        upper(it.body))
    ]
    v(-sp-1)
    line(length: 30mm, stroke: 1pt + c-action)
    v(sp-2)
  }

  show heading.where(level: 3): it => block[
    #text(font: font-body, size: size-h3, weight: "bold",
      fill: c-navy, it.body)
  ]

  // ---------------- Inline elements ------------------
  show link: it => text(fill: c-link, underline(it.body))
  show raw.where(block: false): it => box(
    fill: c-panel,
    inset: (x: 3pt),
    outset: (y: 2pt),
    radius: 2pt,
    text(font: font-mono, size: 0.9em, fill: c-ink, it),
  )
  show raw.where(block: true): it => block(
    width: 100%,
    fill: c-panel,
    inset: sp-3,
    radius: 3pt,
    stroke: (left: 2pt + c-action),
    text(font: font-mono, size: 9pt, fill: c-ink, it),
  )

  // ---------------- Table styling ------------------
  set table(stroke: none, inset: (x: 8pt, y: 6pt))
  show table.cell.where(y: 0): set text(weight: "bold", fill: c-page,
    size: size-small)

  body
}

// ---------------------------------------------------------------------------
// Cover page
// ---------------------------------------------------------------------------

#let cover-page(
  title: "",
  subtitle: "",
  // Required. Callers MUST pass `doc-date: data.doc-date` so the
  // bibliographic "Document date" row always flows from the data
  // SoT. Earlier revisions of this helper defaulted to a hardcoded
  // ISO string; that default silently rotted whenever `data.doc-date`
  // moved forward and a caller forgot the parameter. An assertion
  // below converts that silent-rot into a build-time failure.
  doc-date: none,
  version: "RC-5.2 (0.5.2)",
  author: "John Morrissey",
  affiliation: none,
  audience: none,
  // Bibliographic-metadata classification value rendered in the
  // cover's document-control block. Defaults to `none` — the
  // running classification band is the PSPF marking, the bib row
  // is optional supplementary metadata. Documents that want the
  // bib row should pass `classification: default-classification`
  // (from theme.typ) rather than re-typing the string, so the
  // bib row cannot drift away from the band's value.
  classification: none,
  status: none,
  distribution: none,
  history: (),
  draft: false,
  // ISO-style date string to render inside the cover's DRAFT banner
  // when `draft: true`. None means the banner is rendered without an
  // explicit date. Pass `draft-date: data.draft-date` to keep this
  // in sync with the data SoT.
  draft-date: none,
  hero: none,
) = {
  assert(doc-date != none,
    message: "cover-page: `doc-date` is required — pass `doc-date: data.doc-date` from the data SoT.")

  set page(
    paper: "a4",
    margin: (top: 0mm, bottom: 0mm, left: 0mm, right: 0mm),
    header: none,
    footer: none,
  )

  // Pale-green text on the deep-green hero — both values are DTA
  // dark-theme tokens (`--ct-color-dark-background-light` and the
  // pale-green border-light) being used here as on-dark text colours.
  let eyebrow-colour = rgb("#c8ebd7")     // ~10:1 on c-navy — AAA
  let subtitle-colour = rgb("#b3d3c8")    // ~7.5:1 on c-navy — AAA

  // OFFICIAL classification band at the very top of the cover, in
  // white on the dark navy/green block (cover is page 1, so the
  // running-header band is suppressed; this is the cover's own
  // classification marking).
  block(
    width: 100%,
    fill: c-navy,
    inset: (left: 22mm, right: 22mm, top: 10mm, bottom: 6mm),
    classification-band(fill-colour: white),
  )

  // Top navy block (title area). Height sized to accommodate DRAFT
  // banner (when present) + eyebrow + title + subtitle without
  // overflowing into the body area below.
  block(
    width: 100%, height: 105mm,
    fill: c-navy,
    inset: (left: 22mm, right: 22mm, top: 8mm, bottom: 18mm),
    {
      set text(fill: white)
      if draft {
        block(
          width: 100%, fill: c-draft-bg, inset: sp-3, radius: 2pt,
          stroke: 1pt + c-draft-fg,
          {
            set align(center)
            text(size: 12pt, weight: "bold", fill: c-draft-fg,
              tracking: 3pt, [DRAFT — AWAITING REVIEW])
            if draft-date != none {
              v(2pt)
              text(size: 9pt, weight: "bold", fill: c-draft-fg,
                tracking: 1pt, [Stamped #draft-date])
            }
            v(2pt)
            text(size: 9pt, fill: white,
              [Claims about assurance status, test counts, and residual risk
              are under operator review. Not for external distribution.])
          }
        )
        v(sp-5)
      } else {
        v(sp-3)
      }
      text(size: size-eyebrow, weight: "medium", tracking: 3pt,
        fill: eyebrow-colour, upper("ELSPETH release documentation"))
      v(sp-3)
      text(font: font-body, size: size-display, weight: "bold",
        fill: white, title)
      if subtitle != "" {
        v(sp-2)
        text(size: 14pt, weight: "regular", fill: subtitle-colour, subtitle)
      }
    },
  )

  // Body.
  block(
    width: 100%,
    inset: (left: 22mm, right: 22mm, top: sp-6, bottom: sp-5),
    {
      if hero != none {
        hero
        v(sp-5)
      }

      // Australian-Government-style document-control block on the
      // cover. Classification first (PSPF convention), then status,
      // then the bibliographic and audience fields, then distribution.
      let cls-row = if classification != none {
        (("Classification", classification),)
      } else { () }
      let st-row = if status != none {
        (("Status", status),)
      } else { () }
      let meta-rows = (
        ("Document date", doc-date),
        ("Release covered", version),
      )
      let extra = if audience != none {
        (("Audience", audience),)
      } else { () }
      let aff = if affiliation != none {
        (("Prepared at", affiliation),)
      } else { () }
      let author-row = (("Author of record", author),)
      let dist-row = if distribution != none {
        (("Distribution", distribution),)
      } else { () }
      let bib-rows = meta-rows + extra + aff + author-row
      let all-rows = cls-row + st-row + bib-rows + dist-row

      grid(
        columns: (auto, 1fr),
        column-gutter: sp-4,
        row-gutter: sp-2,
        ..all-rows.map(((label, value)) => (
          text(size: size-eyebrow, fill: c-muted, weight: "medium",
            tracking: 1pt, upper(label)),
          text(size: size-body, fill: c-ink, value),
        )).flatten(),
      )
    },
  )

  // Footer pinned at the bottom of the cover: handling caveat then
  // the OFFICIAL classification band, mirroring interior-page footers.
  place(
    bottom + left, dx: 22mm, dy: -14mm,
    block(width: 170mm,
      {
        line(length: 100%, stroke: 0.5pt + c-rule)
        v(sp-1)
        text(size: size-footer, fill: c-muted, handling-caveat)
        v(sp-2)
        classification-band()
      },
    ),
  )
  pagebreak()
}

// ---------------------------------------------------------------------------
// Cover hero — abstract SDA flow rendered with native Typst shapes.
// Pure decoration; alt text describes the diagram for assistive
// technology.
// ---------------------------------------------------------------------------

#let cover-hero-sda() = {
  // The hero is decorative; an audit-relevant text description follows
  // below the cover meta block in the document body. We mark the
  // graphics as a PDF artifact so the tag tree does not require alt
  // text, and we keep the visual rendering for sighted readers.
  set align(center)
  pdf.artifact(
    block(
      width: 100%,
      inset: (x: 0pt, y: sp-3),
      {
        set align(center + horizon)
        let pills = (("Sense", c-action), ("Decide", c-navy),
          ("Act", c-action), ("Audit", c-accent))
        stack(dir: ltr, spacing: 12pt,
          ..pills.enumerate().map(((i, item)) => {
            let (label, colour) = item
            let pill = box(
              width: 26mm, height: 13mm,
              fill: colour,
              radius: 3pt,
              inset: 4pt,
              {
                set align(center + horizon)
                set text(fill: white, size: 10.5pt, weight: "bold",
                  tracking: 1pt)
                upper(label)
              },
            )
            if i == 3 {
              stack(dir: ttb, spacing: 4pt,
                text(size: size-eyebrow, fill: c-muted,
                  tracking: 1pt, upper("plus")),
                pill,
              )
            } else if i < 3 {
              stack(dir: ltr, spacing: 8pt,
                pill,
                align(horizon, text(size: 18pt, fill: c-muted, sym.arrow.r)),
              )
            } else {
              pill
            }
          })
        )
      },
    ),
  )
}

// ---------------------------------------------------------------------------
// Components: metric cards, callouts, accent rules
// ---------------------------------------------------------------------------

// metric-card: a label/value/sub-caption tile with a coloured left
// stripe. The stripe carries the *semantic* signal (e.g. supported /
// risk / advisory) while the headline value's *typographic emphasis*
// is what draws the eye. Splitting `colour` (stripe) from
// `value-colour` (text) lets a "risk" card flag risk via a grey or
// red stripe without demoting the headline number itself. If
// `value-colour` is left as `none`, it falls back to `colour` —
// every existing call site keeps its current rendering.
#let metric-card(label, value, sub: none, colour: c-action,
    value-colour: none, height: auto) = {
  let resolved-value-colour = if value-colour == none { colour }
    else { value-colour }
  block(
    width: 100%,
    height: height,
    inset: (x: sp-3, y: sp-3),
    radius: 4pt,
    fill: c-panel,
    stroke: (left: 3pt + colour),
  )[
    #text(size: size-eyebrow, fill: c-muted, weight: "medium",
      tracking: 1pt, upper(label))
    #v(sp-1)
    #text(size: 22pt, weight: "bold", fill: resolved-value-colour, value)
    #if sub != none {
      v(sp-1)
      text(size: size-small, fill: c-ink-soft, sub)
    }
  ]
}

#let callout(kind: "note", title: none, body) = {
  // Palette rationale:
  //   note     -> c-action (info blue) — neutral informational accent.
  //   advisory -> c-accent (DTA warning orange) — "please take note".
  //   risk     -> c-accent (DTA warning orange) — future-tense warning.
  //               Previously c-notyet (grey); grey was semantically weak
  //               for risk, and DTA's error-red is reserved for past-tense
  //               errors (already broken), not future-tense warnings.
  //   success  -> c-supported (DTA success teal) — positive outcome.
  //   error    -> c-error (DTA error red) — actual failure semantics.
  let palette = (
    note: c-action,
    advisory: c-accent,
    risk: c-accent,
    success: c-supported,
    error: c-error,
  )
  let colour = palette.at(kind, default: c-action)
  block(
    width: 100%,
    inset: sp-3,
    radius: 3pt,
    fill: c-panel,
    stroke: (left: 3pt + colour),
  )[
    #if title != none {
      text(size: size-eyebrow, fill: colour, weight: "bold",
        tracking: 1pt, upper(title))
      v(sp-1)
    }
    #body
  ]
}

// Fixed width so every pill in the deployment-readiness table (or any
// other status row) renders at identical size regardless of label
// length. Sized to accommodate the widest expected label
// ("Supported"/"Required") with comfortable padding.
//
// The fixed width has historically been a silent-clip hazard: a future
// label longer than the inner content area would clip without any
// build-time signal. The `context { measure() ; assert() }` block below
// converts that silent failure into a build-time error.
#let status-pill-width = 26mm
// Inner usable width after horizontal inset (6pt on each side).
// Conservative — leaves the 3pt rounded corner clear of the glyph.
#let status-pill-inner-max = status-pill-width - 14pt

// Glyph set rationale:
//   The original `● / ◐ / ○` (filled disk / half-fill / open disk) suffered
//   a legibility regression at 9pt on coloured pill backgrounds: the
//   ◐ (U+25D0 LEFT HALF BLACK CIRCLE) renders correctly but at small size,
//   on a saturated background, the *open* (right) half's outline becomes
//   visually invisible — the glyph reads as a left-pointing chevron rather
//   than a half-fill. Confirmed visually at 300 DPI on RC-5.2 PDFs.
//
//   The square-fill set below preserves the metaphor (filled / partial /
//   open) with three glyphs that share an identical outer bounding box
//   and read unambiguously at small size on any background:
//     ■ (U+25A0 BLACK SQUARE)            — fully filled
//     ▣ (U+25A3 WHITE SQUARE CONTAINING BLACK SMALL SQUARE) — partial
//     □ (U+25A1 WHITE SQUARE)            — empty
//   The ▣ glyph carries its own visible inner-and-outer geometry, so
//   "partial" never collapses into the "filled" or "empty" reading even
//   at low resolution.
//
//   These codepoints fall outside Public Sans's character set (which
//   covers only U+25CA LOZENGE in the Misc Symbols block). They render
//   from Lato or Liberation Sans via the body-font fallback chain in
//   `tokens.typ`. The fallback is verified at build time by inspecting
//   embedded fonts; see `build.sh` preflight.
#let status-pill(label, kind: "supported") = {
  let colour = if kind == "supported" { c-supported }
    else if kind == "optional" { c-optional }
    else { c-notyet }
  let glyph = if kind == "supported" { "■" }
    else if kind == "optional" { "▣" }
    else { "□" }
  let content = text(fill: white, size: size-small, weight: "bold",
    [#glyph #h(3pt) #label])
  context {
    let m = measure(content)
    assert(m.width <= status-pill-inner-max,
      message: "status-pill: label '" + label + "' renders at "
        + repr(m.width) + " which exceeds the inner width "
        + repr(status-pill-inner-max) + " (pill outer "
        + repr(status-pill-width) + "). Either shorten the label or "
        + "widen `status-pill-width` in theme.typ.")
    box(
      width: status-pill-width,
      fill: colour,
      inset: (x: 6pt, y: 2pt),
      outset: (y: 2pt),
      radius: 3pt,
      align(center, content),
    )
  }
}

// ---------------------------------------------------------------------------
// Accessible chart wrapper
// ---------------------------------------------------------------------------
//
// cetz-plot 0.1.3 internally produces equations (axis tick labels) that
// it does not annotate with alt text. Under PDF/UA-1 strict export
// this fails the build. The accessible pattern is:
//
//   1. Render the visual chart inside `pdf.artifact()` — the tag tree
//      treats it as decoration, so the screen-reader user does not
//      encounter unlabelled equations.
//   2. Provide the underlying data immediately afterwards in an
//      accessible `data-table` (or pass `data: none` if data is in
//      the surrounding text already).
//   3. The caption stays in the tag tree as the figure's accessible
//      label, supplemented by the data table.
//
// This satisfies WCAG 2.2 success criterion 1.1.1 (the data is reachable
// in an accessible alternative) and PDF/UA-1 (no untagged content with
// missing alt).
//
// Workaround scope: this wrapper exists because cetz-plot 0.1.x emits
// raw equations for axis ticks (no tag-tree alt text). When cetz-plot
// reaches a version that tags axis labels with accessible text, this
// wrapper is no longer required and `pdf.artifact()` can be lifted —
// the chart body would then participate in the tag tree directly.
// Bump the version pin in `velocity.typ` / `executive-summary.typ` /
// `versions.lock` together and re-test PDF/UA-1 strict before
// unwrapping.

#let chart-figure(
  caption: [],
  description: none,
  data: none,
  data-caption: none,
  width: 100%,
  body,
) = {
  // Group description + chart + caption inside an unbreakable block so
  // the visual stays together; the underlying-data table is emitted
  // outside that block and is allowed to break across pages.
  block(width: width, breakable: false, {
    if description != none {
      block(inset: (bottom: sp-2),
        text(size: size-small, fill: c-ink-soft, description))
    }
    pdf.artifact(body)
    block(inset: (top: sp-2),
      text(size: size-caption, fill: c-muted, style: "italic", caption))
  })
  if data != none {
    v(sp-2)
    block(
      width: 100%,
      fill: c-panel,
      inset: sp-3,
      radius: 3pt,
      stroke: (left: 2pt + c-rule),
      breakable: true,
      {
        text(size: size-eyebrow, fill: c-muted, weight: "medium",
          tracking: 1pt, upper(if data-caption != none { data-caption }
          else { "Underlying data" }))
        v(sp-1)
        data
      },
    )
  }
}

// ---------------------------------------------------------------------------
// Table helper — branded header row, alternating fills, breakable.
// ---------------------------------------------------------------------------

// data-table: branded header row with optional independent header
// alignment. `align-rules` controls the body-row alignment per
// column. `header-align`, when supplied, overrides the header row's
// alignment per column independently — useful when the body is
// right-aligned for numerics but the header reads better centred.
// When `header-align` is `none` (default), the header inherits
// `align-rules` so existing call sites are unchanged.
#let data-table(columns: none, header: (), align-rules: none,
    header-align: none, ..rows) = {
  let data = rows.pos()
  let n = if columns == none { header.len() } else { columns }
  // Defensive: a mismatched header / columns / align-rules length is a
  // structural authoring bug — Typst's table() will paper over it by
  // truncating or repeating, and the resulting visual error (misaligned
  // columns, wrong header fill) is easy to miss in PDF review. Fail
  // loudly at compile time instead of silently rendering a wrong table.
  assert(header.len() == n,
    message: "data-table: header has " + str(header.len())
      + " entries but columns=" + str(n) + ". They must match.")
  if align-rules != none {
    assert(align-rules.len() == n,
      message: "data-table: align-rules has " + str(align-rules.len())
        + " entries but columns=" + str(n) + ". They must match.")
  }
  if header-align != none {
    assert(header-align.len() == n,
      message: "data-table: header-align has " + str(header-align.len())
        + " entries but columns=" + str(n) + ". They must match.")
  }
  let body-aligns = if align-rules == none {
    range(0, n).map(_ => left + horizon)
  } else { align-rules }
  let head-aligns = if header-align == none { body-aligns }
    else { header-align }
  table(
    columns: n,
    align: (col, row) => if row == 0 { head-aligns.at(col) }
                          else { body-aligns.at(col) },
    stroke: none,
    inset: (x: 8pt, y: 6pt),
    fill: (col, row) => {
      if row == 0 { c-navy }
      else if calc.even(row) { c-panel }
      else { white }
    },
    table.header(
      ..header.map(h => text(fill: white, weight: "bold",
        size: size-small, h)),
    ),
    ..data.flatten(),
  )
}

// ---------------------------------------------------------------------------
// Native-Typst body diagrams (cetz)
// ---------------------------------------------------------------------------
//
// These diagrams replace the previous Mermaid PNGs. The migration was driven
// by four concrete wins:
//   1. Vector at any zoom — no more diagonal-arrowhead aliasing at >200%.
//   2. Font fidelity — same Public Sans render path as body text, no
//      themeCSS-override gymnastics required to defeat Mermaid's neutral
//      theme + headless-chromium serif fallback.
//   3. Build simplification — mmdc / puppeteer / headless chromium drop
//      out of the toolchain. typst is the only renderer.
//   4. Deterministic — same Typst version produces byte-identical output.
//
// Caller contract:
//   Wrap the call site in `figure(... caption: ..., supplement: [Figure])`
//   to keep the alt-text path uniform with image-based figures. The
//   accessible description still lives in the figure caption.

// Layout helpers — keep magic numbers in one place. cetz coordinates
// are in cm; one cm at the page width below renders as ~3.78mm.

// Helper: stage rectangle with a centered label. Two important
// shadowing traps inside a cetz canvas:
//   1. Typst-builtin `align()` is shadowed by cetz's `align` (a
//      directive, not a content wrapper). We pass labels as plain
//      text content and let cetz's `content()` center on the anchor.
//   2. `import cetz.draw: *` exports a `fill` symbol (cetz's fill
//      function). A parameter named `fill` is silently shadowed by
//      that import, so `fill: fill` passes cetz's function value to
//      `rect`'s fill argument and produces the obscure
//      "expected color, found function" error from canvas.typ:165.
//      We use `fillc` and `strokec` to avoid the collision.
#let _diagram-node(pos, size, fillc, body, strokec: none, font-size: 10pt) = {
  import cetz.draw: *
  let (x, y) = pos
  let (w, h) = size
  rect((x, y), (x + w, y + h),
    fill: fillc,
    stroke: if strokec == none { none } else { 1pt + strokec },
    radius: 1mm)
  content((x + w / 2, y + h / 2),
    text(font: font-body, size: font-size, weight: "bold",
      fill: if fillc == white { c-ink } else { white }, body))
}

// SDA flow — replaces diagrams/sda-flow.png. Renders the external →
// Sense → Decide → Act → consumer chain with all three stages dashed-
// arrowing into the audit cylinder. The audit-trail node is drawn as
// a cylinder primitive to preserve the database-shape semantic the
// Mermaid version carried.
#let diagram-sda-flow() = cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // External source (top, white box, dark stroke).
  rect((0, 11.5), (5.5, 13.2), fill: white,
    stroke: 1pt + c-muted, radius: 1mm)
  content((2.75, 12.35),
    text(font: font-body, size: 9.5pt, weight: "bold",
      fill: c-ink)[
        External source system\
        (Dataverse, CSV, JSON,\
        blob, queue, API)
      ])

  // ELSPETH pipeline subgraph border.
  rect((-0.4, 2.8), (5.9, 10.8),
    fill: rgb("#fafafa"),
    stroke: (paint: c-rule, thickness: 0.5pt, dash: "dashed"),
    radius: 1.5mm)
  content((2.75, 10.45),
    text(font: font-body, size: size-eyebrow, fill: c-muted,
      weight: "medium", tracking: 1pt, upper("ELSPETH pipeline")))

  // Three stages.
  _diagram-node((0, 8.2), (5.5, 1.7), c-navy,
    [Sense\ Source\ load + validate])
  _diagram-node((0, 5.7), (5.5, 1.7), c-navy,
    [Decide\ Transforms / Gates\ rules · models · LLM])
  _diagram-node((0, 3.2), (5.5, 1.7), c-navy,
    [Act\ Sinks\ write + dispatch])

  // Downstream consumer (white, outside pipeline).
  rect((0.5, 0.6), (4.5, 2.1), fill: white,
    stroke: 1pt + c-muted, radius: 1mm)
  content((2.5, 1.35),
    text(font: font-body, size: 9.5pt, weight: "bold",
      fill: c-ink)[Downstream\ consumer])

  // Audit cylinder — body rect + top ellipse rim + bottom front-arc.
  // We use `circle(center, radius: (rx, ry))` to draw the elliptical
  // rims (cetz 0.4's circle accepts an (rx, ry) tuple for ellipses).
  // The top rim sits on top of the rect (full ellipse visible); the
  // bottom rim is drawn as a half-ellipse via path composition so
  // only the front half shows. The cylinder is the canonical
  // ER-diagram "database" shape.
  let audit-x = 7.5
  let audit-w = 4.2
  let audit-top = 7.5
  let audit-bottom = 3.5
  let audit-rx = audit-w / 2
  let audit-ry = 0.4
  let audit-cx = audit-x + audit-w / 2
  let audit-fill = c-accent
  // Body rect (the cylinder's vertical wall).
  rect((audit-x, audit-bottom), (audit-x + audit-w, audit-top),
    fill: audit-fill, stroke: 2pt + c-navy)
  // Top rim — full ellipse, painted over the rect's top edge.
  circle((audit-cx, audit-top), radius: (audit-rx, audit-ry),
    fill: audit-fill, stroke: 2pt + c-navy)
  // Bottom rim — front half only. We paint a full ellipse in the
  // body's fill colour to erase the rect's bottom edge in front,
  // then stroke just the front arc.
  circle((audit-cx, audit-bottom), radius: (audit-rx, audit-ry),
    fill: audit-fill, stroke: none)
  arc((audit-cx - audit-rx, audit-bottom), start: 180deg, stop: 360deg,
    radius: (audit-rx, audit-ry),
    stroke: 2pt + c-navy, anchor: "origin")
  content((audit-cx, 5.5),
    text(font: font-body, size: 9.5pt, weight: "bold",
      fill: white)[Audit trail\ (Landscape DB)\ hash · lineage · provenance])

  // Solid arrows: External → Sense → Decide → Act → Consumer.
  line((2.75, 11.5), (2.75, 9.9), mark: (end: "stealth", fill: c-ink-soft),
    stroke: 1pt + c-ink-soft)
  line((2.75, 8.2), (2.75, 7.4), mark: (end: "stealth", fill: c-ink-soft),
    stroke: 1pt + c-ink-soft)
  line((2.75, 5.7), (2.75, 4.9), mark: (end: "stealth", fill: c-ink-soft),
    stroke: 1pt + c-ink-soft)
  line((2.75, 3.2), (2.75, 2.1), mark: (end: "stealth", fill: c-ink-soft),
    stroke: 1pt + c-ink-soft)

  // Dashed "record" arrows: each stage → audit cylinder.
  let record-stroke = (paint: c-muted, thickness: 0.8pt, dash: "dashed")
  let record-mark = (end: "stealth", fill: c-muted)
  line((5.5, 9.05), (audit-x + audit-w / 2, audit-top - 0.05),
    stroke: record-stroke, mark: record-mark)
  line((5.5, 6.55), (audit-x + audit-w / 2 - 0.4, audit-top - 0.6),
    stroke: record-stroke, mark: record-mark)
  line((5.5, 4.05), (audit-x + 0.2, audit-bottom + 0.3),
    stroke: record-stroke, mark: record-mark)

  // Edge labels for the record arrows. Positioned at the midpoint of
  // each dashed line, offset slightly from the line so they read as
  // associated rather than overlapping.
  let label = (pos, body) => content(pos,
    box(fill: white, inset: 1pt,
      text(font: font-body, size: 7.5pt, fill: c-muted,
        style: "italic", body)))
  label((6.5, 8.9), [record])
  label((6.2, 6.4), [record])
  label((6.0, 3.9), [record])
})

// Trust-tier diagram — three vertical tier subgraphs with inner
// content boxes and labeled boundary edges, plus a dashed read-guard
// back-arrow from Tier 1 to Tier 2.
#let diagram-trust-tiers() = cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Tier 3 (top): zero-trust external, pale orange fill.
  let t3-fill = rgb("#fde7d4")
  let t3-stroke = c-accent
  rect((0, 10.5), (12, 13), fill: t3-fill, stroke: 1pt + t3-stroke,
    radius: 1.5mm)
  content((1.5, 12.6), anchor: "north-west",
    text(font: font-body, size: size-eyebrow, fill: t3-stroke,
      weight: "bold", tracking: 1pt,
      upper("Tier 3 — External (Zero trust)")))
  // Inner content boxes — two side-by-side.
  rect((0.5, 10.9), (5.7, 12.1), fill: white,
    stroke: 0.5pt + t3-stroke, radius: 1mm)
  content((3.1, 11.5),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: c-ink)[Source plugins\ CSV · JSON · Dataverse · Blob · Web])
  rect((6.3, 10.9), (11.5, 12.1), fill: white,
    stroke: 0.5pt + t3-stroke, radius: 1mm)
  content((8.9, 11.5),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: c-ink)[Validate · coerce · quarantine\ Record absence as None])

  // Tier 2 (middle): elevated trust, pale teal fill.
  let t2-fill = rgb("#e9f2ef")
  let t2-stroke = c-action
  rect((0, 6), (12, 8.5), fill: t2-fill, stroke: 1pt + t2-stroke,
    radius: 1.5mm)
  content((1.5, 8.1), anchor: "north-west",
    text(font: font-body, size: size-eyebrow, fill: t2-stroke,
      weight: "bold", tracking: 1pt,
      upper("Tier 2 — Pipeline (Elevated trust)")))
  rect((0.5, 6.4), (5.7, 7.6), fill: white,
    stroke: 0.5pt + t2-stroke, radius: 1mm)
  content((3.1, 7),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: c-ink)[Transforms · Gates · Aggregations])
  rect((6.3, 6.4), (11.5, 7.6), fill: white,
    stroke: 0.5pt + t2-stroke, radius: 1mm)
  content((8.9, 7),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: c-ink)[Type-safe values\ No coercion · wrap operations])

  // Tier 1 (bottom): full trust, navy fill, white text.
  rect((0, 1.5), (12, 4), fill: c-navy, stroke: 1pt + c-navy,
    radius: 1.5mm)
  content((1.5, 3.6), anchor: "north-west",
    text(font: font-body, size: size-eyebrow, fill: white,
      weight: "bold", tracking: 1pt,
      upper("Tier 1 — Our data (Full trust)")))
  rect((0.5, 1.9), (5.7, 3.1), fill: c-navy-soft,
    stroke: 0.5pt + white, radius: 1mm)
  content((3.1, 2.5),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: white)[Landscape audit DB\ Checkpoint state · Hashes])
  rect((6.3, 1.9), (11.5, 3.1), fill: c-navy-soft,
    stroke: 0.5pt + white, radius: 1mm)
  content((8.9, 2.5),
    text(font: font-body, size: 8.5pt, weight: "bold",
      fill: white)[Crash on any anomaly\ Pristine at all times])

  // Boundary edges: T3 → T2 and T2 → T1 (solid down arrows with labels).
  let boundary-stroke = 1pt + c-ink-soft
  let boundary-mark = (end: "stealth", fill: c-ink-soft)
  line((4, 10.5), (4, 8.5), stroke: boundary-stroke, mark: boundary-mark)
  content((4, 9.5), anchor: "east",
    box(fill: white, inset: 2pt,
      text(font: font-body, size: 8pt, weight: "bold", fill: c-ink-soft,
        [Boundary: validate, coerce, quarantine])))

  line((4, 6), (4, 4), stroke: boundary-stroke, mark: boundary-mark)
  content((4, 5), anchor: "east",
    box(fill: white, inset: 2pt,
      text(font: font-body, size: 8pt, weight: "bold", fill: c-ink-soft,
        [Boundary: read straight, write atomic])))

  // Dashed read-guard back-arrow: T1 → T2 (right side, going up).
  let guard-stroke = (paint: c-accent, thickness: 1pt, dash: "dashed")
  let guard-mark = (end: "stealth", fill: c-accent)
  line((9, 4), (9, 6), stroke: guard-stroke, mark: guard-mark)
  content((9, 5), anchor: "west",
    box(fill: white, inset: 2pt,
      text(font: font-body, size: 8pt, weight: "bold", fill: c-accent,
        [Read guard: TIER_1_ERRORS])))
})
