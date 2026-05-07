// ELSPETH Architecture Pack — Typst template for pandoc output
//
// Design: professional architecture-description document in the style of
// ISO/NIST publications.  Adapted from the Wardline Framework Specification
// template (with permission/shared codebase) by removing spec-specific
// regex highlighters (RFC 2119 keywords, rule-ID chips, §-link styling)
// and the Part I/Part II title-page grid.
//
// Fonts:    TeX Gyre Heros (headings), Libertinus Serif (body), Liberation Mono (code)
// Colours:  deep steel blue #1E3A5F (primary), teal #0A6E72 (accent),
//           warm grey rules and shading
//
// Pandoc variables consumed: title, subtitle, author, date, version, status,
//   running-title, doc-id, document-category, document-type, classification,
//   commit, scope-blurb, org-name, org-tagline, revisions[].

// ─────────────────────────────────────────────────────────────
// COLOUR PALETTE
// ─────────────────────────────────────────────────────────────
#let c-navy    = rgb("#1E3A5F")   // primary — headings, title page, rules
#let c-teal    = rgb("#0A6E72")   // accent — links, code border, highlight
#let c-rule    = rgb("#C8CDD3")   // horizontal rules, table borders
#let c-muted   = rgb("#5A6370")   // secondary text (headers, captions, labels)
#let c-shade   = rgb("#F4F5F6")   // code block and table-header fill
#let c-warning = rgb("#7A3B00")   // DRAFT watermark / status chip

// ─────────────────────────────────────────────────────────────
// HEADLINE VERDICT PALETTE (cover findings panel)
// ─────────────────────────────────────────────────────────────
//
// Each finding's `status` field selects an accent color, a pill
// fill, and a pill label.  Centralising these here keeps the
// cover panel's visual vocabulary consistent and easy to extend
// when new status kinds appear.
#let cover-status-color(s) = if s == "verified" {
  rgb("#166534")
} else if s == "fix-verified" {
  rgb("#854D0E")
} else if s == "gap" {
  rgb("#7A3B00")
} else {
  rgb("#5A6370")
}

#let cover-status-pill-fill(s) = if s == "verified" {
  rgb("#DCFCE7")
} else if s == "fix-verified" {
  rgb("#FEF3C7")
} else if s == "gap" {
  rgb("#FEE2E2")
} else {
  rgb("#F4F5F6")
}

#let cover-status-label(s) = if s == "verified" {
  "Verified"
} else if s == "fix-verified" {
  "Fix verified"
} else if s == "gap" {
  "Gap"
} else {
  s
}

// ─────────────────────────────────────────────────────────────
// DOCUMENT METADATA
// ─────────────────────────────────────────────────────────────
#set document(
  title: "$title$",
  author: "$author$",
  keywords: ("elspeth", "architecture", "audit", "sda pipeline", "evidence-anchored"),
)

// PDF bookmarks for outline navigation in viewers.
#set heading(bookmarked: true)

// ─────────────────────────────────────────────────────────────
// TYPOGRAPHY — BASE
// ─────────────────────────────────────────────────────────────
#set text(
  font: ("Libertinus Serif", "DejaVu Serif"),
  size: 10.5pt,
  lang: "en",
  region: "AU",
  hyphenate: true,
  fill: rgb("#1A1A1A"),
)

#set par(
  justify: true,
  leading: 0.72em,
  spacing: 1.1em,
)

// ─────────────────────────────────────────────────────────────
// PAGE GEOMETRY + HEADER / FOOTER
// ─────────────────────────────────────────────────────────────
//
// State:
//   body-started      — true after the front-matter pages (cover + ToC)
//   current-chapter   — most recent level-1 heading body, used by the
//                       running header.
#let body-started = state("body-started", false)
#let current-chapter = state("current-chapter", none)

#set page(
  paper: "a4",
  // Asymmetric margins: wider left (2.8cm) accommodates a binding gutter
  // for A4 print.  Bottom is generous for footer rule + page-number chip.
  margin: (top: 2.6cm, bottom: 2.8cm, left: 2.8cm, right: 2.2cm),

  header: context {
    let pg = counter(page).get().first()
    if pg > 1 and body-started.get() [
      #line(length: 100%, stroke: 0.5pt + c-rule)
      #v(2pt)
      #set text(7.5pt, font: "TeX Gyre Heros", fill: c-muted, tracking: 0.5pt)
      #let chapter = current-chapter.get()
      #if chapter != none {
        upper(chapter)
      } else {
        upper[$running-title$]
      }
    ]
  },

  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #v(2pt)
      #line(length: 100%, stroke: 0.5pt + c-rule)
      #v(3pt)
      #set text(7.5pt, font: "TeX Gyre Heros", fill: c-muted)
      // Left: document identifier + version
      $doc-id$-$version$
      #h(1fr)
      // Centre: page-number chip
      #box(
        fill: c-navy,
        inset: (x: 6pt, y: 2.5pt),
        radius: 2pt,
      )[
        #set text(7pt, fill: white, weight: "bold", tracking: 0.5pt)
        #counter(page).display("1")
      ]
      #h(1fr)
      // Right: status chip
      #box(
        fill: if "$status$" == "DRAFT" { rgb("#FEF3C7") }
             else if "$status$" == "RELEASE CANDIDATE" { rgb("#DBEAFE") }
             else if "$status$" == "INITIAL ISSUE" { rgb("#E0E7FF") }
             else { rgb("#DCFCE7") },
        inset: (x: 5pt, y: 2pt),
        radius: 2pt,
      )[
        #set text(
          6.5pt,
          fill: if "$status$" == "DRAFT" { c-warning }
               else if "$status$" == "RELEASE CANDIDATE" { rgb("#1E40AF") }
               else if "$status$" == "INITIAL ISSUE" { rgb("#3730A3") }
               else { rgb("#166534") },
          weight: "bold",
          tracking: 0.8pt,
          font: "TeX Gyre Heros",
        )
        #upper[$status$]
      ]
    ]
  },
)

// ─────────────────────────────────────────────────────────────
// RUN-IN HEADINGS (LEAD-INS)
// ─────────────────────────────────────────────────────────────
//
// `#leadin[...]` is invoked by leadins.lua for paragraphs whose
// first inline span is a Strong ending in a period or em-dash —
// the structural anchor pattern used throughout the briefing
// ("**Headline result.** The composer's behaviour…").  Rendered
// in small caps + navy + tracking, the lead-in reads as a section
// label embedded in the prose, distinct from ordinary inline bold
// (which stays plain bold serif).
#let leadin(body) = text(
  font: ("Libertinus Serif", "DejaVu Serif"),
  size: 9.5pt,
  weight: "bold",
  fill: rgb("#1E3A5F"),
  tracking: 0.4pt,
)[#smallcaps[#body]]

// ─────────────────────────────────────────────────────────────
// HEADING STYLES
// ─────────────────────────────────────────────────────────────

// Level 1 — Chapter / Part heading.  Forces a page break, draws an
// accent bar + heading text, and updates the running-header state.
#show heading.where(level: 1): it => {
  current-chapter.update(it.body)
  pagebreak(weak: true)
  v(0.8cm)
  grid(
    columns: (6pt, 1fr),
    column-gutter: 10pt,
    rect(width: 6pt, height: 1.5em, fill: c-navy, stroke: none),
    text(
      font: "TeX Gyre Heros",
      size: 17pt,
      weight: "bold",
      fill: c-navy,
    )[#it.body],
  )
  v(0.5em)
  line(length: 100%, stroke: 0.6pt + c-rule)
  v(0.45cm)
}

// Level 2 — Section heading
#show heading.where(level: 2): it => {
  v(0.9em)
  block(width: 100%)[
    #text(
      font: "TeX Gyre Heros",
      size: 13.5pt,
      weight: "bold",
      fill: c-navy,
    )[#it.body]
    #v(-2pt)
    #line(length: 40pt, stroke: 2pt + c-teal)
  ]
  v(0.35em)
}

// Level 3 — Sub-section heading
#show heading.where(level: 3): it => {
  v(0.75em)
  text(
    font: "TeX Gyre Heros",
    size: 11.5pt,
    weight: "bold",
    fill: c-navy,
  )[#it.body]
  v(0.25em)
}

// Level 4 — Minor heading
#show heading.where(level: 4): it => {
  v(0.5em)
  text(
    font: "TeX Gyre Heros",
    size: 10.5pt,
    weight: "bold",
    fill: c-muted,
  )[#it.body]
  v(0.15em)
}

// ─────────────────────────────────────────────────────────────
// CODE BLOCKS
// ─────────────────────────────────────────────────────────────
#show raw.where(block: true): it => {
  set text(8.5pt, font: ("Liberation Mono", "DejaVu Sans Mono"))
  block(
    width: 100%,
    radius: (right: 3pt),
    clip: true,
    fill: c-shade,
    stroke: (left: 3pt + c-teal),
    inset: (left: 12pt, right: 12pt, top: 10pt, bottom: 10pt),
    it,
  )
}

#show raw.where(block: false): it => {
  set text(8pt, font: ("Liberation Mono", "DejaVu Sans Mono"))
  box(
    fill: c-shade,
    stroke: 0.5pt + c-rule,
    inset: (x: 3pt, y: 1.5pt),
    radius: 2pt,
    baseline: 1.5pt,
    it,
  )
}

// ─────────────────────────────────────────────────────────────
// TABLES
// ─────────────────────────────────────────────────────────────
//
// Pandoc emits: #figure(align(center)[#table(columns: ..., table.header([...]),
//                                              table.hline(), [...])])
//
// Strategy:
//   - set table() globally: no internal stroke, generous inset, alternating fills
//   - row 0 (header): navy fill + white bold sans
//   - other rows: alternating light-grey/white, near-black sans body
//   - tables can break across pages

#set table(
  stroke: (x, y) => (
    top: if y <= 1 { 0.5pt + c-rule } else { 0pt },
    bottom: 0pt,
    left: 0pt,
    right: 0pt,
  ),
  fill: (col, row) => {
    if row == 0 { c-navy }
    else if calc.odd(row) { c-shade }
    else { white }
  },
  inset: (x: 9pt, y: 7pt),
  align: left,
)

#show table.cell: it => {
  if it.y == 0 {
    set text(
      font: "TeX Gyre Heros",
      size: 8.5pt,
      weight: "bold",
      fill: white,
      tracking: 0.2pt,
      hyphenate: false,
    )
    it
  } else {
    set par(justify: false)
    set text(
      font: ("TeX Gyre Heros", "Liberation Sans"),
      size: 9pt,
      fill: rgb("#1A1A1A"),
      hyphenate: false,
    )
    // Shrink inline code in table cells to prevent overflow on long tokens
    show raw.where(block: false): r => {
      set text(6.5pt, font: ("Liberation Mono", "DejaVu Sans Mono"))
      box(
        fill: c-shade,
        stroke: 0.5pt + c-rule,
        inset: (x: 1.5pt, y: 0.5pt),
        radius: 1.5pt,
        baseline: 1pt,
        r,
      )
    }
    it
  }
}

#show table: set block(breakable: true)

// ─────────────────────────────────────────────────────────────
// FIGURES (tables and images) WITH NUMBERING
// ─────────────────────────────────────────────────────────────
#set figure(placement: none)
#show figure: set block(breakable: true)

#show figure.where(kind: table): it => {
  // Caption ABOVE the table, standard for technical specs.
  set align(left)
  block(width: 100%, breakable: true)[
    #context {
      let num = counter(figure.where(kind: table)).display()
      text(
        font: "TeX Gyre Heros",
        size: 8pt,
        fill: c-muted,
        weight: "bold",
      )[Table #num#if it.caption != none [: #it.caption.body]]
    }
    #v(4pt)
    #it.body
  ]
}

#show figure.where(kind: image): it => {
  // Image figure (Mermaid diagrams): centered with "Figure N" caption
  block(width: 100%)[
    #align(center)[#it.body]
    #v(4pt)
    #context {
      let num = counter(figure.where(kind: image)).display()
      align(center)[
        #text(
          font: "TeX Gyre Heros",
          size: 8pt,
          fill: c-muted,
          weight: "bold",
        )[Figure #num#if it.caption != none [: #it.caption.body]]
      ]
    }
  ]
}

// ─────────────────────────────────────────────────────────────
// LISTS
// ─────────────────────────────────────────────────────────────
#set list(
  indent: 1.2em,
  body-indent: 0.6em,
  marker: ([#text(fill: c-teal, size: 7pt)[▸]], [–], [·]),
)

#set enum(
  indent: 1.2em,
  body-indent: 0.6em,
  numbering: "1.",
)

// ─────────────────────────────────────────────────────────────
// DEFINITION LISTS (TERMS)
// ─────────────────────────────────────────────────────────────
#show terms: it => {
  set par(spacing: 0.6em)
  for child in it.children {
    block(spacing: 0.7em)[
      #text(
        font: "TeX Gyre Heros",
        size: 10pt,
        weight: "bold",
        fill: c-navy,
      )[#child.term]
      #block(inset: (left: 1.5em, top: 0.2em))[
        #set text(size: 10pt)
        #child.description
      ]
    ]
  }
}

// ─────────────────────────────────────────────────────────────
// LINKS
// ─────────────────────────────────────────────────────────────
#show link: it => {
  set text(fill: c-teal)
  it
}

// ─────────────────────────────────────────────────────────────
// FOOTNOTES
// ─────────────────────────────────────────────────────────────
#show footnote.entry: it => {
  set text(8pt)
  show raw.where(block: false): r => {
    set text(7pt, font: ("Liberation Mono", "DejaVu Sans Mono"))
    box(
      fill: c-shade,
      stroke: 0.5pt + c-rule,
      inset: (x: 2.5pt, y: 1pt),
      radius: 2pt,
      baseline: 1.5pt,
      r,
    )
  }
  it
}

// ─────────────────────────────────────────────────────────────
// BLOCK QUOTES (used for verdict callouts in the executive summary)
// ─────────────────────────────────────────────────────────────
#show quote.where(block: true): it => {
  pad(left: 0pt)[
    #block(
      stroke: (left: 3pt + c-navy),
      inset: (left: 14pt, right: 8pt, top: 8pt, bottom: 8pt),
      fill: rgb("#EEF2F7"),
      radius: (right: 3pt),
      width: 100%,
    )[
      #set text(size: 10pt, style: "normal")
      #set par(leading: 0.65em, spacing: 0.75em)
      #it.body
    ]
  ]
}

// ─────────────────────────────────────────────────────────────
// OUTLINE (TABLE OF CONTENTS)
// ─────────────────────────────────────────────────────────────
#show outline.entry: it => {
  let level = it.level
  let label = it.element.body
  let target = it.element.location()
  let pg = counter(page).at(target).first()

  if level == 1 {
    v(13.5pt, weak: true)
    link(target)[
      #box(width: 100%)[
        #text(
          font: "TeX Gyre Heros",
          size: 10pt,
          weight: "bold",
          fill: c-navy,
        )[#label]
        #h(1fr)
        #text(
          font: "TeX Gyre Heros",
          size: 10pt,
          weight: "bold",
          fill: c-navy,
        )[#str(pg)]
      ]
    ]
  } else if level == 2 {
    v(6.5pt, weak: true)
    link(target)[
      #box(width: 100%)[
        #h(1.4em)
        #text(
          font: "TeX Gyre Heros",
          size: 9pt,
          fill: rgb("#2D3748"),
        )[#label]
        #box(width: 1fr)[
          #set text(fill: rgb("#C8CDD3"), size: 9pt)
          #repeat[.]
        ]
        #text(
          font: "TeX Gyre Heros",
          size: 9pt,
          fill: c-muted,
        )[#str(pg)]
      ]
    ]
  } else {
    let base-indent = 2.8em
    let extra-indent = if level > 3 { (level - 3) * 1.2em } else { 0em }
    let font-size = if level == 3 { 8.5pt } else { 8pt }
    let text-fill = if level == 3 { c-muted } else { rgb("#8899AA") }

    v(if level == 3 { 5.5pt } else { 4pt }, weak: true)
    link(target)[
      #box(width: 100%)[
        #h(base-indent + extra-indent)
        #text(
          font: "TeX Gyre Heros",
          size: font-size,
          fill: text-fill,
        )[#label]
        #box(width: 1fr)[
          #set text(fill: rgb("#C8CDD3"), size: font-size)
          #repeat[.]
        ]
        #text(
          font: "TeX Gyre Heros",
          size: font-size,
          fill: text-fill,
        )[#str(pg)]
      ]
    ]
  }
  linebreak()
}

// ─────────────────────────────────────────────────────────────
// TITLE PAGE
// ─────────────────────────────────────────────────────────────
//
// Design: full-bleed navy header band placed with negative offset to reach
// past the page margins to the physical edge.  Teal accent stripe below.
// Document-control table and scope blurb in body.  Bottom navy band with
// organisation name + tagline.
//
// Page margin: top=2.6cm, left=2.8cm, right=2.2cm.  A4 = 210×297mm.

// ── Top navy bleed band ───────────────────────────────────────
// Height chosen to accommodate up to two lines of 25pt title text
// without the wrapped second line falling under the teal accent
// stripe (the previous 5.5cm clipped 2-line titles).
#place(
  top + left,
  dx: -2.8cm,
  dy: -2.6cm,
  rect(width: 21cm, height: 6.0cm, fill: c-navy)
)

// ── Teal accent stripe immediately below the navy band ────────
#place(
  top + left,
  dx: -2.8cm,
  dy: -2.6cm + 6.0cm,
  rect(width: 21cm, height: 6pt, fill: c-teal)
)

// ── Bottom navy band (organization branding) ──────────────────
$if(org-name)$
#place(
  top + left,
  dx: -2.8cm,
  dy: -2.6cm + 29.7cm - 1.75cm - 6pt,
  rect(width: 21cm, height: 6pt, fill: c-teal)
)

#place(
  top + left,
  dx: -2.8cm,
  dy: -2.6cm + 29.7cm - 1.75cm,
  rect(width: 21cm, height: 1.75cm, fill: c-navy)
)

#place(
  top + left,
  dx: -2.8cm,
  dy: -2.6cm + 29.7cm - 1.0cm,
  block(width: 21cm, height: auto)[
    #pad(left: 2.8cm, right: 2.2cm)[
      #set text(font: "TeX Gyre Heros")
      #text(
        size: 11pt,
        fill: white,
        weight: "bold",
        tracking: 2pt,
      )[#upper[$org-name$]]
      #h(1fr)
      $if(org-tagline)$
      #text(
        size: 8pt,
        fill: rgb("#7FAFD4"),
        tracking: 0.5pt,
      )[$org-tagline$]
      $endif$
    ]
  ]
)
$endif$

// ── Title text over the navy band ─────────────────────────────
//
// `par(justify: false)` overrides the document-wide justification
// so a long title isn't gap-justified across the line ("ELSPETH
// Composer    —    Evaluation"); titles wrap naturally if they
// exceed the 16.5cm block width.  Tighter `leading` keeps a
// 2-line title within the navy band's visible area.
#place(
  top + left,
  dy: 0.5cm,
  block(width: 16.5cm)[
    #set text(font: "TeX Gyre Heros")
    #set par(justify: false, leading: 0.45em)
    #text(
      size: 7.5pt,
      fill: rgb("#7FAFD4"),
      tracking: 2.5pt,
      weight: "bold",
    )[#upper[$document-category$]]
    #linebreak()
    #v(0.25em)
    #text(
      size: 25pt,
      fill: white,
      weight: "bold",
    )[$title$]
  ]
)

// Spacer to clear the bleed band (navy band + teal stripe + room
// for the subtitle to start cleanly below the rule).
#v(3.4cm + 6pt + 1.2cm)

// ── Subtitle ──────────────────────────────────────────────────
// Display type — disable justify + hyphenation so a short headline
// phrase doesn't gap-justify or hyphenate ("Trajec-tory").
#block(width: 100%)[
  #set par(justify: false, leading: 0.5em)
  #set text(hyphenate: false)
  #text(
    font: "TeX Gyre Heros",
    size: 13.5pt,
    fill: c-navy,
    weight: "bold",
  )[$subtitle$]
]

#v(0.5cm)

#line(length: 100%, stroke: 0.5pt + c-rule)

#v(0.6cm)

// ── Document control table ────────────────────────────────────
#table(
  columns: (130pt, 1fr),
  stroke: none,
  inset: (x: 12pt, y: 6pt),
  fill: (col, row) => if row == 0 { c-navy } else if calc.odd(row) { c-shade } else { white },
  table.cell(colspan: 2)[
    #text(
      font: "TeX Gyre Heros",
      size: 7pt,
      fill: rgb("#8BAFD4"),
      weight: "bold",
      tracking: 1.5pt,
    )[#upper[Document Control]]
  ],
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Status]],
  [#text(font: "TeX Gyre Heros", size: 9pt)[$status$ v$version$]],
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Date]],
  [#text(font: "TeX Gyre Heros", size: 9pt)[$date$]],
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Document type]],
  [#text(font: "TeX Gyre Heros", size: 9pt)[$document-type$]],
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Classification]],
  [#text(font: "TeX Gyre Heros", size: 9pt)[$classification$]],
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Identifier]],
  [#text(font: "TeX Gyre Heros", size: 9pt)[$doc-id$-$version$]],
  $if(commit)$
  [#text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted, weight: "bold")[Codebase HEAD]],
  [#text(font: "Liberation Mono", size: 9pt)[$commit$]],
  $endif$
)

#v(0.55cm)

// ── Scope blurb ───────────────────────────────────────────────
//
// Set in the body serif at 10pt (rather than 9pt grey sans) — this
// is the elevator-pitch paragraph for the briefing, not fine-print.
// Italic kept to differentiate from chapter prose and to signal
// "abstract / framing" by typographic convention.
#block(
  stroke: (left: 3pt + c-teal),
  inset: (left: 14pt, right: 12pt, top: 10pt, bottom: 10pt),
  fill: rgb("#EEF2F7"),
  radius: (right: 3pt),
  width: 100%,
)[
  #set text(
    font: ("Libertinus Serif", "DejaVu Serif"),
    size: 10pt,
    fill: c-navy,
    style: "italic",
  )
  #set par(leading: 0.62em, spacing: 0.7em, justify: true)
  $scope-blurb$
]

// ── Headline verdict cards (if provided) ─────────────────────
//
// Three-card horizontal panel that delivers the briefing's
// verdict at-a-glance.  Wording lives in the metadata YAML and
// should be lifted directly from the briefing's executive summary
// — a verdict panel that paraphrases is worse than no panel.
$if(findings)$
#v(0.4cm)
#text(
  font: "TeX Gyre Heros",
  size: 8pt,
  fill: c-muted,
  weight: "bold",
  tracking: 1pt,
)[#upper[Headline verdict]]
#v(0.2cm)
#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 8pt,
  $for(findings)$
  block(
    fill: rgb("#F8FAFC"),
    stroke: (left: 2.5pt + cover-status-color("$findings.status$")),
    inset: (x: 10pt, y: 9pt),
    radius: (right: 2pt),
    width: 100%,
  )[
    #box(
      fill: cover-status-pill-fill("$findings.status$"),
      inset: (x: 5pt, y: 1.5pt),
      radius: 2pt,
    )[
      #text(
        size: 6.5pt,
        font: "TeX Gyre Heros",
        weight: "bold",
        tracking: 0.7pt,
        fill: cover-status-color("$findings.status$"),
      )[#upper[#cover-status-label("$findings.status$")]]
    ]
    #v(5pt)
    #text(
      font: "TeX Gyre Heros",
      size: 9.5pt,
      fill: c-navy,
      weight: "bold",
    )[$findings.title$]
    #v(3pt)
    #block[
      #set par(leading: 0.55em, justify: false)
      #set text(
        size: 7.8pt,
        fill: c-muted,
        style: "italic",
      )
      $findings.note$
    ]
  ],
  $endfor$
)
$endif$

// ── Revision history (if provided) ───────────────────────────
$if(revisions)$
#v(0.4cm)
#text(
  font: "TeX Gyre Heros",
  size: 8pt,
  fill: c-muted,
  weight: "bold",
  tracking: 1pt,
)[#upper[Revision History]]
#v(0.18cm)
#table(
  columns: (70pt, 85pt, 1fr),
  stroke: none,
  inset: (x: 8pt, y: 5pt),
  fill: (col, row) => if row == 0 { c-navy } else if calc.odd(row) { c-shade } else { white },
  [#text(font: "TeX Gyre Heros", size: 7.5pt, fill: white, weight: "bold")[Version]],
  [#text(font: "TeX Gyre Heros", size: 7.5pt, fill: white, weight: "bold")[Date]],
  [#text(font: "TeX Gyre Heros", size: 7.5pt, fill: white, weight: "bold")[Changes]],
  $for(revisions)$
  [#text(font: "TeX Gyre Heros", size: 8pt)[$revisions.version$]],
  [#text(font: "TeX Gyre Heros", size: 8pt)[$revisions.date$]],
  [#text(font: "TeX Gyre Heros", size: 8pt)[$revisions.changes$]],
  $endfor$
)
$endif$

#v(1fr)

#pagebreak()

// ─────────────────────────────────────────────────────────────
// TABLE OF CONTENTS PAGE
// ─────────────────────────────────────────────────────────────
#block(width: 100%)[
  #text(
    font: "TeX Gyre Heros",
    size: 18pt,
    weight: "bold",
    fill: c-navy,
  )[Contents]
  #v(3pt)
  #line(length: 100%, stroke: 1pt + c-navy)
  #v(4pt)
  #line(length: 100%, stroke: 0.4pt + c-rule)
]

#v(0.6cm)

#outline(
  title: none,
  indent: 0pt,
  depth: 3,
)

#v(1cm)

// ─────────────────────────────────────────────────────────────
// LIST OF TABLES (suppressed entirely when no entries carry a
// caption — an uncaptioned LoT lists "Table 1 … Table N" with no
// information, which is worse than not listing at all)
// ─────────────────────────────────────────────────────────────
#context {
  let tables = query(figure.where(kind: table))
  let captioned = tables.filter(fig => fig.caption != none)
  if captioned.len() > 0 {
    block(width: 100%)[
      #text(
        font: "TeX Gyre Heros",
        size: 11pt,
        weight: "bold",
        fill: c-navy,
      )[List of Tables]
      #v(2pt)
      #line(length: 50pt, stroke: 2pt + c-teal)
    ]
    v(0.35cm)
    for (i, fig) in tables.enumerate() {
      let pg = counter(page).at(fig.location()).first()
      let caption-text = if fig.caption != none { fig.caption.body } else { [] }
      box(width: 100%)[
        #text(font: "TeX Gyre Heros", size: 9pt, weight: "medium", fill: c-navy)[
          Table #(i + 1)#if caption-text != [] [:]
        ]
        #if caption-text != [] [
          #h(3pt)
          #text(font: "TeX Gyre Heros", size: 9pt)[#caption-text]
        ]
        #box(width: 1fr)[
          #set text(fill: rgb("#D0D5DC"), size: 9pt)
          #repeat[.]
        ]
        #text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted)[#str(pg)]
      ]
      linebreak()
      v(1.5pt, weak: true)
    }

    if tables.len() > 40 {
      pagebreak()
    } else {
      v(1cm)
    }
  }
}

// ─────────────────────────────────────────────────────────────
// LIST OF FIGURES (suppressed entirely when no figures exist —
// a "No figures in this document" placeholder wastes a half-page)
// ─────────────────────────────────────────────────────────────
#context {
  let figures = query(figure.where(kind: image))
  if figures.len() > 0 {
    block(width: 100%)[
      #text(
        font: "TeX Gyre Heros",
        size: 11pt,
        weight: "bold",
        fill: c-navy,
      )[List of Figures]
      #v(2pt)
      #line(length: 50pt, stroke: 2pt + c-teal)
    ]
    v(0.35cm)
    for (i, fig) in figures.enumerate() {
      let pg = counter(page).at(fig.location()).first()
      let caption-text = if fig.caption != none { fig.caption.body } else { [] }
      block(spacing: 0.5em)[
        #grid(
          columns: (1fr, auto),
          column-gutter: 8pt,
          [
            #text(font: "TeX Gyre Heros", size: 9pt, weight: "medium", fill: c-navy)[
              Figure #(i + 1)#if caption-text != [] [:]
            ]
            #if caption-text != [] [
              #h(3pt)
              #text(font: "TeX Gyre Heros", size: 9pt)[#caption-text]
            ]
          ],
          [
            #text(font: "TeX Gyre Heros", size: 9pt, fill: c-muted)[#str(pg)]
          ],
        )
      ]
    }
  }
}

#pagebreak()

// ─────────────────────────────────────────────────────────────
// BODY CONTENT (from pandoc)
// ─────────────────────────────────────────────────────────────

// From this point the running header is visible.
#body-started.update(true)

$body$
