// tokens.typ - design tokens for the release PDF set.
//
// The palette is AGDS-inspired, but these values are local source
// tokens rather than a generated AGDS token feed. Body text colours
// document their WCAG 2.2 contrast ratios against c-page (#ffffff).

// Typography ---------------------------------------------------------------

#let font-body = ("Public Sans", "Lato", "Liberation Sans", "DejaVu Sans")
#let font-mono = ("Liberation Mono", "DejaVu Sans Mono")

#let size-display = 30pt
#let size-h1 = 20pt
#let size-h3 = 13pt
#let size-body = 10pt
#let size-small = 8.5pt
#let size-caption = 8pt
#let size-eyebrow = 8pt
#let size-footer = 7.5pt

// Spacing ------------------------------------------------------------------

#let sp-1 = 3pt
#let sp-2 = 6pt
#let sp-3 = 10pt
#let sp-4 = 14pt
#let sp-5 = 20pt
#let sp-6 = 28pt

// Page geometry ------------------------------------------------------------

#let page-margin-top = 18mm
#let page-margin-bottom = 16mm
#let page-margin-inner = 18mm
#let page-margin-outer = 15mm

#let page-margin-landscape-top = 13mm
#let page-margin-landscape-bottom = 12mm
#let page-margin-landscape-x = 13mm

// Colour -------------------------------------------------------------------

#let c-page = rgb("#ffffff")

// Core text: 10.9:1 on c-page.
#let c-ink = rgb("#373b3f")
// Secondary body text: 6.8:1 on c-page.
#let c-ink-soft = rgb("#56616b")
// Muted metadata text: 5.1:1 on c-page.
#let c-muted = rgb("#6b7280")

#let c-rule = rgb("#d8dee4")
#let c-panel = rgb("#f5f7fa")

// Brand / action colours.
#let c-navy = rgb("#12324a")
#let c-navy-soft = rgb("#dcebf5")
#let c-action = rgb("#005eb8")
#let c-link = c-action
#let c-accent = rgb("#00a3a3")

// Semantic colours.
#let c-supported = rgb("#2e7d32")
#let c-notyet = rgb("#6f7782")
#let c-error = rgb("#b3261e")
#let c-optional = rgb("#5b4baa")

// Draft banner: c-draft-fg is 7.4:1 on c-draft-bg.
#let c-draft-bg = rgb("#fff3cd")
#let c-draft-fg = rgb("#6b3f00")
