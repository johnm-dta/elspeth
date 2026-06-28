// velocity.typ — ELSPETH Velocity Report, RC-1 to RC-5
// Audience: engineering reviewers. Data-heavy.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as d
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart, plot

#show: document-frame.with(
  title: "ELSPETH Velocity Report",
  subtitle: "Delivery cadence, RC-1 to RC-5",
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Velocity Report",
  subtitle: "Delivery cadence, RC-1 to RC-5.",
  doc-date: d.doc-date,
  version: "RC-5.2 (0.5.2)",
  author: "John Morrissey (sole contributor)",
  affiliation: "Digital Transformation Agency",
  audience: "DTA Architecture, Security and Technical staff",
  hero: cover-hero-sda(),
)

= Reading the figures

#callout(kind: "advisory", title: "Single-author cadence")[
  All commit volume in this document reflects the work of a single
  author. The per-active-day rates (36--80 commits) should be read
  in that context: they are the cadence of one person, not a team.
  The pattern of low-activity weeks immediately preceding phase
  changes (see #emph[Velocity Observations] below) is consistent
  with sustained solo work, not team scheduling.
]

This document is the #emph[velocity / tempo] view. For cumulative
feature output, see the companion #emph[Progress] document.

= Headline statistics

// Stripe colours rotate across the full DTA palette rather than
// defaulting to teal/blue for every card. Each card's stripe carries
// a distinct semantic register:
//   c-action     — informational quantity (period, mean, days)
//   c-supported  — positive outcome (active days, commit count)
//   c-navy-soft  — structural anchor (total commits)
//   c-accent     — attention-drawing peak (peak day, headline event)
#grid(columns: (1fr, 1fr, 1fr), gutter: sp-3,
  metric-card("Period", "128 days",
    sub: "12 Jan -- 19 May 2026",
    colour: c-action),
  metric-card("Active commit days", "123",
    sub: "5 idle days",
    colour: c-supported),
  metric-card("Total commits", "4,521",
    sub: "Unique across RC history snapshots",
    colour: c-navy-soft),
)

#v(sp-2)

#grid(columns: (1fr, 1fr, 1fr), gutter: sp-3,
  metric-card("Mean per active day", "36.8",
    sub: "Median 27. Mean per calendar day 35.3.",
    colour: c-action),
  metric-card("Peak day", "177 commits",
    sub: "2026-01-20 -- pre-RC-1 LLM pooled-execution land",
    colour: c-accent),
  metric-card("Days >= 50 commits", "31",
    sub: "Days >= 100 commits: 6",
    colour: c-supported),
)

= Top 15 peak days

The peak days where epics land or RC cutovers happen. Attribution is
taken from the dominant feature-commit cluster on that day.

#data-table(
  columns: 4,
  header: ([Rank], [Date], [Commits], [What landed]),
  align-rules: (right + horizon, left + horizon, right + horizon,
    left + horizon),
  ..d.peak-days.map(((rank, date, n, attr)) => (
    [#str(rank)], [#date], [#str(n)], [#attr],
  )),
)

#v(sp-3)

== Peak-day callouts

// Peak-card stripe and headline-number colour now vary per-card so
// the four-card grid expresses more of the DTA palette than the
// previous all-blue treatment:
//   c-accent     — single-day project record (Jan 20, 177 commits)
//   c-action     — informational hot-day (Jan 30 hardening burst)
//   c-supported  — composer maturation day (May 12 MANIFEST pass)
//   c-navy-soft  — release-stamp day (May 14 RC-5.2 cutover)
#let peak-card(date, n, attr, colour: c-action) = block(
  width: 100%,
  inset: sp-3,
  radius: 3pt,
  fill: c-panel,
  stroke: (left: 3pt + colour),
  {
    grid(
      columns: (auto, 1fr),
      column-gutter: sp-4,
      align(top, {
        text(size: 26pt, weight: "bold", fill: colour, str(n))
        v(-sp-2)
        text(size: size-eyebrow, fill: c-muted,
          tracking: 1pt, upper("commits"))
      }),
      align(top, {
        text(size: size-small, fill: c-muted, weight: "medium",
          tracking: 1pt, upper(date))
        v(sp-1)
        text(size: size-body, fill: c-ink, attr)
      }),
    )
  },
)

#grid(columns: (1fr, 1fr), gutter: sp-2,
  peak-card("2026-01-20", 177,
    "Heaviest single day in project history. Pre-RC1 LLM pooled-execution + batch aggregation: AzureLLMTransform batch + pooled, OpenRouter _process_batch, public pooled-execution API.",
    colour: c-accent),
  peak-card("2026-01-30", 156,
    "RC-1 hardening burst: ChaosLLM weighted error selection + metrics export; AuditIntegrityError / OrchestrationInvariantError introduction; Azure Monitor + Datadog exporters.",
    colour: c-action),
  peak-card("2026-05-12", 142,
    "RC-5.2 composer redaction-MANIFEST pass: guided step_2_chosen_plugin; RecipeOfferTurn editable slots; ARG_ERROR canonicalization via Pydantic __cause__.",
    colour: c-supported),
  peak-card("2026-05-14", 97,
    "RC-5.2 release-stamp day: changelog finalize; per-step chat merge; phase3 compose-loop persistence merge.",
    colour: c-navy-soft),
)

// ---------------------------------------------------------------------------
// HERO CHART: daily commits — landscape page.
// ---------------------------------------------------------------------------

#pagebreak()

// Margins sized to fit the OFFICIAL classification marking with
// visible whitespace at the page edges (matches the global landscape
// geometry in tokens.typ, page-margin-landscape-*).
#page(flipped: true, margin: (top: 24mm, bottom: 24mm, x: 18mm))[
  #v(sp-2)
  #block[
    #grid(columns: (3pt, 1fr), column-gutter: sp-3,
      rect(width: 3pt, height: 0.9em, fill: c-action, stroke: none),
      text(font: font-body, size: size-h1, weight: "bold",
        fill: c-navy, [Daily commits, 12 Jan -- 19 May 2026]),
    )
  ]
  #v(sp-2)

  #text(size: size-small, fill: c-ink-soft)[
    Bar chart of commits per calendar day across the entire 128-day
    project period. Five idle days (2026-02-23, 03-11, 03-16, 03-17,
    04-06) show as zero-height bars. Peak: 177 on 2026-01-20.
    Mean across active days: 36.8. Bars at or above 100 commits are
    annotated with their numeric values so the high-tempo days are
    legible without relying on the colour-only legend.
    Underlying daily values appear in the #emph[Full per-day commit
    table] on page #context counter(page).at(<full-per-day-table>).first().
  ]
  #v(sp-3)

  #pdf.artifact(
    cetz.canvas({
      import cetz.draw: *
      // Hand-draw bars via cetz primitives — gives per-bar colour and
      // a guaranteed-clean render under PDF/UA-1.
      let pts = d.daily-commits
      let n = pts.len()
      let chart-w = 24.0
      let chart-h = 10.0
      let y-max = 200.0
      let bar-w = chart-w / n * 0.9

      // Plot area: (0,0) bottom-left, (chart-w, chart-h) top-right.
      group(name: "plot", {
        // Y axis ticks + gridlines.
        for tv in (0, 25, 50, 75, 100, 125, 150, 175, 200) {
          let y = tv / y-max * chart-h
          line((0, y), (chart-w, y),
            stroke: (paint: c-rule, thickness: 0.3pt))
          content((-0.2, y), anchor: "east",
            text(size: 8pt, fill: c-ink-soft, str(tv)))
        }
        // Y axis label.
        content((-1.6, chart-h / 2), anchor: "center",
          angle: 90deg,
          text(size: 9pt, fill: c-ink-soft, "Commits per day"))

        // Bars. Each bar's colour encodes the activity tier and the
        // numeric value of >=100-commit days is annotated above the
        // bar so a deuteranopic / monochrome reader can identify
        // the high-tempo days without resolving the colour legend
        // (WCAG 1.4.1 — information not by colour alone).
        for (i, item) in pts.enumerate() {
          let (date-str, v) = item
          let x = (i + 0.5) / n * chart-w
          let y = v / y-max * chart-h
          let colour = if v >= 100 { c-accent }
            else if v >= 50 { c-action }
            else if v == 0 { c-muted }
            else { c-navy }
          rect((x - bar-w / 2, 0), (x + bar-w / 2, y),
            fill: colour, stroke: none)
          if v >= 100 {
            // Label above the bar in the same accent tone. Adjacent
            // >=100 days would visually collide at the default offset
            // (e.g. 2026-02-02 = 118 vs 2026-02-03 = 125), so when the
            // previous bar is also >= 100, raise this label so the two
            // form a staircase rather than overlapping in place.
            // WCAG 1.4.1: the numeric label augments the colour-only
            // tier signal, so legibility under all conditions matters.
            let prev-v = if i > 0 { pts.at(i - 1).at(1) } else { 0 }
            let label-offset = if prev-v >= 100 { 0.75 } else { 0.2 }
            content((x, y + label-offset), anchor: "south",
              text(size: 7pt, weight: "bold", fill: c-accent, str(v)))
          }
        }

        // Mean reference line (36.8).
        let mean-y = 36.8 / y-max * chart-h
        line((0, mean-y), (chart-w, mean-y),
          stroke: (paint: c-notyet, dash: "dashed", thickness: 0.8pt))

        // X axis baseline.
        line((0, 0), (chart-w, 0),
          stroke: (paint: c-ink-soft, thickness: 0.6pt))

        // X axis ticks at named milestones.
        let tick-positions = (
          ("Jan 12", 0),
          ("Jan 22", 10),
          ("Feb 2", 21),
          ("Feb 22", 41),
          ("Mar 10", 57),
          ("Mar 29", 76),
          ("Apr 17", 95),
          ("May 6", 114),
          ("May 19", 127),
        )
        for (lbl, idx) in tick-positions {
          let x = (idx + 0.5) / n * chart-w
          line((x, 0), (x, -0.2),
            stroke: (paint: c-ink-soft, thickness: 0.5pt))
          content((x, -0.45), anchor: "north",
            text(size: 8pt, fill: c-ink-soft, lbl))
        }
      })
    }),
  )
  #v(sp-2)

  #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: sp-3,
    {
      stack(dir: ltr, spacing: 6pt,
        rect(width: 10pt, height: 10pt, fill: c-accent, stroke: none),
        text(size: size-small, [Day >= 100 commits]),
      )
    },
    {
      stack(dir: ltr, spacing: 6pt,
        rect(width: 10pt, height: 10pt, fill: c-action, stroke: none),
        text(size: size-small, [Day 50--99 commits]),
      )
    },
    {
      stack(dir: ltr, spacing: 6pt,
        rect(width: 10pt, height: 10pt, fill: c-navy, stroke: none),
        text(size: size-small, [Day 1--49 commits]),
      )
    },
    {
      stack(dir: ltr, spacing: 6pt,
        line(start: (0pt, 5pt), end: (12pt, 5pt),
          stroke: (paint: c-notyet, dash: "dashed", thickness: 1pt)),
        text(size: size-small, [Mean (36.8) across active days]),
      )
    },
  )
]

// 7-column table is landscape-only — portrait A4's inner width (171mm)
// forces brutal wrapping on the Phase / Dates / Headline columns.
// Landscape (A4 flipped, inner width ~261mm) gives every column room
// without sacrificing the Headline data. The H1 heading and intro
// paragraph live inside the landscape page block so the section
// reads as a unit; rendering the heading on a portrait page before
// the landscape table would leave an orphan portrait page above.
#page(flipped: true, margin: (top: 24mm, bottom: 24mm, x: 18mm))[
  = Velocity by phase

  The seven canonical phases of the project, with per-day averages.

  #v(sp-3)

  #data-table(
    columns: 7,
    header: ([Phase], [Dates], [Cal. days], [Active], [Commits],
      [Per active day], [Headline]),
    align-rules: (left + horizon, left + horizon, right + horizon,
      right + horizon, right + horizon, right + horizon, left + horizon),
    ..d.phase-totals.map(((phase, dates, cal, active, total, rate, head)) => (
      [#phase], [#dates], [#str(cal)], [#str(active)],
      [#str(total)], [#d.fmt-1dec(rate)], [#head],
    )),
  )
]

// Back to portrait for the chart and observations below.

// Back to portrait for the chart and observations below.

#v(sp-3)

#chart-figure(
  caption: [Per-phase total commits. Architectural-remediation phases
    (RC-3, RC-4 + RC-5 cut, RC-5.1) sit between 475 and 810 commits
    each. Hot cuts (Pre-RC1, RC-1 Hardening, RC-5.2) sit at 638--811
    despite shorter calendar windows.],
  description: [Bar chart of total commits per phase across the seven
    canonical phases. Heights: 782 (Pre-RC1 Foundation), 811 (RC-1
    Hardening), 527 (RC-2 Sub-releases), 475 (RC-3 Series), 478 (RC-4
    plus RC-5 cut), 810 (RC-5.1 Composer Correctness), 638 (RC-5.2
    Composer Maturation).],
  data: data-table(
    header: ([Phase], [Commits], [Active days], [Per day]),
    align-rules: (left + horizon, right + horizon, right + horizon,
      right + horizon),
    ..d.phase-totals.map(((phase, dates, cal, active, total, rate, head)) => (
      [#phase], [#str(total)], [#str(active)], [#d.fmt-1dec(rate)],
    )),
  ),
  cetz.canvas({
    let phase-rows = d.phase-totals.map(((phase, dates, cal, active, total, rate, head)) => {
      let short = phase
        .replace("Pre-RC1 Foundation", "Pre-RC1")
        .replace("RC-1 Hardening", "RC-1 Harden")
        .replace("RC-2 Sub-releases", "RC-2 Sub")
        .replace("RC-3 Series", "RC-3")
        .replace("RC-4 + RC-5 cut", "RC-4 + cut")
        .replace("RC-5.1 Composer Correctness", "RC-5.1")
        .replace("RC-5.2 Composer Maturation", "RC-5.2")
      (short, total)
    })
    chart.barchart(
      phase-rows,
      mode: "basic",
      size: (14, 6),
      bar-style: i => (fill: c-action, stroke: none),
      label-key: 0,
      value-key: 1,
      x-tick-step: 100,
    )
  }),
)

== Reading the table

#callout(title: "Three velocity tiers")[
  Velocity has three tiers, not two:
  - #strong[Hot cuts (Pre-RC1, RC-1 Hardening, RC-5.2 Composer Maturation)]
    — 71--80 commits / active day. Empty-scaffold-to-shipping
    foundation work, plus high-tempo merge sprints around major
    releases.
  - #strong[Mid-cadence sub-release stream (RC-2)] — 52.7 commits /
    active day. Five rapid sub-releases each \~3--5 days apart, with
    shorter individual review cycles.
  - #strong[Architectural-remediation and correctness sprints
    (RC-3, RC-4 + RC-5 cut, RC-5.1)] — 19--23 commits / active day.
    Smaller, more carefully audited changes; each commit tends to be
    a focused fix with paired tests.

  The current RC-5.2 surge (12--19 May) is the #strong[highest
  sustained tempo since RC-1 cutover] — 79.8 commits / active day over
  8 consecutive active days.
]

= Per-week summary

For a coarser view, the calendar weeks (Monday-anchored) with totals
and themes.

#chart-figure(
  caption: [Weekly commit totals. Mean across all 19 weeks (including
    the partial W19): 238 commits / week.],
  description: [Bar chart of commit totals across 19 calendar weeks.
    Heights span from 53 (W09) to 520 (W03). The mean is shown as a
    horizontal dashed reference line.],
  data: data-table(
    header: ([Week], [Dates], [Commits], [Theme]),
    align-rules: (left + horizon, left + horizon, right + horizon,
      left + horizon),
    ..d.weekly-summary.map(((wk, dates, total, head)) => (
      [#wk], [#dates], [#str(total)], [#head],
    )),
  ),
  cetz.canvas({
    let week-rows = d.weekly-summary.map(((wk, dates, total, head)) => (wk, total))
    chart.barchart(
      week-rows,
      mode: "basic",
      size: (15, 7),
      bar-style: i => (fill: c-action, stroke: none),
      label-key: 0,
      value-key: 1,
      x-tick-step: 100,
    )
  }),
)

= Full per-day commit table <full-per-day-table>

The complete daily series, 12 January -- 19 May 2026.

#data-table(
  columns: 4,
  header: ([Date], [Commits], [Date], [Commits]),
  align-rules: (left + horizon, right + horizon,
    left + horizon, right + horizon),
  ..{
    let pairs = ()
    let half = calc.ceil(d.daily-commits.len() / 2)
    let left-col = d.daily-commits.slice(0, half)
    let right-col = d.daily-commits.slice(half)
    for i in range(0, half) {
      let (d1, c1) = left-col.at(i)
      let right-item = if i < right-col.len() { right-col.at(i) }
        else { ("", "") }
      let (d2, c2) = right-item
      pairs.push(([#d1], [#if c1 == "" { "" } else { str(c1) }],
                  [#d2], [#if c2 == "" { "" } else { str(c2) }]))
    }
    pairs
  }
)

= Velocity observations

== Bimodal distribution

The histogram of active days is bimodal:

- *31 days at >= 50 commits* — these correspond to RC cutovers,
  multi-PR merge days, and bug-burndown sprints. Median commit on
  these days is a one-line code change or a test addition; the volume
  comes from many small commits in series.
- *92 days at < 50 commits* (median across active days is 27) —
  architectural and correctness work, where each commit tends to be a
  focused, reviewed change.

== Idle days

Only *5 calendar days* out of 128 had zero commits:

- 2026-02-23 (post-RC-3.2 quiet)
- 2026-03-11 (post-RC-3.4 break)
- 2026-03-16, 2026-03-17 (mid-March pause before RC-4 acceleration)
- 2026-04-06 (post-RC-5-cut break)

Idle days cluster immediately after a major release. The pattern is
#emph[sprint to release to 1--2 day pause to next-phase planning to
next-phase implementation]. The longest idle stretch was the 2-day
pause around 16--17 March 2026.

== Recovery from low-activity weeks

The lowest-activity calendar week was W06 (16--22 Feb, 57 commits).
It was immediately followed by W07's RC-3.3 kickoff (130 commits) —
a 2.3x rebound. The pattern repeats around W09 to W10 to W11 (53 to
107 to 209 commits over three weeks as RC-4 ramped up). Low-activity
weeks reliably precede a phase change rather than indicate sustained
slowdown.

== Tempo at the end of the period

The last seven days (13--19 May) have averaged *71 commits / day*;
the prior seven (6--12 May) averaged *75 commits / day*. Together
6--19 May holds at *73 commits / day* — the highest sustained 14-day
tempo since the Pre-RC1 run (16--22 Jan averaged *81 commits / day*
at the project's heaviest week). RC-5.2's Composer Phase 1--8 program
is being delivered at near-Pre-RC1 cadence.

= How these numbers were computed

```bash
# Active-day per-day commits across the three RC history snapshots
git log <rc1-history> <rc2-history> <rc5.2-history> \
    --format='%ad' --date=short \
  | sort | uniq -c

# Unique commit count across all three snapshots
git log <rc1-history> <rc2-history> <rc5.2-history> --format='%h' \
  | sort -u | wc -l
```

Output: 123 active-commit days, 4,521 unique commits.

Peak-day attribution is taken from the dominant `feat(...)` /
`add(...)` / `implement(...)` commit cluster on each day, validated
against the CHANGELOG family (the archived `CHANGELOG-RC1.md` and
`CHANGELOG-RC2.md` under `docs-archive/2026-05-19-docs-cleanout/`,
plus the active `/CHANGELOG.md`) for cross-reference.

= Sources

- `docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC1.md`, `docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC2.md`, `/CHANGELOG.md`
- Git history snapshots for RC-1, RC-2, RC-5.2, and `main`
- `docs/release/elspeth-progress-rc1-to-rc5.md` — companion cumulative-output document
