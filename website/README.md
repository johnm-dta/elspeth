# ELSPETH project website

Standalone static site built on the ELSPETH design tokens. No build step.
Serve `website/` with any static server (or GitHub Pages); open `index.html`.

Pages: index (Home), authoring, assurance, use-cases, get-started.
Shared: `site.css`, `site.js` (Lucide icons, light/dark toggle, copy-to-clipboard).

**No third-party requests.** Lucide is vendored at `vendor/lucide.min.js`, and the
webfonts (Inter + JetBrains Mono, both SIL OFL-1.1) are self-hosted as variable
woff2 under `fonts/` — so the site loads nothing from a CDN and leaks no visitor
IPs. Their SIL OFL-1.1 licence (both copyright holders) is bundled at
`fonts/OFL.txt`.

**Design tokens.** `website/styles.css` + `website/tokens/*.css` mirror the
frontend's canonical styles. The structures differ, so re-syncing is a
value-by-value copy, **not** a whole-file overwrite:

- `tokens/{colors,typography,layout,fonts}.css` mirror the app's single
  `src/elspeth/web/frontend/src/styles/tokens.css` (with fonts self-hosted here
  rather than loaded from Google Fonts).
- `tokens/{base,primitives}.css` mirror the app's `shared.css` (reset + shared
  primitive classes).

**Contrast guard.** `website/tokens/` has no equivalent of the app's
`colorContrast.test.ts`, so run `node check-contrast.mjs` after any token or
colour change — it parses the token values plus the `site.css` light-theme
overrides and exits non-zero if the terminal/code text drops below WCAG AA (4.5:1)
in either theme. Wire it into CI alongside the site deploy.
