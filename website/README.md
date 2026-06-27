# ELSPETH marketing website

Standalone static site built on the ELSPETH design tokens. No build step.
Serve `website/` with any static server (or GitHub Pages); open `index.html`.

Pages: index (Home), authoring, assurance, use-cases, get-started.
Shared: `site.css`, `site.js` (Lucide icons + light/dark toggle). Lucide loads
from the unpkg CDN.

**Token source:** `website/tokens/` + `website/styles.css` are a static MIRROR of
the frontend's canonical tokens (`src/elspeth/web/frontend/src/styles/tokens.css`).
They are identical today; if the frontend tokens change, re-copy them here.
