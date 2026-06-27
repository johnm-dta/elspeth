# Tutorial synthetic scrape pages

Three self-contained synthetic "government project brief" pages used by the
first-run guided tutorial's `web_scrape` demo: the tutorial fetches them at
`{base}/tutorial-site/project-N.html` and has an LLM write a short summary of
each.

These **mirror** the application copies in
`src/elspeth/web/frontend/public/tutorial-site/` (which the deployed app serves
at its own origin). They are published here on the public GitHub Pages site so
the tutorial can also run end-to-end from a loopback dev origin — set
`ELSPETH_WEB__TUTORIAL_SAMPLE_BASE_URL=https://johnm-dta.github.io/elspeth`
(see the top-level README, "Web Composer" notes). Re-copy here if the app copies
change.
