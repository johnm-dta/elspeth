// Shared axe-core configuration for the composer a11y audit suite.
//
// Disables `color-contrast` because jsdom does not compute CSS-variable
// values, so contrast checks against design tokens generate false
// positives. Contrast is instead verified by the token-pair math suite
// (src/styles/colorContrast.test.ts, which covers BOTH themes) plus
// visual review. A themed (dark) axe pass is likewise pointless in this
// harness: jsdom never applies the stylesheets, so `data-theme="dark"`
// changes nothing axe can see — composed-surface contrast in both themes
// needs a real-browser pass (Playwright + axe injection against staging,
// the approach the 2026-07-02 live-review harness demonstrated).
//
// Restricts to WCAG 2.0/2.1/2.2 levels A and AA; AAA is too restrictive
// for a developer tool. The audit gate is "no AA violations".
//
// `wcag22aa` (axe-core >= 4.5) is the only WCAG 2.2 tag set axe ships:
// of the six new 2.2 SCs only 2.5.8 Target Size (Minimum) has an
// automated rule (`target-size`); 2.4.11/2.5.7/3.2.6/3.3.7/3.3.8 are
// not automatable and rest on manual review. There is no `wcag22a` tag.

import { configureAxe } from "jest-axe";

export const axe = configureAxe({
  rules: {
    "color-contrast": { enabled: false },
  },
  runOnly: {
    type: "tag",
    values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"],
  },
});
