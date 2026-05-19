// Shared axe-core configuration for the composer a11y audit suite.
//
// Disables `color-contrast` because jsdom does not compute CSS-variable
// values, so contrast checks against design tokens generate false
// positives. Contrast is verified manually against the design-token
// palette during visual review.
//
// Restricts to WCAG 2.0/2.1 levels A and AA; AAA is too restrictive
// for a developer tool. The audit gate is "no AA violations".

import { configureAxe } from "jest-axe";

export const axe = configureAxe({
  rules: {
    "color-contrast": { enabled: false },
  },
  runOnly: {
    type: "tag",
    values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"],
  },
});
