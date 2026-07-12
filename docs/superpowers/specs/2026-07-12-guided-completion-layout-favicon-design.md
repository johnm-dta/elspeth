# Guided Completion Layout and Favicon Design

Date: 2026-07-12
Status: Approved for implementation

## Problem

On the guided workflow's completed “Pipeline ready” surface, the Run Results
panel sits close to the content above but does not participate in the page's
remaining vertical space. On wide, short viewports this leaves the results card
looking detached from the bottom edge. The application also has no favicon, so
browsers request `/favicon.ico` and display a generic tab icon.

## Approved Experience

### Run Results layout

The expanded Run Results panel begins 4px below the preceding completion
content and grows downward to consume all remaining height in the completed
chat panel.

- Scope the fill behavior to the guided completed surface only.
- Preserve the existing desktop and mobile horizontal gutters.
- Remove the expanded panel's bottom gap so its border reaches the completed
  panel's bottom edge.
- Keep Run Results content top-aligned inside the expanded panel.
- Keep overflow inside Run Results scrollable; the panel must not force the
  completed page beyond its height constraint.
- Preserve the collapse affordance. A collapsed Run Results panel remains
  content-height and does not fill the available space.
- If there is no active, recent, or historical run, `InlineRunResults` remains
  absent and the completion surface is unchanged.

This is the approved “fill downward” option. It keeps Pipeline Ready and its
execution evidence adjacent in the reading order and avoids a variable dead
zone between them.

### Favicon

Add an SVG favicon using the approved ELSPETH monogram:

- dark-teal rounded-square field;
- high-contrast cyan capital “E”;
- small light audit dot in the upper-right;
- simple geometry that remains identifiable at 16px;
- served as a same-origin Vite public asset and linked explicitly from
  `index.html` with `type="image/svg+xml"`.

The favicon uses the existing ELSPETH dark-teal/cyan visual language and does
not introduce a new brand palette.

## Implementation Boundaries

- Add a completed-surface modifier in the existing chat layout CSS rather than
  changing `InlineRunResults` behavior globally.
- Reuse the existing `.chat-panel--completed`, `.inline-run-results`, and
  `.inline-run-results--collapsed` hooks.
- Add the favicon under the frontend `public/` directory and add one `<link>`
  in the existing document head.
- No backend, API, state-management, run-history, or completion-flow changes.
- No changes to the Run Results rendering or data-loading logic.

## Responsive and Accessibility Behavior

- Desktop and wide/short viewports use the fill-down layout.
- Existing narrow-screen horizontal margins remain authoritative.
- Scrolling stays on the already keyboard-reachable Run Results region.
- The collapsed state remains compact and keyboard-operable.
- The decorative favicon does not alter the document's accessible name or
  application wordmark.

## Verification

1. Add a regression contract for the completed-surface layout hook and favicon
   declaration where the frontend test conventions support it.
2. Run the affected ChatPanel and InlineRunResults tests.
3. Run frontend lint, typecheck, and the production build.
4. Rebuild the deployed frontend and restart the backend service.
5. Use Playwright to verify the completed page at the supplied wide/short
   viewport and at a narrow viewport:
   - 4px visual separation below the completion content;
   - expanded Run Results reaches the bottom edge;
   - collapsed Run Results remains compact;
   - no unintended page-level overflow;
   - `/favicon.svg` loads successfully and the browser no longer requests a
     missing favicon.

## Out of Scope

- Redesigning Run Results content, toolbar, or output rows.
- Changing the completion summary, workflow stepper, or side rail.
- Adding raster favicon variants, platform-specific app icons, or a broader
  brand asset system.
