// REGISTER ONCE. This is the project's only global vitest matcher
// registration. Adding additional `expect.extend(...)` calls in test
// files (including audit/a11y test files) will shadow these matchers
// and break the "register once" invariant. If you need a new matcher,
// add it HERE — not in a test file. The audit suite
// (`components.a11y.test.tsx`) deliberately does NOT call
// `expect.extend({ toHaveNoViolations })` for this reason; see Task 7
// Step 4 prose for the same warning at the consumer site.

import { expect } from "vitest";
import { toHaveNoViolations } from "jest-axe";

// jest-axe's `toHaveNoViolations` export is ALREADY an `{ toHaveNoViolations:
// fn }` object — wrapping it again as `expect.extend({ toHaveNoViolations })`
// would produce a nested `{ toHaveNoViolations: { toHaveNoViolations: fn } }`
// and `expect.extend` would fail with `expectAssertion.call is not a function`.
// Pass the export through directly. (Plan-spec wording at line 3254 says
// `expect.extend({ toHaveNoViolations })`; the actual API requires the
// unwrapped form. Verified against `node_modules/jest-axe/index.js`.)
expect.extend(toHaveNoViolations);
