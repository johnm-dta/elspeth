/**
 * P3A-002: Mechanical guard — every Alt-key shortcut in TAB_SHORTCUT_MAP must
 * target a live inspector tab.
 *
 * This test is the machine-enforcement of the constraint documented in CLAUDE.md:
 * "if it's not mechanically enforced (by types, tests, CI, or named constants),
 * assume the next session won't know about it either."
 *
 * Placement rationale: kept separate from App.test.tsx because App.test.tsx
 * carries heavy DOM-rendering mocks that are irrelevant here. This test is a
 * pure constant-coherence check — no rendering, no mocks, fast compile-time
 * guard. If TAB_SHORTCUT_MAP gains a dead key (targets a tab that no longer
 * exists in TABS), this test fails immediately with the name of the bad key.
 */

import { describe, it, expect } from "vitest";
import { TAB_SHORTCUT_MAP } from "@/App";
import { TABS } from "@/components/inspector/InspectorPanel";

describe("keyboard shortcut → tab wiring", () => {
  it("every Alt-key shortcut targets a live inspector tab", () => {
    const liveTabIds = new Set(TABS.map((t) => t.id));
    for (const [key, targetTabId] of Object.entries(TAB_SHORTCUT_MAP)) {
      expect(
        liveTabIds,
        `Alt+${key} targets "${targetTabId}" which is not a live tab — update TAB_SHORTCUT_MAP or restore the tab`,
      ).toContain(targetTabId);
    }
  });
});
