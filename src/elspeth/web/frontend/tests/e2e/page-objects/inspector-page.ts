// Page object for the right-side inspector panel (tabs: spec/graph/yaml/runs).
// Navigation/DOM only; assertions live in spec files.

import type { Page, Locator } from "@playwright/test";

export type InspectorTab = "spec" | "graph" | "yaml" | "runs";

export class InspectorPage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  tab(name: InspectorTab): Locator {
    // InspectorPanel.tsx applies role="tab" and id={`inspector-tab-${id}`}.
    return this.page.locator(`#inspector-tab-${name}`);
  }

  panel(name: InspectorTab): Locator {
    return this.page.locator(`#inspector-tabpanel-${name}`);
  }

  validationStatus(): Locator {
    // The validation dot is a <span> with one of four aria-labels:
    // "Not validated" / "Validation passed" / "Validation passed with warnings"
    // / "Validation failed". Use a regex over the four exact strings.
    return this.page.locator(".inspector-validation-dot");
  }

  async openTab(name: InspectorTab): Promise<void> {
    await this.tab(name).click();
  }
}
