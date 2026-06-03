// Page object for the plugin catalog drawer.
// Navigation/DOM only; assertions live in spec files.

import type { Page, Locator } from "@playwright/test";

export class CatalogPage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  toggleButton(): Locator {
    return this.page.getByRole("button", { name: /catalog \(reference\)/i });
  }

  drawer(): Locator {
    // CatalogDrawer renders an off-canvas panel; we identify it by its
    // backdrop testid (catalog-backdrop already exists in the markup
    // per existing Vitest mocks).
    return this.page.locator(".catalog-drawer");
  }

  async open(): Promise<void> {
    await this.toggleButton().click();
    await this.drawer().waitFor({ state: "visible" });
  }
}
