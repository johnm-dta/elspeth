// Page object for the main composer surface. Per E2E discipline
// (e2e-testing-strategies.md "Page Object Anti-Patterns"), this PO does
// navigation and DOM interaction only — no business logic, no assertions.

import type { Page, Locator } from "@playwright/test";

export class ComposerPage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  async goto(sessionId?: string): Promise<void> {
    const path = sessionId === undefined ? "/" : `/#/${sessionId}`;
    await this.page.goto(path);
  }

  async waitForChatReady(): Promise<void> {
    await this.page.getByLabel("Chat panel").waitFor({ state: "visible" });
  }

  async createSession(_title: string): Promise<void> {
    await this.page.getByRole("button", { name: /session switcher/i }).click();
    await this.page.getByRole("menuitem", { name: "+ New session" }).click();
    await this.waitForChatReady();
  }

  async sendMessage(content: string): Promise<void> {
    await this.chatInput().fill(content);
    await this.page.getByRole("button", { name: "Send message" }).click();
  }

  chatInput(): Locator {
    // ChatInput renders a textarea inside an aria-labelled region. Use the
    // textbox role with the existing label "Message" (or fall back to any
    // single textarea inside the chat panel).
    return this.page.getByRole("textbox").filter({ hasText: "" }).first();
  }

  validateButton(): Locator {
    return this.page.getByRole("button", { name: /Validate pipeline/i });
  }

  executeButton(): Locator {
    return this.page.getByRole("button", { name: /Execute pipeline/i });
  }
}
