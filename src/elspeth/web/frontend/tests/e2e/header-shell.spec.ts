import { expect, test } from "@playwright/test";

import { ComposerPage } from "./page-objects/composer-page";

test.describe("header-owned shell", () => {
  test("renders the header-owned session and account controls without the legacy sidebar", async ({
    page,
  }) => {
    const composer = new ComposerPage(page);
    await composer.goto();

    await expect(page.getByRole("banner")).toBeVisible();
    await expect(page.getByRole("button", { name: /untitled/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /account menu/i })).toBeVisible();
    await expect(page.getByLabel("Sessions sidebar")).toHaveCount(0);

    await page.getByRole("button", { name: /account menu/i }).click();
    await expect(
      page.getByRole("button", { name: /switch to (light|dark) theme/i }),
    ).toBeVisible();
  });
});
