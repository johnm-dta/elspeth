// Shared staged-guided tutorial driver used by the standalone staging harness.

export const ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES = [
  /^View prompt$/,
  /^Approve the LLM prompt template$/,
  /^Acknowledge/i,
];

export const GUIDED_STAGE_PRIMARY_ACTION_NAMES = [
  "Confirm wiring",
  "Continue",
  "Let source decide (pass all fields through)",
];

export const STAGED_GUIDED_PHASES = ["Source", "Output", "Transforms"];

export function isComposeRequest(url, method) {
  return (
    method.toUpperCase() === "POST" &&
    /\/api\/sessions\/[0-9a-f-]{36}\/guided\/respond\b/i.test(url)
  );
}

export function isRunRequest(url, method) {
  return method.toUpperCase() === "POST" && url.includes("/api/tutorial/run");
}

async function isVisible(locator) {
  return locator.isVisible().catch(() => false);
}

async function isEnabled(locator) {
  return locator.isEnabled().catch(() => false);
}

async function waitForNonEmptyInput(locator, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await locator.inputValue().catch(() => "");
    if (value.trim() !== "") {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("guided tutorial locked prompt did not populate");
}

export async function resolveVisibleReviews(page) {
  const primaryButtons = ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES.map((name) =>
    page.getByRole("button", { name }),
  );
  const legacyViewToggles = page.getByRole("button", { name: /^View$/ });
  const promptRegions = page.getByRole("region", {
    name: "Prompt template review",
  });
  let actions = 0;

  for (let guard = 0; guard < 12; guard += 1) {
    const toggleCount = await legacyViewToggles.count().catch(() => 0);
    for (let i = 0; i < toggleCount; i += 1) {
      await legacyViewToggles.nth(i).click().catch(() => {});
    }

    const regionCount = await promptRegions.count().catch(() => 0);
    for (let i = 0; i < regionCount; i += 1) {
      await promptRegions
        .nth(i)
        .evaluate((el) => {
          el.scrollTop = el.scrollHeight;
          el.dispatchEvent(new Event("scroll"));
        })
        .catch(() => {});
    }

    let clicked = false;
    for (const buttons of primaryButtons) {
      const total = await buttons.count().catch(() => 0);
      for (let i = 0; i < total; i += 1) {
        const btn = buttons.nth(i);
        if (await isEnabled(btn)) {
          await btn.click().catch(() => {});
          actions += 1;
          clicked = true;
          break;
        }
      }
      if (clicked) {
        break;
      }
    }
    if (!clicked) {
      await page.waitForTimeout(300);
    }
  }

  return actions;
}

export async function driveStagedGuidedTutorial(page, options = {}) {
  const timeoutMs = options.timeoutMs ?? 600_000;
  const decisionScreenshotPath = options.decisionScreenshotPath ?? null;
  const guidedPanel = page.getByLabel(/guided composer/i);
  const runHeading = page.getByRole("heading", {
    name: /Running your pipeline/i,
  });
  const stepChat = page.getByRole("region", { name: "Describe what you want" });
  const stepChatInput = stepChat.getByLabel("Message input");
  const stepChatSend = stepChat.getByRole("button", { name: "Send message" });

  await stepChatInput.waitFor({ state: "visible", timeout: 30_000 });
  await waitForNonEmptyInput(stepChatInput, 30_000);

  const primaries = GUIDED_STAGE_PRIMARY_ACTION_NAMES.map((name) =>
    page.getByRole("button", { name, exact: true }),
  );
  const drivenPhases = new Set(STAGED_GUIDED_PHASES);

  async function currentPhase() {
    const label = page
      .locator(".guided-workflow-step--current .guided-workflow-label")
      .first();
    const text = await label.textContent().catch(() => null);
    return text ? text.trim() : null;
  }

  let lastDrivenPhase = null;
  let assertedSummary = false;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await isVisible(runHeading)) {
      if (!assertedSummary) {
        throw new Error("expected to observe a read-only decision summary");
      }
      return;
    }
    if (!(await isVisible(guidedPanel))) {
      return;
    }

    if (
      !assertedSummary &&
      (await isVisible(page.locator(".guided-schema-summary").first()))
    ) {
      assertedSummary = true;
      if (decisionScreenshotPath !== null) {
        await page
          .screenshot({ path: decisionScreenshotPath, fullPage: true })
          .catch(() => {});
      }
      if ((await page.locator(".guided-schema-input").count().catch(() => 0)) > 0) {
        throw new Error(
          "guided decision rendered an editable form, expected a read-only summary",
        );
      }
    }

    await resolveVisibleReviews(page);

    let advanced = false;
    for (const primary of primaries) {
      const total = await primary.count().catch(() => 0);
      for (let i = 0; i < total; i += 1) {
        const btn = primary.nth(i);
        if (await isEnabled(btn)) {
          await btn.click().catch(() => {});
          advanced = true;
          break;
        }
      }
      if (advanced) {
        break;
      }
    }
    if (advanced) {
      await page.waitForTimeout(750);
      continue;
    }

    const phase = await currentPhase();
    const canSend = await isEnabled(stepChatSend);
    if (
      canSend &&
      phase !== null &&
      drivenPhases.has(phase) &&
      phase !== lastDrivenPhase
    ) {
      await stepChatSend.click().catch(() => {});
      lastDrivenPhase = phase;
      await page.waitForTimeout(2_000);
      continue;
    }

    await page.waitForTimeout(1_000);
  }

  throw new Error("guided walk never reached the run turn before the deadline");
}
