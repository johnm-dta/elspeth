// THROWAWAY instrumented probe (not part of the battery). Drives the real staged
// guided tutorial against staging, screenshotting after every transition and
// dumping the visible-affordance inventory + guided/tutorial network outcomes so
// the FIRST divergence is localized fast. Delete after debugging.
//
// Run: STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass \
//      STAGING_BASE_URL=https://elspeth.foundryside.dev \
//      PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
//      npx playwright test --config=playwright.staging.config.ts tutorial-probe

import { mkdirSync } from "node:fs";

import { test, expect, type Page } from "@playwright/test";

import { harnessCtx, resetToFirstRun, cleanSessions } from "./helpers/tutorial-harness";

const SHOT_DIR =
  process.env.TUTORIAL_PROBE_SHOT_DIR ??
  "/tmp/claude-1000/-home-john-elspeth/92b95b1e-c2e4-4139-98fe-f6d036587be2/scratchpad/tutorial-probe";
let shotN = 0;
async function shot(page: Page, label: string): Promise<void> {
  shotN += 1;
  const name = `${String(shotN).padStart(2, "0")}-${label}`;
  await page.screenshot({ path: `${SHOT_DIR}/${name}.png`, fullPage: true }).catch(() => {});
  console.log(`[shot] ${name}`);
}

// Visible buttons (+ enabled state), headings, and any alert/error text.
async function affordances(page: Page): Promise<string> {
  return page.evaluate(() => {
    const out: string[] = [];
    const vis = (el: Element) => {
      const r = (el as HTMLElement).getBoundingClientRect();
      const s = getComputedStyle(el as HTMLElement);
      return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
    };
    document.querySelectorAll("button").forEach((b) => {
      if (!vis(b)) return;
      out.push(
        `btn[${(b as HTMLButtonElement).disabled ? "OFF" : "on "}]: "${(b.textContent || "").trim().slice(0, 60)}"`,
      );
    });
    document.querySelectorAll("h1,h2,h3").forEach((h) => {
      if (vis(h)) out.push(`hdg: "${(h.textContent || "").trim().slice(0, 90)}"`);
    });
    document
      .querySelectorAll('[role="alert"], .error, [class*="error" i], [class*="Error"]')
      .forEach((e) => {
        if (vis(e)) {
          const t = (e.textContent || "").trim();
          if (t) out.push(`ALERT: "${t.slice(0, 140)}"`);
        }
      });
    return out.join("\n");
  });
}

test.beforeEach(async () => {
  const ctx = await harnessCtx();
  await cleanSessions(ctx);
  await resetToFirstRun(ctx);
  await ctx.dispose();
});

test("probe: walk the staged guided tutorial", async ({ page }) => {
  test.setTimeout(600_000);
  mkdirSync(SHOT_DIR, { recursive: true });

  page.on("console", (m) => {
    if (m.type() === "error") console.log(`[browser:error] ${m.text().slice(0, 220)}`);
  });
  page.on("requestfailed", (r) =>
    console.log(`[reqfail] ${r.method()} ${r.url()} :: ${r.failure()?.errorText}`),
  );
  page.on("response", async (r) => {
    const u = r.url();
    const m = r.request().method();
    if (m !== "GET" && /\/api\/(sessions\/[^/]+\/guided|tutorial)/.test(u)) {
      let extra = "";
      if (!r.ok()) extra = " :: " + (await r.text().catch(() => "")).slice(0, 300);
      console.log(`[resp] ${m} ${u.replace(/^https?:\/\/[^/]+/, "")} -> ${r.status()}${extra}`);
    }
  });

  await page.goto("/");
  await expect(page.getByRole("main", { name: /first-run tutorial/i })).toBeVisible({
    timeout: 20_000,
  });
  await shot(page, "landing");

  await page.getByRole("button", { name: "Let's go" }).click();
  await expect(page.getByLabel(/guided composer/i)).toBeVisible({ timeout: 30_000 });
  await shot(page, "guided-shell");
  console.log("[affordances @ guided-shell]\n" + (await affordances(page)));

  // The tutorial is the NORMAL guided flow with the intent PRELOCKED at every
  // phase: the learner types nothing and never fills the box — on each LLM-driven
  // phase they press Send on the prepopulated worked-example prompt. We mirror
  // tutorial-reliability's driver exactly: locate the step chat under its current
  // region label, do NOT fill the read-only prompt, and wait for it to populate
  // (synthetic URLs are fetched + appended async) before the pump drives by Send.
  const stepChat = page.getByRole("region", { name: "Describe what you want" });
  const stepChatInput = stepChat.getByLabel("Message input");
  const stepChatSend = stepChat.getByRole("button", { name: "Send message" });
  await expect(stepChatInput).toBeVisible({ timeout: 30_000 });
  await expect(stepChatInput).not.toHaveValue("", { timeout: 30_000 });
  await shot(page, "step-1-locked-prompt");

  // Turn-pump: resolve per-stage interpretation reviews, then advance via the
  // enabled stage primary, screenshotting + dumping affordances as state evolves.
  const runHeading = page.getByRole("heading", { name: /Running your pipeline/i });
  const acceptButtons = page.getByRole("button", { name: /^Accept /i });
  const promptRegions = page.getByRole("region", { name: "Prompt template review" });
  const primaries = [
    page.getByRole("button", { name: "Confirm wiring", exact: true }),
    page.getByRole("button", { name: "Continue", exact: true }),
    // Source inspection review (inspect_and_confirm): rendered after the
    // chat-resolved inline source is materialized into a session blob and
    // inspected — confirming the observed columns is the designed answer.
    page.getByRole("button", { name: "Looks right", exact: true }),
    // Component review turns: once the chat-resolved source/output lands as a
    // reviewed component, the stage ends on its review turn — finishing it is
    // the designed advance (mirrors composer-guided-live).
    page.getByRole("button", { name: "Finish sources", exact: true }),
    page.getByRole("button", { name: "Finish outputs", exact: true }),
    // Output required-fields turn: the LLM-built sink is observed-mode, so the
    // designed answer is the escape, not ticking the source's column.
    page.getByRole("button", { name: "Let source decide (pass all fields through)", exact: true }),
  ];

  // The tutorial prompt is prelocked at every phase; the learner drives each
  // LLM-built phase (Source/Output/Transforms) by pressing Send ONCE per phase
  // (re-sending mid-build re-triggers the driver). Wire is confirm-only.
  // Mirrors tutorial-reliability's driveGuidedWalk.
  const drivenPhases = new Set(["Source", "Output", "Transforms"]);
  const currentPhase = async (): Promise<string | null> => {
    const label = page.locator(".guided-workflow-step--current .guided-workflow-label").first();
    const text = await label.textContent().catch(() => null);
    return text ? text.trim() : null;
  };
  let lastDrivenPhase: string | null = null;
  const deadline = Date.now() + 420_000;
  let pass = 0;
  let lastSig = "";
  while (Date.now() < deadline) {
    pass += 1;
    if (await runHeading.isVisible().catch(() => false)) {
      await shot(page, "run-heading-reached");
      console.log("[milestone] reached the run turn — compose phase completed");
      break;
    }
    const rc = await promptRegions.count().catch(() => 0);
    for (let i = 0; i < rc; i++)
      await promptRegions
        .nth(i)
        .evaluate((el) => {
          el.scrollTop = el.scrollHeight;
          el.dispatchEvent(new Event("scroll"));
        })
        .catch(() => {});

    let advanced = false;
    const ab = await acceptButtons.count().catch(() => 0);
    for (let i = 0; i < ab; i++) {
      const b = acceptButtons.nth(i);
      if (await b.isEnabled().catch(() => false)) {
        const label = (await b.textContent().catch(() => "")) ?? "";
        await b.click().catch(() => {});
        console.log(`[advance] clicked Accept: "${label.trim().slice(0, 40)}"`);
        advanced = true;
        break;
      }
    }
    if (!advanced) {
      for (const p of primaries) {
        if ((await p.count().catch(() => 0)) > 0 && (await p.isEnabled().catch(() => false))) {
          const label = (await p.textContent().catch(() => "")) ?? "";
          await p.click().catch(() => {});
          console.log(`[advance] clicked primary: "${label.trim().slice(0, 40)}"`);
          advanced = true;
          break;
        }
      }
    }
    if (!advanced) {
      // No Accept/primary fired — drive the CURRENT LLM phase by Send (once per
      // phase), mirroring tutorial-reliability. A confirm primary appears once
      // the phase result renders.
      const phase = await currentPhase();
      const canSend = await stepChatSend.isEnabled().catch(() => false);
      if (canSend && phase !== null && drivenPhases.has(phase) && phase !== lastDrivenPhase) {
        await stepChatSend.click().catch(() => {});
        lastDrivenPhase = phase;
        console.log(`[advance] sent locked prompt for phase: "${phase}"`);
        advanced = true;
        await page.waitForTimeout(2_000);
      }
    }
    const sig = await affordances(page);
    if (advanced || sig !== lastSig) {
      await shot(page, `pass-${String(pass).padStart(2, "0")}`);
      console.log(`[affordances @ pass ${pass}]\n${sig}`);
      lastSig = sig;
    }
    if (!advanced) await page.waitForTimeout(2500);
  }

  // If we reached the run, confirm it actually completes (run POST + Continue).
  if (await runHeading.isVisible().catch(() => false)) {
    const done = page.getByRole("button", { name: "Continue", exact: true });
    await expect(done).toBeVisible({ timeout: 360_000 }).catch(() => {});
    await shot(page, "run-finished");
    console.log("[affordances @ run-finished]\n" + (await affordances(page)));
  } else {
    await shot(page, "final-no-run");
    console.log("[affordances @ final-no-run]\n" + (await affordances(page)));
    throw new Error("compose phase did not reach the run turn within deadline");
  }
});
