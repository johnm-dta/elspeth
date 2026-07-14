import { describe, expect, it } from "vitest";
import type { FullConfig, FullResult, Suite, TestCase, TestError, TestResult } from "@playwright/test/reporter";

import OidcRedactingReporter, { sanitizeReporterText } from "./oidc-redacting-reporter";

const SECRET = "oidc-credential-sentinel";
const JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJzZW50aW5lbCJ9.signature";

describe("OIDC redacting reporter", () => {
  it("redacts credentials, bearer/JWT material, keyed callback fields, URLs, cookies, and headers", () => {
    const raw = [
      SECRET,
      `Bearer ${JWT}`,
      JWT,
      "https://example.invalid/callback?code=secret-code&state=secret-state#access_token=secret-token",
      "code_verifier=secret-verifier",
      "Cookie: session=secret-cookie",
      "Authorization: secret-header",
      '{"cookie":"session=structured-cookie","x-api-key":"structured-key"}',
    ].join("\n");

    const sanitized = sanitizeReporterText(raw, { OIDC_TEST_PASSWORD: SECRET });

    for (const forbidden of [
      SECRET,
      JWT,
      "secret-code",
      "secret-state",
      "secret-token",
      "secret-verifier",
      "secret-cookie",
      "secret-header",
      "structured-cookie",
      "structured-key",
    ]) {
      expect(sanitized).not.toContain(forbidden);
    }
    expect(sanitizeReporterText("credential=x", { OIDC_TEST_PASSWORD: "x" })).not.toContain("x");
  });

  it("redacts callback fields from JSON and object-like text", () => {
    const raw = [
      '{"code":"json-code-secret","state":"json-state-secret","access_token":"json-access-secret"}',
      "{ id_token: 'object-id-secret', refresh_token: object-refresh-secret }",
    ].join("\n");

    const sanitized = sanitizeReporterText(raw, {});

    for (const forbidden of [
      "json-code-secret",
      "json-state-secret",
      "json-access-secret",
      "object-id-secret",
      "object-refresh-secret",
    ]) {
      expect(sanitized).not.toContain(forbidden);
    }
  });

  it("covers every reporter channel without printing attachment paths or bodies", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const reporter = new OidcRedactingReporter({
      stdout: (value) => stdout.push(value),
      stderr: (value) => stderr.push(value),
      environ: { OIDC_TEST_USERNAME: SECRET, OIDC_TEST_PASSWORD: SECRET },
    });
    const testCase = {
      title: `test ${SECRET}`,
      titlePath: () => ["chromium", "aws-ecs-oidc.staging.spec.ts", `test ${SECRET}`],
      location: { file: "/private/aws-ecs-oidc.staging.spec.ts", line: 1, column: 1 },
    } as unknown as TestCase;
    const result = {
      status: "failed",
      error: { message: `Bearer ${JWT}`, stack: `stack ${SECRET}` },
      attachments: [
        { name: `artifact-${SECRET}`, contentType: "text/plain", path: `/private/${SECRET}`, body: Buffer.from(SECRET) },
      ],
    } as unknown as TestResult;

    reporter.onBegin({} as FullConfig, { allTests: () => [testCase] } as unknown as Suite);
    reporter.onStdOut?.(`stdout ${SECRET}`, testCase, result);
    reporter.onStdErr?.(Buffer.from(`stderr ${JWT}`), testCase, result);
    reporter.onStdOut?.("split-oidc-creden", testCase, result);
    reporter.onStdOut?.("tial-sentinel", testCase, result);
    reporter.onError?.({ message: `global ${SECRET}`, stack: `stack ${JWT}` } as TestError);
    reporter.onTestEnd?.(testCase, result);
    const end = await reporter.onEnd?.({ status: "passed" } as FullResult);

    expect(reporter.printsToStdio?.()).toBe(true);
    expect(stdout[0]).toMatch(/^\[chromium\] aws-ecs-oidc\.staging\.spec\.ts$/);
    expect(end).toEqual({ status: "passed" });
    const rendered = [...stdout, ...stderr].join("\n");
    expect(rendered).not.toContain(SECRET);
    expect(rendered).not.toContain(JWT);
    expect(rendered).not.toContain("/private/");
    expect(rendered).not.toContain("Bearer");
    expect(rendered).not.toContain("split-oidc-credential-sentinel");
    expect(rendered).toContain("oidc_test_stdout_suppressed");
  });

  it("converts its own sink failure into a failed final status", async () => {
    const reporter = new OidcRedactingReporter({
      stdout: () => {
        throw new Error(SECRET);
      },
      stderr: () => undefined,
      environ: {},
    });
    reporter.onBegin({} as FullConfig, { allTests: () => [] } as unknown as Suite);

    await expect(reporter.onEnd?.({ status: "passed" } as FullResult)).resolves.toEqual({ status: "failed" });
  });
});
