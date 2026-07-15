import { basename } from "node:path";

import type {
  FullConfig,
  FullResult,
  Reporter,
  Suite,
  TestCase,
  TestError,
  TestResult,
} from "@playwright/test/reporter";

const CREDENTIAL_ENV_NAMES = [
  "OIDC_TEST_USERNAME",
  "OIDC_TEST_PASSWORD",
  "OIDC_EXPECTED_ISSUER",
  "OIDC_EXPECTED_AUDIENCE",
] as const;
const JWT_PATTERN = /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{2,}\.[A-Za-z0-9_-]{2,}\b/g;
const URL_PATTERN = /https?:\/\/[^\s"'<>]+/gi;
const KEYED_SECRET_PATTERN = /\b(code|state|code_verifier|access_token|refresh_token|id_token|token|password|username)=([^\s&;]+)/gi;
const STRUCTURED_SECRET_PATTERN = /((?:["'](?:code|state|code_verifier|access_token|refresh_token|id_token|token)["']|\b(?:code|state|code_verifier|access_token|refresh_token|id_token|token)\b)\s*:\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^,}\r\n]+)/gi;
const HEADER_PATTERN = /^(authorization|cookie|set-cookie|proxy-authorization|x-api-key)\s*:\s*.*$/gim;
const STRUCTURED_HEADER_PATTERN = /(["'](?:authorization|cookie|set-cookie|proxy-authorization|x-api-key)["']\s*:\s*)(["'][^"'\r\n]*["']|[^,}\r\n]+)/gi;
const MAX_REPORTED_CHARS = 16 * 1024;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function sanitizeReporterText(raw: unknown, environ: Readonly<Record<string, string | undefined>>): string {
  let value = typeof raw === "string" ? raw : raw instanceof Buffer ? raw.toString("utf8") : String(raw);
  for (const name of CREDENTIAL_ENV_NAMES) {
    const secret = environ[name];
    if (secret) {
      value = value.replace(new RegExp(escapeRegExp(secret), "g"), "[REDACTED]");
    }
  }
  value = value
    .replace(HEADER_PATTERN, "$1: [REDACTED]")
    .replace(STRUCTURED_HEADER_PATTERN, '$1"[REDACTED]"')
    .replace(/\bBearer\s+[^\s]+/gi, "[REDACTED_BEARER]")
    .replace(JWT_PATTERN, "[REDACTED_JWT]")
    .replace(KEYED_SECRET_PATTERN, "$1=[REDACTED]")
    .replace(STRUCTURED_SECRET_PATTERN, '$1"[REDACTED]"')
    .replace(URL_PATTERN, "[REDACTED_URL]")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "?");
  return value.slice(0, MAX_REPORTED_CHARS);
}

interface ReporterOptions {
  stdout?: (value: string) => void;
  stderr?: (value: string) => void;
  environ?: Readonly<Record<string, string | undefined>>;
}

export default class OidcRedactingReporter implements Reporter {
  private readonly writeStdout: (value: string) => void;
  private readonly writeStderr: (value: string) => void;
  private readonly environ: Readonly<Record<string, string | undefined>>;
  private reporterFailed = false;

  constructor(options: ReporterOptions = {}) {
    this.writeStdout = options.stdout ?? ((value) => process.stdout.write(`${value}\n`));
    this.writeStderr = options.stderr ?? ((value) => process.stderr.write(`${value}\n`));
    this.environ = options.environ ?? process.env;
  }

  private safeWrite(channel: "stdout" | "stderr", raw: unknown): void {
    try {
      const sanitized = sanitizeReporterText(raw, this.environ);
      (channel === "stdout" ? this.writeStdout : this.writeStderr)(sanitized);
    } catch {
      this.reporterFailed = true;
      try {
        this.writeStderr("oidc_reporter_failure");
      } catch {
        // Playwright swallows reporter exceptions; retain the failed status.
      }
    }
  }

  onBegin(_config: FullConfig, suite: Suite): void {
    for (const test of suite.allTests()) {
      const titlePath = test.titlePath();
      const project = titlePath[0] || "chromium";
      this.safeWrite("stdout", `[${project}] ${basename(test.location.file)}`);
    }
  }

  onStdOut(_chunk: string | Buffer, _test: void | TestCase, _result: void | TestResult): void {
    this.safeWrite("stdout", "oidc_test_stdout_suppressed");
  }

  onStdErr(_chunk: string | Buffer, _test: void | TestCase, _result: void | TestResult): void {
    this.safeWrite("stderr", "oidc_test_stderr_suppressed");
  }

  onError(error: TestError): void {
    this.safeWrite("stderr", `oidc_global_error ${error.message ?? "unknown"} ${error.stack ?? ""}`);
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    this.safeWrite("stdout", `oidc_test_end ${test.title} status=${result.status}`);
    if (result.error) {
      this.safeWrite("stderr", `oidc_test_error ${result.error.message ?? "unknown"} ${result.error.stack ?? ""}`);
    }
    for (const attachment of result.attachments) {
      this.safeWrite(
        "stdout",
        `oidc_attachment name=${attachment.name} content_type=${attachment.contentType} body=[REDACTED] path=[REDACTED]`,
      );
    }
  }

  async onEnd(result: FullResult): Promise<{ status: FullResult["status"] }> {
    const status = this.reporterFailed ? "failed" : result.status;
    this.safeWrite("stdout", `oidc_run_end status=${status}`);
    return { status: this.reporterFailed ? "failed" : status };
  }

  printsToStdio(): boolean {
    return true;
  }
}
