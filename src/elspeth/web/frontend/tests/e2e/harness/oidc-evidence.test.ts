import { chmodSync, lstatSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  OidcEvidenceError,
  buildOidcEvidence,
  validateAccessToken,
  validateAuthConfig,
  writeOidcBearerHandoff,
  writeOidcEvidence,
} from "./oidc-evidence";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) {
    try {
      chmodSync(directory, 0o700);
    } catch {
      // The test may already have removed or replaced the directory.
    }
    rmSync(directory, { force: true, recursive: true });
  }
});

function base64Url(value: object): string {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

function token(claims: Record<string, unknown>): string {
  return `${base64Url({ alg: "none", typ: "JWT" })}.${base64Url(claims)}.signature`;
}

const expected = {
  issuer: "https://cognito-idp.ap-southeast-2.amazonaws.com/pool",
  audience: "client-id",
  authorizationOrigin: "https://acceptance.auth.ap-southeast-2.amazoncognito.com",
} as const;

describe("OIDC config and access-token validation", () => {
  it.each(["aud", "client_id"] as const)("accepts the closed %s audience lane", (audienceClaim) => {
    const claims = validateAccessToken(
      token({
        iss: expected.issuer,
        sub: "subject-123",
        exp: 2_000_000_000,
        token_use: "access",
        [audienceClaim]: expected.audience,
      }),
      { ...expected, audienceClaim },
      1_900_000_000,
    );

    expect(claims.subjectSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.stringify(claims)).not.toContain("subject-123");
  });

  it("requires exact OIDC configuration and same-origin HTTPS endpoints", () => {
    expect(
      validateAuthConfig(
        {
          provider: "oidc",
          oidc_issuer: expected.issuer,
          oidc_client_id: expected.audience,
          authorization_endpoint: `${expected.authorizationOrigin}/oauth2/authorize`,
          token_endpoint: `${expected.authorizationOrigin}/oauth2/token`,
        },
        expected,
      ),
    ).toEqual({
      issuer: expected.issuer,
      audience: expected.audience,
      authorizationOrigin: expected.authorizationOrigin,
      tokenEndpoint: `${expected.authorizationOrigin}/oauth2/token`,
    });
  });

  it.each([
    ["wrong issuer", { iss: "https://wrong.invalid", sub: "subject", exp: 2_000_000_000, aud: expected.audience }],
    ["wrong audience", { iss: expected.issuer, sub: "subject", exp: 2_000_000_000, aud: "wrong" }],
    ["missing subject", { iss: expected.issuer, exp: 2_000_000_000, aud: expected.audience }],
    ["blank subject", { iss: expected.issuer, sub: "", exp: 2_000_000_000, aud: expected.audience }],
    ["expired", { iss: expected.issuer, sub: "subject", exp: 1_800_000_000, aud: expected.audience }],
  ])("rejects %s with a static error", (_label, claims) => {
    const sentinel = "credential-sentinel";
    expect(() =>
      validateAccessToken(token({ ...claims, secret: sentinel }), { ...expected, audienceClaim: "aud" }, 1_900_000_000),
    ).toThrowError(OidcEvidenceError);
    try {
      validateAccessToken(token({ ...claims, secret: sentinel }), { ...expected, audienceClaim: "aud" }, 1_900_000_000);
    } catch (error) {
      expect(String(error)).not.toContain(sentinel);
    }
  });

  it("requires a valid audience mode and access token_use in client_id mode", () => {
    const valid = token({
      iss: expected.issuer,
      sub: "subject",
      exp: 2_000_000_000,
      client_id: expected.audience,
      token_use: "id",
    });
    expect(() => validateAccessToken(valid, { ...expected, audienceClaim: "client_id" }, 1_900_000_000)).toThrow(
      "oidc_token_claims",
    );
    expect(() =>
      validateAccessToken(valid, { ...expected, audienceClaim: "invalid" as "aud" }, 1_900_000_000),
    ).toThrow("oidc_audience_claim");
  });

  it.each(["not-a-jwt", `${"a".repeat(20_000)}.e30.signature`, "a.!!!!.c"])(
    "rejects malformed or oversized JWT material",
    (rawToken) => {
      expect(() => validateAccessToken(rawToken, { ...expected, audienceClaim: "aud" }, 1_900_000_000)).toThrowError(
        OidcEvidenceError,
      );
    },
  );
});

describe("OIDC evidence", () => {
  it("builds the exact closed schema for one of four phases", () => {
    const evidence = buildOidcEvidence({
      phase: "candidate-initial",
      timestamp: "2026-07-14T01:02:03Z",
      ...expected,
      audienceClaim: "client_id",
      subjectSha256: "a".repeat(64),
      authMeStatus: 200,
      sessionCreateStatus: 201,
      sessionReadStatus: 200,
      sessionDeleteStatus: 204,
    });
    expect(Object.keys(evidence).sort()).toEqual(
      [
        "audience",
        "audience_claim",
        "auth_me_status",
        "authorization_origin",
        "issuer",
        "phase",
        "session_create_status",
        "session_delete_status",
        "session_read_status",
        "session_round_trip",
        "subject_sha256",
        "timestamp",
      ].sort(),
    );
    expect(evidence.session_round_trip).toBe(true);
    expect(() => buildOidcEvidence({ ...evidence, phase: "unreviewed" } as never)).toThrow("oidc_evidence_schema");
  });

  it("writes one bounded owner-only file through an owner-only directory", () => {
    const directory = join(tmpdir(), `elspeth-oidc-${process.pid}-${directories.length}`);
    directories.push(directory);
    mkdirSync(directory, { mode: 0o700 });
    const destination = join(directory, "candidate-initial.json");
    const evidence = buildOidcEvidence({
      phase: "candidate-initial",
      timestamp: "2026-07-14T01:02:03Z",
      ...expected,
      audienceClaim: "client_id",
      subjectSha256: "a".repeat(64),
      authMeStatus: 200,
      sessionCreateStatus: 201,
      sessionReadStatus: 200,
      sessionDeleteStatus: 204,
    });

    writeOidcEvidence(destination, evidence);

    expect(lstatSync(destination).mode & 0o777).toBe(0o600);
    expect(JSON.parse(readFileSync(destination, "utf8"))).toEqual(evidence);
    expect(() => writeOidcEvidence(join(directory, "invalid.json"), { ...evidence, audience: "" })).toThrow(
      "oidc_evidence_schema",
    );
  });

  it("writes a bounded bearer handoff only to a new owner-only file", () => {
    const directory = join(tmpdir(), `elspeth-oidc-handoff-${process.pid}-${directories.length}`);
    directories.push(directory);
    mkdirSync(directory, { mode: 0o700 });
    const destination = join(directory, "access-token");
    const accessToken = token({
      iss: expected.issuer,
      sub: "subject",
      exp: 2_000_000_000,
      client_id: expected.audience,
      token_use: "access",
    });

    writeOidcBearerHandoff(destination, accessToken);

    expect(lstatSync(destination).mode & 0o777).toBe(0o600);
    expect(readFileSync(destination, "utf8")).toBe(accessToken);
    expect(() => writeOidcBearerHandoff(destination, accessToken)).toThrow("oidc_bearer_handoff");
  });

  it("rejects symlink, permissive parent, and pre-existing destination attacks", () => {
    const directory = join(tmpdir(), `elspeth-oidc-attacks-${process.pid}-${directories.length}`);
    directories.push(directory);
    mkdirSync(directory, { mode: 0o700 });
    const evidence = buildOidcEvidence({
      phase: "candidate-initial",
      timestamp: "2026-07-14T01:02:03Z",
      ...expected,
      audienceClaim: "client_id",
      subjectSha256: "a".repeat(64),
      authMeStatus: 200,
      sessionCreateStatus: 201,
      sessionReadStatus: 200,
      sessionDeleteStatus: 204,
    });
    const target = join(directory, "target.json");
    const link = join(directory, "link.json");
    writeFileSync(target, "{}", { mode: 0o600 });
    symlinkSync(target, link);
    expect(() => writeOidcEvidence(link, evidence)).toThrow("oidc_evidence_destination");
    expect(() => writeOidcEvidence(target, evidence)).toThrow("oidc_evidence_destination");
    chmodSync(directory, 0o755);
    expect(() => writeOidcEvidence(join(directory, "new.json"), evidence)).toThrow("oidc_evidence_parent");
  });
});
