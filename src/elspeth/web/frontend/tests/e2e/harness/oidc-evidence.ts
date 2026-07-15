import {
  closeSync,
  constants,
  fsyncSync,
  linkSync,
  lstatSync,
  openSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { basename, dirname, join } from "node:path";

const MAX_JWT_BYTES = 16 * 1024;
const MAX_JWT_PAYLOAD_BYTES = 8 * 1024;
const MAX_EVIDENCE_BYTES = 64 * 1024;
const SHA256 = /^[0-9a-f]{64}$/;
const BASE64URL = /^[A-Za-z0-9_-]+$/;

export const OIDC_EVIDENCE_PHASES = [
  "previous-before-candidate",
  "candidate-initial",
  "previous-after-rollback",
  "candidate-after-redeploy",
] as const;

export type OidcEvidencePhase = (typeof OIDC_EVIDENCE_PHASES)[number];
export type OidcAudienceClaim = "aud" | "client_id";

export class OidcEvidenceError extends Error {
  constructor(readonly check: string) {
    super(check);
    this.name = "OidcEvidenceError";
  }
}

interface ExpectedOidc {
  issuer: string;
  audience: string;
  authorizationOrigin: string;
}

interface TokenExpectations extends ExpectedOidc {
  audienceClaim: OidcAudienceClaim;
}

interface AuthConfigDocument {
  provider?: unknown;
  oidc_issuer?: unknown;
  oidc_client_id?: unknown;
  authorization_endpoint?: unknown;
  token_endpoint?: unknown;
}

export interface ValidatedAuthConfig {
  issuer: string;
  audience: string;
  authorizationOrigin: string;
  tokenEndpoint: string;
}

export interface ValidatedAccessToken {
  subjectSha256: string;
}

export interface OidcEvidence {
  phase: OidcEvidencePhase;
  timestamp: string;
  issuer: string;
  authorization_origin: string;
  audience_claim: OidcAudienceClaim;
  audience: string;
  subject_sha256: string;
  auth_me_status: 200;
  session_create_status: 201;
  session_read_status: 200;
  session_delete_status: 204;
  session_round_trip: true;
}

interface BuildOidcEvidenceInput extends ExpectedOidc {
  phase: OidcEvidencePhase;
  timestamp: string;
  audienceClaim: OidcAudienceClaim;
  subjectSha256: string;
  authMeStatus: number;
  sessionCreateStatus: number;
  sessionReadStatus: number;
  sessionDeleteStatus: number;
}

function exactHttpsOrigin(value: unknown, check: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 4096) {
    throw new OidcEvidenceError(check);
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new OidcEvidenceError(check);
  }
  const port = parsed.port === "443" ? "" : parsed.port;
  const canonical = `https://${parsed.hostname.toLowerCase()}${port ? `:${port}` : ""}`;
  if (
    parsed.protocol !== "https:" ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.pathname !== "/" ||
    parsed.search !== "" ||
    parsed.hash !== "" ||
    value !== canonical
  ) {
    throw new OidcEvidenceError(check);
  }
  return canonical;
}

function exactHttpsUrl(value: unknown, expectedOrigin: string, check: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 4096) {
    throw new OidcEvidenceError(check);
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new OidcEvidenceError(check);
  }
  if (
    parsed.protocol !== "https:" ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.origin !== expectedOrigin ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new OidcEvidenceError(check);
  }
  return value;
}

export function validateAuthConfig(config: AuthConfigDocument, expected: ExpectedOidc): ValidatedAuthConfig {
  const authorizationOrigin = exactHttpsOrigin(expected.authorizationOrigin, "oidc_expected_origin");
  if (
    config.provider !== "oidc" ||
    typeof expected.issuer !== "string" ||
    !expected.issuer ||
    typeof expected.audience !== "string" ||
    !expected.audience ||
    config.oidc_issuer !== expected.issuer ||
    config.oidc_client_id !== expected.audience
  ) {
    throw new OidcEvidenceError("oidc_auth_config");
  }
  const authorizationEndpoint = exactHttpsUrl(config.authorization_endpoint, authorizationOrigin, "oidc_auth_config");
  const tokenEndpoint = exactHttpsUrl(config.token_endpoint, authorizationOrigin, "oidc_auth_config");
  if (authorizationEndpoint === tokenEndpoint) {
    throw new OidcEvidenceError("oidc_auth_config");
  }
  return { issuer: expected.issuer, audience: expected.audience, authorizationOrigin, tokenEndpoint };
}

function decodeClaims(rawToken: string): Record<string, unknown> {
  if (Buffer.byteLength(rawToken, "utf8") > MAX_JWT_BYTES) {
    throw new OidcEvidenceError("oidc_token_format");
  }
  const segments = rawToken.split(".");
  if (segments.length !== 3 || segments.some((segment) => !BASE64URL.test(segment))) {
    throw new OidcEvidenceError("oidc_token_format");
  }
  let decoded: Buffer;
  try {
    decoded = Buffer.from(segments[1], "base64url");
  } catch {
    throw new OidcEvidenceError("oidc_token_format");
  }
  if (decoded.length === 0 || decoded.length > MAX_JWT_PAYLOAD_BYTES) {
    throw new OidcEvidenceError("oidc_token_format");
  }
  let claims: unknown;
  try {
    claims = JSON.parse(decoded.toString("utf8"));
  } catch {
    throw new OidcEvidenceError("oidc_token_format");
  }
  if (claims === null || typeof claims !== "object" || Array.isArray(claims)) {
    throw new OidcEvidenceError("oidc_token_format");
  }
  return claims as Record<string, unknown>;
}

export function validateAccessToken(
  rawToken: string,
  expected: TokenExpectations,
  nowEpochSeconds = Math.floor(Date.now() / 1000),
): ValidatedAccessToken {
  if (expected.audienceClaim !== "aud" && expected.audienceClaim !== "client_id") {
    throw new OidcEvidenceError("oidc_audience_claim");
  }
  exactHttpsOrigin(expected.authorizationOrigin, "oidc_expected_origin");
  const claims = decodeClaims(rawToken);
  const subject = claims.sub;
  const expiration = claims.exp;
  if (
    claims.iss !== expected.issuer ||
    claims[expected.audienceClaim] !== expected.audience ||
    typeof subject !== "string" ||
    subject.trim() === "" ||
    Buffer.byteLength(subject, "utf8") > 1024 ||
    typeof expiration !== "number" ||
    !Number.isSafeInteger(expiration) ||
    expiration <= nowEpochSeconds ||
    (expected.audienceClaim === "client_id" && claims.token_use !== "access")
  ) {
    throw new OidcEvidenceError("oidc_token_claims");
  }
  return { subjectSha256: createHash("sha256").update(subject).digest("hex") };
}

function isIsoUtc(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value) && Number.isFinite(Date.parse(value));
}

export function buildOidcEvidence(input: BuildOidcEvidenceInput): OidcEvidence {
  if (
    !OIDC_EVIDENCE_PHASES.includes(input.phase) ||
    !isIsoUtc(input.timestamp) ||
    input.audienceClaim !== "client_id" && input.audienceClaim !== "aud" ||
    typeof input.issuer !== "string" ||
    !input.issuer ||
    typeof input.audience !== "string" ||
    !input.audience ||
    !SHA256.test(input.subjectSha256) ||
    input.authMeStatus !== 200 ||
    input.sessionCreateStatus !== 201 ||
    input.sessionReadStatus !== 200 ||
    input.sessionDeleteStatus !== 204
  ) {
    throw new OidcEvidenceError("oidc_evidence_schema");
  }
  const authorizationOrigin = exactHttpsOrigin(input.authorizationOrigin, "oidc_evidence_schema");
  return {
    phase: input.phase,
    timestamp: input.timestamp,
    issuer: input.issuer,
    authorization_origin: authorizationOrigin,
    audience_claim: input.audienceClaim,
    audience: input.audience,
    subject_sha256: input.subjectSha256,
    auth_me_status: 200,
    session_create_status: 201,
    session_read_status: 200,
    session_delete_status: 204,
    session_round_trip: true,
  };
}

function ownerUid(): number {
  if (typeof process.getuid !== "function") {
    throw new OidcEvidenceError("oidc_evidence_owner");
  }
  return process.getuid();
}

export function writeOidcEvidence(destination: string, evidence: OidcEvidence): void {
  const validated = buildOidcEvidence({
    phase: evidence.phase,
    timestamp: evidence.timestamp,
    issuer: evidence.issuer,
    authorizationOrigin: evidence.authorization_origin,
    audienceClaim: evidence.audience_claim,
    audience: evidence.audience,
    subjectSha256: evidence.subject_sha256,
    authMeStatus: evidence.auth_me_status,
    sessionCreateStatus: evidence.session_create_status,
    sessionReadStatus: evidence.session_read_status,
    sessionDeleteStatus: evidence.session_delete_status,
  });
  if (Object.keys(evidence).sort().join(",") !== Object.keys(validated).sort().join(",") || evidence.session_round_trip !== true) {
    throw new OidcEvidenceError("oidc_evidence_schema");
  }
  const content = `${JSON.stringify(validated)}\n`;
  writeOwnerOnlyFile(destination, content, {
    maxBytes: MAX_EVIDENCE_BYTES,
    parentCheck: "oidc_evidence_parent",
    destinationCheck: "oidc_evidence_destination",
    sizeCheck: "oidc_evidence_size",
    writeCheck: "oidc_evidence_write",
  });
}

interface OwnerOnlyWriteChecks {
  maxBytes: number;
  parentCheck: string;
  destinationCheck: string;
  sizeCheck: string;
  writeCheck: string;
}

function writeOwnerOnlyFile(destination: string, content: string, checks: OwnerOnlyWriteChecks): void {
  const parent = dirname(destination);
  let parentStat;
  try {
    parentStat = lstatSync(parent);
  } catch {
    throw new OidcEvidenceError(checks.parentCheck);
  }
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink() || parentStat.uid !== ownerUid() || (parentStat.mode & 0o077) !== 0) {
    throw new OidcEvidenceError(checks.parentCheck);
  }
  try {
    lstatSync(destination);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw new OidcEvidenceError(checks.destinationCheck);
    }
  }
  try {
    lstatSync(destination);
    throw new OidcEvidenceError(checks.destinationCheck);
  } catch (error) {
    if (error instanceof OidcEvidenceError) throw error;
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw new OidcEvidenceError(checks.destinationCheck);
    }
  }
  if (Buffer.byteLength(content, "utf8") > checks.maxBytes) {
    throw new OidcEvidenceError(checks.sizeCheck);
  }
  const temporary = join(parent, `.${basename(destination)}.${process.pid}.${randomUUID()}.tmp`);
  let fileDescriptor: number | null = null;
  try {
    fileDescriptor = openSync(
      temporary,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o600,
    );
    writeFileSync(fileDescriptor, content, { encoding: "utf8" });
    fsyncSync(fileDescriptor);
    closeSync(fileDescriptor);
    fileDescriptor = null;
    linkSync(temporary, destination);
    unlinkSync(temporary);
    const directoryDescriptor = openSync(parent, constants.O_RDONLY | constants.O_DIRECTORY);
    try {
      fsyncSync(directoryDescriptor);
    } finally {
      closeSync(directoryDescriptor);
    }
  } catch {
    if (fileDescriptor !== null) {
      try {
        closeSync(fileDescriptor);
      } catch {
        // Preserve the static outer failure.
      }
    }
    try {
      unlinkSync(temporary);
    } catch {
      // The file may already have been atomically renamed.
    }
    throw new OidcEvidenceError(checks.writeCheck);
  }
}

export function writeOidcBearerHandoff(destination: string, accessToken: string): void {
  const segments = accessToken.split(".");
  if (
    accessToken.length === 0 ||
    Buffer.byteLength(accessToken, "utf8") > MAX_JWT_BYTES ||
    segments.length !== 3 ||
    segments.some((segment) => !BASE64URL.test(segment))
  ) {
    throw new OidcEvidenceError("oidc_bearer_handoff");
  }
  writeOwnerOnlyFile(destination, accessToken, {
    maxBytes: MAX_JWT_BYTES,
    parentCheck: "oidc_bearer_handoff",
    destinationCheck: "oidc_bearer_handoff",
    sizeCheck: "oidc_bearer_handoff",
    writeCheck: "oidc_bearer_handoff",
  });
}
