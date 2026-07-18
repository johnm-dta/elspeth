type GuidedRetryKind = "guided_reenter" | "state_revert";

interface GuidedRetryDescriptor {
  kind: GuidedRetryKind;
  sessionId: string;
  requestFingerprint: string;
  operationId: string;
  createdAt: number;
}

export interface GuidedRetryHandle {
  kind: GuidedRetryKind;
  sessionId: string;
  requestFingerprint: string;
  operationId: string;
}

export const GUIDED_RETRY_STORAGE_KEY = "elspeth_guided_operation_retries_v1";
const STORAGE_SCHEMA = "guided-operation-retries.v1";
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SESSION_UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const MAX_DESCRIPTOR_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_DESCRIPTORS = 16;
const MAX_STORAGE_BYTES = 8192;

let fallbackDescriptors: GuidedRetryDescriptor[] = [];
let fallbackAuthoritative = false;

function sessionStorageOrNull(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

function isDescriptor(value: unknown): value is GuidedRetryDescriptor {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    (record.kind === "guided_reenter" || record.kind === "state_revert") &&
    typeof record.sessionId === "string" &&
    SESSION_UUID_PATTERN.test(record.sessionId) &&
    typeof record.requestFingerprint === "string" &&
    SHA256_PATTERN.test(record.requestFingerprint) &&
    typeof record.operationId === "string" &&
    UUID_PATTERN.test(record.operationId) &&
    typeof record.createdAt === "number" &&
    Number.isFinite(record.createdAt) &&
    record.createdAt >= 0
  );
}

function encodedEnvelope(descriptors: GuidedRetryDescriptor[]): string {
  return JSON.stringify({ schema: STORAGE_SCHEMA, descriptors });
}

function encodedBytes(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function boundedDescriptors(
  descriptors: readonly GuidedRetryDescriptor[],
  now = Date.now(),
): GuidedRetryDescriptor[] {
  const live = descriptors
    .filter(
      (descriptor) =>
        descriptor.createdAt <= now && now - descriptor.createdAt <= MAX_DESCRIPTOR_AGE_MS,
    )
    .slice(-MAX_DESCRIPTORS);

  while (live.length > 0 && encodedBytes(encodedEnvelope(live)) > MAX_STORAGE_BYTES) {
    live.shift();
  }
  return live;
}

function readDescriptors(): GuidedRetryDescriptor[] {
  const storage = sessionStorageOrNull();
  if (fallbackAuthoritative || storage === null) {
    fallbackDescriptors = boundedDescriptors(fallbackDescriptors);
    return [...fallbackDescriptors];
  }
  let encoded: string | null;
  try {
    encoded = storage.getItem(GUIDED_RETRY_STORAGE_KEY);
  } catch {
    fallbackAuthoritative = true;
    fallbackDescriptors = boundedDescriptors(fallbackDescriptors);
    return [...fallbackDescriptors];
  }
  if (encoded === null) {
    fallbackDescriptors = boundedDescriptors(fallbackDescriptors);
    return [...fallbackDescriptors];
  }
  try {
    if (encodedBytes(encoded) > MAX_STORAGE_BYTES) throw new Error("oversized retry envelope");
    const parsed = JSON.parse(encoded) as unknown;
    if (typeof parsed !== "object" || parsed === null) throw new Error("invalid retry envelope");
    const envelope = parsed as Record<string, unknown>;
    if (envelope.schema !== STORAGE_SCHEMA || !Array.isArray(envelope.descriptors)) {
      throw new Error("invalid retry envelope");
    }
    if (!envelope.descriptors.every(isDescriptor)) throw new Error("invalid retry descriptor");
    const descriptors = boundedDescriptors(envelope.descriptors);
    fallbackDescriptors = [...descriptors];
    if (descriptors.length !== envelope.descriptors.length) writeDescriptors(descriptors);
    return descriptors;
  } catch {
    fallbackDescriptors = [];
    try {
      storage.removeItem(GUIDED_RETRY_STORAGE_KEY);
      fallbackAuthoritative = false;
    } catch {
      fallbackAuthoritative = true;
    }
    return [];
  }
}

function writeDescriptors(descriptors: GuidedRetryDescriptor[]): void {
  const bounded = boundedDescriptors(descriptors);
  fallbackDescriptors = [...bounded];
  const storage = sessionStorageOrNull();
  if (storage === null) {
    fallbackAuthoritative = true;
    return;
  }
  try {
    if (bounded.length === 0) {
      // Persist an empty generation before best-effort physical deletion. If
      // removeItem fails after this write, a fresh module still reads the
      // tombstone and cannot resurrect completed custody. If setItem itself
      // is unavailable, only this live tab's bounded memory can remain
      // authoritative; browser storage offers no durable recovery path.
      storage.setItem(GUIDED_RETRY_STORAGE_KEY, encodedEnvelope([]));
      storage.removeItem(GUIDED_RETRY_STORAGE_KEY);
      fallbackAuthoritative = false;
      return;
    }
    storage.setItem(GUIDED_RETRY_STORAGE_KEY, encodedEnvelope(bounded));
    fallbackAuthoritative = false;
  } catch {
    // A disabled or full sessionStorage leaves its previous envelope stale.
    // Keep the bounded in-memory generation authoritative until a later write
    // successfully synchronises storage with it.
    fallbackAuthoritative = true;
  }
}

function requestFingerprint(value: string): string {
  // This fingerprint is a storage key, not an authenticity boundary. Four
  // domain-separated FNV-1a lanes avoid retaining raw request identifiers;
  // the backend's canonical request hash remains the collision-safe authority
  // and rejects any accidental operation-id reuse with different semantics.
  const bytes = new TextEncoder().encode(value);
  const mask = (1n << 64n) - 1n;
  const lanes: string[] = [];
  for (let domain = 0; domain < 4; domain += 1) {
    let hash = 0xcbf29ce484222325n ^ BigInt(domain);
    for (const byte of bytes) {
      hash ^= BigInt(byte);
      hash = (hash * 0x100000001b3n) & mask;
    }
    lanes.push(hash.toString(16).padStart(16, "0"));
  }
  return lanes.join("");
}

export function acquireGuidedRetry(
  kind: GuidedRetryKind,
  sessionId: string,
  requestIdentity: readonly string[],
): GuidedRetryHandle {
  if (!SESSION_UUID_PATTERN.test(sessionId)) {
    throw new Error("guided retry sessionId must be a canonical UUID");
  }
  const fingerprint = requestFingerprint(
    JSON.stringify({ schema: "guided-operation-request-fingerprint.v1", kind, requestIdentity }),
  );
  const descriptors = readDescriptors();
  const existing = descriptors.find(
    (descriptor) =>
      descriptor.kind === kind &&
      descriptor.sessionId === sessionId &&
      descriptor.requestFingerprint === fingerprint,
  );
  if (existing !== undefined) return { ...existing };

  const descriptor: GuidedRetryDescriptor = {
    kind,
    sessionId,
    requestFingerprint: fingerprint,
    operationId: crypto.randomUUID(),
    createdAt: Date.now(),
  };
  writeDescriptors([
    ...descriptors.filter((candidate) => candidate.sessionId !== sessionId),
    descriptor,
  ]);
  return { ...descriptor };
}

export function clearGuidedRetry(handle: GuidedRetryHandle): void {
  writeDescriptors(
    readDescriptors().filter(
      (descriptor) =>
        !(
          descriptor.kind === handle.kind &&
          descriptor.sessionId === handle.sessionId &&
          descriptor.requestFingerprint === handle.requestFingerprint &&
          descriptor.operationId === handle.operationId
        ),
    ),
  );
}

export function clearAllGuidedRetries(): void {
  writeDescriptors([]);
}

export function isAmbiguousGuidedRetryFailure(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (typeof error !== "object" || error === null) return false;
  const record = error as { name?: unknown; status?: unknown };
  if (record.name === "AbortError" || record.name === "TimeoutError") return true;
  return typeof record.status === "number" && record.status >= 500 && record.status <= 599;
}
