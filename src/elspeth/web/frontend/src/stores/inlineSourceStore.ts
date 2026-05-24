// ============================================================================
// inlineSourceStore — projected per-session view of the inline-blob source
// attached to the current composition state.
//
// This store is a caching/projection layer, NOT a source of truth. The
// derivation that populates it lives in the ChatPanel wiring (computed from
// compositionState.sources + blob metadata). Three
// downstream consumers read from here:
//   - InlineSourceCreatedTurn — confirmation widget after creation.
//   - InlineSourceDisambiguationTurn — ambiguous-input picker.
//   - Audit-readiness panel surface — readiness row for inline source.
//
// The store has THREE responsibilities, intentionally co-located in one
// container:
//   1. Per-session inline-source summary projection (summariesBySession).
//   2. Disambiguation-related message-ID sets (F-11 re-fire guard for
//      "treat as single row"; F-10 escape for "this isn't source data").
//   3. Per-session dismissal timestamp for the fallback prompt (F-20),
//      so a dismissed prompt does not re-fire within the same session
//      regardless of predicate re-evaluation.
// ============================================================================

import { create } from "zustand";
import type {
  BlobCreationModalityWire,
  BlobMetadata,
  InlineSourceProvenance,
  InlineSourceSummary,
} from "@/types/api";

const INLINE_SOURCE_PREVIEW_CHARS = 1024;
const CONTENT_TYPE_TOKEN_RE = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const SHA256_HEX_RE = /^[a-f0-9]{64}$/;
const TEXT_ENCODER = new TextEncoder();

interface InlineSourceProjectionInput {
  metadata: BlobMetadata;
  contentText: string;
  toProvenance: (
    wire: BlobCreationModalityWire,
  ) => InlineSourceProvenance;
}

function parseContentType(mimeType: string): string | null {
  const [rawBaseType, ...rawParams] = mimeType.split(";");
  const baseType = rawBaseType.trim().toLowerCase();
  const slashIndex = baseType.indexOf("/");
  if (slashIndex <= 0 || slashIndex !== baseType.lastIndexOf("/")) {
    return null;
  }
  const type = baseType.slice(0, slashIndex);
  const subtype = baseType.slice(slashIndex + 1);
  if (
    !CONTENT_TYPE_TOKEN_RE.test(type) ||
    !CONTENT_TYPE_TOKEN_RE.test(subtype)
  ) {
    return null;
  }

  for (const rawParam of rawParams) {
    const param = rawParam.trim();
    if (param === "") return null;
    const equalsIndex = param.indexOf("=");
    if (equalsIndex <= 0 || equalsIndex !== param.lastIndexOf("=")) {
      return null;
    }
    const name = param.slice(0, equalsIndex).trim().toLowerCase();
    const value = param.slice(equalsIndex + 1).trim();
    if (!CONTENT_TYPE_TOKEN_RE.test(name) || value === "") return null;
    if (value.startsWith('"')) {
      if (value.length < 2 || !value.endsWith('"')) return null;
      if (/[\r\n]/.test(value.slice(1, -1))) return null;
      continue;
    }
    if (!CONTENT_TYPE_TOKEN_RE.test(value)) return null;
  }

  return baseType;
}

async function sha256Hex(text: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    TEXT_ENCODER.encode(text),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function deriveInlineSourceRowCount(
  mimeType: string,
  text: string,
): number | null {
  const baseMimeType = parseContentType(mimeType);
  if (baseMimeType !== "text/csv") return null;
  const trimmed = text.trim();
  if (trimmed === "") return 0;
  const lines = trimmed.split("\n").length;
  return Math.max(0, lines - 1);
}

export async function projectInlineSourceSummary({
  metadata,
  contentText,
  toProvenance,
}: InlineSourceProjectionInput): Promise<InlineSourceSummary> {
  if (metadata.content_hash === null || metadata.content_hash === "") {
    throw new Error(
      `inline blob ${metadata.id} has no content_hash — ` +
        "audit-trail invariant violated (Tier-1)",
    );
  }
  const baseMimeType = parseContentType(metadata.mime_type);
  if (baseMimeType === null) {
    throw new Error(
      `inline blob ${metadata.id} has invalid MIME metadata ` +
        `${JSON.stringify(metadata.mime_type)} — refusing projection`,
    );
  }
  if (!SHA256_HEX_RE.test(metadata.content_hash)) {
    throw new Error(
      `inline blob ${metadata.id} content_hash is not canonical ` +
        "SHA-256 lowercase hex — refusing projection",
    );
  }
  const actualHash = await sha256Hex(contentText);
  if (actualHash !== metadata.content_hash) {
    throw new Error(
      `inline blob ${metadata.id} content_hash mismatch — ` +
        "metadata does not match preview bytes",
    );
  }

  return {
    blobId: metadata.id,
    filename: metadata.filename,
    mimeType: metadata.mime_type,
    contentPreview: contentText.slice(0, INLINE_SOURCE_PREVIEW_CHARS),
    rowCount: deriveInlineSourceRowCount(metadata.mime_type, contentText),
    contentHash: metadata.content_hash,
    provenance: toProvenance(metadata.creation_modality),
  };
}

interface InlineSourceState {
  // --- Primary projection: per-session inline-source summary ---
  summariesBySession: Record<string, InlineSourceSummary>;
  setSummary: (sessionId: string, summary: InlineSourceSummary) => void;
  clearSummary: (sessionId: string) => void;
  getSummary: (sessionId: string) => InlineSourceSummary | null;

  // --- Disambiguation re-fire guard (F-11) ---
  // Message IDs for which the user explicitly chose "treat as 1 row".
  // The disambiguation predicate in ChatPanel skips these message IDs.
  userRequestedSingleRowForMessageIds: Set<string>;
  addUserRequestedSingleRow: (messageId: string) => void;

  // --- "Not source data" escape (F-10) ---
  // Message IDs for which the user explicitly chose "this isn't source data".
  // The disambiguation predicate and fallback-prompt predicate skip these.
  nonSourceMessageIds: Set<string>;
  addNonSourceMessage: (messageId: string) => void;

  // --- Fallback-prompt dismiss persistence (F-20) ---
  // Keyed by sessionId. A dismissed fallback prompt must not re-fire
  // within the same session regardless of predicate re-evaluation.
  dismissedAt: Map<string, number>;
  markDismissed: (sessionId: string) => void;
  isDismissed: (sessionId: string) => boolean;
}

export const useInlineSourceStore = create<InlineSourceState>((set, get) => ({
  summariesBySession: {},
  setSummary: (sessionId, summary) =>
    set((s) => ({
      summariesBySession: { ...s.summariesBySession, [sessionId]: summary },
    })),
  clearSummary: (sessionId) =>
    set((s) => {
      const next = { ...s.summariesBySession };
      delete next[sessionId];
      return { summariesBySession: next };
    }),
  getSummary: (sessionId) => get().summariesBySession[sessionId] ?? null,

  userRequestedSingleRowForMessageIds: new Set(),
  addUserRequestedSingleRow: (messageId) =>
    set((s) => ({
      userRequestedSingleRowForMessageIds: new Set([
        ...s.userRequestedSingleRowForMessageIds,
        messageId,
      ]),
    })),

  nonSourceMessageIds: new Set(),
  addNonSourceMessage: (messageId) =>
    set((s) => ({
      nonSourceMessageIds: new Set([...s.nonSourceMessageIds, messageId]),
    })),

  dismissedAt: new Map(),
  markDismissed: (sessionId) =>
    set((s) => {
      const next = new Map(s.dismissedAt);
      next.set(sessionId, Date.now());
      return { dismissedAt: next };
    }),
  isDismissed: (sessionId) => get().dismissedAt.has(sessionId),
}));
