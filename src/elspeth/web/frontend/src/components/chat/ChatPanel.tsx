// src/components/chat/ChatPanel.tsx
import { useEffect, useMemo, useRef, useCallback, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import {
  deriveInlineSourceRowCount,
  projectInlineSourceSummary,
  useInlineSourceStore,
} from "@/stores/inlineSourceStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useComposer } from "@/hooks/useComposer";
import { FOCUSABLE_SELECTOR } from "@/hooks/useFocusTrap";
import {
  getBlobMetadata,
  previewBlobContent,
  toInlineSourceProvenance,
} from "@/api/client";
import { MessageBubble } from "./MessageBubble";
import { groupIntoTurns, turnRepresentativeMessage, type ChatTurn } from "./turns";
import { ComposingIndicator } from "./ComposingIndicator";
import { ChatInput } from "./ChatInput";
import { TemplateCards } from "./TemplateCards";
import { BlobManager } from "@/components/blobs/BlobManager";
import { InlineRunResults } from "@/components/execution/InlineRunResults";
import { CompletionSummary } from "./guided/CompletionSummary";
import { ExitToFreeformButton } from "./guided/ExitToFreeformButton";
import { InlineOptOutCheckbox } from "./guided/InlineOptOutCheckbox";
import { PendingProposalsBanner } from "./PendingProposalsBanner";
import { GuidedChatHistory } from "./guided/GuidedChatHistory";
import { GuidedHistory } from "./guided/GuidedHistory";
import { GuidedTurn } from "./guided/GuidedTurn";
import { InlineSourceCreatedTurn } from "./InlineSourceCreatedTurn";
import { InlineSourceDisambiguationTurn } from "./InlineSourceDisambiguationTurn";
import { InlineSourceFallbackPrompt } from "./InlineSourceFallbackPrompt";
import { InterpretationReviewInlineMessage } from "./InterpretationReviewInlineMessage";
import { sortedSourceEntries } from "@/utils/compositionState";
import type {
  BlobMetadata,
  ChatMessage,
  CompositionProposal,
  CompositionState,
  InlineSourceSummary,
} from "@/types/api";
import type { GuidedStep } from "@/types/guided";
import type { ExampleUseCase, RecommendedStartingPoint } from "./templates_data";

function assertNever(value: never): never {
  throw new Error(`Unhandled template starting point: ${value}`);
}

function isTerminalComposerPhase(
  phase: string | null | undefined,
): boolean {
  return phase === "complete" || phase === "failed" || phase === "cancelled";
}

/**
 * Best-effort row-count from CSV-like text content.
 *
 * Returns `null` when the mime type is not CSV-shaped — we'd rather be honest
 * about not knowing than infer a misleading number (see CLAUDE.md "fabrication
 * decision test"). For text/csv we count newline-separated rows after
 * trimming, subtracting one for the header row when there is content.
 *
 * MIME normalisation: the input may be parameterised (e.g.
 * "text/csv; charset=utf-8" — a perfectly valid value the server may
 * record verbatim from the upload's Content-Type header).  We split on
 * `;` and lowercase before the comparison so parameterised CSVs are not
 * silently classified as "unknown row count".  RFC 7231 §3.1.1.1 reserves
 * `;` as the parameter separator; a literal `;` cannot appear inside the
 * type/subtype tokens.
 *
 * Exported for the ChatPanel test seam — the integration tests stub the blob
 * fetchers and feed text directly into this projector.
 */
export function deriveRowCount(
  mimeType: string,
  text: string,
): number | null {
  return deriveInlineSourceRowCount(mimeType, text);
}

// ── Inline-source disambiguation heuristic (Phase 5a Task 4) ─────────────────
//
// Detect whether a pending composer proposal is BOTH (a) an inline-blob
// source-creation proposal AND (b) one whose row-count interpretation
// looks ambiguous enough to warrant explicit confirmation from the user.
//
// Heuristic v1 (intentionally narrow — false negatives are recoverable
// because the standard PendingProposalsBanner still routes the proposal;
// false positives DO produce a disruptive widget where a banner would
// have sufficed):
//
//   * The proposal's `tool_name` must be "set_pipeline" (the only tool
//     surface that carries an inline_blob source today; the inline-blob
//     create path lives inside set_pipeline.source.inline_blob — see
//     src/elspeth/web/composer/redaction.py:_InlineBlobModel).
//   * The proposal's `arguments_redacted_json` must contain an inline
//     blob under `source.inline_blob` (i.e., the proposal's source is
//     an inline-blob, not a blob_id reference or an external source).
//   * The proposal's `summary` must contain a recognised
//     row-count-ambiguity phrase — currently "I read" or "interpreted as".
//     A Phase 5b refactor will replace this with a structured annotation
//     emitted by the composer pipeline; until then the heuristic is the
//     contract.
//
// The canonical demo prompt (create a data source with government-page URLs
// and explicit scrape-contact fields) MUST NOT trip this heuristic —
// that proposal's summary describes LLM-generated source rows, not an
// interpretation of the user's input. The
// `isAmbiguousInlineProposal` unit tests pin this behaviour.
//
// Exported for the ChatPanel test seam.
export function isAmbiguousInlineProposal(
  proposal: CompositionProposal,
): boolean {
  if (proposal.tool_name !== "set_pipeline") return false;

  // Walk the arguments tree without coercion. The redaction layer
  // preserves the inline_blob marker even when the content has been
  // summarised, so a structural check is sufficient and does NOT need
  // to peek at the (possibly redacted) content string.
  const source = proposal.arguments_redacted_json["source"];
  if (typeof source !== "object" || source === null) return false;
  const inlineBlob = (source as Record<string, unknown>)["inline_blob"];
  if (typeof inlineBlob !== "object" || inlineBlob === null) return false;

  const summary = proposal.summary;
  // Case-insensitive substring match. Two recognised ambiguity phrases
  // — composer narration that explicitly frames "I parsed your input
  // as N rows" or "interpreted as N items". The canonical demo
  // proposal does NOT use either phrase (it describes generation, not
  // interpretation).
  const lowered = summary.toLowerCase();
  return lowered.includes("i read") || lowered.includes("interpreted as");
}

// ── Originating user-message resolution ──────────────────────────────────────
//
// The F-10 / F-11 re-fire guards key on the user message ID that
// triggered the assistant's tool-call. The proposal itself only
// carries `tool_call_id`; we recover the user message by walking the
// chat history: find the assistant message bearing that tool call,
// then walk backwards to the nearest preceding user message.
//
// Returns null when:
//   * No assistant message has a tool call with the given id (proposal
//     orphaned from its message — should not happen in production but
//     we surface as null rather than crash).
//   * The assistant message exists but no user message precedes it
//     (also shouldn't happen — the composer always responds to a user
//     turn — but null is the honest signal).
//
// Callers that need a non-null message ID (e.g., to populate the F-10
// guard) MUST handle null by falling back to the standard banner —
// firing the F-10 guard with a synthesised ID would corrupt the
// guard's data model.
export function findOriginatingMessageId(
  messages: ReadonlyArray<ChatMessage>,
  toolCallId: string,
): string | null {
  const assistantIndex = messages.findIndex((m) =>
    m.tool_calls?.some((tc) => tc.id === toolCallId) ?? false,
  );
  if (assistantIndex < 0) return null;
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") return messages[i].id;
  }
  return null;
}

/**
 * Extract the proposed-rows list from an inline-blob proposal's
 * arguments tree. The redaction layer summarises the content string
 * itself, so the only durable row-count signal in the proposal is
 * the structural shape of any inline-blob fields the LLM populated
 * alongside the (now-redacted) content — chiefly the JSON schema's
 * own row metadata. Today the inline_blob redacted shape exposes
 * `filename` / `mime_type` but NOT a parsed row list — see
 * `_InlineBlobModel` in redaction.py.
 *
 * For Phase 5a Task 4 we surface the user's original input as the
 * source of truth for the row list (split on common delimiters) so
 * the widget has something concrete to show. This is a presentation
 * concern only — the authoritative row data lives in the eventual
 * inline_blob.content on the server; the displayed list is a parse
 * preview the user is being asked to confirm.
 *
 * Exported for the ChatPanel test seam.
 */
export function parseProposedRowsFromUserInput(
  userInput: string,
): ReadonlyArray<string> {
  // Strip a leading prose preamble like "check these URLs:" — the
  // rows-of-interest are typically after the first ":" when present.
  const afterColon = userInput.includes(":")
    ? userInput.slice(userInput.indexOf(":") + 1)
    : userInput;
  // Split on commas OR newlines; trim each fragment; drop empties.
  // We intentionally use a permissive split rather than try to do
  // CSV-grade quoting — the user has the "Edit the rows" affordance
  // for cases where the heuristic guesses wrong.
  return afterColon
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// ── Inline-source fallback heuristic (Phase 5a Task 5) ───────────────────────
//
// `looksLikeData` is the safety-net predicate the chat panel runs against
// recent user messages to decide whether to surface the
// InlineSourceFallbackPrompt. The widget is the floor for the
// inline-source-from-chat path: if the composer LLM ignores Task 8's
// prompt nudge and never proposes a `set_pipeline` with an inline_blob
// source for source-shaped typed input, this predicate triggers the
// fallback affordance so the user is not stuck.
//
// CLOSED LIST — two recognised shapes. Both are HIGH-SPECIFICITY signals.
// The Phase 5a Task 5 spec also lists a third clause ("single short typed
// phrase under 200 chars containing no ?"), but that clause matches almost
// every chat message the user could type and would dominate the predicate
// — biasing toward FALSE POSITIVES (a disruptive affordance surfacing on
// every conversational turn), which is the opposite of the spec's
// "bias toward false negatives" framing. Surfacing the fallback on a
// missed URL is recoverable (the user re-types or accepts the fallback);
// surfacing it on every casual message is a UX bug. We deliberately omit
// clause 3; cf. the InlineSourceFallbackPrompt self-review in the
// commit message.
//
//   1. URL — http(s) prefix anywhere in the content.
//   2. Comma-separated list — 2..10 comma-separated tokens (matches a
//      typed list like "alice, bob, carol" but not a normal English
//      sentence with one or two embedded commas because we require the
//      entire trimmed content to consist of comma-separated tokens).
//
/** Min items in a typed list to qualify (2 items = at least one comma). */
const LIST_TOKEN_MIN_COUNT = 2;
/** Max items in a typed list to qualify (spec §Task 5 detection §3). */
const LIST_TOKEN_MAX_COUNT = 10;
/**
 * Per-token max word count.  A typed-source list token is almost always
 * 1..3 words — names ("Alice Smith"), slugs ("government-data"), URLs
 * ("a.com"), or short identifiers.  English prose with an embedded
 * comma ("hello, world how are you doing today") has multi-word tokens
 * (6+ words after the first split).
 *
 * Bias toward false negatives: a list of multi-word phrases longer
 * than 3 words (e.g. "5 government web pages, the local council site,
 * the open-data portal") would fail this check.  That's deliberate
 * — the predicate is the SAFETY NET; missing a phrase-shaped list is
 * recoverable (user re-types more concisely, or the LLM proposes a
 * source on the next turn).  Over-firing on every English sentence
 * with one comma is the worse failure mode.
 */
const LIST_TOKEN_MAX_WORDS = 3;

// Exported for the ChatPanel test seam — unit-tested directly so the
// predicate's shape is pinned without going through the widget render.
export function looksLikeData(content: string): boolean {
  const trimmed = content.trim();
  if (trimmed === "") return false;
  if (/https?:\/\//.test(trimmed)) return true;
  // Comma-separated list of 2..10 short tokens.  We split (not regex-
  // match) so we can apply the per-token word-count cap structurally.
  // The earlier regex-only `^[^,]+(?:, [^,]+){1,9}$` approach matched
  // any prose containing a comma because `[^,]+` is greedy on
  // whitespace; a structural check is clearer and easier to evolve.
  const parts = trimmed.split(",").map((p) => p.trim());
  if (parts.length < LIST_TOKEN_MIN_COUNT) return false;
  if (parts.length > LIST_TOKEN_MAX_COUNT) return false;
  // Every token must be non-empty (rejects trailing-comma artefacts
  // like "a, b,") AND ≤ LIST_TOKEN_MAX_WORDS words (rejects prose
  // with one or two embedded commas).
  for (const part of parts) {
    if (part === "") return false;
    // Split on any whitespace run; filter empties from the leading
    // or trailing edge already handled by trim, but defensive against
    // double-spaces.
    const words = part.split(/\s+/).filter((w) => w.length > 0);
    if (words.length > LIST_TOKEN_MAX_WORDS) return false;
  }
  return true;
}

/** Narrow `source.options["blob_ref"]` (which is `unknown`) to a string. */
function readSourceBlobRef(source: { options: Record<string, unknown> } | null): string | null {
  if (source === null) return null;
  const raw = source.options["blob_ref"];
  return typeof raw === "string" && raw !== "" ? raw : null;
}

function readBlobRef(state: CompositionState | null): string | null {
  if (state === null) return null;
  for (const [, source] of sortedSourceEntries(state)) {
    const ref = readSourceBlobRef(source);
    if (ref !== null) return ref;
  }
  return null;
}

function isInlineSourceBlob(metadata: BlobMetadata): boolean {
  return (
    metadata.created_by === "assistant" &&
    metadata.created_from_message_id !== null
  );
}

/**
 * Per-step placeholder text for the chat input in guided mode (Phase A slice 4).
 *
 * The wording frames what's *useful* to ask at each wizard step.  This is a
 * UX nudge, not a server-enforced scope — the backend still validates
 * step_index against the live session.step and the per-step skill briefing
 * shapes what the LLM will engage with.  Mirrors the playbook fragments in
 * src/elspeth/web/composer/guided/skills/step_*.md.
 *
 * CLOSED LIST — must cover every GuidedStep member.  Adding a new step
 * member without extending this map produces a TypeScript exhaustiveness
 * error at the lookup site (see assertion in the lookup below).
 */
const GUIDED_CHAT_PLACEHOLDERS: Record<GuidedStep, string> = {
  step_1_source: "Ask about source options, columns, or paste a sample row…",
  step_2_sink: "Ask about sink config, outputs, or schema mode…",
  step_2_5_recipe_match: "Ask about the suggested recipe or alternatives…",
  step_3_transforms: "Ask about the proposed transform chain…",
};

interface ChatPanelProps {
  onOpenSecrets?: () => void;
  // Phase 1B Panel UX-M3: the freeform-mode header surfaces a small
  // "Change my default" link beside "Switch to guided" — the third
  // opt-out/opt-in surface spec 05 enumerates (alongside the inline
  // checkbox in the guided body and the Composer-preferences pane in
  // the account menu). The link opens the same modal the account menu
  // does; threaded as a prop from App.tsx for parity with onOpenSecrets.
  onOpenComposerPreferences?: () => void;
}

/**
 * Main chat panel combining the message list, composing indicator, and input.
 *
 * Auto-scrolls to the bottom on new messages unless the user has scrolled up.
 * Focus returns to the ChatInput textarea after the assistant response arrives.
 */
export function ChatPanel({
  onOpenSecrets,
  onOpenComposerPreferences,
}: ChatPanelProps) {
  const messages = useSessionStore((s) => s.messages);
  // Project audit-grade message rows onto user-visible turns. One bubble per
  // turn — see ./turns.ts for the grouping rules. Memoised on the messages
  // reference because the store updates the array on append, not in place.
  const chatTurns = useMemo(() => groupIntoTurns(messages), [messages]);
  // Last complete agent turn id — the inline-source summary attaches to this
  // turn's bubble. null when no complete agent turn exists yet (e.g. fresh
  // session, mid-flight first turn, or session-restore before any chat); the
  // standalone fallback widget further down handles those cases. Recomputed
  // on every chatTurns change since the value rolls forward across turns.
  const inlineSourceTargetTurnId = useMemo<string | null>(() => {
    for (let i = chatTurns.length - 1; i >= 0; i--) {
      const t = chatTurns[i];
      if (t.kind === "agent" && t.isComplete) return t.id;
    }
    return null;
  }, [chatTurns]);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const sessions = useSessionStore((s) => s.sessions);
  const compositionState = useSessionStore((s) => s.compositionState);
  const compositionProposals = useSessionStore((s) => s.compositionProposals);
  const staleProposalIds = useSessionStore((s) => s.staleProposalIds);
  const proposalActionPendingIds = useSessionStore(
    (s) => s.proposalActionPendingIds,
  );
  const acceptProposal = useSessionStore((s) => s.acceptProposal);
  const rejectProposal = useSessionStore((s) => s.rejectProposal);
  const composerProgress = useSessionStore((s) => s.composerProgress);
  const clearError = useSessionStore((s) => s.clearError);
  const forkFromMessage = useSessionStore((s) => s.forkFromMessage);
  // Guided-mode discriminator state.  Selectors are hoisted here (not inside a
  // branch) to comply with React's Rules of Hooks; the discriminator early
  // returns below decide which surface to render based on these values.
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const guidedNextTurn = useSessionStore((s) => s.guidedNextTurn);
  const respondGuided = useSessionStore((s) => s.respondGuided);
  const chatGuided = useSessionStore((s) => s.chatGuided);
  const guidedChatPending = useSessionStore((s) => s.guidedChatPending);
  const guidedResponsePending = useSessionStore((s) => s.guidedResponsePending);
  const enterGuided = useSessionStore((s) => s.enterGuided);

  const activeSessionTitle = sessions.find((s) => s.id === activeSessionId)?.title;
  const {
    sendMessage,
    retryMessage,
    cancelComposition,
    isComposing,
    error,
    errorDetails,
  } = useComposer();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const guidedLogRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [showBlobManager, setShowBlobManager] = useState(false);
  const [inputText, setInputText] = useState("");
  const activeComposerMessage = findActiveComposerMessage(messages);
  const proposalsByToolCallId = useMemo(
    () =>
      new Map(
        compositionProposals.map((proposal) => [
          proposal.tool_call_id,
          proposal,
        ]),
      ),
    [compositionProposals],
  );

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    setShowScrollButton(false);
  }

  // Track whether the user has scrolled up from the bottom
  function handleScroll() {
    const container = scrollContainerRef.current;
    if (!container) return;
    const threshold = 40; // pixels from bottom
    const atBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold;
    setShowScrollButton(!atBottom);
  }

  const shouldShowComposerProgress =
    isComposing || isTerminalComposerPhase(composerProgress?.phase);

  // Auto-scroll to bottom when new messages arrive (unless user scrolled up).
  // Empty sessions render template cards above the sentinel; scrolling to the
  // bottom on first paint clips the top row of cards under the header.
  useEffect(() => {
    if (messages.length === 0 && !isComposing) return;
    if (!showScrollButton) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isComposing, showScrollButton]);

  // Return focus to input when composing ends only if focus stayed in the
  // composer. Do not steal focus from proposal buttons, recovery actions, or
  // side-rail controls the user reached for while the request was running.
  useEffect(() => {
    const active = document.activeElement;
    const safeToRestore =
      active === null ||
      active === document.body ||
      active === inputRef.current;
    if (!isComposing && safeToRestore) {
      inputRef.current?.focus();
    }
  }, [isComposing]);

  // Reset scroll state when switching sessions
  useEffect(() => {
    setShowScrollButton(false);
  }, [activeSessionId]);

  // ── Inline-source projection (Phase 5a Task 3) ─────────────────────────────
  //
  // When the active composition's source blob_ref resolves to an assistant-
  // created blob with chat-message provenance, project that blob's metadata +
  // a bounded content preview into the inlineSourceStore. The summary is rendered
  // inside the agent bubble (MessageBubble's "Sources created" disclosure
  // group) — the store is the projection layer for downstream consumers
  // (Task 4 disambiguation widget, Task 7 audit-readiness row, and the
  // bubble's sources-created group).
  //
  // The effect is cancelled via a `cancelled` flag set in the cleanup so that
  // a session-switch or composition-replace mid-fetch does not race the
  // older response into the newer summary slot.
  //
  // Do not use creation_modality alone as the predicate: ordinary uploaded
  // files and assistant-created inline blobs may both be `verbatim`. The
  // backend distinguishes chat-created inline blobs by `created_by=assistant`
  // plus a non-null `created_from_message_id`; browser uploads and pipeline
  // outputs do not carry that pair. Checking metadata before preview fetch
  // keeps large uploaded sources out of the chat-created-source projection.
  const blobRef = readBlobRef(compositionState);
  const setInlineSourceSummary = useInlineSourceStore((s) => s.setSummary);
  const clearInlineSourceSummary = useInlineSourceStore((s) => s.clearSummary);
  const inlineSourceSummary = useInlineSourceStore((s) =>
    activeSessionId !== null ? s.summariesBySession[activeSessionId] ?? null : null,
  );

  // ── Interpretation review pending events (Phase 5b Task 5) ────────────────
  //
  // Freeform-mode surfacing of LLM-interpretation review affordances.
  // Guided mode renders these inside the GuidedTurn dispatch
  // (InterpretationReviewTurn).  In freeform mode they appear inline in
  // the chat message stream — one InterpretationReviewInlineMessage per
  // pending event, ordered by created_at ascending so the oldest
  // unresolved interpretation surfaces first.
  //
  // The dispatch predicate is structural: the store has at least one
  // pending event for the active session AND the freeform branch is
  // reached (the guided branches return early above).  We do NOT key off
  // the proposal summary text here — interpretation events come from a
  // separate wire route (POST /interpretations/resolve) and live in
  // their own store; an inline_blob proposal whose summary lacks
  // "I read" / "interpreted as" simply does not produce a pending
  // interpretation event, so it cannot trigger this widget.  Task 5
  // test 17 asserts this routing predicate's negative branch.
  //
  // Subscribe to the per-session pending map so a new pending event
  // arriving via store.addPendingEvent / store.refreshAll triggers a
  // re-render.  Reading via a stable selector that returns an empty
  // record (rather than undefined) when the session has no entry yet
  // avoids identity churn from `?? {}` on every render.
  const pendingInterpretationEventsBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const pendingInterpretationEvents = useMemo(() => {
    if (activeSessionId === null) return [];
    const map = pendingInterpretationEventsBySession[activeSessionId];
    if (!map) return [];
    // Sort by created_at ascending (ISO-8601 strings sort lexicographically
    // in chronological order).  Stable order is important because the
    // widget is rendered with the event id as the React key — re-sorts
    // on each render would not remount the components, but the visual
    // top-to-bottom order would shift if a new event arrived with an
    // earlier created_at (which the wire contract permits but is rare).
    return Object.values(map).sort((a, b) =>
      a.created_at < b.created_at ? -1 : a.created_at > b.created_at ? 1 : 0,
    );
  }, [activeSessionId, pendingInterpretationEventsBySession]);

  // ── Interpretation review resolve-success confirmation (Phase 5b.18b.8) ──
  //
  // The interpretation-review widget (InterpretationReviewInlineMessage)
  // unmounts on successful resolve because the parent re-renders with the
  // event removed from `pendingBySession`. The widget's onResolved callback
  // is therefore the only signal we can use to push a confirmation line
  // back into the chat after dismissal — by the time the next render runs,
  // `event.user_term` is no longer reachable.
  //
  // Spec lines 768-774: "Got it — using your interpretation of *<user_term>*."
  // We render this as an assistant-styled chat bubble inside the
  // message-stream region so the user sees a natural continuation of the
  // conversation. The confirmations are LOCAL UI state — NOT pushed to
  // sessionStore.messages and NOT written to the audit trail (the audit
  // trail's interpretation_event row is the canonical record; this is the
  // human-readable echo, an explicit UI nudge per spec line 772).
  //
  // Each confirmation carries a local id so React reconciliation keeps
  // confirmation bubbles stable when new events resolve while older
  // confirmations are still visible. Confirmations persist for the
  // lifetime of the ChatPanel mount; switching sessions or reloading clears
  // them, which matches the "ephemeral UI nudge" intent.
  interface ResolveConfirmation {
    id: string;
    userTerm: string;
  }
  const [resolveConfirmations, setResolveConfirmations] = useState<
    ReadonlyArray<ResolveConfirmation>
  >([]);
  // Monotonic counter for confirmation ids — useId is per-component and
  // would collide across appended entries; crypto.randomUUID is overkill
  // for ephemeral UI state. A ref-backed counter is identity-stable across
  // renders and produces predictable, debuggable ids in DevTools.
  const confirmationIdCounterRef = useRef(0);

  const handleInterpretationResolved = useCallback(
    (resolvedEvent: { user_term: string | null }) => {
      // Skip the confirmation when user_term is null — this is the case
      // for opt-out and auto-interpretation rows, which do not have a
      // user term to echo. The opt-out flow has its own confirm dialog
      // ("Stop reviewing interpretations for this session"); a chat-stream
      // echo would be redundant noise.
      const userTerm = resolvedEvent.user_term;
      if (userTerm === null || userTerm === "") return;
      confirmationIdCounterRef.current += 1;
      const id = `resolve-confirmation-${confirmationIdCounterRef.current}`;
      setResolveConfirmations((prev) => [...prev, { id, userTerm }]);
    },
    [],
  );

  // Reset confirmations on session switch — the confirmations are
  // per-session UI state and showing previous-session confirmations in a
  // new session's chat would be confusing.
  useEffect(() => {
    setResolveConfirmations([]);
    confirmationIdCounterRef.current = 0;
  }, [activeSessionId]);

  // Disambiguation re-fire guards (F-10 / F-11). Subscribed via the
  // store so the widget surface updates when a guard flips — without
  // this, clicking "treat as 1 row" once would not remove the widget
  // for the same proposal/message on the next render until something
  // else triggered a re-render.
  const userRequestedSingleRowForMessageIds = useInlineSourceStore(
    (s) => s.userRequestedSingleRowForMessageIds,
  );
  const nonSourceMessageIds = useInlineSourceStore(
    (s) => s.nonSourceMessageIds,
  );
  const addUserRequestedSingleRow = useInlineSourceStore(
    (s) => s.addUserRequestedSingleRow,
  );
  const addNonSourceMessage = useInlineSourceStore(
    (s) => s.addNonSourceMessage,
  );
  const markFallbackDismissed = useInlineSourceStore((s) => s.markDismissed);
  // Subscribe through the dismissedAt Map so the predicate re-evaluates
  // when a dismissal lands. Calling `isDismissed(sessionId)` inside the
  // selector body itself would not trigger a re-render on store change
  // (the selector returns a primitive boolean derived from the Map but
  // doesn't subscribe to the Map's identity change unless we read the
  // Map directly).
  const fallbackDismissedAt = useInlineSourceStore((s) => s.dismissedAt);

  useEffect(() => {
    if (activeSessionId === null) return;
    if (blobRef === null) {
      // No inline source attached — clear any stale projection.
      clearInlineSourceSummary(activeSessionId);
      return;
    }
    let cancelled = false;
    const sessionId = activeSessionId;
    const targetBlobId = blobRef;
    void (async () => {
      try {
        const meta = await getBlobMetadata(sessionId, targetBlobId);
        if (cancelled) return;
        if (!isInlineSourceBlob(meta)) {
          clearInlineSourceSummary(sessionId);
          return;
        }
        const text = await previewBlobContent(sessionId, targetBlobId);
        if (cancelled) return;
        const summary = await projectInlineSourceSummary({
          metadata: meta,
          contentText: text,
          toProvenance: toInlineSourceProvenance,
        });
        if (cancelled) return;
        setInlineSourceSummary(sessionId, summary);
      } catch (err) {
        // Frontend display-projection failure.  Bound `err` (not bare
        // `catch {}`) so programming errors are debuggable:
        //   * Transient blob-fetch failures (Tier-3 boundary —
        //     authenticated endpoint returning 5xx) are the expected
        //     case; we keep the last-known-good summary in the store.
        //   * `toInlineSourceProvenance` throws on an unknown wire
        //     `CreationModality` value (exhaustiveness `never`).  That
        //     would be a wire-contract drift we MUST see, not silently
        //     swallow.
        //   * The Tier-1 hash-invariant throw above lands here.
        // `console.error` follows the in-codebase frontend convention
        // (see App.tsx [preferences], CatalogDrawer.tsx, etc.) — it is
        // NOT the backend `slog` channel (CLAUDE.md "Telemetry and
        // Logging").  The audit trail of the failure itself lives on
        // the server (blob-fetch attempts are recorded server-side);
        // this is the operational mirror so a developer opening
        // devtools sees the projection failure.
        if (!cancelled) {
          console.error("[inline-source] projection failed:", err);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    activeSessionId,
    blobRef,
    setInlineSourceSummary,
    clearInlineSourceSummary,
  ]);

  // ── Disambiguation candidate set (Phase 5a Task 4) ─────────────────────────
  //
  // A proposal is a disambiguation candidate iff:
  //   1. It is currently pending (status === "pending") and not stale.
  //   2. `isAmbiguousInlineProposal(proposal)` is true (inline blob +
  //      row-count-ambiguity narration phrases).
  //   3. We can resolve a non-null originating user message ID for it
  //      (the F-10/F-11 guards need a stable key).
  //   4. The originating message ID is NOT in either re-fire guard
  //      set — the user has already disambiguated in either direction
  //      and we must not re-prompt.
  //
  // Each surviving proposal yields one widget. The matching proposal
  // IDs are also removed from the standard PendingProposalsBanner
  // input set so a single proposal does not appear in BOTH surfaces.
  //
  // useMemo is used here because the computation walks the message
  // list per proposal; recomputing on every render (especially while
  // typing in the chat input, which fires re-renders via the
  // ChatInput's onChange callback) is wasted work.
  const disambiguationCandidates = useMemo(() => {
    return compositionProposals
      .filter((p) => p.status === "pending")
      .filter((p) => !staleProposalIds.includes(p.id))
      .filter(isAmbiguousInlineProposal)
      .map((proposal) => {
        const messageId = findOriginatingMessageId(
          messages,
          proposal.tool_call_id,
        );
        if (messageId === null) return null;
        if (userRequestedSingleRowForMessageIds.has(messageId)) return null;
        if (nonSourceMessageIds.has(messageId)) return null;
        const userMessage = messages.find((m) => m.id === messageId);
        // userMessage existence is implied by findOriginatingMessageId
        // returning a non-null id (it found the id by walking the same
        // messages array), but TypeScript can't see that — narrow with
        // an explicit guard. A null here would be a bug we want to
        // surface, but rather than crash the chat panel we skip the
        // candidate; the standard banner picks it up.
        if (!userMessage) return null;
        return {
          proposal,
          messageId,
          userInput: userMessage.content,
          proposedRows: parseProposedRowsFromUserInput(userMessage.content),
        };
      })
      .filter(
        (c): c is {
          proposal: CompositionProposal;
          messageId: string;
          userInput: string;
          proposedRows: ReadonlyArray<string>;
        } => c !== null,
      );
  }, [
    compositionProposals,
    staleProposalIds,
    messages,
    userRequestedSingleRowForMessageIds,
    nonSourceMessageIds,
  ]);

  // Proposal IDs claimed by the disambiguation widget — excluded from
  // the standard PendingProposalsBanner so a single proposal does not
  // appear in both surfaces.
  const disambiguationProposalIds = useMemo(
    () => new Set(disambiguationCandidates.map((c) => c.proposal.id)),
    [disambiguationCandidates],
  );

  const bannerProposals = useMemo(
    () =>
      compositionProposals.filter((p) => !disambiguationProposalIds.has(p.id)),
    [compositionProposals, disambiguationProposalIds],
  );

  // ── Disambiguation action handlers ─────────────────────────────────────────
  //
  // Each handler is bound to one button on the widget. The accessible
  // names on those buttons are load-bearing — see the CLOSED LIST
  // comment in InlineSourceDisambiguationTurn.tsx.
  //
  // "Yes — N rows":   delegate to acceptProposal (the standard
  //                   accept-proposal flow lands the inline_blob and
  //                   the inlineSourceStore projection effect picks it
  //                   up on the next composition-state update).
  // "No — treat as 1 row":  reject the proposal + record the F-11 guard
  //                          + ask the LLM to re-interpret as a single
  //                          row. The LLM re-issuing a multi-row
  //                          interpretation for the same message would
  //                          re-fire this widget without the guard.
  // "Edit the rows":   reject the proposal + ask the LLM for an
  //                    in-chat edit of the row list (mirrors the
  //                    chat-mediated edit path Task 3 introduced for
  //                    the post-success surface).
  // "This isn't source data":  record the F-10 guard + tell the LLM
  //                            the message wasn't source data and to
  //                            continue without creating one. We do
  //                            NOT reject the proposal here because
  //                            the F-10 surface implies "stop framing
  //                            my message as source"; the next
  //                            assistant turn will rebuild without
  //                            the inline source and the stale
  //                            proposal will be marked stale by the
  //                            standard rebase pipeline.
  const handleDisambiguationConfirmMultiRow = useCallback(
    (proposalId: string) => {
      void acceptProposal(proposalId);
    },
    [acceptProposal],
  );

  const handleDisambiguationTreatAsOneRow = useCallback(
    (proposalId: string) => {
      const candidate = disambiguationCandidates.find(
        (c) => c.proposal.id === proposalId,
      );
      if (!candidate) return;
      addUserRequestedSingleRow(candidate.messageId);
      void rejectProposal(proposalId);
      sendMessage(
        "Treat my previous message as a single row, not multiple — " +
          "please re-interpret the input as one row and update the source.",
      );
    },
    [
      disambiguationCandidates,
      addUserRequestedSingleRow,
      rejectProposal,
      sendMessage,
    ],
  );

  const handleDisambiguationEditRows = useCallback(
    (proposalId: string) => {
      const candidate = disambiguationCandidates.find(
        (c) => c.proposal.id === proposalId,
      );
      if (!candidate) return;
      void rejectProposal(proposalId);
      const rowsListing = candidate.proposedRows
        .map((row, idx) => `${idx + 1}. ${row}`)
        .join("\n");
      sendMessage(
        `I'd like to edit the proposed row list before continuing.\n\n` +
          `Current rows:\n${rowsListing}\n\n` +
          `Please ask me what changes I want, then update the inline source.`,
      );
    },
    [disambiguationCandidates, rejectProposal, sendMessage],
  );

  const handleDisambiguationNotSourceData = useCallback(
    (messageId: string) => {
      addNonSourceMessage(messageId);
      sendMessage(
        "That message isn't source data — please continue without " +
          "creating a source from it.",
      );
    },
    [addNonSourceMessage, sendMessage],
  );

  // ── Inline-source fallback predicate (Phase 5a Task 5) ───────────────────
  //
  // The fallback prompt fires when ALL of:
  //   1. User has sent ≥1 user message in the session.
  //   2. No source is bound on the composition state.
  //   3. The most recent user message that survives `looksLikeData` is
  //      the actual candidate (we walk the last few user messages so a
  //      transient question turn doesn't suppress the affordance when a
  //      prior URL is still the unresolved input).
  //   4. The composer is not currently responding to that user message.
  //      The fallback is a post-turn safety net, not a mid-compose
  //      competing offer.
  //   5. No source-related tool call is in flight on the latest
  //      assistant message (set_pipeline, set_source_from_blob,
  //      set_source). If one is in flight the LLM is mid-response and
  //      we must not race the affordance against the proposal pipeline.
  //   6. The fallback has not been dismissed for this session (F-20).
  //
  // The candidate is the most recent looksLikeData-positive user message
  // text, walking backwards through the LAST 3 user messages (a 3-turn
  // window — wider than 1 lets a "what does it cost?" follow-up question
  // not suppress the affordance for a still-unresolved URL above it; the
  // spec mentions N=2 turns, we use 3 for the same reason). Older
  // unresolved candidates fade out naturally as the chat scrolls.
  const fallbackCandidate = useMemo(() => {
    const userMessages = messages.filter((m) => m.role === "user");
    if (userMessages.length === 0) return null;
    const recent = userMessages.slice(-3).reverse();
    for (const m of recent) {
      if (looksLikeData(m.content)) return m.content;
    }
    return null;
  }, [messages]);

  // Inflight source-tool-call check — gate on the LATEST assistant
  // message's tool_calls. The ToolCall wire shape carries the function
  // name at `tc.function.name` (LiteLLM convention; see types/index.ts).
  //
  // CLOSED LIST — the three source-mutating tool names. Adding a fourth
  // source-mutating tool to the composer means widening this set; the
  // CLOSED-LIST framing prevents quiet drift.
  const inflightSourceToolNames: ReadonlySet<string> = useMemo(
    () => new Set(["set_pipeline", "set_source_from_blob", "set_source"]),
    [],
  );
  const hasInflightSourceCall = useMemo(() => {
    const lastAssistant = [...messages]
      .reverse()
      .find((m) => m.role === "assistant");
    if (!lastAssistant) return false;
    const calls = lastAssistant.tool_calls ?? [];
    return calls.some((tc) => inflightSourceToolNames.has(tc.function.name));
  }, [messages, inflightSourceToolNames]);

  // Source-bound predicate. The shape mirrors the spec: either no
  // composition state OR no named source OR every source plugin slot is the
  // empty string (the composer's pre-source-bound representation).
  const compositionHasSource =
    compositionState !== null &&
    Object.values(compositionState.sources).some((source) => source.plugin !== "");

  // F-20 session-scoped dismissal. The store action `markDismissed`
  // populates `dismissedAt[sessionId]`; we read via the Map identity
  // we subscribed to above so the predicate re-evaluates on flip.
  const sessionDismissed =
    activeSessionId !== null && fallbackDismissedAt.has(activeSessionId);

  const shouldRenderFallback =
    fallbackCandidate !== null &&
    !isComposing &&
    !hasInflightSourceCall &&
    !compositionHasSource &&
    !sessionDismissed;

  // F-3 — no API jargon in the user-visible chat message. The dispatched
  // chat turn reads as natural language; the composer prompt (Task 8)
  // teaches the LLM to recognise this framing and call set_pipeline
  // with an inline_blob source. The fallback path goes through the
  // SAME tool-use loop as the LLM-initiated path, which is what
  // preserves audit-trail equivalence.
  const handleFallbackAccept = useCallback(
    (text: string) => {
      sendMessage(`Use this as my source data:\n\n${text}`);
    },
    [sendMessage],
  );

  const handleFallbackDismiss = useCallback(() => {
    if (activeSessionId !== null) {
      markFallbackDismissed(activeSessionId);
    }
  }, [activeSessionId, markFallbackDismissed]);

  /**
   * "Edit the list" handler (Phase 5a Task 3 v1).
   *
   * v1 emits a conversational instruction to the composer LLM, pre-loaded
   * with the current preview so the user can amend in the chat textarea.
   * This is the minimal viable path: the composer's `set_pipeline` tool
   * accepts an `inline_blob.content` re-write, so an LLM-mediated edit goes
   * through the existing proposal-approval pipeline (same audit-event
   * lineage as the original creation).
   *
   * A direct in-browser textarea modal is the natural Task 6+ follow-up —
   * but it requires a `setPipeline` action surface in `sessionStore`, which
   * does not exist today (all pipeline mutations flow through the LLM
   * tool-use loop). Shipping the textarea modal here would mean adding that
   * surface, which is out of scope for Task 3. The chat-mediated handler
   * lets Task 6's integration test exercise the click path; the modal
   * upgrade is tracked separately.
   */
  const handleEditInlineSource = useCallback(
    (summary: InlineSourceSummary) => {
      const prompt =
        `I'd like to edit the inline source "${summary.filename}". ` +
        `Current contents:\n\n${summary.contentPreview}\n\n` +
        `Please update it per the changes I describe in my next message.`;
      sendMessage(prompt);
    },
    [sendMessage],
  );

  // Spec §7.4 — maintain focus on the first interactive element of the new turn
  // after step advance.  Without this, a step-advancing button click unmounts
  // the button before the browser can return focus elsewhere, so focus falls to
  // <body>.  Keyboard users then have to Tab from the very top to reach the new
  // turn widget — unacceptable for general a11y.
  //
  // Keyed on step_index: fires only when the guided wizard advances to a new
  // step, not on every store mutation that produces a new TurnPayload object
  // with the same step_index.  The ref-null short-circuit handles all non-guided
  // branches implicitly — guidedLogRef.current is null whenever the
  // chat-panel-guided-log div is not mounted (completed surface, freeform
  // surface, no session).  Observation elspeth-obs-5ea21f94af documents the
  // original defect and the chosen Option (c) implementation.
  useEffect(() => {
    if (!guidedLogRef.current) return;
    guidedLogRef.current.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
    const first = guidedLogRef.current.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
    first?.focus();
  }, [guidedNextTurn?.step_index]);

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content);
      // Explicit send means user has returned to live conversation —
      // force-scroll to bottom and resume auto-scroll.
      setShowScrollButton(false);
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    },
    [sendMessage],
  );

  const handleFork = useCallback(
    (messageId: string, newContent: string) => {
      forkFromMessage(messageId, newContent);
    },
    [forkFromMessage],
  );

  const handleUseAsInput = useCallback(
    (blob: BlobMetadata) => {
      // Insert a helper message referencing the blob by filename.
      // The assistant/composer will use blob tools to wire it as source.
      const prompt = `Please use the file "${blob.filename}" as the pipeline input.`;
      sendMessage(prompt);
      setShowBlobManager(false);
    },
    [sendMessage],
  );

  const handleSelectTemplate = useCallback(
    (
      seedPrompt: string,
      recommendedStartingPoint: ExampleUseCase["recommended_starting_point"],
    ) => {
      const applyStartingPoint = (startingPoint: RecommendedStartingPoint) => {
        switch (startingPoint) {
          case "dynamic_source_from_chat":
            setInputText("");
            sendMessage(seedPrompt);
            return;
          case "csv_upload":
            setInputText(seedPrompt);
            setShowBlobManager(true);
            inputRef.current?.focus();
            return;
          case "api_source":
            setInputText(seedPrompt);
            onOpenSecrets?.();
            inputRef.current?.focus();
            return;
          default:
            assertNever(startingPoint);
        }
      };

      applyStartingPoint(recommendedStartingPoint);
    },
    [onOpenSecrets, sendMessage],
  );

  // No active session: show prompt to select or create one
  if (!activeSessionId) {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--empty"
        aria-label="Chat panel"
      >
        Use the session switcher to select a session or create a new one.
      </div>
    );
  }

  // ── Guided-mode discriminator ────────────────────────────────────────────────
  //
  // Precedence (intentional):
  //   1. terminal.kind === "completed"  → CompletionSummary surface.
  //   2. active guided session + non-null next turn  → GuidedTurn surface.
  //   3. anything else (no guidedSession, exited_to_freeform terminal, or a
  //      transient state where guidedSession is set but guidedNextTurn is null)
  //      → fall through to the freeform body below.
  //
  // The completed branch is checked FIRST so that a stale `guidedNextTurn`
  // alongside a completed terminal still surfaces the summary (correct UX)
  // rather than dispatching a widget.
  //
  // When `terminal.kind === "exited_to_freeform"`, branch 1 does not match
  // (kind !== "completed") and branch 2 does not match (`!guidedSession.terminal`
  // is false because `terminal` is set). Execution falls through to the existing
  // freeform body — which is the correct outcome (the user has exited; show
  // them the chat surface).
  //
  // Both branches preserve `id="chat-main"` so the skip-link target is honoured;
  // the modifier class (`--guided` / `--completed`) provides a per-branch hook
  // for future CSS without coupling layout to the freeform surface.
  if (guidedSession?.terminal?.kind === "completed") {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--completed"
        aria-label="Pipeline summary"
      >
        <GuidedWorkflowStepper activeStep="ready" />
        {error && (
          <GuidedErrorBanner error={error} onDismiss={clearError} />
        )}
        <CompletionSummary terminal={guidedSession.terminal} />
        <InlineRunResults />
      </div>
    );
  }

  if (guidedSession && !guidedSession.terminal && guidedNextTurn) {
    return (
      <div
        id="chat-main"
        className="chat-panel chat-panel--guided"
        aria-label="Guided composer"
      >
        <GuidedWorkflowStepper activeStep={guidedSession.step} />
        {error && (
          <GuidedErrorBanner error={error} onDismiss={clearError} />
        )}
        {/*
          aria-live region scope (mirrors the freeform body's
          `<div className="chat-panel-messages">` region below).

          Only the live turn surface (<GuidedTurn>) lives inside the role="log"
          region.  Rationale:

          * GuidedHistory is historical context — already-resolved turns that
            were announced when they first arrived.  Replaying them through the
            live region on every step transition would create redundant SR
            chatter; keep it outside.
          * ExitToFreeformButton is a persistent affordance (always present
            in guided mode).  It is not "new content" on turn change, so it
            also lives outside the log region.
          * GuidedTurn replaces in place when a new step's payload arrives.
            That replacement IS the "new content" event that SRs need to hear
            about — hence the wrapping log region.

          Load-bearing for `InspectAndConfirmTurn.tsx` — search for the
          "Warnings accessibility" comment block (the widget's warnings <aside>
          deliberately omits its own aria-live region under the convention that
          the parent ChatPanel wraps turn content in one).
        */}
        <GuidedHistory history={guidedSession.history} />
        {/*
          Per-step chat log (Phase A slice 6).  Placed ABOVE the wizard
          turn's role="log" region per handover guidance — the user
          reads the chat above their current control surface, and the
          ChatInput at the bottom of the branch is where they reply.
          GuidedChatHistory carries its OWN role="log" + aria-live so
          new chat turns are announced independently of wizard turn
          advances.  Empty-state returns null; no DOM contribution
          before the first chat exchange.
        */}
        <GuidedChatHistory chatHistory={guidedSession.chat_history} />
        <section
          className="guided-current-decision"
          aria-labelledby="guided-current-decision-heading"
        >
          <div className="guided-current-decision-copy">
            <h2 id="guided-current-decision-heading">
              Current decision
            </h2>
            <p>{GUIDED_STEP_PURPOSES[guidedSession.step]}</p>
          </div>
          <div
            ref={guidedLogRef}
            className="chat-panel-guided-log"
            role="log"
            aria-label="Guided wizard step"
            aria-live="polite"
            aria-relevant="additions"
          >
            <GuidedTurn
              turn={guidedNextTurn}
              onSubmit={(body) => void respondGuided(body)}
              disabled={guidedResponsePending}
            />
          </div>
          {guidedResponsePending && (
            <p className="guided-current-decision-pending" role="status">
              Saving decision...
            </p>
          )}
        </section>
        <ExitToFreeformButton />
        {/* Phase 1B inline opt-out: footer-weight affordance to flip the
            account-level default-mode preference from guided→freeform (or
            back). Same backend row as the Settings → Composer pane. */}
        <InlineOptOutCheckbox />
        {/*
          Per-step conversational chat input (Phase A slice 4).

          Lives below the active wizard turn widget so the widget remains the
          primary control surface; chat is a sidecar.  The textarea is its
          own ChatInput instance separate from the freeform composer's
          ChatInput at the bottom of the freeform branch — they have
          independent uncontrolled state.

          `placeholder` is keyed on the live `guidedSession.step` via the
          GUIDED_CHAT_PLACEHOLDERS map (closed list at module top).  The
          per-step skill briefing on the backend already scopes what the
          LLM will engage with; the placeholder text is a UX nudge that
          mirrors the playbook framing.

          `disabled={guidedChatPending}` blocks rapid double-submits while
          a chat round-trip is in flight.  The store's chatGuided action
          flips the flag back on response (or error).
        */}
        <section
          className="guided-step-chat"
          role="region"
          aria-label="Ask about this step"
        >
          <h2 className="guided-step-chat-heading">Ask about this step</h2>
          <ChatInput
            onSend={(content) => void chatGuided(content)}
            disabled={guidedChatPending}
            inputRef={inputRef}
            placeholder={GUIDED_CHAT_PLACEHOLDERS[guidedSession.step]}
          />
        </section>
        <InlineRunResults />
      </div>
    );
  }

  return (
    <div
      id="chat-main"
      className="chat-panel"
      aria-label="Chat panel"
      // data-composing surfaces the "agent is thinking" state to CSS so the
      // textarea and send-button cursors flip to `progress` while the compose
      // request is in-flight. The ComposingIndicator block below is the
      // primary affordance; the cursor change reinforces "system is busy"
      // for users whose pointer is hovering the input area. See
      // components/chat/chat.css [data-composing="true"] rules.
      data-composing={isComposing ? "true" : undefined}
    >
      {/* Session title header.  The "Switch to guided" affordance lives in
          the header so it's always visible without competing with the chat
          input for vertical real-estate.  Symmetric with the "Exit to
          freeform" button rendered by the guided branch above.

          "Change my default" link (Phase 1B Panel UX-M3): the third
          opt-out/opt-in surface spec 05 enumerates. "Switch to guided"
          is a one-session toggle; this link opens the preferences panel
          where the user can change the *future-default* without
          flipping the current session. Rendered only when a handler is
          wired (App.tsx always wires it; tests that mount ChatPanel in
          isolation may omit it). */}
      <div className="chat-panel-header">
        {activeSessionTitle ? (
          <h2 className="chat-panel-header-title">{activeSessionTitle}</h2>
        ) : (
          <span aria-hidden="true" />
        )}
        <div
          className="chat-panel-header-actions"
          style={{ display: "inline-flex", gap: 8, alignItems: "center" }}
        >
          <button
            type="button"
            className="chat-panel-switch-to-guided"
            onClick={() => void enterGuided()}
          >
            Switch to guided
          </button>
          {onOpenComposerPreferences && (
            <button
              type="button"
              onClick={onOpenComposerPreferences}
              className="chat-panel-change-default"
              title="Change which mode new sessions start in"
              style={{
                background: "transparent",
                border: 0,
                padding: "4px 6px",
                font: "inherit",
                fontSize: 12,
                textDecoration: "underline",
                cursor: "pointer",
                minHeight: 24,
                color: "var(--color-text-muted, #555)",
              }}
            >
              Change my default
            </button>
          )}
        </div>
      </div>

      {/* Error banner. Renders the primary error message plus, when
          present, a bulleted list of structured `errorDetails` (currently
          populated from `validation_errors` on a proposal-accept failure).
          Without the bullets the toast collapses Pydantic's flattened
          error string into one unreadable line. */}
      {error && (
        <div role="alert" className="chat-panel-error">
          <div className="chat-panel-error-body">
            <p className="chat-panel-error-message">{error}</p>
            {errorDetails && errorDetails.length > 0 && (
              <ul className="chat-panel-error-details">
                {errorDetails.map((detail, idx) => (
                  <li key={idx}>{detail}</li>
                ))}
              </ul>
            )}
          </div>
          <button
            onClick={clearError}
            className="chat-panel-error-dismiss"
            aria-label="Dismiss error"
          >
            {"\u00D7"}
          </button>
        </div>
      )}

      {/* Message list */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="chat-panel-messages"
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
        aria-relevant="additions"
      >
        {messages.length === 0 ? (
          <TemplateCards onSelectTemplate={handleSelectTemplate} />
        ) : (
          // Render one bubble per *turn*, not one per audit row. The compose
          // loop persists every LLM round-trip as its own assistant row
          // (Tier-1 audit doctrine); grouping projects the audit stream onto
          // user-visible turns so a single user prompt becomes one user
          // bubble + one agent bubble that aggregates every tool call and the
          // final answer. See ./turns.ts.
          //
          // Atomic-reveal gate: agent turns that are mid-flight (only
          // tool-call rows landed, no LLM text reply yet) are hidden from the
          // timeline. The ComposingIndicator (rendered further down while
          // `isComposing` is true) is the visible affordance for "the agent
          // is thinking" — leaking a half-assembled bubble on top of it
          // creates a confusing race between tool calls and the eventual
          // answer. User and system turns are always complete, so the gate
          // is a no-op for them. See turns.ts → ChatTurn.isComplete.
          chatTurns
            .filter((turn: ChatTurn) => turn.isComplete)
            .map((turn: ChatTurn) => {
              const repr = turnRepresentativeMessage(turn);
              // Attach the inline-source summary to the most recent complete
              // agent turn — that's the turn whose audit narrative includes
              // the source-creation event. The store holds at most one
              // summary per session today; passing it as a list keeps the
              // bubble's contract ready for multi-source turns without a
              // future refactor here. When no agent turn is present (e.g.
              // session-restore loaded a composition before any chat), the
              // summary falls through to the standalone widget rendered
              // below the message stream.
              const sourcesForThisTurn =
                inlineSourceSummary && turn.id === inlineSourceTargetTurnId
                  ? [inlineSourceSummary]
                  : undefined;
              return (
                <MessageBubble
                  key={turn.id}
                  message={repr}
                  isComposing={isComposing}
                  onRetry={turn.kind === "user" ? retryMessage : undefined}
                  onFork={turn.kind === "user" ? handleFork : undefined}
                  proposalsByToolCallId={proposalsByToolCallId}
                  staleProposalIds={staleProposalIds}
                  proposalActionPendingIds={proposalActionPendingIds}
                  onAcceptProposal={acceptProposal}
                  onRejectProposal={rejectProposal}
                  sourcesCreated={sourcesForThisTurn}
                  onEditInlineSource={handleEditInlineSource}
                />
              );
            })
        )}
        {/* Standalone fallback for the inline-source summary: only painted
            when no complete agent turn exists to absorb it into the bubble
            (e.g. session-restore where a composition was loaded before any
            chat turn happened). The hybrid keeps the operator's stated UX —
            sources-created appears inside the bubble like tool calls do —
            while not silently dropping the summary in pre-chat states. */}
        {inlineSourceSummary && inlineSourceTargetTurnId === null && (
          <InlineSourceCreatedTurn
            summary={inlineSourceSummary}
            onEdit={handleEditInlineSource}
          />
        )}
        {/*
          Interpretation-review inline messages (Phase 5b Task 5).

          Freeform-mode rendering of pending interpretation events.  One
          message per event, in created_at-ascending order.  Lives inside
          the chat-panel-messages region (role="log") so the messages
          flow naturally with surrounding assistant turns; the inline
          widget brings its own role="region" with a stable accessible
          name so AT users can jump to it independently of the message
          stream.

          The guided-mode counterpart (InterpretationReviewTurn) is
          dispatched by GuidedTurn higher up in the file's guided
          branch; that branch returns early before reaching this
          freeform body.  Both surfaces consume the same
          interpretationEventsStore so a resolution on either side
          updates the other automatically.
        */}
        {pendingInterpretationEvents.map((event) => (
          <InterpretationReviewInlineMessage
            key={event.id}
            event={event}
            sessionId={activeSessionId}
            // Capture user_term BEFORE the widget unmounts so the
            // confirmation line below can show it. The callback fires
            // for both "Use mine" and "Submit amendment" resolves; we do
            // NOT fire it for opt-out (event.user_term is null for
            // opt-out rows — see InterpretationEvent contract) and the
            // handler skips null/empty user_terms.
            onResolved={() => handleInterpretationResolved(event)}
          />
        ))}
        {/*
          Resolve-success confirmation bubbles (Phase 5b.18b.8).

          One assistant-styled bubble per resolved interpretation. Rendered
          inside the role="log" region so the new bubble is announced to
          AT users on append (aria-live="polite" on the parent). The
          bubbles use the same chat-message--assistant styling as ordinary
          assistant turns so the confirmation visually flows with the
          conversation. These are NOT persisted to sessionStore.messages
          and do NOT round-trip to the server — see the
          handleInterpretationResolved comment above for the rationale.
        */}
        {resolveConfirmations.map((conf) => (
          <div
            key={conf.id}
            className="chat-message chat-message--assistant interpretation-review-confirmation"
            data-testid="interpretation-review-confirmation"
            role="status"
          >
            Got it — using your interpretation of{" "}
            <em className="interpretation-review-confirmation-user-term">
              {conf.userTerm}
            </em>
            .
          </div>
        ))}
        {/*
          Inline-source disambiguation widgets (Phase 5a Task 4).

          One widget per pending+non-stale ambiguous-inline proposal
          that survives the F-10 / F-11 re-fire guards (see
          `disambiguationCandidates` derivation above for the
          predicate). Each widget claims its proposal id; the same id
          is excluded from `bannerProposals` so the standard
          PendingProposalsBanner does not duplicate the action
          surface. Non-ambiguous proposals continue to route through
          the banner unchanged.
        */}
        {disambiguationCandidates.map((candidate) => (
          <InlineSourceDisambiguationTurn
            key={candidate.proposal.id}
            userInput={candidate.userInput}
            proposedRows={candidate.proposedRows}
            proposalId={candidate.proposal.id}
            messageId={candidate.messageId}
            onConfirmMultiRow={handleDisambiguationConfirmMultiRow}
            onTreatAsOneRow={handleDisambiguationTreatAsOneRow}
            onEditRows={handleDisambiguationEditRows}
            onNotSourceData={handleDisambiguationNotSourceData}
          />
        ))}
        {shouldShowComposerProgress && (
          <ComposingIndicator
            latestRequest={activeComposerMessage?.content ?? null}
            compositionState={compositionState}
            composerProgress={composerProgress}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll-to-bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
          className="btn scroll-to-bottom-btn"
        >
          {"\u2193"} Jump to latest
        </button>
      )}

      {/* Blob manager drawer */}
      {showBlobManager && <BlobManager onUseAsInput={handleUseAsInput} />}

      <InlineRunResults />

      {/* Pending-proposal banner — surfaces composer proposals that need
          operator approval, co-located with the input so the user does not
          have to scroll up to find the Accept button on the originating
          tool-call message. Component returns null when nothing is pending. */}
      {/* Phase 5a Task 4: `bannerProposals` excludes any proposal
          currently surfaced by an InlineSourceDisambiguationTurn
          widget above so a single proposal does not appear in BOTH
          surfaces. The widget handlers ultimately funnel through
          acceptProposal / rejectProposal so the audit chain is
          identical regardless of which surface lands the action. */}
      <PendingProposalsBanner
        proposals={bannerProposals}
        staleProposalIds={staleProposalIds}
        proposalActionPendingIds={proposalActionPendingIds}
        onAccept={acceptProposal}
        onReject={rejectProposal}
      />

      {/*
        Inline-source fallback prompt (Phase 5a Task 5).

        LLM-skip safety net. Anchored ABOVE the chat input — the user
        reads the affordance immediately before the surface they would
        otherwise re-type into. The widget renders nothing when
        `shouldRenderFallback` is false; mounting unconditionally with
        the boolean gate keeps the DOM stable across predicate flips
        (no remount churn for the focus/scroll containers around it).
      */}
      <InlineSourceFallbackPrompt
        shouldRender={shouldRenderFallback}
        candidateText={fallbackCandidate ?? ""}
        onAccept={handleFallbackAccept}
        onDismiss={handleFallbackDismiss}
      />

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        disabled={isComposing}
        onCancel={isComposing ? cancelComposition : undefined}
        inputRef={inputRef}
        onToggleBlobManager={() => setShowBlobManager((v) => !v)}
        showBlobManager={showBlobManager}
        onOpenSecrets={onOpenSecrets}
        value={inputText}
        onChange={setInputText}
      />
    </div>
  );
}

type WorkflowStepId = GuidedStep | "ready";

const GUIDED_STEP_PURPOSES: Record<GuidedStep, string> = {
  step_1_source: "Choose the input and confirm what ELSPETH can read.",
  step_2_sink: "Choose the output shape and the fields the pipeline should produce.",
  step_2_5_recipe_match: "Review the suggested recipe before ELSPETH builds the transforms.",
  step_3_transforms: "Review the transform chain that turns source data into the output.",
};

const GUIDED_WORKFLOW_STEPS: ReadonlyArray<{
  id: WorkflowStepId;
  label: string;
}> = [
  { id: "step_1_source", label: "Source" },
  { id: "step_2_sink", label: "Output" },
  { id: "step_2_5_recipe_match", label: "Recipe" },
  { id: "step_3_transforms", label: "Transforms" },
  { id: "ready", label: "Ready" },
];

function GuidedWorkflowStepper({ activeStep }: { activeStep: WorkflowStepId }) {
  const activeIndex = GUIDED_WORKFLOW_STEPS.findIndex((step) => step.id === activeStep);
  return (
    <nav className="guided-workflow" aria-label="Guided workflow progress">
      <ol className="guided-workflow-list" aria-label="Guided workflow">
        {GUIDED_WORKFLOW_STEPS.map((step, index) => {
          const state =
            index < activeIndex
              ? "complete"
              : index === activeIndex
                ? "current"
                : "upcoming";
          return (
            <li
              key={step.id}
              className={`guided-workflow-step guided-workflow-step--${state}`}
              aria-current={state === "current" ? "step" : undefined}
            >
              <span className="guided-workflow-index">{index + 1}</span>
              <span className="guided-workflow-label">{step.label}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function GuidedErrorBanner({
  error,
  onDismiss,
}: {
  error: string;
  onDismiss: () => void;
}) {
  return (
    <div role="alert" className="chat-panel-error">
      <span>{error}</span>
      <button
        onClick={onDismiss}
        className="chat-panel-error-dismiss"
        aria-label="Dismiss error"
      >
        {"\u00D7"}
      </button>
    </div>
  );
}

function findActiveComposerMessage(messages: ChatMessage[]): ChatMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user" && message.local_status === "pending") {
      return message;
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") {
      return message;
    }
  }
  return null;
}
