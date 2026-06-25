import { useEffect, useRef, useState } from "react";
import { getTutorialSample, startGuidedSession } from "@/api/client";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { CANONICAL_TUTORIAL_PROMPT } from "./tutorialMachine";
import type {
  GuidedRespondRequest,
  GuidedStep,
  InspectAndConfirmPayload,
  MultiSelectWithCustomPayload,
  SchemaFormPayload,
  SingleSelectPayload,
  TurnPayload,
} from "@/types/guided";

interface TutorialGuidedShellProps {
  sessionId: string;
  onCompleted: (sessionId: string) => void;
}

// Backstop against a wizard that never reaches a terminal (a rejected body, a
// malformed turn). Each real phase needs at most ~2 confirms; 24 is generous
// headroom over the source→sink→recipe→transform→wire path so a stuck drive
// halts rather than spinning.
const MAX_AUTODRIVE_STEPS = 24;

/**
 * Tutorial bridge (D9) + PASSIVE auto-drive (p4 Task 8b): renders the welcome
 * bookend, starts a TUTORIAL-profile guided session, EMBEDS the real ChatPanel
 * guided surface (the truest "use the real thing"), and on guided
 * terminal=completed hands the session back to the surviving tutorialMachine
 * run/audit/graduation tail.
 *
 * The headline conceit (spec way 1): the learner specifies NOTHING and WATCHES
 * the LLM build the pipeline. After the guided session starts, `autoDriveTutorial`
 * drives the wizard to completion on its own — it fetches the 8a GET surface for
 * the runtime-resolved synthetic URLs, seeds the FRONTEND scripted intent
 * (`CANONICAL_TUTORIAL_PROMPT`) via per-phase `chatGuided` (URLs appended at the
 * source phase only — the canonical prompt already carries the sink intent), and
 * then drives the explicit per-phase confirms via `respondGuided` through
 * STEP_1 → STEP_2 → STEP_2.5 → STEP_3 → STEP_4 under p1's no-auto-advance
 * contract. The STEP_2.5 recipe accept (`chosen=["accept"]`) is the seam where
 * 8a injects the SSRF `allowed_hosts` server-side — the shell never sends
 * `allowed_hosts` over the wire. `entry_seed` likewise stays server-side only
 * (NOT a field on the TS WorkflowProfile — P6.4 security carry-note); the
 * scripted intent is the byte-stable frontend constant.
 *
 * Per-stage interpretation reviews (D12, advisory/non-blocking) are surfaced as
 * `interpretationEventsStore.pendingBySession` cards that GATE the confirm; the
 * auto-drive resolves them (accepted_as_drafted) before each confirm so the
 * passive walk is never blocked — the same gate ChatPanel's guided branch
 * projects (P4.T2). Rendering stays PASSIVE: ChatPanel renders the live wizard,
 * the learner clicks nothing; affordance suppression in tutorial mode is owned
 * by p2's `isTutorial` prop. Coaching/bookend copy reads off the wire
 * GuidedSession.profile; the welcome framing text comes from the frontend copy.ts.
 */
export function TutorialGuidedShell({
  sessionId,
  onCompleted,
}: TutorialGuidedShellProps): JSX.Element {
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const startGuided = useSessionStore((s) => s.startGuided);
  const startedRef = useRef(false);
  const completedRef = useRef(false);
  // True once this mount has OBSERVED a live (non-null, not-yet-completed)
  // guidedSession. `onCompleted` may fire only for a completion this shell saw
  // transition to while mounted — never when it mounts directly onto an
  // already-completed session.
  const sawActiveRef = useRef(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Start the TUTORIAL-profile guided session exactly once. The start
  // endpoint is idempotent server-side (P7.1): a second POST for a session
  // that already has a persisted GuidedSession returns it unchanged. The
  // startedRef guard avoids a redundant round-trip under StrictMode's
  // double-invoke.
  useEffect(() => {
    if (startedRef.current) {
      return;
    }
    startedRef.current = true;
    void (async () => {
      setStarting(true);
      setError(null);
      // Bind the store's activeSessionId to this tutorial session BEFORE
      // startGuided. startGuided (sessionStore.ts) DISCARDS its fetched guided
      // payload unless get().activeSessionId === the requested id, and ChatPanel
      // renders the empty-session surface (chat-panel--empty) whenever
      // activeSessionId is null. Clear the same session/guided payload that
      // selectSession clears before loading (mirrors selectSession +
      // cleared{Guided,Recovery}State), otherwise a completed guided session
      // from the previous active session can make ChatPanel render the completed
      // surface and fire onCompleted before the new tutorial session has loaded.
      useSessionStore.setState({
        activeSessionId: sessionId,
        messages: [],
        compositionState: null,
        compositionProposals: [],
        composerPreferences: null,
        staleProposalIds: [],
        proposalActionPendingIds: [],
        composerProgress: null,
        stateVersions: [],
        isComposing: false,
        error: null,
        selectedNodeId: null,
        guidedSession: null,
        guidedNextTurn: null,
        guidedTerminal: null,
        guidedChatPending: false,
        guidedResponsePending: false,
        recoveryError: null,
        recoveryStartedCompositionVersion: null,
      });
      let started = false;
      try {
        await startGuidedSession(sessionId, "tutorial");
        await startGuided(sessionId);
        started = true;
      } catch (err) {
        setError(formatError(err));
      } finally {
        setStarting(false);
      }
      // Passive auto-drive (Task 8b): once the guided session is live, drive the
      // wizard to completion on its own. Fire-and-forget so ChatPanel renders the
      // live wizard while the walker advances it — the learner watches, clicks
      // nothing. A drive failure surfaces as a banner but does not crash the shell.
      if (started) {
        void autoDriveTutorial(sessionId).catch((err) => {
          setError(formatError(err));
        });
      }
    })();
  }, [sessionId, startGuided]);

  // Hand off to the run/audit/graduation tail when guided reaches completion —
  // but ONLY on a completion this mount OBSERVED transition to. The back-nav
  // GET path remounts this shell against the PERSISTED completed guided session
  // (startGuided clears guidedSession to null, then sets it to the completed
  // payload), so the shell mounts onto `terminal=completed` without ever seeing
  // a live wizard. Firing onCompleted there bounces the user straight back to
  // run (no-op flash from run-Back; guided skipped from audit-Back). Gate on
  // sawActiveRef: a completed session that was never preceded by a live,
  // not-yet-completed session during this mount must NOT graduate. Note the
  // `terminal.kind === "completed"` guard also (deliberately) excludes
  // `exited_to_freeform` — leaving the wizard for freeform is not a graduation.
  useEffect(() => {
    if (completedRef.current) {
      return;
    }
    const current = useSessionStore.getState();
    if (
      current.activeSessionId !== sessionId ||
      current.guidedSession !== guidedSession
    ) {
      return;
    }
    const kind = guidedSession?.terminal?.kind;
    // Record that we observed a live wizard: a non-null session that has not
    // yet completed. The mount-effect's clear-to-null step leaves guidedSession
    // null (not "active"), so requiring non-null here keeps the back-nav path
    // from spuriously marking the wizard observed.
    if (guidedSession !== undefined && guidedSession !== null && kind !== "completed") {
      sawActiveRef.current = true;
    }
    if (kind === "completed" && sawActiveRef.current) {
      completedRef.current = true;
      onCompleted(sessionId);
    }
  }, [guidedSession, onCompleted, sessionId]);

  const bookends = guidedSession?.profile?.bookends ?? true;

  return (
    <section
      className="tutorial-guided-shell"
      aria-label="Guided pipeline composer"
    >
      {bookends && (
        <p className="tutorial-kicker">
          Let's build your first pipeline one stage at a time.
        </p>
      )}
      <p role="status" className="sr-only">
        {starting ? "Starting guided composer" : ""}
      </p>
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
      )}
      <ChatPanel isTutorial />
    </section>
  );
}

function formatError(err: unknown): string {
  if (
    typeof err === "object" &&
    err !== null &&
    "detail" in err &&
    typeof (err as { detail?: unknown }).detail === "string"
  ) {
    return (err as { detail: string }).detail;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return "The guided tutorial could not be started.";
}

/**
 * The passive auto-confirm walker (Task 8b, mechanism B — per-phase chat).
 *
 * A GENERIC state-driven loop (mirrors the harness `driveGuidedWalk`, but drives
 * the store actions instead of DOM clicks — the harness phase-walker is test-only
 * and not reusable at runtime). Each iteration reads the LIVE wizard state, then:
 *
 *   1. resolves any pending D12 interpretation-review cards (advisory; they gate
 *      the confirm) via accepted_as_drafted — so the passive walk is never blocked;
 *   2. at the SOURCE / SINK phases, authors the phase via ONE `chatGuided` seeding
 *      the canonical prompt (URLs appended at the source phase only); the LLM-primary
 *      per-phase driver extracts its phase's intent (p1's chat-apply);
 *   3. otherwise confirms the presented turn via `respondGuided` to advance —
 *      STEP_2.5 recipe accept (`chosen=["accept"]`) inserts web_scrape+llm (8a
 *      injects allowed_hosts there), STEP_3 chain accept, STEP_4 wire confirm.
 *
 * Reads the LIVE turn every iteration (no stale optimistic-concurrency token), and
 * halts on a wire error, a non-advancing turn, or the step cap rather than spinning.
 * Returns when the wizard reaches a terminal (completion graduation is owned by the
 * separate effect that observes the live→completed transition).
 */
async function autoDriveTutorial(sessionId: string): Promise<void> {
  const initial = useSessionStore.getState().guidedSession;
  // Not a live wizard (back-nav mounted onto a completed/persisted session, or a
  // start that produced no live turn) → nothing to drive. Bailing BEFORE the GET
  // surface fetch keeps the non-driving paths network-free.
  if (initial === null || initial === undefined || initial.terminal !== null) {
    return;
  }

  const sample = await getTutorialSample(sessionId);
  const urls = sample.sample_urls;
  const sourceIntent =
    urls.length > 0 ? `${CANONICAL_TUTORIAL_PROMPT}\n${urls.join("\n")}` : CANONICAL_TUTORIAL_PROMPT;

  const chatted = new Set<GuidedStep>();

  for (let i = 0; i < MAX_AUTODRIVE_STEPS; i += 1) {
    const store = useSessionStore.getState();
    if (store.error !== null) {
      return; // a wire failure halts the passive drive
    }
    const guided = store.guidedSession;
    if (guided === null || guided === undefined || guided.terminal !== null) {
      return; // completed / exited / lost
    }
    const step = guided.step;

    // (1) Resolve pending D12 interpretation reviews (the confirm gate). Advisory
    // and non-blocking: accept the LLM's draft so the walk advances. Mirrors the
    // harness `resolveVisibleReviews`.
    const interp = useInterpretationEventsStore.getState();
    const pending = interp.pendingBySession[sessionId];
    if (pending !== undefined) {
      const pendingIds = Object.keys(pending);
      if (pendingIds.length > 0) {
        for (const eventId of pendingIds) {
          await interp.resolveEvent(sessionId, eventId, { choice: "accepted_as_drafted" });
        }
        continue;
      }
    }

    // (2) SOURCE / SINK phases: author via one per-phase chat (mechanism B). The
    // canonical prompt carries both the source intent (URLs) and the sink intent
    // ("write the rows to a json file"); each phase's driver extracts its part.
    // URLs ride only the SOURCE chat.
    if ((step === "step_1_source" || step === "step_2_sink") && !chatted.has(step)) {
      chatted.add(step);
      await store.chatGuided(step === "step_1_source" ? sourceIntent : CANONICAL_TUTORIAL_PROMPT);
      continue;
    }

    // (3) Confirm the presented turn to advance.
    const nextTurn = store.guidedNextTurn;
    if (nextTurn === null || nextTurn === undefined) {
      return; // nothing to confirm and not an unauthored chat phase
    }
    const signatureBefore = `${step}|${nextTurn.type}`;
    await store.respondGuided(buildConfirmBody(nextTurn));

    // No-progress guard: if the same (step, turn) is still presented and no new
    // review card appeared, the drive cannot advance — halt rather than spin.
    const after = useSessionStore.getState();
    const afterGuided = after.guidedSession;
    const afterPending = useInterpretationEventsStore.getState().pendingBySession[sessionId];
    const afterHasPending = afterPending !== undefined && Object.keys(afterPending).length > 0;
    if (
      !afterHasPending &&
      afterGuided !== null &&
      afterGuided !== undefined &&
      afterGuided.terminal === null &&
      `${afterGuided.step}|${after.guidedNextTurn?.type ?? ""}` === signatureBefore
    ) {
      return;
    }
  }
}

/**
 * Build the canonical "advance" body for a presented guided turn — byte-matching
 * the leaf widget's own submit handler so the passive walk sends exactly what the
 * manual form would. The STEP_2.5 recipe accept carries `chosen=["accept"]` +
 * the recipe slots (the seam 8a server-side-injects `allowed_hosts` into).
 */
function buildConfirmBody(turn: TurnPayload): GuidedRespondRequest {
  const base: GuidedRespondRequest = {
    chosen: null,
    edited_values: null,
    custom_inputs: null,
    accepted_step_index: null,
    edit_step_index: null,
    control_signal: null,
  };
  switch (turn.type) {
    case "inspect_and_confirm": {
      const payload = turn.payload as InspectAndConfirmPayload;
      return { ...base, edited_values: { columns: payload.observed.columns } };
    }
    case "schema_form": {
      const payload = turn.payload as SchemaFormPayload;
      if (payload.mode === "recipe_decision") {
        return {
          ...base,
          chosen: ["accept"],
          edited_values: {
            recipe_name: payload.recipe_context.recipe_name,
            slots: recipeAcceptSlots(payload),
          },
        };
      }
      // plugin_options: accept the source/sink form as authored by the chat-apply.
      return {
        ...base,
        edited_values: {
          plugin: payload.plugin,
          options: { ...payload.prefilled },
          observed_columns: [],
          sample_rows: [],
        },
      };
    }
    case "recipe_offer": {
      // Dispatched to the recipe_decision SchemaFormTurn (GuidedTurn routing).
      const payload = turn.payload as Extract<SchemaFormPayload, { mode: "recipe_decision" }>;
      return {
        ...base,
        chosen: ["accept"],
        edited_values: {
          recipe_name: payload.recipe_context.recipe_name,
          slots: recipeAcceptSlots(payload),
        },
      };
    }
    case "multi_select_with_custom": {
      const payload = turn.payload as MultiSelectWithCustomPayload;
      // Accept the offered default field set (the widget's confirm path); the
      // recipe overrides the pipeline shape at STEP_2.5 regardless.
      return { ...base, chosen: [...payload.default_chosen], custom_inputs: [] };
    }
    case "propose_chain":
      // Accept the recipe-provided transform chain (Phase-4 accept-all path).
      return { ...base, chosen: ["accept"] };
    case "confirm_wiring":
      return { ...base, chosen: ["confirm"] };
    case "single_select": {
      // Not reached on the tutorial path (the SOURCE/SINK initial single_select
      // turns are authored via chat, not confirmed). Defensive: pick the first
      // offered option so a stray single_select still advances.
      const payload = turn.payload as SingleSelectPayload;
      return { ...base, chosen: payload.options.length > 0 ? [payload.options[0].id] : [] };
    }
    case "interpretation_review":
      // D12 reviews are resolved as pending cards before the confirm (see the
      // walker), never via respondGuided — and the backend never emits this as a
      // guided next_turn. Reaching here is an invariant violation.
      throw new Error("auto-drive: interpretation_review is not a respond turn");
    default: {
      const _exhaustive: never = turn.type;
      throw new Error(`auto-drive: unhandled turn type ${String(_exhaustive)}`);
    }
  }
}

/**
 * The recipe-accept slots: the server-prefilled slots plus any knob defaults —
 * mirrors SchemaFormTurn's `initialValues` for the recipe_decision form, i.e.
 * "accept the offered recipe form as-is". Unsatisfied slots without a prefill or
 * default are left for the recipe offer to supply server-side (the passive walk
 * authors no slot values of its own).
 */
function recipeAcceptSlots(
  payload: Extract<SchemaFormPayload, { mode: "recipe_decision" }>,
): Record<string, unknown> {
  const slots: Record<string, unknown> = { ...payload.prefilled };
  for (const field of payload.knobs.fields) {
    if (!Object.prototype.hasOwnProperty.call(slots, field.name) && field.default !== undefined) {
      slots[field.name] = field.default;
    }
  }
  return slots;
}
