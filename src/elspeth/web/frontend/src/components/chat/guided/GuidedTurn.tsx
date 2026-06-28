// ============================================================================
// GuidedTurn dispatcher
//
// Routes a TurnPayload to its matching leaf widget based on turn.type.
// This component owns no state; it is a pure switch on the wire type.
//
// Routing table:
//   "single_select"           -> SingleSelectTurn
//   "inspect_and_confirm"     -> InspectAndConfirmTurn
//   "multi_select_with_custom"-> MultiSelectWithCustomTurn
//   "schema_form"             -> SchemaFormTurn
//   "propose_chain"           -> ProposeChainTurn
//   "recipe_offer"            -> SchemaFormTurn (mode="recipe_decision")
//   "confirm_wiring"          -> WireStageTurn
//
// Exhaustiveness assertion:
//   The `default:` branch contains `const _exhaustive: never = turn.type`.
//   TypeScript will refuse to compile if a new TurnType is added to
//   guided.ts without a matching case here -- making future omissions a
//   build failure rather than a silent runtime gap.  The throw is also
//   retained for the runtime path (belt-and-suspenders against a JS caller
//   that bypasses type checking).
//
// A/B decision -- Option A (payload: unknown, per-case casts):
//   TurnPayload.payload remains typed as `unknown` in guided.ts.  Each case
//   casts to the appropriate per-type interface (e.g. `as SingleSelectPayload`).
//   Option B (discriminated union) was evaluated but rejected because the
//   existing test fixtures in client.guided.test.ts (line 71) and
//   sessionStore.guided.test.ts (line 54-56) use partial payload literals
//   that do not conform to the full per-type payload shapes (missing required
//   fields such as `question` and `allow_custom` on SingleSelectPayload).
//   Updating those fixtures would be a non-trivial cascade across multiple
//   test files with no correctness gain: `payload: unknown` at the store and
//   API layer is intentional (the dispatcher is the first consumer that needs
//   typed payloads).  Option A limits the cast surface to this one file, where
//   a backend schema mismatch will surface immediately as a runtime render
//   failure rather than silently producing wrong output.
// ============================================================================

import type {
  TurnPayload,
  GuidedRespondRequest,
  SingleSelectPayload,
  InspectAndConfirmPayload,
  MultiSelectWithCustomPayload,
  SchemaFormPayload,
  ProposeChainPayload,
  WireStageData,
} from "@/types/guided";
import type { InterpretationEvent } from "@/types/interpretation";
import { SingleSelectTurn } from "./SingleSelectTurn";
import { InspectAndConfirmTurn } from "./InspectAndConfirmTurn";
import { MultiSelectWithCustomTurn } from "./MultiSelectWithCustomTurn";
import { SchemaFormTurn } from "./SchemaFormTurn";
import { ProposeChainTurn } from "./ProposeChainTurn";
import { InterpretationReviewTurn } from "./InterpretationReviewTurn";
import { WireStageTurn } from "./WireStageTurn";

interface GuidedTurnProps {
  turn: TurnPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
  /** Tutorial mode — forwarded to leaf widgets that surface worked-example
   * teaching copy (e.g. SchemaFormTurn's on_validation_failure caveat). */
  isTutorial?: boolean;
}

function guidedTurnInstanceKey(turn: TurnPayload): string {
  return JSON.stringify([turn.step_index, turn.type, turn.payload]);
}

export function GuidedTurn({ turn, onSubmit, disabled = false, isTutorial = false }: GuidedTurnProps) {
  const guardedSubmit = (body: GuidedRespondRequest) => {
    if (disabled) return;
    onSubmit(body);
  };
  // Stateful leaf widgets initialise local input state from their payload.
  // Key by live turn identity so same-type payload changes remount the leaf
  // instead of carrying stale local form/custom state into the next turn.
  const turnInstanceKey = guidedTurnInstanceKey(turn);
  switch (turn.type) {
    case "single_select":
      return (
        <SingleSelectTurn
          key={turnInstanceKey}
          payload={turn.payload as SingleSelectPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "inspect_and_confirm":
      return (
        <InspectAndConfirmTurn
          key={turnInstanceKey}
          payload={turn.payload as InspectAndConfirmPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "multi_select_with_custom":
      return (
        <MultiSelectWithCustomTurn
          key={turnInstanceKey}
          payload={turn.payload as MultiSelectWithCustomPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "schema_form":
      return (
        <SchemaFormTurn
          key={turnInstanceKey}
          payload={turn.payload as SchemaFormPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "propose_chain":
      return (
        <ProposeChainTurn
          key={turnInstanceKey}
          payload={turn.payload as ProposeChainPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "recipe_offer":
      return (
        <SchemaFormTurn
          key={turnInstanceKey}
          payload={turn.payload as SchemaFormPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "interpretation_review": {
      // Phase 5b Task 4 — the interpretation-review widget owns its own
      // wire submission (POST resolve / POST opt_out), not the guided
      // /respond round-trip the other widget surfaces feed.  The event
      // payload IS the turn payload (the backend includes the
      // InterpretationEvent verbatim so the widget can render without a
      // follow-up GET).  We extract sessionId from event.session_id —
      // it's an authoritative server-emitted Tier-1 field.
      const event = turn.payload as InterpretationEvent;
      return (
        <InterpretationReviewTurn
          key={turnInstanceKey}
          event={event}
          sessionId={event.session_id}
        />
      );
    }
    case "confirm_wiring":
      return (
        <WireStageTurn
          key={turnInstanceKey}
          data={turn.payload as WireStageData}
          confirmDisabled={disabled}
          onConfirm={() =>
            guardedSubmit({
              chosen: ["confirm"],
              edited_values: null,
              custom_inputs: null,
              accepted_step_index: null,
              edit_step_index: null,
              control_signal: null,
            })
          }
        />
      );
    default: {
      // Exhaustiveness check: if a new TurnType is added to guided.ts without
      // a matching case here, TypeScript will report a compile error on this
      // line because `turn.type` narrows to `never` only when all cases are
      // handled.  The throw is retained for the runtime path (JS callers,
      // stale type declarations) per the CLAUDE.md offensive-programming rule.
      const _exhaustive: never = turn.type;
      throw new Error(
        `GuidedTurn: unknown turn type: ${String(_exhaustive)} (exhaustiveness check failed)`,
      );
    }
  }
}
