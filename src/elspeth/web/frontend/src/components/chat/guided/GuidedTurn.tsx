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
//   "propose_pipeline"        -> unavailable until the durable Task 4 adapter
//   "confirm_wiring"          -> WireStageTurn
//
// ============================================================================

import type { TurnPayload, GuidedRespondAction } from "@/types/guided";
import { SingleSelectTurn } from "./SingleSelectTurn";
import { InspectAndConfirmTurn } from "./InspectAndConfirmTurn";
import { MultiSelectWithCustomTurn } from "./MultiSelectWithCustomTurn";
import { SchemaFormTurn } from "./SchemaFormTurn";
import { WireStageTurn, type WireBlockerLink } from "./WireStageTurn";

interface GuidedTurnProps {
  turn: TurnPayload;
  onSubmit: (body: GuidedRespondAction) => void;
  disabled?: boolean;
  /** Tutorial mode — forwarded to leaf widgets that surface worked-example
   * teaching copy (e.g. SchemaFormTurn's on_validation_failure caveat). */
  isTutorial?: boolean;
  /** Pending acknowledgements blocking the confirm_wiring turn — forwarded to
   * WireStageTurn's named-blocker panel (other turn types ignore it; their
   * disabled state is transient in-flight, not an acknowledgement gate). */
  wirePendingAcknowledgements?: WireBlockerLink[];
  /** Client-known validation blockers for confirm_wiring (persisted
   * composition invalid) — forwarded to WireStageTurn. */
  wireValidationIssues?: string[];
}

function guidedTurnInstanceKey(turn: TurnPayload): string {
  return JSON.stringify([turn.step_index, turn.type, turn.payload]);
}

export function GuidedTurn({
  turn,
  onSubmit,
  disabled = false,
  isTutorial = false,
  wirePendingAcknowledgements,
  wireValidationIssues,
}: GuidedTurnProps) {
  const guardedSubmit = (body: GuidedRespondAction) => {
    if (disabled) return;
    onSubmit(body);
  };
  // Stateful leaf widgets initialise local input state from their payload.
  // Key by live turn identity so same-type payload changes remount the leaf
  // instead of carrying stale local form/custom state into the next turn.
  const turnInstanceKey = guidedTurnInstanceKey(turn);
  switch (turn.type) {
    case "single_select":
      // Tutorial is a PASSIVE teaching device: the learner advances by pressing
      // Send (the locked prompt builds the step), never by picking from this
      // menu. The chips are a live, submit-on-click RIVAL driver whose options
      // don't even include the scripted source (the source step builds a
      // web_scrape; the menu lists azure_blob/csv/dataverse/json/text), so
      // clicking ANY chip submits an off-script choice and derails the scripted
      // build. Omit the pick widget in tutorial mode — the decision collapses to
      // its heading + "press Send" caption. Live guided KEEPS the menu (there it
      // is the real path for both audiences).
      if (isTutorial) return null;
      return (
        <SingleSelectTurn
          key={turnInstanceKey}
          payload={turn.payload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "inspect_and_confirm":
      return (
        <InspectAndConfirmTurn
          key={turnInstanceKey}
          payload={turn.payload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "multi_select_with_custom":
      return (
        <MultiSelectWithCustomTurn
          key={turnInstanceKey}
          payload={turn.payload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "schema_form":
      return (
        <SchemaFormTurn
          key={turnInstanceKey}
          payload={turn.payload}
          onSubmit={guardedSubmit}
          disabled={disabled}
          isTutorial={isTutorial}
        />
      );
    case "propose_pipeline":
      throw new Error(
        "GuidedTurn: propose_pipeline rendering is unavailable until durable proposal routing is enabled",
      );
    case "confirm_wiring":
      return (
        <WireStageTurn
          key={turnInstanceKey}
          data={turn.payload}
          confirmDisabled={disabled}
          pendingAcknowledgements={wirePendingAcknowledgements}
          validationIssues={wireValidationIssues}
          onConfirm={() =>
            guardedSubmit({
              chosen: ["confirm"],
              edited_values: null,
              custom_inputs: null,
              proposal_id: null,
              draft_hash: null,
              edit_target: null,
              control_signal: null,
            })
          }
          onAskAdvisor={() =>
            guardedSubmit({
              chosen: null,
              edited_values: null,
              custom_inputs: null,
              proposal_id: null,
              draft_hash: null,
              edit_target: null,
              control_signal: "request_advisor",
            })
          }
          onExitToFreeform={() =>
            guardedSubmit({
              chosen: null,
              edited_values: null,
              custom_inputs: null,
              proposal_id: null,
              draft_hash: null,
              edit_target: null,
              control_signal: "exit_to_freeform",
            })
          }
          onCompleteWithoutSignoff={() =>
            guardedSubmit({
              chosen: ["complete_without_signoff"],
              edited_values: null,
              custom_inputs: null,
              proposal_id: null,
              draft_hash: null,
              edit_target: null,
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
      const _exhaustive: never = turn;
      throw new Error(
        `GuidedTurn: unknown turn: ${String(_exhaustive)} (exhaustiveness check failed)`,
      );
    }
  }
}
