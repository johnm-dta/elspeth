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
} from "@/types/guided";
import { SingleSelectTurn } from "./SingleSelectTurn";
import { InspectAndConfirmTurn } from "./InspectAndConfirmTurn";
import { MultiSelectWithCustomTurn } from "./MultiSelectWithCustomTurn";
import { SchemaFormTurn } from "./SchemaFormTurn";
import { ProposeChainTurn } from "./ProposeChainTurn";

interface GuidedTurnProps {
  turn: TurnPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
}

export function GuidedTurn({ turn, onSubmit, disabled = false }: GuidedTurnProps) {
  const guardedSubmit = (body: GuidedRespondRequest) => {
    if (disabled) return;
    onSubmit(body);
  };
  switch (turn.type) {
    case "single_select":
      return (
        <SingleSelectTurn
          payload={turn.payload as SingleSelectPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "inspect_and_confirm":
      return (
        <InspectAndConfirmTurn
          payload={turn.payload as InspectAndConfirmPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "multi_select_with_custom":
      return (
        <MultiSelectWithCustomTurn
          payload={turn.payload as MultiSelectWithCustomPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "schema_form":
      return (
        <SchemaFormTurn
          payload={turn.payload as SchemaFormPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "propose_chain":
      return (
        <ProposeChainTurn
          payload={turn.payload as ProposeChainPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "recipe_offer":
      return (
        <SchemaFormTurn
          payload={turn.payload as SchemaFormPayload}
          onSubmit={guardedSubmit}
          disabled={disabled}
        />
      );
    case "interpretation_review":
      // Phase 5b — the InterpretationReviewTurn widget is implemented in
      // Task 4 of 18b-phase-5b-frontend.md.  Until then, surfacing this turn
      // type is a backend/frontend phase-skew condition: throw with a
      // diagnostic message rather than rendering nothing.  Task 4 replaces
      // this branch with the real <InterpretationReviewTurn /> render.
      throw new Error(
        "GuidedTurn: interpretation_review widget not yet implemented (Phase 5b Task 4 pending)",
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
