// ============================================================================
// Guided-mode test fixtures — shared helpers for SingleSelectTurn,
// MultiSelectWithCustomTurn, SchemaFormTurn, ProposeChainTurn, RecipeOfferTurn,
// InspectAndConfirmTurn (Tasks 7.2-7.7).
//
// Convention: anything that pins the GuidedRespondRequest wire shape lives
// here, not inside individual widget tests. The whole point of these tests is
// to detect drift in the wire contract — if a 7th GuidedRespondRequest field
// lands one day, only THIS file should need updating, not 6 widget tests.
// ============================================================================

import type { GuidedRespondRequest } from "@/types/guided";

/**
 * The four GuidedRespondRequest fields that none of the chip-group / form
 * widgets ever set (only `chosen` and `custom_inputs` vary across them).
 *
 * Spread into an `expect(...).toEqual({...})` body to assert the unused
 * fields are explicitly null on the wire (not `undefined`, not omitted).
 */
export function nullResponse(): Pick<
  GuidedRespondRequest,
  "edited_values" | "accepted_step_index" | "edit_step_index" | "control_signal"
> {
  return {
    edited_values: null,
    accepted_step_index: null,
    edit_step_index: null,
    control_signal: null,
  };
}
