// ============================================================================
// Guided-mode test fixtures — shared helpers for SingleSelectTurn,
// MultiSelectWithCustomTurn, SchemaFormTurn, ProposeChainTurn,
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
 *
 * USAGE: spread FIRST in the toEqual literal so explicitly-set fields override.
 *
 *   // CORRECT — explicit edited_values wins over nullResponse's edited_values=null
 *   expect(body).toEqual({
 *     ...nullResponse(),                  // <-- first
 *     chosen: null,
 *     custom_inputs: null,
 *     edited_values: { columns, samples, warnings },  // <-- overrides
 *   });
 *
 *   // WRONG — spreading last re-nulls edited_values; the test silently asserts null
 *   expect(body).toEqual({
 *     chosen: null,
 *     edited_values: { columns, samples, warnings },
 *     ...nullResponse(),                  // <-- clobbers edited_values
 *   });
 *
 * InspectAndConfirmTurn (Task 7.3) sets `edited_values` on both submit paths
 * and tripped this exact bug during initial development. Widgets in 7.4-7.7
 * that set fields beyond `chosen` / `custom_inputs` MUST observe the same
 * spread order.
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
