// ============================================================================
// PipelineGloss — one-line plain-language description of the pipeline (Slice C)
//
// Sits above the guided graph as the canonical "what I built" summary. Pure
// presentation over compositionState via pipelineGloss(); humanising stays in
// the guided wrapper (NOT the shared GraphView, which would bleed into the live
// composer).
// ============================================================================

import type { CompositionState } from "@/types/index";
import { pipelineGloss } from "./pipelineGloss";

interface PipelineGlossProps {
  compositionState: CompositionState | null | undefined;
}

export function PipelineGloss({
  compositionState,
}: PipelineGlossProps): JSX.Element {
  return (
    <p className="pipeline-gloss" data-testid="pipeline-gloss">
      {pipelineGloss(compositionState)}
    </p>
  );
}
