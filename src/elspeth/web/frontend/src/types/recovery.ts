import type { ChatMessage, CompositionState } from "./index";

export interface FailedTurn {
  assistant_message_id: string | null;
  tool_calls_attempted: number;
  tool_responses_persisted: number;
  transcript_url: string | null;
}

export interface ComposerRecoveryErrorFields {
  partial_state?: CompositionState | null;
  failed_turn?: FailedTurn | null;
  partial_state_save_failed?: boolean;
  partial_state_save_error?: string | null;
}

export type RecoveryTranscriptRow = ChatMessage & {
  role: "assistant" | "tool";
};
