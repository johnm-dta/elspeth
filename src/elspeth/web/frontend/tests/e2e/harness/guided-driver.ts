// Shared selector contracts for the staging guided tutorial driver.

export const ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES: readonly RegExp[] = [
  /^View prompt$/,
  /^Approve the LLM prompt template$/,
  /^Acknowledge/i,
];

export function isAcknowledgementPrimaryActionName(name: string): boolean {
  return ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES.some((pattern) =>
    pattern.test(name),
  );
}
