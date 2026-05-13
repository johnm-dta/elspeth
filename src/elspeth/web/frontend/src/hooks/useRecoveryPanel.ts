import { useCallback, useState } from "react";
import type { ApiError, CompositionState, ComposerRecoveryError } from "@/types/api";
import { isComposerRecoveryError } from "@/types/recovery";

export interface ApplyRecoveryResult {
  applied: boolean;
  needsConfirmation: boolean;
}

interface UseRecoveryPanelOptions {
  currentCompositionVersion: number | null;
  recoveryStartedCompositionVersion: number | null;
  onApplyState: (state: CompositionState) => void;
  onDiscard?: () => void;
}

interface UseRecoveryPanelResult {
  isOpen: boolean;
  recoveryError: ComposerRecoveryError | null;
  openFromError: (apiError: ApiError) => boolean;
  requestApply: () => ApplyRecoveryResult;
  confirmApply: () => boolean;
  cancelApplyConfirmation: () => void;
  discard: () => void;
  needsApplyConfirmation: boolean;
}

const NO_APPLY: ApplyRecoveryResult = {
  applied: false,
  needsConfirmation: false,
};

export function useRecoveryPanel({
  currentCompositionVersion,
  recoveryStartedCompositionVersion,
  onApplyState,
  onDiscard,
}: UseRecoveryPanelOptions): UseRecoveryPanelResult {
  const [recoveryError, setRecoveryError] =
    useState<ComposerRecoveryError | null>(null);
  const [needsApplyConfirmation, setNeedsApplyConfirmation] = useState(false);

  const close = useCallback(() => {
    setRecoveryError(null);
    setNeedsApplyConfirmation(false);
  }, []);

  const openFromError = useCallback((apiError: ApiError): boolean => {
    if (!isComposerRecoveryError(apiError)) {
      return false;
    }
    setRecoveryError(apiError);
    setNeedsApplyConfirmation(false);
    return true;
  }, []);

  const applyNow = useCallback((): boolean => {
    if (recoveryError === null) {
      return false;
    }
    onApplyState(recoveryError.partial_state);
    close();
    return true;
  }, [close, onApplyState, recoveryError]);

  const requestApply = useCallback((): ApplyRecoveryResult => {
    if (recoveryError === null) {
      return NO_APPLY;
    }
    if (currentCompositionVersion !== recoveryStartedCompositionVersion) {
      setNeedsApplyConfirmation(true);
      return { applied: false, needsConfirmation: true };
    }
    applyNow();
    return { applied: true, needsConfirmation: false };
  }, [
    applyNow,
    currentCompositionVersion,
    recoveryError,
    recoveryStartedCompositionVersion,
  ]);

  const confirmApply = useCallback((): boolean => {
    return applyNow();
  }, [applyNow]);

  const cancelApplyConfirmation = useCallback(() => {
    setNeedsApplyConfirmation(false);
  }, []);

  const discard = useCallback(() => {
    close();
    onDiscard?.();
  }, [close, onDiscard]);

  return {
    isOpen: recoveryError !== null,
    recoveryError,
    openFromError,
    requestApply,
    confirmApply,
    cancelApplyConfirmation,
    discard,
    needsApplyConfirmation,
  };
}
