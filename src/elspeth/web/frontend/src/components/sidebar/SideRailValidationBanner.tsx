import { ValidationResultBanner } from "@/components/execution/ValidationResult";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

export function SideRailValidationBanner(): JSX.Element | null {
  const compositionState = useSessionStore((s) => s.compositionState);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const error = useExecutionStore((s) => s.error);

  if (!error && !validationResult) {
    return null;
  }

  function handleValidationComponentClick(componentId: string): void {
    const isNode =
      compositionState?.nodes.some((node) => node.id === componentId) ?? false;
    if (isNode) {
      useSessionStore.getState().selectNode(componentId);
    }
  }

  return (
    <div className="side-rail-validation-banner">
      {error && (
        <div
          role="alert"
          className="validation-banner validation-banner-fail inspector-error-banner"
        >
          {error}
        </div>
      )}
      {validationResult && (
        <ValidationResultBanner
          result={validationResult}
          nodes={compositionState?.nodes}
          onComponentClick={handleValidationComponentClick}
        />
      )}
    </div>
  );
}
