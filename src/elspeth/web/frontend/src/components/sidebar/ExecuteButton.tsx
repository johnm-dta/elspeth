import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

export function ExecuteButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const progress = useExecutionStore((s) => s.progress);
  const execute = useExecutionStore((s) => s.execute);

  if (!activeSessionId) return null;

  const canExecute =
    validationResult?.is_valid === true &&
    !isExecuting &&
    progress?.status !== "running";

  return (
    <button
      type="button"
      className={`btn side-rail-execute-btn ${canExecute && !isExecuting ? "btn-primary" : ""}`}
      onClick={() => execute(activeSessionId)}
      disabled={!canExecute}
      aria-label="Run pipeline"
    >
      {isExecuting ? (
        <>
          <span
            className="spinner"
            role="status"
            aria-label="Starting pipeline"
          />
          Starting...
        </>
      ) : (
        "Run pipeline"
      )}
    </button>
  );
}
