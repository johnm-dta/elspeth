// ============================================================================
// ValidationResult Banner
//
// Inline banner displayed between the inspector header and tab content.
// Renders Stage 2 validation results with per-component attribution.
//
// Pass: green banner with checkmark, summary, and check details.
// Fail: red banner with per-component error list, component_id mapped to
// display name from CompositionState, and suggested fixes from backend.
//
// The Execute button enables/disables based on this result.
// ============================================================================

import type {
  ValidationResult as ValidationResultType,
  ValidationWarning,
  NodeSpec,
} from "@/types/index";

interface ValidationResultProps {
  result: ValidationResultType;
  /** Nodes from CompositionState for mapping component_id to display name */
  nodes?: NodeSpec[];
  /** All graph components from CompositionState keyed by navigable component_id */
  componentNames?: Record<string, string>;
  /** Callback when user clicks an error/warning to navigate to that component */
  onComponentClick?: (componentId: string) => void;
}

/**
 * Resolve a component_id to a human-readable display name.
 * Falls back to the raw component_id if no matching node is found.
 */
function resolveComponentName(
  componentId: string | null,
  nodes: NodeSpec[] | undefined,
  componentNames: Record<string, string> | undefined,
): string {
  if (!componentId) return "unknown";
  if (
    componentNames &&
    Object.prototype.hasOwnProperty.call(componentNames, componentId)
  ) {
    return componentNames[componentId];
  }
  if (!nodes) return componentId;
  const node = nodes.find((n) => n.id === componentId);
  return node ? `${node.node_type}:${node.id}` : componentId;
}

function isNavigableComponent(
  componentId: string | null,
  nodes: NodeSpec[] | undefined,
  componentNames: Record<string, string> | undefined,
): componentId is string {
  if (!componentId) return false;
  if (
    componentNames &&
    Object.prototype.hasOwnProperty.call(componentNames, componentId)
  ) {
    return true;
  }
  return nodes?.some((n) => n.id === componentId) ?? false;
}

export function ValidationResultBanner({
  result,
  nodes,
  componentNames,
  onComponentClick,
}: ValidationResultProps) {
  if (result.is_valid) {
    return (
      <div
        role="status"
        className="validation-banner validation-banner-pass validation-banner-content"
      >
        <div className="validation-banner-header">
          <span aria-hidden="true">{"\u2713"}</span>
          <span className="validation-banner-summary">
            {result.summary ?? "Validation passed"}
          </span>
        </div>
        {result.checks.length > 0 && (
          <ul className="validation-banner-checks">
            {result.checks.map((check, i) => (
              <li key={i} className="validation-banner-check-item">
                <span aria-hidden="true">
                  {check.passed ? "\u2713" : "\u2717"}
                </span>{" "}
                {check.name}: {check.detail}
              </li>
            ))}
          </ul>
        )}
        {result.warnings && result.warnings.length > 0 && (
          <div className="validation-banner-warnings-section">
            <div className="validation-banner-warnings-title">
              Warnings ({result.warnings.length}):
            </div>
            <ul className="validation-banner-warnings-list">
              {result.warnings.map((warn: ValidationWarning, i: number) => {
                const isClickable =
                  Boolean(onComponentClick) &&
                  isNavigableComponent(warn.component_id, nodes, componentNames);
                const content = (
                  <>
                    <strong>
                      [{warn.component_type ?? "unknown"}]{" "}
                      {resolveComponentName(
                        warn.component_id,
                        nodes,
                        componentNames,
                      )}:
                    </strong>{" "}
                    {warn.message}
                    {warn.suggestion && (
                      <div className="validation-banner-suggestion">
                        Suggestion: {warn.suggestion}
                      </div>
                    )}
                  </>
                );

                return (
                  <li key={i} className="validation-banner-warn-item">
                    {isClickable ? (
                      <button
                        onClick={() => {
                          if (warn.component_id) {
                            onComponentClick?.(warn.component_id);
                          }
                        }}
                        className="validation-banner-component-btn validation-banner-component-btn--warning"
                        title={`Click to select ${warn.component_id} in the pipeline view`}
                      >
                        {content}
                      </button>
                    ) : (
                      content
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      role="alert"
      className="validation-banner validation-banner-fail"
    >
      <div className="validation-banner-fail-title">
        Validation failed
      </div>
      <ul className="validation-banner-fail-list">
        {result.errors.map((err, i) => {
          const isClickable =
            Boolean(onComponentClick) &&
            isNavigableComponent(err.component_id, nodes, componentNames);
          const content = (
            <>
              <strong>
                [{err.component_type ?? "unknown"}]{" "}
                {resolveComponentName(err.component_id, nodes, componentNames)}:
              </strong>{" "}
              {err.message}
              {err.suggestion && (
                <div className="validation-banner-suggestion">
                  Suggestion: {err.suggestion}
                </div>
              )}
            </>
          );

          return (
            <li key={i} className="validation-banner-error-item">
              {isClickable ? (
                <button
                  onClick={() => {
                    if (err.component_id) {
                      onComponentClick?.(err.component_id);
                    }
                  }}
                  className="validation-banner-component-btn validation-banner-component-btn--error"
                  title={`Click to select ${err.component_id} in the pipeline view`}
                >
                  {content}
                </button>
              ) : (
                content
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
