import { useState } from "react";
import { ValidationResultBanner } from "@/components/execution/ValidationResult";
import { useComposer } from "@/hooks/useComposer";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionState, ValidationEntryDTO } from "@/types/index";
import {
  sortedSourceEntries,
  sourceComponentId,
} from "@/utils/compositionState";

interface SuggestionListProps {
  suggestions: ValidationEntryDTO[];
}

function SuggestionList({ suggestions }: SuggestionListProps): JSX.Element {
  const { sendMessage, isComposing } = useComposer();
  const [expanded, setExpanded] = useState(suggestions.length <= 2);

  function handleApply(suggestion: ValidationEntryDTO): void {
    const prompt = `Please apply this suggestion to the pipeline:\n\n**${suggestion.component}:** ${suggestion.message}`;
    void sendMessage(prompt);
  }

  function handleKeyDown(event: React.KeyboardEvent): void {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setExpanded((prev) => !prev);
    }
  }

  return (
    <div className="side-rail-suggestion-banner">
      <div
        className="side-rail-suggestion-header"
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => setExpanded((prev) => !prev)}
        onKeyDown={handleKeyDown}
      >
        <span>Suggestions ({suggestions.length})</span>
        <span className="side-rail-suggestion-chevron">
          {expanded ? "▴" : "▾"}
        </span>
      </div>
      {expanded && (
        <ul className="side-rail-suggestion-list">
          {suggestions.map((s, i) => (
            <li key={i} className="side-rail-suggestion-item">
              <span className="side-rail-suggestion-item-text">
                <strong>{s.component}:</strong> {s.message}
              </span>
              <button
                className="side-rail-suggestion-apply-btn"
                disabled={isComposing}
                onClick={() => handleApply(s)}
              >
                {isComposing ? "Applying..." : "Apply"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function buildValidationComponentNames(
  compositionState: CompositionState | null,
): Record<string, string> {
  if (!compositionState) {
    return {};
  }

  const componentNames: Record<string, string> = {};
  for (const [sourceName] of sortedSourceEntries(compositionState)) {
    componentNames[sourceComponentId(sourceName)] = `source:${sourceName}`;
  }
  for (const node of compositionState.nodes) {
    componentNames[node.id] = `${node.node_type}:${node.id}`;
  }
  for (const output of compositionState.outputs) {
    componentNames[output.name] = `sink:${output.name}`;
  }
  return componentNames;
}

export function SideRailValidationBanner(): JSX.Element | null {
  const compositionState = useSessionStore((s) => s.compositionState);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const error = useExecutionStore((s) => s.error);
  const suggestions = compositionState?.validation_suggestions ?? [];
  const validationComponentNames =
    buildValidationComponentNames(compositionState);

  if (!error && !validationResult && suggestions.length === 0) {
    return null;
  }

  function handleValidationComponentClick(componentId: string): void {
    if (
      Object.prototype.hasOwnProperty.call(
        validationComponentNames,
        componentId,
      )
    ) {
      useSessionStore.getState().selectNode(componentId);
      window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    }
  }

  return (
    <div className="side-rail-validation-banner">
      {error && (
        <div
          role="alert"
          className="validation-banner validation-banner-fail side-rail-error-banner"
        >
          {error}
        </div>
      )}
      {validationResult && (
        <ValidationResultBanner
          result={validationResult}
          nodes={compositionState?.nodes}
          componentNames={validationComponentNames}
          onComponentClick={handleValidationComponentClick}
        />
      )}
      {suggestions.length > 0 && <SuggestionList suggestions={suggestions} />}
    </div>
  );
}
