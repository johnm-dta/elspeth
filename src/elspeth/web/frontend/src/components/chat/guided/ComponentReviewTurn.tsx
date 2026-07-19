import { useId } from "react";
import type {
  ComponentReviewPayload,
  GuidedComponentAction,
  GuidedRespondAction,
} from "@/types/guided";

interface ComponentReviewTurnProps {
  payload: ComponentReviewPayload;
  onSubmit: (body: GuidedRespondAction) => void;
  disabled?: boolean;
}

function responseFor(component_action: GuidedComponentAction): GuidedRespondAction {
  return {
    chosen: null,
    edited_values: null,
    custom_inputs: null,
    proposal_id: null,
    draft_hash: null,
    edit_target: null,
    control_signal: null,
    component_action,
  };
}

function pluralLabel(kind: ComponentReviewPayload["component_kind"]): string {
  return kind === "source" ? "sources" : "outputs";
}

export function ComponentReviewTurn({
  payload,
  onSubmit,
  disabled = false,
}: ComponentReviewTurnProps) {
  const headingId = useId();
  const stableIds = payload.items.map((item) => item.stable_id);
  if (stableIds.length === 0 || new Set(stableIds).size !== stableIds.length) {
    throw new Error("ComponentReviewTurn requires unique server-authored component identities");
  }
  const allowed = new Set(payload.allowed_actions);
  const plural = pluralLabel(payload.component_kind);

  function submit(componentAction: GuidedComponentAction): void {
    if (disabled) return;
    onSubmit(responseFor(componentAction));
  }

  function move(stableId: string, offset: -1 | 1): void {
    const currentIndex = stableIds.indexOf(stableId);
    const nextIndex = currentIndex + offset;
    if (currentIndex < 0 || nextIndex < 0 || nextIndex >= stableIds.length) return;
    const reordered = [...stableIds];
    [reordered[currentIndex], reordered[nextIndex]] = [
      reordered[nextIndex],
      reordered[currentIndex],
    ];
    submit({
      action: "reorder",
      component_kind: payload.component_kind,
      stable_ids: reordered as [string, ...string[]],
    });
  }

  return (
    <section className="guided-turn guided-component-review" aria-labelledby={headingId}>
      <h3 id={headingId} className="guided-component-review-heading">Review {plural}</h3>
      <ol className="guided-component-review-list">
        {payload.items.map((item) => {
          const currentIndex = stableIds.indexOf(item.stable_id);
          return (
            <li
              key={item.stable_id}
              className="guided-component-review-item"
              aria-label={`${item.name}, ${item.plugin}, reviewed`}
            >
              <div className="guided-component-review-summary">
                <strong>{item.name}</strong>
                <span>{item.plugin}</span>
                <span>{item.status}</span>
              </div>
              <div className="guided-component-review-item-actions">
                {allowed.has("edit") && (
                  <button
                    type="button"
                    className="guided-component-review-btn"
                    onClick={() =>
                      submit({
                        action: "edit",
                        target: { kind: payload.component_kind, stable_id: item.stable_id },
                      })
                    }
                    disabled={disabled}
                  >
                    Edit {item.name}
                  </button>
                )}
                {allowed.has("remove") && payload.items.length > 1 && (
                  <button
                    type="button"
                    className="guided-component-review-btn guided-component-review-btn--remove"
                    onClick={() =>
                      submit({
                        action: "remove",
                        target: { kind: payload.component_kind, stable_id: item.stable_id },
                      })
                    }
                    disabled={disabled}
                  >
                    Remove {item.name}
                  </button>
                )}
                {allowed.has("reorder") && payload.items.length > 1 && (
                  <>
                    <button
                      type="button"
                      className="guided-component-review-btn"
                      aria-label={`Move ${item.name} up`}
                      onClick={() => move(item.stable_id, -1)}
                      disabled={disabled || currentIndex === 0}
                    >
                      Move up
                    </button>
                    <button
                      type="button"
                      className="guided-component-review-btn"
                      aria-label={`Move ${item.name} down`}
                      onClick={() => move(item.stable_id, 1)}
                      disabled={disabled || currentIndex === stableIds.length - 1}
                    >
                      Move down
                    </button>
                  </>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      <div className="guided-component-review-actions">
        {allowed.has("add") && (
          <button
            type="button"
            className="guided-component-review-btn"
            onClick={() => submit({ action: "add", component_kind: payload.component_kind })}
            disabled={disabled}
          >
            Add {payload.component_kind}
          </button>
        )}
        {allowed.has("finish") && (
          <button
            type="button"
            className="guided-component-review-btn guided-component-review-btn--finish"
            onClick={() => submit({ action: "finish", component_kind: payload.component_kind })}
            disabled={disabled}
          >
            Finish {plural}
          </button>
        )}
      </div>
    </section>
  );
}
