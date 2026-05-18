/**
 * Example use case cards shown in the empty chat state.
 *
 * Displays six audit-domain exemplars from templates_data.ts (sourced from
 * README.md §"Example Use Cases"), reducing the "blank page" problem and
 * communicating ELSPETH's audit-domain strengths to new users.
 */

import { TEMPLATES, type ExampleUseCase } from "./templates_data";

interface TemplateCardsProps {
  // aria-label leads with domain so screen-reader users hear the topic first.
  onSelectTemplate: (
    seedPrompt: string,
    recommendedStartingPoint: ExampleUseCase["recommended_starting_point"],
  ) => void;
}

export function TemplateCards({ onSelectTemplate }: TemplateCardsProps) {
  return (
    <div className="template-cards-container">
      <div className="template-cards-heading">
        <h2 className="template-cards-title">Welcome to ELSPETH</h2>
        <p className="template-cards-subtitle">
          ELSPETH builds <strong>auditable</strong> pipelines. Start from a
          domain example below, or describe your own pipeline in the chat.
        </p>
      </div>

      <div
        className="template-cards-grid"
        role="group"
        aria-label="Example use cases"
      >
        {TEMPLATES.map((template) => (
          <button
            key={template.id}
            // aria-label leads with domain so SR users hear the topic first,
            // followed by the one-line description for SDA context.
            aria-label={`${template.domain}: ${template.description}`}
            onClick={() =>
              onSelectTemplate(
                template.seed_prompt,
                template.recommended_starting_point,
              )
            }
            className="template-card"
          >
            <div className="template-card-header">
              <span className="template-card-icon" aria-hidden="true">
                {template.icon}
              </span>
              <span className="template-card-title">{template.domain}</span>
            </div>
            <span className="template-card-description">
              {template.description}
            </span>
            <dl className="template-card-sda">
              <div>
                <dt>Sense</dt>
                <dd>{template.sense}</dd>
              </div>
              <div>
                <dt>Decide</dt>
                <dd>{template.decide}</dd>
              </div>
              <div>
                <dt>Act</dt>
                <dd>{template.act}</dd>
              </div>
            </dl>
          </button>
        ))}
      </div>
    </div>
  );
}
