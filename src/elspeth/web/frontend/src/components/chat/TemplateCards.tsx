/**
 * Example use case cards shown in the empty chat state.
 *
 * Displays audit-domain exemplars from templates_data.ts (seeded from
 * README.md §"Example Use Cases"), reducing the "blank page" problem and
 * communicating ELSPETH's audit-domain strengths to new users.
 */

import { TEMPLATES, type ExampleUseCase } from "./templates_data";

interface TemplateCardsProps {
  // Reserved for the future user-storable favourites flow. The current
  // example tiles are intentionally static: they establish visual context
  // without implying that these exact demo pipelines are one-click starters.
  onSelectTemplate: (
    seedPrompt: string,
    recommendedStartingPoint: ExampleUseCase["recommended_starting_point"],
  ) => void;
}

export function TemplateCards({ onSelectTemplate }: TemplateCardsProps) {
  // TODO(user-storable-favourites): Re-enable tile activation only when this
  // surface can show operator-saved favourites rather than generic examples.
  void onSelectTemplate;

  return (
    <div className="template-cards-container">
      <div className="template-cards-heading">
        <h2 className="template-cards-title">Welcome to ELSPETH</h2>
        <p className="template-cards-subtitle">
          ELSPETH builds <strong>auditable</strong> pipelines. Consider one of
          the domain examples below, or describe your own pipeline in the chat.
        </p>
      </div>

      <div
        className="template-cards-grid"
        role="group"
        aria-label="Example use cases"
      >
        {TEMPLATES.map((template) => (
          <article
            key={template.id}
            // aria-label leads with domain so SR users hear the topic first,
            // followed by the one-line description for SDA context.
            aria-label={`${template.domain}: ${template.description}`}
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
          </article>
        ))}
      </div>
    </div>
  );
}
