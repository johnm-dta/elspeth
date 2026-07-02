/**
 * Example use case cards shown in the empty chat state.
 *
 * Displays audit-domain exemplars from templates_data.ts (seeded from
 * README.md §"Example Use Cases"), reducing the "blank page" problem and
 * communicating ELSPETH's audit-domain strengths to new users.
 *
 * Each card carries an explicit "Use this example" action
 * (elspeth-b948756c5a): activating it hands the template's seed prompt and
 * recommended starting point to the parent, which applies the starting
 * point (send the prompt, open the blob manager, or open secrets — see
 * ChatPanel.handleSelectTemplate). The card body itself stays a
 * non-interactive article so the SDA breakdown remains browsable context
 * rather than an ambiguous click target.
 */

import { TEMPLATES, type ExampleUseCase } from "./templates_data";

interface TemplateCardsProps {
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
          ELSPETH builds <strong>auditable</strong> pipelines. Start from one
          of the domain examples below, or describe your own pipeline in the
          chat.
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
            <button
              type="button"
              className="btn btn-small template-card-action"
              // Visible text stays in the accessible name (WCAG 2.5.3); the
              // domain suffix disambiguates the twelve otherwise-identical
              // button names for AT users.
              aria-label={`Use this example: ${template.domain}`}
              onClick={() =>
                onSelectTemplate(
                  template.seed_prompt,
                  template.recommended_starting_point,
                )
              }
            >
              Use this example
            </button>
          </article>
        ))}
      </div>
    </div>
  );
}
