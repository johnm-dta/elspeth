/**
 * Pipeline template cards shown in the empty chat state.
 *
 * Provides quick-start prompts for common pipeline patterns,
 * reducing the "blank page" problem for new users.
 */

interface PipelineTemplate {
  id: string;
  title: string;
  description: string;
  prompt: string;
  icon: string;
}

const TEMPLATES: PipelineTemplate[] = [
  {
    id: "csv-to-api",
    title: "CSV to API",
    description: "Read CSV, validate data, send to REST API",
    prompt:
      "Create a pipeline that reads a CSV file, validates the data, and sends it to a REST API.",
    icon: "\u{1F4CA}", // bar chart
  },
  {
    id: "llm-classification",
    title: "LLM Classification",
    description: "Classify rows using an LLM, output to CSV",
    prompt:
      "Create a pipeline that reads data, classifies each row using an LLM, and outputs results to CSV.",
    icon: "\u{1F916}", // robot
  },
  {
    id: "data-validation",
    title: "Data Validation",
    description: "Validate against schema, quarantine invalid rows",
    prompt:
      "Create a pipeline that validates CSV data against a schema and quarantines invalid rows.",
    icon: "\u{2705}", // checkmark
  },
  {
    id: "batch-etl",
    title: "Batch ETL",
    description: "Extract, transform, and load between destinations",
    prompt:
      "Create a pipeline that extracts data from a database, transforms it, and loads to another destination.",
    icon: "\u{1F504}", // arrows cycle
  },
  {
    id: "web-scraping",
    title: "Web Scraping",
    description: "Scrape web pages, extract structured data",
    prompt:
      "Create a pipeline that scrapes web pages and extracts structured data.",
    icon: "\u{1F310}", // globe
  },
  {
    id: "file-aggregation",
    title: "File Aggregation",
    description: "Combine multiple files, dedupe, output summary",
    prompt:
      "Create a pipeline that reads multiple CSV files, deduplicates records, and outputs a combined summary.",
    icon: "\u{1F4C1}", // folder
  },
];

interface TemplateCardsProps {
  onSelectTemplate: (prompt: string) => void;
}

export function TemplateCards({ onSelectTemplate }: TemplateCardsProps) {
  return (
    <div className="template-cards-container">
      <div className="template-cards-heading">
        <h2 className="template-cards-title">Welcome to ELSPETH</h2>
        <p className="template-cards-subtitle">
          Choose a template to get started, or describe your own pipeline below.
        </p>
      </div>

      <div
        className="template-cards-grid"
        role="group"
        aria-label="Pipeline templates"
      >
        {TEMPLATES.map((template) => (
          <button
            key={template.id}
            onClick={() => onSelectTemplate(template.prompt)}
            className="template-card"
          >
            <div className="template-card-header">
              <span className="template-card-icon" aria-hidden="true">
                {template.icon}
              </span>
              <span className="template-card-title">{template.title}</span>
            </div>
            <span className="template-card-description">
              {template.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
