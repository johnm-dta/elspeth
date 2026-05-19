import { TURN_3_PRIMARY_BUTTON } from "./copy";
import type { TutorialBuiltSummary } from "./tutorialMachine";

interface TutorialTurn3GraphProps {
  summary: TutorialBuiltSummary;
  onContinue: () => void;
}

export function TutorialTurn3Graph({
  summary,
  onContinue,
}: TutorialTurn3GraphProps): JSX.Element {
  const stages = [
    {
      label: summary.sourceLabel,
      caption:
        summary.urls.length > 0
          ? `${summary.urls.length} rows`
          : "source rows",
    },
    ...summary.transforms.map((transform, index) => ({
      label: transform,
      caption: index === 0 ? "fetch" : "rate",
    })),
    { label: summary.sinkLabel, caption: "write" },
  ].slice(0, 5);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-graph-title">
      <p className="tutorial-kicker">Graph</p>
      <h2 id="tutorial-graph-title">Here is your pipeline as a graph.</h2>
      <div className="tutorial-graph" aria-label="Tutorial pipeline graph">
        {stages.map((stage, index) => (
          <div className="tutorial-graph-stage" key={`${stage.label}-${index}`}>
            <div className="tutorial-graph-node">{stage.label}</div>
            <span>{stage.caption}</span>
          </div>
        ))}
      </div>
      <p>
        This is the source to transform to sink shape from the welcome screen,
        with one extra transform because the pipeline fetches pages before it
        asks the LLM to rate them.
      </p>
      <div className="tutorial-actions">
        <button type="button" className="btn btn-primary" onClick={onContinue}>
          {TURN_3_PRIMARY_BUTTON}
        </button>
      </div>
    </section>
  );
}
