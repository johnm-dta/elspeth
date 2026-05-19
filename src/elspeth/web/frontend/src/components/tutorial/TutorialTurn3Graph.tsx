import { Fragment, useEffect, useRef } from "react";
import { TURN_3_PRIMARY_BUTTON } from "./copy";
import type { TutorialBuiltSummary } from "./tutorialMachine";

interface TutorialTurn3GraphProps {
  summary: TutorialBuiltSummary;
  onContinue: () => void;
  onBack: () => void;
}

export function TutorialTurn3Graph({
  summary,
  onContinue,
  onBack,
}: TutorialTurn3GraphProps): JSX.Element {
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

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
      <h2 id="tutorial-graph-title" ref={headingRef} tabIndex={-1}>
        Here is your pipeline as a graph.
      </h2>
      <ol className="tutorial-graph" aria-label="Pipeline stages">
        {stages.map((stage, index) => {
          const isLast = index === stages.length - 1;
          return (
            <Fragment key={`${stage.label}-${index}`}>
              <li className="tutorial-graph-stage">
                <div className="tutorial-graph-node">{stage.label}</div>
                <span>{stage.caption}</span>
              </li>
              {!isLast && (
                <span
                  className="tutorial-graph-chevron"
                  aria-hidden="true"
                >
                  {/* Chevron is decoration; the <ol> conveys the order
                      semantics to assistive tech. Inline SVG so forced-colors
                      mode can recolour via currentColor. */}
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 18 18"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="6 4 12 9 6 14" />
                  </svg>
                </span>
              )}
            </Fragment>
          );
        })}
      </ol>
      <p>
        This is the source to transform to sink shape from the welcome screen,
        with one extra transform because the pipeline fetches pages before it
        asks the LLM to rate them.
      </p>
      <div className="tutorial-actions">
        <button type="button" className="btn btn-primary" onClick={onContinue}>
          {TURN_3_PRIMARY_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-link-button"
          onClick={onBack}
        >
          Back
        </button>
      </div>
    </section>
  );
}
