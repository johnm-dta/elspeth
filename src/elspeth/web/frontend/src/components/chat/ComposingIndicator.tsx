// src/components/chat/ComposingIndicator.tsx

import type { ComposerProgressSnapshot, CompositionState } from "@/types/api";

interface ComposingIndicatorProps {
  latestRequest?: string | null;
  compositionState?: CompositionState | null;
  composerProgress?: ComposerProgressSnapshot | null;
}

interface RequestFocus {
  headline: string;
  focus: string;
  nextMove: string;
}

interface WorkingView {
  headline: string;
  evidence: string[];
  likelyNext: string;
}

function plural(count: number, singular: string, pluralLabel = `${singular}s`): string {
  return count === 1 ? `1 ${singular}` : `${count} ${pluralLabel}`;
}

function setupCount(count: number, singular: string, pluralLabel = `${singular}s`): string {
  if (count === 0) {
    return `no ${pluralLabel}`;
  }
  return plural(count, singular, pluralLabel);
}

function describeCurrentSetup(compositionState: CompositionState | null | undefined): string {
  const input = compositionState?.source ? "input configured" : "no input yet";
  const steps = setupCount(compositionState?.nodes.length ?? 0, "processing step");
  const outputs = setupCount(compositionState?.outputs.length ?? 0, "output");
  return `Current setup: ${input}, ${steps}, ${outputs}.`;
}

function describeRequestFocus(latestRequest: string | null | undefined): RequestFocus {
  const normalized = latestRequest?.toLocaleLowerCase() ?? "";

  if (normalized.includes("html") && normalized.includes("json")) {
    return {
      headline: "Working on: convert HTML into JSON",
      focus: "Request focus: turn HTML content into structured JSON.",
      nextMove: "Likely next move: choose an input, extract the useful fields, then save structured JSON.",
    };
  }

  if (/\b(database|sql|table|query)\b/.test(normalized)) {
    return {
      headline: "Working on: database-backed data flow",
      focus: "Request focus: read data from a database source.",
      nextMove: "Likely next move: identify the input query, shape the records, then send them to an output.",
    };
  }

  if (/\b(scrape|website|web page|url|fetch)\b/.test(normalized)) {
    return {
      headline: "Working on: web content pipeline",
      focus: "Request focus: fetch or parse web content.",
      nextMove: "Likely next move: choose a web input, extract the useful content, then structure the result.",
    };
  }

  if (/\b(output|save|export|write|artifact)\b/.test(normalized)) {
    return {
      headline: "Working on: saved output",
      focus: "Request focus: produce or update saved output.",
      nextMove: "Likely next move: check the current pipeline shape and wire the final output.",
    };
  }

  if (/\b(file|csv|excel|upload|input)\b/.test(normalized)) {
    return {
      headline: "Working on: file input pipeline",
      focus: "Request focus: use a supplied file as input.",
      nextMove: "Likely next move: connect the file, inspect its fields, then add the needed processing steps.",
    };
  }

  return {
    headline: "Working through your request",
    focus: "Request focus: update the pipeline from your latest message.",
    nextMove: "Likely next move: compare your request with the current setup, then update the graph or explain what is missing.",
  };
}

function backendWorkingView(
  composerProgress: ComposerProgressSnapshot | null | undefined,
): WorkingView | null {
  if (!composerProgress || composerProgress.phase === "idle") {
    return null;
  }

  return {
    headline: composerProgress.headline,
    evidence:
      composerProgress.evidence.length > 0
        ? composerProgress.evidence
        : ["ELSPETH has accepted the compose request for this session."],
    likelyNext:
      composerProgress.likely_next ??
      "ELSPETH will continue through the visible composer workflow.",
  };
}

function heuristicWorkingView(
  latestRequest: string | null | undefined,
  compositionState: CompositionState | null | undefined,
): WorkingView {
  const requestFocus = describeRequestFocus(latestRequest);
  return {
    headline: requestFocus.headline,
    evidence: [
      requestFocus.focus,
      describeCurrentSetup(compositionState),
    ],
    likelyNext: requestFocus.nextMove,
  };
}

/**
 * Animated three-dot composing indicator shown while the backend
 * is processing the LLM tool-use loop. Uses the .composing-dot CSS
 * class from App.css for staggered bounce animation.
 * Announces to screen readers via aria-live.
 */
export function ComposingIndicator({
  latestRequest = null,
  compositionState = null,
  composerProgress = null,
}: ComposingIndicatorProps) {
  const workingView =
    backendWorkingView(composerProgress) ??
    heuristicWorkingView(latestRequest, compositionState);

  return (
    <div
      className="composing-indicator composing-row"
      aria-live="polite"
      role="status"
    >
      <div className="composing-bubble">
        <div className="composing-pulse" aria-hidden="true">
          <span className="composing-dot" />
          <span className="composing-dot" />
          <span className="composing-dot" />
        </div>
        <div className="composing-working-view">
          <div className="composing-label">Working on...</div>
          <div className="composing-title">{workingView.headline}</div>
          <div className="composing-section">
            <div className="composing-label">What ELSPETH can see</div>
            <ul className="composing-evidence">
              {workingView.evidence.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="composing-section">
            <div className="composing-label">Likely next</div>
            <div className="composing-text">{workingView.likelyNext}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
