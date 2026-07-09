// src/components/chat/ComposingIndicator.tsx

import { useEffect, useRef, useState } from "react";

import type { ComposerProgressSnapshot, CompositionState } from "@/types/api";
import { hasSources } from "@/utils/compositionState";

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
  /**
   * Provenance of the view (elspeth-b189b5b3b8 part c): "backend" views carry
   * evidence the server actually reported for this compose request;
   * "estimated" views are keyword-guessed from the user's message text and
   * must not read as if ELSPETH reported them. Rendering italicises estimated
   * views and appends a visible "(estimated)" marker to the headline.
   */
  source: "backend" | "estimated";
}

function isTerminalPhase(
  phase: ComposerProgressSnapshot["phase"] | undefined,
): boolean {
  return phase === "complete" || phase === "failed" || phase === "cancelled";
}

function terminalStatusLabel(
  phase: ComposerProgressSnapshot["phase"] | undefined,
): string {
  if (phase === "failed") return "Failed";
  if (phase === "cancelled") return "Stopped";
  return "Updated";
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
  const input = hasSources(compositionState) ? "input configured" : "no input yet";
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
    source: "backend",
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
    source: "estimated",
  };
}

/** Format elapsed whole seconds as a mm:ss readout (65 → "01:05"). */
export function formatElapsed(totalSeconds: number): string {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(clamped / 60);
  const seconds = clamped % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

/**
 * Elapsed-time readout for the in-flight compose card (elspeth-b189b5b3b8
 * part a): a slow turn must not read identically to a stalled request.
 * Counts from the moment the indicator becomes active (non-terminal) and
 * stops when a terminal phase lands.
 *
 * The ticking readout is aria-hidden: the indicator sits in a role="status"
 * live region and a once-per-second text mutation would spam screen readers
 * with announcements. Sighted users get the timer; AT users get the phase
 * headline changes, which already convey progress.
 *
 * Exported for the guided pending strip (GuidedPendingStrip.tsx), which
 * shares the same aria-hidden/mount-reset semantics.
 */
export function ElapsedReadout() {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <span className="composing-elapsed" aria-hidden="true">
      {formatElapsed(elapsedSeconds)}
    </span>
  );
}

/**
 * Animated three-dot composing indicator shown while the backend
 * is processing the LLM tool-use loop. Uses the .composing-dot CSS
 * class from styles/animations.css for staggered bounce animation.
 *
 * This component carries its own non-interactive role="status" summary and is
 * mounted OUTSIDE ChatPanel's role="log" messages container
 * (elspeth-76a0cc485e): nesting a status region inside an aria-live log risks
 * double announcements on AT that honours both regions.
 */
export function ComposingIndicator({
  latestRequest = null,
  compositionState = null,
  composerProgress = null,
}: ComposingIndicatorProps) {
  const workingView =
    backendWorkingView(composerProgress) ??
    heuristicWorkingView(latestRequest, compositionState);
  const isTerminal = isTerminalPhase(composerProgress?.phase);
  const isEstimated = workingView.source === "estimated";
  const progressKey = latestRequest ?? composerProgress?.request_id ?? "idle";
  const [detailsOpen, setDetailsOpen] = useState(isTerminal);

  useEffect(() => {
    setDetailsOpen(isTerminal);
  }, [isTerminal, progressKey]);

  return (
    <div
      className={`composing-indicator composing-row${isTerminal ? " composing-indicator--terminal" : ""}`}
    >
      <div className="composing-bubble">
        {isTerminal ? (
          <div className="composing-terminal-mark" aria-hidden="true">
            {terminalStatusLabel(composerProgress?.phase)}
          </div>
        ) : (
          <div className="composing-pulse" aria-hidden="true">
            <span className="composing-dot" />
            <span className="composing-dot" />
            <span className="composing-dot" />
          </div>
        )}
        <div
          className={`composing-working-view${isEstimated ? " composing-working-view--estimated" : ""}`}
        >
          <div className="composing-status-summary" role="status">
            <div className="composing-label">
              {isTerminal ? (
                "Last composer update"
              ) : (
                <>
                  Working on...
                  {/* Mount lifecycle doubles as the reset: the readout only
                      renders while non-terminal, so a terminal phase unmounts
                      it and the next compose remounts it from 00:00. */}
                  <ElapsedReadout />
                </>
              )}
            </div>
            <div className="composing-title">
              {workingView.headline}
              {isEstimated && (
                <span className="composing-estimated-tag"> (estimated)</span>
              )}
            </div>
          </div>

          <button
            type="button"
            className="composing-details-toggle"
            aria-expanded={detailsOpen}
            onClick={() => setDetailsOpen((open) => !open)}
          >
            {detailsOpen ? "Hide details" : "Show details"}
          </button>

          {detailsOpen && (
            <div className="composing-details">
              <div className="composing-section">
                <div className="composing-label">
                  {isEstimated ? "Best guess from your request" : "What ELSPETH can see"}
                </div>
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
          )}
        </div>
      </div>
    </div>
  );
}
