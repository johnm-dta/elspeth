import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import type { CompositionState } from "@/types/index";
import { hasCompositionContent, hasSources } from "@/utils/compositionState";

interface MiniLane {
  label: string;
  colorVar: string;
}

interface GraphMiniViewProps {
  /**
   * Phase 6B FIX-C — optional composition override for read-only
   * surfaces (e.g. SharedInspectView) that render a frozen snapshot
   * rather than the live store state. When supplied, this value is
   * used instead of `useSessionStore.compositionState`. The store is
   * still subscribed-to in the render to keep behaviour identical for
   * the regular composer path; the subscription is cheap and only
   * matters for the shared route, which mounts once per token.
   *
   * IMPORTANT: when override is supplied, the click affordance
   * dispatches `OPEN_GRAPH_MODAL_EVENT` as usual; the receiving
   * GraphModal is store-coupled and will render the LIVE composition
   * rather than the frozen one. The SharedInspectView callsite must
   * either suppress the modal mount or pass a corresponding override
   * downstream — for FIX-C scope, the mini view is read-only display
   * and the click is unwired in the shared surface (the SharedInspect
   * subtree does not mount a GraphModal).
   */
  compositionStateOverride?: CompositionState | null;
}

export function GraphMiniView({
  compositionStateOverride,
}: GraphMiniViewProps = {}): JSX.Element {
  const storeCompositionState = useSessionStore((s) => s.compositionState);
  const compositionState =
    compositionStateOverride !== undefined
      ? compositionStateOverride
      : storeCompositionState;

  if (!hasCompositionContent(compositionState)) {
    return (
      <div className="graph-mini graph-mini--empty" data-testid="graph-mini-empty">
        <span>No pipeline yet</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="graph-mini"
      onClick={() =>
        window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT))
      }
      aria-label="Pipeline graph (click to expand)"
    >
      <MiniSvg state={compositionState} />
    </button>
  );
}

function buildLanes(state: CompositionState): MiniLane[] {
  const lanes: MiniLane[] = [];
  if (hasSources(state)) {
    const count = Object.keys(state.sources).length;
    lanes.push({
      label: count === 1 ? "src" : `${count} src`,
      colorVar: "--color-accent",
    });
  }
  if (state.nodes.length > 0) {
    lanes.push({
      label: `${state.nodes.length} tx`,
      colorVar: "--color-info",
    });
  }
  if (state.outputs.length > 0) {
    lanes.push({
      label: state.outputs.length === 1 ? "sink" : `${state.outputs.length} sinks`,
      colorVar: "--color-success",
    });
  }
  return lanes;
}

function MiniSvg({ state }: { state: CompositionState }): JSX.Element {
  const lanes = buildLanes(state);
  const width = 240;
  const height = 80;
  const laneWidth = width / Math.max(lanes.length, 1);

  return (
    <svg width={width} height={height} role="img" aria-hidden="true">
      {lanes.map((lane, index) => {
        const x = index * laneWidth + 8;
        const rectWidth = laneWidth - 16;
        return (
          <g key={lane.label}>
            <rect
              x={x}
              y={height / 2 - 16}
              width={rectWidth}
              height={32}
              rx={6}
              fill={`var(${lane.colorVar})`}
              opacity={0.85}
            />
            <text
              x={x + rectWidth / 2}
              y={height / 2 + 4}
              textAnchor="middle"
              fontSize={12}
              fill="white"
            >
              {lane.label}
            </text>
            {index < lanes.length - 1 && (
              <line
                x1={x + rectWidth}
                y1={height / 2}
                x2={(index + 1) * laneWidth + 8}
                y2={height / 2}
                stroke="var(--color-text-muted)"
                strokeWidth={2}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
