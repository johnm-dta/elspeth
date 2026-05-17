import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import type { CompositionState } from "@/types/index";

interface MiniLane {
  label: string;
  colorVar: string;
}

export function GraphMiniView(): JSX.Element {
  const compositionState = useSessionStore((s) => s.compositionState);

  if (
    !compositionState ||
    (!compositionState.source &&
      compositionState.nodes.length === 0 &&
      compositionState.outputs.length === 0)
  ) {
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
  if (state.source) {
    lanes.push({ label: "src", colorVar: "--color-accent" });
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
