import type { WireStageData } from "@/types/guided";

export interface WireEdge {
  from: string;
  to: string;
  label: string;
  satisfied: boolean | null;
  missing_fields: string[];
}

function pairKey(from: string, to: string): string {
  return `${from}\u0000${to}`;
}

export function reconstructWireEdges(data: WireStageData): WireEdge[] {
  const consumerByLabel = new Map<string, string>();
  for (const node of data.topology.nodes) {
    if (node.input !== null) {
      consumerByLabel.set(node.input, node.id);
    }
  }
  for (const output of data.topology.outputs) {
    consumerByLabel.set(output.sink_name, output.id);
  }

  const contractByPair = new Map(
    data.edge_contracts.map((contract) => [
      pairKey(contract.from, contract.to),
      contract,
    ]),
  );

  const edges: WireEdge[] = [];
  const pushEdge = (from: string, label: string | null) => {
    if (label === null) return;
    const to = consumerByLabel.get(label);
    if (to === undefined) return;
    const contract = contractByPair.get(pairKey(from, to));
    edges.push({
      from,
      to,
      label,
      satisfied: contract?.satisfied ?? null,
      missing_fields: contract?.missing_fields ?? [],
    });
  };

  for (const source of Object.values(data.topology.sources)) {
    pushEdge(source.id, source.on_success);
  }
  for (const node of data.topology.nodes) {
    pushEdge(node.id, node.on_success);
    pushEdge(node.id, node.on_error);
    for (const label of Object.values(node.routes ?? {})) {
      pushEdge(node.id, label);
    }
    for (const label of node.fork_to ?? []) {
      pushEdge(node.id, label);
    }
  }

  return edges;
}

export interface WireStageTurnProps {
  data: WireStageData;
  onConfirm: () => void;
  confirmDisabled: boolean;
}

function edgeStatus(edge: WireEdge): string {
  if (edge.satisfied === true) return "(connected)";
  if (edge.satisfied === false) return "(not satisfied)";
  return "(contract unchecked)";
}

function warningText(warning: Record<string, unknown>): string {
  const message = warning.message;
  if (typeof message === "string") return message;
  return JSON.stringify(warning);
}

export function WireStageTurn({
  data,
  onConfirm,
  confirmDisabled,
}: WireStageTurnProps) {
  const edges = reconstructWireEdges(data);

  return (
    <div className="guided-turn wire-stage">
      <h3>Review wiring</h3>

      {data.warnings.length > 0 ? (
        <ul className="wire-stage__warnings">
          {data.warnings.map((warning, index) => (
            <li key={index}>{warningText(warning)}</li>
          ))}
        </ul>
      ) : null}

      <ul className="wire-stage__edges">
        {edges.map((edge) => (
          <li
            key={`${edge.from}\u0000${edge.label}\u0000${edge.to}`}
            aria-label={`${edge.from} to ${edge.to}`}
          >
            <span>{edge.from}</span>
            <span>{" -> "}</span>
            <span>{edge.to}</span>
            <span> via {edge.label} </span>
            <span>{edgeStatus(edge)}</span>
            {edge.missing_fields.length > 0 ? (
              <span> Missing fields: {edge.missing_fields.join(", ")}</span>
            ) : null}
          </li>
        ))}
      </ul>

      <button
        type="button"
        className="guided-turn-primary"
        onClick={onConfirm}
        disabled={confirmDisabled}
      >
        Confirm wiring
      </button>
    </div>
  );
}
