/**
 * SharedInspectView — Phase 6B Task 8 (refactored under FIX-C).
 *
 * Read-only inspect view rendered when the URL hash is in the form
 * `#/shared/{token}`. Mounted by App.tsx as a top-level branch that
 * short-circuits the regular composer UI.
 *
 * The token is a CAPABILITY, not an authenticator — the reviewer must
 * still be logged in. AuthGuard wraps this view so unauthenticated
 * users hit the login flow; on successful auth they land back here
 * via the persisted hash (the AuthGuard preserves the hash through
 * the login redirect via sessionStorage).
 *
 * FIX-C refactor (2026-05-19):
 *
 *   The initial Phase 6B Task 8 implementation rendered the readiness
 *   panel as an inline 3-column `<table>` and the YAML as a raw
 *   `<pre><code>` block, both inline in this file. The spec-compliance
 *   reviewer flagged that as NON-COMPLIANT vs. the plan's "shared
 *   inspect mounts a read-only AuditReadinessPanel + reused YamlView"
 *   description. The plan's literal references were aspirational —
 *   `InspectorPanel` doesn't exist in this codebase, and YamlView /
 *   GraphMiniView are session-store-coupled — so the FIX-C remediation
 *   extracted reusable primitives:
 *
 *     - `AuditReadinessRow` (extracted from AuditReadinessPanel)
 *     - `SharedAuditReadinessPanel` (renders 6 rows from a frozen
 *       snapshot, no live overlays, wrapped in ReadOnlyProvider)
 *     - `YamlDisplay` (extracted from YamlView; pure renderer, no
 *       store coupling)
 *     - `GraphMiniView.compositionStateOverride` prop
 *     - `ReadOnlyContext` / `useReadOnly`
 *
 *   The view is wrapped in `<ReadOnlyProvider value={true}>` so any
 *   descendant that respects the read-only signal disables its action
 *   affordances automatically.
 *
 * The view renders:
 *
 *   * Pipeline metadata (name, description).
 *   * `<SharedAuditReadinessPanel>` for the frozen six-row panel.
 *   * `<GraphMiniView compositionStateOverride={...} />` for the
 *     structural view (the GraphModal that the click would dispatch
 *     is NOT mounted in this surface — see GraphMiniView's prop
 *     docstring).
 *   * `<YamlDisplay>` for the frozen YAML.
 *
 * The composer chat panel, run controls, and edit affordances are
 * deliberately ABSENT — this is an inspect-only view, not a fork
 * surface (multi-user collaborative editing is out of scope per
 * plan 19a §"Scope boundaries").
 *
 * Error handling:
 *
 * * 401 (tampered/expired token) → "This share link is no longer valid"
 *   with a "Return to my workspace" CTA.
 * * 404 (blob reaped) → "This shared snapshot is no longer available"
 *   with a "Ask the sender for a fresh link" message.
 * * Other / network → "Couldn't load the shared pipeline" with retry.
 */

import { useEffect, useState } from "react";

import { fetchSharedInspect } from "@/api/shareableReviews";
import { ReadOnlyProvider } from "@/contexts/ReadOnlyContext";
import { SharedAuditReadinessPanel } from "./SharedAuditReadinessPanel";
import { YamlDisplay } from "@/components/inspector/YamlDisplay";
import { GraphMiniView } from "@/components/sidebar/GraphMiniView";
import type {
  ApiError,
  CompositionState,
  SharedInspectResponse,
} from "@/types/api";

interface SharedInspectViewProps {
  token: string;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "loaded"; response: SharedInspectResponse }
  | { kind: "error"; status: number | null; message: string };

function _isApiError(value: unknown): value is ApiError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { status?: unknown }).status === "number"
  );
}

function _classifyError(exc: unknown): { status: number | null; message: string } {
  if (_isApiError(exc)) {
    if (exc.status === 401) {
      return {
        status: 401,
        message:
          "This share link is no longer valid. The link may have expired or " +
          "been revoked. Ask the sender for a fresh link.",
      };
    }
    if (exc.status === 404) {
      return {
        status: 404,
        message:
          "This shared snapshot is no longer available. The original " +
          "operator may need to re-share — ask the sender for a fresh link.",
      };
    }
    return {
      status: exc.status,
      message: exc.detail ?? "Couldn't load the shared pipeline.",
    };
  }
  if (exc instanceof Error) {
    return { status: null, message: exc.message };
  }
  return { status: null, message: "Couldn't load the shared pipeline." };
}

function _returnToWorkspaceUrl(): string {
  return `${window.location.origin}${window.location.pathname}`;
}

/**
 * Validate and narrow the wire `composition_snapshot` (typed as
 * `Record<string, unknown>` on the response) into a `CompositionState`
 * shape that GraphMiniView's MiniSvg can read. This is a Tier-1
 * inbound boundary — the snapshot was canonicalised at mark-time, so
 * any shape drift is a system bug (audit-blob serializer changed
 * without updating the consumer). We assert offensively rather than
 * silently coerce.
 */
function _narrowCompositionSnapshot(
  snapshot: Record<string, unknown>,
): CompositionState {
  const nodes = (snapshot as { nodes?: unknown }).nodes;
  const outputs = (snapshot as { outputs?: unknown }).outputs;
  if (!Array.isArray(nodes)) {
    throw new Error(
      "shared composition_snapshot is missing `nodes` array — wire shape drift",
    );
  }
  if (!Array.isArray(outputs)) {
    throw new Error(
      "shared composition_snapshot is missing `outputs` array — wire shape drift",
    );
  }
  // MiniSvg reads only .source (truthiness), .nodes.length, and
  // .outputs.length — fields validated above. The cast is safe because
  // the upstream invariant (set_pipeline.py / shareable-reviews
  // serializer) writes a strict CompositionState shape; this boundary
  // catches drift before the renderer sees an invalid object.
  return snapshot as unknown as CompositionState;
}

export function SharedInspectView({ token }: SharedInspectViewProps): JSX.Element {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setState({ kind: "loading" });
    fetchSharedInspect(token, controller.signal)
      .then((response) => {
        if (cancelled) return;
        setState({ kind: "loaded", response });
      })
      .catch((exc: unknown) => {
        if (cancelled) return;
        const { status, message } = _classifyError(exc);
        setState({ kind: "error", status, message });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [token]);

  if (state.kind === "loading") {
    return (
      <main
        role="main"
        className="shared-inspect-view"
        data-testid="shared-inspect-loading"
        aria-busy="true"
      >
        <p>Loading shared pipeline…</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main
        role="main"
        className="shared-inspect-view shared-inspect-view--error"
        data-testid="shared-inspect-error"
      >
        <h1>Shared link unavailable</h1>
        <p role="alert">{state.message}</p>
        <p>
          <a href={_returnToWorkspaceUrl()} data-testid="shared-inspect-return-link">
            Return to my workspace
          </a>
        </p>
      </main>
    );
  }

  const { response } = state;
  const pipelineName =
    typeof response.pipeline_metadata.name === "string"
      ? response.pipeline_metadata.name
      : "Untitled pipeline";
  const pipelineDescription =
    typeof response.pipeline_metadata.description === "string"
      ? response.pipeline_metadata.description
      : "";

  // Narrow the frozen composition snapshot from the wire (typed as
  // Record<string, unknown>) to the CompositionState shape MiniSvg
  // reads. Done at the boundary — any drift crashes loudly here.
  const frozenComposition = _narrowCompositionSnapshot(
    response.composition_snapshot,
  );

  return (
    <ReadOnlyProvider value={true}>
      <main
        role="main"
        className="shared-inspect-view shared-inspect-view--loaded"
        data-testid="shared-inspect-loaded"
      >
        <header>
          <p className="shared-inspect-banner" role="status">
            Read-only shared view. Shared by{" "}
            <strong>{response.created_by_user_id}</strong> on{" "}
            <time dateTime={response.created_at}>
              {new Date(response.created_at).toLocaleString()}
            </time>
            ; expires on{" "}
            <time dateTime={response.expires_at}>
              {new Date(response.expires_at).toLocaleString()}
            </time>
            .
          </p>
          <h1 data-testid="shared-inspect-pipeline-name">{pipelineName}</h1>
          {pipelineDescription !== "" && (
            <p data-testid="shared-inspect-pipeline-description">
              {pipelineDescription}
            </p>
          )}
        </header>

        <section
          aria-label="Pipeline structure (read-only)"
          data-testid="shared-inspect-graph"
        >
          <h2>Pipeline structure</h2>
          <GraphMiniView compositionStateOverride={frozenComposition} />
        </section>

        <section
          aria-label="Audit readiness panel (read-only)"
          data-testid="shared-inspect-audit-readiness"
        >
          <p>
            The owner reviewed this readiness panel at the moment of marking
            the pipeline ready for review. The values below are frozen — they
            reflect the owner's mark-time view, not the live state.
          </p>
          <SharedAuditReadinessPanel snapshot={response.audit_readiness} />
        </section>

        <section
          aria-label="Pipeline YAML (read-only)"
          data-testid="shared-inspect-yaml"
        >
          <h2>Pipeline YAML</h2>
          <YamlDisplay yaml={response.yaml} filename="pipeline.yaml" />
        </section>

        <footer>
          <p>
            <a
              href={_returnToWorkspaceUrl()}
              data-testid="shared-inspect-return-link"
            >
              Return to my workspace
            </a>
          </p>
        </footer>
      </main>
    </ReadOnlyProvider>
  );
}
