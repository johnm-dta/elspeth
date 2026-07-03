import { type ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";

interface LayoutProps {
  chat: ReactNode;
  /**
   * The right-rail content, or `null` to omit the rail COLUMN entirely
   * (single-column shell). App passes null while a guided build is active —
   * the guided workspace inside the chat panel carries its own rail
   * (.guided-workspace-rail), and rendering both puts two rails side by
   * side (isGuidedBuildActive is the shared predicate).
   */
  siderail: ReactNode | null;
}

/**
 * Fixed composer shell: chat takes the fluid column, SideRail owns the
 * dedicated right rail. The default-mode banner stays in the chat column so it
 * consumes chat height rather than adding untracked vertical space above the
 * composer grid.
 *
 * Column sizing lives entirely in CSS (shared.css .app-layout, rail width via
 * the --siderail-width token) so the ≤960px responsive collapse — single
 * column, rail stacked under the chat — can restructure the grid with a media
 * query. Do not reintroduce an inline grid-template-columns style here: inline
 * styles out-specify every stylesheet rule and made the shell impossible to
 * collapse (elspeth-49dd290c7a).
 */
export function Layout({
  chat,
  siderail,
}: LayoutProps): JSX.Element {
  return (
    <div
      className={
        siderail == null ? "app-layout app-layout--chat-only" : "app-layout"
      }
    >
      <div className="layout-chat" data-testid="layout-chat">
        <DefaultModeChangedBanner />
        <ErrorBoundary label="Chat panel">
          {chat}
        </ErrorBoundary>
      </div>

      {siderail != null && (
        <div className="layout-siderail">
          <ErrorBoundary label="Side rail">
            {siderail}
          </ErrorBoundary>
        </div>
      )}
    </div>
  );
}
