import { type ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";

interface LayoutProps {
  chat: ReactNode;
  siderail: ReactNode;
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
    <div className="app-layout">
      <div className="layout-chat" data-testid="layout-chat">
        <DefaultModeChangedBanner />
        <ErrorBoundary label="Chat panel">
          {chat}
        </ErrorBoundary>
      </div>

      <div className="layout-siderail">
        <ErrorBoundary label="Side rail">
          {siderail}
        </ErrorBoundary>
      </div>
    </div>
  );
}
