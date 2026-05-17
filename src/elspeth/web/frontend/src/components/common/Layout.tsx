import { type ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";

const SIDERAIL_WIDTH = 320;

interface LayoutProps {
  chat: ReactNode;
  siderail: ReactNode;
}

/**
 * Fixed composer shell: chat takes the fluid column, SideRail owns the
 * dedicated right rail. The default-mode banner stays in the chat column so it
 * consumes chat height rather than adding untracked vertical space above the
 * composer grid.
 */
export function Layout({
  chat,
  siderail,
}: LayoutProps): JSX.Element {
  return (
    <div
      className="app-layout"
      style={{ gridTemplateColumns: `minmax(0, 1fr) ${SIDERAIL_WIDTH}px` }}
    >
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
