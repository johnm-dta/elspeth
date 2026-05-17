// ============================================================================
// AppHeader
//
// Thin top-level header. Three regions left-to-right:
//   - ELSPETH brand
//   - HeaderSessionSwitcher
//   - HeaderVersionSelector
//   - UserMenu
// ============================================================================

import { HeaderSessionSwitcher } from "@/components/sessions/HeaderSessionSwitcher";
import { HeaderVersionSelector } from "@/components/header/HeaderVersionSelector";
import { UserMenu } from "@/components/common/UserMenu";

interface AppHeaderProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

export function AppHeader({
  onOpenSettings,
  onSignOut,
}: AppHeaderProps): JSX.Element {
  return (
    <header className="app-header" role="banner">
      <div className="app-header-left">
        <span className="app-header-brand">ELSPETH</span>
        <HeaderSessionSwitcher />
        <span className="app-header-separator" aria-hidden="true" />
        <HeaderVersionSelector />
      </div>
      <div className="app-header-right">
        <UserMenu onOpenSettings={onOpenSettings} onSignOut={onSignOut} />
      </div>
    </header>
  );
}
