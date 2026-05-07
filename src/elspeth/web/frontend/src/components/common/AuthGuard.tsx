import type { ReactNode } from "react";
import { useAuth } from "../../hooks/useAuth";
import { LoginPage } from "../auth/LoginPage";

interface AuthGuardProps {
  children: ReactNode;
}

/**
 * Auth gate component. Renders at the top of the component tree.
 * - Shows a loading spinner while checking stored credentials.
 * - Renders LoginPage if the user is not authenticated.
 * - Renders children if authenticated.
 */
export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Checking authentication"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
        }}
      >
        <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <>{children}</>;
}
