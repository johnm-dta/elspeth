import { useEffect } from "react";
import { useAuthStore, selectIsAuthenticated } from "@/stores/authStore";

/**
 * Hook for auth lifecycle. Calls loadFromStorage on mount,
 * then validates the stored token via GET /api/auth/me.
 * Returns auth state and actions.
 */
export function useAuth() {
  const loadFromStorage = useAuthStore((s) => s.loadFromStorage);
  const isAuthenticated = useAuthStore(selectIsAuthenticated);
  const isLoading = useAuthStore((s) => s.isLoading);
  const user = useAuthStore((s) => s.user);
  const loginError = useAuthStore((s) => s.loginError);
  const login = useAuthStore((s) => s.login);
  const loginWithToken = useAuthStore((s) => s.loginWithToken);
  const logout = useAuthStore((s) => s.logout);

  useEffect(() => {
    loadFromStorage();
  }, [loadFromStorage]);

  return { isAuthenticated, isLoading, user, loginError, login, loginWithToken, logout };
}
