import { create } from "zustand";
import type { UserProfile, ApiError } from "../types/index";
import * as api from "../api/client";
import { usePreferencesStore } from "./preferencesStore";
import { usePluginCatalogStore } from "./pluginCatalogStore";

const TOKEN_KEY = "auth_token";

interface AuthState {
  token: string | null;
  user: UserProfile | null;
  loginError: string | null;
  isLoading: boolean;

  login: (username: string, password: string) => Promise<boolean>;
  loginWithToken: (token: string) => Promise<void>;
  logout: () => Promise<void>;
  loadFromStorage: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  user: null,
  loginError: null,
  isLoading: true, // starts true; loadFromStorage resolves it

  async login(username: string, password: string) {
    // Deliberately does NOT touch isLoading: that flag drives AuthGuard's
    // "Checking authentication" spinner, which REPLACES the LoginPage in
    // the tree. Flipping it during an interactive attempt unmounted the
    // form, so a failed login remounted a blank LoginPage and wiped the
    // username the user had just typed (WCAG 3.3.7 Redundant Entry,
    // elspeth-d49f8ad511). In-flight progress is the form's own concern
    // (its submit button shows "Signing in…").
    //
    // Returns true on success so the form can clear only the password
    // (never the username) after a failed attempt.
    set({ loginError: null });
    try {
      const { access_token } = await api.login(username, password);
      localStorage.setItem(TOKEN_KEY, access_token);
      set({ token: access_token });

      const user = await api.fetchCurrentUser();
      usePluginCatalogStore.getState().clear();
      set({ user, isLoading: false });
      return true;
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 401
          ? "Invalid username or password."
          : apiErr.detail ?? "Login failed. Please try again.";
      set({ token: null, user: null, loginError: message, isLoading: false });
      usePluginCatalogStore.getState().clear();
      localStorage.removeItem(TOKEN_KEY);
      return false;
    }
  },

  async loginWithToken(token: string) {
    usePluginCatalogStore.getState().clear();
    localStorage.setItem(TOKEN_KEY, token);
    set({ token, loginError: null, isLoading: true });
    try {
      const user = await api.fetchCurrentUser();
      set({ user, isLoading: false });
    } catch {
      set({
        token: null,
        user: null,
        loginError: "Authentication failed. Please try signing in again.",
        isLoading: false,
      });
      localStorage.removeItem(TOKEN_KEY);
    }
  },

  async logout() {
    localStorage.removeItem(TOKEN_KEY);
    usePluginCatalogStore.getState().clear();
    set({ token: null, user: null, loginError: null, isLoading: false });
    const [
      { useSessionStore },
      { useExecutionStore },
      { useBlobStore },
      { useSecretsStore },
      { useShareableReviewStore },
    ] = await Promise.all([
      import("./sessionStore"),
      import("./executionStore"),
      import("./blobStore"),
      import("./secretsStore"),
      import("./shareableReviewStore"),
    ]);
    useSessionStore.getState().reset?.();
    useExecutionStore.getState().reset?.();
    useBlobStore.getState().reset();
    useSecretsStore.getState().reset();
    useShareableReviewStore.getState().reset();
    usePreferencesStore.getState().reset();
  },

  async loadFromStorage() {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      usePluginCatalogStore.getState().clear();
      usePreferencesStore.getState().reset();
      set({ isLoading: false });
      return;
    }
    set({ token });
    usePluginCatalogStore.getState().clear();
    try {
      const user = await api.fetchCurrentUser();
      set({ user, isLoading: false });
    } catch {
      // Token invalid or expired -- clear it
      localStorage.removeItem(TOKEN_KEY);
      usePreferencesStore.getState().reset();
      set({ token: null, user: null, isLoading: false });
    }
  },
}));

/**
 * Selector: true when the user is authenticated.
 * Usage: const isAuthenticated = useAuthStore(selectIsAuthenticated);
 */
export const selectIsAuthenticated = (state: AuthState): boolean =>
  state.token !== null && state.user !== null;
