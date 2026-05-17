import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { SWITCH_TAB_EVENT } from "@/components/common/CommandPalette";
import { resetStore } from "@/test/store-helpers";
import { useSessionStore } from "@/stores/sessionStore";
import { useHashRouter } from "./useHashRouter";

const REDIRECT_TOAST_DISMISSED_KEY = "elspeth_redirect_toast_dismissed";

describe("useHashRouter removed tab redirects", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    localStorage.clear();
    window.history.replaceState(null, "", "/");
  });

  it("redirects stale Runs hashes to Graph and exposes a migration toast", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    window.history.replaceState(null, "", "#/session-1/runs");
    const tabRequests: string[] = [];
    const recordTabRequest = (event: Event) => {
      tabRequests.push((event as CustomEvent<string>).detail);
    };
    window.addEventListener(SWITCH_TAB_EVENT, recordTabRequest);

    try {
      const { result } = renderHook(() => useHashRouter());

      await waitFor(() => {
        expect(tabRequests).toEqual(["graph"]);
      });
      await waitFor(() => {
        expect(result.current.redirectToast?.message).toMatch(/Runs tab was removed/i);
      });
    } finally {
      window.removeEventListener(SWITCH_TAB_EVENT, recordTabRequest);
    }
  });

  it("redirects stale Spec hashes to Graph and exposes a migration toast", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    window.history.replaceState(null, "", "#/session-1/spec");
    const tabRequests: string[] = [];
    const recordTabRequest = (event: Event) => {
      tabRequests.push((event as CustomEvent<string>).detail);
    };
    window.addEventListener(SWITCH_TAB_EVENT, recordTabRequest);

    try {
      const { result } = renderHook(() => useHashRouter());

      await waitFor(() => {
        expect(tabRequests).toEqual(["graph"]);
      });
      await waitFor(() => {
        expect(result.current.redirectToast?.message).toMatch(/Spec tab was removed/i);
      });
    } finally {
      window.removeEventListener(SWITCH_TAB_EVENT, recordTabRequest);
    }
  });

  it("does not show the removed-tab toast after the user dismisses it", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    window.history.replaceState(null, "", "#/session-1/runs");
    const { result, unmount } = renderHook(() => useHashRouter());

    await waitFor(() => {
      expect(result.current.redirectToast).not.toBeNull();
    });

    act(() => {
      result.current.redirectToast?.dismiss();
    });

    expect(localStorage.getItem(REDIRECT_TOAST_DISMISSED_KEY)).toBe("1");
    expect(result.current.redirectToast).toBeNull();

    unmount();
    renderHook(() => useHashRouter());

    await waitFor(() => {
      expect(result.current?.redirectToast).toBeNull();
    });
  });
});
