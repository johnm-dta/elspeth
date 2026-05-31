import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

import {
  useSharedToken,
  _parseSharedTokenForTesting,
} from "./useSharedToken";

function _setHash(hash: string) {
  window.location.hash = hash;
}

describe("_parseSharedTokenForTesting (pure parser)", () => {
  it("returns null for empty hash", () => {
    expect(_parseSharedTokenForTesting("")).toBeNull();
  });

  it("returns null for plain session-id hash #/{sessionId}", () => {
    expect(_parseSharedTokenForTesting("#/sess-uuid")).toBeNull();
  });

  it("extracts the token from #/shared/{token}", () => {
    expect(_parseSharedTokenForTesting("#/shared/abc-xyz")).toBe("abc-xyz");
  });

  it("URL-decodes the token", () => {
    expect(_parseSharedTokenForTesting("#/shared/abc%2Fdef%2B")).toBe("abc/def+");
  });

  it("returns null when the prefix is present but the token is empty", () => {
    expect(_parseSharedTokenForTesting("#/shared/")).toBeNull();
  });

  it("returns null for malformed percent-encoding", () => {
    // %ZZ is not a valid percent-encoded byte; decodeURIComponent throws.
    expect(_parseSharedTokenForTesting("#/shared/abc%ZZ")).toBeNull();
  });
});

describe("useSharedToken", () => {
  beforeEach(() => {
    _setHash("");
  });

  it("returns null on mount when hash is empty", () => {
    const { result } = renderHook(() => useSharedToken());
    expect(result.current).toBeNull();
  });

  it("returns the token on mount when hash is #/shared/{token}", () => {
    _setHash("#/shared/initial-token");
    const { result } = renderHook(() => useSharedToken());
    expect(result.current).toBe("initial-token");
  });

  it("updates when the hash changes", () => {
    const { result } = renderHook(() => useSharedToken());
    expect(result.current).toBeNull();
    act(() => {
      _setHash("#/shared/tok-after");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(result.current).toBe("tok-after");
  });

  it("clears to null when leaving the shared route", () => {
    _setHash("#/shared/tok");
    const { result } = renderHook(() => useSharedToken());
    expect(result.current).toBe("tok");
    act(() => {
      _setHash("#/some-session-uuid");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(result.current).toBeNull();
  });
});
