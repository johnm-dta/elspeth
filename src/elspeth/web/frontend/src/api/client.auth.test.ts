import { beforeEach, describe, expect, it, vi } from "vitest";
import { verifyEmail } from "./client";
import { useAuthStore } from "@/stores/authStore";
import { resetStore } from "@/test/store-helpers";

describe("api/client auth helpers", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
    localStorage.clear();
    resetStore(useAuthStore);
  });

  it("does not log out an existing session when email verification returns 401", async () => {
    const user = {
      user_id: "alice",
      username: "alice",
      display_name: "Alice",
      email: "alice@example.com",
      groups: [],
    };
    localStorage.setItem("auth_token", "still-valid");
    useAuthStore.setState({
      token: "still-valid",
      user,
      loginError: null,
      isLoading: false,
    } as never);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Email verification token expired" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(verifyEmail("expired-token")).rejects.toMatchObject({
      status: 401,
    });
    expect(useAuthStore.getState().token).toBe("still-valid");
    expect(useAuthStore.getState().user).toEqual(user);
    expect(localStorage.getItem("auth_token")).toBe("still-valid");
  });
});
