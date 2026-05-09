// ============================================================================
// ChatInput — listener-stabilisation regression coverage.
//
// Pins the contract that the PREFILL_CHAT_INPUT_EVENT listener must (a) be
// registered exactly once for the lifetime of the component (not re-registered
// on every parent re-render in controlled mode), and (b) always resolve to the
// latest setText / onChange handler — i.e. the ref-trampoline pattern at
// ChatInput.tsx:51-52 is load-bearing.  A behavioural test that only fired one
// event would still pass if a future refactor reverted to closing over setText
// directly; this test fires the event AFTER a parent re-render to catch that.
// ============================================================================

import { useRef, useState, type RefObject } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatInput } from "./ChatInput";
import { useSessionStore } from "@/stores/sessionStore";
import { useBlobStore } from "@/stores/blobStore";
import { resetStore } from "@/test/store-helpers";
import { PREFILL_CHAT_INPUT_EVENT } from "@/components/catalog/PluginCard";

describe("ChatInput — controlled-mode prefill listener", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useBlobStore);
  });

  function ControlledHarness() {
    const [value, setValue] = useState("");
    const [renderTick, setRenderTick] = useState(0);
    const inputRef = useRef<HTMLTextAreaElement>(
      null,
    ) as RefObject<HTMLTextAreaElement>;
    // CRITICAL: onChange closes over `renderTick`.  This is the discriminator
    // that turns the ref-trampoline test from a tautology into a real test.
    // - With the trampoline, setTextRef.current points at the LATEST onChange,
    //   which captures the LATEST renderTick.  After 3 rerenders, prefill
    //   writes "${detail}:3".
    // - Without the trampoline, the listener closes over the FIRST onChange,
    //   which captures renderTick=0.  Prefill writes "${detail}:0".
    // The suffix asymmetry is what proves the trampoline is load-bearing.
    return (
      <div>
        <button
          type="button"
          data-testid="force-rerender"
          onClick={() => setRenderTick((n) => n + 1)}
        >
          rerender {renderTick}
        </button>
        <ChatInput
          onSend={vi.fn()}
          disabled={false}
          inputRef={inputRef}
          value={value}
          onChange={(next) => setValue(`${next}:${renderTick}`)}
        />
      </div>
    );
  }

  it("populates the textarea when PREFILL_CHAT_INPUT_EVENT fires", async () => {
    render(<ControlledHarness />);

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    expect(textarea.value).toBe("");

    act(() => {
      window.dispatchEvent(
        new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
          detail: "Add csv as the source",
        }),
      );
    });

    // renderTick=0 at this point — harness writes "${detail}:0".
    expect(textarea.value).toBe("Add csv as the source:0");
  });

  it("uses the LATEST setText after parent re-renders (ref-trampoline must be load-bearing)", async () => {
    // Regression: a previous bug closed over setText directly in the effect.
    // In controlled mode, setText identity changes on every parent render.
    // Without the ref trampoline (ChatInput.tsx:51-52), the listener would
    // hold a stale closure pointing to the FIRST onChange — which captures
    // renderTick=0.  After 3 rerenders, prefill should write "${detail}:3"
    // (latest tick) if the trampoline works.  If the listener has stale
    // closure, it writes "${detail}:0" and this test fails.
    const user = userEvent.setup();
    render(<ControlledHarness />);

    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));

    act(() => {
      window.dispatchEvent(
        new CustomEvent(PREFILL_CHAT_INPUT_EVENT, { detail: "after rerenders" }),
      );
    });

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    // With the trampoline: latest onChange ran, captured renderTick=3.
    // Without the trampoline: stale onChange ran, captured renderTick=0.
    expect(textarea.value).toBe("after rerenders:3");
  });

  it("registers the listener exactly once across re-renders", async () => {
    // Verify the [] dep array on the prefill effect: addEventListener must
    // not fire on every render.  We spy on window.addEventListener and count
    // PREFILL_CHAT_INPUT_EVENT registrations.
    const addSpy = vi.spyOn(window, "addEventListener");
    const user = userEvent.setup();
    render(<ControlledHarness />);

    const initial = addSpy.mock.calls.filter(
      ([type]) => type === PREFILL_CHAT_INPUT_EVENT,
    ).length;
    expect(initial).toBe(1);

    await user.click(screen.getByTestId("force-rerender"));
    await user.click(screen.getByTestId("force-rerender"));

    const after = addSpy.mock.calls.filter(
      ([type]) => type === PREFILL_CHAT_INPUT_EVENT,
    ).length;
    expect(after).toBe(1);

    addSpy.mockRestore();
  });

  it("throws TypeError on non-string event detail (CLAUDE.md trust-tier: internal contract violations crash)", () => {
    // WHATWG DOM spec: event listener errors do NOT propagate through
    // dispatchEvent — the caller continues; the error is reported via
    // window.onerror / 'error' event.  Capture that report to prove the
    // crash actually fires and is loud (DevTools-visible) rather than
    // silently caught.
    const errorEvents: ErrorEvent[] = [];
    const errorListener = (e: ErrorEvent) => {
      errorEvents.push(e);
      e.preventDefault(); // suppress jsdom's "unhandled error" stderr noise
    };
    window.addEventListener("error", errorListener);

    render(<ControlledHarness />);

    // Dispatch the malformed event — listener throws synchronously.
    window.dispatchEvent(
      new CustomEvent(PREFILL_CHAT_INPUT_EVENT, {
        detail: { not: "a string" } as unknown as string,
      }),
    );

    // The TypeError must have been reported as an unhandled error.
    expect(errorEvents).toHaveLength(1);
    expect(errorEvents[0].error).toBeInstanceOf(TypeError);
    expect(errorEvents[0].error.message).toContain("PREFILL_CHAT_INPUT_EVENT");

    const textarea = screen.getByLabelText(/message input/i) as HTMLTextAreaElement;
    // No silent state mutation: the bogus value must not have been written.
    expect(textarea.value).toBe("");

    window.removeEventListener("error", errorListener);
  });
});
