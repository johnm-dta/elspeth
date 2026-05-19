import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineChatSourceEntry } from "./InlineChatSourceEntry";
import { PREFILL_CHAT_INPUT_EVENT } from "./PluginCard";

describe("InlineChatSourceEntry", () => {
  let handler: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    handler = vi.fn();
    window.addEventListener(PREFILL_CHAT_INPUT_EVENT, handler as EventListener);
  });
  afterEach(() => {
    window.removeEventListener(PREFILL_CHAT_INPUT_EVENT, handler as EventListener);
  });

  it("renders the entry title and description", () => {
    render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
    expect(screen.getByText(/for a URL, sentence, or single record/i)).toBeInTheDocument();
  });

  it("renders a distinct visual style (not a regular plugin-card)", () => {
    const { container } = render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    // The synthetic-entry class is what differentiates it visually.
    expect(container.firstChild).toHaveClass("inline-chat-source-entry");
  });

  it("dispatches PREFILL_CHAT_INPUT_EVENT and closes the drawer on click", async () => {
    const onCloseDrawer = vi.fn();
    render(<InlineChatSourceEntry onCloseDrawer={onCloseDrawer} />);
    await userEvent.click(screen.getByRole("button", { name: /try it/i }));
    expect(handler).toHaveBeenCalled();
    expect(onCloseDrawer).toHaveBeenCalled();
  });

  it("dispatches a non-empty string detail", async () => {
    render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: /try it/i }));
    const event = handler.mock.calls[0][0] as CustomEvent<string>;
    expect(typeof event.detail).toBe("string");
    expect(event.detail.length).toBeGreaterThan(10);
  });
});
