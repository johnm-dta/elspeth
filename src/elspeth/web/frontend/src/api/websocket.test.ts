import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { connectToRun } from "./websocket";

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  readonly url: string;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  close(): void {
    this.onclose?.({ code: 1000 } as CloseEvent);
  }

  open(): void {
    this.onopen?.(new Event("open"));
  }

  closeWith(code: number): void {
    this.onclose?.({ code } as CloseEvent);
  }
}

describe("connectToRun", () => {
  function callbacks() {
    return {
      onProgress: vi.fn(),
      onError: vi.fn(),
      onComplete: vi.fn(),
      onCancelled: vi.fn(),
      onFailed: vi.fn(),
      onAuthFailure: vi.fn(),
      onConnected: vi.fn(),
      onDisconnected: vi.fn(),
      onRunUnavailable: vi.fn(),
    };
  }

  async function flushPromises(): Promise<void> {
    await Promise.resolve();
    await Promise.resolve();
  }

  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("requests a fresh opaque ticket for each socket URL", async () => {
    const handlers = callbacks();
    const getTicket = vi.fn()
      .mockResolvedValueOnce("ticket-1")
      .mockResolvedValueOnce("ticket-2");

    connectToRun("run-1", getTicket, handlers);
    await flushPromises();

    expect(MockWebSocket.instances).toHaveLength(1);
    let url = new URL(MockWebSocket.instances[0].url);
    expect(url.pathname).toBe("/ws/runs/run-1");
    expect(url.searchParams.get("ticket")).toBe("ticket-1");
    expect(url.searchParams.has("token")).toBe(false);

    MockWebSocket.instances[0].closeWith(1006);
    vi.advanceTimersByTime(1000);
    await flushPromises();

    expect(getTicket).toHaveBeenCalledTimes(2);
    expect(MockWebSocket.instances).toHaveLength(2);
    url = new URL(MockWebSocket.instances[1].url);
    expect(url.searchParams.get("ticket")).toBe("ticket-2");
    expect(url.searchParams.has("token")).toBe(false);
  });

  it("notifies callers when an abnormal close enters reconnect and when the socket opens again", async () => {
    const handlers = callbacks();

    connectToRun("run-1", vi.fn().mockResolvedValue("ticket-1"), handlers);
    await flushPromises();
    expect(MockWebSocket.instances).toHaveLength(1);

    MockWebSocket.instances[0].open();
    expect(handlers.onConnected).toHaveBeenCalledTimes(1);

    MockWebSocket.instances[0].closeWith(1006);
    expect(handlers.onDisconnected).toHaveBeenCalledTimes(1);
    expect(MockWebSocket.instances).toHaveLength(1);

    vi.advanceTimersByTime(1000);
    await flushPromises();
    expect(MockWebSocket.instances).toHaveLength(2);

    MockWebSocket.instances[1].open();
    expect(handlers.onConnected).toHaveBeenCalledTimes(2);
  });

  it("does not reconnect after the run-not-found close code", async () => {
    const handlers = callbacks();

    connectToRun("missing-run", vi.fn().mockResolvedValue("ticket-1"), handlers);
    await flushPromises();
    expect(MockWebSocket.instances).toHaveLength(1);

    MockWebSocket.instances[0].closeWith(4004);
    expect(handlers.onDisconnected).not.toHaveBeenCalled();
    expect(handlers.onAuthFailure).not.toHaveBeenCalled();
    expect(handlers.onRunUnavailable).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(1000);
    await flushPromises();
    expect(MockWebSocket.instances).toHaveLength(1);
  });
});
