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
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("notifies callers when an abnormal close enters reconnect and when the socket opens again", () => {
    const callbacks = {
      onProgress: vi.fn(),
      onError: vi.fn(),
      onComplete: vi.fn(),
      onCancelled: vi.fn(),
      onFailed: vi.fn(),
      onAuthFailure: vi.fn(),
      onConnected: vi.fn(),
      onDisconnected: vi.fn(),
    };

    connectToRun("run-1", "token-1", callbacks);
    expect(MockWebSocket.instances).toHaveLength(1);

    MockWebSocket.instances[0].open();
    expect(callbacks.onConnected).toHaveBeenCalledTimes(1);

    MockWebSocket.instances[0].closeWith(1006);
    expect(callbacks.onDisconnected).toHaveBeenCalledTimes(1);
    expect(MockWebSocket.instances).toHaveLength(1);

    vi.advanceTimersByTime(1000);
    expect(MockWebSocket.instances).toHaveLength(2);

    MockWebSocket.instances[1].open();
    expect(callbacks.onConnected).toHaveBeenCalledTimes(2);
  });

  it("does not reconnect after the run-not-found close code", () => {
    const callbacks = {
      onProgress: vi.fn(),
      onError: vi.fn(),
      onComplete: vi.fn(),
      onCancelled: vi.fn(),
      onFailed: vi.fn(),
      onAuthFailure: vi.fn(),
      onConnected: vi.fn(),
      onDisconnected: vi.fn(),
    };

    connectToRun("missing-run", "token-1", callbacks);
    expect(MockWebSocket.instances).toHaveLength(1);

    MockWebSocket.instances[0].closeWith(4004);
    expect(callbacks.onDisconnected).not.toHaveBeenCalled();
    expect(callbacks.onAuthFailure).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1000);
    expect(MockWebSocket.instances).toHaveLength(1);
  });
});
