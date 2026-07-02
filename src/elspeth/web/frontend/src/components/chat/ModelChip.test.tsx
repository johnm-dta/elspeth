import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelChip } from "./ModelChip";
import * as apiClient from "@/api/client";
import type { SystemStatus } from "@/types/index";

vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    fetchSystemStatus: vi.fn(),
  };
});

const fetchSystemStatus = vi.mocked(apiClient.fetchSystemStatus);

function makeStatus(overrides: Partial<SystemStatus> = {}): SystemStatus {
  return {
    composer_available: true,
    composer_model: "anthropic/claude-sonnet-4.6",
    composer_provider: "openrouter",
    composer_reason: null,
    composer_missing_keys: [],
    ...overrides,
  };
}

describe("ModelChip", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("shows the composer model from system status with an accessible label", async () => {
    fetchSystemStatus.mockResolvedValue(makeStatus());

    render(<ModelChip />);

    await waitFor(() => {
      expect(
        screen.getByLabelText("Composer model: anthropic/claude-sonnet-4.6"),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("anthropic/claude-sonnet-4.6")).toBeInTheDocument();
  });

  it("renders nothing while the model is unknown or the status call fails", async () => {
    fetchSystemStatus.mockRejectedValue(new Error("offline"));

    const { container } = render(<ModelChip />);

    // Absence of chrome, never a fabricated model name.
    await waitFor(() => {
      expect(fetchSystemStatus).toHaveBeenCalled();
    });
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the deployment reports no composer model", async () => {
    fetchSystemStatus.mockResolvedValue(makeStatus({ composer_model: "" }));

    const { container } = render(<ModelChip />);

    await waitFor(() => {
      expect(fetchSystemStatus).toHaveBeenCalled();
    });
    expect(container).toBeEmptyDOMElement();
  });
});
