import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SecretsPanel } from "./SecretsPanel";
import { useSecretsStore } from "@/stores/secretsStore";

// Mock the API client
vi.mock("@/api/client", () => ({
  listSecrets: vi.fn().mockResolvedValue([]),
  createSecret: vi.fn().mockResolvedValue(undefined),
  deleteSecret: vi.fn().mockResolvedValue(undefined),
}));

describe("SecretsPanel", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    useSecretsStore.setState({
      secrets: [
        { name: "OPENAI_API_KEY", scope: "user" as const, available: true, source_kind: "user" },
        { name: "SERVER_KEY", scope: "server" as const, available: true, source_kind: "env" },
      ],
      isLoading: false,
      error: null,
      // Prevent loadSecrets from firing in useEffect
      loadSecrets: vi.fn(),
    });
  });

  it("renders the dialog with title", () => {
    render(<SecretsPanel onClose={onClose} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("API Keys & Secrets")).toBeInTheDocument();
  });

  it("shows secret inventory", () => {
    render(<SecretsPanel onClose={onClose} />);
    expect(screen.getByText("OPENAI_API_KEY")).toBeInTheDocument();
    expect(screen.getByText("SERVER_KEY")).toBeInTheDocument();
  });

  it("closes on Escape key", async () => {
    const user = userEvent.setup();
    render(<SecretsPanel onClose={onClose} />);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes on backdrop click", async () => {
    const user = userEvent.setup();
    render(<SecretsPanel onClose={onClose} />);
    // The backdrop has role="presentation"
    const backdrop = screen.getByRole("presentation");
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes on X button click", async () => {
    const user = userEvent.setup();
    render(<SecretsPanel onClose={onClose} />);
    const closeBtn = screen.getByLabelText("Close secrets panel");
    await user.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clears value field after successful submission (security contract)", async () => {
    const user = userEvent.setup();
    const createSecret = vi.fn().mockResolvedValue(undefined);
    useSecretsStore.setState({ createSecret });

    render(<SecretsPanel onClose={onClose} />);

    const nameInput = screen.getByLabelText("Name");
    const valueInput = screen.getByLabelText("Value");
    const submitBtn = screen.getByRole("button", { name: /save/i });

    await user.type(nameInput, "MY_KEY");
    await user.type(valueInput, "secret-value-123");
    await user.click(submitBtn);

    // After submission, both fields should be cleared
    expect(nameInput).toHaveValue("");
    expect(valueInput).toHaveValue("");
  });

  it("uses password input type for the value field", () => {
    render(<SecretsPanel onClose={onClose} />);
    const valueInput = screen.getByLabelText("Value");
    expect(valueInput).toHaveAttribute("type", "password");
  });
});
