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
        {
          name: "OPENAI_API_KEY",
          scope: "user" as const,
          available: true,
          source_kind: "user",
          reason: null,
        },
        {
          name: "SERVER_KEY",
          scope: "server" as const,
          available: true,
          source_kind: "env",
          reason: null,
        },
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

  it("surfaces human-readable reasons for unavailable secrets", () => {
    useSecretsStore.setState({
      secrets: [
        {
          name: "OPENROUTER_API_KEY",
          scope: "server" as const,
          available: false,
          source_kind: "env",
          reason: "fingerprint_resolver_not_configured",
        },
        {
          name: "ANTHROPIC_API_KEY",
          scope: "server" as const,
          available: false,
          source_kind: "env",
          reason: "env_var_not_set",
        },
        {
          name: "BROKEN_USER_KEY",
          scope: "user" as const,
          available: false,
          source_kind: "user",
          reason: "value_decryption_failed",
        },
      ],
    });

    render(<SecretsPanel onClose={onClose} />);

    expect(
      screen.getByText("Fingerprint resolver is not configured"),
    ).toBeInTheDocument();
    expect(screen.getByText("Environment variable is not set")).toBeInTheDocument();
    expect(
      screen.getByText("Stored value could not be decrypted"),
    ).toBeInTheDocument();
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

  it("associates name-specific form errors with the name field", () => {
    useSecretsStore.setState({ error: "Secret name already exists." });

    render(<SecretsPanel onClose={onClose} />);

    const alert = screen.getByRole("alert");
    const nameInput = screen.getByLabelText("Name");
    const valueInput = screen.getByLabelText("Value");

    expect(alert).toHaveAttribute("id", "secret-form-error");
    expect(nameInput).toHaveAttribute("aria-invalid", "true");
    expect(nameInput).toHaveAttribute("aria-describedby", "secret-form-error");
    expect(valueInput).not.toHaveAttribute("aria-invalid");
    expect(valueInput).not.toHaveAttribute("aria-describedby");
  });

  describe("submit failure recovery", () => {
    it("re-enables the form after createSecret throws", async () => {
      const user = userEvent.setup();
      const failing = vi.fn(async () => {
        throw new Error("network");
      });
      useSecretsStore.setState({ createSecret: failing });

      render(<SecretsPanel onClose={onClose} />);

      await user.type(screen.getByLabelText("Name"), "OPENAI_API_KEY");
      await user.type(screen.getByLabelText("Value"), "sk-test");
      await user.click(screen.getByRole("button", { name: /save/i }));

      expect(failing).toHaveBeenCalledOnce();

      // After failure the fields are cleared (security contract) and
      // isSubmitting is reset to false. Re-filling the fields must
      // produce an enabled submit button — if isSubmitting were stuck
      // the button would remain disabled regardless of field content.
      await user.type(screen.getByLabelText("Name"), "ANOTHER_KEY");
      await user.type(screen.getByLabelText("Value"), "sk-other");
      expect(screen.getByRole("button", { name: /save/i })).not.toBeDisabled();
    });
  });
});
