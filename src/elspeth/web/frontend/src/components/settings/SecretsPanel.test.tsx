import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within, waitFor } from "@testing-library/react";
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

  it("uses theme token references without hard-coded fallback colours", () => {
    useSecretsStore.setState({
      secrets: [
        {
          name: "SET_KEY",
          scope: "user" as const,
          available: true,
          source_kind: "user",
          reason: null,
        },
        {
          name: "MISSING_KEY",
          scope: "server" as const,
          available: false,
          source_kind: "env",
          reason: "env_var_not_set",
        },
      ],
    });

    render(<SecretsPanel onClose={onClose} />);

    const dialogStyle = screen.getByRole("dialog").getAttribute("style");
    const availableStyle = screen.getByRole("img", { name: "Available" }).getAttribute("style");
    const unavailableStyle = screen.getByRole("img", { name: "Unavailable" }).getAttribute("style");

    expect(dialogStyle).toContain("var(--color-surface)");
    expect(availableStyle).toContain("var(--color-success)");
    expect(availableStyle).toContain("var(--color-success-bg)");
    expect(unavailableStyle).toContain("var(--color-text-muted)");
    expect(`${dialogStyle} ${availableStyle} ${unavailableStyle}`).not.toMatch(
      /#16a34a|#9ca3af|#fff/i,
    );
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
    it("preserves the name and re-enables the form after createSecret throws", async () => {
      const user = userEvent.setup();
      const failing = vi.fn(async () => {
        throw new Error("network");
      });
      useSecretsStore.setState({ createSecret: failing });

      render(<SecretsPanel onClose={onClose} />);

      const nameInput = screen.getByLabelText("Name");
      const valueInput = screen.getByLabelText("Value");

      await user.type(nameInput, "OPENAI_API_KEY");
      await user.type(valueInput, "sk-test");
      await user.click(screen.getByRole("button", { name: /save/i }));

      expect(failing).toHaveBeenCalledOnce();

      // WCAG 3.3.7: even when the store contract is violated (a thrown error
      // rather than a recorded one), the name survives so the user can retry
      // without re-typing. SECURITY: the value is always cleared.
      expect(nameInput).toHaveValue("OPENAI_API_KEY");
      expect(valueInput).toHaveValue("");

      // isSubmitting must have reset — re-filling the value re-enables submit.
      // (If isSubmitting were stuck the button would stay disabled.)
      await user.type(valueInput, "sk-retry");
      expect(screen.getByRole("button", { name: /save/i })).not.toBeDisabled();
    });

    it("preserves the name and keeps the inline error on it when the store records a failure (WCAG 3.3.7)", async () => {
      const user = userEvent.setup();
      // Mirror the real store contract: createSecret swallows the error and
      // records it on the store rather than re-throwing.
      const failing = vi.fn(async () => {
        useSecretsStore.setState({ error: "Failed to save secret." });
      });
      useSecretsStore.setState({ createSecret: failing });

      render(<SecretsPanel onClose={onClose} />);

      const nameInput = screen.getByLabelText("Name");
      const valueInput = screen.getByLabelText("Value");

      await user.type(nameInput, "OPENAI_API_KEY");
      await user.type(valueInput, "sk-test");
      await user.click(screen.getByRole("button", { name: /save/i }));

      // WCAG 3.3.7: the name persists so the inline error keeps describing a
      // populated field; SECURITY: the value is still wiped.
      expect(nameInput).toHaveValue("OPENAI_API_KEY");
      expect(valueInput).toHaveValue("");
      await waitFor(() => {
        expect(nameInput).toHaveAttribute("aria-invalid", "true");
      });
      expect(nameInput).toHaveAttribute("aria-describedby", "secret-form-error");
    });
  });

  describe("delete confirmation (WCAG 3.3.4)", () => {
    it("does not delete immediately — it opens a danger confirm dialog naming the secret", async () => {
      const user = userEvent.setup();
      const deleteSecret = vi.fn().mockResolvedValue(undefined);
      useSecretsStore.setState({ deleteSecret });

      render(<SecretsPanel onClose={onClose} />);

      await user.click(screen.getByLabelText("Delete secret OPENAI_API_KEY"));

      // The destructive action is staged, not fired.
      expect(deleteSecret).not.toHaveBeenCalled();
      expect(screen.getByRole("alertdialog")).toBeInTheDocument();
      expect(
        screen.getByText(/Delete secret "OPENAI_API_KEY"\?/),
      ).toBeInTheDocument();
    });

    it("deletes the named secret only after confirming", async () => {
      const user = userEvent.setup();
      const deleteSecret = vi.fn().mockResolvedValue(undefined);
      useSecretsStore.setState({ deleteSecret });

      render(<SecretsPanel onClose={onClose} />);

      await user.click(screen.getByLabelText("Delete secret OPENAI_API_KEY"));
      const dialog = screen.getByRole("alertdialog");
      await user.click(within(dialog).getByRole("button", { name: "Delete secret" }));

      expect(deleteSecret).toHaveBeenCalledTimes(1);
      expect(deleteSecret).toHaveBeenCalledWith("OPENAI_API_KEY");
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    });

    it("leaves the secret intact when the dialog is cancelled", async () => {
      const user = userEvent.setup();
      const deleteSecret = vi.fn().mockResolvedValue(undefined);
      useSecretsStore.setState({ deleteSecret });

      render(<SecretsPanel onClose={onClose} />);

      await user.click(screen.getByLabelText("Delete secret OPENAI_API_KEY"));
      const dialog = screen.getByRole("alertdialog");
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

      expect(deleteSecret).not.toHaveBeenCalled();
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    });

    it("Escape cancels the pending delete without tearing down the panel", async () => {
      const user = userEvent.setup();
      const deleteSecret = vi.fn().mockResolvedValue(undefined);
      useSecretsStore.setState({ deleteSecret });

      render(<SecretsPanel onClose={onClose} />);

      await user.click(screen.getByLabelText("Delete secret OPENAI_API_KEY"));
      expect(screen.getByRole("alertdialog")).toBeInTheDocument();

      await user.keyboard("{Escape}");

      // The confirm dialog owns Escape: it closes, the delete never fires, and
      // the surrounding panel stays open (onClose must NOT have been called).
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
      expect(deleteSecret).not.toHaveBeenCalled();
      expect(onClose).not.toHaveBeenCalled();
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    it("only user-scoped secrets expose a delete control", () => {
      render(<SecretsPanel onClose={onClose} />);
      // OPENAI_API_KEY is user-scoped → deletable; SERVER_KEY is server-scoped → not.
      expect(
        screen.getByLabelText("Delete secret OPENAI_API_KEY"),
      ).toBeInTheDocument();
      expect(
        screen.queryByLabelText("Delete secret SERVER_KEY"),
      ).not.toBeInTheDocument();
    });
  });
});
