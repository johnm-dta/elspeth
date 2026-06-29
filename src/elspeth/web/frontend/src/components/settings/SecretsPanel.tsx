// src/components/settings/SecretsPanel.tsx
import { useState, useEffect, useCallback, useRef } from "react";
import { useSecretsStore } from "@/stores/secretsStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import type { SecretInventoryItem } from "@/types/api";

interface SecretsPanelProps {
  onClose: () => void;
}

const SECRET_FORM_ERROR_ID = "secret-form-error";

interface SecretFormErrorTargets {
  name: boolean;
  value: boolean;
}

function ScopeBadge({ scope }: { scope: SecretInventoryItem["scope"] }) {
  const colors: Record<SecretInventoryItem["scope"], { bg: string; text: string }> = {
    user: { bg: "var(--color-accent-muted)", text: "var(--color-accent)" },
    server: { bg: "var(--color-info-bg)", text: "var(--color-info)" },
    org: { bg: "var(--color-surface-raised)", text: "var(--color-text-secondary)" },
  };
  const { bg, text } = colors[scope] ?? colors.org;
  return (
    <span
      className="secrets-scope-badge"
      style={{ backgroundColor: bg, color: text }}
    >
      {scope}
    </span>
  );
}

function AvailabilityDot({ available }: { available: boolean }) {
  // Two orthogonal cues so the on/off distinction survives a monochromatic
  // palette and colour-vision deficiency:
  //   - filled disc with a soft halo  → "lit"
  //   - hollow ring on transparent bg → "off"
  return (
    <span
      role="img"
      aria-label={available ? "Available" : "Unavailable"}
      title={available ? "Available" : "Not set"}
      style={{
        display: "inline-block",
        width: 12,
        height: 12,
        borderRadius: "50%",
        boxSizing: "border-box",
        backgroundColor: available
          ? "var(--color-success)"
          : "transparent",
        border: available
          ? "1px solid var(--color-success)"
          : "1.5px solid var(--color-text-muted)",
        boxShadow: available
          ? "0 0 0 2px var(--color-success-bg)"
          : "none",
        flexShrink: 0,
      }}
    />
  );
}

function reasonLabel(reason: SecretInventoryItem["reason"]): string | null {
  if (reason === null) return null;
  const labels: Record<NonNullable<SecretInventoryItem["reason"]>, string> = {
    fingerprint_resolver_not_configured: "Fingerprint resolver is not configured",
    env_var_not_set: "Environment variable is not set",
    value_decryption_failed: "Stored value could not be decrypted",
  };
  return labels[reason];
}

function secretFormErrorTargets(error: string | null): SecretFormErrorTargets {
  if (!error) {
    return { name: false, value: false };
  }

  const normalized = error.toLowerCase();
  if (
    normalized.includes("secret name") ||
    normalized.includes("name already exists")
  ) {
    return { name: true, value: false };
  }
  if (normalized.includes("secret value")) {
    return { name: false, value: true };
  }
  if (normalized.includes("failed to save secret")) {
    return { name: true, value: true };
  }

  return { name: false, value: false };
}

/**
 * Secrets settings panel — modal overlay.
 *
 * Write-only entry form for user-scoped secrets plus an inventory display
 * showing all available secret references (metadata only, never values).
 *
 * SECURITY:
 * - Value input uses type="password" — no browser autocomplete for the secret value.
 * - Value field is cleared immediately after submission, whether the API call succeeds or fails.
 * - The store never retains the value after the API call completes.
 * - No "show password" toggle is provided.
 */
export function SecretsPanel({ onClose }: SecretsPanelProps) {
  const { secrets, isLoading, error, loadSecrets, createSecret, deleteSecret } =
    useSecretsStore();
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Name of the secret awaiting a delete confirmation (null = no pending delete).
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const formErrorTargets = secretFormErrorTargets(error);
  const hasFormErrorTarget = formErrorTargets.name || formErrorTargets.value;
  useFocusTrap(modalRef, true, "#secret-name");

  useEffect(() => {
    loadSecrets();
  }, [loadSecrets]);

  // Close on Escape key. While the delete-confirm dialog is open it owns Escape
  // (it cancels the pending delete); don't also tear down the whole panel.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && pendingDelete === null) onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, pendingDelete]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!name.trim() || !value) return;
      setIsSubmitting(true);
      let succeeded = false;
      try {
        await createSecret(name.trim(), value);
        // createSecret does not re-throw — it reports failure via state.error
        // (and clears it at the start of each attempt). Treat the save as
        // successful only when no error was recorded.
        succeeded = !useSecretsStore.getState().error;
      } catch {
        // Defensive: if the store contract changes to re-throw, fall through
        // with succeeded=false so the name is preserved for retry.
      } finally {
        // SECURITY: clear value immediately — it must never linger in component
        // state regardless of whether the API call succeeded or failed.
        setValue("");
        // WCAG 3.3.7: keep the name on failure so the user can correct and retry
        // without re-typing, and so the inline error (which targets the name
        // field) keeps describing a populated input. Clear it only on success.
        if (succeeded) {
          setName("");
        }
        setIsSubmitting(false);
      }
    },
    [name, value, createSecret],
  );

  // WCAG 3.3.4: deleting a credential is irreversible and breaks every pipeline
  // referencing it, so it goes through a danger confirmation rather than firing
  // straight from the "×" button.
  const confirmDelete = useCallback(() => {
    if (pendingDelete !== null) {
      deleteSecret(pendingDelete);
    }
    setPendingDelete(null);
  }, [pendingDelete, deleteSecret]);

  return (
    <>
      {/* Backdrop */}
      <div
        role="presentation"
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.45)",
          zIndex: 100,
        }}
      />

      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-label="Secrets settings"
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 101,
          width: 480,
          maxWidth: "calc(100vw - 32px)",
          maxHeight: "calc(100vh - 64px)",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "var(--color-surface)",
          borderRadius: 8,
          boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
          border: "1px solid var(--color-border)",
          fontSize: 13,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div className="secrets-panel-header">
          <h2 className="secrets-panel-title">
            API Keys &amp; Secrets
          </h2>
          <button
            onClick={onClose}
            aria-label="Close secrets panel"
            className="secrets-panel-close"
          >
            ×
          </button>
        </div>

        {/* Scrollable body */}
        <div className="secrets-panel-body">
          {/* Entry form */}
          <section aria-labelledby="secrets-add-heading">
            <h3
              id="secrets-add-heading"
              className="secrets-section-heading"
            >
              Add or update a secret
            </h3>
            <form onSubmit={handleSubmit} noValidate>
              <div className="secrets-form-fields">
                <div>
                  <label
                    htmlFor="secret-name"
                    className="secrets-form-label"
                  >
                    Name
                  </label>
                  <input
                    id="secret-name"
                    type="text"
                    aria-invalid={formErrorTargets.name ? true : undefined}
                    aria-describedby={
                      formErrorTargets.name ? SECRET_FORM_ERROR_ID : undefined
                    }
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="e.g. OPENAI_API_KEY"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="secrets-form-input"
                  />
                </div>
                <div>
                  <label
                    htmlFor="secret-value"
                    className="secrets-form-label"
                  >
                    Value
                  </label>
                  {/* SECURITY: type="password" — value never displayed in plaintext.
                      No "show" toggle is intentional. */}
                  <input
                    id="secret-value"
                    type="password"
                    aria-invalid={formErrorTargets.value ? true : undefined}
                    aria-describedby={
                      formErrorTargets.value ? SECRET_FORM_ERROR_ID : undefined
                    }
                    autoComplete="new-password"
                    placeholder="Paste your secret value here"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    className="secrets-form-input"
                  />
                </div>
                <button
                  type="submit"
                  disabled={!name.trim() || !value || isSubmitting}
                  className="btn btn-primary secrets-submit-btn"
                >
                  {isSubmitting ? "Saving…" : "Save secret"}
                </button>
              </div>
            </form>
          </section>

          {/* Error banner */}
          {error && (
            <div
              id={hasFormErrorTarget ? SECRET_FORM_ERROR_ID : undefined}
              role="alert"
              style={{
                marginTop: 12,
                padding: "6px 10px",
                borderRadius: 4,
                backgroundColor: "var(--color-error-bg)",
                color: "var(--color-error)",
                fontSize: 12,
              }}
            >
              {error}
            </div>
          )}

          {/* Inventory */}
          <section aria-labelledby="secrets-inventory-heading" style={{ marginTop: 20 }}>
            <h3
              id="secrets-inventory-heading"
              className="secrets-section-heading"
            >
              Secret inventory
            </h3>

            {isLoading ? (
              <div
                role="status"
                aria-live="polite"
                className="secrets-loading"
              >
                Loading…
              </div>
            ) : secrets.length === 0 ? (
              <div className="secrets-empty">
                No secrets configured. Add one above.
              </div>
            ) : (
              <ul role="list" className="secrets-list">
                {[...secrets]
                  .sort((a, b) => {
                    // Available first, then alphabetical by name.
                    if (a.available !== b.available) return a.available ? -1 : 1;
                    return a.name.localeCompare(b.name);
                  })
                  .map((secret) => {
                  const unavailableReason = reasonLabel(secret.reason);
                  return (
                    <li key={secret.name} className="secrets-list-item">
                      <AvailabilityDot available={secret.available} />

                      <span className="secrets-list-detail">
                        <span className="secrets-list-name">
                          {secret.name}
                        </span>
                        {!secret.available && unavailableReason && (
                          <span className="secrets-unavailable-reason">
                            {unavailableReason}
                          </span>
                        )}
                      </span>

                      <ScopeBadge scope={secret.scope} />

                      {/* Server-scoped and org-scoped secrets are read-only — no delete */}
                      {secret.scope === "user" && (
                        <button
                          onClick={() => setPendingDelete(secret.name)}
                          aria-label={`Delete secret ${secret.name}`}
                          title="Delete"
                          className="secrets-delete-btn"
                        >
                          ×
                        </button>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <p className="secrets-footnote">
            Secrets are encrypted at rest. Values are never shown after saving.
            Server-scoped secrets are configured by an administrator and cannot
            be deleted here.
          </p>
        </div>
      </div>

      {/* WCAG 3.3.4: irreversible delete is gated behind a danger confirmation. */}
      {pendingDelete !== null && (
        <ConfirmDialog
          title="Delete this secret?"
          message={`Delete secret "${pendingDelete}"? Pipelines that reference it will fail.`}
          confirmLabel="Delete secret"
          cancelLabel="Cancel"
          variant="danger"
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
