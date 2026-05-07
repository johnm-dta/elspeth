// src/components/settings/SecretsPanel.tsx
import { useState, useEffect, useCallback, useRef } from "react";
import { useSecretsStore } from "@/stores/secretsStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import type { SecretInventoryItem } from "@/types/api";

interface SecretsPanelProps {
  onClose: () => void;
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
  return (
    <span
      aria-label={available ? "Available" : "Unavailable"}
      title={available ? "Available" : "Not set"}
      style={{
        display: "inline-block",
        width: 12,
        height: 12,
        borderRadius: "50%",
        backgroundColor: available
          ? "var(--color-success, #16a34a)"
          : "var(--color-text-muted, #9ca3af)",
        flexShrink: 0,
      }}
    />
  );
}

/**
 * Secrets settings panel — modal overlay.
 *
 * Write-only entry form for user-scoped secrets plus an inventory display
 * showing all available secret references (metadata only, never values).
 *
 * SECURITY:
 * - Value input uses type="password" — no browser autocomplete for the secret value.
 * - Value field is cleared immediately after successful submission.
 * - The store never retains the value after the API call completes.
 * - No "show password" toggle is provided.
 */
export function SecretsPanel({ onClose }: SecretsPanelProps) {
  const { secrets, isLoading, error, loadSecrets, createSecret, deleteSecret } =
    useSecretsStore();
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);
  useFocusTrap(modalRef, true, "#secret-name");

  useEffect(() => {
    loadSecrets();
  }, [loadSecrets]);

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!name.trim() || !value) return;
      setIsSubmitting(true);
      await createSecret(name.trim(), value);
      // SECURITY: clear value immediately after submission — it must never
      // linger in component state or the store after the API call.
      setValue("");
      setName("");
      setIsSubmitting(false);
    },
    [name, value, createSecret],
  );

  const handleDelete = useCallback(
    (secretName: string) => {
      deleteSecret(secretName);
    },
    [deleteSecret],
  );

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
          backgroundColor: "var(--color-surface, #fff)",
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
                {secrets.map((secret) => (
                  <li key={secret.name} className="secrets-list-item">
                    <AvailabilityDot available={secret.available} />

                    <span className="secrets-list-name">
                      {secret.name}
                    </span>

                    <ScopeBadge scope={secret.scope} />

                    {/* Server-scoped and org-scoped secrets are read-only — no delete */}
                    {secret.scope === "user" && (
                      <button
                        onClick={() => handleDelete(secret.name)}
                        aria-label={`Delete secret ${secret.name}`}
                        title="Delete"
                        className="secrets-delete-btn"
                      >
                        ×
                      </button>
                    )}
                  </li>
                ))}
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
    </>
  );
}
