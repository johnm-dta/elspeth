// src/components/chat/ModelChip.tsx
//
// Persistent composer-model identity chip for the chat header
// (elspeth-e9f7678de8). ELSPETH is an auditability product whose runs record
// model identity — the authoring surface should show which model is doing the
// composing, not leave it discoverable only through an LLM-side list_models
// lookup.
//
// The model comes from GET /api/system/status (`composer_model`), the same
// deployment-level source App.tsx polls for the composer-availability banner.
// It is deployment configuration (ELSPETH_WEB__COMPOSER_MODEL), not per-turn
// state, so a single fetch on mount is sufficient. When the status endpoint
// is unreachable or reports no model, the chip renders nothing — absence of
// chrome, never a fabricated model name.

import { useEffect, useState } from "react";

import { fetchSystemStatus } from "@/api/client";

export function ModelChip() {
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await fetchSystemStatus();
        if (!cancelled && typeof status?.composer_model === "string" && status.composer_model.length > 0) {
          setModel(status.composer_model);
        }
      } catch {
        // Status endpoint unreachable — App.tsx's health banner owns that
        // failure story; the chip simply stays absent.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (model === null) return null;

  return (
    <span className="chat-model-chip" aria-label={`Composer model: ${model}`}>
      <span className="chat-model-chip-label" aria-hidden="true">
        Model:
      </span>{" "}
      {model}
    </span>
  );
}
