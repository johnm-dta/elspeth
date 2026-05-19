// ============================================================================
// InlineSourceCreatedTurn.test.tsx — Phase 5a Task 3
//
// Surfaces a reviewable confirmation widget after an inline-blob source is
// attached to the composition. The widget is INFORMATIONAL (not a
// pending-approval surface): it shows what was created and, for LLM-authored
// content, exposes an "Edit the list" affordance that lets the user amend the
// generated rows.
//
// Test coverage mirrors the six concerns from Phase 5a Task 3:
//   1. Visible filename, MIME type, and row-count for a verbatim source.
//   2. Edit affordance HIDDEN for verbatim provenance (no user authorship to
//      amend).
//   3. Edit affordance VISIBLE and wired for llm-generated provenance.
//   4. SHA-256 hash hidden inside a collapsed audit-info disclosure (F-21:
//      keep noisy provenance out of the default reading order).
//   5. Content preview clipped at 280 chars (F-22: no full-payload leak into
//      the DOM; the underlying blob bytes live behind an authenticated
//      endpoint, not in the widget tree).
//   6. Root element announces itself via role="region" with an accessible
//      name (F-18: assistive-tech can navigate to "Source created" without
//      reading every adjacent widget).
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InlineSourceCreatedTurn } from "./InlineSourceCreatedTurn";
import type { InlineSourceSummary } from "@/types/api";

describe("InlineSourceCreatedTurn", () => {
  // Realistic SHA-256 prefix.  We use a longer, hex-shaped string (instead
  // of the earlier "h1" placeholder) so the regex used in the audit-info
  // disclosure test (`/abc123def456789/`) can't false-positive on
  // unrelated DOM text.
  const verbatim: InlineSourceSummary = {
    blobId: "b1",
    filename: "chat.csv",
    mimeType: "text/csv",
    contentPreview: "url\nhttps://finance.gov.au",
    rowCount: 1,
    contentHash: "abc123def456789",
    provenance: "verbatim",
  };

  const llmGenerated: InlineSourceSummary = {
    ...verbatim,
    blobId: "b2",
    contentPreview: "url\n...gov.au\n...gov.au\n...gov.au\n...gov.au\n...gov.au",
    rowCount: 5,
    provenance: "llm-generated",
  };

  it("renders the filename, MIME type, and row count for a verbatim source", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(
      screen.queryByRole("heading", { name: "Source file" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/chat\.csv/)).toBeInTheDocument();
    expect(screen.getByText(/text\/csv/)).toBeInTheDocument();
    expect(screen.getByText(/1 row/)).toBeInTheDocument();
  });

  it("labels the inline preview as content before rendering source bytes", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(
      screen.getByRole("heading", { name: /source contents/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("inline-source-preview")).toHaveTextContent(
      "https://finance.gov.au",
    );
  });

  it("does NOT show an Edit button for verbatim provenance", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(
      screen.queryByRole("button", { name: /edit the list/i }),
    ).toBeNull();
  });

  it("DOES show an Edit button for llm-generated provenance", () => {
    const onEdit = vi.fn();
    render(<InlineSourceCreatedTurn summary={llmGenerated} onEdit={onEdit} />);
    const button = screen.getByRole("button", { name: /edit the list/i });
    fireEvent.click(button);
    expect(onEdit).toHaveBeenCalledWith(llmGenerated);
  });

  it("renders the SHA-256 hash inside a collapsed audit-info disclosure (F-21)", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    const disclosure = screen.getByText(/show audit info/i);
    expect(disclosure).toBeInTheDocument();
    expect(screen.queryByText(/abc123def456789/)).not.toBeInTheDocument();
    fireEvent.click(disclosure);
    expect(screen.getByText(/abc123def456789/)).toBeInTheDocument();
  });

  it("renders the content preview clipped (no full-payload leak in DOM)", () => {
    const huge: InlineSourceSummary = {
      ...verbatim,
      contentPreview: "x".repeat(500),
    };
    render(<InlineSourceCreatedTurn summary={huge} onEdit={vi.fn()} />);
    const preview = screen.getByTestId("inline-source-preview");
    expect(preview.textContent?.length).toBeLessThanOrEqual(280);
  });

  it("announces itself via role=region with aria-label (F-18)", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(
      screen.getByRole("region", { name: /source created/i }),
    ).toBeInTheDocument();
  });

  it("shows an Edit button for llm-generated-then-amended provenance (F-4 re-edit)", () => {
    const amended: InlineSourceSummary = {
      ...verbatim,
      provenance: "llm-generated-then-amended",
    };
    render(<InlineSourceCreatedTurn summary={amended} onEdit={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /edit the list/i }),
    ).toBeInTheDocument();
  });

  it("does NOT show an Edit button for disambiguated provenance", () => {
    const disambiguated: InlineSourceSummary = {
      ...verbatim,
      provenance: "disambiguated",
    };
    render(<InlineSourceCreatedTurn summary={disambiguated} onEdit={vi.fn()} />);
    expect(
      screen.queryByRole("button", { name: /edit the list/i }),
    ).toBeNull();
  });

  it("renders 'N rows' (plural) for rowCount > 1", () => {
    render(<InlineSourceCreatedTurn summary={llmGenerated} onEdit={vi.fn()} />);
    expect(screen.getByText(/5 rows/)).toBeInTheDocument();
  });

  it("renders 'unknown row count' when rowCount is null", () => {
    const unknown: InlineSourceSummary = { ...verbatim, rowCount: null };
    render(<InlineSourceCreatedTurn summary={unknown} onEdit={vi.fn()} />);
    expect(screen.getByText(/unknown row count/i)).toBeInTheDocument();
  });
});
