// Focus-management contract follows the project's modal-dialog pattern — see CommandPalette.tsx and RecoveryPanel.tsx for canonical examples.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { ExplainDialog } from "./ExplainDialog";
import { useAuditReadinessStore, getInitialState } from "../../stores/auditReadinessStore";
import * as api from "../../api/auditReadiness";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

describe("ExplainDialog", () => {
  beforeEach(() => {
    // Canonical reset: getInitialState() returns the per-session keyed shape
    // (snapshotsBySession / explainsBySession / isLoadingBySession /
    // isLoadingExplainBySession / errorBySession / explainErrorBySession).
    // DO NOT hand-roll a setState literal with `as never` — the store has no
    // flat `isLoading` / `error` / `isLoadingExplain` / `explainError` fields,
    // and `as never` would silently mask the drift.
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("fetches the narrative on mount and renders it", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record:\n\n• Source data — 5 URLs.",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByText(/ELSPETH will record/)).toBeInTheDocument();
    expect(screen.getByText(/Source data — 5 URLs/)).toBeInTheDocument();
  });

  it("uses the cached narrative when present without refetching", async () => {
    useAuditReadinessStore.setState({
      explainsBySession: {
        [SESSION_ID]: {
          session_id: SESSION_ID,
          composition_version: 1,
          narrative: "cached narrative",
        },
      },
    } as never);
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    // Confirm the render has settled before asserting the API was not called.
    // A bare `not.toHaveBeenCalled()` check before render settles races the
    // useEffect; wait for the text to appear first.
    await waitFor(() => expect(screen.getByText("cached narrative")).toBeInTheDocument());
    expect(api.fetchAuditReadinessExplain).not.toHaveBeenCalled();
  });

  it("does not render a cached narrative when its version differs from the requested version", async () => {
    useAuditReadinessStore.setState({
      explainsBySession: {
        [SESSION_ID]: {
          session_id: SESSION_ID,
          composition_version: 1,
          narrative: "stale v1 narrative",
        },
      },
    } as never);
    vi.mocked(api.fetchAuditReadinessExplain).mockReturnValueOnce(
      new Promise(() => {}),
    );

    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={2}
        onClose={() => {}}
      />,
    );

    await screen.findByText(/Generating explanation/i);
    expect(screen.queryByText("stale v1 narrative")).not.toBeInTheDocument();
  });

  it("renders a loading state while the fetch is pending, then transitions to content on resolve", async () => {
    let resolve!: (v: { session_id: string; composition_version: number; narrative: string }) => void;
    vi.mocked(api.fetchAuditReadinessExplain).mockReturnValueOnce(
      new Promise((r) => {
        resolve = r;
      }),
    );
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );

    // Wait for the loadExplain effect to fire and the store to flip
    // isLoadingExplainBySession[SESSION_ID] = true before asserting the
    // loading indicator. A bare synchronous assertion races the effect.
    await waitFor(() =>
      expect(screen.getByText(/Generating explanation/i)).toBeInTheDocument(),
    );

    resolve({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "done.",
    });

    // Confirm the transition from loading to content actually happens —
    // without this waitFor, the test passes regardless of whether the
    // post-resolve render works correctly.
    await waitFor(() =>
      expect(screen.getByText("done.")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Generating explanation/i)).not.toBeInTheDocument();
  });

  it("renders an error when the fetch fails", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockRejectedValueOnce({
      status: 500,
      detail: "boom",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/boom/);
  });

  it("fires onClose when Close is clicked", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("uses role=dialog and is labelled by the heading", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const labelId = dialog.getAttribute("aria-labelledby")!;
    expect(document.getElementById(labelId)).toHaveTextContent(/What this pipeline will record/i);
  });

  it("moves focus into the dialog on mount (focus-trap contract)", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    await waitFor(() => {
      expect(dialog.contains(document.activeElement)).toBe(true);
    });
  });

  it("fires onClose when Escape is pressed", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("restores focus to the opener element when the dialog is closed", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });

    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open explain
          </button>
          {open ? (
            <ExplainDialog
              sessionId={SESSION_ID}
              compositionVersion={1}
              onClose={() => setOpen(false)}
            />
          ) : null}
        </>
      );
    }

    const user = userEvent.setup();
    render(<Harness />);
    const opener = screen.getByRole("button", { name: "Open explain" });
    opener.focus();
    await user.click(opener);
    // Dialog is open — confirm focus is inside it
    const dialog = screen.getByRole("dialog");
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true));
    // Close via Escape
    await user.keyboard("{Escape}");
    // Focus must return to the opener
    await waitFor(() => expect(opener).toHaveFocus());
  });

  // W6 — added 2026-05-17

  it("refetches when compositionVersion changes", async () => {
    // Cache semantics: loadExplain caches by (sessionId, composition_version).
    // The short-circuit is: if cached.composition_version === compositionVersion, return.
    // Therefore a version change 1 → 2 misses the cache and a new fetch fires.
    vi.mocked(api.fetchAuditReadinessExplain)
      .mockResolvedValueOnce({
        session_id: SESSION_ID,
        composition_version: 1,
        narrative: "v1 narrative",
      })
      .mockResolvedValueOnce({
        session_id: SESSION_ID,
        composition_version: 2,
        narrative: "v2 narrative",
      });

    const { rerender } = render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    await screen.findByText("v1 narrative");
    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(1);

    rerender(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={2}
        onClose={() => {}}
      />,
    );
    await screen.findByText("v2 narrative");
    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(2);
  });

  it("closes when the backdrop is clicked", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    // The backdrop is aria-hidden="true" (decorative) so we query by class.
    // The implementation attaches onClick={onClose} to the backdrop div.
    // If the implementation changes to a different backdrop pattern, update this selector.
    const backdrop = document.querySelector(".explain-dialog-backdrop") as HTMLElement;
    expect(backdrop).not.toBeNull();
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders the backdrop and the dialog as siblings, not as parent/child (P0.5)", async () => {
    // P0.5: the prior version nested .explain-dialog-backdrop inside
    // the role="dialog" div. That confused the modal's a11y tree —
    // some screen readers infer the dialog boundary from the
    // containing element, not from role. The new pattern matches
    // SecretsPanel: backdrop and dialog are siblings inside a
    // fragment. A regression that re-nests them would silently
    // re-introduce the a11y drift; this test pins the structure.
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    await screen.findByText("x");
    const backdrop = document.querySelector(
      ".explain-dialog-backdrop",
    ) as HTMLElement;
    const dialog = screen.getByRole("dialog");
    expect(backdrop).not.toBeNull();
    // Sibling check: backdrop's parent must NOT be the dialog div
    // (the old, broken structure). Both backdrop and dialog must
    // share the same parent — RTL renders the fragment's children
    // directly into the test container.
    expect(backdrop.parentElement).not.toBe(dialog);
    expect(backdrop.parentElement).toBe(dialog.parentElement);
  });

  it("closes when Escape is pressed even if focus is not inside the dialog (P0.5 — document-level listener)", async () => {
    // P0.5: the prior Escape handler was bound via onKeyDown on the
    // dialog div, which only fires if focus is inside the dialog
    // tree. After the restructure Escape is registered on document,
    // so an Escape press while focus has drifted to <body> still
    // closes the dialog. This guards against focus-trap escape bugs
    // in nested content.
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    // Force focus OUTSIDE the dialog to simulate a focus-trap break.
    document.body.focus();
    // Use a raw KeyboardEvent on document — userEvent.keyboard
    // dispatches via the active element, which the prior onKeyDown
    // implementation would have caught only inside the dialog.
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("renders a fallback error message when the ApiError has no detail", async () => {
    // Fallback text: auditReadinessStore.ts → apiErr.detail ?? "Failed to load the explain narrative."
    vi.mocked(api.fetchAuditReadinessExplain).mockRejectedValueOnce({ status: 500 });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Failed to load the explain narrative.");
  });
});
