import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  ComposerPreferencesForm,
  ComposerPreferencesPanel,
} from "./ComposerPreferencesPanel";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

// API mock — the real preferencesStore module imports these helpers and
// would receive undefined without the entries.
vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
}));

describe("ComposerPreferencesForm", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    // Reset sessionStore so activeSessionId is null and reads from
    // useSessionStore.getState() are predictable.
    useSessionStore.setState({ activeSessionId: null });
    vi.clearAllMocks();
  });

  it("renders the current default-mode selection (freeform)", () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/freeform/i)).toBeChecked();
    expect(screen.getByLabelText(/guided/i)).not.toBeChecked();
  });

  it("writes the new default-mode on selection (forwards activeSessionId watermark)", async () => {
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);

    render(<ComposerPreferencesForm />);
    await userEvent.click(screen.getByLabelText(/freeform/i));

    // Second arg is the activeSessionId watermark used by the
    // DefaultModeChangedBanner timing predicate. Null here because no
    // session is active in the test setup.
    expect(setDefault).toHaveBeenCalledWith("freeform", null);
  });

  it("forwards the active session id when one is set (banner timing watermark)", async () => {
    useSessionStore.setState({ activeSessionId: "sess-xyz" });
    const setDefault = vi
      .spyOn(usePreferencesStore.getState(), "setDefaultMode")
      .mockResolvedValueOnce(undefined);

    render(<ComposerPreferencesForm />);
    await userEvent.click(screen.getByLabelText(/freeform/i));

    expect(setDefault).toHaveBeenCalledWith("freeform", "sess-xyz");
  });

  it("disables inputs while writing", () => {
    usePreferencesStore.setState({ writing: true });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/guided/i)).toBeDisabled();
    expect(screen.getByLabelText(/freeform/i)).toBeDisabled();
  });

  it("returns null before preferences load (defaultMode null)", () => {
    usePreferencesStore.setState({ loaded: false, defaultMode: null });
    const { container } = render(<ComposerPreferencesForm />);
    // Component must gate on loaded === true; defaultMode is null pre-bootstrap.
    expect(container.firstChild).toBeNull();
  });

  it("surfaces writeError via role=alert when a PATCH fails (Panel a11y F2)", () => {
    usePreferencesStore.setState({
      writeError: "Couldn't save your preference: 503 Service Unavailable",
    });
    render(<ComposerPreferencesForm />);
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert).toHaveTextContent(/503 Service Unavailable/);
  });

  it("shows Reset tutorial after completion and calls resetTutorial", async () => {
    usePreferencesStore.setState({
      tutorialCompletedAt: "2026-05-19T12:00:00Z",
      tutorialCompleted: true,
    });
    const resetTutorial = vi
      .spyOn(usePreferencesStore.getState(), "resetTutorial")
      .mockResolvedValueOnce(undefined);

    render(<ComposerPreferencesForm />);
    await userEvent.click(screen.getByRole("button", { name: /reset tutorial/i }));

    expect(resetTutorial).toHaveBeenCalledTimes(1);
  });

  it("shows Reset tutorial while a tutorial is IN PROGRESS (the wedged-resume escape hatch)", () => {
    // Completion-gating hid the escape hatch from exactly the users who
    // needed it: a wedged mid-tutorial resume (persisted session swept out
    // from under it) left NO affordance anywhere — the tutorial suppresses
    // skip/exit past Welcome.
    usePreferencesStore.setState({
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      tutorialStage: "guided",
      tutorialSessionId: "sess-in-progress",
    });
    render(<ComposerPreferencesForm />);
    expect(
      screen.getByRole("button", { name: /reset tutorial/i }),
    ).toBeInTheDocument();
  });

  it("shows Reset tutorial for a fresh user too — the affordance is unconditional (operator requirement)", () => {
    render(<ComposerPreferencesForm />);
    expect(
      screen.getByRole("button", { name: /reset tutorial/i }),
    ).toBeInTheDocument();
  });
});

// ── Modal chrome (Panel test analyzer #2) ────────────────────────────────
//
// These tests cover the ComposerPreferencesPanel wrapper specifically, not
// the inner ComposerPreferencesForm. Until this round, the modal chrome
// (Escape, backdrop click, role=dialog, aria-labelledby) was only
// transitively covered by E2E — fast unit feedback was missing.
describe("ComposerPreferencesPanel — modal chrome", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      writing: false,
      writeError: null,
      optedOutAtSessionId: null,
    });
    useSessionStore.setState({ activeSessionId: null });
    vi.clearAllMocks();
  });

  it("renders as role=dialog with aria-modal and aria-labelledby pointing to the title", () => {
    render(<ComposerPreferencesPanel onClose={vi.fn()} />);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const titleId = dialog.getAttribute("aria-labelledby");
    expect(titleId).toBe("composer-preferences-title");
    // The element that owns the id must exist and carry the title text
    // — aria-labelledby is dead-pointer-prone otherwise.
    const title = document.getElementById(titleId!);
    expect(title).not.toBeNull();
    expect(title).toHaveTextContent(/composer preferences/i);
  });

  it("Escape calls onClose (modal dismissal contract)", async () => {
    const onClose = vi.fn();
    render(<ComposerPreferencesPanel onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("backdrop click calls onClose", async () => {
    const onClose = vi.fn();
    render(<ComposerPreferencesPanel onClose={onClose} />);
    // The backdrop is the role=presentation div sibling of the dialog.
    // queryAllByRole returns it; click the first match.
    const backdrop = document.querySelector('[role="presentation"]') as
      | HTMLElement
      | null;
    expect(backdrop).not.toBeNull();
    await userEvent.click(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("close button (×) calls onClose with accessible label", async () => {
    const onClose = vi.fn();
    render(<ComposerPreferencesPanel onClose={onClose} />);
    // Accessible name carries the panel context for screen readers
    // (rather than just "×" which would announce as "close" with no
    // owner).
    const closeBtn = screen.getByRole("button", {
      name: /close composer preferences panel/i,
    });
    expect(closeBtn).toBeInTheDocument();
    await userEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("embeds the inner ComposerPreferencesForm", () => {
    render(<ComposerPreferencesPanel onClose={vi.fn()} />);
    // The inner form's radios are present — the wrapper composes the
    // form correctly.
    expect(screen.getByLabelText(/guided/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/freeform/i)).toBeInTheDocument();
  });
});
