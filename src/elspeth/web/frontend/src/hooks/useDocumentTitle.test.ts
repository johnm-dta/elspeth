import { describe, expect, it } from "vitest";
import { renderHook } from "@testing-library/react";
import {
  PRODUCT_TITLE,
  formatDocumentTitle,
  useDocumentTitle,
} from "./useDocumentTitle";

describe("formatDocumentTitle", () => {
  it("composes '{session} — ELSPETH' while a session is active", () => {
    expect(formatDocumentTitle("Weather Data Enrichment")).toBe(
      "Weather Data Enrichment — ELSPETH",
    );
  });

  it("falls back to the bare product name without a session", () => {
    expect(formatDocumentTitle(null)).toBe(PRODUCT_TITLE);
  });

  it("treats an empty title as no session (defensive)", () => {
    expect(formatDocumentTitle("")).toBe(PRODUCT_TITLE);
  });
});

describe("useDocumentTitle", () => {
  it("sets document.title on mount", () => {
    renderHook(() => useDocumentTitle("First — ELSPETH"));
    expect(document.title).toBe("First — ELSPETH");
  });

  it("updates document.title reactively on session switch and rename", () => {
    const { rerender } = renderHook(
      ({ title }: { title: string }) => useDocumentTitle(title),
      { initialProps: { title: formatDocumentTitle("First") } },
    );
    expect(document.title).toBe("First — ELSPETH");

    rerender({ title: formatDocumentTitle("Renamed") });
    expect(document.title).toBe("Renamed — ELSPETH");

    rerender({ title: formatDocumentTitle(null) });
    expect(document.title).toBe("ELSPETH");
  });
});
