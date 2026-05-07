import { describe, expect, it } from "vitest";
import {
  MIN_FUZZY_CONFIDENCE,
  confidenceFromScore,
  fuzzyMatch,
} from "./fuzzyScore";

describe("fuzzyMatch", () => {
  it("empty query matches anything with score 0", () => {
    expect(fuzzyMatch("", "csv_source")).toBe(0);
    expect(fuzzyMatch("", "")).toBe(0);
  });

  it("exact prefix scores 0", () => {
    expect(fuzzyMatch("csv", "csv_source")).toBe(0);
    expect(fuzzyMatch("ab", "ab")).toBe(0);
  });

  it("returns -1 when any query char is missing", () => {
    expect(fuzzyMatch("xyz", "csv_source")).toBe(-1);
    expect(fuzzyMatch("xq", "abc")).toBe(-1);
  });

  it("is case-insensitive", () => {
    expect(fuzzyMatch("CSV", "csv_source")).toBe(0);
    expect(fuzzyMatch("csv", "CSV_SOURCE")).toBe(0);
  });

  it("ranks exact-prefix above scattered matches", () => {
    const exact = fuzzyMatch("csv", "csv_source");
    const scattered = fuzzyMatch("csv", "console_set_value");
    expect(exact).toBeLessThan(scattered);
  });

  it("scores the issue's named false-positive with a non-zero penalty", () => {
    // The bug: "ab" matched "azzbzz" with no penalty before this fix.
    // After the fix, the score is non-zero and the confidence is below 1.
    const score = fuzzyMatch("ab", "azzbzz");
    expect(score).toBeGreaterThan(0);
    expect(confidenceFromScore(score, "azzbzz".length)).toBeLessThan(1);
  });
});

describe("confidenceFromScore", () => {
  it("perfect score yields confidence 1.0", () => {
    expect(confidenceFromScore(0, 10)).toBe(1);
  });

  it("no-match (-1) yields confidence 0", () => {
    expect(confidenceFromScore(-1, 10)).toBe(0);
  });

  it("worst-case scatter falls below MIN_FUZZY_CONFIDENCE", () => {
    // Query "abcde" scattered across "azzbzzczzdzze" (length 13) — every
    // matched char is preceded by gap chars. Confidence must be below the
    // configured noise floor so it gets trimmed by the catalog.
    const target = "azzbzzczzdzze";
    const score = fuzzyMatch("abcde", target);
    expect(score).toBeGreaterThan(0);
    const confidence = confidenceFromScore(score, target.length);
    expect(confidence).toBeLessThan(MIN_FUZZY_CONFIDENCE);
  });

  it("legitimate end-of-string match stays above MIN_FUZZY_CONFIDENCE", () => {
    // Query "csv" inside "azure_csv" — match is at the end, so score is
    // moderate, but a user typing "csv" should still see this result.
    const target = "azure_csv";
    const score = fuzzyMatch("csv", target);
    const confidence = confidenceFromScore(score, target.length);
    expect(confidence).toBeGreaterThan(MIN_FUZZY_CONFIDENCE);
  });

  it("clamps to 0 when score exceeds target length", () => {
    expect(confidenceFromScore(100, 10)).toBe(0);
  });

  it("treats empty target sensibly (score 0 → 1, otherwise 0)", () => {
    expect(confidenceFromScore(0, 0)).toBe(1);
    expect(confidenceFromScore(5, 0)).toBe(0);
  });
});
