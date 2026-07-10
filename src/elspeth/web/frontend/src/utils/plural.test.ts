import { describe, expect, it } from "vitest";

import { plural } from "./plural";

describe("plural", () => {
  it("renders the singular form for a count of 1", () => {
    expect(plural(1, "row")).toBe("1 row");
  });

  it("renders the default '+s' plural for counts other than 1", () => {
    expect(plural(0, "row")).toBe("0 rows");
    expect(plural(2, "row")).toBe("2 rows");
    expect(plural(-1, "row")).toBe("-1 rows");
  });

  it("supports an explicit plural label for irregular nouns", () => {
    expect(plural(1, "child", "children")).toBe("1 child");
    expect(plural(3, "child", "children")).toBe("3 children");
    expect(plural(0, "child", "children")).toBe("0 children");
  });

  it("matches every existing call site's output for representative inputs", () => {
    // ImportYamlModal.tsx pluraliseCount(count, singular)
    expect(plural(1, "source")).toBe("1 source");
    expect(plural(3, "source")).toBe("3 sources");
    // RecoveryDiff.tsx pluralize(count, noun)
    expect(plural(1, "addition")).toBe("1 addition");
    expect(plural(2, "addition")).toBe("2 additions");
    // InlineRunResults.tsx pluralRows(count) === plural(count, "row")
    expect(plural(1, "row")).toBe("1 row");
    expect(plural(5, "row")).toBe("5 rows");
  });
});
