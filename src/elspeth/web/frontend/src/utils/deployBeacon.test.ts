// Deploy-cache coherence beacon (operator-hit: after a rebuild, an open tab
// kept the previous bundle — lazy chunks 404'd against the wiped dist and
// newly shipped features were silently absent, reading as a product hang).
// The tab compares its own entry-script identity against the polled
// frontend_build and offers a refresh once the mismatch is STABLE.

import { describe, expect, it } from "vitest";

import {
  STALE_BUILD_POLLS_REQUIRED,
  nextStaleBuildStreak,
  ownFrontendBuild,
} from "./deployBeacon";

function documentWithEntry(src: string | null): Document {
  const doc = document.implementation.createHTMLDocument("t");
  if (src !== null) {
    const script = doc.createElement("script");
    script.setAttribute("type", "module");
    script.setAttribute("src", src);
    doc.head.appendChild(script);
  }
  return doc;
}

describe("ownFrontendBuild", () => {
  it("parses the hashed entry asset name from this page's module script", () => {
    expect(ownFrontendBuild(documentWithEntry("/assets/index-BQurQ9Qw.js"))).toBe(
      "index-BQurQ9Qw.js",
    );
  });

  it("disarms (null) in dev where the module script is unhashed source", () => {
    expect(ownFrontendBuild(documentWithEntry("/src/main.tsx"))).toBeNull();
    expect(ownFrontendBuild(documentWithEntry(null))).toBeNull();
  });
});

describe("nextStaleBuildStreak", () => {
  it("increments only across consecutive polls with a genuine mismatch", () => {
    const own = "index-BQurQ9Qw.js";
    let streak = 0;
    streak = nextStaleBuildStreak(streak, own, "index-Bk0OsIay.js");
    streak = nextStaleBuildStreak(streak, own, "index-Bk0OsIay.js");
    streak = nextStaleBuildStreak(streak, own, "index-Bk0OsIay.js");
    expect(streak).toBeGreaterThanOrEqual(STALE_BUILD_POLLS_REQUIRED);
  });

  it("resets on a single-poll flap back to a match (mid-deploy transient)", () => {
    const own = "index-BQurQ9Qw.js";
    let streak = nextStaleBuildStreak(0, own, "index-Bk0OsIay.js");
    streak = nextStaleBuildStreak(streak, own, own);
    expect(streak).toBe(0);
  });

  it("never counts when either identity is unknown (dev tab or beacon-less backend)", () => {
    expect(nextStaleBuildStreak(2, null, "index-Bk0OsIay.js")).toBe(0);
    expect(nextStaleBuildStreak(2, "index-BQurQ9Qw.js", null)).toBe(0);
    expect(nextStaleBuildStreak(2, "index-BQurQ9Qw.js", undefined)).toBe(0);
  });
});
