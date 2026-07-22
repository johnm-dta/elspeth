// Deploy-cache coherence beacon.
//
// After a rebuild, an open tab keeps running its previous bundle against the
// new dist: vite's emptyOutDir wipes the old hashed assets, so lazy-loaded
// chunks 404 (the backend serves its JSON 404 → MIME refusal), and newly
// shipped features are silently absent — which reads as a product hang.
//
// The tab's OWN identity is its hashed entry-script name, read from the DOM
// copy of index.html that loaded it; the SERVED identity arrives on the
// /api/system/status payload the health check already polls (~10s), parsed
// server-side from the same built index.html at startup. Both derive from
// the identical artifact family, so the identities can never drift by
// construction. Either side null (dev tab, dist-less backend) disarms the
// beacon entirely.

export const STALE_BUILD_POLLS_REQUIRED = 3;

export function ownFrontendBuild(doc: Document = document): string | null {
  const script = doc.querySelector('script[type="module"][src*="assets/index-"]');
  const src = script?.getAttribute("src") ?? null;
  if (src === null) return null;
  const match = /(index-[A-Za-z0-9_-]+\.js)(?:\?.*)?$/.exec(src);
  return match ? match[1] : null;
}

// Streak reducer for the debounce: a mismatch must hold across
// STALE_BUILD_POLLS_REQUIRED consecutive polls before the banner shows, so a
// mid-deploy transient (status flapping during the restart window) never
// flashes it. Any match or unknown identity resets the streak; the caller
// LATCHES once the threshold is reached (a genuinely new deploy stays new —
// only a refresh clears it).
export function nextStaleBuildStreak(
  current: number,
  own: string | null,
  served: string | null | undefined,
): number {
  if (own === null || served === null || served === undefined || own === served) {
    return 0;
  }
  return current + 1;
}
