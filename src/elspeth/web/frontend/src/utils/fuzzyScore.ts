/**
 * Fuzzy matching with consecutive-character bonus scoring.
 *
 * Single source of truth for catalog and command-palette fuzzy search.
 * Score is **lower-is-better**: a perfect prefix match scores 0, scattered
 * matches accumulate the gap distance between each matched character.
 *
 * `confidenceFromScore` maps the raw score into a normalized [0, 1] confidence
 * so consumers can apply a threshold without knowing the absolute score range
 * (which depends on target length).
 */

/**
 * Lower bound below which fuzzy matches are treated as noise.
 *
 * Empirical: a 5-char query scattered across a 13-char target with maximum
 * gaps yields confidence ≈ 0.077, which is the kind of result a user would
 * read as a false positive. Set above that floor.
 */
export const MIN_FUZZY_CONFIDENCE = 0.1;

/**
 * Score a fuzzy match between query and target.
 *
 * Returns -1 if any query character is missing from target (in order).
 * Otherwise returns a non-negative score where 0 means "all query characters
 * matched consecutively starting at index 0" (best) and higher values reflect
 * larger gaps between matched characters (worse).
 *
 * Case-insensitive.
 */
export function fuzzyMatch(query: string, target: string): number {
  const q = query.toLowerCase();
  const t = target.toLowerCase();

  if (q.length === 0) return 0;

  let qIdx = 0;
  let score = 0;
  let lastMatchIdx = -1;

  for (let tIdx = 0; tIdx < t.length && qIdx < q.length; tIdx++) {
    if (t[tIdx] === q[qIdx]) {
      // Consecutive matches add nothing; gaps add the distance jumped.
      if (lastMatchIdx === tIdx - 1) {
        score += 0;
      } else {
        score += tIdx - lastMatchIdx;
      }
      lastMatchIdx = tIdx;
      qIdx++;
    }
  }

  return qIdx === q.length ? score : -1;
}

/**
 * Convert a raw fuzzy score into a normalized confidence in [0, 1].
 *
 * Confidence 1.0 = perfect match. Confidence 0.0 = no match (or absolute
 * worst-case scatter). Higher is better. Use with `MIN_FUZZY_CONFIDENCE` to
 * trim noise without depending on target-length specifics.
 */
export function confidenceFromScore(
  score: number,
  targetLength: number,
): number {
  if (score < 0) return 0;
  if (targetLength <= 0) return score === 0 ? 1 : 0;
  return Math.max(0, 1 - score / targetLength);
}
