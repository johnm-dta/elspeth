import { useEffect } from "react";

/**
 * Product name used as the document.title base — matches the <title> in
 * index.html so pre-hydration tabs and in-app tabs read consistently.
 */
export const PRODUCT_TITLE = "ELSPETH";

/**
 * Compose the browser-tab title for the current app context.
 *
 * "{Session title} — ELSPETH" while a session is active so multi-tabbed /
 * bookmarked sessions (the whole point of the #/{sessionId} deep links) are
 * distinguishable; the bare product name otherwise (elspeth-42f63fa312).
 */
export function formatDocumentTitle(sessionTitle: string | null): string {
  return sessionTitle !== null && sessionTitle !== ""
    ? `${sessionTitle} — ${PRODUCT_TITLE}`
    : PRODUCT_TITLE;
}

/** Reactively set document.title. Re-runs on every title change (session
 *  switch, rename, post-compose auto-title refresh). */
export function useDocumentTitle(title: string): void {
  useEffect(() => {
    document.title = title;
  }, [title]);
}
