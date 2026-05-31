/**
 * ReadOnlyContext — propagates a read-only signal through a React
 * subtree.
 *
 * Phase 6B FIX-C: the SharedInspectView wraps its inspect-mode subtree
 * in `<ReadOnlyProvider value={true}>` so any descendant component
 * (notably `AuditReadinessRow`) that supports action affordances
 * (Explain, opening a detail drawer, etc.) can disable them via
 * `useReadOnly()`. The default context value is `false`, so
 * components mounted in the regular composer (which does not wrap
 * itself in a provider) behave unchanged.
 *
 * The signal is intentionally a single boolean — there is exactly one
 * read-only mode in the product today (the shared-inspect view).
 * If future surfaces need finer-grained capabilities (e.g. "yaml only
 * read-only", "everything read-only except annotation"), this contract
 * can extend to a structured value without breaking the boolean
 * consumers (callers using `useReadOnly()` get the unchanged false
 * default).
 */

import { createContext, useContext, type ReactNode } from "react";

const ReadOnlyContext = createContext<boolean>(false);

interface ReadOnlyProviderProps {
  children: ReactNode;
  /** Defaults to `true` — the only reason to instantiate the provider
   *  is to flip the signal on for a subtree. Pass `false` only in
   *  tests that need to explicitly verify the un-set default. */
  value?: boolean;
}

export function ReadOnlyProvider({
  children,
  value = true,
}: ReadOnlyProviderProps): JSX.Element {
  return (
    <ReadOnlyContext.Provider value={value}>
      {children}
    </ReadOnlyContext.Provider>
  );
}

/**
 * Returns the current read-only signal. Defaults to `false` when no
 * provider is present higher in the tree — the composer surface
 * mounts components without a provider and gets the default.
 */
export function useReadOnly(): boolean {
  return useContext(ReadOnlyContext);
}
