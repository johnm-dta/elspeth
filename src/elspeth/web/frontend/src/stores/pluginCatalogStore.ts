import { create, type StoreApi, type UseBoundStore } from "zustand";

import * as api from "@/api/client";
import type {
  ApiError,
  PluginPolicyResponse,
  PluginSchemaInfo,
  PluginSummary,
} from "@/types/index";

export const PLUGIN_CATALOG_INVALIDATED_EVENT =
  "elspeth:plugin-catalog-invalidated";

type PluginKind = "source" | "transform" | "sink";

export interface PluginCatalogLoadRequest {
  principal: string;
  /** Optional caller-observed fingerprint, used to discard known-stale data immediately. */
  fingerprint?: string;
  force?: boolean;
}

export interface PluginCatalogState {
  key: string | null;
  principal: string | null;
  fingerprint: string | null;
  policy: PluginPolicyResponse | null;
  sources: PluginSummary[] | null;
  transforms: PluginSummary[] | null;
  sinks: PluginSummary[] | null;
  schemas: Record<string, PluginSchemaInfo>;
  schemaLoading: Record<string, boolean>;
  schemaErrors: Record<string, boolean>;
  isLoading: boolean;
  error: string | null;
  load: (request: PluginCatalogLoadRequest) => Promise<void>;
  loadSchema: (kind: PluginKind, name: string) => Promise<void>;
  invalidate: () => Promise<void>;
  clear: () => void;
  dispose: () => void;
}

type PluginCatalogStore = UseBoundStore<StoreApi<PluginCatalogState>>;

function catalogKey(principal: string, fingerprint: string): string {
  return `${principal}:${fingerprint}`;
}

function emptyCatalogState() {
  return {
    key: null,
    principal: null,
    fingerprint: null,
    policy: null,
    sources: null,
    transforms: null,
    sinks: null,
    schemas: {},
    schemaLoading: {},
    schemaErrors: {},
    isLoading: false,
    error: null,
  };
}

export function createPluginCatalogStore(): PluginCatalogStore {
  let generation = 0;
  let activeLoad: {
    principal: string;
    fingerprint: string | undefined;
    promise: Promise<void>;
  } | null = null;
  let removeInvalidationListener = () => {};

  const store = create<PluginCatalogState>((set, get) => ({
    ...emptyCatalogState(),

    async load(request) {
      const { principal, fingerprint, force = false } = request;
      const current = get();
      const expectedKey = fingerprint === undefined
        ? null
        : catalogKey(principal, fingerprint);
      if (
        !force &&
        expectedKey !== null &&
        current.key === expectedKey &&
        current.sources !== null &&
        current.transforms !== null &&
        current.sinks !== null
      ) {
        return;
      }
      if (
        !force &&
        activeLoad !== null &&
        activeLoad.principal === principal &&
        activeLoad.fingerprint === fingerprint
      ) {
        return activeLoad.promise;
      }

      const requestGeneration = ++generation;
      const knownStale =
        current.principal !== principal ||
        (fingerprint !== undefined && current.fingerprint !== fingerprint);
      if (knownStale) {
        set({
          ...emptyCatalogState(),
          principal,
          isLoading: true,
        });
      } else {
        set({ isLoading: true, error: null });
      }

      const promise = (async () => {
        try {
          for (let attempt = 0; attempt < 3; attempt += 1) {
            const policyResponse = await api.fetchPluginPolicy();
            if (requestGeneration !== generation) return;
            const policy = policyResponse.data;
            if (policyResponse.snapshotFingerprint !== policy.snapshot_fingerprint) {
              continue;
            }

            const canonicalPrincipal = policy.principal_scope;
            const nextKey = catalogKey(
              canonicalPrincipal,
              policy.snapshot_fingerprint,
            );
            const afterPolicy = get();
            if (
              !force &&
              afterPolicy.key === nextKey &&
              afterPolicy.sources !== null &&
              afterPolicy.transforms !== null &&
              afterPolicy.sinks !== null
            ) {
              set({ policy, isLoading: false, error: null });
              return;
            }

            set({
              key: nextKey,
              principal: canonicalPrincipal,
              fingerprint: policy.snapshot_fingerprint,
              policy,
              sources: null,
              transforms: null,
              sinks: null,
              schemas: {},
              schemaLoading: {},
              schemaErrors: {},
              isLoading: true,
              error: null,
            });

            const [sources, transforms, sinks] = await Promise.all([
              api.listSources(),
              api.listTransforms(),
              api.listSinks(),
            ]);
            if (requestGeneration !== generation) return;
            const fingerprints = [
              sources.snapshotFingerprint,
              transforms.snapshotFingerprint,
              sinks.snapshotFingerprint,
            ];
            if (fingerprints.some((value) => value !== policy.snapshot_fingerprint)) {
              continue;
            }
            set({
              sources: sources.data,
              transforms: transforms.data,
              sinks: sinks.data,
              isLoading: false,
              error: null,
            });
            return;
          }
          throw new Error("Plugin catalog snapshot changed repeatedly while loading.");
        } catch {
          if (requestGeneration !== generation) return;
          set({
            sources: null,
            transforms: null,
            sinks: null,
            isLoading: false,
            error: "Failed to load plugin catalog.",
          });
        }
      })();
      activeLoad = { principal, fingerprint, promise };
      try {
        await promise;
      } finally {
        if (activeLoad?.promise === promise) activeLoad = null;
      }
    },

    async loadSchema(kind, name) {
      const cacheKey = `${kind}:${name}`;
      const state = get();
      const owningCatalogKey = state.key;
      if (
        owningCatalogKey === null ||
        state.schemas[cacheKey] !== undefined ||
        state.schemaLoading[cacheKey]
      ) {
        return;
      }
      set({
        schemaLoading: { ...state.schemaLoading, [cacheKey]: true },
        schemaErrors: { ...state.schemaErrors, [cacheKey]: false },
      });
      try {
        const schema = await api.getPluginSchema(kind, name);
        if (get().key !== owningCatalogKey) return;
        if (schema.snapshotFingerprint !== state.fingerprint) {
          await get().load({ principal: state.principal!, force: true });
          if (get().fingerprint !== schema.snapshotFingerprint) return;
        }
        set((latest) => ({
          schemas: { ...latest.schemas, [cacheKey]: schema.data },
          schemaLoading: { ...latest.schemaLoading, [cacheKey]: false },
          schemaErrors: { ...latest.schemaErrors, [cacheKey]: false },
        }));
      } catch (error) {
        if (get().key !== owningCatalogKey) return;
        const responseFingerprint = (error as ApiError).snapshot_fingerprint;
        if (
          responseFingerprint !== undefined &&
          responseFingerprint !== state.fingerprint
        ) {
          await get().load({ principal: state.principal!, force: true });
          return;
        }
        set((latest) => ({
          schemaLoading: { ...latest.schemaLoading, [cacheKey]: false },
          schemaErrors: { ...latest.schemaErrors, [cacheKey]: true },
        }));
      }
    },

    async invalidate() {
      const principal = get().principal;
      generation += 1;
      activeLoad = null;
      set({
        ...emptyCatalogState(),
        principal,
        isLoading: principal !== null,
      });
      if (principal !== null) {
        await get().load({ principal, force: true });
      }
    },

    clear() {
      generation += 1;
      activeLoad = null;
      set(emptyCatalogState());
    },

    dispose() {
      removeInvalidationListener();
      removeInvalidationListener = () => {};
      get().clear();
    },
  }));

  if (typeof window !== "undefined") {
    const handleInvalidation = () => {
      void store.getState().invalidate();
    };
    window.addEventListener(PLUGIN_CATALOG_INVALIDATED_EVENT, handleInvalidation);
    removeInvalidationListener = () => {
      window.removeEventListener(
        PLUGIN_CATALOG_INVALIDATED_EVENT,
        handleInvalidation,
      );
    };
  }
  return store;
}

export const usePluginCatalogStore = createPluginCatalogStore();
