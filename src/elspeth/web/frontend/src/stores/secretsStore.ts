// src/stores/secretsStore.ts
import { create } from "zustand";
import type { SecretInventoryItem } from "@/types/api";
import * as api from "@/api/client";

interface SecretsState {
  secrets: SecretInventoryItem[];
  isLoading: boolean;
  error: string | null;

  loadSecrets: () => Promise<void>;
  createSecret: (name: string, value: string) => Promise<void>;
  deleteSecret: (name: string) => Promise<void>;
  reset: () => void;
}

const initialState = {
  secrets: [] as SecretInventoryItem[],
  isLoading: false,
  error: null as string | null,
};

export const useSecretsStore = create<SecretsState>((set) => ({
  ...initialState,

  async loadSecrets() {
    set({ isLoading: true, error: null });
    try {
      const secrets = await api.listSecrets();
      set({ secrets, isLoading: false });
    } catch {
      set({ error: "Failed to load secrets.", isLoading: false });
    }
  },

  async createSecret(name: string, value: string) {
    set({ error: null });
    try {
      await api.createSecret(name, value);
      // Value is gone — only the ack comes back. Refresh inventory.
      const secrets = await api.listSecrets();
      set({ secrets });
    } catch {
      set({ error: "Failed to save secret." });
    }
  },

  async deleteSecret(name: string) {
    try {
      await api.deleteSecret(name);
      set((state) => ({
        secrets: state.secrets.filter((s) => s.name !== name),
      }));
    } catch {
      set({ error: "Failed to delete secret." });
    }
  },

  reset() {
    set(initialState);
  },
}));
