// src/stores/blobStore.ts
import { create } from "zustand";
import type { BlobMetadata } from "@/types/api";
import * as api from "@/api/client";

interface BlobState {
  blobs: BlobMetadata[];
  isLoading: boolean;
  error: string | null;

  loadBlobs: (sessionId: string) => Promise<void>;
  uploadBlob: (sessionId: string, file: File) => Promise<BlobMetadata>;
  deleteBlob: (sessionId: string, blobId: string) => Promise<void>;
  downloadBlob: (sessionId: string, blobId: string) => Promise<void>;
  clearBlobs: () => void;
  reset: () => void;
}

const initialState = {
  blobs: [] as BlobMetadata[],
  isLoading: false,
  error: null as string | null,
};

export const useBlobStore = create<BlobState>((set) => ({
  ...initialState,

  async loadBlobs(sessionId: string) {
    set({ isLoading: true, error: null });
    try {
      const blobs = await api.listBlobs(sessionId);
      set({ blobs, isLoading: false });
    } catch {
      set({ error: "Failed to load files.", isLoading: false });
    }
  },

  async uploadBlob(sessionId: string, file: File) {
    set({ error: null });
    try {
      const blob = await api.uploadBlob(sessionId, file);
      set((state) => ({ blobs: [blob, ...state.blobs] }));
      return blob;
    } catch (err) {
      const detail =
        (err as { status?: number }).status === 413
          ? "File exceeds the maximum upload size."
          : (err as { status?: number }).status === 415
            ? "Unsupported file type. Please use CSV, JSON, JSONL, or plain text."
            : "Upload failed. Please try again.";
      set({ error: detail });
      throw err;
    }
  },

  async deleteBlob(sessionId: string, blobId: string) {
    try {
      await api.deleteBlob(sessionId, blobId);
      set((state) => ({
        blobs: state.blobs.filter((b) => b.id !== blobId),
      }));
    } catch (err) {
      const detail =
        (err as { status?: number }).status === 409
          ? "Cannot delete — file is linked to an active run."
          : "Failed to delete file.";
      set({ error: detail });
    }
  },

  async downloadBlob(sessionId: string, blobId: string) {
    try {
      const { data, filename } = await api.downloadBlobContent(
        sessionId,
        blobId,
      );
      // Trigger browser download via object URL
      const url = URL.createObjectURL(data);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      set({ error: "Download failed." });
    }
  },

  clearBlobs() {
    set({ blobs: [], error: null });
  },

  reset() {
    set(initialState);
  },
}));
