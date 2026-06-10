/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const backendPort = process.env.PLAYWRIGHT_BACKEND_PORT ?? "8451";
const frontendPort = Number(process.env.PLAYWRIGHT_FRONTEND_PORT ?? "5173");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts", "./src/test/a11y/setup.ts"],
    // src/** for app unit tests, plus the e2e harness's pure logic (the
    // outcome classifier) which deserves fast unit coverage without Playwright.
    include: ["src/**/*.test.{ts,tsx}", "tests/e2e/harness/**/*.test.ts"],
    css: false,
  },
  server: {
    port: frontendPort,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
      "/ws": {
        target: `ws://127.0.0.1:${backendPort}`,
        ws: true,
      },
    },
  },
});
