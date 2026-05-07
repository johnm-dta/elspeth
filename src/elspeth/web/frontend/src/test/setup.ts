import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement window.matchMedia.
// Provide a minimal stub so components that use media queries (e.g. Layout)
// can render without throwing.
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }),
});
