import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted Inter + JetBrains Mono @font-face rules — imported first so
// the fonts are declared before any component styles reference them.
import "./styles/fonts.css";
import App from "./App";

const root = document.getElementById("root");
if (!root) {
  throw new Error("Root element #root not found in document");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>
);
