(function () {
  try {
    var stored = localStorage.getItem("elspeth_theme");
    var prefersLight =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-color-scheme: light)").matches;
    var resolved =
      stored === "light"
        ? "light"
        : stored === "dark"
          ? "dark"
          : prefersLight
            ? "light"
            : "dark";

    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.style.colorScheme = resolved;
  } catch (_) {
    // localStorage, matchMedia, or document may be unavailable in constrained browsers.
  }
})();
