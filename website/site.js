// site.js — shared marketing-site behaviour: render icons + theme toggle.
// The initial theme is set pre-paint by an inline <head> script (reads
// localStorage + prefers-color-scheme); this only handles the toggle and
// persists the choice so it survives navigation.
(function () {
  if (window.lucide) window.lucide.createIcons();
  var t = document.getElementById("theme-toggle");
  if (!t) return;
  function setIcon() {
    var dark = document.documentElement.getAttribute("data-theme") !== "light";
    t.innerHTML = '<i data-lucide="' + (dark ? "moon" : "sun") + '"></i>';
    if (window.lucide) window.lucide.createIcons();
  }
  setIcon(); // reflect the resolved theme on load
  t.addEventListener("click", function () {
    var root = document.documentElement;
    var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("elspeth-theme", next); } catch (e) {}
    setIcon();
  });
})();
