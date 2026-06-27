// site.js — shared marketing-site behaviour: render icons + theme toggle.
(function () {
  if (window.lucide) window.lucide.createIcons();
  var t = document.getElementById("theme-toggle");
  if (t) {
    t.addEventListener("click", function () {
      var root = document.documentElement;
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      t.innerHTML = '<i data-lucide="' + (next === "dark" ? "moon" : "sun") + '"></i>';
      if (window.lucide) window.lucide.createIcons();
    });
  }
})();
