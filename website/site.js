// site.js — shared project-site behaviour: render icons, theme toggle, and
// copy-to-clipboard on command/config blocks. The initial theme is set pre-paint
// by an inline <head> script (reads localStorage + prefers-color-scheme); this
// only handles the toggle, persists the choice, and progressively enhances code
// blocks marked `.copyable`.
(function () {
  // --- theme toggle --------------------------------------------------------
  var t = document.getElementById("theme-toggle");
  if (t) {
    setIcon();
    t.addEventListener("click", function () {
      var root = document.documentElement;
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("elspeth-theme", next); } catch (e) {}
      setIcon();
    });
  }
  function setIcon() {
    var dark = document.documentElement.getAttribute("data-theme") !== "light";
    t.innerHTML = '<i data-lucide="' + (dark ? "moon" : "sun") + '"></i>';
    // Reflect the action the control performs, for screen readers (A11y).
    t.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }

  // --- copy-to-clipboard ---------------------------------------------------
  document.querySelectorAll("pre.copyable").forEach(function (pre) {
    // Capture the payload before injecting the button; strip leading shell
    // prompts ("$ ") per line so pasted commands are clean. Config blocks have
    // no prompt and copy verbatim.
    var payload = pre.textContent.replace(/^[ \t]*\$[ \t]/gm, "").replace(/\s+$/, "");
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.setAttribute("aria-label", "Copy to clipboard");
    btn.innerHTML = '<i data-lucide="copy"></i>';
    pre.appendChild(btn);
    btn.addEventListener("click", function () {
      copyText(payload).then(function (ok) {
        if (!ok) return;
        flash(btn);
      });
    });
  });

  function flash(btn) {
    btn.classList.add("copied");
    btn.setAttribute("aria-label", "Copied");
    btn.innerHTML = '<i data-lucide="check"></i>';
    if (window.lucide) window.lucide.createIcons();
    setTimeout(function () {
      btn.classList.remove("copied");
      btn.setAttribute("aria-label", "Copy to clipboard");
      btn.innerHTML = '<i data-lucide="copy"></i>';
      if (window.lucide) window.lucide.createIcons();
    }, 1600);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; },
        function () { return fallbackCopy(text); }
      );
    }
    return Promise.resolve(fallbackCopy(text));
  }

  // Fallback for non-secure contexts (e.g. opening the files over file://).
  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "absolute";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  // Render all icons last, after the toggle + copy buttons are in the DOM.
  if (window.lucide) window.lucide.createIcons();
})();
