/*
 * theme.js: the light/dark switcher shared by every page of the site.
 *
 * Load it in <head> (not deferred): it sets <html data-theme> before the page
 * paints, so a reader who chose dark never sees a flash of light. Any element
 * with a data-theme-toggle attribute becomes the switcher; clicking it cycles
 * auto (follow the system) -> light -> dark. The choice is remembered for the
 * whole site, and a "primer:theme" event lets pages redraw canvases.
 */
(function () {
  var KEY = "primer-theme";
  var ORDER = ["auto", "light", "dark"];
  var LABEL = { auto: "◐ Auto", light: "☀ Light", dark: "☾ Dark" };
  var root = document.documentElement;

  function stored() {
    try {
      // The paper companions used their own key before the site shared one.
      return localStorage.getItem(KEY) || localStorage.getItem("papers-theme") || "auto";
    } catch (_) {
      return "auto";
    }
  }

  var mode = ORDER.indexOf(stored()) >= 0 ? stored() : "auto";

  function apply() {
    if (mode === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
    var buttons = document.querySelectorAll("[data-theme-toggle]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].textContent = LABEL[mode];
      buttons[i].setAttribute("aria-label", "Theme: " + mode + ". Click to change.");
      buttons[i].title = "Theme: " + mode + " (click to change)";
    }
    document.dispatchEvent(new CustomEvent("primer:theme", { detail: mode }));
  }

  apply();
  document.addEventListener("DOMContentLoaded", apply);
  document.addEventListener("click", function (e) {
    var button = e.target.closest && e.target.closest("[data-theme-toggle]");
    if (!button) return;
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
    try {
      localStorage.setItem(KEY, mode);
    } catch (_) {}
    apply();
  });
})();
