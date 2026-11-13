// Click any chart to open an enlarged copy in a modal.
// A plain click (no drag) on the plot opens it; legend / modebar clicks keep their normal behavior.
(function () {
  var IGNORE = ".legend, .modebar, .modebar-container, .updatemenu-container, " +
               ".rangeslider-container, .slider-container, .chart-expand-btn";
  var overlay, body, titleEl, down = null;

  function buildModal() {
    overlay = document.createElement("div");
    overlay.className = "chart-modal-overlay";
    overlay.innerHTML =
      '<div class="chart-modal" role="dialog" aria-modal="true">' +
      '  <div class="chart-modal-header">' +
      '    <div class="chart-modal-title"></div>' +
      '    <button class="chart-modal-close" title="Tutup (Esc)">✕</button>' +
      "  </div>" +
      '  <div class="chart-modal-body"></div>' +
      "</div>";
    document.body.appendChild(overlay);
    body = overlay.querySelector(".chart-modal-body");
    titleEl = overlay.querySelector(".chart-modal-title");
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay || e.target.closest(".chart-modal-close")) closeModal();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && overlay.classList.contains("open")) closeModal();
    });
  }

  function figureTitle(layout) {
    var t = layout && layout.title;
    var text = typeof t === "string" ? t : (t && t.text) || "";
    return text.replace(/<[^>]*>/g, " ");
  }

  function openModal(gd) {
    if (!window.Plotly || !gd || !gd.data) return;
    if (!overlay) buildModal();
    var data = JSON.parse(JSON.stringify(gd.data));
    var layout = JSON.parse(JSON.stringify(gd.layout || {}));
    titleEl.textContent = figureTitle(layout);
    delete layout.width;
    delete layout.height;
    layout.autosize = true;
    layout.title = Object.assign({}, typeof layout.title === "object" ? layout.title : {}, { text: "" });
    layout.margin = Object.assign({}, layout.margin, { t: 50 });
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    window.Plotly.newPlot(body, data, layout, { responsive: true, displaylogo: false });
  }

  function closeModal() {
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    if (window.Plotly) window.Plotly.purge(body);
  }

  function graphFrom(target) {
    if (!target || !target.closest || target.closest(".chart-modal-overlay")) return null;
    return target.closest(".js-plotly-plot");
  }

  // Track mousedown/mouseup at document level (capture) so Plotly's drag layer doesn't swallow it.
  document.addEventListener("mousedown", function (e) {
    var gd = graphFrom(e.target);
    down = gd && !e.target.closest(IGNORE) && e.button === 0
      ? { gd: gd, x: e.clientX, y: e.clientY, t: Date.now() }
      : null;
  }, true);

  document.addEventListener("mouseup", function (e) {
    if (!down) return;
    var d = down;
    down = null;
    var moved = Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y);
    if (moved < 6 && Date.now() - d.t < 500) {
      setTimeout(function () { openModal(d.gd); }, 0);
    }
  }, true);

  // Small expand button on each chart, shown on hover.
  function addButtons() {
    document.querySelectorAll(".dash-graph").forEach(function (g) {
      if (g.closest(".chart-modal-overlay") || g.querySelector(".chart-expand-btn")) return;
      var btn = document.createElement("button");
      btn.className = "chart-expand-btn";
      btn.title = "Perbesar chart";
      btn.textContent = "⤢";
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        openModal(g.querySelector(".js-plotly-plot"));
      });
      g.appendChild(btn);
    });
  }

  new MutationObserver(addButtons).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", addButtons);
})();
