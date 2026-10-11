/* NFLELO ASCII density-map hero. Renders assets/ascii-hero.json (brightness 0-9 per character cell, sampled by
   tools/make_ascii_assets.js from the pattern engine's `terrain` render) as monospace text sized to its frame.
   Each cell takes a character from the ramp " .:-=+*#%@" by brightness, colored along primary -> secondary -> accent
   and fading toward the canvas as it dims. Most cells stay empty. A few cells are inverse video (accent background).
   The frame is real box-drawing characters. Decorative only (aria-hidden); no dependencies beyond the optional NFL.getJSON. */
(function () {
  'use strict';

  var fig = document.getElementById('ascii');
  if (!fig) return;
  var pre = fig.querySelector('.ascii-pre');
  var ADV = 0.6, LINE = 1.14;                 // JetBrains Mono: advance 0.6em, line-height 1.14 (--aa-adv, --aa-line); the advance is re-measured below
  var field = null, ramp = ' .:-=+*#%@', srcCols = 0, srcRows = 0, win = [0.5, 0.5];
  var pal = null, cum = null;

  // The real advance of the loaded face (a fallback monospace may differ from 0.6em).
  function measureAdv() {
    var probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;font:400 100px/1 ' + getComputedStyle(pre).fontFamily;
    probe.textContent = '┤' + new Array(40).join('─');
    document.body.appendChild(probe);
    var a = probe.getBoundingClientRect().width / 40 / 100;
    probe.remove();
    return a > 0.4 && a < 0.8 ? a : ADV;
  }

  // ------------------------------------------------------------------ colors from the tokens
  function tokenRGB(name) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    var m = /^#([0-9a-f]{6})$/i.exec(v);
    if (!m) return [128, 128, 128];
    var n = parseInt(m[1], 16);
    return [n >> 16 & 255, n >> 8 & 255, n & 255];
  }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function css(c) { return 'rgb(' + Math.round(c[0]) + ',' + Math.round(c[1]) + ',' + Math.round(c[2]) + ')'; }
  function palette() {
    if (pal) return pal;
    pal = { primary: tokenRGB('--primary'), secondary: tokenRGB('--secondary'), accent: tokenRGB('--accent'), canvas: tokenRGB('--canvas'), ink: tokenRGB('--ink') };
    return pal;
  }
  // gradient position t in 0..1: primary at 0, secondary at 0.5, accent at 1
  function grad(t) {
    var p = palette();
    return t < 0.5 ? mix(p.primary, p.secondary, t / 0.5) : mix(p.secondary, p.accent, (t - 0.5) / 0.5);
  }

  // ------------------------------------------------------------------ deterministic noise
  function hash(x, y) {
    var h = (x * 374761393 + y * 668265263) | 0;
    h = (h ^ (h >>> 13)) * 1274126177 | 0;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }

  // ------------------------------------------------------------------ render
  function esc(ch) { return ch === '<' ? '&lt;' : ch === '&' ? '&amp;' : ch; }

  function render() {
    if (!field) return;
    var W = fig.clientWidth, H = fig.clientHeight;
    if (W < 120 || H < 80) return;
    var adv = measureAdv(), base = W < 560 ? 11 : 14;
    var cols = Math.max(24, Math.floor(W / (base * adv)));
    var fs = W / (cols * adv);                       // fit the frame to the width exactly
    var rows = Math.max(8, Math.floor(H / (fs * LINE)));
    pre.style.fontSize = fs.toFixed(3) + 'px';
    var ic = cols - 2, ir = rows - 2;                // inside the frame
    // source window: k source cells per target cell, positioned by the page's data-win
    var k = Math.max(1, Math.min(2, Math.floor(srcCols / ic), Math.floor(srcRows / ir)));
    var ww = Math.min(srcCols, ic * k), wh = Math.min(srcRows, ir * k);
    var x0 = Math.round((srcCols - ww) * win[0]), y0 = Math.round((srcRows - wh) * win[1]);
    var p = palette(), out = [];

    var title = '┤ TERRAIN ├', foot = '┤ ' + ramp.replace(/ /g, '·') + ' ├';
    var top = '┌' + rep('─', 2) + title + rep('─', Math.max(0, ic - 2 - title.length)) + '┐';
    out.push('<span class="fr">' + top.replace(title, '</span><span class="tt">' + title + '</span><span class="fr">') + '</span>');

    var inv = 0, maxInv = Math.max(2, Math.round(ic * ir / 900));
    for (var y = 0; y < ir; y++) {
      var line = '<span class="fr">│</span>', run = '', runCol = '';
      for (var x = 0; x < ic; x++) {
        var s = 0, n = 0;
        for (var yy = 0; yy < k; yy++) for (var xx = 0; xx < k; xx++) {
          var sx = x0 + Math.min(ww - 1, Math.floor(x * ww / ic) + xx), sy = y0 + Math.min(wh - 1, Math.floor(y * wh / ir) + yy);
          s += field[sy * srcCols + sx]; n++;
        }
        var q = quantile(s / n), r = hash(x + x0 * 7, y + y0 * 13);
        // sparse: only the brightest part of the tonal range draws; the rest is empty with the odd dot
        var v = Math.max(0, Math.min(1, (q - 0.45) / 0.45)), idx = Math.round(v * 9), ch = ' ', col = null, invert = false;
        if (idx >= 1 && r < 0.62 + 0.38 * v) ch = ramp.charAt(Math.min(9, idx));
        else if (q > 0.3 && q <= 0.5 && r < 0.14) ch = r < 0.05 ? ':' : '.';
        if (ch !== ' ') {
          var lvl = Math.max(0.25, Math.min(1, idx / 9));
          var t = Math.min(0.97, 0.06 + 0.78 * (x / Math.max(1, ic - 1)) + 0.14 * (1 - v));
          var c = idx >= 9 ? p.ink : mix(p.canvas, grad(t), 0.3 + 0.7 * Math.pow(lvl, 0.8));
          col = css(c);
          if (idx >= 8 && inv < maxInv && hash(x * 31 + 7, y * 17 + 3) < 0.03) { invert = true; inv++; }
        }
        if (invert) {
          if (run) { line += span(run, runCol); run = ''; }
          line += '<span class="inv">' + esc(ch) + '</span>';
          continue;
        }
        if (col !== runCol && run) { line += span(run, runCol); run = ''; }
        runCol = col; run += esc(ch);
      }
      if (run) line += span(run, runCol);
      out.push(line + '<span class="fr">│</span>');
    }
    var bot = '└' + rep('─', Math.max(0, ic - foot.length - 2)) + foot + rep('─', 2) + '┘';
    if (ic > foot.length + 8) out.push('<span class="fr">' + bot.replace(foot, '</span><span class="tt">' + foot + '</span><span class="fr">') + '</span>');
    else out.push('<span class="fr">└' + rep('─', ic) + '┘</span>');
    pre.innerHTML = out.join('\n');
  }
  // brightness (0-9, fractional) -> share of cells darker than it, from the field's own histogram
  function quantile(b) {
    var t = Math.max(0, Math.min(9.999, b + 0.5)), d = Math.floor(t), f = t - d;   // each digit is the middle of its bin
    return cum[d] + f * (cum[d + 1] - cum[d]);
  }
  function span(text, col) { return col ? '<span style="color:' + col + '">' + text + '</span>' : text; }
  function rep(ch, n) { var s = ''; for (var i = 0; i < n; i++) s += ch; return s; }

  // ------------------------------------------------------------------ load and watch
  var w = (fig.getAttribute('data-win') || '0.5,0.5').split(',');
  win = [parseFloat(w[0]) || 0, parseFloat(w[1]) || 0];
  var get = window.NFL && window.NFL.getJSON ? window.NFL.getJSON : function (u) { return fetch(u).then(function (r) { return r.json(); }); };
  get('assets/ascii-hero.json').then(function (d) {
    ramp = d.ramp || ramp; srcCols = d.cols; srcRows = d.rows;
    field = new Uint8Array(srcCols * srcRows);
    d.field.forEach(function (row, y) { for (var x = 0; x < srcCols; x++) field[y * srcCols + x] = row.charCodeAt(x) - 48; });
    var hist = new Array(10).fill(0);
    for (var i = 0; i < field.length; i++) hist[field[i]]++;
    cum = [0]; for (var k = 0; k < 10; k++) cum.push(cum[k] + hist[k] / field.length);
    render();
    if (window.ResizeObserver) {
      var last = fig.clientWidth + 'x' + fig.clientHeight;
      new ResizeObserver(function () {
        var now = fig.clientWidth + 'x' + fig.clientHeight;
        if (now !== last) { last = now; render(); }
      }).observe(fig);
    } else window.addEventListener('resize', render);
  }).catch(function () { fig.hidden = true; });
})();
