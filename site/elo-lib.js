/* NFELO ratings helpers, shared by app.js (home) and elo.js (the Elo page): number and date formatting, the
   hand-built SVG line chart, the Elo data loader, and the section reveal. Needs common.js first. Exposed as window.NFL.R.
   Charts follow one spec: 2px lines, hairline solid grid, 2px surface rings on dots, one tooltip style,
   a table view under each chart, and chart areas set in bordered wells. Series are told apart by color, shape and dash. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear;
  var tipShow = N.tipShow, tipHide = N.tipHide;
  var mark = N.mark, moveMark = N.moveMark;
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ formatting

  // ------------------------------------------------------------------ formatting

  function f1(n) { return n.toFixed(1); }
  function signed(n, d) { d = d == null ? 1 : d; var s = Math.abs(n).toFixed(d); return (+s === 0 ? '' : n > 0 ? '+' : '-') + s; }
  function pct(p, d) { return (p * 100).toFixed(d == null ? 0 : d) + '%'; }
  function rec(w, l, t) { return w + '-' + l + (t ? '-' + t : ''); }
  function winPct(w, l, t) { return ((w + t / 2) / (w + l + t) * 100).toFixed(1) + '%'; }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  function parseDate(iso) { var p = iso.split('-'); return new Date(+p[0], +p[1] - 1, +p[2]); }
  function longDate(iso) { return parseDate(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }); }
  function shortDay(iso) { return parseDate(iso).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }); }
  function clock(hhmm) {
    if (!hhmm) return '';
    var p = hhmm.split(':'), hr = +p[0];
    return ((hr + 11) % 12 + 1) + ':' + p[1] + ' ' + (hr < 12 ? 'AM' : 'PM') + ' ET';
  }
  function ordinalSpots(n) { return n === 0 ? 'unchanged' : (n > 0 ? 'up ' : 'down ') + plural(Math.abs(n), 'spot', 'spots'); }

  function dayLong(iso) { return parseDate(iso).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' }); }

  // ------------------------------------------------------------------ generic line chart

  // ------------------------------------------------------------------ generic line chart

  function niceStep(span, n) {
    var raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
  }

  function ticksFor(lo, hi, n) {
    var step = niceStep(hi - lo, n), out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  /* o: { series:[{id,label,cls,style,pts:[{x,y,...}]}], x:[a,b], y:[a,b], height(w), xTicks:[], yTicks:[],
          xFmt, yFmt, ref:{y,label}, tip(x, rows)->{title,rows}, aria, markers:[{x,y,text,dy}], endLabels } */
  function lineChart(host, o) {
    clear(host);
    var W = Math.max(280, host.clientWidth || 640);
    var H = o.height(W);
    var m = { t: o.yLabel ? 32 : 12, r: o.endLabels ? 68 : 16, b: 28, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var sx = function (v) { return m.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * iw; };
    var sy = function (v) { return m.t + (1 - (v - o.y[0]) / (o.y[1] - o.y[0])) * ih; };

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', tabindex: 0, 'aria-label': o.aria }, host);

    if (o.yLabel) sv('text', { class: 'tick a-start', x: 4, y: 14, text: o.yLabel }, svg);
    o.yTicks.forEach(function (v) {
      sv('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, svg);
      sv('text', { class: 'tick a-end', x: m.l - 8, y: sy(v) + 4, text: o.yFmt(v) }, svg);
    });
    // Skip x ticks that would sit within 44px of the previous one (narrow screens).
    var lastTickX = -1e9;
    o.xTicks.forEach(function (v) {
      if (sx(v) - lastTickX < 44) return;
      lastTickX = sx(v);
      sv('text', { class: 'tick a-mid', x: sx(v), y: H - 8, text: o.xFmt(v) }, svg);
    });
    if (o.ref) {
      sv('line', { class: 'refline', x1: m.l, x2: W - m.r, y1: sy(o.ref.y), y2: sy(o.ref.y) }, svg);
      // Put the label in the corner (left/right, above/below the line) that touches the fewest data points.
      var lw = o.ref.label.length * 6.6 + 8, ry = sy(o.ref.y), best = null;
      [[1, -1], [1, 1], [0, -1], [0, 1]].forEach(function (c) {
        var x0 = c[0] ? W - m.r - 4 - lw : m.l + 2, x1 = x0 + lw;
        var y0 = c[1] < 0 ? ry - 18 : ry + 2, y1 = y0 + 16, hits = 0;
        o.series.forEach(function (sr) {
          sr.pts.forEach(function (p) { if (p.y != null && sx(p.x) >= x0 && sx(p.x) <= x1 && sy(p.y) >= y0 && sy(p.y) <= y1) hits++; });
        });
        if (!best || hits < best.hits) best = { hits: hits, right: c[0], above: c[1] < 0 };
      });
      if (best.hits === 0) sv('text', { class: 'tick halo ' + (best.right ? 'a-end' : 'a-start'), x: best.right ? W - m.r - 4 : m.l + 4,
        y: best.above ? ry - 5 : ry + 13, text: o.ref.label }, svg);
    }

    o.series.forEach(function (s) {
      // Points with y == null break the line into separate segments.
      var d = '', pen = false;
      s.pts.forEach(function (p) {
        if (p.y == null) { pen = false; return; }
        d += (pen ? 'L' : 'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1);
        pen = true;
      });
      if (s.dotsOnly) {
        s.pts.forEach(function (p) { if (p.y != null) mark(svg, s.cls, sx(p.x), sy(p.y), 3.5, 'plain'); });
      } else {
        sv('path', { class: 'line ' + s.cls, d: d }, svg);
      }
      var last = s.pts.filter(function (p) { return p.y != null; }).pop();
      if (!s.dotsOnly) mark(svg, s.cls, sx(last.x), sy(last.y), 5);
      if (o.endLabels) {
        sv('text', { class: 'lbl', x: sx(last.x) + 10, y: sy(last.y) + 4, text: s.short || s.label }, svg);
      }
    });

    (o.markers || []).forEach(function (mk) {
      mark(svg, mk.cls, sx(mk.x), sy(mk.y), 6);
      var anchor = sx(mk.x) > W - 90 ? 'end' : sx(mk.x) < m.l + 70 ? 'start' : 'middle';
      sv('text', { class: 'lbl', x: sx(mk.x), y: sy(mk.y) + mk.dy, 'text-anchor': anchor, text: mk.text }, svg);
    });

    // Hover layer: a crosshair snaps to the nearest x; the tooltip lists every series at that x.
    var maps = o.series.map(function (s) { var mp = new Map(); s.pts.forEach(function (p) { mp.set(p.x, p); }); return mp; });
    var xs = Array.from(new Set([].concat.apply([], o.series.map(function (s) { return s.pts.map(function (p) { return p.x; }); })))).sort(function (a, b) { return a - b; });
    var cross = sv('line', { class: 'cross', y1: m.t, y2: m.t + ih, visibility: 'hidden' }, svg);
    var dots = o.series.map(function (s) { var d = mark(svg, s.cls, 0, 0, 6); d.setAttribute('visibility', 'hidden'); return d; });
    var cur = xs.length - 1;

    function nearest(px) {
      var v = o.x[0] + (px - m.l) / iw * (o.x[1] - o.x[0]), lo = 0, hi = xs.length - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] < v) lo = mid; else hi = mid; }
      return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
    }

    function show(i, cx, cy) {
      cur = i;
      var x = xs[i];
      cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.setAttribute('visibility', 'visible');
      var rows = [];
      o.series.forEach(function (s, k) {
        var p = maps[k].get(x);
        if (!p || p.y == null) { dots[k].setAttribute('visibility', 'hidden'); if (p) rows.push({ p: p, s: s }); return; }
        moveMark(dots[k], sx(x), sy(p.y)); dots[k].setAttribute('visibility', 'visible');
        rows.push({ p: p, s: s });
      });
      var c = o.tip(x, rows);
      if (cx == null) { var r = svg.getBoundingClientRect(); cx = r.left + sx(x) * (r.width / W); cy = r.top + m.t + 10; }
      tipShow(cx, cy, c.title, c.rows);
    }

    function hide() {
      cross.setAttribute('visibility', 'hidden');
      dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); });
      tipHide();
    }

    svg.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect();
      var px = (e.clientX - r.left) * (W / r.width);
      if (px < m.l - 4 || px > W - m.r + 4) return hide();
      show(nearest(px), e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () { if (!svg.matches(':hover')) show(cur); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 5 : 1, i = cur;
      if (e.key === 'ArrowLeft') i = Math.max(0, cur - step);
      else if (e.key === 'ArrowRight') i = Math.min(xs.length - 1, cur + step);
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = xs.length - 1;
      else if (e.key === 'Escape') return hide();
      else return;
      e.preventDefault();
      show(i);
    });
    return svg;
  }

  // ------------------------------------------------------------------ data and page helpers

  var D = {};          // loaded data, by file name (plus D.ml)
  var byTeam = {};     // ladder row by franchise
  var histBy = {};     // history row by franchise

  function setHeadline(id, text) { var n = $('#' + id); if (n && text) n.textContent = text; }

  // Bars grow once, when their section first scrolls into view (CSS does the easing).
  function initReveal() {
    if (REDUCED || !('IntersectionObserver' in window)) return;
    document.documentElement.classList.add('anim');
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
    document.querySelectorAll('main > .sec').forEach(function (sec) { io.observe(sec); });
  }

  // Fetches data/<file>.json for each name, plus the optional ml.json (it exists only once the prediction ledger does).
  // Fills D, byTeam and histBy. Resolves to D.
  function load(files) {
    var mlFile = N.getJSON('data/ml.json', true);
    return Promise.all(files.map(function (f) { return N.getJSON('data/' + f + '.json'); }).concat([mlFile])).then(function (all) {
      files.forEach(function (f, i) { D[f] = all[i]; });
      D.ml = all[files.length];
      if (D.ladder) D.ladder.teams.forEach(function (t) { byTeam[t.team] = t; });
      if (D.history) D.history.teams.forEach(function (t) { histBy[t.team] = t; });
      return D;
    });
  }

  N.R = {
    D: D, byTeam: byTeam, histBy: histBy, REDUCED: REDUCED, load: load,
    f1: f1, signed: signed, pct: pct, rec: rec, winPct: winPct, plural: plural,
    parseDate: parseDate, longDate: longDate, shortDay: shortDay, dayLong: dayLong, clock: clock, ordinalSpots: ordinalSpots,
    niceStep: niceStep, ticksFor: ticksFor, lineChart: lineChart, setHeadline: setHeadline, initReveal: initReveal
  };
})();
