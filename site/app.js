/* NFLELO front end. No dependencies: reads ./data/*.json, draws hand-built SVG/HTML.
   Charts follow one spec: 2px lines, hairline solid grid, 4px rounded bar ends,
   2px surface rings on dots, one tooltip style, a table view under each chart. */
(function () {
  'use strict';

  var FILES = ['meta', 'ladder', 'upcoming', 'history', 'luck', 'tapestry', 'records', 'scorecard', 'headlines'];
  var SVGNS = 'http://www.w3.org/2000/svg';
  var BG_DARK = [28, 33, 38], BG_LIGHT = [245, 247, 249];   // --neutral-dark, --neutral-light
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ dom helpers

  function $(sel, root) { return (root || document).querySelector(sel); }

  function h(tag, props) {
    var n = document.createElement(tag);
    props = props || {};
    Object.keys(props).forEach(function (k) {
      var v = props[k];
      if (v === false || v == null) return;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k === 'style') n.setAttribute('style', v);
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    });
    for (var i = 2; i < arguments.length; i++) append(n, arguments[i]);
    return n;
  }

  function append(n, kid) {
    if (kid == null || kid === false) return;
    if (Array.isArray(kid)) { kid.forEach(function (k) { append(n, k); }); return; }
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }

  function sv(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] == null) return;
      if (k === 'text') n.textContent = attrs[k]; else n.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(n);
    return n;
  }

  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

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

  // ------------------------------------------------------------------ team colors for data marks
  // Team colors are only used inside marks. Dark team colors are lifted (or light ones
  // darkened) until they hold 3:1 against the page surface. Both variants go in as CSS
  // custom properties so the theme switches without a redraw.

  function rgb(hex) { var n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
  function toHex(c) { return '#' + c.map(function (v) { return ('0' + v.toString(16)).slice(-2); }).join(''); }
  function lin(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function lum(c) { return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]); }
  function contrast(a, b) { var x = lum(a), y = lum(b); if (x < y) { var t = x; x = y; y = t; } return (x + 0.05) / (y + 0.05); }
  function mixc(a, b, t) { return a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); }); }
  function ensure(c, surface, toward) {
    var t = 0, out = c;
    while (contrast(out, surface) < 3 && t < 1) { t += 0.04; out = mixc(c, toward, t); }
    return toHex(out);
  }
  var colorCache = {};
  function markVars(hex) {
    if (!colorCache[hex]) {
      var c = rgb(hex);
      colorCache[hex] = '--c-d:' + ensure(c, BG_DARK, [255, 255, 255]) + ';--c-l:' + ensure(c, BG_LIGHT, [0, 0, 0]);
    }
    return colorCache[hex];
  }

  // ------------------------------------------------------------------ tooltip

  var tip = $('#tip'), live = $('#live');

  function tipContent(title, rows) {
    clear(tip);
    if (title) tip.appendChild(h('div', { class: 'tt', text: title }));
    rows.forEach(function (r) {
      var key = r.cls ? h('i', { class: r.cls, style: r.style }) : null;
      tip.appendChild(h('div', { class: 'tr' }, key, h('b', { text: r.value }), r.label ? h('span', { text: r.label }) : null));
    });
  }

  function tipPlace(x, y) {
    tip.hidden = false;
    var w = tip.offsetWidth, hh = tip.offsetHeight;
    var left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (left < 8) left = 8;
    if (top + hh > window.innerHeight - 8) top = y - hh - 14;
    if (top < 8) top = 8;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }

  function tipShow(x, y, title, rows) {
    tipContent(title, rows);
    tipPlace(x, y);
    live.textContent = (title ? title + '. ' : '') + rows.map(function (r) { return (r.label ? r.label + ' ' : '') + r.value; }).join('. ');
  }

  function tipHide() { tip.hidden = true; }

  // Per-mark tooltip for rows, bars and cells: pointer and keyboard focus show the same thing.
  function bindTip(node, build) {
    function at(e) { var c = build(); if (c) tipShow(e.clientX, e.clientY, c.title, c.rows); }
    node.addEventListener('pointermove', at);
    node.addEventListener('pointerleave', tipHide);
    node.addEventListener('focus', function () {
      var r = node.getBoundingClientRect(), c = build();
      if (c) tipShow(r.left + Math.min(r.width, 160), r.top + r.height / 2, c.title, c.rows);
    });
    node.addEventListener('blur', tipHide);
  }

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
    var m = { t: 12, r: o.endLabels ? 68 : 16, b: 28, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var sx = function (v) { return m.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * iw; };
    var sy = function (v) { return m.t + (1 - (v - o.y[0]) / (o.y[1] - o.y[0])) * ih; };

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', tabindex: 0, 'aria-label': o.aria }, host);

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
        s.pts.forEach(function (p) { if (p.y != null) sv('circle', { class: 'dot ' + s.cls + ' plain', cx: sx(p.x), cy: sy(p.y), r: 3 }, svg); });
      } else {
        sv('path', { class: 'line ' + s.cls, style: s.style, d: d }, svg);
      }
      var last = s.pts.filter(function (p) { return p.y != null; }).pop();
      if (!s.dotsOnly) sv('circle', { class: 'dot ' + s.cls, style: s.style, cx: sx(last.x), cy: sy(last.y), r: 4 }, svg);
      if (o.endLabels) {
        sv('text', { class: 'lbl', x: sx(last.x) + 10, y: sy(last.y) + 4, text: s.short || s.label }, svg);
      }
    });

    (o.markers || []).forEach(function (mk) {
      sv('circle', { class: 'dot ' + mk.cls, style: mk.style, cx: sx(mk.x), cy: sy(mk.y), r: 5 }, svg);
      var anchor = sx(mk.x) > W - 90 ? 'end' : sx(mk.x) < m.l + 70 ? 'start' : 'middle';
      sv('text', { class: 'lbl', x: sx(mk.x), y: sy(mk.y) + mk.dy, 'text-anchor': anchor, text: mk.text }, svg);
    });

    // Hover layer: a crosshair snaps to the nearest x; the tooltip lists every series at that x.
    var maps = o.series.map(function (s) { var mp = new Map(); s.pts.forEach(function (p) { mp.set(p.x, p); }); return mp; });
    var xs = Array.from(new Set([].concat.apply([], o.series.map(function (s) { return s.pts.map(function (p) { return p.x; }); })))).sort(function (a, b) { return a - b; });
    var cross = sv('line', { class: 'cross', y1: m.t, y2: m.t + ih, visibility: 'hidden' }, svg);
    var dots = o.series.map(function (s) { return sv('circle', { class: 'dot ' + s.cls, style: s.style, r: 5, visibility: 'hidden' }, svg); });
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
        dots[k].setAttribute('cx', sx(x)); dots[k].setAttribute('cy', sy(p.y)); dots[k].setAttribute('visibility', 'visible');
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

  // Re-render charts when their container width changes.
  var resizers = [];
  function watch(host, draw) {
    draw();
    var last = host.clientWidth;
    resizers.push(function () {
      if (!host.isConnected) return false;
      if (host.clientWidth !== last) { last = host.clientWidth; draw(); }
      return true;
    });
  }
  var rafId = 0;
  window.addEventListener('resize', function () {
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(function () { resizers = resizers.filter(function (f) { return f(); }); });
  });

  // ------------------------------------------------------------------ table views

  function tableView(summary, cols, rowsFn) {
    var d = h('details', { class: 'tv' }, h('summary', { text: summary }));
    d.addEventListener('toggle', function () {
      if (!d.open || d.querySelector('table')) return;
      var thead = h('thead', {}, h('tr', {}, cols.map(function (c) { return h('th', { class: c.r ? 'r' : '', scope: 'col', text: c.t }); })));
      var tbody = h('tbody', {}, rowsFn().map(function (r) {
        return h('tr', {}, r.map(function (v, i) { return h('td', { class: (cols[i].r ? 'r ' : '') + (cols[i].n ? 'n' : ''), text: v }); }));
      }));
      d.appendChild(h('div', { class: 'tv-wrap' }, h('table', {}, thead, tbody)));
    });
    return d;
  }

  // ------------------------------------------------------------------ state

  var D = {};          // loaded data
  var byTeam = {};     // ladder row by franchise
  var histBy = {};     // history row by franchise
  var pick = null;     // setter for the explorer

  function name(code) { return byTeam[code] ? byTeam[code].name : code; }

  // ------------------------------------------------------------------ headlines, motion helpers

  function setHeadline(id, text) { var n = $('#' + id); if (n && text) n.textContent = text; }

  // Counts a number up from zero once (skipped under prefers-reduced-motion). The final text is
  // always what assistive tech reads: the animated span is hidden and a plain copy is added.
  function countUp(to, dec, pre, suf) {
    var finalText = pre + to.toFixed(dec) + suf;
    var live = h('span', { 'aria-hidden': 'true', text: finalText });
    var wrap = h('span', {}, live, h('span', { class: 'sr', text: finalText }));
    if (REDUCED) return wrap;
    live.textContent = pre + (0).toFixed(dec) + suf;
    var t0 = null, dur = 900;
    function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      live.textContent = p < 1 ? pre + (to * e).toFixed(dec) + suf : finalText;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    return wrap;
  }

  // Bars grow once, when their section first scrolls into view (CSS does the easing).
  function initReveal() {
    if (REDUCED || !('IntersectionObserver' in window)) return;
    document.documentElement.classList.add('anim');
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
    document.querySelectorAll('main > .sec').forEach(function (sec) { io.observe(sec); });
  }

  // ------------------------------------------------------------------ header

  function renderHeader() {
    var meta = D.meta, teams = D.ladder.teams;
    var top = teams.filter(function (t) { return t.rank === 1; })[0];
    var riser = teams.slice().sort(function (a, b) { return b.rating_change - a.rating_change || a.rank - b.rank; })[0];
    var sc = meta.scorecard;

    var st = $('#status');
    clear(st);
    st.append('Through ' + meta.season + ' Week ' + meta.week + ' · updated Wednesdays');
    st.append(h('br'));
    st.append(h('span', { class: 'note', text: 'Ratings as of ' + longDate(meta.as_of) }));

    var tiles = $('#tiles');
    clear(tiles);
    tiles.append(
      tile('#1 team', [top.team, h('small', {}, countUp(top.rating, 1, '', ''))], top.name + ', ' + ordinalSpots(top.rank_change) + ' this week.'),
      tile('Biggest riser this week', [riser.team, h('small', {}, '\u25b2 ', countUp(riser.rating_change, 1, '', ''))],
        riser.name + ', now #' + riser.rank + ' (' + ordinalSpots(riser.rank_change) + ').'),
      tile('Home-field edge now', [countUp(meta.hfa_pts, 1, '', ''), h('small', { text: 'Elo pts' })],
        'Worth ' + pct(meta.hfa_win_pct, 1) + ' to the home team in a game between equal teams.'),
      tile('Elo vs Vegas pick accuracy', [
        h('span', { class: 'duo' }, countUp(sc.elo_accuracy_market_games * 100, 1, '', '%'), h('small', { text: 'Elo' })),
        h('span', { class: 'slash', text: '/' }),
        h('span', { class: 'duo' }, countUp(sc.market_accuracy * 100, 1, '', '%'), h('small', { text: 'Vegas' }))],
        sc.test_seasons[0] + '-' + sc.test_seasons[1] + ' regular seasons, ' + sc.n_market.toLocaleString('en-US') + ' games with a moneyline.', true)
    );
    document.title = 'NFLELO: ' + top.team + ' leads at ' + f1(top.rating) + ', through ' + meta.season + ' Week ' + meta.week;
    $('#credit').textContent = meta.credit;
  }

  function tile(label, big, sub, dual) {
    return h('div', { class: 'tile' },
      h('h3', { text: label }),
      h('div', { class: 'big' + (dual ? ' dual' : '') }, big),
      h('p', { class: 'sub', text: sub }));
  }

  // ------------------------------------------------------------------ ladder

  var GROUPS = [
    ['All', function () { return true; }],
    ['AFC', function (t) { return t.conf === 'AFC'; }],
    ['NFC', function (t) { return t.conf === 'NFC'; }]
  ].concat(['AFC East', 'AFC North', 'AFC South', 'AFC West', 'NFC East', 'NFC North', 'NFC South', 'NFC West'].map(function (d) {
    return [d, function (t) { return t.div === d; }];
  }));

  function renderLadder() {
    var teams = D.ladder.teams, active = 0;
    var maxDev = Math.max.apply(null, teams.map(function (t) { return Math.abs(t.rating - 1500); }));
    var R = Math.max(150, Math.ceil(maxDev / 50) * 50);
    var step = R <= 150 ? 50 : 100;
    var sparkAll = [].concat.apply([], teams.map(function (t) { return t.spark; }));

    var riser = teams.reduce(function (a, t) { return t.rating_change > a.rating_change ? t : a; }, teams[0]);
    var faller = teams.reduce(function (a, t) { return t.rating_change < a.rating_change ? t : a; }, teams[0]);
    var chips = $('#ladder-filters');
    var body = $('#ladder-body');

    function drawChips() {
      clear(chips);
      GROUPS.forEach(function (g, i) {
        chips.appendChild(h('button', { class: 'chip', type: 'button', 'aria-pressed': i === active ? 'true' : 'false', text: g[0],
          onclick: function () { active = i; drawChips(); drawRows(); } }));
      });
    }

    function axis() {
      var a = h('div', { class: 'axis', 'aria-hidden': 'true' });
      for (var v = -R; v <= R; v += step) a.appendChild(h('span', { style: 'left:' + ((v / R / 2 + 0.5) * 100) + '%', text: String(1500 + v) }));
      return a;
    }

    function spark(t) {
      var vals = t.spark, lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      if (hi - lo < 60) { var mid = (hi + lo) / 2; lo = mid - 30; hi = mid + 30; }
      var w = 168, hh = 28, px = function (i) { return 4 + i * (w - 8) / Math.max(1, vals.length - 1); };
      var py = function (v) { return hh - 4 - (v - lo) / (hi - lo) * (hh - 8); };
      var from = Math.max(0, t.spark_from - 1);
      var path = function (a, b) { var s = ''; for (var i = a; i <= b; i++) s += (i === a ? 'M' : 'L') + px(i).toFixed(1) + ' ' + py(vals[i]).toFixed(1); return s; };
      var svg = sv('svg', { class: 'spark', viewBox: '0 0 ' + w + ' ' + hh, 'aria-hidden': 'true' });
      if (from > 0) sv('path', { class: 'prior', d: path(0, from) }, svg);
      sv('path', { class: 'now', d: path(from, vals.length - 1) }, svg);
      sv('circle', { cx: px(vals.length - 1), cy: py(vals[vals.length - 1]), r: 3 }, svg);
      return svg;
    }

    function row(t) {
      var dev = t.rating - 1500, wPct = Math.min(50, Math.abs(dev) / R * 50);
      var fill = h('span', { class: 'fill mk ' + (dev >= 0 ? 'pos' : 'neg'), style: markVars(t.color) + ';' +
        (dev >= 0 ? 'left:50%;' : 'left:' + (50 - wPct) + '%;') + 'width:' + wPct + '%' });
      var up = t.rating_change > 0, flat = t.rating_change === 0;
      var label = t.name + ', rank ' + t.rank + ', rating ' + f1(t.rating) + ', ' + (flat ? 'no change' : (up ? 'up ' : 'down ') + f1(Math.abs(t.rating_change))) +
        ' this week, record ' + rec(t.w, t.l, t.t) + '. Open in the team explorer.';
      var tagText = t === riser && riser.rating_change > 0 ? 'Biggest rise' : t === faller && faller.rating_change < 0 ? 'Biggest fall' : '';
      if (tagText) label = tagText + '. ' + label;
      return h('li', {}, h('button', { class: 'lrow' + (t.rank <= 3 ? ' top' : '') + (tagText ? ' has-tag' : ''), type: 'button', 'aria-label': label, onclick: function () { pick(t.team, true); } },
        h('span', { class: 'rank', text: t.rank }),
        h('span', { class: 'team' }, h('span', { class: 'abbr', text: t.team }), h('span', { class: 'nm', text: t.name }), h('span', { class: 'rec-m', text: rec(t.w, t.l, t.t) }),
          tagText ? h('span', { class: 'tag-fill' + (tagText === 'Biggest fall' ? ' fall' : '') }, h('span', { class: 'full', text: tagText }), h('span', { class: 'short', text: tagText === 'Biggest fall' ? 'Fall' : 'Rise' })) : null),
        h('span', { class: 'bar' }, fill),
        h('span', { class: 'val', text: f1(t.rating) }),
        h('span', { class: 'chg ' + (flat ? 'flat' : up ? 'up' : 'down'), text: flat ? '-' : (up ? '▲ ' : '▼ ') + f1(Math.abs(t.rating_change)) }),
        h('span', { class: 'mv ' + (t.rank_change === 0 ? 'flat' : t.rank_change > 0 ? 'up' : 'down'), text: t.rank_change === 0 ? '-' : signed(t.rank_change, 0) }),
        h('span', { class: 'rec', text: rec(t.w, t.l, t.t) }),
        spark(t)));
    }

    function drawRows() {
      var list = teams.filter(GROUPS[active][1]);
      clear(body);
      body.removeAttribute('aria-busy');
      body.appendChild(h('div', { class: 'ladder' },
        h('div', { class: 'lhead', 'aria-hidden': 'true' },
          h('span', { text: '#' }), h('span', { text: 'Team' }), axis(),
          h('span', { class: 'r', text: 'Rating' }), h('span', { class: 'r', text: 'Week' }),
          h('span', { class: 'r hide-m', text: 'Rank' }), h('span', { class: 'r hide-m', text: 'W-L' }), h('span', { class: 'hide-m sp', text: 'Last 17' })),
        h('ol', {}, list.map(row))));
      body.appendChild(h('p', { class: 'fnote', text: 'Week is the rating change over the last 7 days; Rank is places gained or lost. The trend line covers each team\'s last 17 games, with this season in blue.' }));
    }

    drawChips();
    drawRows();
  }

  // ------------------------------------------------------------------ this week

  function lineText(home, away, spread) {
    if (spread == null) return 'No line';
    if (Math.abs(spread) < 0.05) return 'Pick\'em';
    return spread > 0 ? home + ' -' + f1(spread) : away + ' -' + f1(-spread);
  }

  function renderWeek() {
    var u = D.upcoming, body = $('#week-body');
    clear(body);
    body.removeAttribute('aria-busy');
    if (!u.week || !u.games.length) {
      body.appendChild(h('p', { class: 'note', text: 'No games are left on the schedule.' }));
      return;
    }
    var left = u.games.length;

    $('#week-deck').textContent = 'Week ' + u.week + (u.played ? ': ' + plural(left, 'game', 'games') + ' still to play, ' + u.played + ' already in the ratings' : ': ' + plural(left, 'game', 'games')) +
      '. Elo win chance and point spread next to the Vegas line.';

    var grid = h('div', { class: 'games' });
    u.games.forEach(function (g) {
      var home = g.p_home, away = 1 - g.p_home;
      var tag = g.flagged ? 'Gap ' + f1(Math.abs(g.diff)) : (g.neutral ? 'Neutral site' : '');
      var label = g.away + ' at ' + g.home + '. Elo gives ' + g.home + ' ' + pct(home) + '. Elo line ' + lineText(g.home, g.away, g.elo_spread) +
        (g.vegas_spread != null ? ', Vegas line ' + lineText(g.home, g.away, g.vegas_spread) : '') + '.';
      var lines = h('dl', { class: 'lines' },
        h('div', {}, h('dt', { text: 'Elo line' }), h('dd', { text: lineText(g.home, g.away, g.elo_spread) })),
        g.vegas_spread != null ? h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: lineText(g.home, g.away, g.vegas_spread) })) : h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: 'None yet' })),
        g.diff != null ? h('div', {}, h('dt', { text: 'Gap' }), h('dd', { text: f1(Math.abs(g.diff)) })) : null);
      var card = h('article', { class: 'game', 'aria-label': label },
        h('div', { class: 'when' }, h('span', { text: shortDay(g.date) + (g.time ? ', ' + clock(g.time) : '') }),
          tag ? h('span', { class: g.flagged ? 'tag' : '', text: tag }) : null),
        h('div', { class: 'vs' }, g.away, h('span', { class: 'at', text: 'at' }), g.home),
        h('div', { class: 'pbar', role: 'img', 'aria-hidden': 'true' },
          h('i', { class: 'mk', style: markVars(byTeam[g.away].color) + ';width:' + (away * 100).toFixed(1) + '%' }),
          h('i', { class: 'mk', style: markVars(byTeam[g.home].color) + ';width:' + (home * 100).toFixed(1) + '%' })),
        h('div', { class: 'plabels' },
          h('span', {}, g.away + ' ', h('b', { text: pct(away) })),
          h('span', {}, h('b', { text: pct(home) }), ' ' + g.home)),
        lines);
      grid.appendChild(card);
    });
    body.appendChild(grid);
    body.appendChild(h('p', { class: 'fnote', text: 'Lines name the favorite and the points it is favored by. Elo spread is the rating gap (with home-field edge, none at neutral sites) divided by 25. Kickoff times are Eastern.' }));
  }

  // ------------------------------------------------------------------ explorer

  var YEAR_TICKS = [1970, 1980, 1990, 2000, 2010, 2020];

  function renderExplorer() {
    var sel = $('#team-select'), body = $('#explorer-body');
    var teams = D.history.teams.slice().sort(function (a, b) { return a.name < b.name ? -1 : 1; });
    teams.forEach(function (t) { sel.appendChild(h('option', { value: t.team, text: t.name })); });
    var all = [].concat.apply([], D.history.teams.map(function (t) { return t.r; }));
    var yLo = Math.floor((Math.min.apply(null, all) - 20) / 50) * 50, yHi = Math.ceil((Math.max.apply(null, all) + 20) / 50) * 50;
    var xMax = D.meta.season + 1;

    function draw(code) {
      var t = histBy[code], L = byTeam[code];
      clear(body);
      body.removeAttribute('aria-busy');
      var pts = t.s.map(function (s, i) { return { x: s + (t.w[i] - 1) / 19, y: t.r[i], s: s, w: t.w[i], i: i }; });
      var host = h('div', { class: 'chart' });
      var style = markVars(t.color);
      var bestIdx = lastIndexOfSeason(t, t.best.season), worstIdx = lastIndexOfSeason(t, t.worst.season);
      var left = h('div', {}), split = h('div', { class: 'split' }, left);
      left.appendChild(host);
      body.appendChild(split);
      setHeadline('explorer-h', D.headlines.explorer[code]);
      watch(host, function () {
        lineChart(host, {
          series: [{ id: code, label: t.name, cls: 'mk', style: style, pts: pts }],
          x: [1970, xMax], y: [yLo, yHi],
          height: function (w) { return w < 560 ? 280 : w > 900 ? 460 : 380; },
          xTicks: YEAR_TICKS, yTicks: ticksFor(yLo, yHi, 6),
          xFmt: String, yFmt: String,
          ref: { y: 1500, label: '1500 average' },
          aria: t.name + ' Elo rating by game week since ' + t.first_season + '. Best season ' + t.best.season + ' at ' + f1(t.best.rating) + ', worst ' + t.worst.season + ' at ' + f1(t.worst.rating) + '. Focus the chart and use the arrow keys to read values.',
          markers: [
            { x: pts[bestIdx].x, y: pts[bestIdx].y, cls: 'mk', style: style, text: 'Best ' + t.best.season, dy: -12 },
            { x: pts[worstIdx].x, y: pts[worstIdx].y, cls: 'mk', style: style, text: 'Worst ' + t.worst.season, dy: 20 }
          ],
          tip: function (x, rows) {
            var p = rows[0].p;
            return { title: p.s + ' Week ' + p.w, rows: [{ value: f1(p.y), label: signed(p.y - 1500) + ' vs 1500', cls: 'mk', style: style }] };
          }
        });
      });

      var cur = L;
      split.appendChild(h('dl', { class: 'facts' },
        fact('Rating now', f1(cur.rating), 'Rank ' + cur.rank + ' of 32'),
        fact('Best season', String(t.best.season), f1(t.best.rating) + ', ' + rec(t.best.w, t.best.l, t.best.t)),
        fact('Worst season', String(t.worst.season), f1(t.worst.rating) + ', ' + rec(t.worst.w, t.worst.l, t.worst.t)),
        fact('All-time record', rec(t.all_time.w, t.all_time.l, t.all_time.t), winPct(t.all_time.w, t.all_time.l, t.all_time.t) + ' of ' + t.all_time.games.toLocaleString('en-US') + ' games')));
      var notes = 'The horizontal line is the 1500 league average. Regular season only. Season-end ratings exclude ' + D.meta.season + (D.meta.season_partial ? ', which is still in progress.' : '.');
      if (t.first_season > 1970) notes += ' ' + t.name + ' first played in ' + t.first_season + ' and started at ' + D.meta.config.expansion_start + '.';
      left.appendChild(h('p', { class: 'fnote', text: notes }));
      left.appendChild(tableView('View season-end ratings as a table', [{ t: 'Season' }, { t: 'Rating', r: 1, n: 1 }, { t: 'Vs 1500', r: 1, n: 1 }], function () {
        var rows = [], seen = {};
        for (var i = t.s.length - 1; i >= 0; i--) {
          var s = t.s[i];
          if (seen[s] || (D.meta.season_partial && s === D.meta.season)) continue;
          seen[s] = 1;
          rows.push([String(s), f1(t.r[i]), signed(t.r[i] - 1500)]);
        }
        return rows;
      }));
    }

    sel.addEventListener('change', function () { draw(sel.value); });
    pick = function (code, scroll) {
      sel.value = code;
      draw(code);
      if (scroll) {
        var sec = $('#explorer');
        sec.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' });
        sel.focus({ preventScroll: true });
      }
    };
    var first = D.ladder.teams[0].team;
    sel.value = first;
    draw(first);
  }

  function lastIndexOfSeason(t, season) {
    for (var i = t.s.length - 1; i >= 0; i--) if (t.s[i] === season) return i;
    return t.s.length - 1;
  }

  function fact(label, value, small) {
    return h('div', { class: 'fact' }, h('dt', { text: label }), h('dd', {}, value, h('small', { text: small })));
  }

  // ------------------------------------------------------------------ luck

  // Three luckiest and three unluckiest teams for the selected season (from the same JSON rows as the chart).
  function luckyBox(s) {
    var rows = s.teams;
    var lucky = rows.filter(function (r) { return r.luck > 0; }).slice(0, 3);
    var unlucky = rows.filter(function (r) { return r.luck < 0; }).slice(-3).reverse();
    function list(title, items, cls) {
      return h('div', {}, h('h3', { text: title }), h('ol', {}, items.map(function (r) {
        return h('li', {}, h('span', { class: 't' }, h('b', { text: r.team }), byTeam[r.team].name),
          h('span', { class: 'v ' + cls, text: signed(r.luck) }));
      })));
    }
    return h('div', { class: 'lucky' }, list('Luckiest, wins above expected', lucky, 'up'), list('Unluckiest, wins below expected', unlucky, 'down'));
  }

  function renderLuck() {
    var body = $('#luck-body'), toggle = $('#luck-toggle');
    var keys = Object.keys(D.luck.seasons).sort();
    var active = keys[keys.length - 1];
    var R = Math.max(2, Math.ceil(Math.max.apply(null, [].concat.apply([], keys.map(function (k) { return D.luck.seasons[k].teams.map(function (r) { return Math.abs(r.luck); }); })))));

    function labelFor(k) {
      return k + (+k === D.meta.season && D.meta.season_partial ? ' to date' : ' full season');
    }

    function drawToggle() {
      clear(toggle);
      keys.forEach(function (k) {
        toggle.appendChild(h('button', { class: 'chip', type: 'button', 'aria-pressed': k === active ? 'true' : 'false', text: labelFor(k),
          onclick: function () { active = k; drawToggle(); drawBars(); } }));
      });
    }

    function drawBars() {
      var s = D.luck.seasons[active];
      clear(body);
      body.removeAttribute('aria-busy');
      var head = h('div', { class: 'lk-head', 'aria-hidden': 'true' }, h('span', { text: 'Team' }),
        h('span', { class: 'ends' }, h('span', { text: 'Fewer wins than expected' }), h('span', { text: 'More wins than expected' })), h('span', { class: 'r', style: 'text-align:right', text: 'Luck' }));
      var list = h('ol', { 'aria-label': 'Luck by team, ' + labelFor(active) });
      s.teams.forEach(function (r) {
        var t = byTeam[r.team], w = Math.min(50, Math.abs(r.luck) / R * 50);
        var item = h('li', { class: 'lk-row', tabindex: 0, 'aria-label': t.name + ': ' + r.actual + ' wins, ' + f1(r.expected) + ' expected, luck ' + signed(r.luck) },
          h('span', { class: 'abbr', text: r.team }),
          h('span', { class: 'bar' }, h('span', { class: 'fill mk ' + (r.luck >= 0 ? 'pos' : 'neg'), style: markVars(t.color) + ';' + (r.luck >= 0 ? 'left:50%' : 'left:' + (50 - w) + '%') + ';width:' + w + '%' })),
          h('span', { class: 'val', text: signed(r.luck) }));
        bindTip(item, function () {
          return { title: t.name + ', ' + plural(r.games, 'game', 'games'), rows: [
            { value: signed(r.luck), label: 'wins vs expected' }, { value: String(r.actual), label: 'actual wins' }, { value: f1(r.expected), label: 'expected wins' }] };
        });
        list.appendChild(item);
      });
      setHeadline('luck-h', D.headlines.luck[active]);
      var left = h('div', {}, h('div', { class: 'lk' }, head, list), h('p', { class: 'fnote', text: D.luck.note + ' Both seasons share one scale.' }));
      body.appendChild(h('div', { class: 'split' }, left, luckyBox(s)));
      left.appendChild(tableView('View luck as a table', [{ t: 'Team' }, { t: 'Games', r: 1, n: 1 }, { t: 'Actual wins', r: 1, n: 1 }, { t: 'Expected wins', r: 1, n: 1 }, { t: 'Luck', r: 1, n: 1 }], function () {
        return s.teams.map(function (r) { return [r.team, String(r.games), String(r.actual), f1(r.expected), signed(r.luck)]; });
      }));
    }

    drawToggle();
    drawBars();
  }

  // ------------------------------------------------------------------ history: tapestry + parity

  function heatFill(v) {
    var dev = (v - 1500) / 250, a = Math.min(1, Math.abs(dev)), p = Math.round(Math.pow(a, 0.8) * 100);
    return 'color-mix(in oklab, var(--h-mid) ' + (100 - p) + '%, var(' + (dev >= 0 ? '--h-pos' : '--h-neg') + ') ' + p + '%)';
  }

  function renderHistory() {
    var body = $('#history-body'), T = D.tapestry;
    clear(body);
    body.removeAttribute('aria-busy');

    var scale = h('div', { class: 'scale', 'aria-label': 'Color scale: red below 1500, gray at 1500, blue above' }, h('span', { text: 'Rating' }));
    [1250, 1350, 1500, 1650, 1750].forEach(function (v) {
      scale.appendChild(h('span', { class: 'sw' }, h('i', { style: 'background:' + heatFill(v) }), String(v)));
    });
    var wrap = h('div', { class: 'heat-wrap' });
    var heat = h('div', { class: 'heat' });
    wrap.appendChild(heat);
    body.appendChild(wrap);
    body.appendChild(scale);
    body.appendChild(h('p', { class: 'fnote', text: 'Each cell is a franchise\'s rating at the end of that regular season. Empty cells are seasons before the franchise existed.' + (T.partial_season ? ' The ' + T.partial_season + ' column is the rating so far.' : '') }));

    var nT = T.teams.length, nS = T.seasons.length;
    var CH = 14, LBL = 38, TOP = 22, GAP_DIV = 6, GAP_CONF = 12;
    var rowY = [], totalH = 0;
    function layout() {
      rowY = []; var y = TOP;
      T.teams.forEach(function (code, i) {
        if (i > 0) y += (i % 16 === 0) ? GAP_CONF : (i % 4 === 0 ? GAP_DIV : 0);
        rowY.push(y); y += CH;
      });
      totalH = y + 4;
    }

    function draw() {
      clear(heat);
      var avail = wrap.clientWidth - LBL;
      var cw = Math.max(13, Math.floor(avail / nS));
      CH = Math.min(24, Math.max(14, Math.round(cw * 0.6)));   // cells grow with the width
      layout();
      var W = LBL + cw * nS;
      var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + totalH, width: W, height: totalH, role: 'img', tabindex: 0,
        'aria-label': 'Heatmap of end-of-season Elo ratings, ' + nT + ' franchises by ' + nS + ' seasons from ' + T.seasons[0] + ' to ' + T.seasons[nS - 1] + '. Use the arrow keys to move between cells.' }, heat);
      T.seasons.forEach(function (s, c) {
        if (s % 10 === 0 || s === T.seasons[nS - 1]) sv('text', { class: 'tick a-mid', x: LBL + c * cw + cw / 2, y: 12, text: s === T.partial_season ? s + '*' : String(s) }, svg);
      });
      T.teams.forEach(function (code, r) {
        sv('text', { class: 'tick a-end', x: LBL - 8, y: rowY[r] + CH - 3, text: code }, svg);
        T.values[r].forEach(function (v, c) {
          if (v == null) return;
          sv('rect', { class: 'cell', x: LBL + c * cw, y: rowY[r], width: cw - 1, height: CH - 1, style: 'fill:' + heatFill(v) }, svg);
        });
      });
      var cursor = sv('rect', { class: 'cursor', width: cw - 1, height: CH - 1, visibility: 'hidden' }, svg);
      var pos = { r: 0, c: nS - 1 };

      function show(r, c, cx, cy) {
        pos = { r: r, c: c };
        var v = T.values[r][c], code = T.teams[r], season = T.seasons[c];
        cursor.setAttribute('x', LBL + c * cw); cursor.setAttribute('y', rowY[r]); cursor.setAttribute('visibility', 'visible');
        if (cx == null) { var b = svg.getBoundingClientRect(); cx = b.left + (LBL + c * cw + cw) * (b.width / W); cy = b.top + rowY[r] * (b.height / totalH); }
        tipShow(cx, cy, name(code) + ', ' + season + (season === T.partial_season ? ' (so far)' : ''),
          v == null ? [{ value: 'No team yet', label: '' }] : [{ value: f1(v), label: signed(v - 1500) + ' vs 1500' }]);
      }
      function hide() { cursor.setAttribute('visibility', 'hidden'); tipHide(); }

      svg.addEventListener('pointermove', function (e) {
        var b = svg.getBoundingClientRect(), px = (e.clientX - b.left) * (W / b.width), py = (e.clientY - b.top) * (totalH / b.height);
        var c = Math.floor((px - LBL) / cw), r = -1;
        for (var i = 0; i < nT; i++) if (py >= rowY[i] && py < rowY[i] + CH) { r = i; break; }
        if (c < 0 || c >= nS || r < 0) return hide();
        show(r, c, e.clientX, e.clientY);
      });
      svg.addEventListener('pointerleave', hide);
      svg.addEventListener('focus', function () { if (!svg.matches(':hover')) show(pos.r, pos.c); });
      svg.addEventListener('blur', hide);
      svg.addEventListener('keydown', function (e) {
        var r = pos.r, c = pos.c;
        if (e.key === 'ArrowLeft') c = Math.max(0, c - 1);
        else if (e.key === 'ArrowRight') c = Math.min(nS - 1, c + 1);
        else if (e.key === 'ArrowUp') r = Math.max(0, r - 1);
        else if (e.key === 'ArrowDown') r = Math.min(nT - 1, r + 1);
        else if (e.key === 'Escape') return hide();
        else return;
        e.preventDefault();
        show(r, c);
      });
    }
    var last = wrap.clientWidth;
    draw();
    resizers.push(function () {
      if (!wrap.isConnected) return false;
      if (wrap.clientWidth !== last) { last = wrap.clientWidth; draw(); }
      return true;
    });

    // Parity: spread of end-of-season ratings. Lower means a tighter league.
    body.appendChild(h('h3', { class: 'sub-h', text: 'League parity by season' }));
    body.appendChild(h('p', { class: 'deck', style: 'margin-top:-6px;margin-bottom:12px', text: 'Standard deviation of end-of-season ratings. A lower line means the league was more bunched together.' }));
    var pHost = h('div', { class: 'chart' });
    body.appendChild(pHost);
    var pts = T.seasons.map(function (s, i) { return { x: s, y: T.parity[i], count: T.teams_count[i] }; }).filter(function (p) { return p.x !== T.partial_season; });
    var pLo = 60, pHi = 150;
    watch(pHost, function () {
      lineChart(pHost, {
        series: [{ id: 'parity', label: 'Rating spread', cls: 's1', pts: pts }],
        x: [pts[0].x, pts[pts.length - 1].x], y: [pLo, pHi],
        height: function (w) { return w < 560 ? 220 : 280; },
        xTicks: YEAR_TICKS.concat([2025]), yTicks: ticksFor(pLo, pHi, 4), xFmt: String, yFmt: String,
        aria: 'Line chart of the spread of end-of-season Elo ratings by season, from ' + f1(pts[0].y) + ' in ' + pts[0].x + ' to ' + f1(pts[pts.length - 1].y) + ' in ' + pts[pts.length - 1].x + '.',
        tip: function (x, rows) { var p = rows[0].p; return { title: String(x), rows: [{ value: f1(p.y), label: 'rating spread', cls: 's1' }, { value: String(p.count), label: 'teams' }] }; }
      });
    });
    body.appendChild(h('p', { class: 'fnote', text: 'Expansion years read high because new teams start at ' + D.meta.config.expansion_start + ', well below the average.' }));
    body.appendChild(tableView('View parity as a table', [{ t: 'Season' }, { t: 'Rating spread', r: 1, n: 1 }, { t: 'Teams', r: 1, n: 1 }], function () {
      return pts.slice().reverse().map(function (p) { return [String(p.x), f1(p.y), String(p.count)]; });
    }));
  }

  // ------------------------------------------------------------------ records

  function gameCell(g) {
    var awayWon = g.winner === g.away;
    return h('span', {},
      h('span', { class: awayWon ? 'w' : 'l', text: g.away }), ' ', h('span', { class: awayWon ? 'w-sc' : 'l-sc', text: String(g.away_score) }),
      ' at ',
      h('span', { class: awayWon ? 'l' : 'w', text: g.home }), ' ', h('span', { class: awayWon ? 'l-sc' : 'w-sc', text: String(g.home_score) }));
  }

  function recTable(title, sub, cols, rows) {
    var thead = h('thead', {}, h('tr', {}, cols.map(function (c) { return h('th', { class: c.r ? 'r' : '', scope: 'col', text: c.t }); })));
    var tbody = h('tbody', {}, rows.map(function (r) {
      return h('tr', {}, r.map(function (v, i) { return h('td', { class: (cols[i].r ? 'r ' : '') + (cols[i].n ? 'n' : '') }, v); }));
    }));
    return h('div', {}, h('h3', { text: title }), h('p', { class: 'sub', text: sub }), h('div', { class: 'tbl-wrap' }, h('table', {}, thead, tbody)));
  }

  function renderRecords() {
    var R = D.records, body = $('#records-body');
    clear(body);
    body.removeAttribute('aria-busy');
    var seasonCols = [{ t: 'Team' }, { t: 'Season', n: 1 }, { t: 'Rating', r: 1, n: 1 }, { t: 'Record', r: 1, n: 1 }];
    var seasonRow = function (s) { return [h('b', { text: s.team }), String(s.season), f1(s.rating), rec(s.w, s.l, s.t)]; };
    body.appendChild(h('div', { class: 'rec4' },
      recTable('Biggest upsets', 'Lowest pre-game win chance for the team that won. Winner in bold.',
        [{ t: 'Date', n: 1 }, { t: 'Game', n: 0 }, { t: 'Win chance', r: 1, n: 1 }],
        R.upsets.map(function (g) { return [g.date, gameCell(g), pct(g.p_win, 1)]; })),
      recTable('Biggest rating swings', 'Rating points the winner gained in one game.',
        [{ t: 'Date', n: 1 }, { t: 'Game', n: 0 }, { t: 'Gain', r: 1, n: 1 }],
        R.swings.map(function (g) { return [g.date, gameCell(g), '+' + f1(g.gain)]; })),
      recTable('Best team-seasons', 'Highest end-of-season rating.', seasonCols, R.best_seasons.map(seasonRow)),
      recTable('Worst team-seasons', 'Lowest end-of-season rating.', seasonCols, R.worst_seasons.map(seasonRow))));
    body.appendChild(h('p', { class: 'fnote', text: 'Upsets include playoff games; the other tables use regular-season games, because playoff games do not change ratings.' }));
  }

  // ------------------------------------------------------------------ scorecard

  function renderScorecard() {
    var body = $('#scorecard-body'), S = D.scorecard.seasons, sc = D.meta.scorecard, cfg = D.meta.config;
    clear(body);
    body.removeAttribute('aria-busy');
    var span = sc.test_seasons[0] + '-' + sc.test_seasons[1];

    body.appendChild(h('div', { class: 'score' },
      h('div', {}, h('h3', { text: 'Brier score, lower is better (' + span + ')' }),
        h('div', { class: 'row' },
          h('span', {}, h('b', { class: 'num', text: sc.elo_brier.toFixed(4) }), 'Elo v2'),
          h('span', {}, h('b', { class: 'num', text: sc.market_brier.toFixed(4) }), 'Vegas market'),
          sc.legacy_brier ? h('span', {}, h('b', { class: 'num', text: sc.legacy_brier.toFixed(4) }), '2025 model') : null)),
      h('div', {}, h('h3', { text: 'Correct picks' }),
        h('div', { class: 'row' },
          h('span', {}, h('b', { class: 'num', text: pct(sc.elo_accuracy_market_games, 1) }), 'Elo v2'),
          h('span', {}, h('b', { class: 'num', text: pct(sc.market_accuracy, 1) }), 'Vegas favorite'))),
      h('div', {}, h('h3', { text: 'Gap to the market' }),
        h('div', { class: 'row' }, h('span', {}, h('b', { class: 'num', text: '+' + sc.brier_gap_vs_market.toFixed(4) }), 'Brier, plus or minus ' + sc.brier_gap_se.toFixed(4))))));

    var pair = h('div', { class: 'pair' });
    var brHost = h('div', { class: 'chart' }), hfaHost = h('div', { class: 'chart' });
    pair.appendChild(h('div', {}, h('h3', { text: 'Brier score by season' }), h('p', { class: 'sub', text: 'Elo against the Vegas moneyline, vig removed. Lower is better.' }),
      legend([['s1', 'Elo v2'], ['s2', 'Vegas market']]), brHost,
      h('p', { class: 'fnote', text: 'No betting lines in the data for 2008; 2006 is partial.' }),
      tableView('View Brier scores as a table', [{ t: 'Season' }, { t: 'Games', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }, { t: 'Market', r: 1, n: 1 }], function () {
        return S.filter(function (r) { return r.season >= firstMkt; }).reverse().map(function (r) {
          return r.market_brier == null ? [String(r.season), '-', '-', '-'] : [String(r.season), String(r.n_market), r.elo_brier.toFixed(4), r.market_brier.toFixed(4)];
        });
      })));
    pair.appendChild(h('div', {}, h('h3', { text: 'Home-field edge by season' }), h('p', { class: 'sub', text: 'Chance the home team wins between equal teams, from Elo\'s learned edge, next to how often home teams actually won.' }),
      legend([['s1', 'Elo learned edge'], ['sm dot', 'Actual home win rate']]), hfaHost,
      tableView('View home-field edge as a table', [{ t: 'Season' }, { t: 'Elo edge (pts)', r: 1, n: 1 }, { t: 'Implied win %', r: 1, n: 1 }, { t: 'Actual win %', r: 1, n: 1 }], function () {
        return S.slice().reverse().map(function (r) { return [String(r.season), f1(r.hfa_pts), pct(r.hfa_win_pct, 1), pct(r.home_win_rate, 1)]; });
      })));
    body.appendChild(pair);

    var firstMkt = S.filter(function (r) { return r.market_brier != null; })[0].season;
    var mk = S.filter(function (r) { return r.season >= firstMkt; });   // keeps seasons with no lines as gaps
    watch(brHost, function () {
      lineChart(brHost, {
        series: [
          { id: 'elo', label: 'Elo v2', short: 'Elo', cls: 's1', pts: mk.map(function (r) { return { x: r.season, y: r.elo_brier }; }) },
          { id: 'mkt', label: 'Vegas market', short: 'Vegas', cls: 's2', pts: mk.map(function (r) { return { x: r.season, y: r.market_brier }; }) }],
        x: [mk[0].season, mk[mk.length - 1].season], y: [0.19, 0.25], endLabels: true,
        height: function (w) { return w < 560 ? 240 : w > 1300 ? 360 : 300; },
        xTicks: [2006, 2010, 2015, 2020, 2025], yTicks: [0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25], xFmt: String, yFmt: function (v) { return v.toFixed(2); },
        aria: 'Line chart of Brier score by season, Elo against the Vegas market, ' + mk[0].season + ' to ' + mk[mk.length - 1].season + '. The market is lower in most seasons. Use the table below for exact values.',
        tip: function (x, rows) { return { title: String(x), rows: rows.map(function (r) { return { value: r.p.y == null ? 'No lines' : r.p.y.toFixed(4), label: r.s.label, cls: r.s.cls }; }) }; }
      });
    });
    watch(hfaHost, function () {
      lineChart(hfaHost, {
        series: [
          { id: 'hfa', label: 'Elo learned edge', short: 'Elo', cls: 's1', pts: S.map(function (r) { return { x: r.season, y: r.hfa_win_pct * 100, pts: r.hfa_pts }; }) },
          { id: 'act', label: 'Actual home win rate', short: 'Actual', cls: 'sm', dotsOnly: true, pts: S.map(function (r) { return { x: r.season, y: r.home_win_rate * 100 }; }) }],
        x: [S[0].season, S[S.length - 1].season], y: [48, 64], endLabels: true,
        height: function (w) { return w < 560 ? 240 : w > 1300 ? 360 : 300; },
        xTicks: [1970, 1980, 1990, 2000, 2010, 2020], yTicks: [48, 52, 56, 60, 64], xFmt: String, yFmt: function (v) { return v + '%'; },
        aria: 'Line chart of home-field edge by season from ' + S[0].season + ' to ' + S[S.length - 1].season + ', falling from about ' + pct(S[0].hfa_win_pct) + ' to about ' + pct(S[S.length - 1].hfa_win_pct) + '. Use the table below for exact values.',
        tip: function (x, rows) {
          return { title: String(x), rows: rows.map(function (r) {
            return { value: r.p.y.toFixed(1) + '%', label: r.s.label + (r.p.pts != null ? ' (' + f1(r.p.pts) + ' pts)' : ''), cls: r.s.cls };
          }) };
        }
      });
    });

    body.appendChild(h('div', { class: 'method' },
      h('div', {},
        h('h3', { text: 'How it works' }),
        h('p', {}, h('b', { text: 'Ratings. ' }), 'Every team starts at 1500, the league average. After each game the winner takes points from the loser. The bigger the surprise and the bigger the margin, the more points move. The base size is K = ' + cfg.k + '.'),
        h('p', {}, h('b', { text: 'Offseason. ' }), 'Between seasons every rating moves ' + Math.round(cfg.lam * 100) + '% of the way back to 1500 (lambda = ' + cfg.lam.toFixed(2) + ') because rosters turn over.'),
        h('p', {}, h('b', { text: 'Home field. ' }), 'The home team gets a rating bonus before each prediction, none at neutral sites. The bonus is learned: it started at ' + cfg.hfa_init + ' points in 1970, moves up or down as home teams win more or less than expected, and is ' + f1(D.meta.hfa_pts) + ' now. Elo spread is the rating gap divided by 25.'),
        h('p', {}, h('b', { text: 'Data. ' }), 'Game results from 1999 on come from nflverse. Results for 1970 to 1998, including playoffs, come from the FiveThirtyEight NFL Elo game file. Both are CC BY 4.0.'),
        h('p', {}, h('b', { text: 'New teams. ' }), 'Franchises that join after 1970 start at ' + cfg.expansion_start + '. The model was tuned on ' + sc.tune_seasons[0] + '-' + sc.tune_seasons[1] + ' and scored on ' + span + '.'),
        h('p', {}, h('b', { text: 'Limits. ' }), 'The Vegas line is still better: a Brier gap of ' + sc.brier_gap_vs_market.toFixed(4) + '. Elo does not know about injuries, starting quarterbacks, or weather.')),
      h('div', {},
        h('h3', { text: 'Links' }),
        h('ul', {},
          h('li', {}, h('a', { href: D.meta.repo, text: 'Code and method on GitHub' }), h('small', { text: 'github.com/walk-the-program/NFLELO' })),
          h('li', {}, h('a', { href: 'https://nflverse.nflverse.com/', text: 'nflverse' }), h('small', { text: 'Schedules, scores, and betting lines, 1999 on. CC BY 4.0.' })),
          h('li', {}, h('a', { href: 'https://github.com/fivethirtyeight/data', text: 'FiveThirtyEight' }), h('small', { text: 'NFL game results 1970\u20131998: FiveThirtyEight, CC BY 4.0.' })),
          h('li', {}, h('a', { href: 'data/ladder.json', text: 'Ladder data (JSON)' }), h('small', { text: 'Rebuilt every Wednesday' }))))));
  }

  function legend(items) {
    return h('div', { class: 'legend' }, items.map(function (it) { return h('span', {}, h('i', { class: it[0] }), it[1]); }));
  }

  // ------------------------------------------------------------------ nav highlight

  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.nav-links a'));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    if (!('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        links.forEach(function (a) { a.removeAttribute('aria-current'); });
        var a = map[en.target.id];
        if (a) {
          a.setAttribute('aria-current', 'true');
          var box = a.parentNode.parentNode;
          if (box.scrollWidth > box.clientWidth) box.scrollLeft = a.offsetLeft - 60;
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['ladder', 'week', 'explorer', 'luck', 'history', 'records', 'scorecard'].forEach(function (id) {
      var b = $('#' + id + '-body');
      if (!b) return;
      clear(b);
      b.removeAttribute('aria-busy');
      b.appendChild(h('p', { class: 'note err', text: 'The data files could not be loaded. Serve this folder over http, for example with python3 -m http.server --directory site.' }));
    });
    $('#status').textContent = 'Data unavailable';
  }

  Promise.all(FILES.map(function (f) {
    return fetch('data/' + f + '.json').then(function (r) { if (!r.ok) throw new Error(f + ' ' + r.status); return r.json(); });
  })).then(function (all) {
    FILES.forEach(function (f, i) { D[f] = all[i]; });
    D.ladder.teams.forEach(function (t) { byTeam[t.team] = t; });
    D.history.teams.forEach(function (t) { histBy[t.team] = t; });
    initReveal();
    renderHeader();
    renderLadder();
    renderWeek();
    renderExplorer();
    renderLuck();
    renderHistory();
    renderRecords();
    renderScorecard();
    setHeadline('ladder-h', D.headlines.ladder);
    setHeadline('week-h', D.headlines.week);
    setHeadline('history-h', D.headlines.history);
    setHeadline('records-h', D.headlines.records);
    setHeadline('scorecard-h', D.headlines.scorecard);
    initNav();
  }).catch(fail);
})();
