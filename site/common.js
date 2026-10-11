/* NFLELO shared front-end helpers, used by app.js (ratings page), models.js (models page) and the four model
   pages (plays.js, winprob.js, fourth.js, playcalling.js):
   DOM builders, the tooltip, resize watching, table views, legends, the section nav highlight and
   the one place that fetches data. No dependencies. Exposed as window.NFL. */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';

  // ------------------------------------------------------------------ data
  // Every data file is revalidated on each load (`no-cache`): after the weekly update, browsers must not
  // keep showing last week's numbers from a heuristic cache. Optional files resolve to null when missing.
  function getJSON(url, optional) {
    return fetch(url, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) { if (optional) return null; throw new Error(url + ' ' + r.status); }
      return r.json();
    }).catch(function (e) { if (optional) return null; throw e; });
  }

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

  // Chart area: a bordered well around a chart host (the host keeps its own width for drawing).
  function well(host) { return h('div', { class: 'well' }, host); }


  // ------------------------------------------------------------------ series marks
  // One shape per series so color is never the only cue: Elo circle, Model square, Vegas triangle, third series diamond.
  var SHAPE = { s1: 'circle', s3: 'square', s2: 'triangle', sm: 'diamond' };
  function markPath(shape, r) {
    if (shape === 'square') return 'M' + (-r) + ' ' + (-r) + 'H' + r + 'V' + r + 'H' + (-r) + 'Z';
    if (shape === 'triangle') return 'M0 ' + (-r * 1.2).toFixed(2) + 'L' + (r * 1.15).toFixed(2) + ' ' + (r * 0.9).toFixed(2) + 'L' + (-r * 1.15).toFixed(2) + ' ' + (r * 0.9).toFixed(2) + 'Z';
    if (shape === 'diamond') return 'M0 ' + (-r * 1.25).toFixed(2) + 'L' + (r * 1.25).toFixed(2) + ' 0L0 ' + (r * 1.25).toFixed(2) + 'L' + (-r * 1.25).toFixed(2) + ' 0Z';
    return 'M' + (-r) + ' 0A' + r + ' ' + r + ' 0 1 0 ' + r + ' 0A' + r + ' ' + r + ' 0 1 0 ' + (-r) + ' 0Z';
  }
  // Adds a series mark to an SVG parent at (cx, cy). extra: more classes, e.g. 'plain'.
  function mark(parent, series, cx, cy, r, extra) {
    return sv('path', { class: 'dot ' + series + (extra ? ' ' + extra : ''), d: markPath(SHAPE[series] || 'circle', r), transform: 'translate(' + cx + ' ' + cy + ')' }, parent);
  }
  function moveMark(el, cx, cy) { el.setAttribute('transform', 'translate(' + cx + ' ' + cy + ')'); }

  // ------------------------------------------------------------------ Halftone field dividers
  // A strip of the brand's Halftone field pattern (see brand.css .splash) before every section but the first.
  // Decorative only. Safe to call again after sections are inserted.
  function initSlabs() {
    var secs = document.querySelectorAll('main > .sec');
    Array.prototype.forEach.call(secs, function (sec, i) {
      if (i === 0) return;
      var prev = sec.previousElementSibling;
      if (prev && prev.classList.contains('splash-div')) return;
      var d = document.createElement('div');
      d.className = 'splash splash-div';
      d.setAttribute('aria-hidden', 'true');
      sec.parentNode.insertBefore(d, sec);
    });
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
    rows = rows.filter(Boolean);
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

  // ------------------------------------------------------------------ resize, table views, legends

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

  function legend(items) {
    return h('div', { class: 'legend' }, items.map(function (it) { return h('span', {}, h('i', { class: it[0] }), it[1]); }));
  }

  // ------------------------------------------------------------------ nav highlight

  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.nav-links a')).filter(function (a) { return a.getAttribute('href').charAt(0) === '#'; });
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

  // ------------------------------------------------------------------ shared by the four model pages
  // (plays, winprob, fourth, playcalling): formatting, segmented controls, hero tiles, interval charts,
  // credit and caveat blocks, the "data not available yet" state and the nav scroll.

  var REPO = 'https://github.com/walk-the-program/NFLELO';
  function repoUrl(path) { return REPO + '/blob/main/' + path; }
  function commitUrl(hash) { return REPO + '/commit/' + hash; }
  // Signed fixed-point text with an ASCII minus; a value that rounds to zero is shown as +0.
  function signed(x, d) { var r = Math.round(x * Math.pow(10, d)) / Math.pow(10, d); return (r >= 0 ? '+' : '-') + Math.abs(r).toFixed(d); }
  function commas(n) { return Number(n).toLocaleString('en-US'); }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  function ciText(ci, d) { return '[' + signed(ci.lo, d) + ', ' + signed(ci.hi, d) + ']'; }
  function setText(id, text) { var n = $('#' + id); if (n && text != null) n.textContent = text; return n; }
  function done(id) { var n = $('#' + id); if (n) n.removeAttribute('aria-busy'); return n; }

  // Segmented control (joined boxes, ink-filled selected segment). options: [{value, label, title}].
  // Returns {el, set(value), value(), disable(fn)}; onchange(value) fires on user choice only.
  function seg(label, options, value, onchange, cls) {
    var el = h('div', { class: 'b-seg ' + (cls || ''), role: 'group', 'aria-label': label });
    var cur = value, btns = {};
    options.forEach(function (o) {
      var b = h('button', { type: 'button', 'aria-pressed': o.value === cur ? 'true' : 'false', title: o.title, text: o.label });
      b.addEventListener('click', function () { if (b.disabled || o.value === cur) return; api.set(o.value); onchange(o.value); });
      btns[o.value] = b; el.appendChild(b);
    });
    var api = {
      el: el,
      value: function () { return cur; },
      set: function (v) { cur = v; options.forEach(function (o) { btns[o.value].setAttribute('aria-pressed', o.value === v ? 'true' : 'false'); }); },
      disable: function (fn) { options.forEach(function (o) { var off = !!fn(o.value); btns[o.value].disabled = off; btns[o.value].classList.toggle('is-off', off); }); }
    };
    return api;
  }

  // Hero tiles: [{label, big, sub}] drawn into #tiles.
  function heroTiles(list) {
    var tiles = $('#tiles');
    if (!tiles) return;
    clear(tiles);
    list.forEach(function (t) {
      tiles.appendChild(h('div', { class: 'tile' }, h('h3', {}, t.label), h('div', { class: 'big num' }, t.big), h('p', { class: 'sub' }, t.sub)));
    });
  }

  // License chip in the hero kicker row and the credit line in the footer, both from the JSON.
  function licenseUI(lic) {
    if (!lic) return;
    var chip = $('#lic');
    if (chip) { clear(chip); chip.appendChild(h('a', { href: lic.url, rel: 'noopener', text: lic.license })); }
    setText('credit', lic.credit);
  }

  function receiptLinks(receipt) {
    if (!receipt) return null;
    return h('span', { class: 'rcpt' },
      h('a', { href: repoUrl(receipt.run), text: 'Run record' }), ', ',
      h('a', { href: commitUrl(receipt.commit), text: 'commit ' + receipt.commit }),
      ', logged ' + String(receipt.logged_at || '').slice(0, 10));
  }

  function caveatBox(list, title, cls) {
    return h('div', { class: 'caveats ' + (cls || '') }, h('h3', { text: title || 'Read this first' }),
      h('ul', {}, list.map(function (c) { return h('li', { text: c }); })));
  }

  // "Data not available yet": hero line plus every body id.
  function unavailable(what, bodyIds) {
    setText('hero-line', 'Data not available yet');
    setText('hero-deck', what + ' has not been published yet. It is written by scripts/export_ml_pages.py and appears here after the next update.');
    (bodyIds || []).forEach(function (id) {
      var b = done(id);
      if (!b) return;
      clear(b);
      b.appendChild(h('p', { class: 'note empty', text: 'Data not available yet. ' + what + ' will appear here once it is published.' }));
    });
  }

  // Interval chart: one row per estimate, a dot at the difference and a whisker for its 95% interval,
  // against a labelled zero line. rows: [{label, ci:{diff,lo,hi}, series:'s3'|'sm', note}].
  function ciChart(host, rows, opts) {
    clear(host);
    opts = opts || {};
    var d = opts.decimals == null ? 3 : opts.decimals;
    var W = Math.max(260, host.clientWidth || 600), narrow = W < 520, rowH = narrow ? 70 : 58, m = { t: 6, r: 14, b: 40, l: 14 };
    var H = m.t + rows.length * rowH + m.b;
    var lo = 0, hi = 0;
    rows.forEach(function (r) { lo = Math.min(lo, r.ci.lo); hi = Math.max(hi, r.ci.hi); });
    var span = (hi - lo) || 1; lo -= span * 0.08; hi += span * 0.08;
    var step = niceStep((hi - lo) / Math.max(2, Math.floor(W / 110)));
    var iw = W - m.l - m.r;
    function sx(v) { return m.l + (v - lo) / (hi - lo) * iw; }
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'group',
      'aria-label': (opts.title || 'Estimates') + ' with 95% intervals. ' + rows.map(function (r) { return r.label + ' ' + signed(r.ci.diff, d) + ', interval ' + signed(r.ci.lo, d) + ' to ' + signed(r.ci.hi, d); }).join('; ') + '.' }, host);
    var lastX = -1e9;
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      var x = sx(v);
      sv('line', { class: Math.abs(v) < step / 1e3 ? 'refline' : 'grid', x1: x, x2: x, y1: m.t, y2: H - m.b }, svg);
      if (x - lastX >= 52) { sv('text', { class: 'tick a-mid', x: x, y: H - 20, text: Math.abs(v) < step / 1e3 ? '0' : signed(v, d) }, svg); lastX = x; }
    }
    if (opts.axis) sv('text', { class: 'tick a-mid', x: m.l + iw / 2, y: H - 3, text: opts.axis }, svg);
    rows.forEach(function (r, i) {
      var y0 = m.t + i * rowH, cy = y0 + (narrow ? 54 : 40), ci = r.ci;
      var g = sv('g', { class: 'mrow', tabindex: 0, role: 'img', 'aria-label': r.label + ' ' + signed(ci.diff, d) + ', 95% interval ' + signed(ci.lo, d) + ' to ' + signed(ci.hi, d) + (ci.excludes_zero ? ', excludes zero' : ', includes zero') }, svg);
      sv('rect', { class: 'rowhit', x: 0, y: y0 + 1, width: W, height: rowH - 2 }, g);
      sv('text', { class: 'lbl', x: m.l, y: y0 + 20, text: r.label }, g);
      sv('text', { class: 'lbl rv', x: narrow ? m.l : W - m.r, y: y0 + (narrow ? 38 : 20), 'text-anchor': narrow ? 'start' : 'end', text: signed(ci.diff, d) + '  ' + ciText(ci, d) }, g);
      sv('line', { class: 'wh ' + (r.series || 's3'), x1: sx(ci.lo), x2: sx(ci.hi), y1: cy, y2: cy }, g);
      sv('line', { class: 'wh ' + (r.series || 's3'), x1: sx(ci.lo), x2: sx(ci.lo), y1: cy - 6, y2: cy + 6 }, g);
      sv('line', { class: 'wh ' + (r.series || 's3'), x1: sx(ci.hi), x2: sx(ci.hi), y1: cy - 6, y2: cy + 6 }, g);
      mark(g, r.series || 's3', sx(ci.diff), cy, 7);
      bindTip(g, function () {
        var t = [{ value: signed(ci.diff, d), label: opts.measure || 'estimate', cls: r.series === 'sm' ? 'sm' : 's3' },
          { value: ciText(ci, d), label: '95% interval, ' + (ci.excludes_zero ? 'excludes zero' : 'includes zero') }];
        if (r.note) t.push({ value: '', label: r.note });
        return { title: r.label, rows: t };
      });
    });
    return svg;
  }
  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }

  // A table view that re-renders when the underlying selection changes (tableView() builds once).
  function tableViewLive(summary, colsFn, rowsFn) {
    var d = h('details', { class: 'tv' }, h('summary', { text: summary }));
    function draw() {
      var old = d.querySelector('.tv-wrap');
      if (old) old.remove();
      var cols = colsFn();
      d.appendChild(h('div', { class: 'tv-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, cols.map(function (c) { return h('th', { class: c.r ? 'r' : '', scope: 'col', text: c.t }); }))),
        h('tbody', {}, rowsFn().map(function (r) { return h('tr', {}, r.map(function (v, i) { return h('td', { class: (cols[i].r ? 'r ' : '') + (cols[i].n ? 'n' : ''), text: v }); })); })))));
    }
    d.addEventListener('toggle', function () { if (d.open) draw(); });
    return { el: d, refresh: function () { if (d.open) draw(); } };
  }

  // Team display names come from ladder.json when present; otherwise the abbreviation stands in.
  var teamNames = {};
  function loadTeamNames() {
    return getJSON('data/ladder.json', true).then(function (l) {
      if (l && l.teams) l.teams.forEach(function (t) { teamNames[t.team] = t.name; });
      return teamNames;
    });
  }
  function teamName(abbr) { return teamNames[abbr] || abbr; }

  // Field position from yards to the opponent's goal line: 75 -> "own 25", 50 -> "midfield", 35 -> "opp 35".
  function fieldLabel(yl) { return yl > 50 ? 'own ' + (100 - yl) : yl === 50 ? 'midfield' : 'opp ' + yl; }
  function ordinal(n) { return n + (['th', 'st', 'nd', 'rd'][n] || 'th'); }

  // On load, scroll the nav strip so the current page link is visible (narrow screens scroll it sideways).
  function navCurrent() {
    var a = document.querySelector('.nav-links a[aria-current="page"]');
    if (!a) return;
    var box = a.parentNode.parentNode;
    if (box.scrollWidth > box.clientWidth) box.scrollLeft = Math.max(0, a.offsetLeft - 60);
  }
  navCurrent();
  initSlabs();

  window.NFL = {
    getJSON: getJSON, $: $, h: h, append: append, sv: sv, clear: clear, well: well,
    tipShow: tipShow, tipHide: tipHide, bindTip: bindTip, live: live,
    watch: watch, tableView: tableView, legend: legend, initNav: initNav,
    REPO: REPO, repoUrl: repoUrl, commitUrl: commitUrl, signed: signed, commas: commas, pct: pct, ciText: ciText,
    setText: setText, done: done, seg: seg, heroTiles: heroTiles, licenseUI: licenseUI, receiptLinks: receiptLinks,
    caveatBox: caveatBox, unavailable: unavailable, ciChart: ciChart, tableViewLive: tableViewLive,
    loadTeamNames: loadTeamNames, teamName: teamName, fieldLabel: fieldLabel, ordinal: ordinal,
    mark: mark, moveMark: moveMark, initSlabs: initSlabs
  };
})();
