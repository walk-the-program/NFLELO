/* NFLELO shared front-end helpers, used by app.js (ratings page) and models.js (models page):
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

  // Chart area: a recessed well around a chart host (the host keeps its own width for drawing).
  function well(host) { return h('div', { class: 'well' }, host); }

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

  window.NFL = {
    getJSON: getJSON, $: $, h: h, append: append, sv: sv, clear: clear, well: well,
    tipShow: tipShow, tipHide: tipHide, bindTip: bindTip, live: live,
    watch: watch, tableView: tableView, legend: legend, initNav: initNav
  };
})();
