/* NFLELO models page. Reads ./data/research.json (written by scripts/export_research.py) and draws the
   scoreboard cards, the game-model holdout chart, the findings and the receipts. No dependencies.
   The small DOM, tooltip and table-view helpers mirror the ones in app.js (the two pages share styles.css). */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var REPO = 'https://github.com/walk-the-program/NFLELO';

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
  function setText(id, text) { var n = $('#' + id); if (n && text) n.textContent = text; }
  function done(id) { var n = $('#' + id); if (n) n.removeAttribute('aria-busy'); return n; }

  // ------------------------------------------------------------------ formatting

  // Signed fixed-point text with an ASCII minus; a value that rounds to zero is shown as +0.
  function signed(x, d) { var r = Math.round(x * Math.pow(10, d)) / Math.pow(10, d); return (r >= 0 ? '+' : '-') + Math.abs(r).toFixed(d); }
  function commas(n) { return Number(n).toLocaleString('en-US'); }

  function intervalText(p) { return '[' + signed(p.ci_low, p.decimals) + ', ' + signed(p.ci_high, p.decimals) + ']'; }
  function estimateText(p) { return p.kind === 'calibration' ? p.estimate.toFixed(p.decimals) : signed(p.estimate, p.decimals); }
  function limitText(p) { return p.kind === 'calibration' ? 'limit ' + p.limit.toFixed(p.decimals) : intervalText(p); }
  function repoUrl(path) { return REPO + '/blob/main/' + path; }
  function commitUrl(hash) { return REPO + '/commit/' + hash; }

  // ------------------------------------------------------------------ tooltip (same behavior as the main page)

  var tip = $('#tip'), live = $('#live');

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
    clear(tip);
    if (title) tip.appendChild(h('div', { class: 'tt', text: title }));
    rows.forEach(function (r) {
      tip.appendChild(h('div', { class: 'tr' }, r.cls ? h('i', { class: r.cls }) : null, h('b', { text: r.value }), r.label ? h('span', { text: r.label }) : null));
    });
    tipPlace(x, y);
    live.textContent = (title ? title + '. ' : '') + rows.map(function (r) { return (r.label ? r.label + ' ' : '') + r.value; }).join('. ');
  }

  function tipHide() { tip.hidden = true; }

  function bindTip(node, build) {
    node.addEventListener('pointermove', function (e) { var c = build(); if (c) tipShow(e.clientX, e.clientY, c.title, c.rows); });
    node.addEventListener('pointerleave', tipHide);
    node.addEventListener('focus', function () {
      var r = node.getBoundingClientRect(), c = build();
      if (c) tipShow(r.left + Math.min(r.width, 160), r.top + r.height / 2, c.title, c.rows);
    });
    node.addEventListener('blur', tipHide);
  }

  // ------------------------------------------------------------------ resize and table views

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

  // ------------------------------------------------------------------ state

  var R = null;   // research.json

  // ------------------------------------------------------------------ hero

  function renderHero() {
    var c = R.counts, hl = R.headlines;
    setText('hero-line', hl.hero);
    setText('hero-deck', hl.hero_deck + ' Every model was built on earlier seasons, then scored once on ' + R.holdout_seasons[0] + ' to ' + R.holdout_seasons[1] + ' under rules committed to a public repository first.');
    var tiles = $('#tiles');
    clear(tiles);
    function tile(label, big, sub) {
      return h('div', { class: 'tile' }, h('h3', { text: label }), h('div', { class: 'big' }, String(big)), h('p', { class: 'sub' }, sub));
    }
    var dev = c.negative + c.shadow;
    tiles.append(
      tile('Tested once', c.holdout_tested, 'models scored on seasons they never saw'),
      tile('Passed', c.holdout_pass, 'met every rule set in advance'),
      tile('Failed', c.holdout_fail, 'missed a rule, published anyway'),
      tile('Not on the holdout', dev, c.negative + ' negative result, ' + c.shadow + ' shadow model running live'));
  }

  // ------------------------------------------------------------------ scoreboard

  var FILTERS = [
    ['all', 'All', function () { return true; }],
    ['pass', 'Passed', function (m) { return m.verdict === 'pass'; }],
    ['fail', 'Failed', function (m) { return m.verdict === 'fail'; }],
    ['dev', 'Not on the holdout', function (m) { return !m.tested_holdout; }]
  ];

  function card(m) {
    var p = m.primary;
    var links = [h('a', { href: repoUrl(m.method_doc), text: 'Method' })];
    if (m.rules_commit) links.push(h('a', { href: commitUrl(m.rules_commit), text: 'Rules ' + m.rules_commit }));
    else links.push(h('span', { class: 'none', text: 'No holdout run' }));
    if (m.result_commit) links.push(h('a', { href: commitUrl(m.result_commit), text: 'Result ' + m.result_commit }));
    return h('article', { class: 'mcard', 'data-verdict': m.verdict, 'data-holdout': m.tested_holdout ? '1' : '0', 'aria-labelledby': 'mc-' + m.id },
      h('div', { class: 'mc-top' },
        h('span', { class: 'mc-id', text: m.short }),
        h('span', { class: 'vchip v-' + m.verdict, text: m.verdict_label })),
      h('h3', { class: 'mc-name', id: 'mc-' + m.id, text: m.name }),
      h('p', { class: 'mc-what', text: m.what }),
      h('div', { class: 'mc-metric' },
        h('p', { class: 'mc-lab', text: p.label }),
        h('p', { class: 'mc-val num', text: estimateText(p) }),
        h('p', { class: 'mc-ci num', text: (p.kind === 'calibration' ? '' : '95% interval ') + limitText(p) }),
        h('p', { class: 'mc-n', text: commas(p.n) + ' ' + p.n_unit + ', ' + p.window })),
      h('div', { class: 'mc-text' },
        h('p', { class: 'mc-read', text: m.reading }),
        h('ul', { class: 'mc-also' }, m.also.map(function (a) { return h('li', { text: a.text }); }))),
      h('div', { class: 'mc-links' }, links));
  }

  function renderScoreboard() {
    var body = done('scoreboard-body'), M = R.models, c = R.counts;
    clear(body);
    setText('scoreboard-h', R.headlines.scoreboard);

    var grid = h('div', { class: 'mcards', id: 'mcards' }, M.map(card));
    var chips = $('#model-filters');
    clear(chips);
    FILTERS.forEach(function (f, i) {
      var n = M.filter(f[2]).length;
      var b = h('button', { class: 'chip', type: 'button', 'aria-pressed': i === 0 ? 'true' : 'false', text: f[1] + ' ' + n });
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(chips.children, function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        Array.prototype.forEach.call(grid.children, function (el, k) { el.hidden = !f[2](M[k]); });
        live.textContent = f[1] + ': ' + n + ' of ' + M.length + ' models shown.';
      });
      chips.appendChild(b);
    });

    body.appendChild(grid);
    body.appendChild(tableView('View all models as a table',
      [{ t: 'Model' }, { t: 'Verdict' }, { t: 'Measure' }, { t: 'Estimate', r: 1, n: 1 }, { t: '95% interval or limit', r: 1, n: 1 }, { t: 'Sample' }, { t: 'Rules commit' }, { t: 'Result commit' }],
      function () {
        return M.map(function (m) {
          var p = m.primary;
          return [m.short + ' ' + m.name, m.verdict_label, p.label, estimateText(p), limitText(p),
            commas(p.n) + ' ' + p.n_unit + ', ' + p.window, m.rules_commit || 'none', m.result_commit || 'none'];
        });
      }));
    body.appendChild(h('p', { class: 'fnote', text: 'Every number comes from the committed run registry (experiments/runs) or, where noted, from the project scoreboard. The models built on participation data (on-field ratings, the play model, the fourth-down tool and early-down play calling) are released under CC BY-SA; the rest under CC BY.' }));

    body.appendChild(h('dl', { class: 'gloss' },
      h('h3', { text: 'Reading the numbers' }),
      gl('Brier score', 'The average squared miss of a win probability. Lower is better. Always guessing 50% scores 0.25.'),
      gl('95% interval', 'The range that would hold the true difference in 95 of 100 repeats of the test. If it includes zero, the edge is not proven.'),
      gl('EPA', 'Expected points added: how much a play changed the offense\'s chance to score, in points.'),
      gl('CRPS', 'A score for a whole forecast distribution, here the spread of yards a play might gain. Lower is better.'),
      gl('Calibration error (ECE)', 'How far stated chances sit from what really happened. 0 is perfect; the limit allows for luck at that sample size.'),
      gl('RAPM', 'Regularized adjusted plus-minus: a player\'s effect on results, adjusted for who else was on the field.'),
      gl('Shadow model', 'A model logged before kickoff next to the live one, scored on the same games, but not used for the live forecast.')));
  }

  function gl(term, text) { return h('div', {}, h('dt', { text: term }), h('dd', { text: text })); }

  // ------------------------------------------------------------------ game-model chart

  function gameChart(host, C) {
    clear(host);
    var W = Math.max(280, host.clientWidth || 640);
    var rowH = W < 560 ? 80 : 92, m = { t: 8, r: 18, b: 32, l: 18 };
    var rows = C.rows, H = m.t + rows.length * rowH + m.b;
    var vals = [];
    rows.forEach(function (r) { vals.push(r.brier); if (r.whisker_low != null) vals.push(r.whisker_low, r.whisker_high); });
    var lo = Math.round(Math.floor((Math.min.apply(null, vals) - 0.002) / 0.005 + 1e-9) * 0.005 * 1000) / 1000;
    var hi = Math.round(Math.ceil((Math.max.apply(null, vals) + 0.002) / 0.005 - 1e-9) * 0.005 * 1000) / 1000;
    var iw = W - m.l - m.r;
    function sx(v) { return m.l + (v - lo) / (hi - lo) * iw; }

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'group',
      'aria-label': 'Brier score on the ' + C.n + ' games of ' + C.window[0] + ' to ' + C.window[1] + ', by model. Lower is better. ' +
        rows.map(function (r) { return r.label + ' ' + r.brier.toFixed(4); }).join(', ') + '. Use the table below for exact values.' }, host);

    var lastX = -1e9;
    for (var v = lo; v <= hi + 1e-9; v += 0.005) {
      var x = sx(v);
      sv('line', { class: 'grid', x1: x, x2: x, y1: m.t, y2: H - m.b }, svg);
      if (x - lastX >= 44) { sv('text', { class: 'tick a-mid', x: x, y: H - 10, text: v.toFixed(3) }, svg); lastX = x; }
    }
    var model = rows.filter(function (r) { return r.id === 'model'; })[0];
    sv('line', { class: 'refline', x1: sx(model.brier), x2: sx(model.brier), y1: m.t, y2: H - m.b }, svg);

    rows.forEach(function (r, i) {
      var y0 = m.t + i * rowH, cy = y0 + 56;
      var g = sv('g', { class: 'mrow', tabindex: 0, role: 'img',
        'aria-label': r.label + ' Brier ' + r.brier.toFixed(4) + (r.gap != null ? '; ' + signed(r.gap, 4) + ' against the game model, 95% interval ' + signed(r.gap_ci_low, 4) + ' to ' + signed(r.gap_ci_high, 4) : '; this is the reference row') }, svg);
      sv('rect', { class: 'rowhit', x: 0, y: y0 + 2, width: W, height: rowH - 4, fill: 'transparent' }, g);
      sv('text', { class: 'lbl', x: m.l, y: y0 + 22, text: r.label }, g);
      sv('text', { class: 'lbl', x: W - m.r, y: y0 + 22, 'text-anchor': 'end', text: r.brier.toFixed(4) }, g);
      if (r.whisker_low != null) {
        sv('line', { class: 'wh ' + r.series, x1: sx(r.whisker_low), x2: sx(r.whisker_high), y1: cy, y2: cy }, g);
        sv('line', { class: 'wh ' + r.series, x1: sx(r.whisker_low), x2: sx(r.whisker_low), y1: cy - 6, y2: cy + 6 }, g);
        sv('line', { class: 'wh ' + r.series, x1: sx(r.whisker_high), x2: sx(r.whisker_high), y1: cy - 6, y2: cy + 6 }, g);
      }
      sv('circle', { class: 'dot ' + r.series, cx: sx(r.brier), cy: cy, r: 7 }, g);
      bindTip(g, function () {
        var rowsT = [{ value: r.brier.toFixed(4), label: 'Brier score', cls: r.series }, { value: r.logloss.toFixed(4), label: 'Log loss' }];
        if (r.gap != null) rowsT.push({ value: signed(r.gap, 4), label: 'gap to the game model, 95% interval ' + signed(r.gap_ci_low, 4) + ' to ' + signed(r.gap_ci_high, 4) });
        return { title: r.label, rows: rowsT };
      });
    });
    return svg;
  }

  function renderHoldout() {
    var body = done('holdout-body'), C = R.chart;
    clear(body);
    setText('holdout-h', R.headlines.chart);
    var elo = C.rows.filter(function (r) { return r.id === 'elo'; })[0];
    var mkt = C.rows.filter(function (r) { return r.id === 'market'; })[0];
    var host = h('div', { class: 'chart' });
    var facts = h('dl', { class: 'facts' },
      fact('Model against Elo', signed(-elo.gap, 4), 'Brier, 95% interval ' + signed(-elo.gap_ci_high, 4) + ' to ' + signed(-elo.gap_ci_low, 4)),
      fact('Seasons won against Elo', C.model_beats_elo_seasons + ' of ' + C.seasons_total, commas(C.n) + ' games, ' + C.window[0] + ' to ' + C.window[1]),
      fact('Market against model', signed(mkt.gap, 4), 'Brier, 95% interval ' + signed(mkt.gap_ci_low, 4) + ' to ' + signed(mkt.gap_ci_high, 4)));
    body.appendChild(h('div', { class: 'split' },
      h('div', {},
        legend([['dot s1', 'Elo v2'], ['dot s3', 'NFLELO game model'], ['dot s2', 'Vegas market, benchmark only']]),
        host,
        h('p', { class: 'fnote', text: 'Brier score, lower is better; the axis starts at 0.200, not zero. The model\'s own score has no whisker. ' + C.whisker_note }),
        tableView('View the comparison as a table', [{ t: 'Season' }, { t: 'Games', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }, { t: 'Game model', r: 1, n: 1 }, { t: 'Vegas market', r: 1, n: 1 }], function () {
          var all = [[C.window[0] + ' to ' + C.window[1], commas(C.n), elo.brier.toFixed(4), C.rows.filter(function (r) { return r.id === 'model'; })[0].brier.toFixed(4), mkt.brier.toFixed(4)]];
          return all.concat(C.seasons.map(function (s) { return [String(s.season), String(s.n), s.elo.toFixed(4), s.model.toFixed(4), s.market.toFixed(4)]; }));
        })),
      facts));
    watch(host, function () { gameChart(host, C); });
  }

  function fact(label, value, small) {
    return h('div', { class: 'fact' }, h('dt', { text: label }), h('dd', { class: 'num' }, value, h('small', { text: small })));
  }

  // ------------------------------------------------------------------ evidence

  function renderEvidence() {
    var body = done('evidence-body');
    clear(body);
    body.appendChild(h('ol', { class: 'finds' }, R.findings.map(function (f) {
      return h('li', { class: 'find' },
        h('span', { class: 'fn', 'aria-hidden': 'true', text: String(f.n) }),
        h('div', {},
          h('h3', { text: f.title }),
          h('p', { text: f.text }),
          h('p', { class: 'fnote' }, 'Source: ', h('a', { href: repoUrl(f.source), text: f.source }))));
    })));
  }

  // ------------------------------------------------------------------ receipts

  function liveBox() {
    var L = R.live;
    if (!L) return null;
    var box = h('div', { class: 'livebox' }, h('h3', { text: 'Live ' + L.season + ' record' }));
    var m = L.model, s = L.shadow;
    if (!m.n_scored) {
      box.appendChild(h('p', { class: 'sub', text: 'The live record starts with week ' + L.from_week + (L.from_date ? ' (' + L.from_date + ')' : '') + '. No ' + L.season + ' games have been scored yet; ' + commas(L.ledger_rows) + ' predictions are logged so far, each written before kickoff.' }));
    } else {
      box.appendChild(h('p', { class: 'sub', text: 'Scored on ' + commas(m.n_scored) + ' completed ' + L.season + ' games since week ' + L.from_week + ', using each game\'s last prediction before kickoff.' }));
      box.appendChild(h('div', { class: 'score' },
        h('div', {}, h('h3', { text: 'Brier score, lower is better' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: m.brier.toFixed(4) }), 'Game model'),
            h('span', {}, h('b', { class: 'num', text: m.elo_brier.toFixed(4) }), 'Elo v2'))),
        s && s.n_scored ? h('div', {}, h('h3', { text: 'Shadow model, same games' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: s.brier.toFixed(4) }), 'Kalman (' + commas(s.n_scored) + ' games)'),
            h('span', {}, h('b', { class: 'num', text: s.model_brier_same_games.toFixed(4) }), 'Game model, same games'))) : null));
    }
    if (s && !s.n_scored) box.appendChild(h('p', { class: 'fnote', text: 'The shadow model (' + s.name + ') is logged next to the game model and has no scored games yet.' }));
    return box;
  }

  function renderReceipts() {
    var body = done('receipts-body'), L = R.live, F = R.forward_test;
    clear(body);
    var links = [
      h('li', {}, h('a', { href: repoUrl('context/ml-results.md'), text: 'The full scoreboard (markdown)' }), h('small', { text: 'Every model and its locked result, failures included' })),
      h('li', {}, h('a', { href: REPO + '/tree/main/experiments/runs', text: 'Run registry' }), h('small', { text: 'One JSON file per run, with its data hashes and the commit it ran on' }))
    ];
    if (L && L.ledger_url) links.push(h('li', {}, h('a', { href: L.ledger_url, text: 'Prediction ledger (CSV)' }), h('small', { text: 'Every ' + L.season + ' prediction, written before kickoff' })));
    if (L && L.shadow && L.shadow.ledger_url) links.push(h('li', {}, h('a', { href: L.shadow.ledger_url, text: 'Shadow ledger (CSV)' }), h('small', { text: 'The Kalman shadow model\'s predictions' })));
    links.push(h('li', {}, h('a', { href: REPO, text: 'Code and method on GitHub' }), h('small', { text: 'github.com/walk-the-program/NFLELO' })));

    body.appendChild(h('div', { class: 'method' },
      h('div', {},
        h('h3', { text: 'The process' }),
        h('p', {}, h('b', { text: 'A locked holdout. ' }), 'The ' + R.holdout_seasons[0] + ' to ' + R.holdout_seasons[1] + ' seasons were set aside before the models were built. Models were developed and tuned on earlier seasons only. Each one then got a single run on the locked seasons, and that number is reported whatever it turned out to be.'),
        h('p', {}, h('b', { text: 'Rules first. ' }), 'Before each run, the pass and fail rules were committed to the public repository. The "Rules" hash on each card is that commit; the "Result" hash is the one that recorded the outcome. Anyone can check that the first came before the second.'),
        h('p', {}, h('b', { text: 'Dated rule changes. ' }), 'When a rule had to change, it changed before the holdout number existed, with the date and reason written down. Examples: the calibration check became one-sided (a forecast should not be marked down for being better calibrated than chance), a 0.01 floor was set for very large samples, and punts are judged on yards error. One bar was simply too tight: the game model\'s calibration limit of 0.02 cannot be met reliably with 1,615 games. It is recorded as a flawed criterion, not quietly moved.'),
        h('p', {}, h('b', { text: 'Leak checks. ' }), 'Every input is built as of the start of that game\'s week, and a test that corrupts future data proves nothing leaks in. Four real leaks were caught and fixed along the way, such as roster statuses that were season-end values before 2016 and missing formations that gave away fumbles.'),
        h('p', {}, h('b', { text: 'A live ledger. ' }), 'Predictions for the ' + (L ? L.season : '2026') + ' season are written to an append-only ledger before kickoff and pushed to GitHub, so the timestamps can be audited. It is the only test that cannot be rehearsed.'),
        liveBox(),
        h('div', { class: 'livebox' },
          h('h3', { text: 'Forward test: ' + F.name }),
          h('p', { class: 'sub' }, h('b', { text: F.status + '. ' }), F.text, ' ', F.dev_note, ' ',
            h('a', { href: commitUrl(F.frozen_commit), text: 'Frozen commit ' + F.frozen_commit }), '.'))),
      h('div', {},
        h('h3', { text: 'Links' }),
        h('ul', {}, links))));
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

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['scoreboard', 'holdout', 'evidence', 'receipts'].forEach(function (id) {
      var b = done(id + '-body');
      if (!b) return;
      clear(b);
      b.appendChild(h('p', { class: 'note err', text: 'The results file could not be loaded. Serve this folder over http, for example with python3 -m http.server --directory site.' }));
    });
    setText('hero-line', 'Results unavailable');
  }

  fetch('data/meta.json').then(function (r) { return r.ok ? r.json() : null; }).then(function (m) {
    if (m && m.credit) setText('credit', m.credit);
  }).catch(function () {});

  fetch('data/research.json').then(function (r) { if (!r.ok) throw new Error('research.json ' + r.status); return r.json(); }).then(function (d) {
    R = d;
    renderHero();
    renderScoreboard();
    renderHoldout();
    renderEvidence();
    renderReceipts();
    initNav();
  }).catch(fail);
})();
