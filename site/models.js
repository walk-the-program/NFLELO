/* NFLELO models page. Reads ./data/research.json (written by scripts/export_research.py) and draws the
   scoreboard cards, the game-model holdout chart and the findings. No dependencies; the DOM,
   tooltip, table-view and fetch helpers come from common.js, shared with the ratings page. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, bindTip = N.bindTip, live = N.live;
  var watch = N.watch, tableView = N.tableView, legend = N.legend;

  function setText(id, text) { var n = $('#' + id); if (n && text) n.textContent = text; }
  function done(id) { var n = $('#' + id); if (n) n.removeAttribute('aria-busy'); return n; }

  // ------------------------------------------------------------------ formatting

  // Signed fixed-point text with an ASCII minus; a value that rounds to zero is shown as +0.
  function signed(x, d) { var r = Math.round(x * Math.pow(10, d)) / Math.pow(10, d); return (r >= 0 ? '+' : '-') + Math.abs(r).toFixed(d); }
  function commas(n) { return Number(n).toLocaleString('en-US'); }

  function intervalText(p) { return '[' + signed(p.ci_low, p.decimals) + ', ' + signed(p.ci_high, p.decimals) + ']'; }
  function estimateText(p) { return p.kind === 'calibration' ? p.estimate.toFixed(p.decimals) : signed(p.estimate, p.decimals); }
  function limitText(p) { return p.kind === 'calibration' ? 'limit ' + p.limit.toFixed(p.decimals) : intervalText(p); }
  function repoUrl(path) { return 'https://github.com/walk-the-program/NFLELO/blob/main/' + path; }

  // ------------------------------------------------------------------ state

  var R = null;   // research.json

  // ------------------------------------------------------------------ hero

  function renderHero() {
    var c = R.counts, hl = R.headlines;
    setText('hero-line', hl.hero);
    setText('hero-deck', hl.hero_deck);
    var tiles = $('#tiles');
    clear(tiles);
    function tile(label, big, sub, bar, barText) {
      return h('div', { class: 'tile' }, h('h3', { text: label }), h('div', { class: 'big' }, String(big)), bar != null ? N.tileBar(bar, barText) : null, h('p', { class: 'sub' }, sub));
    }
    var dev = c.negative + c.shadow;
    tiles.append(
      tile('Tested once', c.holdout_tested, 'models scored on seasons they never saw', c.holdout_tested ? c.holdout_pass / c.holdout_tested : null, c.holdout_pass + ' of ' + c.holdout_tested + ' passed'),
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
      h('div', { class: 'mc-links' }, h('a', { href: repoUrl(m.method_doc), text: 'Method' })));
  }

  function summaryCard(c) {
    var rows = [[c.pass, 'passed'], [c.fail, 'failed'], [c.negative, 'development only'], [c.shadow, 'running live']];
    return h('article', { class: 'mcard mc-sum', 'aria-labelledby': 'mc-sum' },
      h('div', { class: 'mc-top' }, h('span', { class: 'mc-id', text: 'All models' })),
      h('h3', { class: 'mc-name', id: 'mc-sum', text: c.total + ' models' }),
      h('p', { class: 'mc-what', text: rows.map(function (r) { return r[0] + ' ' + r[1]; }).join(', ') + '.' }),
      h('ul', { class: 'mc-metric mc-tally', 'aria-label': 'Models by outcome' }, rows.map(function (r) {
        return h('li', {}, h('b', { class: 'num', text: String(r[0]) }), h('span', { text: r[1] }));
      })),
      h('div', { class: 'mc-text' },
        h('p', { class: 'mc-read', text: c.holdout_tested + ' were scored once on the locked seasons, ' + c.holdout_pass + ' passed. The rest are development-only results or are still running live.' })));
  }

  function renderScoreboard() {
    var body = done('scoreboard-body'), M = R.models, c = R.counts;
    clear(body);
    setText('scoreboard-h', R.headlines.scoreboard);

    // The 12th tile: a summary computed from the data, shown on the All view only. It also rounds the grid to full rows.
    var sum = summaryCard(c), shown = M.map(card).concat([sum]);
    var grid = h('div', { class: 'mcards', id: 'mcards' }, shown);
    var chips = $('#model-filters');
    clear(chips);
    FILTERS.forEach(function (f, i) {
      var n = M.filter(f[2]).length;
      var b = h('button', { type: 'button', 'aria-pressed': i === 0 ? 'true' : 'false', text: f[1] + ' ' + n });
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(chips.children, function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        M.forEach(function (m, k) { grid.children[k].hidden = !f[2](m); });
        sum.hidden = f[0] !== 'all';
        live.textContent = f[1] + ': ' + n + ' of ' + M.length + ' models shown.';
      });
      chips.appendChild(b);
    });

    body.appendChild(grid);
    body.appendChild(tableView('View all models as a table',
      [{ t: 'Model' }, { t: 'Verdict' }, { t: 'Measure' }, { t: 'Estimate', r: 1, n: 1 }, { t: '95% interval or limit', r: 1, n: 1 }, { t: 'Sample' }],
      function () {
        return M.map(function (m) {
          var p = m.primary;
          return [m.short + ' ' + m.name, m.verdict_label, p.label, estimateText(p), limitText(p),
            commas(p.n) + ' ' + p.n_unit + ', ' + p.window];
        });
      }));
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
      N.mark(g, r.series, sx(r.brier), cy, 7);
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
        well(host),
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
          h('p', { text: f.text })));
    })));
  }

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['scoreboard', 'holdout', 'evidence'].forEach(function (id) {
      var b = done(id + '-body');
      if (!b) return;
      clear(b);
      b.appendChild(h('p', { class: 'note err', text: 'The results file could not be loaded. Serve this folder over http, for example with python3 -m http.server --directory site.' }));
    });
    setText('hero-line', 'Results unavailable');
  }

  N.getJSON('data/research.json').then(function (d) {
    R = d;
    renderHero();
    renderScoreboard();
    renderHoldout();
    renderEvidence();
    N.initNav();
  }).catch(fail);
})();
