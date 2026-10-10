/* NFLELO win probability page. Reads ./data/winprob.json (written by scripts/export_ml_pages.py): one win
   probability line per completed 2026 game, an excitement leaderboard, and the model's test against nflfastR.
   No dependencies; helpers come from common.js. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, tipHide = N.tipHide, watch = N.watch;
  var signed = N.signed, commas = N.commas, pct = N.pct;
  var setText = N.setText, done = N.done;

  var W = null;             // winprob.json
  var byId = {};
  var curGame = null;       // selected game object
  var ui = {};
  var sortKey = 'excitement', showAll = false;

  function matchup(g) { return g.away + ' at ' + g.home; }
  function score(g) { return g.away + ' ' + g.away_score + ', ' + g.home + ' ' + g.home_score; }
  function gameTime(t) {
    if (t >= 3600) return 'End of regulation';
    var q = Math.min(4, Math.floor(t / 900) + 1), rem = 900 - (t - (q - 1) * 900);
    return 'Q' + q + ' ' + Math.floor(rem / 60) + ':' + ('0' + Math.floor(rem % 60)).slice(-2);
  }
  function winner(g) { return g.tie ? null : g.home_score > g.away_score ? g.home : g.away; }
  function pts(x) { return (x * 100).toFixed(1); }

  // ------------------------------------------------------------------ hero

  function renderHero() {
    var H = W.headline, d = H.ours_minus_nflfastr, G = W.games, top = W.games_by_excitement[0];
    setText('hero-line', 'Our win probability beats nflfastR\'s public model on 2020 to 2025: Brier ' + H.brier_ours.toFixed(4) + ' against ' + H.brier_nflfastr.toFixed(4) + ' (difference ' + signed(d.diff, 4) + ', 95% interval ' + signed(d.lo, 4) + ' to ' + signed(d.hi, 4) + '), with no betting-market input.');
    setText('hero-deck', 'Pick any completed ' + W.season + ' game and watch each team\'s chance of winning move snap by snap. The model reads the game state and a pregame strength estimate, nothing from the betting market.');
    var swingMax = G.reduce(function (a, g) { return !a || g.swing.abs > a.swing.abs ? g : a; }, null);
    N.heroTiles([
      { label: 'Games charted', big: commas(G.length), sub: 'completed ' + W.season + ' regular-season games, through week ' + W.updated_through.week + ' (last game ' + W.updated_through.last_game_date + ')' },
      { label: 'Most exciting game', big: top ? top.away + ' at ' + top.home : 'n/a', sub: top ? 'Week ' + top.week + ', final ' + top.away_score + '-' + top.home_score + ', excitement index ' + top.excitement.toFixed(1) : '' },
      { label: 'Biggest single swing', big: swingMax ? pts(swingMax.swing.abs) + ' pts' : 'n/a', sub: swingMax ? 'win probability on one play, ' + matchup(swingMax) + ', week ' + swingMax.week + ', ' + swingMax.swing.clock : '' },
      { label: 'Test against nflfastR', big: signed(d.diff, 4), sub: 'Brier on ' + commas(H.plays) + ' plays, 2020 to 2025. Negative is better.' }
    ]);
    N.licenseUI(W.license);
  }

  // ------------------------------------------------------------------ game chart

  function buildChart() {
    var body = done('chart-body');
    clear(body);
    var games = W.games;
    if (!games.length) { body.appendChild(h('p', { class: 'note empty', text: 'No completed games yet.' })); return; }

    var sel = h('select', { id: 'game-pick', 'aria-label': 'Choose a game' });
    var weeks = {};
    games.forEach(function (g) { (weeks[g.week] = weeks[g.week] || []).push(g); });
    Object.keys(weeks).sort(function (a, b) { return a - b; }).forEach(function (wk) {
      var og = h('optgroup', { label: 'Week ' + wk });
      weeks[wk].forEach(function (g) { og.appendChild(h('option', { value: g.game_id, text: matchup(g) + ', ' + g.away_score + '-' + g.home_score + (g.overtime ? ' (OT)' : '') })); });
      sel.appendChild(og);
    });
    sel.addEventListener('change', function () { pick(sel.value); });
    var prev = h('button', { type: 'button', class: 'btn', text: 'Previous game' });
    var next = h('button', { type: 'button', class: 'btn', text: 'Next game' });
    prev.addEventListener('click', function () { step(-1); });
    next.addEventListener('click', function () { step(1); });
    ui.sel = sel;

    ui.head = h('p', { class: 'ro-sub' });
    ui.chart = h('div', { class: 'chart' });
    ui.facts = h('dl', { class: 'statrow' });
    ui.note = h('p', { class: 'swing-note' });
    ui.ot = h('p', { class: 'fnote', hidden: true });
    ui.tv = N.tableViewLive('View this game as a table',
      function () { return [{ t: 'Game time' }, { t: 'Seconds', r: 1, n: 1 }, { t: curGame.home + ' win probability', r: 1, n: 1 }, { t: curGame.away + ' win probability', r: 1, n: 1 }]; },
      function () { return curGame.t.map(function (t, i) { return [gameTime(t), String(t), pct(curGame.wp[i], 1), pct(1 - curGame.wp[i], 1)]; }); });

    body.appendChild(h('div', { class: 'pick' },
      h('div', { class: 'field' }, h('label', { for: 'game-pick', text: 'Game' }), h('div', { class: 'sel' }, sel)),
      h('div', { class: 'pinrow', style: 'margin-top:0' }, prev, next)));
    body.appendChild(h('div', { style: 'margin-top:24px' },
      ui.head,
      h('div', { style: 'margin-top:14px' }, N.legend([['s3', 'Home win probability, our model'], ['ring', 'Biggest swing']])),
      well(ui.chart), ui.ot, ui.note, ui.facts, ui.tv.el));
    // the legend swatch for the biggest swing is a ring, drawn in CSS
    var best = W.games_by_excitement[0];
    watch(ui.chart, function () { drawChart(); });
    pick(best ? best.game_id : games[0].game_id, true);
  }

  function step(dir) {
    var ids = W.games.map(function (g) { return g.game_id; }), i = ids.indexOf(curGame.game_id) + dir;
    if (i < 0) i = ids.length - 1; if (i >= ids.length) i = 0;
    pick(ids[i]);
  }

  function pick(id, first) {
    var g = byId[id];
    if (!g) return;
    curGame = g;
    ui.sel.value = id;
    var win = winner(g), lowest = Math.min.apply(null, g.wp.map(function (v, i) { return win === g.home ? v : 1 - v; }));
    var rank = W.games_by_excitement.filter(function (x) { return x.game_id === id; })[0];
    var line;
    if (!win) line = 'Week ' + g.week + ': ' + matchup(g) + ' ended tied ' + g.away_score + '-' + g.home_score;
    else {
      var lose = win === g.home ? g.away : g.home, ws = Math.max(g.home_score, g.away_score), ls = Math.min(g.home_score, g.away_score);
      line = 'Week ' + g.week + ': ' + win + (lowest < 0.25 ? ' came back from ' + Math.round(lowest * 100) + '% win probability to beat ' + lose + ' ' + ws + '-' + ls : ' beat ' + lose + ' ' + ws + '-' + ls + ' and never fell below ' + Math.round(lowest * 100) + '% win probability');
    }
    setText('chart-h', line);
    ui.head.textContent = N.teamName(g.away) + ' at ' + N.teamName(g.home) + ', ' + g.date + '. Home win probability by game time; the line starts from the pregame estimate and ends at the result.';
    var sw = g.swing;
    clear(ui.note);
    ui.note.append(h('b', { text: 'Biggest swing: ' }), sw.clock + ', ' + sw.offense + ' ball. ' + g.home + '\'s win probability moved from ' + pct(sw.wp_before) + ' to ' + pct(sw.wp_after) + ' (' + signed(sw.delta_home * 100, 1) + ' points). ' + sw.desc);
    ui.ot.hidden = !g.overtime;
    ui.ot.textContent = g.overtime ? 'This game went to overtime. Overtime is not modelled: the line stops at the end of regulation and jumps to the result.' : '';
    clear(ui.facts);
    [
      ['Final', g.away + ' ' + g.away_score + ', ' + g.home + ' ' + g.home_score, 'Week ' + g.week + (g.overtime ? ', overtime' : '')],
      ['Pregame, ' + g.home + ' to win', pct(g.pregame_home, 0), 'A4s game model, logged before kickoff'],
      ['Excitement index', g.excitement.toFixed(1), rank ? 'rank ' + rank.rank + ' of ' + W.games.length + ' this season' : ''],
      [win ? win + '\'s lowest point' : 'Lowest point', g.winner_min_wp == null ? 'n/a' : pct(g.winner_min_wp, 0), win ? 'win probability, a comeback measure' : 'no winner'],
      ['Biggest swing', pts(sw.abs) + ' pts', sw.clock]
    ].forEach(function (f) { ui.facts.appendChild(h('div', {}, h('dt', { text: f[0] }), h('dd', { class: 'num' }, f[1], h('small', { text: f[2] })))); });
    drawChart();
    ui.tv.refresh();
    renderLeaderRows();
    N.live.textContent = line + '.';
  }

  var cursorIdx = -1;
  function drawChart() {
    var host = ui.chart, g = curGame;
    if (!host || !g) return;
    clear(host);
    var Wd = Math.max(300, host.clientWidth || 700), Ht = Wd < 560 ? 300 : Wd > 1400 ? 460 : 400;
    var m = { t: 18, r: 14, b: 40, l: 46 };
    var iw = Wd - m.l - m.r, ih = Ht - m.t - m.b;
    function sx(t) { return m.l + t / 3600 * iw; }
    function sy(p) { return m.t + (1 - p) * ih; }
    var svg = sv('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, width: Wd, height: Ht, tabindex: 0, role: 'application',
      'aria-label': 'Win probability for ' + matchup(g) + ' by game time. Use the left and right arrow keys to move through the game; the table below lists every point.' }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
      sv('line', { class: p === 0.5 ? 'refline' : 'grid', x1: m.l, x2: Wd - m.r, y1: sy(p), y2: sy(p) }, svg);
      sv('text', { class: 'tick a-end', x: m.l - 6, y: sy(p) + 4, text: Math.round(p * 100) + '%' }, svg);
    });
    [900, 1800, 2700].forEach(function (t) { sv('line', { class: t === 1800 ? 'wp-half' : 'wp-q', x1: sx(t), x2: sx(t), y1: m.t, y2: m.t + ih }, svg); });
    ['Q1', 'Q2', 'Q3', 'Q4'].forEach(function (q, i) { sv('text', { class: 'tick a-mid', x: sx(i * 900 + 450), y: Ht - 18, text: q }, svg); });
    sv('text', { class: 'tick a-mid', x: m.l + iw / 2, y: Ht - 3, text: 'Regulation game time', style: 'fill:var(--muted)' }, svg);
    sv('text', { class: 'wp-side', x: m.l + 8, y: m.t + 16, text: g.home + ' win' }, svg);
    sv('text', { class: 'wp-side', x: m.l + 8, y: m.t + ih - 8, text: g.away + ' win' }, svg);

    var d = '', i;
    for (i = 0; i < g.t.length; i++) d += (i ? 'L' : 'M') + sx(g.t[i]).toFixed(1) + ',' + sy(g.wp[i]).toFixed(1) + ' ';
    sv('path', { class: 'wp-area', d: d + 'L' + sx(g.t[g.t.length - 1]).toFixed(1) + ',' + sy(0.5) + ' L' + sx(g.t[0]).toFixed(1) + ',' + sy(0.5) + ' Z' }, svg);
    sv('path', { class: 'wp-line', d: d }, svg);

    // biggest swing: the largest single step, marked at the state it ended in
    var bi = 0, bd = -1;
    for (i = 0; i < g.wp.length - 1; i++) { var dd = Math.abs(g.wp[i + 1] - g.wp[i]); if (dd > bd) { bd = dd; bi = i; } }
    var bx = sx(g.t[bi + 1]), by = sy(g.wp[bi + 1]);
    sv('line', { x1: sx(g.t[bi]), y1: sy(g.wp[bi]), x2: bx, y2: by, stroke: 'var(--ink)', 'stroke-width': 3.5, 'stroke-linecap': 'butt' }, svg);
    sv('circle', { class: 'wp-swing', cx: bx, cy: by, r: 7 }, svg);
    var above = g.wp[bi + 1] < 0.7, anchorEnd = bx > Wd - 130, anchorStart = bx < m.l + 70;
    sv('text', { class: 'tick halo', x: bx + (anchorEnd ? -12 : anchorStart ? 12 : 0), y: above ? by - 14 : by + 22, 'text-anchor': anchorEnd ? 'end' : anchorStart ? 'start' : 'middle', text: 'Biggest swing', style: 'fill:var(--ink);font-weight:700' }, svg);
    var end = g.wp.length - 1;
    N.mark(svg, 's3', sx(g.t[end]), sy(g.wp[end]), 6);

    var cross = sv('line', { class: 'cross', x1: 0, x2: 0, y1: m.t, y2: m.t + ih, visibility: 'hidden' }, svg);
    var dot = N.mark(svg, 's3', 0, 0, 7); dot.setAttribute('visibility', 'hidden');
    function showIdx(k, cx, cy) {
      cursorIdx = k;
      var x = sx(g.t[k]);
      cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
      N.moveMark(dot, x, sy(g.wp[k])); dot.setAttribute('visibility', 'visible');
      var rows = [{ value: pct(g.wp[k]), label: g.home + ' win', cls: 's3' }, { value: pct(1 - g.wp[k]), label: g.away + ' win' }];
      if (k === bi) rows.push({ value: '', label: 'Biggest swing starts here: ' + g.swing.desc });
      else if (k === bi + 1) rows.push({ value: '', label: 'After the biggest swing (' + g.swing.clock + ')' });
      tipShow(cx, cy, gameTime(g.t[k]), rows);
    }
    function idxAt(clientX) {
      var r = svg.getBoundingClientRect(), x = (clientX - r.left) * (Wd / r.width);
      var t = Math.max(0, Math.min(3600, (x - m.l) / iw * 3600)), lo = 0, hi = g.t.length - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (g.t[mid] <= t) lo = mid; else hi = mid; }
      return Math.abs(g.t[lo] - t) <= Math.abs(g.t[hi] - t) ? lo : hi;
    }
    function hide() { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); tipHide(); }
    svg.addEventListener('pointermove', function (e) { showIdx(idxAt(e.clientX), e.clientX, e.clientY); });
    svg.addEventListener('pointerleave', hide);
    function focusAt(k) {
      var r = svg.getBoundingClientRect(), sc = r.width / Wd;
      showIdx(k, r.left + sx(g.t[k]) * sc, r.top + sy(g.wp[k]) * sc);
    }
    svg.addEventListener('focus', function () { focusAt(Math.min(g.t.length - 1, Math.max(0, cursorIdx < 0 ? bi : cursorIdx))); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (e) {
      var k = cursorIdx < 0 ? bi : cursorIdx, n = g.t.length - 1;
      if (e.key === 'ArrowRight') k += e.shiftKey ? 10 : 1;
      else if (e.key === 'ArrowLeft') k -= e.shiftKey ? 10 : 1;
      else if (e.key === 'Home') k = 0; else if (e.key === 'End') k = n;
      else if (e.key === 's' || e.key === 'S') k = bi;
      else return;
      e.preventDefault(); focusAt(Math.max(0, Math.min(n, k)));
    });
    cursorIdx = -1;
  }

  // ------------------------------------------------------------------ excitement leaderboard

  var SORTS = {
    excitement: { label: 'Excitement', fn: function (a, b) { return b.excitement - a.excitement; } },
    swing: { label: 'Biggest swing', fn: function (a, b) { return b.biggest_swing - a.biggest_swing; } },
    low: { label: 'Winner\'s low point', fn: function (a, b) { return (a.winner_min_wp == null ? 2 : a.winner_min_wp) - (b.winner_min_wp == null ? 2 : b.winner_min_wp); } }
  };

  function buildLeaderboard() {
    var body = done('excite-body');
    clear(body);
    var L = W.games_by_excitement;
    if (!L.length) { body.appendChild(h('p', { class: 'note empty', text: 'No completed games yet.' })); return; }
    var max = Math.max.apply(null, L.map(function (g) { return g.excitement; }));
    ui.max = max;
    ui.rows = h('div', { class: 'xlist' });
    ui.heads = h('div', { class: 'xhead' },
      h('span', { class: 'r', text: '#' }), h('span', { text: 'Game' }), h('span', { class: 'hide-m', text: 'Week' }),
      sortHead('excitement'), sortHead('swing', 'r hide-m'), sortHead('low', 'r hide-m'));
    ui.more = h('button', { type: 'button', class: 'btn xmore' });
    ui.more.addEventListener('click', function () { showAll = !showAll; renderLeaderRows(); });
    body.appendChild(h('p', { class: 'fnote', style: 'margin-top:0', text: 'Through week ' + W.updated_through.week + ': ' + commas(L.length) + ' games, sorted by the column marked. The excitement index is the sum of the snap-to-snap changes in home win probability; the biggest swing is the largest single change; the winner\'s low point is the lowest win probability the eventual winner had.' }));
    ui.xseg = N.seg('Sort the games', Object.keys(SORTS).map(function (k) { return { value: k, label: SORTS[k].label }; }), sortKey, function (v) { sortKey = v; showAll = false; renderLeaderRows(); }, 'xseg');
    body.appendChild(ui.xseg.el);
    body.appendChild(h('div', { style: 'margin-top:18px' }, ui.heads, ui.rows));
    body.appendChild(ui.more);
    var hb = W.updated_through.held_back || [];
    if (hb.length) {
      body.appendChild(h('p', { class: 'fnote heldback', text: 'Held back this update: ' + hb.map(heldName).join(', ') + '. ' + (hb.length === 1 ? 'It is' : 'They are') + ' final in nflverse but wait for the rating update that supplies the pregame input, so ' + (hb.length === 1 ? 'it appears' : 'they appear') + ' on the next update.' }));
    }
    body.appendChild(h('p', { class: 'fnote', text: 'Excitement measures movement in the model\'s estimate, not quality of play or how close the score was.' }));
    renderLeaderRows();
  }

  function heldName(id) { var p = id.split('_'); return p.length === 4 ? 'week ' + Number(p[1]) + ', ' + p[2] + ' at ' + p[3] : id; }

  function sortHead(k, cls) {
    var b = h('button', { type: 'button', class: 'sortb' + (k === sortKey ? ' on' : ''), 'data-dir': 'desc', text: SORTS[k].label });
    b.addEventListener('click', function () { sortKey = k; showAll = false; renderLeaderRows(); });
    b.dataset.k = k;
    return h('span', { class: cls || '' }, b);
  }

  function renderLeaderRows() {
    if (!ui.rows) return;
    if (ui.xseg) ui.xseg.set(sortKey);
    var L = W.games_by_excitement.slice().sort(SORTS[sortKey].fn);
    Array.prototype.forEach.call(ui.heads.querySelectorAll('.sortb'), function (b) {
      var on = b.dataset.k === sortKey;
      b.classList.toggle('on', on);
      b.setAttribute('data-dir', sortKey === 'low' ? 'asc' : 'desc');
      b.parentNode.setAttribute('aria-sort', on ? (sortKey === 'low' ? 'ascending' : 'descending') : 'none');
    });
    var n = showAll ? L.length : Math.min(10, L.length);
    clear(ui.rows);
    L.slice(0, n).forEach(function (g) {
      var full = byId[g.game_id];
      var row = h('button', { type: 'button', class: 'xrow', 'aria-current': curGame && curGame.game_id === g.game_id ? 'true' : null, 'aria-label': 'Open ' + matchup(g) + ', week ' + g.week + ', in the chart. Excitement ' + g.excitement.toFixed(1) },
        h('span', { class: 'rk num', text: String(g.rank) }),
        h('span', { class: 'gm' }, matchup(g), h('small', {}, g.away_score + '-' + g.home_score + (g.overtime ? ' (OT)' : '') + ', week ' + g.week, h('span', { class: 'mini', text: ', swing ' + pts(g.biggest_swing) + ' pts, winner low ' + (g.winner_min_wp == null ? 'n/a' : pct(g.winner_min_wp, 0)) }))),
        h('span', { class: 'wk num', text: 'Wk ' + g.week }),
        h('span', { class: 'xv' }, h('span', { class: 'num', text: g.excitement.toFixed(1) }), h('span', { class: 'xb' }, h('i', { style: 'width:' + (g.excitement / ui.max * 100).toFixed(1) + '%' }))),
        h('span', { class: 'sw num' }, pts(g.biggest_swing) + ' pts'),
        h('span', { class: 'lw num' }, g.winner_min_wp == null ? 'n/a' : pct(g.winner_min_wp, 0)));
      row.addEventListener('click', function () {
        pick(g.game_id);
        var c = document.getElementById('chart');
        if (c) c.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      });
      ui.rows.appendChild(row);
    });
    ui.more.hidden = L.length <= 10;
    ui.more.textContent = showAll ? 'Show the top 10' : 'Show all ' + L.length + ' games';
  }

  // ------------------------------------------------------------------ receipts and method

  function buildReceipts() {
    var body = done('receipts-body'), H = W.headline, d = H.ours_minus_nflfastr;
    clear(body);
    setText('receipts-h', 'Ours beat nflfastR on seasons it never saw');
    var host = h('div', { class: 'chart ci' });
    body.appendChild(h('div', {},
      h('p', { class: 'stat-note', style: 'margin-top:0', text: H.plain }),
      well(host),
      h('p', { class: 'fnote', text: 'Brier score difference, ours minus nflfastR\'s public win probability, on ' + commas(H.plays) + ' plays from ' + commas(d.clusters) + ' games (' + H.window + '). The Brier score is the average squared miss of the win probability, so negative means ours is closer. The interval is 95%, resampled by game.' }),
      h('dl', { class: 'statrow' },
        stat('Brier, ours', H.brier_ours.toFixed(4), 'lower is better'),
        stat('Brier, nflfastR', H.brier_nflfastr.toFixed(4), 'public model, same plays'),
        stat('Difference', signed(d.diff, 4), '95% interval ' + signed(d.lo, 4) + ' to ' + signed(d.hi, 4)),
        stat('Calibration error (ECE)', H.ece_ours.toFixed(4), 'how far stated chances sit from results'))));
    watch(host, function () { N.ciChart(host, [{ label: 'Ours minus nflfastR', ci: d, series: 's3' }], { decimals: 4, axis: 'Brier score, ours minus nflfastR', measure: 'Brier difference', title: 'Ours minus nflfastR' }); });
  }
  function stat(label, value, small) { return h('div', {}, h('dt', { text: label }), h('dd', { class: 'num' }, value, h('small', { text: small }))); }

  function buildHow() {
    var dyn = $('#how-dyn'), F = W.model_fit, D = W.definitions;
    dyn.appendChild(h('div', {},
      h('h3', { class: 'sub-h', text: 'The model' }),
      h('p', { class: 'stat-note', text: 'Fit on ' + F.trained_on[0] + ' to ' + F.trained_on[1] + ' plays (' + commas(F.n_train) + ' snaps) as a monotone gradient-boosted model, frozen before the ' + W.season + ' season. Inputs: ' + F.inputs + '. Updated through week ' + W.updated_through.week + '.' })));
    dyn.appendChild(N.caveatBox(W.caveats, 'Caveats'));
    dyn.appendChild(h('dl', { class: 'gloss' },
      h('h3', { text: 'What each number means' }),
      Object.keys(D).map(function (k) { return h('div', {}, h('dt', { text: k }), h('dd', { text: D[k] })); })));
    $('#how-rcpt').appendChild(N.receiptLinks(W.headline.receipt));
    setText('how-credit', W.license.credit);
  }

  // ------------------------------------------------------------------ boot

  function render() {
    W.games.forEach(function (g) { byId[g.game_id] = g; });
    renderHero();
    buildLeaderboard();
    buildChart();
    buildReceipts();
    buildHow();
    renderLeaderRows();
  }

  var ALL = ['chart-body', 'excite-body', 'receipts-body'];
  Promise.all([N.getJSON('data/winprob.json', true), N.loadTeamNames()]).then(function (r) {
    if (!r[0] || !r[0].games) { N.unavailable('The win probability data', ALL); return; }
    W = r[0];
    render();
  }).catch(function (err) { console.error(err); N.unavailable('The win probability data', ALL); });
})();
