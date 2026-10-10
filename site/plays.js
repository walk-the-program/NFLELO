/* NFLELO play outcomes page. Reads ./data/plays.json (written by scripts/export_ml_pages.py) and draws the play
   console: the model's whole yard distribution for one chosen situation, with an optional pinned setting to compare.
   No dependencies; DOM, tooltip, table-view and fetch helpers come from common.js. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, tipHide = N.tipHide, watch = N.watch;
  var signed = N.signed, commas = N.commas, pct = N.pct, ciText = N.ciText;
  var setText = N.setText, done = N.done;

  var P = null;            // plays.json
  var idx = {};            // cell lookup
  var S = { down: 1, ydstogo: 10, yl: 80, call: 'pass', pers: '11' };
  var pinned = null;       // a copy of S, or null
  var ui = {};             // control handles and output nodes
  var NBIN = 52;           // yard bins; slot layout adds a gap, TD, a gap and turnover
  var SLOT_TD = 53, SLOT_TOV = 56, SLOTS = 57;
  function isGap(s) { return s === 52 || s === 54 || s === 55; }

  function key(s) { return [s.down, s.ydstogo, s.yl, s.call, s.pers].join('|'); }
  function cellFor(s) { return idx[key(s)] || null; }
  function binP(c, i) { var k = i - c.bins_from; return k >= 0 && k < c.bins.length ? c.bins[k] : 0; }
  function downName(d) { return N.ordinal(d); }
  function situation(s) { return downName(s.down) + ' and ' + s.ydstogo + ' at ' + N.fieldLabel(s.yl); }
  function binName(i) {
    if (i === 0) return '10 or more yards lost';
    if (i === 51) return '41 or more yards';
    var y = P.bins.yards[i];
    return y < 0 ? 'Loss of ' + (-y) + (y === -1 ? ' yard' : ' yards') : y === 0 ? 'No gain' : 'Gain of ' + y + (y === 1 ? ' yard' : ' yards');
  }
  function binShort(i) { return i === 0 ? '10+ lost' : i === 51 ? '41+' : (P.bins.yards[i] > 0 ? '+' : '') + P.bins.yards[i]; }

  // ------------------------------------------------------------------ hero and receipts

  function renderHero() {
    var hc = P.headline.call, hs = P.headline.situation, d = hc.crps_model_minus_baseline;
    var every = /every season/.test(P.headline.plain || '');
    setText('hero-line', 'On the locked 2020 to 2025 seasons the play model beat the historical baseline: ' + hc.crps_model.toFixed(3) + ' against ' + hc.crps_baseline.toFixed(3) + ' yards of average error' + (every ? ', in every season' : '') + '.');
    setText('hero-deck', 'Set a situation and see the whole spread of yards the play model expects, not just an average. It sees only what is known before the snap, and it was tested on seasons it never saw.');
    N.heroTiles([
      { label: 'Gain over the baseline', big: signed(d.diff, 3), sub: 'yards of CRPS, 95% interval ' + signed(d.lo, 3) + ' to ' + signed(d.hi, 3) + '. Negative is better.' },
      { label: 'Model error (CRPS)', big: [hc.crps_model.toFixed(3), h('small', { text: ' yd' })], sub: 'against ' + hc.crps_baseline.toFixed(3) + ' for the down, distance and field position baseline' },
      { label: 'Plays scored once', big: commas(hc.plays), sub: P.headline.window.replace(', locked holdout, scored once', '') },
      { label: 'Situations to explore', big: commas(P.grid.cells), sub: 'trained on ' + commas(P.n_train_plays) + ' plays, ' + P.trained_on[0] + ' to ' + P.trained_on[1] }
    ]);
    N.licenseUI(P.license);
  }

  function renderHow() {
    var hc = P.headline.call, hs = P.headline.situation;
    var dyn = $('#how-dyn');
    var host = h('div', { class: 'chart ci' });
    dyn.appendChild(h('div', {},
      h('h3', { class: 'sub-h', text: 'The receipt' }),
      h('p', { class: 'stat-note', text: P.headline.plain }),
      well(host),
      h('p', { class: 'fnote', text: 'Model minus baseline, in yards of CRPS (average error of the whole yard distribution). Negative means the model is closer. The call view is the one on this page; the situation view leaves out the called play type. Intervals are 95%, resampled by game.' }),
      h('dl', { class: 'statrow' },
        stat('Call view, model', hc.crps_model.toFixed(3) + ' yd', 'baseline ' + hc.crps_baseline.toFixed(3)),
        stat('Situation view, model', hs.crps_model.toFixed(3) + ' yd', 'baseline ' + hs.crps_baseline.toFixed(3)),
        stat('Chain-moving chance, calibration error', hc.p_first_ece.toFixed(4), hc.p_first_calibration_ok ? 'within the ' + hc.p_first_ece_floor.toFixed(2) + ' floor set before the test' : 'above the ' + hc.p_first_ece_floor.toFixed(2) + ' floor'),
        stat('Plays tested', commas(hc.plays), commas(hc.crps_model_minus_baseline.clusters) + ' games'))));
    watch(host, function () {
      N.ciChart(host, [
        { label: 'Call view (this page)', ci: hc.crps_model_minus_baseline, series: 's3' },
        { label: 'Situation view', ci: hs.crps_model_minus_baseline, series: 's3' }
      ], { decimals: 3, axis: 'yards of CRPS, model minus baseline', measure: 'CRPS difference', title: 'Model minus baseline' });
    });
    dyn.appendChild(N.caveatBox(P.caveats, 'Caveats'));
    dyn.appendChild(h('div', {},
      h('h3', { class: 'sub-h', text: 'Held neutral' }),
      h('p', { class: 'stat-note', text: heldText(true) })));
    $('#how-rcpt').appendChild(N.receiptLinks(hc.receipt));
    setText('how-credit', P.license.credit);
  }
  function stat(label, value, small) { return h('div', {}, h('dt', { text: label }), h('dd', { class: 'num' }, value, h('small', { text: small }))); }

  function heldText(long) {
    var g = P.held_neutral, clock = String(g.clock || '').split(' (')[0];
    var t = (g.score_diff === 0 ? 'tied game' : 'score difference ' + g.score_diff) + ', ' + clock + ', ' + String(g.timeouts) + ' timeouts, ' + (g.roof || 'outdoors') + ' at ' + g.temp_f + 'F and ' + g.wind_mph + ' mph wind, league-average teams, home field averaged.';
    return long ? 'Everything not on the console is held at: ' + t + ' Defensive personnel, formation and box count are not fixed: each result averages the model over 32 real defensive looks that fit the offense and the call. Home field is averaged.' : 'Held neutral: ' + t.replace(', home field averaged.', '.');
  }

  // ------------------------------------------------------------------ console

  function buildConsole() {
    var body = done('console-body');
    clear(body);
    var G = P.grid;

    ui.down = N.seg('Down', G.down.map(function (d) { return { value: d, label: N.ordinal(d) }; }), S.down, function (v) { S.down = v; update(); });
    ui.dist = N.seg('Yards to go', G.ydstogo.map(function (d) { return { value: d, label: String(d) }; }), S.ydstogo, function (v) { S.ydstogo = v; update(); });
    ui.call = N.seg('Called play', G.call.map(function (c) { return { value: c, label: c === 'run' ? 'Run' : 'Pass' }; }), S.call, function (v) { S.call = v; update(); });
    var persKeys = Object.keys(G.personnel);
    ui.pers = N.seg('Offensive personnel', persKeys.map(function (k) { return { value: k, label: k, title: G.personnel[k] }; }), S.pers, function (v) { S.pers = v; update(); });

    var yls = G.yardline_100.slice().sort(function (a, b) { return b - a; });  // own end zone first
    ui.yls = yls;
    var input = h('input', { type: 'range', min: 0, max: yls.length - 1, step: 1, value: yls.indexOf(S.yl), 'aria-label': 'Field position' });
    input.addEventListener('input', function () { S.yl = yls[Number(input.value)]; update(); });
    ui.slider = input;
    ui.range = h('div', { class: 'sf-range single' }, h('div', { class: 'sf-range__track' }), h('div', { class: 'sf-range__fill' }), input);
    var ticks = h('div', { class: 'ticks', 'aria-hidden': 'true' }, yls.map(function (y, i) { return h('span', { style: '--i:' + i, text: N.fieldLabel(y).replace('midfield', '50') }); }));

    ui.fieldVal = h('span', { class: 'val' });
    ui.persCap = h('p', { class: 'cap' });
    ui.distCap = h('p', { class: 'cap' });
    ui.pinBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', text: 'Pin this setting' });
    ui.pinBtn.addEventListener('click', function () { pinned = pinned ? null : JSON.parse(JSON.stringify(S)); update(); });
    ui.pinTxt = h('p', { class: 'pinned-txt' });

    var controls = h('div', { class: 'console-in' },
      ctl('Down', ui.down.el),
      ctl('Yards to go', ui.dist.el, ui.distCap),
      h('div', { class: 'ctl' },
        h('div', { class: 'ctl-h' }, h('span', { class: 'label', text: 'Field position' }), ui.fieldVal),
        ui.range, ticks),
      ctl('Called play', ui.call.el),
      ctl('Offensive personnel', ui.pers.el, ui.persCap),
      h('div', { class: 'pinrow' }, ui.pinBtn, ui.pinTxt));

    ui.title = h('h3', { class: 'ro-h' });
    ui.sub = h('p', { class: 'ro-sub' });
    ui.thin = h('div', { class: 'thin', role: 'status', hidden: true });
    ui.chart = h('div', { class: 'chart' });
    ui.bigs = h('dl', { class: 'bigs' });
    ui.tv = N.tableViewLive('View this distribution as a table',
      function () { var c = [{ t: 'Result' }, { t: 'This setting', r: 1, n: 1 }]; if (pinned) c.push({ t: 'Pinned', r: 1, n: 1 }); return c; },
      tableRows);
    var out = h('div', { class: 'console-out', 'aria-live': 'polite' },
      ui.title, ui.sub, ui.thin, well(ui.chart), ui.bigs, ui.tv.el,
      h('p', { class: 'held', text: heldText(false) + ' Result probabilities are model estimates for the snap, not promises.' }));

    body.appendChild(h('div', { class: 'console sf-glass' }, controls, out));
    watch(ui.chart, function () { drawChart(); });
    update();
  }

  function ctl(label, el, cap) { return h('div', { class: 'ctl' }, h('div', { class: 'ctl-h' }, h('span', { class: 'label', text: label })), el, cap || null); }

  function update() {
    // distances longer than the distance to the goal line do not exist: disable them and snap
    ui.dist.disable(function (v) { return v > S.yl; });
    if (S.ydstogo > S.yl) {
      var ok = P.grid.ydstogo.filter(function (v) { return v <= S.yl; });
      S.ydstogo = ok[ok.length - 1];
    }
    ui.dist.set(S.ydstogo);
    ui.slider.value = ui.yls.indexOf(S.yl);
    var f = ui.yls.indexOf(S.yl) / (ui.yls.length - 1);
    ui.range.style.setProperty('--hi', f);
    ui.slider.setAttribute('aria-valuetext', N.fieldLabel(S.yl));
    ui.fieldVal.textContent = N.fieldLabel(S.yl) + (S.ydstogo === S.yl ? ', goal to go' : '');
    ui.persCap.textContent = P.grid.personnel[S.pers] || '';
    ui.distCap.textContent = P.grid.ydstogo.some(function (v) { return v > S.yl; }) ? 'Distances longer than the field position are not possible and are dimmed.' : '';
    ui.pinBtn.textContent = pinned ? 'Clear pin' : 'Pin this setting';
    ui.pinBtn.setAttribute('aria-pressed', pinned ? 'true' : 'false');
    ui.pinTxt.textContent = pinned ? 'Pinned (outline): ' + desc(pinned) + '. The solid bars show the current setting.' : 'Pin a setting, then change the console to compare. The pinned distribution is drawn as an outline.';

    var c = cellFor(S);
    var call = S.call === 'run' ? 'run' : 'pass';
    var goal = S.ydstogo === S.yl;
    ui.title.textContent = desc(S);
    ui.sub.textContent = c ? 'Spread of yards on the snap, from the model, as of the situation held neutral below.' : 'This combination is not in the grid.';
    setText('console-h', c ? situation(S) + ': a ' + call + ' from ' + S.pers + ' personnel gains about ' + c.exp_yards.toFixed(1) + ' yards and ' + (goal ? 'scores' : 'reaches the line to gain or scores') + ' ' + pct(c.p_first_or_td, 0) + ' of the time' : 'No data for this setting');

    clear(ui.thin);
    if (c && c.n_similar < 30) {
      ui.thin.hidden = false;
      ui.thin.appendChild(h('span', { class: 'vchip v-negative', text: 'Thin support' }));
      ui.thin.appendChild(h('span', { text: c.n_similar === 0
        ? 'Thin support: few similar real plays. There are none in 2016 to 2025 with this down, distance, call and personnel near this yard line, so this estimate rests on the model\'s smoothing, not on direct data.'
        : 'Thin support: few similar real plays. Only ' + c.n_similar + ' in 2016 to 2025 had this down, distance, call and personnel near this yard line, so treat the shape as approximate.' }));
    } else ui.thin.hidden = true;

    drawStats(c);
    drawChart();
    ui.tv.refresh();
  }

  function desc(s) { return situation(s) + ', ' + s.call + ' from ' + s.pers + ' personnel'; }

  function drawStats(c) {
    var pc = pinned ? cellFor(pinned) : null;
    var items = [
      ['Expected yards', function (x) { return x.exp_yards.toFixed(1); }, function (a, b) { return signed(a.exp_yards - b.exp_yards, 1); }],
      ['First down or TD', function (x) { return pct(x.p_first_or_td); }, function (a, b) { return signed((a.p_first_or_td - b.p_first_or_td) * 100, 1) + ' pts'; }],
      ['20 or more yards', function (x) { return pct(x.p_20plus); }, function (a, b) { return signed((a.p_20plus - b.p_20plus) * 100, 1) + ' pts'; }],
      ['Yards lost', function (x) { return pct(x.p_loss); }, function (a, b) { return signed((a.p_loss - b.p_loss) * 100, 1) + ' pts'; }],
      ['Turnover', function (x) { return pct(x.p_turnover); }, function (a, b) { return signed((a.p_turnover - b.p_turnover) * 100, 1) + ' pts'; }]
    ];
    clear(ui.bigs);
    items.forEach(function (it) {
      ui.bigs.appendChild(h('div', {}, h('dt', { text: it[0] }),
        h('dd', {}, c ? it[1](c) : 'n/a', h('span', { class: 'pin', text: c && pc ? 'Pinned ' + it[1](pc) + ' (' + it[2](c, pc) + ')' : '' }))));
    });
  }

  function tableRows() {
    var c = cellFor(S), pc = pinned ? cellFor(pinned) : null;
    if (!c) return [];
    var rows = [];
    for (var i = 0; i < NBIN; i++) {
      var a = binP(c, i), b = pc ? binP(pc, i) : 0;
      if (a < 0.0005 && b < 0.0005) continue;
      var r = [binName(i), pct(a, 2)];
      if (pinned) r.push(pc ? pct(b, 2) : 'n/a');
      rows.push(r);
    }
    var td = [ 'Touchdown', pct(c.p_td, 2)]; if (pinned) td.push(pc ? pct(pc.p_td, 2) : 'n/a'); rows.push(td);
    var tv = ['Turnover (separate model, overlaps the rows above)', pct(c.p_turnover, 2)]; if (pinned) tv.push(pc ? pct(pc.p_turnover, 2) : 'n/a'); rows.push(tv);
    return rows;
  }

  // ------------------------------------------------------------------ chart

  var cur = -1;      // keyboard / pointer cursor slot
  function slotName(c, slot) {
    if (slot < NBIN) return binName(slot);
    return slot === SLOT_TD ? 'Touchdown' : slot === SLOT_TOV ? 'Turnover (separate model)' : null;
  }
  function slotP(c, slot) {
    if (!c) return null;
    return slot < NBIN ? binP(c, slot) : slot === SLOT_TD ? c.p_td : slot === SLOT_TOV ? c.p_turnover : null;
  }

  function drawChart() {
    var host = ui.chart;
    if (!host) return;
    clear(host);
    var c = cellFor(S), pc = pinned ? cellFor(pinned) : null;
    if (!c) { host.appendChild(h('p', { class: 'note', text: 'No distribution for this combination. Change the down, distance or field position.' })); return; }
    var W = Math.max(280, host.clientWidth || 640), narrow = W < 560, H = narrow ? 290 : 340;
    var m = { t: 30, r: 10, b: narrow ? 74 : 58, l: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, bw = iw / SLOTS;
    var vals = [c.p_td, c.p_turnover];
    for (var i = 0; i < NBIN; i++) vals.push(binP(c, i), pc ? binP(pc, i) : 0);
    if (pc) vals.push(pc.p_td, pc.p_turnover);
    var vmax = Math.max.apply(null, vals);
    var step = vmax > 0.3 ? 0.1 : vmax > 0.12 ? 0.05 : vmax > 0.05 ? 0.02 : 0.01;
    var top = Math.ceil(vmax / step) * step;
    function sy(p) { return m.t + ih - p / top * ih; }
    function sx(slot) { return m.l + slot * bw; }

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, tabindex: 0, role: 'application',
      'aria-label': 'Yard distribution for ' + desc(S) + '. Use the left and right arrow keys to read each result; the table below lists the exact values.' }, host);

    for (var v = 0; v <= top + 1e-9; v += step) {
      sv('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, svg);
      sv('text', { class: 'tick a-end', x: m.l - 6, y: sy(v) + 4, text: Math.round(v * 100) + '%' }, svg);
    }
    // first-down line and the region it covers (the line to gain through the touchdown bar)
    var fdx = sx(S.ydstogo + 10);
    var zx0 = fdx, zx1 = sx(SLOT_TD + 1);
    sv('rect', { class: 'fd-zone', x: zx0, y: m.t, width: zx1 - zx0, height: ih }, svg);
    sv('line', { class: 'fd-line', x1: fdx, x2: fdx, y1: m.t - 6, y2: m.t + ih }, svg);
    var lab = S.ydstogo === S.yl ? 'goal line' : 'line to gain';
    var anchorEnd = fdx > W - 150;
    sv('text', { class: 'tick halo', x: fdx + (anchorEnd ? -6 : 6), y: m.t - 12, 'text-anchor': anchorEnd ? 'end' : 'start', text: lab, style: 'fill:var(--ink)' }, svg);

    // bars
    var modal = 0, modalP = -1;
    for (var k = 0; k < NBIN; k++) {
      var p = binP(c, k);
      if (p > modalP) { modalP = p; modal = k; }
      if (p <= 0) continue;
      sv('rect', { class: 'yb', x: sx(k) + 0.5, y: sy(p), width: Math.max(1.5, bw - 1), height: Math.max(1, m.t + ih - sy(p)), rx: Math.min(2, bw / 2) }, svg);
    }
    sv('rect', { class: 'yb', x: sx(SLOT_TD) + 1, y: sy(c.p_td), width: Math.max(2, bw - 2), height: Math.max(1, m.t + ih - sy(c.p_td)), rx: 2 }, svg);
    sv('rect', { class: 'yb tov', x: sx(SLOT_TOV) + 1.5, y: sy(c.p_turnover), width: Math.max(2, bw - 3), height: Math.max(1, m.t + ih - sy(c.p_turnover)), rx: 2 }, svg);

    // pinned outline: a stepped line over the yard bars, rectangles for TD and turnover
    if (pc) {
      var d = '';
      for (var j = 0; j < NBIN; j++) {
        var x0 = sx(j), x1 = sx(j + 1), y = sy(binP(pc, j));
        d += (j === 0 ? 'M' : 'L') + x0 + ',' + y + ' L' + x1 + ',' + y + ' ';
      }
      sv('path', { d: d, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, svg);
      sv('rect', { class: 'yb pin', x: sx(SLOT_TD) + 1, y: sy(pc.p_td), width: Math.max(2, bw - 2), height: Math.max(1, m.t + ih - sy(pc.p_td)), rx: 2 }, svg);
      sv('rect', { class: 'yb pin', x: sx(SLOT_TOV) + 1.5, y: sy(pc.p_turnover), width: Math.max(2, bw - 3), height: Math.max(1, m.t + ih - sy(pc.p_turnover)), rx: 2, 'stroke-dasharray': '3 2' }, svg);
    }

    // axis labels
    sv('line', { class: 'refline', x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
    [[0, '10+ lost'], [10, '0'], [20, '10'], [30, '20'], [40, '30']].forEach(function (t) {
      sv('text', { class: 'tick a-mid', x: sx(t[0]) + bw / 2, y: m.t + ih + 16, text: t[1] }, svg);
    });
    sv('text', { class: 'tick a-end', x: sx(52), y: m.t + ih + 16, text: '41+' }, svg);
    sv('text', { class: 'tick a-start', x: sx(SLOT_TD) + 1, y: m.t + ih + 16, text: 'TD', style: 'fill:var(--ink-2);font-weight:700' }, svg);
    sv('text', { class: 'tick a-end', x: sx(SLOT_TOV + 1), y: m.t + ih + (narrow ? 50 : 32), text: 'Turnover (separate model)', style: 'fill:var(--ink-2);font-weight:700' }, svg);
    sv('text', { class: 'tick a-start', x: m.l, y: m.t + ih + 32, text: 'Yards gained on the snap' }, svg);

    // direct labels: the most likely yard result, the touchdown and the turnover
    function vlabel(slot, p, anchor) {
      if (p < 0.004) return;
      sv('text', { class: 'tick halo', x: sx(slot) + bw / 2, y: sy(p) - 5, 'text-anchor': anchor || 'middle', text: pct(p, p < 0.1 ? 1 : 0), style: 'fill:var(--ink);font-weight:700' }, svg);
    }
    vlabel(modal, modalP, modal < 6 ? 'start' : 'middle');
    vlabel(SLOT_TD, c.p_td, 'middle');
    vlabel(SLOT_TOV, c.p_turnover, 'end');

    // cursor
    var cursor = sv('rect', { class: 'cursor-bar', x: 0, y: m.t, width: bw, height: ih, rx: 2, visibility: 'hidden' }, svg);
    function showSlot(slot, cx, cy) {
      var name = slotName(c, slot);
      if (name == null) { cursor.setAttribute('visibility', 'hidden'); tipHide(); return; }
      cur = slot;
      cursor.setAttribute('x', sx(slot)); cursor.setAttribute('visibility', 'visible');
      var rows = [{ value: pct(slotP(c, slot), 2), label: 'this setting', cls: 's3' }];
      if (pc) rows.push({ value: pct(slotP(pc, slot), 2), label: 'pinned', cls: 'sm' });
      if (slot < NBIN && P.bins.yards[slot] >= S.ydstogo) rows.push({ value: '', label: 'reaches the line to gain' });
      tipShow(cx, cy, name, rows);
    }
    function slotAt(clientX) {
      var r = svg.getBoundingClientRect();
      var x = (clientX - r.left) * (W / r.width);
      var slot = Math.floor((x - m.l) / bw);
      return slot < 0 || slot >= SLOTS ? -1 : slot;
    }
    svg.addEventListener('pointermove', function (e) { var s = slotAt(e.clientX); if (s < 0 || isGap(s)) { cursor.setAttribute('visibility', 'hidden'); tipHide(); } else showSlot(s, e.clientX, e.clientY); });
    svg.addEventListener('pointerleave', function () { cursor.setAttribute('visibility', 'hidden'); tipHide(); });
    function focusTip(slot) {
      var r = svg.getBoundingClientRect(), sc = r.width / W;
      showSlot(slot, r.left + (sx(slot) + bw / 2) * sc, r.top + (m.t + ih / 2) * sc);
    }
    svg.addEventListener('focus', function () { focusTip(cur >= 0 && cur < SLOTS ? cur : modal); });
    svg.addEventListener('blur', function () { cursor.setAttribute('visibility', 'hidden'); tipHide(); });
    svg.addEventListener('keydown', function (e) {
      var s = cur >= 0 ? cur : modal, dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (e.key === 'Home') s = 0; else if (e.key === 'End') s = SLOT_TOV; else if (dir) {
        do { s += dir; } while (isGap(s));
        s = Math.max(0, Math.min(SLOT_TOV, s));
      } else return;
      e.preventDefault(); focusTip(s);
    });
  }

  // ------------------------------------------------------------------ boot

  function render() {
    P.cells.forEach(function (c) { idx[key({ down: c.down, ydstogo: c.ydstogo, yl: c.yardline_100, call: c.call, pers: c.personnel })] = c; });
    renderHero();
    buildConsole();
    renderHow();
  }

  N.getJSON('data/plays.json', true).then(function (d) {
    if (!d || !d.cells) { N.unavailable('The play outcome model', ['console-body']); return; }
    P = d;
    render();
  }).catch(function (err) {
    console.error(err);
    N.unavailable('The play outcome model', ['console-body']);
  });
})();
