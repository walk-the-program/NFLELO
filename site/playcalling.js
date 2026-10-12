/* NFLELO play calling page. Reads ./data/playcalling.json (written by scripts/export_ml_pages.py): the
   off-policy test of the early-down play-calling recommendations, the 60-situation grid by season and score state.
   No dependencies; helpers come from common.js. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, tipHide = N.tipHide, bindTip = N.bindTip, watch = N.watch;
  var signed = N.signed, commas = N.commas, pct = N.pct, ciText = N.ciText;
  var setText = N.setText, done = N.done;

  var PC = null;
  var ui = {};
  var score = 'within 3', season = '2025', selCell = { dd: '1-10+', zone: 'own 21-50' };

  var DD = [['1-10+', '1st and 10+', '10 or more to go'], ['1-short', '1st, short', 'under 10 to go'], ['2-long', '2nd and long', '8 or more to go'], ['2-mid', '2nd and mid', '4 to 7 to go'], ['2-short', '2nd and short', '1 to 3 to go']];
  var ZONES = [['own 1-20', 'Own 1-20'], ['own 21-50', 'Own 21-50'], ['opp 49-21', 'Opp 49-21'], ['red zone', 'Red zone']];
  var SCORES = [['trail 4+', 'Trailing by 4+'], ['within 3', 'Within 3'], ['lead 4+', 'Leading by 4+']];

  function actName(a) { var p = a.split('_'); return (p[0] === 'oth' ? 'Other' : p[0]) + ' ' + p[1]; }
  function actLong(a) { var p = a.split('_'), g = PC.definitions.actions.groups[p[0]] || p[0]; return g + ', ' + (PC.definitions.actions.calls[p[1]] || p[1]); }
  function cellOf(dd, zone, sc) { return PC.cells.filter(function (c) { return c.down_distance === dd && c.zone === zone && c.score === sc; })[0]; }
  function epa(x) { return signed(x, 3); }

  // ------------------------------------------------------------------ caveats, hero

  function renderCaveats() {
    var body = done('caveats-body');
    clear(body);
    body.appendChild(N.caveatBox(['Observational estimate. Defenses would adapt.'], 'Caveat', 'big'));
  }

  function renderHero() {
    var H = PC.headline, o = H.ope_epa_per_play, p = H.placebo_epa_per_play, s1 = H.sensitivity['overlap_threshold_0.10'], s2 = H.sensitivity.weights_trimmed_at_10;
    setText('hero-line', 'Following the recommendations gains about ' + o.diff.toFixed(2) + ' points per early-down play on unseen seasons (95% CI ' + epa(o.lo) + ' to ' + epa(o.hi) + ').');
    setText('hero-deck', 'Best run or pass and personnel group, by situation.');
    var lo = Math.min(s1.diff, s2.diff), hi = Math.max(s1.diff, s2.diff);
    N.heroTiles([
      { label: 'Estimated gain per play', big: epa(o.diff), sub: 'EPA per play, 95% CI ' + epa(o.lo) + ' to ' + epa(o.hi) },
      { label: 'Placebo (shuffled actions)', big: epa(p.diff), sub: '95% CI ' + epa(p.lo) + ' to ' + epa(p.hi) + (p.excludes_zero ? '' : ', includes zero') },
      { label: 'Stress tests', big: epa(lo) + ' to ' + epa(hi), sub: 'two stricter checks; the gain holds' },
      { label: 'Where it applies', big: pct(H.share_of_plays_policy_applies, 0), sub: 'of plays change; ' + epa(H.epa_per_affected_play) + ' EPA each' }
    ]);
    N.licenseUI(PC.license);
  }

  // ------------------------------------------------------------------ the test

  function buildOpe() {
    var body = done('ope-body'), H = PC.headline, o = H.ope_epa_per_play;
    clear(body);
    setText('ope-h', 'The gain holds; the placebo shows none');
    var host = h('div', { class: 'chart ci' }), host2 = h('div', { class: 'chart ci' });
    var s1 = H.sensitivity['overlap_threshold_0.10'], s2 = H.sensitivity.weights_trimmed_at_10;
    var seasons = Object.keys(H.per_season);
    var weak = seasons.filter(function (k) { return !H.per_season[k].ope_epa.excludes_zero; });
    var plac = seasons.filter(function (k) { return H.per_season[k].placebo_epa.excludes_zero; });
    var seasonNote = [weak.length ? 'Gain interval includes zero in ' + weak.join(' and ') : 'Gain interval excludes zero every season', plac.length ? 'placebo interval excludes zero in ' + plac.join(' and ') : ''].filter(Boolean).join('; ') + '.';
    body.appendChild(h('div', { class: 'split' },
      h('div', {},
        h('h3', { class: 'sub-h', style: 'margin-top:0', text: 'All of 2020 to 2025' }),
        well(host),
        h('p', { class: 'fnote', text: 'EPA per early-down play over the actual mix of calls. 95% intervals, ' + commas(o.clusters) + ' games.' }),
        h('dl', { class: 'statrow' },
          stat('Success rate gain', signed(H.ope_success_rate.diff * 100, 1) + ' pp', '95% interval ' + signed(H.ope_success_rate.lo * 100, 1) + ' to ' + signed(H.ope_success_rate.hi * 100, 1)),
          stat('Plays the policy changes', pct(H.share_of_plays_policy_applies, 0), epa(H.epa_per_affected_play) + ' EPA per affected play'),
          stat('Plays tested', commas(H.plays), 'early downs, ' + PC.seasons[0] + ' to ' + PC.seasons[PC.seasons.length - 1]))),
      h('div', {},
        h('h3', { class: 'sub-h', style: 'margin-top:0', text: 'Season by season' }),
        well(host2),
        h('p', { class: 'fnote', text: seasonNote }),
        N.tableView('View the seasons as a table', [{ t: 'Season' }, { t: 'Plays', r: 1, n: 1 }, { t: 'Gain', r: 1, n: 1 }, { t: 'Gain 95% interval', r: 1, n: 1 }, { t: 'Placebo', r: 1, n: 1 }, { t: 'Placebo 95% interval', r: 1, n: 1 }, { t: 'Share with a clear best', r: 1, n: 1 }], function () {
          return seasons.map(function (k) { var s = H.per_season[k]; return [k, commas(s.plays), epa(s.ope_epa.diff), ciText(s.ope_epa, 3), epa(s.placebo_epa.diff), ciText(s.placebo_epa, 3), pct(s.clear_share, 0)]; });
        }))));
    watch(host, function () {
      N.ciChart(host, [
        { label: 'Gain from the recommendations', ci: o, series: 's3' },
        { label: 'Placebo, shuffled actions', ci: H.placebo_epa_per_play, series: 'sm' },
        { label: 'Stress test: stricter overlap rule', ci: s1, series: 's3' },
        { label: 'Stress test: weights trimmed', ci: s2, series: 's3' }
      ], { decimals: 3, axis: 'EPA per early-down play', measure: 'EPA per play', title: 'Gain and checks' });
    });
    watch(host2, function () {
      N.ciChart(host2, seasons.map(function (k) { return { label: k, ci: H.per_season[k].ope_epa, series: 's3' }; }), { decimals: 3, axis: 'EPA per early-down play, by season', measure: 'EPA per play', title: 'Gain by season' });
    });
  }
  function stat(label, value, small) { return h('div', {}, h('dt', { text: label }), h('dd', { class: 'num' }, value, h('small', { text: small }))); }

  // ------------------------------------------------------------------ situation grid

  function buildCells() {
    var body = done('cells-body');
    clear(body);
    ui.score = N.seg('Score state', SCORES.map(function (s) { return { value: s[0], label: s[1] }; }), score, function (v) { score = v; drawGrid(); });
    ui.season = N.seg('Season', PC.seasons.map(function (s) { return { value: String(s), label: String(s) }; }), season, function (v) { season = v; drawGrid(); });
    ui.grid = h('div', { class: 'pgrid', role: 'group', 'aria-label': 'Best action in each situation' });
    ui.detail = h('div', { class: 'pdetail', 'aria-live': 'polite' });
    ui.tv = N.tableViewLive('View all 60 situations for this season as a table',
      function () { return [{ t: 'Situation' }, { t: 'Best action' }, { t: 'Result' }, { t: 'Gain (EPA)', r: 1, n: 1 }, { t: '95% interval', r: 1, n: 1 }, { t: 'Training plays', r: 1, n: 1 }]; },
      function () {
        return PC.cells.map(function (c) {
          var b = c.by_season[season];
          return [c.cell, b.best ? actName(b.best) : 'none', b.label, b.gain == null ? 'n/a' : epa(b.gain), b.gain == null ? 'n/a' : '[' + signed(b.gain_lo, 3) + ', ' + signed(b.gain_hi, 3) + ']', commas(b.n_train_plays)];
        });
      });
    body.appendChild(h('div', { class: 'pc-controls' },
      h('div', { class: 'ctl' }, h('div', { class: 'ctl-h' }, h('span', { class: 'label', text: 'Score state' })), ui.score.el),
      h('div', { class: 'ctl' }, h('div', { class: 'ctl-h' }, h('span', { class: 'label', text: 'Season' })), ui.season.el)));
    body.appendChild(h('div', { class: 'legend', style: 'margin-top:20px;margin-bottom:0' },
      h('span', {}, h('i', { class: 'sc pass' }), 'Clear best is a pass'),
      h('span', {}, h('i', { class: 'sc run' }), 'Clear best is a run'),
      h('span', {}, h('i', { class: 'sc none' }), 'No clear best'),
      h('span', {}, h('i', { class: 'sc pass t0' }), h('i', { class: 'sc pass t1' }), h('i', { class: 'sc pass' }), 'Gain (EPA per play): pale under 0.07, outlined to 0.14, solid above')));
    body.appendChild(ui.grid);
    body.appendChild(ui.detail);
    body.appendChild(ui.tv.el);
    drawGrid();
  }

  function drawGrid() {
    ui.score.set(score); ui.season.set(season);
    var g = ui.grid;
    clear(g);
    var clearN = 0, passN = 0;
    PC.cells.forEach(function (c) { var b = c.by_season[season]; if (b.clear) { clearN++; if (/_pass$/.test(b.best)) passN++; } });
    var bs = PC.cells.filter(function (c) { return c.score === score; }).filter(function (c) { return c.by_season[season].clear; }).length;
    setText('cells-h', season + ': a clear best action in ' + clearN + ' of ' + PC.cells.length + ' situations, ' + passN + ' of them passes');
    g.appendChild(h('span'));
    ZONES.forEach(function (z) { g.appendChild(h('span', { class: 'gh', text: z[1] })); });
    DD.forEach(function (d) {
      g.appendChild(h('span', { class: 'gl' }, h('span', {}, d[1], h('small', { text: d[2] }))));
      ZONES.forEach(function (z) {
        var c = cellOf(d[0], z[0], score), b = c.by_season[season];
        var on = selCell.dd === d[0] && selCell.zone === z[0];
        var cls, tier = '';
        if (b.clear) { cls = /_pass$/.test(b.best) ? 'pass' : 'run'; tier = ' t' + (b.gain < 0.07 ? 0 : b.gain < 0.14 ? 1 : 2); }   // gain in three discrete steps
        else cls = 'none';
        var btn = h('button', { type: 'button', class: 'pcell k-' + cls + tier, 'aria-pressed': on ? 'true' : 'false',
          'aria-label': d[1] + ', ' + z[1] + ': ' + (b.clear ? actName(b.best) + ', gain ' + epa(b.gain) + ' EPA per play' : 'no clear best') },
          b.clear ? [h('span', { class: 'pa', text: actName(b.best) }), h('span', { class: 'pg2' }, epa(b.gain), h('small', { text: 'EPA per play' }))]
                  : [h('span', { class: 'pa', text: 'No clear best' }), h('span', { class: 'pn', text: 'n ' + commas(b.n_train_plays) })]);
        btn.addEventListener('click', function () { selCell = { dd: d[0], zone: z[0] }; drawGrid(); });
        bindTip(btn, function () {
          var rows = b.clear ? [{ value: actName(b.best), label: actLong(b.best) }, { value: epa(b.gain), label: 'EPA per play, 95% interval [' + signed(b.gain_lo, 3) + ', ' + signed(b.gain_hi, 3) + ']' }] : [{ value: 'No clear best', label: 'the actions cannot be ranked' }];
          rows.push({ value: commas(b.n_train_plays), label: 'training plays, ' + (PC.seasons[0] - 4) + ' to ' + (Number(season) - 1) });
          return { title: d[1] + ', ' + z[1] + ', ' + score + ', ' + season, rows: rows };
        });
        g.appendChild(btn);
      });
    });
    drawDetail();
    ui.tv.refresh();
  }

  function drawDetail() {
    var c = cellOf(selCell.dd, selCell.zone, score), b = c.by_season[season], dd = DD.filter(function (d) { return d[0] === selCell.dd; })[0], z = ZONES.filter(function (x) { return x[0] === selCell.zone; })[0];
    var sc = SCORES.filter(function (s) { return s[0] === score; })[0][1];
    clear(ui.detail);
    ui.detail.appendChild(h('h3', { text: dd[1] + ', ' + z[1] + ', ' + sc.toLowerCase() + ', ' + season }));
    ui.detail.appendChild(h('p', { text: (b.clear ? 'Clear best: ' + actLong(b.best) + ', ' + epa(b.gain) + ' EPA per play (95% CI ' + signed(b.gain_lo, 3) + ' to ' + signed(b.gain_hi, 3) + ').' : 'No clear best: the top interval includes zero.') + ' ' + commas(b.n_train_plays) + ' training plays, ' + b.candidates + ' actions.' }));
    ui.detail.appendChild(h('div', { class: 'tbl-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, [['Action', ''], ['Gain (EPA per play)', 'r'], ['95% interval', 'r'], ['Share of plays that called it', 'r']].map(function (t) { return h('th', { class: t[1], scope: 'col', text: t[0] }); }))),
      h('tbody', {}, b.actions.map(function (a) {
        return h('tr', {}, h('td', { title: actLong(a[0]) }, h('i', { class: 'sc ' + (/_pass$/.test(a[0]) ? 'pass' : 'run') }), actName(a[0])), h('td', { class: 'r n', text: epa(a[1]) }), h('td', { class: 'r n', text: '[' + signed(a[2], 3) + ', ' + signed(a[3], 3) + ']' }), h('td', { class: 'r n', text: pct(a[4], 0) }));
      })))));
  }

  // ------------------------------------------------------------------ method

  // ------------------------------------------------------------------ boot

  var ALL = ['caveats-body', 'ope-body', 'cells-body'];
  N.loadTeamNames().then(function () { return N.getJSON('data/playcalling.json', true); }).then(function (d) {
    if (!d || !d.cells) { N.unavailable('The play-calling data', ALL); return; }
    PC = d;
    season = String(PC.seasons[PC.seasons.length - 1]);
    renderHero(); renderCaveats(); buildOpe(); buildCells();
  }).catch(function (err) { console.error(err); N.unavailable('The play-calling data', ALL); });
})();
