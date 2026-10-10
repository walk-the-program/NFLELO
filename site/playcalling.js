/* NFLELO play calling page. Reads ./data/playcalling.json (written by scripts/export_ml_pages.py): the
   off-policy test of the early-down play-calling recommendations, the 60-situation grid by season and score state,
   and a descriptive 2025 team view. No dependencies; helpers come from common.js. */
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
  var teamSort = 'pass_rate';

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
    body.appendChild(N.caveatBox(PC.caveats, 'Read this first: what these numbers can and cannot tell you', 'big'));
  }

  function renderHero() {
    var H = PC.headline, o = H.ope_epa_per_play, p = H.placebo_epa_per_play, s1 = H.sensitivity['overlap_threshold_0.10'], s2 = H.sensitivity.weights_trimmed_at_10;
    setText('hero-line', 'Following the recommendations would have gained about ' + o.diff.toFixed(2) + ' expected points per early-down play on seasons the method never saw (95% interval ' + epa(o.lo) + ' to ' + epa(o.hi) + '). A placebo with shuffled actions shows no gain.');
    setText('hero-deck', 'On first and second down, which personnel group and run or pass gained the most, situation by situation, estimated from earlier seasons and tested on later ones. It is an observational estimate, so read the caveats first.');
    var lo = Math.min(s1.diff, s2.diff), hi = Math.max(s1.diff, s2.diff);
    N.heroTiles([
      { label: 'Estimated gain per play', big: epa(o.diff), sub: 'EPA per early-down play, 95% interval ' + epa(o.lo) + ' to ' + epa(o.hi) + ', ' + commas(H.plays) + ' plays' },
      { label: 'Placebo (shuffled actions)', big: epa(p.diff), sub: '95% interval ' + epa(p.lo) + ' to ' + epa(p.hi) + (p.excludes_zero ? ', excludes zero' : ', includes zero: no gain') },
      { label: 'Stress tests', big: epa(lo) + ' to ' + epa(hi), sub: 'with a stricter overlap rule and with weights trimmed; the gain holds' },
      { label: 'Where it applies', big: pct(H.share_of_plays_policy_applies, 0), sub: 'of early-down plays differ from the recommendation; ' + epa(H.epa_per_affected_play) + ' EPA per affected play' }
    ]);
    N.licenseUI(PC.license);
  }

  // ------------------------------------------------------------------ the test

  function buildOpe() {
    var body = done('ope-body'), H = PC.headline, o = H.ope_epa_per_play;
    clear(body);
    setText('ope-h', 'The gain holds up on unseen seasons, and a placebo shows nothing');
    var host = h('div', { class: 'chart ci' }), host2 = h('div', { class: 'chart ci' });
    var s1 = H.sensitivity['overlap_threshold_0.10'], s2 = H.sensitivity.weights_trimmed_at_10;
    var seasons = Object.keys(H.per_season);
    var weak = seasons.filter(function (k) { return !H.per_season[k].ope_epa.excludes_zero; });
    var plac = seasons.filter(function (k) { return H.per_season[k].placebo_epa.excludes_zero; });
    var seasonNote = 'In ' + (weak.length ? weak.join(' and ') + ' the gain\'s interval includes zero' : 'every season the gain\'s interval excludes zero') + (plac.length ? '; in ' + plac.join(' and ') + ' the placebo\'s interval excludes zero (a hint of leftover bias)' : '') + '. The pooled result is tighter than any single season.';
    body.appendChild(h('div', { class: 'split' },
      h('div', {},
        h('h3', { class: 'sub-h', style: 'margin-top:0', text: 'All of 2020 to 2025' }),
        well(host),
        h('p', { class: 'fnote', text: 'EPA per early-down play, gain over the teams\' actual mix of calls, off-policy estimate with a 95% interval resampled by game (' + commas(o.clusters) + ' games). The placebo shuffles actions within situations; it should show zero. Pass rule, fixed in advance: ' + H.rule + '.' }),
        h('dl', { class: 'statrow' },
          stat('Success rate gain', signed(H.ope_success_rate.diff * 100, 1) + ' pts', '95% interval ' + signed(H.ope_success_rate.lo * 100, 1) + ' to ' + signed(H.ope_success_rate.hi * 100, 1)),
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
      h('span', {}, h('i', { class: 'sc', style: 'border:1px dashed var(--rule-strong)' }), 'No clear best: the data cannot rank the actions'),
      h('span', { text: 'Deeper color means a larger estimated gain' })));
    body.appendChild(ui.grid);
    body.appendChild(ui.detail);
    body.appendChild(h('p', { class: 'fnote', text: 'Gain is expected points added per play against the teams\' current mix on the same plays. A situation is "clear best" only when the best action\'s 95% interval is above zero. "No clear best" is not evidence that the actions are equal. Recommendations for a season use earlier seasons only.' }));
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
        var cls, k = '';
        if (b.clear) { cls = /_pass$/.test(b.best) ? 'pass' : 'run'; k = '--k:' + (0.22 + 0.4 * Math.min(Math.max(b.gain, 0) / 0.2, 1)).toFixed(2); }
        else cls = 'none';
        var btn = h('button', { type: 'button', class: 'pcell k-' + cls, style: k, 'aria-pressed': on ? 'true' : 'false',
          'aria-label': d[1] + ', ' + z[1] + ': ' + (b.clear ? actName(b.best) + ', gain ' + epa(b.gain) + ' EPA per play' : 'no clear best') },
          b.clear ? [h('span', { class: 'pa', text: actName(b.best) }), h('span', { class: 'pg2' }, epa(b.gain), h('small', { text: 'EPA per play' }))]
                  : [h('span', { class: 'pa', text: 'No clear best' }), h('span', { class: 'pn', text: 'n ' + commas(b.n_train_plays) })]);
        btn.addEventListener('click', function () { selCell = { dd: d[0], zone: z[0] }; drawGrid(); });
        bindTip(btn, function () {
          var rows = b.clear ? [{ value: actName(b.best), label: actLong(b.best) }, { value: epa(b.gain), label: 'EPA per play, 95% interval [' + signed(b.gain_lo, 3) + ', ' + signed(b.gain_hi, 3) + ']' }] : [{ value: 'No clear best', label: 'the data cannot rank the actions here' }];
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
    ui.detail.appendChild(h('p', { text: (b.clear ? 'Clear best: ' + actLong(b.best) + ', a gain of ' + epa(b.gain) + ' EPA per play (95% interval ' + signed(b.gain_lo, 3) + ' to ' + signed(b.gain_hi, 3) + ').' : 'No clear best: the leading action\'s interval includes zero, so the data cannot rank the actions here.') + ' Estimated from ' + commas(b.n_train_plays) + ' plays in earlier seasons; ' + b.candidates + ' candidate actions.' }));
    ui.detail.appendChild(h('div', { class: 'tbl-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, [['Action', ''], ['Gain (EPA per play)', 'r'], ['95% interval', 'r'], ['Share of plays that called it', 'r']].map(function (t) { return h('th', { class: t[1], scope: 'col', text: t[0] }); }))),
      h('tbody', {}, b.actions.map(function (a) {
        return h('tr', {}, h('td', { title: actLong(a[0]) }, h('i', { class: 'sc ' + (/_pass$/.test(a[0]) ? 'pass' : 'run') }), actName(a[0])), h('td', { class: 'r n', text: epa(a[1]) }), h('td', { class: 'r n', text: '[' + signed(a[2], 3) + ', ' + signed(a[3], 3) + ']' }), h('td', { class: 'r n', text: pct(a[4], 0) }));
      })))));
  }

  // ------------------------------------------------------------------ team view

  function buildTeams() {
    var body = done('teams-body'), T = PC.team_view_2025;
    clear(body);
    setText('teams-h', 'Passing in the pass-recommended situations: ' + T.teams[0].team + ' led at ' + pct(T.teams[0].pass_rate, 0) + ', ' + T.teams[T.teams.length - 1].team + ' trailed at ' + pct(T.teams[T.teams.length - 1].pass_rate, 0));
    ui.tlist = h('div', { class: 'tlist' });
    var sortSeg = N.seg('Sort teams', [{ value: 'pass_rate', label: 'Pass rate' }, { value: 'name', label: 'Team A to Z' }], teamSort, function (v) { teamSort = v; drawTeams(); });
    var cellsTv = N.tableView('View the ' + T.pass_cells.length + ' situations where the 2025 recommendation is a pass', [{ t: 'Situation' }, { t: 'Recommended action' }], function () { return T.pass_cells.map(function (c) { return [c.cell, actLong(c.recommended)]; }); });
    body.appendChild(h('div', {},
      h('p', { class: 'fnote', style: 'margin-top:0' }, h('span', { class: 'me', text: 'descriptive' }), ' Observed ' + T.season + ' play calls. Pass rate is each team\'s share of passes in the ' + T.pass_cells.length + ' situations where the 2025 recommendation is a pass (league ' + pct(T.league.pass_rate, 0) + ' over ' + commas(T.league.plays) + ' plays). The gap to the league does not adjust for each team\'s mix of situations, and it is not a model evaluation.'),
      h('div', { style: 'margin-top:16px;width:fit-content;max-width:100%' }, sortSeg.el),
      h('div', { style: 'margin-top:22px' }, ui.tlist),
      cellsTv));
    drawTeams();
  }

  function drawTeams() {
    var T = PC.team_view_2025, lg = T.league, list = T.teams.slice();
    if (teamSort === 'name') list.sort(function (a, b) { return a.team.localeCompare(b.team); }); else list.sort(function (a, b) { return b.pass_rate - a.pass_rate; });
    var lo = Math.floor(Math.min.apply(null, T.teams.map(function (t) { return t.pass_rate; })) * 20) / 20, hi = Math.ceil(Math.max.apply(null, T.teams.map(function (t) { return t.pass_rate; })) * 20) / 20;
    function at(v) { return ((v - lo) / (hi - lo) * 100).toFixed(2) + '%'; }
    clear(ui.tlist);
    ui.tlist.appendChild(h('div', { class: 'tvrow tvhead' }, h('span', { text: 'Team' }),
      h('span', { class: 'axisrow', text: pct(lo, 0) + ' pass rate to ' + pct(hi, 0) + '; the tick is the league (' + pct(lg.pass_rate, 0) + ')' }),
      h('span', { class: 'r', text: 'Pass rate' }), h('span', { class: 'r', text: 'vs league' }), h('span', { class: 'r em', text: 'Exact match' })));
    list.forEach(function (t) {
      var row = h('div', { class: 'tvrow', tabindex: 0, 'aria-label': N.teamName(t.team) + ', pass rate ' + pct(t.pass_rate, 1) + ', ' + signed(t.vs_league * 100, 1) + ' points against the league' },
        h('span', {}, h('span', { class: 'abbr', text: t.team }), h('span', { class: 'nm', text: N.teamName(t.team).replace(/^.* /, '') })),
        h('span', { class: 'dp' }, h('b', { style: 'left:' + at(lg.pass_rate) }), h('i', { style: 'left:' + at(t.pass_rate) })),
        h('span', { class: 'r num', text: pct(t.pass_rate, 0) }), h('span', { class: 'r num', text: signed(t.vs_league * 100, 1) + ' pts' }), h('span', { class: 'r em num', text: pct(t.exact_match_rate, 0) }));
      bindTip(row, function () {
        return { title: N.teamName(t.team) + ', 2025 (descriptive)', rows: [
          { value: pct(t.pass_rate, 1), label: 'pass rate in the pass-recommended situations (' + t.plays + ' plays)', cls: 's3' },
          { value: signed(t.vs_league * 100, 1) + ' pts', label: 'against the league, not adjusted for situation mix' },
          { value: pct(t.exact_match_rate, 1), label: 'used the exact recommended personnel and call' },
          { value: pct(t.early_down_pass_rate_all_cells, 1), label: 'pass rate on all early downs' }] };
      });
      ui.tlist.appendChild(row);
    });
  }

  // ------------------------------------------------------------------ method

  function buildHow() {
    var dyn = $('#how-dyn'), D = PC.definitions;
    dyn.appendChild(h('div', {},
      h('h3', { class: 'sub-h', text: 'The sample' }),
      h('p', { class: 'stat-note', text: D.sample + '. Each of the ' + D.cells.count + ' situations is a down-and-distance group, a field zone and a score state, with twelve candidate actions (a personnel group plus run or pass).' })));
    dyn.appendChild(h('dl', { class: 'gloss' }, h('h3', { text: 'What each term means' }),
      ['gain', 'best', 'clear', 'candidate', 'by_season'].map(function (k) { return h('div', {}, h('dt', { text: k.replace('_', ' ') }), h('dd', { text: D[k] })); })));
    dyn.appendChild(h('p', { class: 'fnote', text: 'The caveats are at the top of this page. The advice in these tables is a statement about the data, not a game plan: defenses would adapt to a team that always followed it.' }));
    $('#how-rcpt').appendChild(N.receiptLinks(PC.headline.receipt));
    setText('how-credit', PC.license.credit);
  }

  // ------------------------------------------------------------------ boot

  var ALL = ['caveats-body', 'ope-body', 'cells-body', 'teams-body'];
  N.loadTeamNames().then(function () { return N.getJSON('data/playcalling.json', true); }).then(function (d) {
    if (!d || !d.cells) { N.unavailable('The play-calling data', ALL); return; }
    PC = d;
    season = String(PC.seasons[PC.seasons.length - 1]);
    renderHero(); renderCaveats(); buildOpe(); buildCells(); buildTeams(); buildHow();
  }).catch(function (err) { console.error(err); N.unavailable('The play-calling data', ALL); });
})();
