/* NFELO Elo page: team explorer, luck board, league parity, the Elo scorecard (Brier by season, home-field edge,
   live record) and the method. Moved out of the home page; formatting, the line chart and the data loader are shared
   with it through elo-lib.js. No dependencies. */
(function () {
  'use strict';

  var FILES = ['meta', 'ladder', 'history', 'luck', 'parity', 'scorecard', 'headlines'];

  var N = window.NFL, R = N.R;
  var $ = N.$, h = N.h, clear = N.clear, well = N.well;
  var bindTip = N.bindTip;
  var watch = N.watch, tableView = N.tableView, legend = N.legend, specList = N.specList;
  var D = R.D, byTeam = R.byTeam, histBy = R.histBy;
  var f1 = R.f1, signed = R.signed, pct = R.pct, rec = R.rec, winPct = R.winPct, plural = R.plural, dayLong = R.dayLong;
  var lineChart = R.lineChart, ticksFor = R.ticksFor, setHeadline = R.setHeadline, REDUCED = R.REDUCED;

  var pick = null;     // setter for the explorer

  // ------------------------------------------------------------------ hero: the headline scorecard numbers

  function renderHero() {
    var sc = D.meta.scorecard, meta = D.meta;
    var span = sc.test_seasons[0] + ' to ' + sc.test_seasons[1];
    function duo(a, b, al, bl) {
      return [h('span', { class: 'duo' }, a, h('small', { text: al })), h('span', { class: 'slash', text: '/' }), h('span', { class: 'duo' }, b, h('small', { text: bl }))];
    }
    function tile(label, big, sub, dual) {
      return h('div', { class: 'tile' }, h('h3', { text: label }), h('div', { class: 'big' + (dual ? ' dual' : '') }, big), h('p', { class: 'sub', text: sub }));
    }
    $('#hero-line').textContent = 'Elo trails the Vegas benchmark by ' + sc.brier_gap_vs_market.toFixed(3) + ' Brier on ' + span + ' games.';
    var tiles = $('#tiles');
    clear(tiles);
    tiles.append(
      tile('Brier score, lower is better', duo(sc.elo_brier.toFixed(4), sc.market_brier.toFixed(4), 'Elo', 'Vegas'), span + ', ' + sc.n_market.toLocaleString('en-US') + ' games.', true),
      tile('Correct picks', duo(pct(sc.elo_accuracy_market_games, 1), pct(sc.market_accuracy, 1), 'Elo', 'Vegas'), 'Vegas favorite vs Elo favorite.', true),
      tile('Gap to the market', ['+' + sc.brier_gap_vs_market.toFixed(3)], 'Brier, plus or minus ' + sc.brier_gap_se.toFixed(3) + '.'),
      tile('Home-field edge now', [f1(meta.hfa_pts), h('small', { text: 'Elo pts' })], pct(meta.hfa_win_pct, 1) + ' home win chance, equal teams.'));
    $('#credit').textContent = meta.credit.replace(/(\d)[\u2013\u2014](\d)/g, '$1 to $2');   // house style: no dashes in copy
  }

  // ------------------------------------------------------------------ method

  // Elo settings come from meta.json (so they follow the config); the evaluation numbers are from context/data-and-elo.md
  // (test 2010 to 2025, regular season), apart from those already in meta.scorecard.
  var LOG_LOSS = 0.6317;

  function renderMethod() {
    var body = $('#method-body'), cfg = D.meta.config, sc = D.meta.scorecard;
    clear(body);
    body.removeAttribute('aria-busy');
    body.appendChild(h('h3', { text: 'Elo model' }));
    body.appendChild(specList([
      ['K factor', String(cfg.k)],
      ['Season regression', 'lambda ' + cfg.lam.toFixed(2) + ' toward the mean each offseason'],
      ['Home-field edge', 'Learned online: starts at ' + cfg.hfa_init + ' pts, update rate ' + cfg.k_hfa],
      ['Expansion teams', 'Start at ' + cfg.expansion_start],
      ['Margin of victory', '538-style multiplier'],
      ['Games rated', 'Regular season only; no playoff updates']
    ]));
    body.appendChild(h('h3', { text: 'Data' }));
    body.appendChild(specList([
      ['1999 to now', 'nflverse schedule and results (CC BY 4.0)'],
      ['1970 to 1998', 'FiveThirtyEight NFL Elo game file (CC BY 4.0)'],
      ['Benchmark', 'Vegas moneylines, vig removed (nflverse schedule file)']
    ]));
    body.appendChild(h('h3', { text: 'Held-out result' }));
    body.appendChild(specList([
      ['Tuning window', sc.tune_seasons[0] + ' to ' + sc.tune_seasons[1] + ' regular season'],
      ['Test window', sc.test_seasons[0] + ' to ' + sc.test_seasons[1] + ' regular season, ' + sc.n_all.toLocaleString('en-US') + ' games'],
      ['Brier / log loss', sc.elo_brier.toFixed(4) + ' / ' + LOG_LOSS.toFixed(4)],
      ['Pick accuracy', pct(sc.elo_accuracy, 1)],
      ['Legacy model', 'Brier ' + sc.legacy_brier.toFixed(4) + ' (fixed home edge)']
    ]));
  }

  // Live record block for the scorecard: empty-state copy until the first live week is scored.
  function liveBlock() {
    var ml = D.ml, L = ml.live.final, W = ml.live.wednesday;
    var box = h('div', { class: 'livebox' }, h('h3', { text: 'Live ' + ml.season + ' record' }));
    var start = 'Live record from Week ' + ml.live_from_week + (ml.live_from_date ? ' (' + dayLong(ml.live_from_date) + ')' : '') + '. Predictions are logged before kickoff.';
    if (!L.n) {
      box.appendChild(h('p', { class: 'sub', text: start + ' No games scored yet.' }));
    } else {
      var mk = L.market_games;
      box.appendChild(h('p', { class: 'sub', text: start + ' ' + plural(L.n, 'game', 'games') + ' scored so far, a small sample.' }));
      box.appendChild(h('div', { class: 'score' },
        h('div', {}, h('h3', { text: 'Brier score, lower is better' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: (mk ? mk.model.brier : L.model.brier).toFixed(4) }), 'NFELO model'),
            h('span', {}, h('b', { class: 'num', text: (mk ? mk.elo.brier : L.elo.brier).toFixed(4) }), 'Elo'),
            mk ? h('span', {}, h('b', { class: 'num', text: mk.market.brier.toFixed(4) }), 'Vegas market') : null)),
        h('div', {}, h('h3', { text: 'Correct picks' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: pct(L.model.accuracy, 1) }), 'NFELO model'),
            h('span', {}, h('b', { class: 'num', text: pct(L.elo.accuracy, 1) }), 'Elo'),
            mk ? h('span', {}, h('b', { class: 'num', text: pct(mk.market.accuracy, 1) }), 'Vegas') : null))));
      var notes = [];
      if (mk && mk.n !== L.n) notes.push('Brier uses the ' + mk.n + ' games with a market line; picks use all ' + L.n + '.');
      if (W.n) notes.push('Wednesday predictions alone: NFELO model ' + W.model.brier.toFixed(4) + ', Elo ' + W.elo.brier.toFixed(4) + ' Brier on ' + plural(W.n, 'game', 'games') + '.');
      if (notes.length) box.appendChild(h('p', { class: 'fnote', text: notes.join(' ') }));
      // Shadow model: logged before kickoff in its own ledger, scored on the same games; hidden until one is scored.
      var S = ml.shadow;
      if (S && S.n) box.appendChild(h('p', { class: 'fnote' }, 'Shadow model Brier ' + S.shadow.brier.toFixed(4) + ', NFELO model ' + S.model.brier.toFixed(4) + ', on the same ' + plural(S.n, 'game', 'games') + '.'));
    }
    return box;
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
      var bestIdx = lastIndexOfSeason(t, t.best.season), worstIdx = lastIndexOfSeason(t, t.worst.season);
      var left = h('div', {}), split = h('div', { class: 'split' }, left);
      left.appendChild(well(host));
      body.appendChild(split);
      setHeadline('explorer-h', D.headlines.explorer[code]);
      watch(host, function () {
        lineChart(host, {
          series: [{ id: code, label: t.name, cls: 's1', pts: pts }],
          x: [1970, xMax], y: [yLo, yHi],
          height: function (w) { return w < 560 ? 280 : w > 900 ? 460 : 380; },
          xTicks: YEAR_TICKS, yTicks: ticksFor(yLo, yHi, 6),
          xFmt: String, yFmt: String,
          ref: { y: 1500, label: '1500 average' },
          aria: t.name + ' Elo rating by game week since ' + t.first_season + '. Best season ' + t.best.season + ' at ' + f1(t.best.rating) + ', worst ' + t.worst.season + ' at ' + f1(t.worst.rating) + '. Focus the chart and use the arrow keys to read values.',
          markers: [
            { x: pts[bestIdx].x, y: pts[bestIdx].y, cls: 's1', text: 'Best ' + t.best.season, dy: -12 },
            { x: pts[worstIdx].x, y: pts[worstIdx].y, cls: 's1', text: 'Worst ' + t.worst.season, dy: 20 }
          ],
          tip: function (x, rows) {
            var p = rows[0].p;
            return { title: p.s + ' Week ' + p.w, rows: [{ value: f1(p.y), label: signed(p.y - 1500) + ' vs 1500', cls: 's1' }] };
          }
        });
      });

      var cur = L;
      split.appendChild(h('dl', { class: 'facts' },
        fact('Rating now', f1(cur.rating), 'Rank ' + cur.rank + ' of 32'),
        fact('Best season', String(t.best.season), f1(t.best.rating) + ', ' + rec(t.best.w, t.best.l, t.best.t)),
        fact('Worst season', String(t.worst.season), f1(t.worst.rating) + ', ' + rec(t.worst.w, t.worst.l, t.worst.t)),
        fact('All-time record', rec(t.all_time.w, t.all_time.l, t.all_time.t), winPct(t.all_time.w, t.all_time.l, t.all_time.t) + ' of ' + t.all_time.games.toLocaleString('en-US') + ' games')));
      var notes = 'Line at 1500, the average. Regular season; ' + (D.meta.season_partial ? D.meta.season + ' excluded.' : 'all seasons.');
      if (t.first_season > 1970) notes += ' ' + t.name + ' joined in ' + t.first_season + ' at ' + D.meta.config.expansion_start + '.';
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
        toggle.appendChild(h('button', { type: 'button', 'aria-pressed': k === active ? 'true' : 'false', text: labelFor(k),
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
          h('span', { class: 'bar' }, h('span', { class: 'fill ' + (r.luck >= 0 ? 'pos' : 'neg'), style: (r.luck >= 0 ? 'left:50%' : 'left:' + (50 - w) + '%') + ';width:' + w + '%' })),
          h('span', { class: 'val', text: signed(r.luck) }));
        bindTip(item, function () {
          return { title: t.name + ', ' + plural(r.games, 'game', 'games'), rows: [
            { value: signed(r.luck), label: 'wins vs expected' }, { value: String(r.actual), label: 'actual wins' }, { value: f1(r.expected), label: 'expected wins' }] };
        });
        list.appendChild(item);
      });
      setHeadline('luck-h', D.headlines.luck[active]);
      var left = h('div', {}, h('div', { class: 'lk' }, head, list), h('p', { class: 'fnote', text: 'Both seasons share one scale.' }));
      body.appendChild(h('div', { class: 'split' }, left, luckyBox(s)));
      left.appendChild(tableView('View luck as a table', [{ t: 'Team' }, { t: 'Games', r: 1, n: 1 }, { t: 'Actual wins', r: 1, n: 1 }, { t: 'Expected wins', r: 1, n: 1 }, { t: 'Luck', r: 1, n: 1 }], function () {
        return s.teams.map(function (r) { return [r.team, String(r.games), String(r.actual), f1(r.expected), signed(r.luck)]; });
      }));
    }

    drawToggle();
    drawBars();
  }

  // ------------------------------------------------------------------ history: league parity

  function renderHistory() {
    var body = $('#history-body'), T = D.parity;
    clear(body);
    body.removeAttribute('aria-busy');

    // Parity: spread of end-of-season ratings. Lower means a tighter league.
    var pHost = h('div', { class: 'chart' });
    body.appendChild(well(pHost));
    var pts = T.seasons.map(function (s, i) { return { x: s, y: T.parity[i], count: T.teams_count[i] }; }).filter(function (p) { return p.x !== T.partial_season; });
    var pLo = 60, pHi = 150;
    watch(pHost, function () {
      lineChart(pHost, {
        series: [{ id: 'parity', label: 'Rating spread', cls: 's1', pts: pts }],
        x: [pts[0].x, pts[pts.length - 1].x], y: [pLo, pHi], yLabel: 'Std. dev. of team ratings (Elo points)',
        height: function (w) { return w < 560 ? 240 : 300; },
        xTicks: YEAR_TICKS.concat([2025]), yTicks: ticksFor(pLo, pHi, 4), xFmt: String, yFmt: String,
        aria: 'Line chart of the standard deviation of end-of-season Elo ratings by season, from ' + f1(pts[0].y) + ' in ' + pts[0].x + ' to ' + f1(pts[pts.length - 1].y) + ' in ' + pts[pts.length - 1].x + '. Lower means a more even league.',
        tip: function (x, rows) { var p = rows[0].p; return { title: String(x), rows: [{ value: f1(p.y), label: 'std. dev. (Elo points)', cls: 's1' }, { value: String(p.count), label: 'teams' }] }; }
      });
    });
    body.appendChild(h('p', { class: 'fnote', text: 'Lower is more even. Expansion years read high: new teams start at ' + D.meta.config.expansion_start + '.' }));
    body.appendChild(tableView('View parity as a table', [{ t: 'Season' }, { t: 'Std. dev. (Elo points)', r: 1, n: 1 }, { t: 'Teams', r: 1, n: 1 }], function () {
      return pts.slice().reverse().map(function (p) { return [String(p.x), f1(p.y), String(p.count)]; });
    }));
  }

  // ------------------------------------------------------------------ scorecard

  function renderScorecard() {
    var body = $('#scorecard-body'), S = D.scorecard.seasons;
    clear(body);
    body.removeAttribute('aria-busy');

    if (D.ml) body.appendChild(liveBlock());

    var pair = h('div', { class: 'pair' });
    var brHost = h('div', { class: 'chart' }), hfaHost = h('div', { class: 'chart' });
    pair.appendChild(h('div', {}, h('h3', { text: 'Brier score by season' }), h('p', { class: 'sub', text: 'Lower is better.' }),
      legend([['s1', 'Elo'], ['s2', 'Vegas market']]), well(brHost),
      h('p', { class: 'fnote', text: 'No market lines for 2008; 2006 is partial.' }),
      tableView('View Brier scores as a table', [{ t: 'Season' }, { t: 'Games', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }, { t: 'Market', r: 1, n: 1 }], function () {
        return S.filter(function (r) { return r.season >= firstMkt; }).reverse().map(function (r) {
          return r.market_brier == null ? [String(r.season), '-', '-', '-'] : [String(r.season), String(r.n_market), r.elo_brier.toFixed(4), r.market_brier.toFixed(4)];
        });
      })));
    pair.appendChild(h('div', {}, h('h3', { text: 'Home-field edge by season' }), h('p', { class: 'sub', text: 'Home win chance between equal teams vs actual home win rate.' }),
      legend([['s1', 'Elo learned edge'], ['sm dot', 'Actual home win rate']]), well(hfaHost),
      tableView('View home-field edge as a table', [{ t: 'Season' }, { t: 'Elo edge (pts)', r: 1, n: 1 }, { t: 'Implied win %', r: 1, n: 1 }, { t: 'Actual win %', r: 1, n: 1 }], function () {
        return S.slice().reverse().map(function (r) { return [String(r.season), f1(r.hfa_pts), pct(r.hfa_win_pct, 1), pct(r.home_win_rate, 1)]; });
      })));
    body.appendChild(pair);

    var firstMkt = S.filter(function (r) { return r.market_brier != null; })[0].season;
    var mk = S.filter(function (r) { return r.season >= firstMkt; });   // keeps seasons with no lines as gaps
    watch(brHost, function () {
      lineChart(brHost, {
        series: [
          { id: 'elo', label: 'Elo', short: 'Elo', cls: 's1', pts: mk.map(function (r) { return { x: r.season, y: r.elo_brier }; }) },
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
  }

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['explorer', 'luck', 'history', 'scorecard', 'method'].forEach(function (id) {
      var b = $('#' + id + '-body');
      if (!b) return;
      clear(b);
      b.removeAttribute('aria-busy');
      b.appendChild(h('p', { class: 'note err', text: 'The data files could not be loaded. Serve this folder over http, for example with python3 -m http.server --directory site.' }));
    });
    $('#hero-line').textContent = 'Data unavailable';
  }

  R.load(FILES).then(function () {
    R.initReveal();
    renderHero();
    renderExplorer();
    renderLuck();
    renderHistory();
    renderScorecard();
    renderMethod();
    setHeadline('history-h', D.headlines.history);
    setHeadline('scorecard-h', D.headlines.scorecard);
    var want = new URLSearchParams(location.search).get('team');
    if (want && histBy[want]) pick(want, location.hash === '#explorer');
    N.initNav();
  }).catch(fail);
})();
