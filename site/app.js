/* NFELO home page: the hero matchup, power ladder, this week's games, projected wins and playoff odds.
   No dependencies: reads ./data/*.json (through common.js); formatting, the data loader and reveal come from elo-lib.js. */
(function () {
  'use strict';

  var FILES = ['meta', 'ladder', 'upcoming', 'headlines'];

  // ------------------------------------------------------------------ shared helpers (common.js, elo-lib.js)

  var N = window.NFL, R = N.R;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear;
  var bindTip = N.bindTip, tableView = N.tableView, legend = N.legend;
  var D = R.D, byTeam = R.byTeam;
  var f1 = R.f1, signed = R.signed, pct = R.pct, rec = R.rec, plural = R.plural;
  var longDate = R.longDate, shortDay = R.shortDay, clock = R.clock, ordinalSpots = R.ordinalSpots;
  var setHeadline = R.setHeadline, REDUCED = R.REDUCED;

  // ------------------------------------------------------------------ motion helpers

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

  // ------------------------------------------------------------------ header and matchup instrument

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
      tile('Biggest riser this week', [riser.team, h('small', {}, '▲ ', countUp(riser.rating_change, 1, '', ''))],
        riser.name + ', now #' + riser.rank + ' (' + ordinalSpots(riser.rank_change) + ').'),
      tile('Home-field edge now', [countUp(meta.hfa_pts, 1, '', ''), h('small', { text: 'Elo pts' })],
        pct(meta.hfa_win_pct, 1) + ' home win chance, equal teams.'),
      tile('Elo vs Vegas pick accuracy', [
        h('span', { class: 'duo' }, countUp(sc.elo_accuracy_market_games * 100, 1, '', '%'), h('small', { text: 'Elo' })),
        h('span', { class: 'slash', text: '/' }),
        h('span', { class: 'duo' }, countUp(sc.market_accuracy * 100, 1, '', '%'), h('small', { text: 'Vegas' }))],
        sc.test_seasons[0] + '-' + sc.test_seasons[1] + ', ' + sc.n_market.toLocaleString('en-US') + ' games.', true)
    );
    document.title = 'NFELO: ' + top.team + ' leads at ' + f1(top.rating) + ', through ' + meta.season + ' Week ' + meta.week;
    $('#credit').textContent = meta.credit.replace(/(\d)[\u2013\u2014](\d)/g, '$1 to $2');   // house style: no dashes in copy
  }

  function tile(label, big, sub, dual) {
    return h('div', { class: 'tile' },
      h('h3', { text: label }),
      h('div', { class: 'big' + (dual ? ' dual' : '') }, big),
      h('p', { class: 'sub', text: sub }));
  }

  // The matchup instrument: one game (or team) on a 0 to 100% track with the model as a raised knob and
  // Elo and Vegas as ticks. Three views; each builds a plain description `v` that drawInstrument() renders.
  //   v = { eyebrow, away, awayRole, home, homeRole, pm, pe, pv, knob, stats[], aria, link:{href,text}, ticks }
  // pm is the knob (model, or Elo when there is no model), pe the Elo tick, pv the Vegas tick; null hides a mark.

  function nameParts(abbr) {
    var t = byTeam[abbr];
    if (!t) return { city: abbr, nick: abbr, rec: '' };
    var w = t.name.split(' '), nick = w.pop();
    return { city: w.join(' '), nick: nick, rec: rec(t.w, t.l, t.t) + ' · Elo ' + Math.round(t.rating) };
  }

  var hero = { view: 'week', views: {}, showElo: true, showVegas: true };

  // Unplayed games only (upcoming.json drops played ones), so a game already played never shows.
  // Picks the largest gap: model vs Vegas, else Elo vs Vegas, else model vs Elo, else the first game.
  function weekView() {
    var u = D.upcoming, M = mlWeek();
    if (!u.week || !u.games.length) return { empty: 'No games are left on the schedule.' };
    var mlBy = {};
    if (M) M.games.forEach(function (x) { mlBy[x.game_id] = x; });
    var best = null;
    u.games.forEach(function (g) {
      var mg = mlBy[g.game_id];
      var pm = mg && mg.p_home_model != null ? mg.p_home_model : null, pe = g.p_home;
      var pv = g.p_market_home != null ? g.p_market_home : (mg && mg.p_market_home != null ? mg.p_market_home : null);
      var kind = pm != null && pv != null ? 3 : pv != null ? 2 : pm != null ? 1 : 0;
      var gap = kind === 3 ? pm - pv : kind === 2 ? pe - pv : kind === 1 ? pm - pe : 0;
      var c = { g: g, mg: mg, pm: pm, pe: pe, pv: pv, kind: kind, size: Math.abs(gap) };
      if (!best || c.kind > best.kind || (c.kind === best.kind && c.size > best.size)) best = c;
    });
    var g = best.g, hasModel = best.pm != null;
    var basis = best.kind === 3 ? 'largest NFELO model vs Vegas gap' : best.kind === 2 ? 'largest Elo vs Vegas gap' : best.kind === 1 ? 'largest NFELO model vs Elo gap' : 'first game up';
    var p = hasModel ? best.pm : best.pe;
    var gapPts = best.kind === 3 ? (best.pm - best.pv) * 100 : best.kind === 2 ? (best.pe - best.pv) * 100 : null;
    var lead = nameParts(g.home).nick;
    var stats = [{ l: (hasModel ? 'NFELO model' : 'Elo') + ' · ' + lead + ' win', v: pct(p, 1), c: hasModel ? 'var(--ink)' : 'var(--primary)', clip: hasModel ? 'none' : 'circle(50%)', main: true }];
    if (hasModel) stats.push({ l: 'Elo', v: pct(best.pe, 1), c: 'var(--primary)', clip: 'circle(50%)', key: 'elo' });
    stats.push(best.pv != null ? { l: 'Vegas', v: pct(best.pv, 1), c: 'var(--muted)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas' } : { l: 'Vegas', v: 'No line yet', c: 'var(--muted)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas', na: true });
    stats.push({ l: (hasModel ? 'NFELO model' : 'Elo') + ' vs Vegas', v: gapPts == null ? 'Needs a line' : signed(gapPts, 1) + ' pp', gap: true, na: gapPts == null });
    return {
      eyebrow: 'Week ' + u.week + ' · ' + shortDay(g.date) + ' · ' + basis,
      away: g.away, awayRole: 'Away', home: g.home, homeRole: g.neutral ? 'Home, neutral site' : 'Home',
      pm: p, pe: hasModel ? best.pe : null, pv: best.pv, knob: hasModel ? 'model' : 'elo', stats: stats,
      aria: g.home + ' win probability. ' + (hasModel ? 'NFELO model ' + pct(best.pm, 1) + ', Elo ' + pct(best.pe, 1) : 'Elo ' + pct(best.pe, 1)) + (best.pv != null ? ', Vegas ' + pct(best.pv, 1) : '. No Vegas line yet') + '.',
      axis: nameParts(g.home).nick + ' win probability, %',
      link: { href: '#week', text: 'All games this week' }, ticks: true
    };
  }

  // The team projected for the most wins, and its next game.
  function restView() {
    var R = D.ml && D.ml.rest;
    if (!R || !R.teams.length) return null;
    var rt = R.teams.slice().sort(function (a, b) { return b.proj_model - a.proj_model || (a.team < b.team ? -1 : 1); })[0];
    if (!rt.games || !rt.games.length) return null;
    var gm = rt.games[0], homeAbbr = gm.home ? rt.team : gm.opp, awayAbbr = gm.home ? gm.opp : rt.team;
    var pm = gm.home ? gm.p_model : 1 - gm.p_model, pe = gm.home ? gm.p_elo : 1 - gm.p_elo, pv = null;
    var ug = D.upcoming.games.filter(function (x) { return x.game_id === gm.game_id; })[0];
    if (ug && ug.p_market_home != null) pv = ug.p_market_home;
    var nick = nameParts(homeAbbr).nick;
    return {
      eyebrow: 'Week ' + gm.week + ' · next game for the projected wins leader',
      away: awayAbbr, awayRole: 'Away', home: homeAbbr, homeRole: gm.neutral ? 'Home, neutral site' : 'Home',
      pm: pm, pe: pe, pv: pv, knob: 'model',
      stats: [
        { l: 'NFELO model · ' + nick + ' win', v: pct(pm, 1), c: 'var(--ink)', clip: 'none', main: true },
        { l: 'Elo', v: pct(pe, 1), c: 'var(--primary)', clip: 'circle(50%)', key: 'elo' },
        pv != null ? { l: 'Vegas', v: pct(pv, 1), c: 'var(--muted)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas' } : { l: 'Vegas', v: 'No line yet', c: 'var(--muted)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas', na: true },
        { l: nameParts(rt.team).nick + ' projected wins', v: f1(rt.proj_model) }],
      aria: homeAbbr + ' win probability. NFELO model ' + pct(pm, 1) + ', Elo ' + pct(pe, 1) + (pv != null ? ', Vegas ' + pct(pv, 1) : '. No Vegas line yet') + '.',
      axis: nick + ' win probability, %',
      link: { href: $('#rest') ? '#rest' : '#week', text: 'Rest of season' }, ticks: true
    };
  }

  // The team most likely to make the playoffs.
  function oddsView() {
    var O = D.ml && D.ml.playoff_odds;
    if (!O || !O.conferences) return null;
    var all = [];
    Object.keys(O.conferences).forEach(function (k) { all = all.concat(O.conferences[k]); });
    all.sort(function (a, b) { return b.playoffs - a.playoffs || (a.team < b.team ? -1 : 1); });
    var o = all[0];
    if (!o) return null;
    return {
      eyebrow: 'Season simulation · most likely playoff team',
      away: o.team, awayRole: 'Chance to make the playoffs', home: null,
      pm: o.playoffs, pe: null, pv: null, knob: 'model',
      stats: [
        { l: 'Make playoffs', v: pct(o.playoffs, 1), c: 'var(--ink)', clip: 'none', main: true },
        { l: 'Win division', v: pct(o.division, 1) },
        { l: 'Reach Super Bowl', v: pct(o.reach_sb, 1) },
        { l: 'Win Super Bowl', v: pct(o.win_sb, 1) }],
      aria: byTeam[o.team].name + ' chance to make the playoffs: ' + pct(o.playoffs, 1) + '.',
      axis: byTeam[o.team].name.split(' ').pop() + ' chance to make the playoffs, %',
      link: { href: '#odds', text: 'Playoff odds' }, ticks: false
    };
  }

  function drawTeam(el, abbr, role) {
    clear(el);
    if (!abbr) return;
    var p = nameParts(abbr);
    el.append(h('p', { class: 'label role', text: role }), h('p', { class: 'city', text: p.city }), h('h3', { class: 'nick', text: p.nick }), h('p', { class: 'mx-rec', text: p.rec }));
  }

  function setAt(id, p) { var n = $('#' + id); n.style.setProperty('--at', p == null ? 0.5 : Math.max(0, Math.min(1, p))); }

  function drawInstrument() {
    var v = hero.views[hero.view];
    $('#mx-view').querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-view') === hero.view)); });
    var meter = $('#mx-meter'), stats = $('#mx-stats'), sw = $('#mx-switches');
    if (v.empty) {
      $('#mx-eyebrow').textContent = v.empty;
      drawTeam($('#mx-away'), null); drawTeam($('#mx-home'), null);
      meter.hidden = true; stats.hidden = true; sw.hidden = true;
      $('#mx-open').hidden = true;
      return;
    }
    meter.hidden = false; stats.hidden = false; $('#mx-open').hidden = false;
    $('#mx-eyebrow').textContent = v.eyebrow;
    drawTeam($('#mx-away'), v.away, v.awayRole);
    drawTeam($('#mx-home'), v.home, v.homeRole || '');
    $('#mx-meter-img').classList.toggle('knob-elo', v.knob === 'elo');
    setAt('mx-pos-model', v.pm); setAt('mx-pos-elo', v.pe); setAt('mx-pos-vegas', v.pv);
    $('#mx-pos-elo').hidden = !(v.ticks && hero.showElo && v.pe != null);
    $('#mx-pos-vegas').hidden = !(v.ticks && hero.showVegas && v.pv != null);
    sw.hidden = !v.ticks;
    $('#mx-meter-img').setAttribute('aria-label', v.aria);
    $('#mx-axis').textContent = v.axis || '';
    clear(stats);
    v.stats.forEach(function (s) {
      var off = (s.key === 'elo' && !hero.showElo) || (s.key === 'vegas' && !hero.showVegas);
      var dd = h('dd', { text: s.v });
      var wrap = h('div', { class: 'stat' + (s.main ? ' stat-main' : '') + (s.gap ? ' stat-gap' : '') + (s.na ? ' is-na' : '') + (off ? ' is-off' : ''), style: s.c ? '--c:' + s.c + (s.clip === 'circle(50%)' ? ';--clip:none;--rad:50%;--bd:var(--ink)' : ';--clip:' + (s.clip || 'none')) : null },
        h('dt', { class: 'label', text: s.l }), dd);
      stats.appendChild(wrap);
    });
    var open = $('#mx-open');
    open.setAttribute('href', v.link.href);
    open.firstChild.textContent = v.link.text + ' ';
  }

  function renderHero() {
    var bar = $('#mx-view');
    if (!bar) return;
    hero.views.week = weekView();
    var rv = restView(), ov = oddsView();
    if (rv) hero.views.rest = rv;
    if (ov) hero.views.playoffs = ov;
    clear(bar);
    [['week', 'This week'], ['rest', 'Rest of season'], ['playoffs', 'Playoff odds']].forEach(function (t) {
      if (!hero.views[t[0]]) return;
      bar.appendChild(h('button', { type: 'button', 'data-view': t[0], 'aria-pressed': 'false', text: t[1], onclick: function () { hero.view = t[0]; drawInstrument(); } }));
    });
    bar.hidden = Object.keys(hero.views).length < 2;
    $('#mx-sw-elo').addEventListener('change', function (e) { hero.showElo = e.target.checked; drawInstrument(); });
    $('#mx-sw-vegas').addEventListener('change', function (e) { hero.showVegas = e.target.checked; drawInstrument(); });
    drawInstrument();
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
        chips.appendChild(h('button', { type: 'button', 'aria-pressed': i === active ? 'true' : 'false', text: g[0],
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
      var fill = h('span', { class: 'fill ' + (dev >= 0 ? 'pos' : 'neg'), style: (dev >= 0 ? 'left:50%;' : 'left:' + (50 - wPct) + '%;') + 'width:' + wPct + '%' });
      var up = t.rating_change > 0, flat = t.rating_change === 0;
      var label = t.name + ', rank ' + t.rank + ', rating ' + f1(t.rating) + ', ' + (flat ? 'no change' : (up ? 'up ' : 'down ') + f1(Math.abs(t.rating_change))) +
        ' this week, record ' + rec(t.w, t.l, t.t) + '. Open in the team explorer.';
      var tagText = t === riser && riser.rating_change > 0 ? 'Biggest rise' : t === faller && faller.rating_change < 0 ? 'Biggest fall' : '';
      if (tagText) label = tagText + '. ' + label;
      return h('li', {}, h('button', { class: 'lrow' + (t.rank <= 3 ? ' top' : '') + (tagText ? ' has-tag' : ''), type: 'button', 'aria-label': label, onclick: function () { location.href = 'elo.html?team=' + t.team + '#explorer'; } },
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
      body.appendChild(h('p', { class: 'fnote', text: 'Week: 7-day rating change. Rank: places gained or lost.' }));
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
    var M = mlWeek(), mlBy = {};
    if (M) M.games.forEach(function (x) { mlBy[x.game_id] = x; });

    $('#week-deck').textContent = 'Week ' + u.week + ': ' + plural(left, 'game', 'games') + ' to play' + (u.played ? ', ' + u.played + ' already in the ratings.' : '.');

    var grid = h('div', { class: 'games' });
    u.games.forEach(function (g) {
      var home = g.p_home, away = 1 - g.p_home, mg = M ? mlBy[g.game_id] : null;
      var tag = M ? (mg && mg.flagged ? 'Gap ' + f1(Math.abs(mg.gap)) + ' pp' : (g.neutral ? 'Neutral site' : ''))
        : (g.flagged && g.gap != null ? 'Gap ' + f1(Math.abs(g.gap)) + ' pp' : (g.neutral ? 'Neutral site' : ''));
      var flagged = M ? !!(mg && mg.flagged) : g.flagged;
      var label = g.away + ' at ' + g.home + '. Elo gives ' + g.home + ' ' + pct(home) + '. Elo line ' + lineText(g.home, g.away, g.elo_spread) +
        (g.vegas_spread != null ? ', Vegas line ' + lineText(g.home, g.away, g.vegas_spread) : '') + '.';
      if (mg && mg.p_home_model != null) label += ' The NFELO model gives ' + g.home + ' ' + pct(mg.p_home_model) + (mg.spread_model != null ? ', model line ' + lineText(g.home, g.away, mg.spread_model) : '') + '.';
      if (mg && mg.qb_change) label += ' ' + qbText(mg.qb_change) + '.';
      var lines = M ? tri(g, mg) : h('dl', { class: 'lines' },
        h('div', {}, h('dt', { text: 'Elo line' }), h('dd', { text: lineText(g.home, g.away, g.elo_spread) })),
        g.vegas_spread != null ? h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: lineText(g.home, g.away, g.vegas_spread) })) : h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: 'None yet' })),
        g.diff != null ? h('div', {}, h('dt', { text: 'Spread gap' }), h('dd', { text: f1(Math.abs(g.diff)) + ' pts' })) : null);
      var card = h('article', { class: 'game', 'aria-label': label },
        h('div', { class: 'when' }, h('span', { text: shortDay(g.date) + (g.time ? ', ' + clock(g.time) : '') }),
          tag ? h('span', { class: flagged ? 'tag' : '', text: tag }) : null),
        h('div', { class: 'vs' }, g.away, h('span', { class: 'at', text: 'at' }), g.home),
        mg && mg.qb_change ? h('p', { class: 'qbtag', text: qbText(mg.qb_change) }) : null,
        h('div', { class: 'pbar', role: 'img', 'aria-hidden': 'true' },
          h('i', { style: 'width:' + (away * 100).toFixed(1) + '%' }),
          h('i', { style: 'width:' + (home * 100).toFixed(1) + '%' })),
        h('div', { class: 'plabels' },
          h('span', {}, g.away + ' ', h('b', { text: pct(away) })),
          h('span', {}, h('b', { text: pct(home) }), ' ' + g.home)),
        lines);
      grid.appendChild(card);
    });
    body.appendChild(grid);
    var anySpread = M && M.games.some(function (x) { return x.spread_model != null; });
    body.appendChild(h('p', { class: 'fnote', text: M
      ? 'Home-team win chances. Gap tags: the biggest model-Vegas differences, in percentage points. Times Eastern.'
      : 'Home-team win chances. Gap tags: the biggest Elo-Vegas differences, in percentage points. Times Eastern.' }));
  }

  // ------------------------------------------------------------------ ML model (optional: only when data/ml.json exists)

  function mlWeek() {
    var M = D.ml && D.ml.week;
    return M && M.week === D.upcoming.week ? M : null;
  }

  function qbText(q) {
    return 'QB change: ' + (q.name || 'new starter') + ', ' + q.team;
  }

  // Elo, Model, Vegas side by side: the home team's win chance, then each line.
  function tri(g, mg) {
    var pm = mg && mg.p_home_model != null ? mg.p_home_model : null;
    function td(text, na) { return h('td', { class: na ? 'na' : '', text: text }); }
    function head(cls, text) { return h('th', { scope: 'col' }, h('i', { class: cls, 'aria-hidden': 'true' }), text); }
    return h('table', { class: 'tri' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, h('span', { class: 'sr', text: 'Measure' })), head('k-s1', 'Elo'), head('k-s3', 'NFELO model'), head('k-s2', 'Vegas'))),
      h('tbody', {},
        h('tr', {}, h('th', { scope: 'row' }, g.home, h('span', { class: 'sr', text: ' win chance' })), td(pct(g.p_home)), pm != null ? td(pct(pm)) : td('None yet', true),
          g.p_market_home != null ? td(pct(g.p_market_home)) : td('None yet', true)),
        h('tr', {}, h('th', { scope: 'row', text: 'Line' }), td(lineText(g.home, g.away, g.elo_spread)),
          mg && mg.spread_model != null ? td(lineText(g.home, g.away, mg.spread_model)) : td('Not yet', true),
          g.vegas_spread != null ? td(lineText(g.home, g.away, g.vegas_spread)) : td('None yet', true))));
  }

  function winsText(v) { return Math.abs(v - Math.round(v)) < 0.01 ? String(Math.round(v)) : f1(v); }

  // Inserts an optional section after the section `afterId`, adds its nav link after that section's link,
  // and renumbers every kicker so the sequence stays 01, 02, ...
  function insertSection(sec, afterId, navText) {
    var after = $('#' + afterId);
    after.parentNode.insertBefore(sec, after.nextSibling);
    var navAfter = document.querySelector('.nav-links a[href="#' + afterId + '"]');
    if (navAfter) navAfter.parentNode.parentNode.insertBefore(h('li', {}, h('a', { href: '#' + sec.id, text: navText })), navAfter.parentNode.nextSibling);
    N.initSlabs();
    document.querySelectorAll('main > .sec .kicker .kn').forEach(function (k, i) {
      k.textContent = (i < 9 ? '0' : '') + (i + 1);
    });
  }

  var DIVS = ['AFC East', 'AFC North', 'AFC South', 'AFC West', 'NFC East', 'NFC North', 'NFC South', 'NFC West'];

  // Builds the Rest of season section after This week, adds its nav link, and renumbers the kickers.
  function renderRest() {
    var R = D.ml && D.ml.rest;
    if (!R || !R.teams.length) return;
    var body = h('div', { id: 'rest-body', class: 'tile-m body' });
    var sec = h('section', { class: 'sec', id: 'rest', 'aria-labelledby': 'rest-h', tabindex: '-1' },
      h('div', { class: 'wrap' }, h('div', { class: 'board sec-board' },
        h('header', { class: 'sec-head tile-m' }, h('div', { class: 'sec-copy' },
          h('p', { class: 'kicker' }, h('span', { class: 'kn', text: '00' }), h('span', { class: 'kt', text: 'Rest of season' })),
          h('h2', { class: 'headline', id: 'rest-h', text: R.headline || 'Rest of season' }),
          h('p', { class: 'deck', text: 'Projected final wins.' }))),
        body)));
    insertSection(sec, 'week', 'Rest of season');

    var MAXW = 17;
    function hasRange(r) { return r.range_model != null && r.range_elo != null; }
    function rangeOf(a) { return winsText(a[0]) + ' to ' + winsText(a[1]); }
    function x(v) { return (Math.max(0, Math.min(MAXW, v)) / MAXW * 100).toFixed(2) + '%'; }
    function bar(cls, label, v, val, rg) {
      return h('span', { class: 'wl ' + cls },
        h('span', { class: 'wk', text: label }),
        h('span', { class: 'wb' },
          rg ? h('i', { class: 'rg', style: 'left:' + x(rg[0]) + ';width:' + x(rg[1] - rg[0]) }) : null,
          h('i', { class: 'fl', style: 'width:' + x(v) })),
        h('span', { class: 'wv', text: val }));
    }
    function row(r) {
      var t = byTeam[r.team], base = r.w + r.t / 2, record = rec(r.w, r.l, r.t), rg = hasRange(r);
      var summary = h('summary', {},
        h('span', { class: 'sr', text: t.name + ', ' + record + '. Projected ' + f1(r.proj_model) + ' wins by the NFELO model' + (rg ? ' (likely ' + rangeOf(r.range_model) + ')' : '') + ', ' + f1(r.proj_elo) + ' by Elo' + (rg ? ' (likely ' + rangeOf(r.range_elo) + ')' : '') + ', ' + plural(r.remaining, 'game', 'games') + ' left. Show remaining games.' }),
        h('span', { class: 'who', 'aria-hidden': 'true' }, h('span', { class: 'abbr', text: r.team }), h('span', { class: 'meta', text: r.remaining + ' left' })),
        h('span', { class: 'mid', 'aria-hidden': 'true' },
          bar('w-so', 'Wins', base, record),
          bar('w-md', 'NFELO model', r.proj_model, f1(r.proj_model), rg ? r.range_model : null),
          bar('w-el', 'Elo', r.proj_elo, f1(r.proj_elo), rg ? r.range_elo : null)));
      bindTip(summary, function () {
        return { title: t.name, rows: [
          { value: f1(r.proj_model), label: 'model projection', cls: 'wmd' },
          { value: f1(r.proj_elo), label: 'Elo projection', cls: 'wel' },
          hasRange(r) ? { value: rangeOf(r.range_model), label: 'model likely range', cls: 'wrg' } : null,
          hasRange(r) ? { value: rangeOf(r.range_elo), label: 'Elo likely range', cls: 'wrg' } : null,
          { value: record, label: 'so far', cls: 'wso' },
          { value: String(r.remaining), label: r.remaining === 1 ? 'game left' : 'games left' }] };
      });
      var games = h('div', { class: 'rs-games' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Week' }), h('th', { scope: 'col', text: 'Game' }),
          h('th', { scope: 'col', class: 'r', text: 'NFELO model' }), h('th', { scope: 'col', class: 'r', text: 'Elo' }))),
        h('tbody', {}, r.games.map(function (g) {
          return h('tr', {}, h('td', { class: 'n', text: String(g.week) }),
            h('td', { text: (g.home ? 'vs ' : 'at ') + g.opp + (g.neutral ? ' (neutral)' : '') }),
            h('td', { class: 'r n', text: pct(g.p_model) }), h('td', { class: 'r n', text: pct(g.p_elo) }));
        }))));
      return h('details', { class: 'rs' }, summary, games);
    }

    var ranged = R.teams.some(hasRange);
    body.appendChild(legend([['wso', 'Wins so far'], ['wmd', 'NFELO model'], ['wel', 'Elo']].concat(ranged ? [['wrg', 'Likely range']] : [])));
    var grid = h('div', { class: 'divs' });
    DIVS.forEach(function (d) {
      var teams = R.teams.filter(function (r) { return byTeam[r.team] && byTeam[r.team].div === d; });
      teams.sort(function (a, b) { return b.proj_model - a.proj_model || (a.team < b.team ? -1 : 1); });
      grid.appendChild(h('div', { class: 'dv' }, h('h3', {}, d, h('span', { text: 'Projected wins' })), teams.map(row)));
    });
    body.appendChild(grid);
    body.appendChild(h('p', { class: 'fnote', text: ranged ? 'Pale band: likely range, 8 in 10 simulated seasons.' : 'Bars run from 0 to 17 wins.' }));
    body.appendChild(tableView('View projected wins as a table', [{ t: 'Team' }, { t: 'Record', r: 1, n: 1 }, { t: 'Left', r: 1, n: 1 }, { t: 'NFELO model', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }].concat(ranged ? [{ t: 'Model range', r: 1, n: 1 }, { t: 'Elo range', r: 1, n: 1 }] : []), function () {
      return R.teams.map(function (r) { return [byTeam[r.team].name, rec(r.w, r.l, r.t), String(r.remaining), f1(r.proj_model), f1(r.proj_elo)].concat(ranged ? (hasRange(r) ? [rangeOf(r.range_model), rangeOf(r.range_elo)] : ['', '']) : []); });
    }));
  }

  // ------------------------------------------------------------------ playoff odds (optional: ml.json with a simulation run)

  function oddsPct(p) {
    if (p <= 0) return '0%';
    if (p >= 1) return '100%';
    if (p < 0.005) return '<1%';
    if (p > 0.995) return '>99%';
    return Math.round(p * 100) + '%';
  }

  function oddsChange(d) {
    if (d == null) return null;
    var v = Math.round(d * 100);
    return { v: v, text: v === 0 ? '0' : (v > 0 ? '+' : '-') + Math.abs(v), sr: v === 0 ? 'no change' : (v > 0 ? 'up ' : 'down ') + plural(Math.abs(v), 'percentage point', 'percentage points') };
  }

  function shortRun(iso) {
    return new Date(iso).toLocaleDateString('en-US', { timeZone: 'America/New_York', weekday: 'short', month: 'short', day: 'numeric' });
  }

  function renderOdds() {
    var O = D.ml && D.ml.playoff_odds;
    if (!O || !O.conferences) return;
    var hasBase = !!O.baseline_run_utc;
    var body = h('div', { id: 'odds-body', class: 'tile-m body' });
    var sec = h('section', { class: 'sec', id: 'odds', 'aria-labelledby': 'odds-h', tabindex: '-1' },
      h('div', { class: 'wrap' }, h('div', { class: 'board sec-board' },
        h('header', { class: 'sec-head tile-m' }, h('div', { class: 'sec-copy' },
        h('p', { class: 'kicker' }, h('span', { class: 'kn', text: '00' }), h('span', { class: 'kt', text: 'Playoff odds' })),
        h('h2', { class: 'headline', id: 'odds-h', text: O.headline ? O.headline.split('. ')[0].replace(/\.?$/, '.') : 'Playoff odds' }),
        h('p', { class: 'deck', text: O.n_sims.toLocaleString('en-US') + ' simulated seasons.' + (hasBase ? ' Change in points since ' + shortRun(O.baseline_run_utc) + '.' : '') }))),
        body)));
    insertSection(sec, $('#rest') ? 'rest' : 'week', 'Playoff odds');

    var COLS = [
      { k: 'team', t: 'Team' },
      { k: 'playoffs', t: 'Playoffs', bar: 1 },
      { k: 'd_playoffs', t: 'Change', chg: 1 },
      { k: 'division', t: 'Division' },
      { k: 'seed1', t: '#1 seed' },
      { k: 'win_sb', t: 'Win SB' }
    ].filter(function (c) { return hasBase || !c.chg; });

    function block(conf) {
      var rows = O.conferences[conf].slice(), sortKey = 'playoffs', desc = true;
      var tbody = h('tbody'), heads = [];
      function draw() {
        rows.sort(function (a, b) {
          var x = a[sortKey], y = b[sortKey];
          if (sortKey === 'team') { x = byTeam[a.team].name; y = byTeam[b.team].name; return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1); }
          if (x == null) x = -Infinity; if (y == null) y = -Infinity;
          return (desc ? y - x : x - y) || b.playoffs - a.playoffs || (a.team < b.team ? -1 : 1);
        });
        clear(tbody);
        rows.forEach(function (r) {
          var t = byTeam[r.team], ch = oddsChange(r.d_playoffs);
          var cells = COLS.map(function (c) {
            if (c.k === 'team') return h('th', { scope: 'row', class: 'tm' }, h('span', { class: 'abbr', text: r.team }), h('span', { class: 'nm', text: t.name }));
            if (c.chg) return h('td', { class: 'r n chg ' + (ch && ch.v > 0 ? 'up' : ch && ch.v < 0 ? 'down' : '') },
              ch ? [h('span', { 'aria-hidden': 'true', text: (ch.v > 0 ? '▲ ' : ch.v < 0 ? '▼ ' : '') + ch.text }), h('span', { class: 'sr', text: ch.sr })] : h('span', { class: 'na', text: 'New' }));
            if (c.bar) return h('td', { class: 'r n pc' }, h('span', { class: 'ob', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (r[c.k] * 100).toFixed(1) + '%' })), h('b', { text: oddsPct(r[c.k]) }));
            return h('td', { class: 'r n', text: oddsPct(r[c.k]) });
          });
          var tr = h('tr', { tabindex: '0' }, cells);
          bindTip(tr, function () {
            return { title: t.name + ', ' + r.div, rows: [
              { value: oddsPct(r.playoffs), label: 'make the playoffs', cls: 's3' },
              { value: oddsPct(r.division), label: 'win the division' },
              { value: oddsPct(r.seed1), label: 'get the #1 seed' },
              { value: oddsPct(r.reach_sb), label: 'reach the Super Bowl' },
              { value: oddsPct(r.win_sb), label: 'win the Super Bowl' },
              { value: f1(r.wins_mean), label: 'wins on average, likely ' + winsText(r.wins_p10) + ' to ' + winsText(r.wins_p90) },
              ch ? { value: ch.text, label: 'points of playoff chance since ' + shortRun(O.baseline_run_utc) } : null] };
          });
          tbody.appendChild(tr);
        });
        heads.forEach(function (hd) {
          var on = hd.k === sortKey;
          hd.th.setAttribute('aria-sort', on ? (desc ? 'descending' : 'ascending') : 'none');
          hd.btn.classList.toggle('on', on);
          hd.btn.setAttribute('data-dir', on ? (desc ? 'desc' : 'asc') : '');
        });
      }
      var thead = h('thead', {}, h('tr', {}, COLS.map(function (c) {
        var btn = h('button', { type: 'button', class: 'sortb', text: c.t, onclick: function () {
          if (sortKey === c.k) desc = !desc; else { sortKey = c.k; desc = c.k !== 'team'; }
          draw();
        } });
        var th = h('th', { scope: 'col', class: c.k === 'team' ? '' : 'r' }, btn);
        heads.push({ k: c.k, th: th, btn: btn });
        return th;
      })));
      draw();
      return h('div', { class: 'oc' }, h('h3', {}, conf, h('span', { text: (D.ml.season >= 2020 ? 7 : 6) + ' make it' })),
        h('div', { class: 'tbl-wrap' }, h('table', { class: 'odds' }, h('caption', { class: 'sr', text: conf + ' playoff odds' }), thead, tbody)));
    }

    body.appendChild(legend([['s3 blk', 'Chance to make the playoffs']].concat(hasBase ? [['up-k', 'Up since last run'], ['down-k', 'Down']] : [])));
    body.appendChild(h('div', { class: 'odds-grid' }, ['AFC', 'NFC'].map(block)));
    body.appendChild(tableView('View playoff odds as a table',
      [{ t: 'Team' }, { t: 'Conf' }, { t: 'Playoffs', r: 1, n: 1 }, { t: 'Division', r: 1, n: 1 }, { t: '#1 seed', r: 1, n: 1 }, { t: 'Reach SB', r: 1, n: 1 }, { t: 'Win SB', r: 1, n: 1 }, { t: 'Mean wins', r: 1, n: 1 }, { t: 'Likely wins', r: 1, n: 1 }].concat(hasBase ? [{ t: 'Change', r: 1, n: 1 }] : []),
      function () {
        return ['AFC', 'NFC'].reduce(function (acc, c) {
          return acc.concat(O.conferences[c].map(function (r) {
            var ch = oddsChange(r.d_playoffs);
            return [byTeam[r.team].name, c, pct(r.playoffs, 1), pct(r.division, 1), pct(r.seed1, 1), pct(r.reach_sb, 1), pct(r.win_sb, 1), f1(r.wins_mean), winsText(r.wins_p10) + ' to ' + winsText(r.wins_p90)].concat(hasBase ? [ch ? ch.text : 'New'] : []);
          }));
        }, []);
      }));
  }

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['ladder', 'week'].forEach(function (id) {
      var b = $('#' + id + '-body');
      if (!b) return;
      clear(b);
      b.removeAttribute('aria-busy');
      b.appendChild(h('p', { class: 'note err', text: 'The data files could not be loaded. Serve this folder over http, for example with python3 -m http.server --directory site.' }));
    });
    $('#status').textContent = 'Data unavailable';
    var eb = $('#mx-eyebrow');
    if (eb) eb.textContent = 'Data unavailable';
  }

  // ml.json is optional: it exists only once the ML prediction ledger does. Without it the page is unchanged.
  R.load(FILES).then(function () {
    renderRest();
    renderOdds();
    R.initReveal();
    renderHeader();
    renderLadder();
    renderWeek();
    renderHero();
    setHeadline('ladder-h', D.headlines.ladder);
    setHeadline('week-h', (mlWeek() && mlWeek().headline) || D.headlines.week);
    N.initNav();
  }).catch(fail);
})();
