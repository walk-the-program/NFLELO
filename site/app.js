/* NFLELO ratings page. No dependencies: reads ./data/*.json (through common.js), draws hand-built SVG/HTML.
   Charts follow one spec: 2px lines, hairline solid grid, 2px surface rings on dots, one tooltip style,
   a table view under each chart, and chart areas set in bordered wells. Series are told apart by color, shape and dash. */
(function () {
  'use strict';

  var FILES = ['meta', 'ladder', 'upcoming', 'history', 'luck', 'tapestry', 'records', 'scorecard', 'headlines'];
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ shared helpers (common.js)

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, tipHide = N.tipHide, bindTip = N.bindTip;
  var watch = N.watch, tableView = N.tableView, legend = N.legend, mark = N.mark, moveMark = N.moveMark;

  // ------------------------------------------------------------------ formatting

  function f1(n) { return n.toFixed(1); }
  function signed(n, d) { d = d == null ? 1 : d; var s = Math.abs(n).toFixed(d); return (+s === 0 ? '' : n > 0 ? '+' : '-') + s; }
  function pct(p, d) { return (p * 100).toFixed(d == null ? 0 : d) + '%'; }
  function rec(w, l, t) { return w + '-' + l + (t ? '-' + t : ''); }
  function winPct(w, l, t) { return ((w + t / 2) / (w + l + t) * 100).toFixed(1) + '%'; }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  function parseDate(iso) { var p = iso.split('-'); return new Date(+p[0], +p[1] - 1, +p[2]); }
  function longDate(iso) { return parseDate(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }); }
  function shortDay(iso) { return parseDate(iso).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }); }
  function clock(hhmm) {
    if (!hhmm) return '';
    var p = hhmm.split(':'), hr = +p[0];
    return ((hr + 11) % 12 + 1) + ':' + p[1] + ' ' + (hr < 12 ? 'AM' : 'PM') + ' ET';
  }
  function ordinalSpots(n) { return n === 0 ? 'unchanged' : (n > 0 ? 'up ' : 'down ') + plural(Math.abs(n), 'spot', 'spots'); }

  // ------------------------------------------------------------------ generic line chart

  function niceStep(span, n) {
    var raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
  }

  function ticksFor(lo, hi, n) {
    var step = niceStep(hi - lo, n), out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  /* o: { series:[{id,label,cls,style,pts:[{x,y,...}]}], x:[a,b], y:[a,b], height(w), xTicks:[], yTicks:[],
          xFmt, yFmt, ref:{y,label}, tip(x, rows)->{title,rows}, aria, markers:[{x,y,text,dy}], endLabels } */
  function lineChart(host, o) {
    clear(host);
    var W = Math.max(280, host.clientWidth || 640);
    var H = o.height(W);
    var m = { t: 12, r: o.endLabels ? 68 : 16, b: 28, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var sx = function (v) { return m.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * iw; };
    var sy = function (v) { return m.t + (1 - (v - o.y[0]) / (o.y[1] - o.y[0])) * ih; };

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', tabindex: 0, 'aria-label': o.aria }, host);

    o.yTicks.forEach(function (v) {
      sv('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, svg);
      sv('text', { class: 'tick a-end', x: m.l - 8, y: sy(v) + 4, text: o.yFmt(v) }, svg);
    });
    // Skip x ticks that would sit within 44px of the previous one (narrow screens).
    var lastTickX = -1e9;
    o.xTicks.forEach(function (v) {
      if (sx(v) - lastTickX < 44) return;
      lastTickX = sx(v);
      sv('text', { class: 'tick a-mid', x: sx(v), y: H - 8, text: o.xFmt(v) }, svg);
    });
    if (o.ref) {
      sv('line', { class: 'refline', x1: m.l, x2: W - m.r, y1: sy(o.ref.y), y2: sy(o.ref.y) }, svg);
      // Put the label in the corner (left/right, above/below the line) that touches the fewest data points.
      var lw = o.ref.label.length * 6.6 + 8, ry = sy(o.ref.y), best = null;
      [[1, -1], [1, 1], [0, -1], [0, 1]].forEach(function (c) {
        var x0 = c[0] ? W - m.r - 4 - lw : m.l + 2, x1 = x0 + lw;
        var y0 = c[1] < 0 ? ry - 18 : ry + 2, y1 = y0 + 16, hits = 0;
        o.series.forEach(function (sr) {
          sr.pts.forEach(function (p) { if (p.y != null && sx(p.x) >= x0 && sx(p.x) <= x1 && sy(p.y) >= y0 && sy(p.y) <= y1) hits++; });
        });
        if (!best || hits < best.hits) best = { hits: hits, right: c[0], above: c[1] < 0 };
      });
      if (best.hits === 0) sv('text', { class: 'tick halo ' + (best.right ? 'a-end' : 'a-start'), x: best.right ? W - m.r - 4 : m.l + 4,
        y: best.above ? ry - 5 : ry + 13, text: o.ref.label }, svg);
    }

    o.series.forEach(function (s) {
      // Points with y == null break the line into separate segments.
      var d = '', pen = false;
      s.pts.forEach(function (p) {
        if (p.y == null) { pen = false; return; }
        d += (pen ? 'L' : 'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1);
        pen = true;
      });
      if (s.dotsOnly) {
        s.pts.forEach(function (p) { if (p.y != null) mark(svg, s.cls, sx(p.x), sy(p.y), 3.5, 'plain'); });
      } else {
        sv('path', { class: 'line ' + s.cls, d: d }, svg);
      }
      var last = s.pts.filter(function (p) { return p.y != null; }).pop();
      if (!s.dotsOnly) mark(svg, s.cls, sx(last.x), sy(last.y), 5);
      if (o.endLabels) {
        sv('text', { class: 'lbl', x: sx(last.x) + 10, y: sy(last.y) + 4, text: s.short || s.label }, svg);
      }
    });

    (o.markers || []).forEach(function (mk) {
      mark(svg, mk.cls, sx(mk.x), sy(mk.y), 6);
      var anchor = sx(mk.x) > W - 90 ? 'end' : sx(mk.x) < m.l + 70 ? 'start' : 'middle';
      sv('text', { class: 'lbl', x: sx(mk.x), y: sy(mk.y) + mk.dy, 'text-anchor': anchor, text: mk.text }, svg);
    });

    // Hover layer: a crosshair snaps to the nearest x; the tooltip lists every series at that x.
    var maps = o.series.map(function (s) { var mp = new Map(); s.pts.forEach(function (p) { mp.set(p.x, p); }); return mp; });
    var xs = Array.from(new Set([].concat.apply([], o.series.map(function (s) { return s.pts.map(function (p) { return p.x; }); })))).sort(function (a, b) { return a - b; });
    var cross = sv('line', { class: 'cross', y1: m.t, y2: m.t + ih, visibility: 'hidden' }, svg);
    var dots = o.series.map(function (s) { var d = mark(svg, s.cls, 0, 0, 6); d.setAttribute('visibility', 'hidden'); return d; });
    var cur = xs.length - 1;

    function nearest(px) {
      var v = o.x[0] + (px - m.l) / iw * (o.x[1] - o.x[0]), lo = 0, hi = xs.length - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] < v) lo = mid; else hi = mid; }
      return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
    }

    function show(i, cx, cy) {
      cur = i;
      var x = xs[i];
      cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.setAttribute('visibility', 'visible');
      var rows = [];
      o.series.forEach(function (s, k) {
        var p = maps[k].get(x);
        if (!p || p.y == null) { dots[k].setAttribute('visibility', 'hidden'); if (p) rows.push({ p: p, s: s }); return; }
        moveMark(dots[k], sx(x), sy(p.y)); dots[k].setAttribute('visibility', 'visible');
        rows.push({ p: p, s: s });
      });
      var c = o.tip(x, rows);
      if (cx == null) { var r = svg.getBoundingClientRect(); cx = r.left + sx(x) * (r.width / W); cy = r.top + m.t + 10; }
      tipShow(cx, cy, c.title, c.rows);
    }

    function hide() {
      cross.setAttribute('visibility', 'hidden');
      dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); });
      tipHide();
    }

    svg.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect();
      var px = (e.clientX - r.left) * (W / r.width);
      if (px < m.l - 4 || px > W - m.r + 4) return hide();
      show(nearest(px), e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () { if (!svg.matches(':hover')) show(cur); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 5 : 1, i = cur;
      if (e.key === 'ArrowLeft') i = Math.max(0, cur - step);
      else if (e.key === 'ArrowRight') i = Math.min(xs.length - 1, cur + step);
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = xs.length - 1;
      else if (e.key === 'Escape') return hide();
      else return;
      e.preventDefault();
      show(i);
    });
    return svg;
  }

  // ------------------------------------------------------------------ state

  var D = {};          // loaded data
  var byTeam = {};     // ladder row by franchise
  var histBy = {};     // history row by franchise
  var pick = null;     // setter for the explorer

  function name(code) { return byTeam[code] ? byTeam[code].name : code; }

  // ------------------------------------------------------------------ headlines, motion helpers

  function setHeadline(id, text) { var n = $('#' + id); if (n && text) n.textContent = text; }

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

  // Bars grow once, when their section first scrolls into view (CSS does the easing).
  function initReveal() {
    if (REDUCED || !('IntersectionObserver' in window)) return;
    document.documentElement.classList.add('anim');
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
    document.querySelectorAll('main > .sec').forEach(function (sec) { io.observe(sec); });
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
        'Worth ' + pct(meta.hfa_win_pct, 1) + ' to the home team in a game between equal teams.'),
      tile('Elo vs Vegas pick accuracy', [
        h('span', { class: 'duo' }, countUp(sc.elo_accuracy_market_games * 100, 1, '', '%'), h('small', { text: 'Elo' })),
        h('span', { class: 'slash', text: '/' }),
        h('span', { class: 'duo' }, countUp(sc.market_accuracy * 100, 1, '', '%'), h('small', { text: 'Vegas' }))],
        sc.test_seasons[0] + '-' + sc.test_seasons[1] + ' regular seasons, ' + sc.n_market.toLocaleString('en-US') + ' games with a moneyline.', true)
    );
    document.title = 'NFLELO: ' + top.team + ' leads at ' + f1(top.rating) + ', through ' + meta.season + ' Week ' + meta.week;
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
    var basis = best.kind === 3 ? 'largest model vs Vegas gap' : best.kind === 2 ? 'largest Elo vs Vegas gap' : best.kind === 1 ? 'largest model vs Elo gap' : 'first game up';
    var p = hasModel ? best.pm : best.pe;
    var gapPts = best.kind === 3 ? (best.pm - best.pv) * 100 : best.kind === 2 ? (best.pe - best.pv) * 100 : null;
    var lead = nameParts(g.home).nick;
    var stats = [{ l: (hasModel ? 'Model' : 'Elo') + ' · ' + lead + ' win', v: pct(p, 1), c: hasModel ? 'var(--primary)' : 'var(--ink)', clip: hasModel ? 'none' : 'circle(50%)', main: true }];
    if (hasModel) stats.push({ l: 'Elo', v: pct(best.pe, 1), c: 'var(--ink)', clip: 'circle(50%)', key: 'elo' });
    stats.push(best.pv != null ? { l: 'Vegas', v: pct(best.pv, 1), c: 'var(--secondary)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas' } : { l: 'Vegas', v: 'No line yet', c: 'var(--secondary)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas', na: true });
    stats.push({ l: (hasModel ? 'Model' : 'Elo') + ' vs Vegas', v: gapPts == null ? 'Needs a line' : signed(gapPts, 1) + ' pts', gap: true, na: gapPts == null });
    return {
      eyebrow: 'Week ' + u.week + ' · ' + shortDay(g.date) + ' · ' + basis,
      away: g.away, awayRole: 'Away', home: g.home, homeRole: g.neutral ? 'Home, neutral site' : 'Home',
      pm: p, pe: hasModel ? best.pe : null, pv: best.pv, knob: hasModel ? 'model' : 'elo', stats: stats,
      aria: g.home + ' win probability. ' + (hasModel ? 'Model ' + pct(best.pm, 1) + ', Elo ' + pct(best.pe, 1) : 'Elo ' + pct(best.pe, 1)) + (best.pv != null ? ', Vegas ' + pct(best.pv, 1) : '. No Vegas line yet') + '.',
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
        { l: 'Model · ' + nick + ' win', v: pct(pm, 1), c: 'var(--primary)', clip: 'none', main: true },
        { l: 'Elo', v: pct(pe, 1), c: 'var(--ink)', clip: 'circle(50%)', key: 'elo' },
        pv != null ? { l: 'Vegas', v: pct(pv, 1), c: 'var(--secondary)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas' } : { l: 'Vegas', v: 'No line yet', c: 'var(--secondary)', clip: 'polygon(50% 0,100% 100%,0 100%)', key: 'vegas', na: true },
        { l: nameParts(rt.team).nick + ' projected wins', v: f1(rt.proj_model) }],
      aria: homeAbbr + ' win probability. Model ' + pct(pm, 1) + ', Elo ' + pct(pe, 1) + (pv != null ? ', Vegas ' + pct(pv, 1) : '. No Vegas line yet') + '.',
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
        { l: 'Make playoffs', v: pct(o.playoffs, 1), c: 'var(--primary)', clip: 'none', main: true },
        { l: 'Win division', v: pct(o.division, 1) },
        { l: 'Reach Super Bowl', v: pct(o.reach_sb, 1) },
        { l: 'Win Super Bowl', v: pct(o.win_sb, 1) }],
      aria: byTeam[o.team].name + ' chance to make the playoffs: ' + pct(o.playoffs, 1) + '.',
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
    clear(stats);
    v.stats.forEach(function (s) {
      var off = (s.key === 'elo' && !hero.showElo) || (s.key === 'vegas' && !hero.showVegas);
      var dd = h('dd', { text: s.v });
      var wrap = h('div', { class: 'stat' + (s.main ? ' stat-main' : '') + (s.gap ? ' stat-gap' : '') + (s.na ? ' is-na' : '') + (off ? ' is-off' : ''), style: s.c ? '--c:' + s.c + ';--clip:' + (s.clip || 'none') : null },
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
      return h('li', {}, h('button', { class: 'lrow' + (t.rank <= 3 ? ' top' : '') + (tagText ? ' has-tag' : ''), type: 'button', 'aria-label': label, onclick: function () { pick(t.team, true); } },
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
      body.appendChild(h('p', { class: 'fnote', text: 'Week is the rating change over the last 7 days; Rank is places gained or lost. The trend line covers each team\'s last 17 games, with this season highlighted.' }));
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

    $('#week-deck').textContent = 'Week ' + u.week + (u.played ? ': ' + plural(left, 'game', 'games') + ' still to play, ' + u.played + ' already in the ratings' : ': ' + plural(left, 'game', 'games')) +
      (M ? '. Win chance and point spread from Elo and the model, next to the Vegas line.' : '. Elo win chance and point spread next to the Vegas line.');

    var grid = h('div', { class: 'games' });
    u.games.forEach(function (g) {
      var home = g.p_home, away = 1 - g.p_home, mg = M ? mlBy[g.game_id] : null;
      var tag = M ? (mg && mg.flagged ? 'Gap ' + Math.round(Math.abs(mg.gap)) + ' pts' : (g.neutral ? 'Neutral site' : ''))
        : (g.flagged ? 'Gap ' + f1(Math.abs(g.diff)) : (g.neutral ? 'Neutral site' : ''));
      var flagged = M ? !!(mg && mg.flagged) : g.flagged;
      var label = g.away + ' at ' + g.home + '. Elo gives ' + g.home + ' ' + pct(home) + '. Elo line ' + lineText(g.home, g.away, g.elo_spread) +
        (g.vegas_spread != null ? ', Vegas line ' + lineText(g.home, g.away, g.vegas_spread) : '') + '.';
      if (mg && mg.p_home_model != null) label += ' The model gives ' + g.home + ' ' + pct(mg.p_home_model) + (mg.spread_model != null ? ', model line ' + lineText(g.home, g.away, mg.spread_model) : '') + '.';
      if (mg && mg.qb_change) label += ' ' + qbText(mg.qb_change) + '.';
      var lines = M ? tri(g, mg) : h('dl', { class: 'lines' },
        h('div', {}, h('dt', { text: 'Elo line' }), h('dd', { text: lineText(g.home, g.away, g.elo_spread) })),
        g.vegas_spread != null ? h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: lineText(g.home, g.away, g.vegas_spread) })) : h('div', {}, h('dt', { text: 'Vegas line' }), h('dd', { text: 'None yet' })),
        g.diff != null ? h('div', {}, h('dt', { text: 'Gap' }), h('dd', { text: f1(Math.abs(g.diff)) })) : null);
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
      ? 'Win chances are the home team\'s. Vegas is the moneyline with the bookmaker\'s margin removed. Lines name the favorite and the points it is favored by' + (anySpread ? '; the model\'s line is the average final margin its score model expects, and it shows Not yet for games predicted before that model went live' : '; the model has no point spread yet') + '. Gap is how far the model\'s win chance is from Vegas, in percentage points, for the three biggest. A QB change tag marks a starter whose play the team\'s recent numbers do not reflect. Kickoff times are Eastern.'
      : 'Lines name the favorite and the points it is favored by. Elo spread is the rating gap (with home-field edge, none at neutral sites) divided by 25. Kickoff times are Eastern.' }));
  }

  // ------------------------------------------------------------------ ML model (optional: only when data/ml.json exists)

  function mlWeek() {
    var M = D.ml && D.ml.week;
    return M && M.week === D.upcoming.week ? M : null;
  }

  function qbText(q) {
    return 'QB change: ' + (q.name || 'new starter') + ', ' + q.team;
  }

  function runTime(iso) {
    return new Date(iso).toLocaleString('en-US', { timeZone: 'America/New_York', weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' ET';
  }

  function dayLong(iso) { return parseDate(iso).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' }); }

  // Elo, Model, Vegas side by side: the home team's win chance, then each line.
  function tri(g, mg) {
    var pm = mg && mg.p_home_model != null ? mg.p_home_model : null;
    function td(text, na) { return h('td', { class: na ? 'na' : '', text: text }); }
    function head(cls, text) { return h('th', { scope: 'col' }, h('i', { class: cls, 'aria-hidden': 'true' }), text); }
    return h('table', { class: 'tri' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, h('span', { class: 'sr', text: 'Measure' })), head('k-s1', 'Elo'), head('k-s3', 'Model'), head('k-s2', 'Vegas'))),
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
    document.querySelectorAll('main > .sec .kicker').forEach(function (k, i) {
      k.textContent = '[' + (i < 9 ? '0' : '') + (i + 1) + '] ' + k.textContent.replace(/^\[\d+\]\s*/, '');
    });
  }

  var DIVS = ['AFC East', 'AFC North', 'AFC South', 'AFC West', 'NFC East', 'NFC North', 'NFC South', 'NFC West'];

  // Builds the Rest of season section after This week, adds its nav link, and renumbers the kickers.
  function renderRest() {
    var R = D.ml && D.ml.rest;
    if (!R || !R.teams.length) return;
    var body = h('div', { id: 'rest-body', class: 'body' });
    var sec = h('section', { class: 'sec', id: 'rest', 'aria-labelledby': 'rest-h', tabindex: '-1' },
      h('div', { class: 'wrap' },
        h('p', { class: 'kicker', text: 'Rest of season' }),
        h('h2', { class: 'headline', id: 'rest-h', text: R.headline || 'Rest of season' }),
        h('p', { class: 'deck', text: 'Projected final wins: wins so far plus the model\'s win chance in every game left, with Elo\'s projection beside it. Open a team for its remaining games.' }),
        body));
    insertSection(sec, 'week', 'Rest of season');

    var MAXW = 17;
    function hasRange(r) { return r.wins_p10 != null && r.wins_p90 != null; }
    function rangeText(r) { return winsText(r.wins_p10) + ' to ' + winsText(r.wins_p90); }
    function x(v) { return (Math.max(0, Math.min(MAXW, v)) / MAXW * 100).toFixed(2) + '%'; }
    function row(r) {
      var t = byTeam[r.team], base = r.w + r.t / 2, record = rec(r.w, r.l, r.t);
      var summary = h('summary', {},
        h('span', { class: 'sr', text: t.name + ', ' + record + '. Projected ' + f1(r.proj_model) + ' wins by the model, ' + f1(r.proj_elo) + ' by Elo' + (hasRange(r) ? ', likely range ' + rangeText(r) + ' wins' : '') + ', ' + plural(r.remaining, 'game', 'games') + ' left. Show remaining games.' }),
        h('span', { class: 'abbr', 'aria-hidden': 'true', text: r.team }),
        h('span', { class: 'mid', 'aria-hidden': 'true' },
          h('span', { class: 'wb' },
            h('i', { class: 'so', style: 'width:' + x(base) }),
            h('i', { class: 'pj', style: 'left:calc(' + x(base) + ' + 2px);width:max(0px, calc(' + x(r.proj_model - base) + ' - 2px))' }),
            hasRange(r) ? h('i', { class: 'rg', style: 'left:' + x(r.wins_p10) + ';width:' + x(r.wins_p90 - r.wins_p10) }) : null,
            h('i', { class: 'et', style: 'left:' + x(r.proj_elo) })),
          h('span', { class: 'meta', text: record + ', ' + r.remaining + ' left' })),
        h('span', { class: 'pv', 'aria-hidden': 'true' }, h('b', { text: f1(r.proj_model) }), h('small', { text: 'Elo ' + f1(r.proj_elo) })));
      bindTip(summary, function () {
        return { title: t.name, rows: [
          { value: f1(r.proj_model), label: 'model projection', cls: 's3' },
          { value: f1(r.proj_elo), label: 'Elo projection', cls: 's1' },
          hasRange(r) ? { value: rangeText(r), label: 'likely range (10th to 90th percentile)', cls: 'rg' } : null,
          { value: record, label: 'so far', cls: 'sm' },
          { value: String(r.remaining), label: r.remaining === 1 ? 'game left' : 'games left' }] };
      });
      var games = h('div', { class: 'rs-games' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Week' }), h('th', { scope: 'col', text: 'Game' }),
          h('th', { scope: 'col', class: 'r', text: 'Model' }), h('th', { scope: 'col', class: 'r', text: 'Elo' }))),
        h('tbody', {}, r.games.map(function (g) {
          return h('tr', {}, h('td', { class: 'n', text: String(g.week) }),
            h('td', { text: (g.home ? 'vs ' : 'at ') + g.opp + (g.neutral ? ' (neutral)' : '') }),
            h('td', { class: 'r n', text: pct(g.p_model) }), h('td', { class: 'r n', text: pct(g.p_elo) }));
        }))));
      return h('details', { class: 'rs' }, summary, games);
    }

    var ranged = R.teams.some(hasRange);
    body.appendChild(legend([['sm blk', 'Wins so far'], ['s3 blk', 'Model projection'], ['tk', 'Elo projection']].concat(ranged ? [['rg', 'Likely range, 10th to 90th percentile']] : [])));
    var grid = h('div', { class: 'divs' });
    DIVS.forEach(function (d) {
      var teams = R.teams.filter(function (r) { return byTeam[r.team] && byTeam[r.team].div === d; });
      teams.sort(function (a, b) { return b.proj_model - a.proj_model || (a.team < b.team ? -1 : 1); });
      grid.appendChild(h('div', { class: 'dv' }, h('h3', {}, d, h('span', { text: 'Projected wins' })), teams.map(row)));
    });
    body.appendChild(grid);
    body.appendChild(h('p', { class: 'fnote', text: R.note + ' Bars run from 0 to 17 wins. Model numbers are from the run on ' + runTime(D.ml.last_run_utc) + '.' + (ranged ? ' The thin line is the likely range of final wins from the season simulation: in 8 of 10 simulated seasons the team lands inside it.' : '') }));
    body.appendChild(tableView('View projected wins as a table', [{ t: 'Team' }, { t: 'Record', r: 1, n: 1 }, { t: 'Left', r: 1, n: 1 }, { t: 'Model', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }].concat(ranged ? [{ t: 'Likely range', r: 1, n: 1 }] : []), function () {
      return R.teams.map(function (r) { return [byTeam[r.team].name, rec(r.w, r.l, r.t), String(r.remaining), f1(r.proj_model), f1(r.proj_elo)].concat(ranged ? [hasRange(r) ? rangeText(r) : ''] : []); });
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
    return { v: v, text: v === 0 ? '0' : (v > 0 ? '+' : '-') + Math.abs(v), sr: v === 0 ? 'no change' : (v > 0 ? 'up ' : 'down ') + plural(Math.abs(v), 'point', 'points') };
  }

  function shortRun(iso) {
    return new Date(iso).toLocaleDateString('en-US', { timeZone: 'America/New_York', weekday: 'short', month: 'short', day: 'numeric' });
  }

  function renderOdds() {
    var O = D.ml && D.ml.playoff_odds;
    if (!O || !O.conferences) return;
    var hasBase = !!O.baseline_run_utc;
    var body = h('div', { id: 'odds-body', class: 'body' });
    var sec = h('section', { class: 'sec', id: 'odds', 'aria-labelledby': 'odds-h', tabindex: '-1' },
      h('div', { class: 'wrap' },
        h('p', { class: 'kicker', text: 'Playoff odds' }),
        h('h2', { class: 'headline', id: 'odds-h', text: O.headline || 'Playoff odds' }),
        h('p', { class: 'deck', text: 'Each team\'s chances from ' + O.n_sims.toLocaleString('en-US') + ' simulations of the rest of the season, seeded with the NFL\'s tiebreakers. ' +
          (hasBase ? 'Change is in percentage points since the run on ' + runTime(O.baseline_run_utc) + '.' : 'Change since last week appears after the next Wednesday run.') + ' Select a column to sort.' }),
        body));
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
    body.appendChild(h('p', { class: 'fnote' },
      'Each simulated season gives every team one random boost or drag for the rest of the year (a spread of ' + f1(O.tau_rest) + ' points), because ratings are estimates and teams change. ' +
      'Then every game left is played from the model\'s range of final scores, ties included; the tiebreakers seed the field, and the higher seed hosts each playoff game. Last run: ' + runTime(O.run_at_utc) + '. ',
      h('a', { href: O.history_url, text: 'Every run is kept in a public file' }), '.'));
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

  function atsText(a) { return a.w + '-' + a.l + (a.push ? '-' + a.push : ''); }

  // Live record block for the scorecard: empty-state copy until the first live week is scored.
  function liveBlock() {
    var ml = D.ml, L = ml.live.final, W = ml.live.wednesday;
    var box = h('div', { class: 'livebox' }, h('h3', { text: 'Live ' + ml.season + ' record' }));
    var start = 'The live record starts with Week ' + ml.live_from_week + (ml.live_from_date ? ', ' + dayLong(ml.live_from_date) : '') + '.';
    if (!L.n) {
      box.appendChild(h('p', { class: 'sub', text: start + ' Each prediction is written to a public ledger before kickoff, and only those count. Weeks 1 to ' + (ml.live_from_week - 1) + ' were never predicted in advance, so they are left out. No live games have been scored yet.' }));
    } else {
      var mk = L.market_games;
      box.appendChild(h('p', { class: 'sub', text: start + ' Scored on the last prediction made before each kickoff, ' + plural(L.n, 'game', 'games') + ' so far. That is a small sample, so treat the gaps as noise for now.' }));
      box.appendChild(h('div', { class: 'score' },
        h('div', {}, h('h3', { text: 'Brier score, lower is better' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: (mk ? mk.model.brier : L.model.brier).toFixed(4) }), 'Model'),
            h('span', {}, h('b', { class: 'num', text: (mk ? mk.elo.brier : L.elo.brier).toFixed(4) }), 'Elo v2'),
            mk ? h('span', {}, h('b', { class: 'num', text: mk.market.brier.toFixed(4) }), 'Vegas market') : null)),
        h('div', {}, h('h3', { text: 'Correct picks' }),
          h('div', { class: 'row' },
            h('span', {}, h('b', { class: 'num', text: pct(L.model.accuracy, 1) }), 'Model'),
            h('span', {}, h('b', { class: 'num', text: pct(L.elo.accuracy, 1) }), 'Elo v2'),
            mk ? h('span', {}, h('b', { class: 'num', text: pct(mk.market.accuracy, 1) }), 'Vegas') : null)),
        h('div', {}, h('h3', { text: 'Against the Vegas spread' }),
          h('div', { class: 'row' },
            L.ats_model && L.ats_model.games ? h('span', {}, h('b', { class: 'num', text: atsText(L.ats_model) }), 'Model') : null,
            h('span', {}, h('b', { class: 'num', text: atsText(L.ats_elo) }), 'Elo v2')))));
      var notes = [];
      if (mk && mk.n !== L.n) notes.push('Brier scores use the ' + mk.n + ' games with a moneyline; correct picks for the model and Elo use all ' + L.n + '.');
      if (W.n) notes.push('Wednesday predictions alone: model ' + W.model.brier.toFixed(4) + ', Elo ' + W.elo.brier.toFixed(4) + ' Brier on ' + plural(W.n, 'game', 'games') + '. The gap to the final numbers is what late quarterback news was worth.');
      var am = L.ats_model;
      notes.push(am && am.games
        ? 'Against the spread: wins and losses' + (am.push || L.ats_elo.push ? ' and pushes' : '') + ' when taking the side the forecast\'s line favors over the Vegas line. The model\'s line exists for ' + plural(am.games, 'scored game', 'scored games') + (am.games < L.n ? ' (predictions made before its score model went live have none)' : '') + '.'
        : 'The model\'s point spread began after the first live predictions, so only Elo has a record against the spread so far.');
      box.appendChild(h('p', { class: 'fnote', text: notes.join(' ') }));
      // Shadow model: logged before kickoff in its own ledger, scored on the same games; hidden until one is scored.
      var S = ml.shadow;
      if (S && S.n) box.appendChild(h('p', { class: 'fnote' }, 'Shadow model (Kalman, testing for 2027): Brier ' + S.shadow.brier.toFixed(4) + ' against the model\'s ' + S.model.brier.toFixed(4) + ' on the same ' + plural(S.n, 'game', 'games') + '. It does not change the model\'s picks. ', h('a', { href: S.ledger_url, text: 'See its ledger' }), '.'));
    }
    box.appendChild(h('p', { class: 'fnote' }, 'Last model run: ' + runTime(ml.last_run_utc) + '. ', h('a', { href: ml.ledger_url, text: 'See the prediction ledger' }), '.'));
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
      var notes = 'The horizontal line is the 1500 league average. Regular season only. Season-end ratings exclude ' + D.meta.season + (D.meta.season_partial ? ', which is still in progress.' : '.');
      if (t.first_season > 1970) notes += ' ' + t.name + ' first played in ' + t.first_season + ' and started at ' + D.meta.config.expansion_start + '.';
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
      var left = h('div', {}, h('div', { class: 'lk' }, head, list), h('p', { class: 'fnote', text: D.luck.note + ' Both seasons share one scale.' }));
      body.appendChild(h('div', { class: 'split' }, left, luckyBox(s)));
      left.appendChild(tableView('View luck as a table', [{ t: 'Team' }, { t: 'Games', r: 1, n: 1 }, { t: 'Actual wins', r: 1, n: 1 }, { t: 'Expected wins', r: 1, n: 1 }, { t: 'Luck', r: 1, n: 1 }], function () {
        return s.teams.map(function (r) { return [r.team, String(r.games), String(r.actual), f1(r.expected), signed(r.luck)]; });
      }));
    }

    drawToggle();
    drawBars();
  }

  // ------------------------------------------------------------------ history: tapestry + parity

  // Seven discrete bins, tokens only. Below 1500: ink, muted, line. Near 1500: canvas. Above: tint, secondary, primary.
  // Lightness steps the same way on both sides, so the scale reads without color.
  var HEAT = [
    { max: 1350, fill: 'var(--ink)', label: 'Below 1350' },
    { max: 1425, fill: 'var(--muted)', label: '1350 to 1425' },
    { max: 1475, fill: 'var(--line)', label: '1425 to 1475' },
    { max: 1525, fill: 'var(--canvas)', label: '1475 to 1525' },
    { max: 1575, fill: 'var(--pri-tint)', label: '1525 to 1575' },
    { max: 1650, fill: 'var(--secondary)', label: '1575 to 1650' },
    { max: Infinity, fill: 'var(--primary)', label: 'Above 1650' }
  ];
  function heatBin(v) { for (var i = 0; i < HEAT.length; i++) if (v < HEAT[i].max) return HEAT[i]; return HEAT[HEAT.length - 1]; }
  function heatFill(v) { return heatBin(v).fill; }

  function renderHistory() {
    var body = $('#history-body'), T = D.tapestry;
    clear(body);
    body.removeAttribute('aria-busy');

    var scale = h('div', { class: 'scale', 'aria-label': 'Color scale in seven steps: dark gray below 1350, lighter gray to 1475, white from 1475 to 1525, then pink, orange and red above 1525' }, h('span', { text: 'Rating' }));
    HEAT.forEach(function (b) {
      scale.appendChild(h('span', { class: 'sw' }, h('i', { style: 'background:' + b.fill }), b.label));
    });
    var wrap = h('div', { class: 'heat-wrap' });
    var heat = h('div', { class: 'heat' });
    wrap.appendChild(heat);
    body.appendChild(well(wrap));
    body.appendChild(scale);
    body.appendChild(h('p', { class: 'fnote', text: 'Each cell is a franchise\'s rating at the end of that regular season. Empty cells are seasons before the franchise existed.' + (T.partial_season ? ' The ' + T.partial_season + ' column is the rating so far.' : '') }));

    var nT = T.teams.length, nS = T.seasons.length;
    var CH = 14, LBL = 38, TOP = 22, GAP_DIV = 6, GAP_CONF = 12;
    var rowY = [], totalH = 0;
    function layout() {
      rowY = []; var y = TOP;
      T.teams.forEach(function (code, i) {
        if (i > 0) y += (i % 16 === 0) ? GAP_CONF : (i % 4 === 0 ? GAP_DIV : 0);
        rowY.push(y); y += CH;
      });
      totalH = y + 4;
    }

    function draw() {
      clear(heat);
      var avail = wrap.clientWidth - LBL;
      var cw = Math.max(13, Math.floor(avail / nS));
      CH = Math.min(24, Math.max(14, Math.round(cw * 0.6)));   // cells grow with the width
      layout();
      var W = LBL + cw * nS;
      var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + totalH, width: W, height: totalH, role: 'img', tabindex: 0,
        'aria-label': 'Heatmap of end-of-season Elo ratings, ' + nT + ' franchises by ' + nS + ' seasons from ' + T.seasons[0] + ' to ' + T.seasons[nS - 1] + '. Use the arrow keys to move between cells.' }, heat);
      T.seasons.forEach(function (s, c) {
        if (s % 10 === 0 || s === T.seasons[nS - 1]) sv('text', { class: 'tick a-mid', x: LBL + c * cw + cw / 2, y: 12, text: s === T.partial_season ? s + '*' : String(s) }, svg);
      });
      T.teams.forEach(function (code, r) {
        sv('text', { class: 'tick a-end', x: LBL - 8, y: rowY[r] + CH - 3, text: code }, svg);
        T.values[r].forEach(function (v, c) {
          if (v == null) return;
          sv('rect', { class: 'cell', x: LBL + c * cw, y: rowY[r], width: cw - 1, height: CH - 1, style: 'fill:' + heatFill(v) }, svg);
        });
      });
      var cursor = sv('rect', { class: 'cursor', width: cw - 1, height: CH - 1, visibility: 'hidden' }, svg);
      var pos = { r: 0, c: nS - 1 };

      function show(r, c, cx, cy) {
        pos = { r: r, c: c };
        var v = T.values[r][c], code = T.teams[r], season = T.seasons[c];
        cursor.setAttribute('x', LBL + c * cw); cursor.setAttribute('y', rowY[r]); cursor.setAttribute('visibility', 'visible');
        if (cx == null) { var b = svg.getBoundingClientRect(); cx = b.left + (LBL + c * cw + cw) * (b.width / W); cy = b.top + rowY[r] * (b.height / totalH); }
        tipShow(cx, cy, name(code) + ', ' + season + (season === T.partial_season ? ' (so far)' : ''),
          v == null ? [{ value: 'No team yet', label: '' }] : [{ value: f1(v), label: signed(v - 1500) + ' vs 1500' }]);
      }
      function hide() { cursor.setAttribute('visibility', 'hidden'); tipHide(); }

      svg.addEventListener('pointermove', function (e) {
        var b = svg.getBoundingClientRect(), px = (e.clientX - b.left) * (W / b.width), py = (e.clientY - b.top) * (totalH / b.height);
        var c = Math.floor((px - LBL) / cw), r = -1;
        for (var i = 0; i < nT; i++) if (py >= rowY[i] && py < rowY[i] + CH) { r = i; break; }
        if (c < 0 || c >= nS || r < 0) return hide();
        show(r, c, e.clientX, e.clientY);
      });
      svg.addEventListener('pointerleave', hide);
      svg.addEventListener('focus', function () { if (!svg.matches(':hover')) show(pos.r, pos.c); });
      svg.addEventListener('blur', hide);
      svg.addEventListener('keydown', function (e) {
        var r = pos.r, c = pos.c;
        if (e.key === 'ArrowLeft') c = Math.max(0, c - 1);
        else if (e.key === 'ArrowRight') c = Math.min(nS - 1, c + 1);
        else if (e.key === 'ArrowUp') r = Math.max(0, r - 1);
        else if (e.key === 'ArrowDown') r = Math.min(nT - 1, r + 1);
        else if (e.key === 'Escape') return hide();
        else return;
        e.preventDefault();
        show(r, c);
      });
    }
    watch(wrap, draw);

    // Parity: spread of end-of-season ratings. Lower means a tighter league.
    body.appendChild(h('h3', { class: 'sub-h', text: 'League parity by season' }));
    body.appendChild(h('p', { class: 'deck', style: 'margin-top:-6px;margin-bottom:12px', text: 'Standard deviation of end-of-season ratings. A lower line means the league was more bunched together.' }));
    var pHost = h('div', { class: 'chart' });
    body.appendChild(well(pHost));
    var pts = T.seasons.map(function (s, i) { return { x: s, y: T.parity[i], count: T.teams_count[i] }; }).filter(function (p) { return p.x !== T.partial_season; });
    var pLo = 60, pHi = 150;
    watch(pHost, function () {
      lineChart(pHost, {
        series: [{ id: 'parity', label: 'Rating spread', cls: 's1', pts: pts }],
        x: [pts[0].x, pts[pts.length - 1].x], y: [pLo, pHi],
        height: function (w) { return w < 560 ? 220 : 280; },
        xTicks: YEAR_TICKS.concat([2025]), yTicks: ticksFor(pLo, pHi, 4), xFmt: String, yFmt: String,
        aria: 'Line chart of the spread of end-of-season Elo ratings by season, from ' + f1(pts[0].y) + ' in ' + pts[0].x + ' to ' + f1(pts[pts.length - 1].y) + ' in ' + pts[pts.length - 1].x + '.',
        tip: function (x, rows) { var p = rows[0].p; return { title: String(x), rows: [{ value: f1(p.y), label: 'rating spread', cls: 's1' }, { value: String(p.count), label: 'teams' }] }; }
      });
    });
    body.appendChild(h('p', { class: 'fnote', text: 'Expansion years read high because new teams start at ' + D.meta.config.expansion_start + ', well below the average.' }));
    body.appendChild(tableView('View parity as a table', [{ t: 'Season' }, { t: 'Rating spread', r: 1, n: 1 }, { t: 'Teams', r: 1, n: 1 }], function () {
      return pts.slice().reverse().map(function (p) { return [String(p.x), f1(p.y), String(p.count)]; });
    }));
  }

  // ------------------------------------------------------------------ records

  function gameCell(g) {
    var awayWon = g.winner === g.away;
    return h('span', {},
      h('span', { class: awayWon ? 'w' : 'l', text: g.away }), ' ', h('span', { class: awayWon ? 'w-sc' : 'l-sc', text: String(g.away_score) }),
      ' at ',
      h('span', { class: awayWon ? 'l' : 'w', text: g.home }), ' ', h('span', { class: awayWon ? 'l-sc' : 'w-sc', text: String(g.home_score) }));
  }

  function recTable(title, sub, cols, rows) {
    var thead = h('thead', {}, h('tr', {}, cols.map(function (c) { return h('th', { class: c.r ? 'r' : '', scope: 'col', text: c.t }); })));
    var tbody = h('tbody', {}, rows.map(function (r) {
      return h('tr', {}, r.map(function (v, i) { return h('td', { class: (cols[i].r ? 'r ' : '') + (cols[i].n ? 'n' : '') }, v); }));
    }));
    return h('div', {}, h('h3', { text: title }), h('p', { class: 'sub', text: sub }), h('div', { class: 'tbl-wrap' }, h('table', {}, thead, tbody)));
  }

  function renderRecords() {
    var R = D.records, body = $('#records-body');
    clear(body);
    body.removeAttribute('aria-busy');
    var seasonCols = [{ t: 'Team' }, { t: 'Season', n: 1 }, { t: 'Rating', r: 1, n: 1 }, { t: 'Record', r: 1, n: 1 }];
    var seasonRow = function (s) { return [h('b', { text: s.team }), String(s.season), f1(s.rating), rec(s.w, s.l, s.t)]; };
    body.appendChild(h('div', { class: 'rec4' },
      recTable('Biggest upsets', 'Lowest pre-game win chance for the team that won. Winner in bold.',
        [{ t: 'Date', n: 1 }, { t: 'Game', n: 0 }, { t: 'Win chance', r: 1, n: 1 }],
        R.upsets.map(function (g) { return [g.date, gameCell(g), pct(g.p_win, 1)]; })),
      recTable('Biggest rating swings', 'Rating points the winner gained in one game.',
        [{ t: 'Date', n: 1 }, { t: 'Game', n: 0 }, { t: 'Gain', r: 1, n: 1 }],
        R.swings.map(function (g) { return [g.date, gameCell(g), '+' + f1(g.gain)]; })),
      recTable('Best team-seasons', 'Highest end-of-season rating.', seasonCols, R.best_seasons.map(seasonRow)),
      recTable('Worst team-seasons', 'Lowest end-of-season rating.', seasonCols, R.worst_seasons.map(seasonRow))));
    body.appendChild(h('p', { class: 'fnote', text: 'Upsets include playoff games; the other tables use regular-season games, because playoff games do not change ratings.' }));
  }

  // ------------------------------------------------------------------ scorecard

  function renderScorecard() {
    var body = $('#scorecard-body'), S = D.scorecard.seasons, sc = D.meta.scorecard, cfg = D.meta.config;
    clear(body);
    body.removeAttribute('aria-busy');
    var span = sc.test_seasons[0] + '-' + sc.test_seasons[1];

    body.appendChild(h('div', { class: 'score' },
      h('div', {}, h('h3', { text: 'Brier score, lower is better (' + span + ')' }),
        h('div', { class: 'row' },
          h('span', {}, h('b', { class: 'num', text: sc.elo_brier.toFixed(4) }), 'Elo v2'),
          h('span', {}, h('b', { class: 'num', text: sc.market_brier.toFixed(4) }), 'Vegas market'),
          sc.legacy_brier ? h('span', {}, h('b', { class: 'num', text: sc.legacy_brier.toFixed(4) }), '2025 model') : null)),
      h('div', {}, h('h3', { text: 'Correct picks' }),
        h('div', { class: 'row' },
          h('span', {}, h('b', { class: 'num', text: pct(sc.elo_accuracy_market_games, 1) }), 'Elo v2'),
          h('span', {}, h('b', { class: 'num', text: pct(sc.market_accuracy, 1) }), 'Vegas favorite'))),
      h('div', {}, h('h3', { text: 'Gap to the market' }),
        h('div', { class: 'row' }, h('span', {}, h('b', { class: 'num', text: '+' + sc.brier_gap_vs_market.toFixed(4) }), 'Brier, plus or minus ' + sc.brier_gap_se.toFixed(4))))));
    if (D.ml) body.appendChild(liveBlock());

    var pair = h('div', { class: 'pair' });
    var brHost = h('div', { class: 'chart' }), hfaHost = h('div', { class: 'chart' });
    pair.appendChild(h('div', {}, h('h3', { text: 'Brier score by season' }), h('p', { class: 'sub', text: 'Elo against the Vegas moneyline, vig removed. Lower is better.' }),
      legend([['s1', 'Elo v2'], ['s2', 'Vegas market']]), well(brHost),
      h('p', { class: 'fnote', text: 'No betting lines in the data for 2008; 2006 is partial.' }),
      tableView('View Brier scores as a table', [{ t: 'Season' }, { t: 'Games', r: 1, n: 1 }, { t: 'Elo', r: 1, n: 1 }, { t: 'Market', r: 1, n: 1 }], function () {
        return S.filter(function (r) { return r.season >= firstMkt; }).reverse().map(function (r) {
          return r.market_brier == null ? [String(r.season), '-', '-', '-'] : [String(r.season), String(r.n_market), r.elo_brier.toFixed(4), r.market_brier.toFixed(4)];
        });
      })));
    pair.appendChild(h('div', {}, h('h3', { text: 'Home-field edge by season' }), h('p', { class: 'sub', text: 'Chance the home team wins between equal teams, from Elo\'s learned edge, next to how often home teams actually won.' }),
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
          { id: 'elo', label: 'Elo v2', short: 'Elo', cls: 's1', pts: mk.map(function (r) { return { x: r.season, y: r.elo_brier }; }) },
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

    body.appendChild(h('div', { class: 'method' },
      h('div', {},
        h('h3', { text: 'How it works' }),
        h('p', {}, h('b', { text: 'Ratings. ' }), 'Every team starts at 1500, the league average. After each game the winner takes points from the loser. The bigger the surprise and the bigger the margin, the more points move. The base size is K = ' + cfg.k + '.'),
        h('p', {}, h('b', { text: 'Offseason. ' }), 'Between seasons every rating moves ' + Math.round(cfg.lam * 100) + '% of the way back to 1500 (lambda = ' + cfg.lam.toFixed(2) + ') because rosters turn over.'),
        h('p', {}, h('b', { text: 'Home field. ' }), 'The home team gets a rating bonus before each prediction, none at neutral sites. The bonus is learned: it started at ' + cfg.hfa_init + ' points in 1970, moves up or down as home teams win more or less than expected, and is ' + f1(D.meta.hfa_pts) + ' now. Elo spread is the rating gap divided by 25.'),
        h('p', {}, h('b', { text: 'Data. ' }), 'Game results from 1999 on come from nflverse. Results for 1970 to 1998, including playoffs, come from the FiveThirtyEight NFL Elo game file. Both are CC BY 4.0.'),
        h('p', {}, h('b', { text: 'New teams. ' }), 'Franchises that join after 1970 start at ' + cfg.expansion_start + '. The model was tuned on ' + sc.tune_seasons[0] + '-' + sc.tune_seasons[1] + ' and scored on ' + span + '.'),
        h('p', {}, h('b', { text: 'Limits. ' }), 'The Vegas line is still better: a Brier gap of ' + sc.brier_gap_vs_market.toFixed(4) + '. Elo does not know about injuries, starting quarterbacks, or weather.'),
        D.ml ? h('p', {}, h('b', { text: 'Model. ' }), 'The second forecast is a logistic regression on three inputs: Elo\'s win chance, each team\'s play-by-play efficiency adjusted for its opponents, and how much better or worse this week\'s starting quarterback is than the passing the team\'s recent numbers reflect. It is fit on the 2001 to ' + (D.ml.season - 1) + ' regular seasons and never sees betting lines. On ' + D.ml.holdout.seasons[0] + ' to ' + D.ml.holdout.seasons[1] + ' games it had never seen, it scored a Brier of ' + D.ml.holdout.model_brier.toFixed(4) + ' against ' + D.ml.holdout.elo_brier.toFixed(4) + ' for Elo and ' + D.ml.holdout.market_brier.toFixed(4) + ' for Vegas. Every prediction is added to a ', h('a', { href: D.ml.ledger_url, text: 'public ledger' }), ' before kickoff, and the git history shows when. Last run: ' + runTime(D.ml.last_run_utc) + '.') : null,
        D.ml && D.ml.playoff_odds ? h('p', {}, h('b', { text: 'Playoff odds. ' }), 'A second model predicts the final score margin, not just the winner, using the same three inputs. Its average is the model\'s line, and its spread of outcomes favors the scores football produces most, such as 3 and 7 points. The season is then played out ' + D.ml.playoff_odds.n_sims.toLocaleString('en-US') + ' times. In each run every team gets one random boost or drag that lasts the rest of the year, because a rating is an estimate and teams get better or worse. The NFL\'s own tiebreakers set the seeds; run on every real season from 2006 to 2025, they reproduce every actual playoff field and seeding. The size of that boost or drag was set by replaying weeks 4, 8, 12 and 16 of the 2006 to 2019 seasons; without it, teams given 80 to 90% made the playoffs only about 74% of the time. Checked once on 2020 to 2025, seasons it never saw, the odds matched how often teams really made the playoffs. Early in the season they are rougher: odds from week 4 miss by more than odds from week 12. The model\'s line is no worse than Elo\'s and was closer in 5 of those 6 seasons, but its edge, about 0.06 points a game, is too small to call proven. Vegas lines are still closer than both.') : null),
      h('div', {},
        h('h3', { text: 'Links' }),
        h('ul', {},
          h('li', {}, h('a', { href: D.meta.repo, text: 'Code and method on GitHub' }), h('small', { text: 'github.com/walk-the-program/NFLELO' })),
          h('li', {}, h('a', { href: 'https://nflverse.nflverse.com/', text: 'nflverse' }), h('small', { text: 'Schedules, scores, and betting lines, 1999 on. CC BY 4.0.' })),
          h('li', {}, h('a', { href: 'https://github.com/fivethirtyeight/data', text: 'FiveThirtyEight' }), h('small', { text: 'NFL game results 1970 to 1998: FiveThirtyEight, CC BY 4.0.' })),
          h('li', {}, h('a', { href: 'data/ladder.json', text: 'Ladder data (JSON)' }), h('small', { text: 'Rebuilt every Wednesday' })),
          D.ml ? h('li', {}, h('a', { href: D.ml.ledger_url, text: 'Prediction ledger (CSV)' }), h('small', { text: 'Every model prediction, written before kickoff' })) : null))));
  }

  // ------------------------------------------------------------------ boot

  function fail(err) {
    console.error(err);
    ['ladder', 'week', 'explorer', 'luck', 'history', 'records', 'scorecard'].forEach(function (id) {
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
  var mlFile = N.getJSON('data/ml.json', true);

  Promise.all(FILES.map(function (f) { return N.getJSON('data/' + f + '.json'); }).concat([mlFile])).then(function (all) {
    FILES.forEach(function (f, i) { D[f] = all[i]; });
    D.ml = all[FILES.length];
    D.ladder.teams.forEach(function (t) { byTeam[t.team] = t; });
    D.history.teams.forEach(function (t) { histBy[t.team] = t; });
    renderRest();
    renderOdds();
    initReveal();
    renderHeader();
    renderLadder();
    renderWeek();
    renderHero();
    renderExplorer();
    renderLuck();
    renderHistory();
    renderRecords();
    renderScorecard();
    setHeadline('ladder-h', D.headlines.ladder);
    setHeadline('week-h', (mlWeek() && mlWeek().headline) || D.headlines.week);
    setHeadline('history-h', D.headlines.history);
    setHeadline('records-h', D.headlines.records);
    setHeadline('scorecard-h', D.headlines.scorecard);
    N.initNav();
  }).catch(fail);
})();
