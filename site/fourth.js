/* NFLELO fourth-down page. Reads ./data/fourth.json (written by scripts/export_ml_pages.py): the decision chart,
   the 2026 teams and plays, and the status of the forward test. Every number here is model-estimated; the page
   says so wherever numbers appear. No dependencies; helpers come from common.js. */
(function () {
  'use strict';

  var N = window.NFL;
  var $ = N.$, h = N.h, sv = N.sv, clear = N.clear, well = N.well;
  var tipShow = N.tipShow, tipHide = N.tipHide, watch = N.watch;
  var signed = N.signed, commas = N.commas, pct = N.pct;
  var setText = N.setText, done = N.done;

  var F = null;                 // fourth.json
  var ui = {};
  var presetKey = null;
  var P = {};                   // preset lookups: P[key] = {byCell, rows}
  var cursor = { yl: 40, d: 3 };
  var teamSort = { key: 'wp_lost_total', dir: 'desc' };
  var TOSS = 0.02, LEAN = 0.04;   // win-probability margins: under TOSS is a toss-up, under LEAN a lean
  var teamFilter = '', playMode = 'all', playLimit = 25;

  var OPT = { go: 'Go for it', fg: 'Field goal', punt: 'Punt' };
  var OPT_SHORT = { go: 'Go', fg: 'Kick', punt: 'Punt' };
  var CHOICE = { go: 'Go', fg: 'Field goal', punt: 'Punt' };
  function me() { return h('span', { class: 'me', text: 'model-estimated' }); }
  function cat(k) { return h('span', { class: 'sc', style: 'background:var(--cat-' + k + ')' }); }
  function pts(x) { return (x * 100).toFixed(1); }

  // ------------------------------------------------------------------ status, hero

  function renderHero() {
    var L = F.live, s = L.summary, a = F.headline.holdout_audit_v1;
    var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], run = F.status.earliest_run;
    setText('hero-line', 'First model failed its calibration test. Revised model under test until ' + MONTHS[Number(run.slice(5, 7)) - 1] + ' ' + run.slice(0, 4) + '.');
    setText('hero-deck', 'Go, kick or punt: the best option by yard line and distance.');
    N.heroTiles([
      { label: 'Matched the model', big: pct(s.matched_share, 0), bar: s.matched_share, barText: 'Through week ' + L.updated_through.week, sub: 'of ' + commas(s.fourth_downs) + ' fourth downs. Model-estimated.' },
      { label: 'Close calls', big: pct(s.tossup_share, 0), sub: 'toss-ups, margin under ' + F.tossup_rule.margin_below + '. Model-estimated.' },
      { label: 'Win probability given up', big: s.wp_lost_per_team_game.toFixed(3), sub: 'wins per team-game, ' + s.wp_lost_total.toFixed(1) + ' total. Model-estimated.' },
      { label: '2020 to 2025 audit', big: a.wp_lost_per_team_game.toFixed(3), sub: 'wins per team-game, ' + commas(a.fourth_downs) + ' fourth downs; ' + a.wp_lost_by_choice.punt.toFixed(0) + ' of ' + a.wp_lost_total.toFixed(0) + ' wins from punts. Model-estimated.' }
    ]);
    N.licenseUI(F.license);
  }

  // ------------------------------------------------------------------ decision chart

  function buildPresets() {
    F.chart.presets.forEach(function (p) {
      var byCell = {}, n = p.yardline_100.length;
      for (var i = 0; i < n; i++) byCell[p.yardline_100[i] + '|' + p.ydstogo[i]] = i;
      P[p.key] = { p: p, byCell: byCell, n: n };
    });
  }
  function cellAt(yl, d) { var q = P[presetKey]; var i = q.byCell[yl + '|' + d]; return i == null ? null : q.p; }
  function rec(yl, d) {
    var q = P[presetKey], i = q.byCell[yl + '|' + d];
    if (i == null) return null;
    var p = q.p;
    return { i: i, yl: yl, d: d, best: p.best[i], margin: p.margin[i], go: p.wp_go[i], fg: p.wp_fg[i], punt: p.wp_punt[i], pc: p.p_conv[i], pf: p.p_fg[i], close: p.close_call[i] };
  }
  function second(r) {
    var o = ['go', 'fg', 'punt'].filter(function (k) { return k !== r.best; });
    return o.sort(function (a, b) { return r[b] - r[a]; })[0];
  }
  function readout(r) {
    return h('span', {}, h('b', { text: N.ordinal(4) + ' and ' + r.d + ' at ' + N.fieldLabel(r.yl) + ': ' }),
      'best option is ' + OPT[r.best].toLowerCase() + ' at ' + pct(r[r.best]) + ' win probability, ahead of ' + OPT[second(r)].toLowerCase() + ' by ' + pts(r.margin) + ' percentage points' + (r.close ? ' (a toss-up)' : '') + '. Go ' + pct(r.go) + ', field goal ' + pct(r.fg) + ', punt ' + pct(r.punt) + '. Chance to convert if going: ' + pct(r.pc, 0) + '; chance to make the kick: ' + pct(r.pf, 0) + '. ');
  }

  function buildChart() {
    var body = done('chart-body'), C = F.chart;
    TOSS = F.tossup_rule.margin_below;
    clear(body);
    buildPresets();
    presetKey = C.presets[0].key;
    var SHORT = { tied_q2: 'Tied, Q2', down4_early_q4: 'Down 4, early Q4', up3_late_q4: 'Up 3, late Q4' };
    ui.preset = N.seg('Game state', C.presets.map(function (p) { return { value: p.key, label: SHORT[p.key] || p.label, title: p.label + ' (' + p.state.averaged_over + ')' }; }),
      presetKey, function (v) { presetKey = v; drawHeat(); ui.tv.refresh(); updateState(); });
    ui.state = h('p', { class: 'fnote', style: 'margin-top:12px' });
    ui.detail = h('div', { class: 'pdetail fx-detail', 'aria-live': 'polite', text: 'Hover, tap or use arrow keys on a square.' });
    ui.heat = h('div', { class: 'chart hm-wrap' });
    ui.tv = N.tableViewLive('View this chart as a table',
      function () { return [{ t: 'Yard line' }, { t: 'To go', r: 1, n: 1 }, { t: 'Best' }, { t: 'Margin (pp)', r: 1, n: 1 }, { t: 'Go win prob.', r: 1, n: 1 }, { t: 'Field goal win prob.', r: 1, n: 1 }, { t: 'Punt win prob.', r: 1, n: 1 }, { t: 'P(convert)', r: 1, n: 1 }, { t: 'P(make kick)', r: 1, n: 1 }, { t: 'Toss-up' }]; },
      function () {
        var q = P[presetKey], out = [];
        for (var i = 0; i < q.n; i++) {
          var p = q.p;
          out.push([N.fieldLabel(p.yardline_100[i]), String(p.ydstogo[i]), OPT[p.best[i]], pts(p.margin[i]), pct(p.wp_go[i]), pct(p.wp_fg[i]), pct(p.wp_punt[i]), pct(p.p_conv[i], 0), pct(p.p_fg[i], 0), p.close_call[i] ? 'yes' : 'no']);
        }
        return out;
      });
    body.appendChild(h('div', { class: 'fx-bar' },
      h('div', {}, h('p', { class: 'label', style: 'margin-bottom:8px', text: 'Game state' }), ui.preset.el)));
    body.appendChild(ui.state);
    body.appendChild(h('p', { class: 'fnote' }, me(), ' Point estimates. Held fixed: ' + C.as_of.replace(/^week \d+ of \d+: /, '') + '.'));
    body.appendChild(h('div', { class: 'fx-key', role: 'group', 'aria-label': 'Chart key' },
      h('span', {}), h('span', { class: 'kh', text: 'Go for it' }), h('span', { class: 'kh', text: 'Field goal' }), h('span', { class: 'kh', text: 'Punt' }),
      h('span', { class: 'kr', text: 'Clear call' }), h('i', { class: 'kc go' }), h('i', { class: 'kc fg' }), h('i', { class: 'kc punt' }),
      h('span', { class: 'kr', text: 'Lean' }), h('i', { class: 'kc go lean' }), h('i', { class: 'kc fg lean' }), h('i', { class: 'kc punt lean' }),
      h('span', { class: 'kr', text: 'Toss-up' }), h('i', { class: 'kc toss' }), h('span', { class: 'kn2', text: 'no option clearly best' })));
    body.appendChild(h('p', { class: 'fnote', text: 'Margin of the best option over the second best, in win probability: clear ' + LEAN + ' or more, lean ' + TOSS + ' to ' + LEAN + ', toss-up under ' + TOSS + '.' }));
    body.appendChild(well(ui.heat));
    body.appendChild(ui.detail);
    body.appendChild(ui.tv.el);
    watch(ui.heat, function () { drawHeat(); });
    updateState();
  }

  function updateState() {
    var p = P[presetKey].p, close = 0;
    for (var i = 0; i < p.close_call.length; i++) if (p.close_call[i]) close++;
    ui.state.textContent = close + ' of ' + p.best.length + ' situations are toss-ups';
    setText('chart-h', 'Go, kick or punt: the best option by yard line and distance');
  }

  function drawHeat() {
    var host = ui.heat;
    if (!host) return;
    clear(host);
    var q = P[presetKey], p = q.p;
    var Wd = Math.max(host.clientWidth || 700, 600), ch = Wd > 1100 ? 30 : 26;
    var m = { l: 44, r: 8, t: 8, b: 56 };
    var iw = Wd - m.l - m.r, cw = iw / 99, Ht = m.t + 10 * ch + m.b;
    var svg = sv('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, width: Wd, height: Ht, tabindex: 0, role: 'application', class: 'hm',
      'aria-label': 'Fourth-down decision map, ' + p.label + '. Rows are yards to go from 1 to 10, columns are yard lines from your own goal line to the opponent\'s. Use the arrow keys to move between squares; the table below lists every square. Point estimates, model-estimated, no uncertainty band.' }, host);
    ui.svg = svg;
    // Lean cells: the action's color as a diagonal hatch on near-white. One pattern per action, shared across cells.
    var defs = sv('defs', {}, svg);
    ['go', 'fg', 'punt'].forEach(function (k) {
      var pat = sv('pattern', { id: 'hatch-' + k, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      sv('rect', { width: 6, height: 6, style: 'fill:var(--ab-tile)' }, pat);
      sv('rect', { width: 2.6, height: 6, style: 'fill:var(--cat-' + k + ')' }, pat);
    });
    function cx(yl) { return m.l + (99 - yl) * cw; }
    function cy(d) { return m.t + (10 - d) * ch; }
    for (var d = 1; d <= 10; d++) sv('text', { class: 'tick a-end', x: m.l - 8, y: cy(d) + ch / 2 + 4, text: String(d) }, svg);
    sv('text', { class: 'tick', transform: 'rotate(-90 12 ' + (m.t + 5 * ch) + ')', x: 12, y: m.t + 5 * ch + 4, 'text-anchor': 'middle', text: 'Yards to go' }, svg);
    var rects = {};
    for (var i = 0; i < q.n; i++) {
      var yl = p.yardline_100[i], dd = p.ydstogo[i];
      var lv = p.close_call[i] ? 0 : p.margin[i] < LEAN ? 1 : 2;   // three discrete steps: toss-up, lean, clear
      var r = sv('rect', { class: 'hc c-' + p.best[i] + ' lv-' + lv, x: cx(yl) + 0.3, y: cy(dd) + 0.3, width: Math.max(1, cw - 0.6), height: ch - 0.6 }, svg);
      rects[yl + '|' + dd] = r;
    }
    [90, 80, 70, 60, 50, 40, 30, 20, 10].forEach(function (yl) {
      sv('text', { class: 'tick a-mid', x: cx(yl) + cw / 2, y: m.t + 10 * ch + 16, text: N.fieldLabel(yl).replace('midfield', '50') }, svg);
    });
    sv('text', { class: 'tick a-start', x: m.l, y: Ht - 10, text: 'Own goal line', style: 'fill:var(--muted)' }, svg);
    sv('text', { class: 'tick a-end', x: Wd - m.r, y: Ht - 10, text: 'Opponent\'s goal line', style: 'fill:var(--muted)' }, svg);
    sv('text', { class: 'tick a-mid', x: m.l + iw / 2, y: Ht - 10, text: 'Field position', style: 'fill:var(--muted)' }, svg);

    var sel = null;
    function mark(yl, dd) {
      if (sel) sel.classList.remove('on');
      sel = rects[yl + '|' + dd] || null;
      if (sel) { sel.classList.add('on'); sel.parentNode.appendChild(sel); }
    }
    function tip(r, x, y) {
      var rows = ['go', 'fg', 'punt'].map(function (o) { return { value: pct(r[o]), label: OPT[o].toLowerCase() + (o === r.best ? ' (best)' : ''), cls: 'sw-' + o }; });
      rows.push({ value: pct(r.pc, 0), label: 'chance to convert if going' });
      rows.push({ value: pct(r.pf, 0), label: 'chance to make the kick' });
      rows.push({ value: pts(r.margin) + ' pp', label: r.close ? 'margin: a toss-up' : 'margin over the second best' });
      tipShow(x, y, '4th and ' + r.d + ' at ' + N.fieldLabel(r.yl) + ', model-estimated', rows);
    }
    function pick(yl, dd, x, y) {
      var r = rec(yl, dd);
      if (!r) return false;
      cursor.yl = yl; cursor.d = dd; mark(yl, dd);
      tip(r, x, y);
      clear(ui.detail); ui.detail.append(readout(r), me());
      return true;
    }
    function cellAtPoint(e) {
      var b = svg.getBoundingClientRect(), sc = Wd / b.width;
      var x = (e.clientX - b.left) * sc, y = (e.clientY - b.top) * (Ht / b.height);
      var col = Math.floor((x - m.l) / cw), row = Math.floor((y - m.t) / ch);
      if (col < 0 || col > 98 || row < 0 || row > 9) return null;
      return { yl: 99 - col, d: 10 - row };
    }
    svg.addEventListener('pointermove', function (e) { var c = cellAtPoint(e); if (!c || !pick(c.yl, c.d, e.clientX, e.clientY)) { tipHide(); mark(-1, -1); } });
    svg.addEventListener('pointerleave', function () { tipHide(); mark(-1, -1); });
    svg.addEventListener('click', function (e) { var c = cellAtPoint(e); if (c) pick(c.yl, c.d, e.clientX, e.clientY); });
    function focusCell() {
      var b = svg.getBoundingClientRect(), sc = b.width / Wd;
      pick(cursor.yl, cursor.d, b.left + (cx(cursor.yl) + cw / 2) * sc, b.top + (cy(cursor.d) + ch / 2) * sc);
    }
    svg.addEventListener('focus', function () { if (!rec(cursor.yl, cursor.d)) { cursor.yl = 40; cursor.d = 3; } focusCell(); });
    svg.addEventListener('blur', function () { tipHide(); mark(-1, -1); });
    svg.addEventListener('keydown', function (e) {
      var yl = cursor.yl, dd = cursor.d, big = e.shiftKey ? 5 : 1;
      if (e.key === 'ArrowLeft') yl = Math.min(99, yl + big); else if (e.key === 'ArrowRight') yl = Math.max(1, yl - big);
      else if (e.key === 'ArrowUp') dd = Math.min(10, dd + 1); else if (e.key === 'ArrowDown') dd = Math.max(1, dd - 1);
      else return;
      e.preventDefault();
      if (dd > yl) dd = yl;
      cursor.yl = yl; cursor.d = dd; focusCell();
    });
  }

  // ------------------------------------------------------------------ 2026 so far

  var TCOLS = [
    ['team', 'Team', true], ['fourth_downs', 'Fourth downs'], ['model_said_go', 'Model said go'], ['go_rate_when_model_said_go', 'Went when it said go'],
    ['go_rate_when_model_said_go_clear', 'Went, clear calls'], ['matched_share', 'Matched'], ['wp_lost_total', 'Win prob. given up (wins)'], ['wp_lost_per_game', 'Per game']
  ];

  function buildLive() {
    var body = done('ytd-body'), L = F.live, s = L.summary;
    clear(body);
    setText('ytd-h', 'Teams matched the model on ' + pct(s.matched_share, 0) + ' of fourth downs; ' + pct(s.tossup_share, 0) + ' were toss-ups');
    var hb = L.updated_through.held_back || [];
    body.appendChild(h('div', {},
      h('dl', { class: 'statrow', style: 'margin-top:0' },
        stat('Fourth downs', commas(s.fourth_downs), commas(s.games) + ' games, last on ' + L.updated_through.last_game_date),
        stat('Matched the model', pct(s.matched_share, 1), 'model-estimated'),
        stat('Toss-ups', pct(s.tossup_share, 1), 'model-estimated'),
        stat('Win probability given up', s.wp_lost_total.toFixed(1) + ' wins', s.wp_lost_per_team_game.toFixed(3) + ' per team-game, model-estimated')),
      h('div', { class: 'mix', role: 'group', 'aria-label': 'What teams chose against what the model recommended, as counts' },
        mixRow('Teams chose', s.choices), mixRow('Model preferred', s.recommended)),
      h('p', { class: 'fnote', text: 'Team choices are observed; the rest is model-estimated.' })));

    // teams
    ui.teamBody = h('tbody');
    ui.teamHead = h('thead', {}, h('tr', {}, TCOLS.map(function (c, i) {
      var b = h('button', { type: 'button', class: 'sortb', 'data-k': c[0], text: c[1] });
      b.addEventListener('click', function () {
        if (teamSort.key === c[0]) teamSort.dir = teamSort.dir === 'desc' ? 'asc' : 'desc';
        else { teamSort.key = c[0]; teamSort.dir = c[2] ? 'asc' : 'desc'; }
        renderTeams();
      });
      return h('th', { class: i ? 'r' : '', scope: 'col' }, b);
    })));
    body.appendChild(h('div', { style: 'margin-top:48px' },
      h('h3', { class: 'sub-h', style: 'margin-top:0' }, 'Teams ', me()),
      h('p', { class: 'fnote', style: 'margin-top:0', text: 'Select a team to filter the plays below. Model-estimated except fourth-down counts.' }),
      h('div', { class: 'tbl-wrap', style: 'margin-top:14px' }, h('table', { class: 'tmt', 'aria-label': 'Teams, model-estimated' }, ui.teamHead, ui.teamBody))));
    renderTeams();

    // plays
    ui.teamSel = h('select', { id: 'play-team', 'aria-label': 'Filter plays by team' }, h('option', { value: '', text: 'All teams' }),
      F.live.teams.map(function (t) { return h('option', { value: t.team, text: N.teamName(t.team) }); }));
    ui.teamSel.addEventListener('change', function () { teamFilter = ui.teamSel.value; playLimit = 25; renderTeams(); renderPlays(); });
    ui.modeSeg = N.seg('Which plays', [{ value: 'all', label: 'All' }, { value: 'differed', label: 'Differed from the model' }, { value: 'tossup', label: 'Toss-ups' }], playMode, function (v) { playMode = v; playLimit = 25; renderPlays(); });
    ui.modeSeg.el.classList.add('seg');
    ui.plist = h('div', { class: 'plist' });
    ui.pmore = h('button', { type: 'button', class: 'btn xmore', text: 'Show 25 more' });
    ui.pmore.addEventListener('click', function () { playLimit += 25; renderPlays(); });
    ui.pcount = h('p', { class: 'fnote', 'aria-live': 'polite' });
    body.appendChild(h('div', { style: 'margin-top:56px' },
      h('h3', { class: 'sub-h', style: 'margin-top:0' }, 'Plays ', me()),
      h('div', { class: 'pfilters' }, h('div', { class: 'field' }, h('label', { for: 'play-team', text: 'Team' }), h('div', { class: 'sel' }, ui.teamSel)), ui.modeSeg.el),
      ui.pcount, ui.plist, ui.pmore));
    renderPlays();
    if (hb.length) body.appendChild(h('p', { class: 'fnote heldback', text: 'Held back: ' + hb.map(function (id) { var q = id.split('_'); return q.length === 4 ? 'week ' + Number(q[1]) + ', ' + q[2] + ' at ' + q[3] : id; }).join(', ') + '. Added next update.' }));
    body.appendChild(h('p', { class: 'fnote', text: 'Excludes ties, overtime, penalties and kneel-downs.' }));
  }

  function stat(label, value, small) { return h('div', {}, h('dt', { text: label }), h('dd', { class: 'num' }, value, h('small', { text: small }))); }
  function mixRow(label, counts) {
    var tot = counts.go + counts.fg + counts.punt;
    return h('div', { class: 'mix-row' }, h('span', { text: label }), h('div', { class: 'mix-bar' }, ['go', 'fg', 'punt'].map(function (k) {
      return h('span', { class: k, style: 'flex:' + Math.max(counts[k], 0.0001) + ' 1 0', title: OPT[k] + ': ' + counts[k] + ' of ' + tot }, counts[k] / tot > 0.12 ? OPT_SHORT[k] + ' ' + counts[k] : '');
    })));
  }

  function renderTeams() {
    var T = F.live.teams.slice(), k = teamSort.key, dir = teamSort.dir === 'desc' ? -1 : 1;
    T.sort(function (a, b) { var x = a[k], y = b[k]; if (typeof x === 'string') return dir * x.localeCompare(y); return dir * ((x == null ? -1 : x) - (y == null ? -1 : y)) || a.team.localeCompare(b.team); });
    Array.prototype.forEach.call(ui.teamHead.querySelectorAll('.sortb'), function (b) {
      var on = b.dataset.k === k;
      b.classList.toggle('on', on); b.setAttribute('data-dir', teamSort.dir);
      b.parentNode.setAttribute('aria-sort', on ? (teamSort.dir === 'desc' ? 'descending' : 'ascending') : 'none');
    });
    clear(ui.teamBody);
    T.forEach(function (t) {
      var btn = h('button', { type: 'button', class: 'tmbtn', 'aria-pressed': teamFilter === t.team ? 'true' : 'false', 'aria-label': 'Show only ' + N.teamName(t.team) + ' plays' }, h('span', { class: 'abbr', text: t.team }), h('span', { class: 'nm', text: N.teamName(t.team).replace(/^.* /, '') }));
      btn.addEventListener('click', function () {
        teamFilter = teamFilter === t.team ? '' : t.team; ui.teamSel.value = teamFilter; playLimit = 25; renderTeams(); renderPlays();
      });
      function r(v) { return h('td', { class: 'r n', text: v }); }
      function rate(v) { return v == null ? 'n/a' : pct(v, 0); }
      ui.teamBody.appendChild(h('tr', { class: teamFilter === t.team ? 'is-sel' : null }, h('th', { scope: 'row' }, btn),
        r(String(t.fourth_downs)), r(String(t.model_said_go)),
        r(t.model_said_go ? t.went_when_model_said_go + ' (' + rate(t.go_rate_when_model_said_go) + ')' : 'n/a'),
        r(t.model_said_go_clear ? rate(t.go_rate_when_model_said_go_clear) + ' of ' + t.model_said_go_clear : 'n/a'),
        r(pct(t.matched_share, 0)), r(t.wp_lost_total.toFixed(2)), r(t.wp_lost_per_game.toFixed(3))));
    });
  }

  function scoreText(d) { return d > 0 ? 'leading by ' + d : d < 0 ? 'trailing by ' + (-d) : 'tied'; }

  function renderPlays() {
    var col = F.live.plays.columns, rows = F.live.plays.rows;
    var list = [];
    rows.forEach(function (r) {
      var o = {}; col.forEach(function (c, i) { o[c] = r[i]; });
      if (teamFilter && o.team !== teamFilter) return;
      if (playMode === 'differed' && o.matched) return;
      if (playMode === 'tossup' && !o.tossup) return;
      list.push(o);
    });
    list.sort(function (a, b) { return b.week - a.week || a.game_id.localeCompare(b.game_id) || a.play_id - b.play_id; });
    var n = Math.min(playLimit, list.length);
    ui.pcount.textContent = commas(list.length) + ' fourth downs' + (teamFilter ? ' for ' + N.teamName(teamFilter) : '') + (playMode === 'differed' ? ' where the team differed from the model' : playMode === 'tossup' ? ' that were toss-ups' : '') + '. Showing ' + n + ', newest first. Model-estimated.';
    clear(ui.plist);
    list.slice(0, n).forEach(function (o) {
      ui.plist.appendChild(h('details', { class: 'prow' },
        h('summary', {},
          h('span', { class: 'pw num', text: 'Wk ' + o.week }),
          h('span', { class: 'ps' }, o.team + (o.home ? ' vs ' : ' at ') + o.opponent, h('small', { text: o.clock + ', 4th and ' + o.ydstogo + ' at ' + N.fieldLabel(o.yardline_100) + ', ' + scoreText(o.score_diff) })),
          h('span', { class: 'pc' }, h('b', {}, cat(o.choice), CHOICE[o.choice]), o.matched ? null : h('span', { class: 'arrow', text: 'model prefers' }), o.matched ? null : h('b', {}, cat(o.recommended), CHOICE[o.recommended]),
            o.tossup ? h('span', { class: 'tag-fill', text: 'Toss-up' }) : null,
            h('small', { text: (o.matched ? 'Matched the model' : 'Chose ' + CHOICE[o.choice].toLowerCase() + '; model preferred ' + CHOICE[o.recommended].toLowerCase()) })),
          h('span', { class: 'pg num' }, pts(o.wp_given_up) + ' pp', h('small', { text: 'given up, model-estimated' }))),
        h('div', { class: 'prow-body' },
          h('div', { class: 'opts' },
            h('span', { class: 'opt-go' }, 'Go: ', h('b', { text: pct(o.wp_go) }), ' win, ' + pct(o.p_convert, 0) + ' to convert'),
            h('span', { class: 'opt-fg' }, 'Field goal: ', h('b', { text: pct(o.wp_fg) }), ', ' + pct(o.p_fg_make, 0) + ' to make'),
            h('span', { class: 'opt-punt' }, 'Punt: ', h('b', { text: pct(o.wp_punt) }))),
          h('span', {}, 'Margin ' + pts(o.margin) + ' pp' + (o.tossup ? ', a toss-up' : '') + '. Model-estimated.'),
          h('span', { text: o.desc }))));
    });
    if (!list.length) ui.plist.appendChild(h('p', { class: 'note empty', text: 'No fourth downs match this filter.' }));
    ui.pmore.hidden = n >= list.length;
    ui.pmore.textContent = 'Show ' + Math.min(25, list.length - n) + ' more';
  }

  // ------------------------------------------------------------------ method

  function buildHow() {
    var dyn = $('#how-dyn'), a = F.headline.holdout_audit_v1, E = F.engine;
    dyn.appendChild(h('div', {},
      h('h3', { class: 'sub-h', text: 'The 2020 to 2025 audit' }),
      h('p', { class: 'stat-note', text: 'Matched on ' + pct(a.matched_share, 0) + ' of ' + commas(a.fourth_downs) + ' fourth downs (' + pct(a.matched_share_clear_calls, 0) + ' of clear calls). Model-estimated.' }),
      h('dl', { class: 'statrow' },
        stat('Total given up', a.wp_lost_total.toFixed(1) + ' wins', 'all teams, model-estimated'),
        stat('By punting', a.wp_lost_by_choice.punt.toFixed(1), 'wins, model-estimated'),
        stat('By kicking field goals', a.wp_lost_by_choice.fg.toFixed(1), 'wins, model-estimated'),
        stat('By going for it', a.wp_lost_by_choice.go.toFixed(1), 'wins, model-estimated'),
        stat('Toss-ups', pct(a.tossup_share, 0), 'of fourth downs')),
      h('p', { class: 'fnote', text: 'First engine, which failed its conversion test. Its win probability model beat nflfastR (Brier ' + F.headline.wp_model.brier_ours.toFixed(4) + ' vs ' + F.headline.wp_model.brier_nflfastr.toFixed(4) + ').' })));
    dyn.appendChild(h('p', { class: 'stat-note', style: 'margin-top:0', text: 'Current engine: ' + E.engine.candidate + ', frozen ' + E.engine.frozen_at.slice(0, 10) + '.' }));
    dyn.appendChild(N.caveatBox(['Teams know things the model does not.'], 'Caveat'));
    dyn.appendChild(h('dl', { class: 'gloss' },
      [['Toss-up', 'margin under ' + F.tossup_rule.margin_below + ' win probability'], ['Margin', 'best option minus second best'], ['Given up', 'best option minus the team\'s choice'], ['Clear call', 'margin of ' + F.tossup_rule.margin_below + ' or more']].map(function (k) { return h('div', {}, h('dt', { text: k[0] }), h('dd', { text: k[1] })); })));
  }

  // ------------------------------------------------------------------ boot

  var ALL = ['chart-body', 'ytd-body'];
  N.loadTeamNames().then(function () { return N.getJSON('data/fourth.json', true); }).then(function (d) {
    if (!d || !d.chart) { N.unavailable('The fourth-down data', ALL); return; }
    F = d;
    renderHero();
    buildChart();
    buildLive();
    buildHow();
  }).catch(function (err) { console.error(err); N.unavailable('The fourth-down data', ALL); });
})();
