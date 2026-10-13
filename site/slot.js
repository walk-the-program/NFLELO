/* Kickoff slots for the home hero. Pure functions, no DOM: loaded by index.html (window.NFL.slot) and by the node test.
   A slot is every game sharing one kickoff instant. The hero shows the earliest slot that is not yet over,
   where "over" means kickoff + HOLD_MS (3.5 hours) is in the past. Games without a kickoff_utc cannot be placed and are skipped. */
(function (root) {
  'use strict';

  var HOLD_MS = 3.5 * 60 * 60 * 1000;

  // games: [{ kickoff_utc: '2026-10-25T17:00:00Z' | null, ... }], nowMs: epoch ms.
  // Returns { startMs, live, games } for the earliest slot still up, or null when there is none.
  function pickSlot(games, nowMs) {
    var by = {}, starts = [];
    (games || []).forEach(function (g) {
      var t = g && g.kickoff_utc ? Date.parse(g.kickoff_utc) : NaN;
      if (isNaN(t)) return;
      if (!by[t]) { by[t] = []; starts.push(t); }
      by[t].push(g);
    });
    starts.sort(function (a, b) { return a - b; });
    for (var i = 0; i < starts.length; i++) {
      if (starts[i] + HOLD_MS > nowMs) return { startMs: starts[i], live: starts[i] <= nowMs, games: by[starts[i]] };
    }
    return null;
  }

  // "Sun 12:00 PM CDT" in the viewer's zone (or the given IANA zone, for tests).
  function clockLabel(ms, tz) {
    var o = { weekday: 'short', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' };
    if (tz) o.timeZone = tz;
    var p = {};
    new Intl.DateTimeFormat('en-US', o).formatToParts(new Date(ms)).forEach(function (x) { p[x.type] = x.value; });
    return p.weekday + ' ' + p.hour + ':' + p.minute + ' ' + String(p.dayPeriod || '').toUpperCase() + ' ' + p.timeZoneName;
  }

  // "Next up · Sun 12:00 PM CDT" or "Live now · kicked off 12:00 PM CDT".
  function slotLabel(slot, tz) {
    var full = clockLabel(slot.startMs, tz);
    return slot.live ? 'Live now · kicked off ' + full.replace(/^\w+ /, '') : 'Next up · ' + full;
  }

  var api = { HOLD_MS: HOLD_MS, pickSlot: pickSlot, clockLabel: clockLabel, slotLabel: slotLabel };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.NFL = root.NFL || {}; root.NFL.slot = api; }
})(typeof window !== 'undefined' ? window : this);
