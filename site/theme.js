/* NFLELO theme: Auto / Dark / Light. Runs in <head> before first paint so the saved choice never flashes.
   Auto follows the OS (prefers-color-scheme); Dark and Light set <html data-theme>. The choice is kept in
   localStorage, and every storage call is wrapped because private modes can throw. */
(function () {
  'use strict';
  var KEY = 'nflelo-theme', root = document.documentElement;
  var CANVAS = { dark: '#222a33', light: '#e6ebf0' };

  function read() {
    try { var v = localStorage.getItem(KEY); return v === 'dark' || v === 'light' ? v : 'auto'; } catch (e) { return 'auto'; }
  }
  function save(v) { try { localStorage.setItem(KEY, v); } catch (e) {} }

  var metaOriginal = null;
  function apply(choice) {
    if (choice === 'dark' || choice === 'light') root.setAttribute('data-theme', choice);
    else root.removeAttribute('data-theme');
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    if (!metaOriginal) metaOriginal = Array.prototype.map.call(metas, function (m) { return m.getAttribute('content'); });
    Array.prototype.forEach.call(metas, function (m, i) {
      m.setAttribute('content', CANVAS[choice] || metaOriginal[i]);
    });
  }

  var choice = read();
  if (choice !== 'auto') root.setAttribute('data-theme', choice);   // before first paint; metas are handled once the DOM exists

  document.addEventListener('DOMContentLoaded', function () {
    apply(choice);
    var buttons = document.querySelectorAll('[data-theme-choice]');
    function sync() {
      Array.prototype.forEach.call(buttons, function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-choice') === choice));
      });
    }
    sync();
    Array.prototype.forEach.call(buttons, function (b) {
      b.addEventListener('click', function () {
        choice = b.getAttribute('data-theme-choice');
        save(choice);
        apply(choice);
        sync();
      });
    });
  });
})();
