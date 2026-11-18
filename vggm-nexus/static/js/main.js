/* VGGM-Nexus project page: placeholder links, BibTeX copy, nav highlight.
   The 4D and 3D viewers are in static/js/viewer.js, the method figures in static/js/method.js. */
(function () {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);

  /* Placeholder links: no navigation until a real href is set. */
  document.querySelectorAll('[data-soon]').forEach(a => {
    a.setAttribute('aria-disabled', 'true');
    a.title = 'Coming soon';
    a.addEventListener('click', e => e.preventDefault());
  });

  /* ------------------------------------------------------------ bibtex + nav */
  function initBibtex() {
    const btn = $('#copy-bibtex');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const text = $('#bibtex-text').innerText.trim();
      try { await navigator.clipboard.writeText(text); btn.textContent = 'Copied'; }
      catch { const r = document.createRange(); r.selectNodeContents($('#bibtex-text')); getSelection().removeAllRanges(); getSelection().addRange(r); btn.textContent = 'Press ⌘/Ctrl+C'; }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
    });
  }

  function initNav() {
    const links = [...document.querySelectorAll('.nav-links a')];
    const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver(entries => {
      for (const e of entries) if (e.isIntersecting) {
        links.forEach(a => a.classList.remove('active'));
        byId.get(e.target.id)?.classList.add('active');
      }
    }, { rootMargin: '-45% 0px -50% 0px' });
    byId.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initBibtex(); initNav();
  });
})();
