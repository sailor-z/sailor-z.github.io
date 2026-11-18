/* Interactive method figures, staged as in the overview video.
   1. Step figures (.stepfig): a paper figure with one stage lit at a time. The stages and their regions
      (fractions x0, y0, x1, y1 of the image) are in the element's data-steps; they are the spotlight regions of
      the video (pr_video/make_video.py, s4a_pipeline and s4b_head).
   2. Mechanisms (.mech): target attention (all frames or the target frame only) and source fusion (child
      queries without and with source features), as in the video's s4c_mechanisms. One query pixel is shared:
      clicking the source frame moves it, and the fusion colours are recomputed from the image there. */
(function () {
  'use strict';
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* Run `enter` the first time the element is mostly in view; `visible(bool)` on every change. */
  function watch(el, enter, visible) {
    let seen = false;
    new IntersectionObserver(entries => {
      for (const e of entries) {
        if (visible) visible(e.isIntersecting);
        if (e.isIntersecting && !seen) { seen = true; enter(); }
      }
    }, { threshold: 0.45 }).observe(el);
  }

  /* ------------------------------------------------------------------ step figures */
  function stepFigure(root) {
    const steps = JSON.parse(root.dataset.steps);
    const canvas = root.querySelector('.stepfig-canvas');
    const img = canvas.querySelector('img');
    const bar = root.querySelector('.stepfig-bar');
    const text = root.querySelector('.stepfig-text');
    const overview = text.textContent.trim();
    let current = -1, timer = null, inView = false, touched = false;

    const spots = steps.map((step, i) => step.regions.map(([x0, y0, x1, y1]) => {
      const lit = document.createElement('div');           // the full image, clipped to one region
      lit.className = 'stepfig-spot';
      lit.style.backgroundImage = `url("${img.getAttribute('src')}")`;
      lit.style.clipPath = `inset(${y0 * 100}% ${(1 - x1) * 100}% ${(1 - y1) * 100}% ${x0 * 100}%)`;
      const hit = document.createElement('div');           // hovering or tapping a region selects its stage
      hit.className = 'stepfig-hit';
      hit.style.cssText = `left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%`;
      hit.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') pick(i); });
      hit.addEventListener('click', () => pick(i));
      canvas.append(lit, hit);
      return lit;
    }));

    const buttons = steps.map((step, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'stepfig-step';
      b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', 'false');
      b.innerHTML = `<span>${i + 1}</span>${step.label}`;
      b.addEventListener('click', () => pick(i));
      bar.append(b);
      return b;
    });
    const all = document.createElement('button');
    all.type = 'button'; all.className = 'stepfig-step stepfig-all';
    all.setAttribute('role', 'tab'); all.setAttribute('aria-selected', 'true');
    all.textContent = 'Whole figure';
    all.addEventListener('click', () => pick(-1));
    bar.append(all);

    function show(i) {
      current = i;
      root.classList.toggle('is-stepping', i >= 0);
      spots.forEach((group, k) => group.forEach(s => s.classList.toggle('on', k === i)));
      buttons.forEach((b, k) => b.setAttribute('aria-selected', String(k === i)));
      all.setAttribute('aria-selected', String(i < 0));
      text.innerHTML = i < 0 ? overview : `<b>${steps[i].label}.</b> ${steps[i].text}`;
    }
    function pick(i) { touched = true; stop(); show(i); }
    function stop() { clearInterval(timer); timer = null; }
    function play() {                                       // until the visitor picks a stage
      if (timer || touched || reduceMotion || !inView) return;
      if (current < 0) show(0);
      timer = setInterval(() => show((current + 1) % steps.length), 3800);
    }
    watch(root, play, v => { inView = v; if (v) play(); else stop(); });
  }

  /* ------------------------------------------------------------------ mechanisms */
  const FRAME = [688, 384];        // the source frame, in the pixels of its image file
  const CROP = 84;                 // side of the square around the query that gives the feature colours

  function rgbToHsv(r, g, b) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return [((h / 6) % 1 + 1) % 1, mx ? d / mx : 0, mx];
  }
  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  }
  const clamp01 = x => Math.min(1, Math.max(0, x));
  /* Display tone for feature cells: a little more saturation and light than the raw image colours. */
  function tone([r, g, b]) {
    const grey = (r + g + b) / 3;
    return `rgb(${[r, g, b].map(c => Math.round(255 * clamp01((grey + 1.3 * (c - grey)) * 1.15 + .03))).join(',')})`;
  }

  function mechanisms(root) {
    const src = root.querySelector('.ta-source'), srcImg = src.querySelector('img'), dot = src.querySelector('.ta-query');
    const frames = [...root.querySelectorAll('.ta-frame')];
    const svg = root.querySelector('.ta-lines'), stage = root.querySelector('.ta-stage');
    const taButtons = [...root.querySelectorAll('[data-ta]')], taCost = root.querySelector('.ta-cost');
    const sfButtons = [...root.querySelectorAll('[data-sf]')], sfText = root.querySelector('.sf-text');
    const grids = [...root.querySelectorAll('.sf-grid')], feats = [...root.querySelectorAll('.sf-feat')];
    const state = { query: [150, 262], target: 4, only: false, fused: false, colours: null, touchedTa: false, touchedSf: false };

    /* --- colours of the child queries, from the image around the query pixel */
    const pixels = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
    function crop() {
      const [qx, qy] = state.query;
      return [Math.round(Math.min(FRAME[0] - CROP, Math.max(0, qx - CROP / 2))),
              Math.round(Math.min(FRAME[1] - CROP, Math.max(0, qy - CROP / 2)))];
    }
    function boxColours(data, n) {                           // n x n block means of the crop, in [0, 1]
      const out = [];
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const x0 = Math.floor(i * CROP / n), x1 = Math.floor((i + 1) * CROP / n);
        const y0 = Math.floor(j * CROP / n), y1 = Math.floor((j + 1) * CROP / n);
        let r = 0, g = 0, b = 0;
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
          const k = 4 * (y * CROP + x); r += data[k]; g += data[k + 1]; b += data[k + 2];
        }
        const count = (x1 - x0) * (y1 - y0) * 255;
        out.push([r / count, g / count, b / count]);
      }
      return out;
    }
    /* Without fusion the children stay close to the root: hue and saturation drift along one diagonal,
       lightness along the other, so the cells are similar but not identical. */
    function unfused(root3, n) {
      if (n === 1) return [root3];
      const [h, s, v] = rgbToHsv(...root3), out = [];
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const a = i / (n - 1), b = j / (n - 1);
        out.push(hsvToRgb(((h + .034 * (a - b)) % 1 + 1) % 1, clamp01(s * (1 + .16 * (a - b))), clamp01(v * (1 + .11 * (1 - a - b)))));
      }
      return out;
    }
    function computeColours() {
      const [cx, cy] = crop();
      let data = null;
      try {
        pixels.canvas.width = pixels.canvas.height = CROP;
        pixels.drawImage(srcImg, cx * srcImg.naturalWidth / FRAME[0], cy * srcImg.naturalHeight / FRAME[1],
                         CROP * srcImg.naturalWidth / FRAME[0], CROP * srcImg.naturalHeight / FRAME[1], 0, 0, CROP, CROP);
        data = pixels.getImageData(0, 0, CROP, CROP).data;
      } catch (e) { data = null; }                            // e.g. the page opened from disk: fixed colours
      const fused = {}, plain = {};
      for (const n of [1, 3, 7]) {
        fused[n] = data ? boxColours(data, n)
          : Array.from({ length: n * n }, (_, k) => hsvToRgb((.08 + .5 * (k % n) / n + .11 * Math.floor(k / n)) % 1, .45, .78));
        plain[n] = unfused(fused[1][0], n);
      }
      state.colours = { fused, plain };
      feats.forEach(el => {                                   // the source features each level adds, coarse to fine
        const res = +el.dataset.res, ctx = el.getContext('2d');
        try {
          ctx.clearRect(0, 0, res, res);
          ctx.drawImage(srcImg, cx * srcImg.naturalWidth / FRAME[0], cy * srcImg.naturalHeight / FRAME[1],
                        CROP * srcImg.naturalWidth / FRAME[0], CROP * srcImg.naturalHeight / FRAME[1], 0, 0, res, res);
        } catch (e) { /* image not ready */ }
      });
    }
    function paintGrids() {
      if (!state.colours) return;
      grids.forEach(grid => {
        const n = +grid.dataset.n, cols = state.colours[state.fused ? 'fused' : 'plain'][n];
        if (grid.children.length !== n * n) {
          grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
          grid.replaceChildren(...Array.from({ length: n * n }, () => document.createElement('i')));
        }
        [...grid.children].forEach((cell, k) => { cell.style.backgroundColor = tone(cols[k]); });
      });
    }

    /* --- lines from the query to the frames it attends to */
    function drawLines() {
      const box = stage.getBoundingClientRect(), s = src.getBoundingClientRect();
      const qx = s.left - box.left + state.query[0] / FRAME[0] * s.width, qy = s.top - box.top + state.query[1] / FRAME[1] * s.height;
      svg.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
      svg.innerHTML = frames.map((f, k) => {
        const r = f.getBoundingClientRect();
        const horizontal = r.left > s.right - 4;              // frames beside the source frame, or below it
        const x = horizontal ? r.left - box.left + 2 : r.left - box.left + r.width / 2;
        const y = horizontal ? r.top - box.top + r.height / 2 : r.top - box.top + 2;
        const cls = k === state.target ? 'target' : 'other';
        return `<line class="${cls}" x1="${qx.toFixed(1)}" y1="${qy.toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`;
      }).join('');
      dot.style.left = `${state.query[0] / FRAME[0] * 100}%`;
      dot.style.top = `${state.query[1] / FRAME[1] * 100}%`;
    }

    function render() {
      root.classList.toggle('ta-only', state.only);
      root.classList.toggle('sf-fused', state.fused);
      frames.forEach((f, k) => {
        f.classList.toggle('is-target', k === state.target);
        f.setAttribute('aria-pressed', String(k === state.target));
      });
      taButtons.forEach(b => b.setAttribute('aria-pressed', String((b.dataset.ta === 'only') === state.only)));
      sfButtons.forEach(b => b.setAttribute('aria-pressed', String((b.dataset.sf === 'with') === state.fused)));
      taCost.innerHTML = state.only
        ? 'VGGM-Nexus attends to the target frame only: <b class="tgt">O(<i>Q</i>·<i>HW</i>)</b>'
        : 'Point-to-point decoders attend to all <i>T</i> frames: <b class="bad">O(<i>T</i>·<i>Q</i>·<i>HW</i>)</b>';
      sfText.innerHTML = state.fused
        ? '<b class="src">With fusion,</b> each child adds source features from its own location.'
        : '<b class="bad">Without fusion,</b> children inherit the same parent features, so they stay similar.';
      drawLines();
      paintGrids();
    }

    taButtons.forEach(b => b.addEventListener('click', () => { state.touchedTa = true; state.only = b.dataset.ta === 'only'; render(); }));
    sfButtons.forEach(b => b.addEventListener('click', () => { state.touchedSf = true; state.fused = b.dataset.sf === 'with'; render(); }));
    frames.forEach((f, k) => f.addEventListener('click', () => { state.touchedTa = true; state.target = k; state.only = true; render(); }));
    src.addEventListener('click', e => {                     // move the query pixel
      const r = srcImg.getBoundingClientRect();
      state.query = [Math.round(clamp01((e.clientX - r.left) / r.width) * FRAME[0]), Math.round(clamp01((e.clientY - r.top) / r.height) * FRAME[1])];
      computeColours(); render();
    });
    new ResizeObserver(() => { computeColours(); render(); }).observe(stage);

    const ready = () => { computeColours(); render(); };
    if (srcImg.complete && srcImg.naturalWidth) ready(); else srcImg.addEventListener('load', ready);
    render();
    /* As in the video: first all frames and no fusion, then the VGGM-Nexus design. */
    watch(root, () => {
      if (reduceMotion) { state.only = state.fused = true; render(); return; }
      setTimeout(() => { if (!state.touchedTa) { state.only = true; render(); } }, 1600);
      setTimeout(() => { if (!state.touchedSf) { state.fused = true; render(); } }, 3000);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.stepfig').forEach(stepFigure);
    document.querySelectorAll('.mech').forEach(mechanisms);
  });
})();
