/* Interactive point-cloud viewers: VGGM-Nexus, V-DPM, 4RC and ARROW in a 2 × 2 grid with one shared orbit camera.
   One viewer is created for every .viewer[data-base] element; its parts are found by data-role inside it.
     4D viewer (static/viewer/, tools/export_viewer_data.py): clips of T frames, time playback and 3D tracks.
     3D viewer (static/viewer3d/, tools/export_viewer3d_data.py): static scenes (T = 1, no tracks), the fused
       pointmap of several views; the view sways a little until the first drag.
   Data: <base>manifest.json + <clip>/<method>.bin.gz:
     tails uint32[K] | pos int16[T*N*3] (quantized, delta-coded over time) | color uint8[N*3] | valid uint8[T*N] */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const VIEW = { elev: 8, azim: -10, dist: 1.15 };       // initial orbit about the scene centre, as the paper panels
const TAIL_RANGE = [0.10, 0.95];                        // turbo range over clip time, as the paper panels
const PANES = ['ours', 'vdpm', 'four_rc', 'arrow'];    // row by row (top to bottom when stacked)
const PANE_ASPECT_MAX = 1.3;                          // panes in a single row are at most this wide per unit height
const PANE_ASPECT_GRID = [1, 1.8];                    // panes in the 2 × 2 grid: at least square, at most a 16:9 clip
const SWAY = { angle: THREE.MathUtils.degToRad(9), period: 9000 };   // static scenes: ± 9° about the vertical axis
const UP = new THREE.Vector3(0, 1, 0);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ------------------------------------------------------------------ colour */
function turbo(x) {          // Google's polynomial approximation of the turbo colormap, sRGB in [0, 1]
  x = Math.min(1, Math.max(0, x));
  const r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [r, g, b].map(v => Math.min(1, Math.max(0, v)));
}

/* ------------------------------------------------------------------ data */
async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const raw = await res.arrayBuffer();
  const head = new Uint8Array(raw, 0, 2);
  if (head[0] !== 0x1f || head[1] !== 0x8b) return raw;            // already decompressed by the server
  if (!('DecompressionStream' in window)) throw new Error('This browser cannot decompress the viewer data.');
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

function decode(buf, meta, T) {
  const { N, K } = meta;
  let off = 0;
  const tails = new Uint32Array(buf, off, K); off += 4 * K;
  const delta = new Int16Array(buf, off, T * N * 3); off += 2 * T * N * 3;
  const color = new Uint8Array(buf, off, N * 3); off += N * 3;
  const valid = new Uint8Array(buf, off, T * N);
  const lo = meta.lo, span = meta.hi.map((h, a) => (h - lo[a]) / 65534);
  const acc = new Int16Array(N * 3);                                  // wraps modulo 2^16, matching the encoder
  const pos = new Float32Array(T * N * 3);
  for (let t = 0; t < T; t++) {
    const d = t * N * 3;
    for (let i = 0; i < N * 3; i++) {
      acc[i] += delta[d + i];
      const a = i % 3;
      pos[d + i] = lo[a] + (acc[i] + 32767) * span[a];
    }
  }
  return { N, K, T, tails, color, valid, pos };
}

/* ------------------------------------------------------------------ three.js objects */
const pointMaterial = () => new THREE.ShaderMaterial({
  uniforms: { uSize: { value: 2 }, uScale: { value: 1 } },
  vertexShader: `
    attribute vec3 aColor; attribute float aValid;
    uniform float uSize; uniform float uScale;
    varying vec3 vColor;
    void main() {
      vColor = aColor;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = uSize * uScale;
      if (aValid < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; }
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
});

function buildPane(data) {
  const { N, K, T, tails, color, valid, pos } = data;
  const geo = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  position.setUsage(THREE.DynamicDrawUsage);
  const vis = new THREE.BufferAttribute(new Uint8Array(N), 1);
  vis.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', position);
  geo.setAttribute('aColor', new THREE.BufferAttribute(color, 3, true));
  geo.setAttribute('aValid', vis);
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -1), 1e3);
  const points = new THREE.Points(geo, pointMaterial());
  points.frustumCulled = false;

  // Tails: frame-major segments (f -> f+1 for every track), so drawRange(0, 2*K*t) shows tracks up to t.
  let lines = null;
  if (K > 0 && T > 1) {
    const seg = new Float32Array((T - 1) * K * 6), col = new Float32Array((T - 1) * K * 6);
    const c = new THREE.Color();
    const tint = f => { const [r, g, b] = turbo(TAIL_RANGE[0] + (TAIL_RANGE[1] - TAIL_RANGE[0]) * f / (T - 1)); c.setRGB(r, g, b, THREE.SRGBColorSpace); return [c.r, c.g, c.b]; };
    for (let f = 0; f < T - 1; f++) {
      const c0 = tint(f), c1 = tint(f + 1);
      for (let k = 0; k < K; k++) {
        const i = tails[k], o = (f * K + k) * 6;
        const a = (f * N + i) * 3, b = ((f + 1) * N + i) * 3;
        const ok = valid[f * N + i] && valid[(f + 1) * N + i];
        for (let j = 0; j < 3; j++) {
          seg[o + j] = pos[a + j];
          seg[o + 3 + j] = ok ? pos[b + j] : pos[a + j];          // degenerate (invisible) when an end is missing
          col[o + j] = c0[j]; col[o + 3 + j] = c1[j];
        }
      }
    }
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute('position', new THREE.BufferAttribute(seg, 3));
    lgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    lgeo.setDrawRange(0, 0);
    lines = new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
    lines.frustumCulled = false;
  }
  const scene = new THREE.Scene();
  scene.add(points);
  if (lines) scene.add(lines);
  return { scene, points, lines, data };
}

function disposePane(p) {
  p.points.geometry.dispose(); p.points.material.dispose();
  if (p.lines) { p.lines.geometry.dispose(); p.lines.material.dispose(); }
}

/* ------------------------------------------------------------------ one viewer */
function createViewer(root) {
  const part = role => root.querySelector(`[data-role="${role}"]`);
  const stage = part('stage'), canvas = part('canvas');
  const BASE = root.dataset.base;
  const ui = {
    chips: part('chips'), play: part('play'), slider: part('slider'), label: part('label'),
    size: part('size'), reset: part('reset'), frame: part('frame'), views: part('views'), fallback: part('fallback'),
    tags: [...stage.querySelectorAll('.pane-tag')],
    status: [...stage.querySelectorAll('.pane-status')],
  };
  const state = { manifest: null, clip: null, t: 0, playing: !reduceMotion, size: +(ui.size?.value || 3),
                  autoFit: true, vertical: false, visible: false, lastTick: 0, request: 0,
                  sway: false, swayStart: 0, swayAngle: 0 };
  const cache = new Map();   // url -> Promise<decoded>
  const panes = PANES.map(() => null);
  let renderer, camera, controls, home = null;

  function load(clip, key) {
    const meta = clip.methods[key];
    const url = `${BASE}${clip.id}/${meta.file}`;
    if (!cache.has(url)) cache.set(url, fetchBuffer(url).then(buf => decode(buf, meta, clip.T)));
    return cache.get(url);
  }

  function setFrame(pane, t) {
    if (!pane) return;
    const { N, pos, valid, K } = pane.data;
    const attr = pane.points.geometry.attributes;
    attr.position.array.set(pos.subarray(t * N * 3, (t + 1) * N * 3));
    attr.position.needsUpdate = true;
    attr.aValid.array.set(valid.subarray(t * N, (t + 1) * N));
    attr.aValid.needsUpdate = true;
    if (pane.lines) pane.lines.geometry.setDrawRange(0, 2 * K * t);      // tracks are always shown, up to frame t
  }

  /* ---------------------------------------------------------------- renderer */
  function initThree() {
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    } catch (e) {
      return fail('The interactive viewer needs WebGL, which is not available in this browser.');
    }
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setScissorTest(true);
    camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 0.8;
    controls.screenSpacePanning = true;
    controls.addEventListener('start', () => { state.autoFit = false; state.sway = false; });
    new ResizeObserver(layout).observe(stage);
    layout();
    return true;
  }

  function grid() {          // panes per row: one column when narrow, 2 × 2 for four panes, otherwise a single row
    const n = PANES.length;
    const cols = state.vertical ? 1 : (n === 4 ? 2 : n);
    return { cols, rows: Math.ceil(n / cols) };
  }

  function layout() {
    if (!renderer || !state.clip) return;
    const w = stage.clientWidth;
    state.vertical = w < 720;
    const { cols, rows } = grid();
    const aspect = state.clip.aspect;
    const paneAspect = state.vertical ? aspect
      : rows > 1 ? Math.min(Math.max(aspect, PANE_ASPECT_GRID[0]), PANE_ASPECT_GRID[1]) : Math.min(aspect, PANE_ASPECT_MAX);
    stage.style.aspectRatio = `${(cols * paneAspect / rows).toFixed(3)} / 1`;
    const h = stage.clientHeight;
    renderer.setSize(w, h, false);
    const pw = w / cols, ph = h / rows;
    camera.aspect = pw / ph;
    camera.updateProjectionMatrix();
    if (state.autoFit && home) fitView();
    // Pane tags, loading labels and the dividers between panes.
    const at = i => [(i % cols) * pw, Math.floor(i / cols) * ph];
    ui.tags.forEach((tag, i) => {
      const [x, y] = at(i);
      tag.style.left = `${x + 12}px`; tag.style.top = `${y + 12}px`;
    });
    ui.status.forEach((s, i) => {
      const [x, y] = at(i);
      s.style.left = `${x + pw / 2}px`; s.style.top = `${y + ph / 2}px`;
    });
    stage.querySelectorAll('.pane-line').forEach(l => l.remove());
    const line = css => { const l = document.createElement('div'); l.className = 'pane-line'; l.style.cssText = css; stage.append(l); };
    for (let c = 1; c < cols; c++) line(`left:${c * pw}px;top:0;bottom:0;width:1px`);
    for (let r = 1; r < rows; r++) line(`top:${r * ph}px;left:0;right:0;height:1px`);
  }

  function render() {
    if (!renderer) return;
    const w = stage.clientWidth, h = stage.clientHeight, n = PANES.length;
    const { cols, rows } = grid();
    const pw = w / cols, ph = h / rows;
    const zoom = home ? home.distance / camera.position.distanceTo(controls.target) : 1;
    const scale = renderer.getPixelRatio() * (ph / 300) * Math.min(4, Math.max(0.5, zoom));   // paper: 3 px on a 300 px panel
    for (let i = 0; i < n; i++) {
      const x = (i % cols) * pw, yTop = Math.floor(i / cols) * ph;
      const y = h - yTop - ph;                       // WebGL origin is bottom-left
      renderer.setViewport(x, y, pw, ph);
      renderer.setScissor(x, y, pw, ph);
      renderer.clear();
      const pane = panes[i];
      if (!pane) continue;
      const u = pane.points.material.uniforms;
      u.uSize.value = state.size; u.uScale.value = scale;
      renderer.render(pane.scene, camera);
    }
  }

  function frameLoop(now) {
    requestAnimationFrame(frameLoop);
    if (!state.visible || !state.clip) return;
    const T = state.clip.T;
    if (state.playing && T > 1 && panes[0]) {
      const fps = T > 16 ? 12 : 8;
      const hold = state.t === T - 1 ? 700 : 0;                  // pause on the last frame before looping
      if (now - state.lastTick > 1000 / fps + hold) {
        state.lastTick = now;
        showFrame((state.t + 1) % T);
      }
    }
    if (state.sway && panes[0]) {          // a static scene turns a little until the first drag, to show its depth
      if (!state.swayStart) state.swayStart = now;
      const angle = SWAY.angle * Math.sin((now - state.swayStart) * 2 * Math.PI / SWAY.period);
      const offset = camera.position.clone().sub(controls.target).applyAxisAngle(UP, angle - state.swayAngle);
      camera.position.copy(controls.target).add(offset);
      state.swayAngle = angle;
    }
    controls.update();
    render();
  }

  /* ---------------------------------------------------------------- UI */
  function fail(message) {
    ui.fallback.textContent = message;
    ui.fallback.hidden = false;
    return false;
  }

  function showFrame(t) {
    state.t = t;
    panes.forEach(p => setFrame(p, t));
    const T = state.clip.T, src = state.clip.source_frames?.[t];
    if (ui.slider) ui.slider.value = String(t);
    if (ui.label) ui.label.textContent = `t = ${t + 1} / ${T}`;
    if (ui.frame) {
      ui.frame.setAttribute('aria-label', `Input frame ${t + 1} of ${T}${src != null && src !== t ? ` (video frame ${src + 1})` : ''}`);
      ui.frame.style.backgroundPosition = `0 ${T > 1 ? (t / (T - 1)) * 100 : 0}%`;
    }
  }

  function setStatus(i, text) {
    ui.status[i].hidden = !text;
    if (text) ui.status[i].textContent = text;
  }

  function resetView() {
    const clip = state.clip;
    const center = new THREE.Vector3(...clip.center);
    const offset = new THREE.Vector3(...(clip.eye || [0, 0, 0])).sub(center);   // start: the source camera at the origin,
    const sph = new THREE.Spherical().setFromVector3(offset);                   // or clip.eye for wide-baseline scenes
    const view = { ...VIEW, ...(clip.view || {}) };
    sph.phi = Math.min(Math.PI - 0.05, Math.max(0.05, sph.phi - THREE.MathUtils.degToRad(view.elev)));
    sph.theta += THREE.MathUtils.degToRad(view.azim);
    sph.radius *= view.dist;
    camera.fov = Math.min(80, (clip.fov_y || 45) * 1.05);
    camera.position.copy(center).add(new THREE.Vector3().setFromSpherical(sph));
    camera.up.set(0, 1, 0);
    controls.target.copy(center);
    camera.updateProjectionMatrix();
    controls.update();
    home = { distance: camera.position.distanceTo(center) };
    state.autoFit = true;
    state.sway = clip.T === 1 && !reduceMotion;
    state.swayStart = 0; state.swayAngle = 0;
    fitView();
  }

  /* Paper framing (eval_teaser/render.make_view): after the orbit, aim at the content and fit the field of view
     to the 1st-99th percentile of the VGGM-Nexus points at t = s plus its tracks over the clip. Scenes with
     clip.fit_radius are fitted to the points within that distance of the centre (the paper's ETH3D rule). */
  function fitView() {
    const pane = panes[0];
    if (!pane || !state.autoFit) return;
    const { N, T, K, pos, valid, tails } = pane.data, s = state.clip.s || 0;
    const c = state.clip.center, R = state.clip.fit_radius;
    camera.updateMatrixWorld();
    const inv = camera.matrixWorldInverse, v = new THREE.Vector3(), a = [], b = [];
    const add = j => {
      v.set(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
      if (R && Math.hypot(v.x - c[0], v.y - c[1], v.z - c[2]) > R) return;
      v.applyMatrix4(inv);
      if (v.z < -1e-3) { a.push(v.x / -v.z); b.push(v.y / -v.z); }
    };
    for (let i = 0, step = Math.max(1, Math.floor(N / 6000)); i < N; i += step) if (valid[s * N + i]) add(s * N + i);
    for (let k = 0; k < K; k += 3) for (let t = 0; t < T; t += 2) if (valid[t * N + tails[k]]) add(t * N + tails[k]);
    if (a.length < 10) return;
    const q = (arr, p) => { const sorted = Float64Array.from(arr).sort(); return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]; };
    const [alo, ahi, blo, bhi] = [q(a, .01), q(a, .99), q(b, .01), q(b, .99)];
    const dist = camera.position.distanceTo(controls.target);
    const dir = new THREE.Vector3((alo + ahi) / 2, (blo + bhi) / 2, -1).normalize().transformDirection(camera.matrixWorld);
    controls.target.copy(camera.position).addScaledVector(dir, dist);
    const half = 1.18 * Math.max((bhi - blo) / 2, (ahi - alo) / 2 / camera.aspect);
    camera.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(half)), 8, 100);
    camera.updateProjectionMatrix();
    controls.update();
    home = { distance: dist };
  }

  async function showPane(i, key) {
    const clip = state.clip, req = state.request;
    setStatus(i, 'Loading…');
    try {
      const data = await load(clip, key);
      if (req !== state.request) return;                                         // superseded
      if (panes[i]) disposePane(panes[i]);
      panes[i] = buildPane(data);
      setFrame(panes[i], state.t);
      setStatus(i, null);
      if (i === 0) fitView();
    } catch (e) {
      console.error(e);
      setStatus(i, location.protocol === 'file:' ? 'Serve the page over HTTP to load 3D data' : 'Could not load data');
    }
  }

  const stripCount = clip => clip.views || clip.T;      // images in <clip>/frames.jpg: input views or video frames

  function selectClip(clip) {
    state.clip = clip;
    state.request += 1;
    panes.forEach((p, i) => { if (p) disposePane(p); panes[i] = null; });
    [...ui.chips.children].forEach(c => c.setAttribute('aria-selected', String(c.dataset.id === clip.id)));
    const strip = `url("${BASE}${clip.id}/frames.jpg")`, n = stripCount(clip);
    if (ui.slider) ui.slider.max = String(clip.T - 1);
    if (ui.frame) {
      ui.frame.style.backgroundImage = strip;
      ui.frame.style.backgroundSize = `100% ${n * 100}%`;
      ui.frame.style.aspectRatio = `${clip.frame_size[0]} / ${clip.frame_size[1]}`;
    }
    if (ui.views) {                                     // every input view of a static scene
      ui.views.setAttribute('aria-label', `The ${n} input views of this scene`);
      ui.views.replaceChildren(...Array.from({ length: n }, (_, v) => {
        const thumb = document.createElement('span');
        thumb.className = 'view-thumb';
        thumb.style.backgroundImage = strip;
        thumb.style.backgroundSize = `100% ${n * 100}%`;
        thumb.style.backgroundPosition = `0 ${n > 1 ? (v / (n - 1)) * 100 : 0}%`;
        thumb.style.aspectRatio = `${clip.frame_size[0]} / ${clip.frame_size[1]}`;
        return thumb;
      }));
    }
    layout();
    resetView();
    showFrame(0);
    PANES.forEach((key, i) => showPane(i, key));
  }

  function setPlaying(on) {
    state.playing = on;
    if (!ui.play) return;
    ui.play.classList.toggle('paused', !on);
    ui.play.setAttribute('aria-label', on ? 'Pause' : 'Play');
  }

  function buildChips(clips) {
    ui.chips.replaceChildren(...clips.map(clip => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'clip-chip'; b.dataset.id = clip.id;
      b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', 'false');
      const thumb = document.createElement('span');
      thumb.className = 'clip-thumb';
      thumb.style.backgroundImage = `url("${BASE}${clip.id}/frames.jpg")`;
      thumb.style.backgroundSize = `100% ${stripCount(clip) * 100}%`;
      const text = document.createElement('span');
      text.innerHTML = `${clip.title}<small>${clip.source} · ${clip.views ? `${clip.views} views` : `${clip.T} frames`}</small>`;
      b.append(thumb, text);
      b.addEventListener('click', () => selectClip(clip));
      return b;
    }));
  }

  function wireControls() {
    ui.play?.addEventListener('click', () => setPlaying(!state.playing));
    ui.slider?.addEventListener('input', () => { setPlaying(false); showFrame(+ui.slider.value); });
    ui.size?.addEventListener('input', () => { state.size = +ui.size.value; });
    ui.reset?.addEventListener('click', resetView);
    stage.addEventListener('keydown', e => {
      if (!state.clip || state.clip.T < 2) return;
      if (e.key === ' ') { e.preventDefault(); setPlaying(!state.playing); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault(); setPlaying(false);
        showFrame((state.t + (e.key === 'ArrowRight' ? 1 : -1) + state.clip.T) % state.clip.T);
      }
    });
    canvas.addEventListener('pointerdown', () => stage.focus({ preventScroll: true }));
  }

  async function start() {
    if (!initThree()) return;
    wireControls();
    setPlaying(state.playing);
    try {
      const res = await fetch(`${BASE}manifest.json`);
      if (!res.ok) throw new Error(res.status);
      state.manifest = await res.json();
    } catch (e) {
      return fail(location.protocol === 'file:'
        ? 'The 3D viewer loads data files, which browsers block for pages opened from disk. Serve this folder over HTTP, e.g. "python -m http.server", or view the published page.'
        : 'Could not load the viewer data.');
    }
    buildChips(state.manifest.clips);
    selectClip(state.manifest.clips[0]);
    requestAnimationFrame(frameLoop);
  }

  // Start when the viewer is about to scroll into view; pause rendering when it leaves.
  let started = false;
  new IntersectionObserver(entries => {
    for (const e of entries) {
      state.visible = e.isIntersecting;
      if (e.isIntersecting && !started) { started = true; start(); }
    }
  }, { rootMargin: '300px 0px' }).observe(root);
}

document.querySelectorAll('.viewer[data-base]').forEach(createViewer);
