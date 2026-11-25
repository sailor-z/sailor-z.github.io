/* Virtual tour: a phone video through the Visual Geometry Group building, reconstructed by VGGM-Nexus
   (outputs/long_sequence_2026-11-24, IMG_6046, chunks of 64 frames). Data from tools/export_tour_data.py:
     <base>manifest.json, points_<k>.bin.gz (int16 xyz | uint8 rgb | uint8 moving per point, frame-ordered),
     tracks.bin.gz (uint16 start[K] | uint16 length[K] | int16 xyz and uint8 valid per track frame), frames_<k>.jpg.
   The reconstruction builds up over time: every frame adds its points; points of moving objects are shown only
   at their own frame. Tracks of moving people are drawn from where each was found, coloured by time along the
   track (turbo, as the 4D viewer), and fade out when the person is gone. The camera path is drawn as it goes.
   Views: Overview (the whole building), Follow (behind the predicted camera); dragging orbits freely. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const UP = new THREE.Vector3(0, 1, 0);
const TAIL = 100;                 // frames of trail kept behind the current frame (10 s)
const GRACE = 3;                  // frames a finished track stays on screen
const SMOOTH = 6;                 // half window (frames) of the camera smoothing used by Follow
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

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

const TURBO_GLSL = `
  vec3 turbo(float x) {            // Google's polynomial approximation, sRGB
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(
      dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2(-152.94239396, 59.28637943)),
      dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2(4.27729857, 2.82956604)),
      dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2(-89.90310912, 27.34824973))), 0.0, 1.0);
  }`;

function createTour(root) {
  const part = role => root.querySelector(`[data-role="${role}"]`);
  const stage = part('stage'), canvas = part('canvas'), BASE = root.dataset.tour;
  const ui = { play: part('play'), slider: part('slider'), label: part('label'), size: part('size'),
               frame: part('frame'), status: part('status'), fallback: part('fallback'),
               views: [...root.querySelectorAll('[data-view]')] };
  const state = { t: 0, playing: false, mode: 'overview', size: +(ui.size?.value || 2.5), loaded: 0, ready: false,
                  visible: false, last: 0, eye: null, look: null };
  let renderer, camera, controls, m, T, fps;
  let points, path, frustum, trails, dots, frameEnd, smooth, sheets = [];

  /* ---------------------------------------------------------------- materials */
  const pointMaterial = new THREE.ShaderMaterial({
    uniforms: { uT: { value: 0 }, uSize: { value: 2.5 }, uScale: { value: 1 } },
    vertexShader: `
      attribute vec3 aColor; attribute float aFrame; attribute float aMove;
      uniform float uT, uSize, uScale;
      varying vec3 vColor;
      void main() {
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        bool shown = aFrame <= uT + 0.5 && (aMove < 0.5 || abs(aFrame - uT) < 0.5);
        gl_PointSize = shown ? uSize * uScale * clamp(4.0 / max(-mv.z, 0.01), 0.6, 1.6) : 0.0;
        if (!shown) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() { vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25) discard; gl_FragColor = vec4(vColor, 1.0); }`,
  });
  const trackUniforms = { uT: { value: 0 }, uDot: { value: 3 }, uHead: { value: 8 } };
  const trackVertex = `
    ${TURBO_GLSL}
    attribute float aTime, aStart, aEnd;
    uniform float uT, uDot, uHead;
    varying vec3 vColor; varying float vKeep;
    void main() {
      vKeep = (aTime <= uT + 0.5 && aTime >= uT - ${TAIL.toFixed(1)} && uT <= aEnd + ${GRACE.toFixed(1)}) ? 1.0 : 0.0;
      vColor = turbo(0.10 + 0.85 * (aTime - aStart) / max(aEnd - aStart, 1.0));
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_Position.z -= 0.003 * gl_Position.w;            // in front of the reconstructed points they lie on
      gl_PointSize = vKeep > 0.5 ? (abs(aTime - uT) < 0.5 ? uHead : uDot) : 0.0;
    }`;
  const trailMaterial = new THREE.ShaderMaterial({ uniforms: trackUniforms, vertexShader: trackVertex, transparent: true, depthWrite: false,
    fragmentShader: `varying vec3 vColor; varying float vKeep; void main() { if (vKeep < 0.5) discard; gl_FragColor = vec4(vColor, 0.9); }` });
  const dotMaterial = new THREE.ShaderMaterial({ uniforms: trackUniforms, vertexShader: trackVertex, depthWrite: false,
    fragmentShader: `varying vec3 vColor; varying float vKeep;
      void main() { vec2 c = gl_PointCoord - 0.5; if (vKeep < 0.5 || dot(c, c) > 0.25) discard; gl_FragColor = vec4(vColor, 1.0); }` });

  /* ---------------------------------------------------------------- scene */
  function initThree() {
    try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); }
    catch (e) { return fail('The virtual tour needs WebGL, which is not available in this browser.'); }
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    camera = new THREE.PerspectiveCamera(55, 1, 0.05, 400);
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true; controls.dampingFactor = 0.12; controls.rotateSpeed = 0.7; controls.zoomSpeed = 0.8;
    controls.screenSpacePanning = true;
    controls.addEventListener('start', () => setMode('free'));
    new ResizeObserver(resize).observe(stage);
    resize();
    return true;
  }

  function resize() {
    if (!renderer) return;
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }

  const vec = (a, i) => new THREE.Vector3(a[i][0], a[i][1], a[i][2]);

  function buildCameras() {
    const P = m.cameras.position, F = m.cameras.forward;
    const line = new Float32Array(T * 3);
    P.forEach((p, i) => line.set(p, 3 * i));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(line, 3));
    path = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x8a949b }));
    path.frustumCulled = false;
    // Follow uses a smoothed path: positions and level viewing directions averaged over ±SMOOTH frames.
    smooth = { pos: [], fwd: [] };
    for (let i = 0; i < T; i++) {
      const p = new THREE.Vector3(), f = new THREE.Vector3();
      let n = 0;
      for (let j = Math.max(0, i - SMOOTH); j <= Math.min(T - 1, i + SMOOTH); j++, n++) { p.add(vec(P, j)); f.add(vec(F, j)); }
      p.divideScalar(n);
      f.addScaledVector(UP, -f.dot(UP)).normalize();
      smooth.pos.push(p); smooth.fwd.push(f);
    }
    // The current camera as a small frustum (portrait, as the video).
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
    frustum = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0xd55e00 }));
    frustum.frustumCulled = false;
  }

  function placeFrustum(t) {
    const P = m.cameras.position, F = m.cameras.forward, U = m.cameras.up;
    const o = vec(P, t), f = vec(F, t).normalize(), u = vec(U, t).normalize(), r = new THREE.Vector3().crossVectors(f, u).normalize();
    const d = 0.45, hh = 0.22, hw = hh * m.frame_size[0] / m.frame_size[1];
    const c = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([x, y]) => o.clone().addScaledVector(f, d).addScaledVector(r, x * hw).addScaledVector(u, y * hh));
    const seg = [];
    c.forEach((q, i) => { seg.push(o, q, q, c[(i + 1) % 4]); });
    const arr = frustum.geometry.attributes.position.array;
    seg.forEach((v, i) => arr.set([v.x, v.y, v.z], 3 * i));
    frustum.geometry.attributes.position.needsUpdate = true;
  }

  function buildPoints() {
    const total = m.blocks.reduce((s, b) => s + b.points, 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(total * 3), 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(new Uint8Array(total * 3), 3, true));
    g.setAttribute('aFrame', new THREE.BufferAttribute(new Float32Array(total), 1));
    g.setAttribute('aMove', new THREE.BufferAttribute(new Float32Array(total), 1));
    g.setDrawRange(0, 0);
    points = new THREE.Points(g, pointMaterial);
    points.frustumCulled = false;
    frameEnd = new Uint32Array(T);           // points of frames 0..t end at frameEnd[t]
    let end = 0;
    m.blocks.forEach(b => b.counts.forEach((c, i) => { end += c; frameEnd[b.first + i] = end; }));
  }

  async function loadBlocks() {
    const lo = m.lo, span = m.hi.map((h, a) => (h - lo[a]) / 65534);
    const attr = points.geometry.attributes;
    let at = 0;
    for (const b of m.blocks) {
      const buf = await fetchBuffer(`${BASE}${b.file}`);
      const n = b.points, q = new Int16Array(buf, 0, 3 * n), rgb = new Uint8Array(buf, 6 * n, 3 * n), mv = new Uint8Array(buf, 9 * n, n);
      const pos = attr.position.array;
      for (let i = 0; i < 3 * n; i++) pos[3 * at + i] = lo[i % 3] + (q[i] + 32767) * span[i % 3];
      attr.aColor.array.set(rgb, 3 * at);
      let k = at;
      b.counts.forEach((c, f) => { attr.aFrame.array.fill(b.first + f, k, k + c); k += c; });
      for (let i = 0; i < n; i++) attr.aMove.array[at + i] = mv[i];
      at += n;
      for (const name of ['position', 'aColor', 'aFrame', 'aMove']) attr[name].needsUpdate = true;
      state.loaded = b.first + b.frames;
      if (!state.ready) { showFrame(state.loaded - 1); setStatus(`Loading ${state.loaded} / ${T} frames`); }
    }
  }

  async function loadTracks() {
    const buf = await fetchBuffer(`${BASE}${m.tracks.file}`);
    const K = m.tracks.count;
    if (!K) return;
    const start = new Uint16Array(buf, 0, K), len = new Uint16Array(buf, 2 * K, K);
    const L = len.reduce((s, v) => s + v, 0);
    const q = new Int16Array(buf, 4 * K, 3 * L), ok = new Uint8Array(buf, 4 * K + 6 * L, L);
    const lo = m.lo, span = m.hi.map((h, a) => (h - lo[a]) / 65534);
    const P = i => [0, 1, 2].map(a => lo[a] + (q[3 * i + a] + 32767) * span[a]);
    const seg = [], segAttr = [], dot = [], dotAttr = [];
    let at = 0;
    for (let k = 0; k < K; k++) {
      const s = start[k], e = s + len[k] - 1;
      for (let j = 0; j < len[k]; j++) {
        if (!ok[at + j]) continue;
        dot.push(...P(at + j)); dotAttr.push(s + j, s, e);
        if (j + 1 < len[k] && ok[at + j + 1]) { seg.push(...P(at + j), ...P(at + j + 1)); segAttr.push(s + j + 1, s, e, s + j + 1, s, e); }
      }
      at += len[k];
    }
    const make = (xyz, extra) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(xyz), 3));
      const ex = Float32Array.from(extra), n = ex.length / 3;
      for (const [name, o] of [['aTime', 0], ['aStart', 1], ['aEnd', 2]]) {
        const v = new Float32Array(n);
        for (let i = 0; i < n; i++) v[i] = ex[3 * i + o];
        g.setAttribute(name, new THREE.BufferAttribute(v, 1));
      }
      return g;
    };
    trails = new THREE.LineSegments(make(seg, segAttr), trailMaterial);
    dots = new THREE.Points(make(dot, dotAttr), dotMaterial);
    trails.frustumCulled = dots.frustumCulled = false;
    trails.renderOrder = dots.renderOrder = 2;
    scene.add(trails, dots);
  }

  const scene = new THREE.Scene();

  /* ---------------------------------------------------------------- views */
  function sceneBox() {      // 2nd to 98th percentile of the loaded points (the camera path while none are loaded)
    const n = state.loaded ? frameEnd[state.loaded - 1] : 0, box = new THREE.Box3();
    if (n < 2000) { m.cameras.position.forEach(p => box.expandByPoint(new THREE.Vector3(...p))); return box.expandByScalar(3); }
    const pos = points.geometry.attributes.position.array, step = Math.max(1, Math.floor(n / 30000)), axes = [[], [], []];
    for (let i = 0; i < n; i += step) for (let a = 0; a < 3; a++) axes[a].push(pos[3 * i + a]);
    const q = (arr, f) => { arr.sort((x, y) => x - y); return arr[Math.floor(f * (arr.length - 1))]; };
    const lo = axes.map(v => q(v, 0.02)), hi = axes.map(v => q(v, 0.98));
    return new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi));
  }

  function overview(immediate) {
    // The whole building, from above and behind the start of the walk, fitted to the reconstructed points.
    const box = sceneBox(), centre = box.getCenter(new THREE.Vector3());
    const radius = 0.5 * box.getSize(new THREE.Vector3()).length();
    const f0 = smooth.fwd[0];
    const dir = f0.clone().multiplyScalar(-Math.cos(0.9)).addScaledVector(UP, Math.sin(0.9))
      .addScaledVector(new THREE.Vector3().crossVectors(f0, UP), 0.35).normalize();
    const half = THREE.MathUtils.degToRad(camera.fov / 2), fit = Math.min(Math.tan(half), Math.tan(half) * camera.aspect);
    aim(centre.clone().addScaledVector(dir, 0.8 * radius / fit), centre, immediate);
  }

  function followTarget(t) {
    const p = smooth.pos[t], f = smooth.fwd[t];
    // Just behind and above the walking camera: indoors, a higher or farther chase view ends up in the ceiling or a wall.
    return { eye: p.clone().addScaledVector(f, -1.6).addScaledVector(UP, 0.3), look: p.clone().addScaledVector(f, 3.0).addScaledVector(UP, -0.15) };
  }

  function aim(eye, look, immediate) {
    if (immediate || !state.eye) { state.eye = eye.clone(); state.look = look.clone(); camera.position.copy(eye); controls.target.copy(look); }
    state.goal = { eye: eye.clone(), look: look.clone() };
  }

  function setMode(mode) {
    state.mode = mode;
    if (camera && state.eye) { state.eye.copy(camera.position); state.look.copy(controls.target); }   // ease from where the view is
    ui.views.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === mode)));
    if (mode === 'overview' && smooth) overview(false);
    if (mode === 'follow' && smooth) { const g = followTarget(state.t); aim(g.eye, g.look, false); }
    if (mode === 'free') state.goal = null;
  }

  /* ---------------------------------------------------------------- frames */
  function setStatus(text) { ui.status.hidden = !text; if (text) ui.status.textContent = text; }
  function fail(text) { ui.fallback.textContent = text; ui.fallback.hidden = false; return false; }
  const clock = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`;

  function showFrame(t) {
    t = Math.max(0, Math.min(T - 1, t));
    state.t = t;
    const shown = Math.min(t, state.loaded - 1);
    points.geometry.setDrawRange(0, shown >= 0 ? frameEnd[shown] : 0);
    pointMaterial.uniforms.uT.value = t;
    trackUniforms.uT.value = t;
    path.geometry.setDrawRange(0, t + 1);
    placeFrustum(t);
    ui.slider.value = String(t);
    ui.label.textContent = `${clock(t / fps)} / ${clock((T - 1) / fps)}`;
    const sheet = m.sheets.find(s => t >= s.first && t < s.first + s.frames);
    if (sheet && ui.frame) {
      const i = t - sheet.first, rows = Math.ceil(sheet.frames / sheet.columns), col = i % sheet.columns, row = Math.floor(i / sheet.columns);
      ui.frame.style.backgroundImage = `url("${BASE}${sheet.file}")`;
      ui.frame.style.backgroundSize = `${sheet.columns * 100}% ${rows * 100}%`;
      ui.frame.style.backgroundPosition = `${sheet.columns > 1 ? col / (sheet.columns - 1) * 100 : 0}% ${rows > 1 ? row / (rows - 1) * 100 : 0}%`;
      ui.frame.setAttribute('aria-label', `Input video frame at ${clock(t / fps)}`);
    }
    if (state.mode === 'follow') { const g = followTarget(t); aim(g.eye, g.look, false); }
  }

  function setPlaying(on) {
    state.playing = on;
    ui.play.classList.toggle('paused', !on);
    ui.play.setAttribute('aria-label', on ? 'Pause' : 'Play the tour');
  }

  function loop(now) {
    requestAnimationFrame(loop);
    if (!state.visible || !m) return;
    const dt = Math.min(0.1, (now - (state.prev || now)) / 1000);
    state.prev = now;
    if (state.playing && now - state.last > 1000 / fps) {
      state.last = now - state.last > 250 ? now : state.last + 1000 / fps;      // keeps pace with the video
      if (state.t >= T - 1) setPlaying(false); else showFrame(state.t + 1);
    }
    if (state.goal) {                               // ease the view towards its goal
      const k = 1 - Math.exp(-dt / (state.mode === 'follow' ? 0.45 : 0.6));
      state.eye.lerp(state.goal.eye, k); state.look.lerp(state.goal.look, k);
      camera.position.copy(state.eye); controls.target.copy(state.look);
    }
    controls.update();
    const ratio = state.playing ? 1 : Math.min(devicePixelRatio || 1, 2);      // full resolution when still
    if (renderer.getPixelRatio() !== ratio) { renderer.setPixelRatio(ratio); resize(); }
    const scale = ratio * Math.max(0.7, stage.clientHeight / 500);
    pointMaterial.uniforms.uSize.value = state.size; pointMaterial.uniforms.uScale.value = scale;
    trackUniforms.uDot.value = 3 * renderer.getPixelRatio(); trackUniforms.uHead.value = 7 * renderer.getPixelRatio();
    renderer.render(scene, camera);
  }

  function wire() {
    ui.play.addEventListener('click', () => {
      if (!state.ready) return;
      if (!state.playing && state.t >= T - 1) { showFrame(0); setMode('follow'); }       // the walk, from its start
      else if (!state.playing && state.mode === 'free') setMode('follow');
      setPlaying(!state.playing);
    });
    ui.slider.addEventListener('input', () => { setPlaying(false); showFrame(+ui.slider.value); });
    ui.size?.addEventListener('input', () => { state.size = +ui.size.value; });
    ui.views.forEach(b => b.addEventListener('click', () => setMode(b.dataset.view)));
    stage.addEventListener('keydown', e => {
      if (!state.ready) return;
      if (e.key === ' ') { e.preventDefault(); ui.play.click(); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); showFrame(state.t + (e.key === 'ArrowRight' ? 1 : -1)); }
    });
    canvas.addEventListener('pointerdown', () => stage.focus({ preventScroll: true }));
  }

  async function start() {
    if (!initThree()) return;
    try {
      const res = await fetch(`${BASE}manifest.json`);
      if (!res.ok) throw new Error(res.status);
      m = await res.json();
    } catch (e) {
      return fail(location.protocol === 'file:'
        ? 'The virtual tour loads data files, which browsers block for pages opened from disk. Serve this folder over HTTP, e.g. "python -m http.server", or view the published page.'
        : 'Could not load the tour data.');
    }
    T = m.frames; fps = m.fps || 10;
    ui.slider.max = String(T - 1);
    ui.frame.style.aspectRatio = `${m.thumb[0]} / ${m.thumb[1]}`;
    m.sheets.forEach(s => { const img = new Image(); img.src = `${BASE}${s.file}`; sheets.push(img); });   // preload
    buildCameras(); buildPoints();
    scene.add(points, path, frustum);
    wire();
    setPlaying(false);
    setMode('overview');
    overview(true);
    requestAnimationFrame(loop);
    setStatus('Loading…');
    try {
      await Promise.all([loadBlocks(), loadTracks()]);
    } catch (e) {
      console.error(e);
      return setStatus('Could not load the tour data');
    }
    state.ready = true;
    setStatus(null);
    showFrame(T - 1);                               // first view: the whole reconstruction; Play starts the walk
    if (state.mode === 'overview') overview(false);
  }

  let started = false;
  new IntersectionObserver(entries => {
    for (const e of entries) {
      state.visible = e.isIntersecting;
      if (e.isIntersecting && !started) { started = true; start(); }
    }
  }, { rootMargin: '300px 0px' }).observe(root);
}

document.querySelectorAll('.viewer[data-tour]').forEach(createTour);
