/**
 * P15 — P-MAI Real-Time Voice Web Client (Gold Core UI)
 *
 * What changed vs. the previous version:
 *  - The whole UI is now built by this file (CSS + DOM injected on load),
 *    so index.html only needs:  <script src="app.js"></script>
 *  - New centre-piece: a canvas gold hologram core (wireframe sphere, spokes,
 *    ring bands with bokeh, tunnel eye, circuit traces, sparks) that reacts to the real audio level of the mic / MAI's voice.
 *  - Output audio now goes through its own AnalyserNode so the core can
 *    pulse with MAI's speech.
 *
 * Unchanged on purpose:
 *  - WebSocket protocol, event names and payloads
 *  - 16 kHz mono PCM capture, buffered full-response playback
 *  - Barge-in, "stop speaking" vs "cancel task"
 *
 * Voice is only a presentation / I/O layer. The backend remains
 * authoritative for routing, planning, tools, memory and task state.
 */

// ============================================================
// State machine
// ============================================================

const STATES = {
  DISCONNECTED: { label: "Disconnected", color: "#6f5a49" },
  CONNECTING: { label: "Connecting", color: "#ffc94d" },
  IDLE: { label: "Ready", color: "#ffbe3a" },
  LISTENING: { label: "Listening", color: "#ffe08a" },
  TRANSCRIBING: { label: "Understanding", color: "#ffd060" },
  PROCESSING: { label: "Thinking", color: "#ffb21f" },
  SPEAKING: { label: "Speaking", color: "#fff3cf" },
  INTERRUPTED: { label: "Interrupted", color: "#ff5a36" },
  ERROR: { label: "Something went wrong", color: "#ff3b30" },
  CLOSED: { label: "Closed", color: "#6f5a49" },
};

// ============================================================
// Core look per state (tweak hue / energy here)
//   hue    : HSL hue of the core (gold ≈ 38–48)
//   sat    : saturation %
//   bright : overall brightness multiplier
//   spin   : sphere rotation speed (rad/s)
//   energy : how strongly audio level deforms the core
//   swirl  : speed of rings / inner spiral
//   tick   : how far the spokes stretch with the spectrum
// ============================================================

const CORE_PROFILES = {
  DISCONNECTED: { hue: 36, sat: 18, bright: 0.45, spin: 0.03, energy: 0.1, swirl: 0.15, tick: 0.25 },
  CONNECTING: { hue: 44, sat: 85, bright: 0.7, spin: 0.15, energy: 0.25, swirl: 0.7, tick: 0.35 },
  IDLE: { hue: 42, sat: 100, bright: 0.85, spin: 0.08, energy: 0.3, swirl: 0.35, tick: 0.4 },
  LISTENING: { hue: 47, sat: 100, bright: 1.0, spin: 0.16, energy: 0.8, swirl: 0.55, tick: 0.9 },
  TRANSCRIBING: { hue: 43, sat: 100, bright: 1.0, spin: 0.45, energy: 0.55, swirl: 1.1, tick: 0.6 },
  PROCESSING: { hue: 38, sat: 100, bright: 1.0, spin: 0.75, energy: 0.65, swirl: 1.5, tick: 0.7 },
  SPEAKING: { hue: 44, sat: 100, bright: 1.15, spin: 0.3, energy: 1.0, swirl: 0.9, tick: 1.0 },
  INTERRUPTED: { hue: 20, sat: 95, bright: 0.8, spin: 0.04, energy: 0.2, swirl: 0.2, tick: 0.3 },
  ERROR: { hue: 4, sat: 95, bright: 0.7, spin: 0.02, energy: 0.15, swirl: 0.1, tick: 0.2 },
  CLOSED: { hue: 36, sat: 18, bright: 0.45, spin: 0.03, energy: 0.1, swirl: 0.15, tick: 0.25 },
};

// ============================================================
// UI markup + styles (injected)
// ============================================================

const MAI_CSS = `
:root{
  --void:#050302; --ink:#f5e6d0; --dim:#a8896b; --ember:#ffb21f;
  --amber:#ffc94d; --hot:#fff3cf; --rust:#7a4a0c;
  --line:rgba(255,190,70,.22); --state:#ffbe3a;
}
html,body{margin:0;height:100%;background:var(--void);overflow:hidden;}
body{font-family:'Chakra Petch','Segoe UI',system-ui,sans-serif;color:var(--ink);-webkit-font-smoothing:antialiased;}

#mai-root{
  position:fixed;inset:0;box-sizing:border-box;display:grid;
  grid-template-columns:minmax(0,1fr) clamp(300px,27vw,400px);
  grid-template-rows:auto minmax(0,1fr) auto;
  grid-template-areas:"top top" "stage feed" "ctrl feed";
  background:radial-gradient(ellipse 70% 80% at 36% 48%,#1c1204 0%,#0a0703 48%,#050302 100%);
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
}

.mai-top{grid-area:top;display:flex;justify-content:space-between;align-items:center;padding:18px 28px;}
.mai-brand{display:flex;align-items:center;gap:12px;font-weight:600;font-size:18px;letter-spacing:.22em;color:var(--hot);}
.mai-brand-mark{width:9px;height:9px;background:var(--ember);transform:rotate(45deg);box-shadow:0 0 12px var(--ember);}
.mai-conn{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--dim);}
.mai-conn-dot{width:8px;height:8px;border-radius:50%;background:#ff3b30;}

.mai-stage{grid-area:stage;position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:0;padding:0 2vw;}
.mai-hud{position:absolute;inset:4px 3%;pointer-events:none;}
.mai-hud .c{position:absolute;width:26px;height:26px;border:1px solid rgba(255,195,80,.45);}
.mai-hud .tl{top:0;left:0;border-right:0;border-bottom:0;}
.mai-hud .tr{top:0;right:0;border-left:0;border-bottom:0;}
.mai-hud .bl{bottom:0;left:0;border-right:0;border-top:0;}
.mai-hud .br{bottom:0;right:0;border-left:0;border-top:0;}
.mai-levelbar{position:absolute;top:10px;left:44px;width:min(34%,320px);height:5px;background:rgba(255,185,60,.12);overflow:hidden;}
.mai-levelbar b{display:block;height:100%;transform-origin:left center;transform:scaleX(0);
  background:repeating-linear-gradient(90deg,var(--ember) 0 6px,transparent 6px 9px);}
.mai-reticle{position:absolute;right:10px;bottom:10px;width:34px;height:34px;color:var(--amber);opacity:.7;animation:mai-spin 14s linear infinite;}
#mai-root[data-state="PROCESSING"] .mai-reticle,
#mai-root[data-state="TRANSCRIBING"] .mai-reticle{animation-duration:2.2s;opacity:1;}

.mai-core{position:relative;width:min(100%,64vh);aspect-ratio:1;flex:0 1 auto;}
.mai-core canvas{position:absolute;inset:0;width:100%;height:100%;display:block;}

.mai-status{display:flex;align-items:center;gap:10px;margin-top:4px;font-size:15px;font-weight:500;color:var(--hot);}
#mai-pulse{width:9px;height:9px;border-radius:50%;background:var(--state);box-shadow:0 0 14px var(--state);transition:background .3s,box-shadow .3s;}
#mai-root[data-state="LISTENING"] #mai-pulse,
#mai-root[data-state="SPEAKING"] #mai-pulse{animation:mai-blink 1s ease-in-out infinite;}
.mai-live{min-height:1.6em;max-width:56ch;margin:8px 0 0;text-align:center;color:var(--dim);font-size:16px;line-height:1.5;}

.mai-feedpanel{grid-area:feed;display:flex;flex-direction:column;min-height:0;margin:0 22px 22px 0;padding-left:20px;border-left:1px solid var(--line);}
.mai-feedpanel h2{margin:6px 0 14px;font-size:13px;font-weight:500;color:var(--dim);}
.mai-feed{flex:1;min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:18px;padding-right:6px;scrollbar-width:thin;scrollbar-color:var(--rust) transparent;}
.mai-feed:empty::before{content:'Say something to begin. Your conversation shows up here.';color:var(--dim);font-size:14px;line-height:1.6;}
.mai-msg{padding-left:12px;border-left:2px solid var(--rust);animation:mai-in .28s ease-out;}
.mai-msg-user{border-left-color:rgba(168,137,107,.5);color:var(--dim);}
.mai-msg-mai{border-left-color:var(--ember);color:var(--ink);}
.mai-msg-label{margin-bottom:4px;font-size:12px;color:var(--dim);}
.mai-msg-text{font-size:15px;line-height:1.55;white-space:pre-wrap;}

.mai-controls{grid-area:ctrl;position:relative;display:flex;align-items:center;justify-content:center;gap:28px;padding:10px 24px 26px;}
.mai-metrics{position:absolute;left:28px;bottom:26px;font-size:12px;color:var(--dim);}
.mai-ghost{padding:10px 16px;background:transparent;border:1px solid var(--line);border-radius:2px;color:var(--amber);font:inherit;font-size:14px;cursor:pointer;opacity:.4;transition:opacity .25s,border-color .2s,background .2s;}
.mai-ghost:hover{border-color:var(--amber);background:rgba(255,190,70,.08);}
#mai-root[data-state="SPEAKING"] #mai-btn-stop,
#mai-root[data-state="SPEAKING"] #mai-btn-cancel,
#mai-root[data-state="PROCESSING"] #mai-btn-cancel,
#mai-root[data-state="TRANSCRIBING"] #mai-btn-cancel{opacity:1;}

.mai-talk{display:flex;flex-direction:column;align-items:center;gap:8px;}
#mai-speak-label{font-size:12px;color:var(--dim);}
.mai-mic{position:relative;width:78px;height:78px;border-radius:50%;border:1.5px solid var(--amber);
  background:radial-gradient(circle at 50% 40%,#2a1a06,#0d0803);color:var(--amber);cursor:pointer;
  display:grid;place-items:center;transition:transform .15s,box-shadow .25s,background .25s,color .25s;}
.mai-mic:hover{transform:scale(1.04);box-shadow:0 0 28px rgba(255,178,31,.45);}
.mai-mic .stop-icon{display:none;}
.mai-mic.recording{background:var(--ember);color:#1a0a02;box-shadow:0 0 34px rgba(255,178,31,.65);}
.mai-mic.recording .mic-icon{display:none;}
.mai-mic.recording .stop-icon{display:block;}
.mai-mic.recording::after{content:'';position:absolute;inset:-2px;border-radius:50%;border:1.5px solid var(--ember);animation:mai-ring 1.4s ease-out infinite;}
.mai-mic:focus-visible,.mai-ghost:focus-visible{outline:2px solid var(--hot);outline-offset:3px;}

.mai-error{display:none;position:fixed;top:64px;left:50%;transform:translateX(-50%);z-index:10;max-width:min(90vw,560px);
  padding:10px 16px;border:1px solid #ff3b30;background:rgba(40,8,4,.92);color:#ffd9cf;font-size:14px;cursor:pointer;}

@keyframes mai-spin{to{transform:rotate(360deg);}}
@keyframes mai-blink{50%{opacity:.35;}}
@keyframes mai-in{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
@keyframes mai-ring{from{transform:scale(1);opacity:.9;}to{transform:scale(1.55);opacity:0;}}

@media (max-width:900px){
  #mai-root{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr) minmax(90px,26vh) auto;grid-template-areas:"top" "stage" "feed" "ctrl";}
  .mai-core{width:min(100%,46vh);}
  .mai-feedpanel{margin:0 18px;padding:10px 0 0;border-left:0;border-top:1px solid var(--line);}
  .mai-metrics{display:none;}
  .mai-controls{gap:14px;}
  .mai-ghost{padding:8px 10px;font-size:13px;}
}
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.001s !important;animation-iteration-count:1 !important;}
}
`;

const MAI_HTML = `
<header class="mai-top">
  <div class="mai-brand"><span class="mai-brand-mark"></span>MAI</div>
  <div class="mai-conn"><span id="mai-conn-dot" class="mai-conn-dot"></span><span id="mai-conn-text">Connecting</span></div>
</header>

<main class="mai-stage">
  <div class="mai-hud" aria-hidden="true">
    <i class="c tl"></i><i class="c tr"></i><i class="c bl"></i><i class="c br"></i>
    <div class="mai-levelbar"><b id="mai-level-fill"></b></div>
    <svg class="mai-reticle" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1">
      <circle cx="20" cy="20" r="17"/><circle cx="20" cy="20" r="10" stroke-dasharray="3 4"/>
      <circle cx="20" cy="20" r="2.5" fill="currentColor"/><path d="M20 0v6M20 34v6M0 20h6M34 20h6"/>
    </svg>
  </div>
  <div class="mai-core"><canvas id="mai-core-canvas" aria-hidden="true"></canvas></div>
  <div class="mai-status" role="status" aria-live="polite">
    <span id="mai-pulse"></span><span id="mai-state-text">Connecting</span>
  </div>
  <p id="mai-live-transcript" class="mai-live"></p>
</main>

<aside class="mai-feedpanel">
  <h2>Conversation</h2>
  <div id="mai-feed" class="mai-feed" aria-live="polite"></div>
</aside>

<footer class="mai-controls">
  <div id="mai-metrics" class="mai-metrics"></div>
  <button id="mai-btn-stop" class="mai-ghost" type="button" title="Stop speaking (Esc)">Stop speaking</button>
  <div class="mai-talk">
    <button id="mai-btn-speak" class="mai-mic" type="button" aria-label="Start talking">
      <svg class="mic-icon" viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>
      </svg>
      <svg class="stop-icon" viewBox="0 0 24 24" width="26" height="26" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
    </button>
    <span id="mai-speak-label">Tap or press Space to talk</span>
  </div>
  <button id="mai-btn-cancel" class="mai-ghost" type="button" title="Cancel the active task">Cancel task</button>
</footer>

<div id="mai-error" class="mai-error" role="alert"></div>
`;

function buildMaiUI() {
  // Replace any previous markup (keep script tags so app.js stays loaded).
  Array.from(document.body.children).forEach((n) => {
    if (n.tagName !== "SCRIPT") n.remove();
  });

  const font = document.createElement("link");
  font.rel = "stylesheet";
  font.href =
    "https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600&display=swap";
  document.head.appendChild(font);

  const style = document.createElement("style");
  style.id = "mai-style";
  style.textContent = MAI_CSS;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.id = "mai-root";
  root.dataset.state = "DISCONNECTED";
  root.innerHTML = MAI_HTML;
  document.body.appendChild(root);
  return root;
}

// ============================================================
// Small math helpers
// ============================================================

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rotX = (a) => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
const rotY = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
const rotZ = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];

function mulM(a, b) {
  return [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
    a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
}

// ============================================================
// Gold hologram core renderer
//
// Modelled on the reference: a fractured wireframe sphere, long
// straight spokes from the centre, thick glowing ring bands full of
// bokeh, circuit-like traces on the shell and a small tunnel "eye".
// Everything reacts to the real audio level / spectrum.
// ============================================================

class EmberCore {
  constructor(canvas, host, levelFill) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.host = host;
    this.levelFill = levelFill;

    this.reduced =
      !!(window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches);

    this.w = 0;
    this.h = 0;
    this.R = 100;

    this.time = 0;
    this.lastTs = 0;
    this.level = 0;
    this.freq = null;
    this.yaw = 0;
    this.swirlT = 0;
    this.tickRot = 0;
    this.hotA = 0;
    this.scan = 0;
    this.spawnAcc = 0;
    this._lastFill = -1;

    this.px = 0;
    this.py = 0;
    this.pz = 0;
    this.ps = 1;

    this.p = Object.assign({}, CORE_PROFILES.DISCONNECTED);
    this.ramp = new Array(8).fill("rgba(0,0,0,0)");

    this.sprite = null;
    this.spriteKey = "";

    this.dots = [];
    this.thin = [];
    this.mid = [];
    this.thick = [];
    this.rects = [];
    this.boke = [];

    this.buildGeometry();

    this.frame = this.frame.bind(this);
    this.resize();

    if (window.ResizeObserver) {
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    } else {
      window.addEventListener("resize", () => this.resize());
    }

    requestAnimationFrame(this.frame);
  }

  // ----------------------------------------------------------
  // Geometry (built once, deterministic)
  // ----------------------------------------------------------

  buildGeometry() {
    const r = mulberry32(2024);
    const TAU = Math.PI * 2;

    const unit = () => {
      const y = r() * 2 - 1;
      const a = r() * TAU;
      const s = Math.sqrt(1 - y * y);
      return [Math.cos(a) * s, y, Math.sin(a) * s];
    };

    const flags = (n, pSkip) => {
      const a = [];
      for (let i = 0; i < n; i++) a.push({ skip: r() < pSkip, h: r() });
      return a;
    };

    // Latitude circles of the wireframe shell
    this.lats = [];
    const LAT = 9;
    for (let i = 0; i < LAT; i++) {
      const phi = -Math.PI / 2 + ((i + 1) / (LAT + 1)) * Math.PI;
      this.lats.push({
        y: Math.sin(phi),
        rad: Math.cos(phi),
        segs: flags(48, 0.28),
        drift: (r() < 0.5 ? -1 : 1) * (0.05 + r() * 0.15),
      });
    }

    // Great circles through the poles (meridians)
    this.lons = [];
    const LON = 7;
    for (let i = 0; i < LON; i++) {
      const th = (i / LON) * Math.PI;
      this.lons.push({ c: Math.cos(th), s: Math.sin(th), segs: flags(64, 0.28) });
    }

    // Straight chords across the sphere (the "cracked glass" look)
    this.chords = [];
    for (let i = 0; i < 70; i++) {
      this.chords.push({ a: unit(), b: unit(), h: r() });
    }

    // Long spokes radiating from the centre (spectrum-driven)
    this.SPK = 80;
    this.spokes = [];
    for (let i = 0; i < this.SPK; i++) {
      this.spokes.push({
        d: unit(),
        r0: 0.1 + r() * 0.15,
        len: 0.5 + r() * 0.9,
        h: r(),
        bin: Math.floor(r() * 48),
      });
    }

    // Thick ring bands: dashed lines + dense particles + bokeh
    const specs = [
      { rx: 1.25, rz: -0.25, rad: 1.12, width: 0.12, spd: 0.12, offs: [-1, -0.5, 0, 0.5, 1], nd: 90, nDots: 260 },
      { rx: 0.55, rz: 1.1, rad: 0.96, width: 0.07, spd: -0.18, offs: [-1, 0, 1], nd: 80, nDots: 170 },
      { rx: 0.2, rz: -1.0, rad: 1.05, width: 0.05, spd: 0.22, offs: [-1, 0, 1], nd: 80, nDots: 140 },
      { rx: 0.9, rz: 0.3, rad: 0.62, width: 0.05, spd: -0.3, offs: [-1, 0, 1], nd: 60, nDots: 90 },
    ];

    this.bands = specs.map((sp) => {
      const dashLines = sp.offs.map(() => {
        const step = TAU / sp.nd;
        const arr = [];
        for (let d = 0; d < sp.nd; d++) {
          arr.push({
            a: d * step + (r() - 0.5) * step * 0.4,
            len: step * (0.25 + r() * 0.6),
            skip: r() < 0.2,
            h: r(),
          });
        }
        return arr;
      });

      const dots = [];
      for (let i = 0; i < sp.nDots; i++) {
        dots.push({
          a: r() * TAU,
          off: r() + r() - 1,
          y: r() - 0.5,
          sz: 0.8 + r() * 1.4,
          h: r(),
          sp: 1 + r() * 3,
          ph: r() * TAU,
          bokeh: r() < 0.1,
        });
      }

      return Object.assign({}, sp, {
        ph: r() * TAU,
        hs: 0.6 + r() * 0.8,
        dashLines,
        dots,
      });
    });

    // The eye: concentric rings facing the viewer
    const eyeDefs = [
      [0.075, "thick", 0.9],
      [0.09, "mid", 0.85],
      [0.105, "thick", 0.95],
      [0.12, "mid", 0.8],
      [0.14, "mid", 0.7],
      [0.2, "thin", 0.45],
      [0.3, "mid", 0.5],
      [0.34, "thin", 0.45],
      [0.42, "thin", 0.35],
      [0.52, "thin", 0.25],
    ];

    this.eyeRings = eyeDefs.map(([rad, arr, v], i) => ({
      r: rad,
      arr,
      v,
      spd: (i % 2 ? -1 : 1) * (0.2 + r() * 0.5),
      ph: r() * TAU,
      segs: Array.from({ length: 64 }, () => r() < (rad < 0.15 ? 0.08 : 0.32)),
    }));

    this.eyeSpiralN = 90;

    // Circuit-like traces that run along the shell
    this.traces = [];
    for (let i = 0; i < 48; i++) {
      let lat = (r() - 0.5) * 2.4;
      let lon = r() * TAU;
      const pts = [{ lat, lon }];
      const n = 5 + Math.floor(r() * 8);
      for (let k = 0; k < n; k++) {
        if (k % 2 === 0) lon += (r() - 0.5) * 0.5;
        else lat = Math.max(-1.4, Math.min(1.4, lat + (r() - 0.5) * 0.3));
        pts.push({ lat, lon });
      }
      this.traces.push({ pts, rad: 1 + r() * 0.07, sp: 0.5 + r() * 2, ph: r() * TAU });
    }

    // Free-floating bokeh discs
    this.bokehs = [];
    for (let i = 0; i < 90; i++) {
      const u = unit();
      const m = Math.cbrt(r()) * 1.25;
      this.bokehs.push({
        x: u[0] * m,
        y: u[1] * m,
        z: u[2] * m,
        size: 5 + r() * 20,
        ph: r() * TAU,
        sp: 0.5 + r() * 2,
      });
    }

    // Glitch fragments
    this.glitch = [];
    for (let i = 0; i < 70; i++) {
      const u = unit();
      const m = 0.9 + r() * 0.35;
      this.glitch.push({
        x: u[0] * m,
        y: u[1] * m,
        z: u[2] * m,
        w: 2 + r() * 9,
        h: 1 + r() * 2,
      });
    }

    this.sparks = [];

    this.tickVals = new Float32Array(this.SPK);
    this.tickTarget = new Float32Array(this.SPK);
  }

  resize() {
    const parent = this.canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    this.w = rect.width;
    this.h = rect.height;

    this.canvas.width = Math.max(1, Math.round(this.w * dpr));
    this.canvas.height = Math.max(1, Math.round(this.h * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.R = Math.min(this.w, this.h) * 0.235;
  }

  // ----------------------------------------------------------
  // Main loop
  // ----------------------------------------------------------

  frame(ts) {
    requestAnimationFrame(this.frame);

    const dt = Math.min(0.05, (ts - (this.lastTs || ts)) / 1000);
    this.lastTs = ts;
    this.time += dt;

    if (!this.w || !this.h) return;

    this.updateParams(dt);
    this.updateAudio(dt);
    this.draw(dt);
  }

  updateParams(dt) {
    const target =
      CORE_PROFILES[this.host.currentState] || CORE_PROFILES.IDLE;
    const k = 1 - Math.exp(-dt * 2.5);
    for (const key in target) {
      this.p[key] += (target[key] - this.p[key]) * k;
    }
  }

  // ----------------------------------------------------------
  // Audio → level + per-spoke spectrum
  // ----------------------------------------------------------

  readAudio() {
    const h = this.host;
    let an = null;

    if (h.isRecording && h.analyserNode) an = h.analyserNode;
    else if (h.isPlayingAudio && h.outAnalyser) an = h.outAnalyser;

    if (!an) return false;

    if (!this.freq || this.freq.length !== an.frequencyBinCount) {
      this.freq = new Uint8Array(an.frequencyBinCount);
    }

    an.getByteFrequencyData(this.freq);
    return true;
  }

  updateAudio(dt) {
    const st = this.host.currentState;
    const t = this.time;
    const SPK = this.SPK;

    let raw = 0;
    const live = this.readAudio();

    if (live) {
      const n = Math.min(48, this.freq.length);
      let s = 0;
      for (let i = 0; i < n; i++) s += this.freq[i];
      raw = Math.min(1, (s / n / 255) * 2.1);

      const use = Math.min(56, this.freq.length);
      for (let i = 0; i < SPK; i++) {
        const bin = Math.min(use - 1, this.spokes[i].bin);
        this.tickTarget[i] = Math.min(1, Math.pow(this.freq[bin] / 255, 1.2) * 1.25);
      }
    } else if (st === "PROCESSING" || st === "TRANSCRIBING") {
      // A scanner sweeps across the spokes while thinking.
      raw = 0.2 + 0.1 * Math.sin(t * 4.5);
      for (let i = 0; i < SPK; i++) {
        const a = (i / SPK) * Math.PI * 2;
        const d = Math.abs(Math.atan2(Math.sin(a - this.scan), Math.cos(a - this.scan)));
        this.tickTarget[i] = 0.05 + 0.7 * Math.exp(-d * d * 5);
      }
    } else if (st === "SPEAKING" || st === "LISTENING") {
      raw = 0.12 + 0.05 * Math.sin(t * 5);
      for (let i = 0; i < SPK; i++) {
        this.tickTarget[i] = 0.08 + 0.08 * Math.abs(Math.sin(i * 0.9 + t * 3));
      }
    } else {
      raw = 0.03 + 0.02 * Math.sin(t * 0.9);
      for (let i = 0; i < SPK; i++) {
        this.tickTarget[i] = 0.03 + 0.03 * Math.sin(i * 0.4 + t * 1.2);
      }
    }

    // Fast attack, slow release.
    const k = 1 - Math.exp(-dt * (raw > this.level ? 20 : 5));
    this.level += (raw - this.level) * k;

    const kt = 1 - Math.exp(-dt * 16);
    for (let i = 0; i < SPK; i++) {
      this.tickVals[i] += (this.tickTarget[i] - this.tickVals[i]) * kt;
    }

    if (this.levelFill) {
      const v = Math.round(this.level * 100) / 100;
      if (v !== this._lastFill) {
        this._lastFill = v;
        this.levelFill.style.transform = `scaleX(${v})`;
      }
    }
  }

  // ----------------------------------------------------------
  // Drawing helpers
  // ----------------------------------------------------------

  // Dark amber → hot yellow-white, like the reference.
  buildRamp() {
    const p = this.p;
    const alphaCap = Math.min(1, p.bright);
    for (let b = 0; b < 8; b++) {
      const f = b / 7;
      const hue = p.hue - (1 - f) * 9 + f * 6;
      const sat = Math.max(0, p.sat - f * 38);
      const light = 26 + f * 62;
      const alpha = 0.14 + f * 0.86 * alphaCap;
      this.ramp[b] = `hsla(${hue.toFixed(1)},${sat.toFixed(1)}%,${light.toFixed(1)}%,${alpha.toFixed(3)})`;
    }
  }

  // Soft bokeh disc with a brighter rim, re-rendered when the hue changes.
  makeSprite() {
    const p = this.p;
    const key = `${Math.round(p.hue / 2)}|${Math.round(p.sat / 8)}`;
    if (key === this.spriteKey && this.sprite) return;
    this.spriteKey = key;

    const c = this.sprite || document.createElement("canvas");
    c.width = 64;
    c.height = 64;

    const g = c.getContext("2d");
    g.clearRect(0, 0, 64, 64);

    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, `hsla(${p.hue},${p.sat}%,62%,0.30)`);
    gr.addColorStop(0.7, `hsla(${p.hue},${p.sat}%,58%,0.45)`);
    gr.addColorStop(0.9, `hsla(${p.hue + 4},${p.sat}%,72%,0.75)`);
    gr.addColorStop(1, `hsla(${p.hue + 4},${p.sat}%,70%,0)`);
    g.fillStyle = gr;
    g.fillRect(0, 0, 64, 64);

    this.sprite = c;
  }

  bk(v) {
    v *= this.p.bright;
    if (v <= 0) return 0;
    if (v >= 0.999) return 7;
    return (v * 8) | 0;
  }

  project(M, x, y, z, cx, cy, f, R) {
    const X = M[0] * x + M[1] * y + M[2] * z;
    const Y = M[3] * x + M[4] * y + M[5] * z;
    const Z = M[6] * x + M[7] * y + M[8] * z;
    const s = f / (f - Z);
    this.px = cx + X * s;
    this.py = cy + Y * s;
    this.pz = Math.max(0, Math.min(1, (Z / (R * 1.4)) * 0.5 + 0.5));
    this.ps = s;
  }

  // Projects both ends of a 3D segment and stores it in `arr`.
  line3(M, x0, y0, z0, x1, y1, z1, arr, v, cx, cy, f, R) {
    this.project(M, x0, y0, z0, cx, cy, f, R);
    const ax = this.px;
    const ay = this.py;
    const az = this.pz;
    this.project(M, x1, y1, z1, cx, cy, f, R);
    arr.push(ax, ay, this.px, this.py, this.bk(v + (az + this.pz) * 0.175));
  }

  flushDots() {
    const ctx = this.ctx;
    const a = this.dots;
    for (let b = 0; b < 8; b++) {
      ctx.fillStyle = this.ramp[b];
      ctx.beginPath();
      for (let i = 0; i < a.length; i += 4) {
        if (a[i + 3] === b) ctx.rect(a[i], a[i + 1], a[i + 2], a[i + 2]);
      }
      ctx.fill();
    }
  }

  flushRects() {
    const ctx = this.ctx;
    const a = this.rects;
    for (let b = 0; b < 8; b++) {
      ctx.fillStyle = this.ramp[b];
      ctx.beginPath();
      for (let i = 0; i < a.length; i += 5) {
        if (a[i + 4] === b) ctx.rect(a[i], a[i + 1], a[i + 2], a[i + 3]);
      }
      ctx.fill();
    }
  }

  flushLines(a, width) {
    const ctx = this.ctx;
    ctx.lineWidth = width;
    for (let b = 0; b < 8; b++) {
      ctx.strokeStyle = this.ramp[b];
      ctx.beginPath();
      for (let i = 0; i < a.length; i += 5) {
        if (a[i + 4] === b) {
          ctx.moveTo(a[i], a[i + 1]);
          ctx.lineTo(a[i + 2], a[i + 3]);
        }
      }
      ctx.stroke();
    }
  }

  // ----------------------------------------------------------
  // Draw
  // ----------------------------------------------------------

  draw(dt) {
    const ctx = this.ctx;
    const W = this.w;
    const H = this.h;
    const cx = W / 2;
    const cy = H / 2;
    const R = this.R;
    const p = this.p;
    const lv = this.level;
    const t = this.time;
    const mot = this.reduced ? 0.35 : 1;
    const en = p.energy * mot;
    const TAU = Math.PI * 2;
    const f = R * 3.4;
    const k = R / 200;
    const state = this.host.currentState;
    const bright = Math.min(1.2, p.bright);
    const capA = Math.min(1, p.bright);

    ctx.clearRect(0, 0, W, H);
    this.buildRamp();
    this.makeSprite();

    this.yaw += dt * p.spin * mot * (1 + lv * 1.2);
    this.swirlT += dt * p.swirl * mot * (1 + lv * 1.8);
    this.tickRot += dt * 0.06 * (1 + p.swirl * 0.4) * mot;
    this.hotA += dt * (0.5 + p.swirl * 0.8 + lv * 2) * mot;
    this.scan += dt * (state === "PROCESSING" ? 3.2 : 1.8);

    const pitch = 0.38 + Math.sin(t * 0.22) * 0.08;
    const Mg = mulM(rotX(pitch), rotY(this.yaw));
    const pulse = 1 + lv * en * 0.06;
    const Rs = R * pulse;

    ctx.globalCompositeOperation = "lighter";

    // ---------- Glows ----------
    const body = ctx.createRadialGradient(cx, cy, 0, cx, cy, Rs * 1.06);
    body.addColorStop(0, `hsla(${p.hue - 4},100%,40%,${0.03 * bright})`);
    body.addColorStop(0.7, `hsla(${p.hue - 4},100%,45%,${0.05 * bright})`);
    body.addColorStop(0.95, `hsla(${p.hue},100%,55%,${(0.12 + lv * 0.1) * bright})`);
    body.addColorStop(1, `hsla(${p.hue},100%,55%,0)`);
    ctx.fillStyle = body;
    ctx.fillRect(0, 0, W, H);

    const halo = ctx.createRadialGradient(cx, cy, Rs * 0.9, cx, cy, Rs * 1.9);
    halo.addColorStop(0, `hsla(${p.hue - 4},100%,50%,${0.08 * bright + lv * 0.05})`);
    halo.addColorStop(1, `hsla(${p.hue - 4},100%,40%,0)`);
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, W, H);

    // Eye glow with a dark hole in the middle
    const ea = Math.min(1, (0.55 + lv * 0.6) * capA);
    const eg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.6);
    eg.addColorStop(0, `hsla(${p.hue},100%,60%,0)`);
    eg.addColorStop(0.1, `hsla(${p.hue},100%,60%,0)`);
    eg.addColorStop(0.22, `hsla(${p.hue + 6},100%,72%,${ea})`);
    eg.addColorStop(0.4, `hsla(${p.hue},100%,52%,${ea * 0.35})`);
    eg.addColorStop(1, `hsla(${p.hue - 4},100%,40%,0)`);
    ctx.fillStyle = eg;
    ctx.fillRect(0, 0, W, H);

    // Silhouette + faint wide arc
    ctx.lineCap = "butt";
    ctx.lineWidth = 1;
    ctx.strokeStyle = `hsla(${p.hue},100%,60%,${0.16 * capA})`;
    ctx.beginPath();
    ctx.arc(cx, cy, Rs, 0, TAU);
    ctx.stroke();

    ctx.lineWidth = R * 0.1;
    ctx.strokeStyle = `hsla(${p.hue},90%,45%,${0.045 + lv * 0.04})`;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.78, -2.7 + this.tickRot * 0.3, -0.2 + this.tickRot * 0.3);
    ctx.stroke();

    // ---------- Collect geometry ----------
    this.dots.length = 0;
    this.thin.length = 0;
    this.mid.length = 0;
    this.thick.length = 0;
    this.rects.length = 0;
    this.boke.length = 0;

    // Wireframe: latitude circles
    for (let i = 0; i < this.lats.length; i++) {
      const lat = this.lats[i];
      const LS = lat.segs.length;
      for (let s = 0; s < LS; s++) {
        const seg = lat.segs[s];
        if (seg.skip) continue;
        const a0 = (s / LS) * TAU + t * lat.drift;
        const a1 = ((s + 1) / LS) * TAU + t * lat.drift;
        const j0 = 1 + en * lv * 0.05 * Math.sin(a0 * 4 + t * 7 + lat.y * 5);
        const j1 = 1 + en * lv * 0.05 * Math.sin(a1 * 4 + t * 7 + lat.y * 5);
        this.line3(
          Mg,
          Math.cos(a0) * lat.rad * Rs * j0, lat.y * Rs * j0, Math.sin(a0) * lat.rad * Rs * j0,
          Math.cos(a1) * lat.rad * Rs * j1, lat.y * Rs * j1, Math.sin(a1) * lat.rad * Rs * j1,
          this.thin, 0.14 + seg.h * 0.22 + lv * 0.35, cx, cy, f, R
        );
      }
    }

    // Wireframe: meridians
    for (let i = 0; i < this.lons.length; i++) {
      const lon = this.lons[i];
      const MS = lon.segs.length;
      for (let s = 0; s < MS; s++) {
        const seg = lon.segs[s];
        if (seg.skip) continue;
        const a0 = (s / MS) * TAU;
        const a1 = ((s + 1) / MS) * TAU;
        const j0 = 1 + en * lv * 0.05 * Math.sin(a0 * 3 + t * 6 + i);
        const j1 = 1 + en * lv * 0.05 * Math.sin(a1 * 3 + t * 6 + i);
        this.line3(
          Mg,
          Math.cos(a0) * lon.c * Rs * j0, Math.sin(a0) * Rs * j0, Math.cos(a0) * lon.s * Rs * j0,
          Math.cos(a1) * lon.c * Rs * j1, Math.sin(a1) * Rs * j1, Math.cos(a1) * lon.s * Rs * j1,
          this.thin, 0.14 + seg.h * 0.22 + lv * 0.35, cx, cy, f, R
        );
      }
    }

    // Straight chords across the sphere
    for (let i = 0; i < this.chords.length; i++) {
      const c = this.chords[i];
      this.line3(
        Mg,
        c.a[0] * Rs, c.a[1] * Rs, c.a[2] * Rs,
        c.b[0] * Rs, c.b[1] * Rs, c.b[2] * Rs,
        this.thin, 0.22 + c.h * 0.3 + lv * 0.3, cx, cy, f, R
      );
    }

    // Spokes from the centre, lengthened by the spectrum
    for (let i = 0; i < this.spokes.length; i++) {
      const sp = this.spokes[i];
      const v = this.tickVals[i];
      const r0 = sp.r0 * R;
      const r1 = (sp.r0 + sp.len * (0.55 + v * p.tick * 1.1)) * Rs;
      this.line3(
        Mg,
        sp.d[0] * r0, sp.d[1] * r0, sp.d[2] * r0,
        sp.d[0] * r1, sp.d[1] * r1, sp.d[2] * r1,
        this.thin, 0.3 + sp.h * 0.25 + v * 0.6, cx, cy, f, R
      );
      if (v > 0.35) {
        this.dots.push(this.px, this.py, (1.4 + v * 2) * k * this.ps, this.bk(0.6 + v * 0.4));
      }
    }

    // Ring bands
    for (let bi = 0; bi < this.bands.length; bi++) {
      const b = this.bands[bi];
      const Mb = mulM(
        Mg,
        mulM(rotZ(b.rz + Math.sin(t * 0.15 + b.ph) * 0.1), rotX(b.rx + Math.sin(t * 0.2 + b.ph) * 0.12))
      );
      const w = b.width * (1 + lv * en * 0.5);
      const baseAng = b.ph + this.swirlT * b.spd * 1.6;
      const hotAng = this.hotA * b.hs + b.ph;

      for (let o = 0; o < b.offs.length; o++) {
        const rad = (b.rad + b.offs[o] * w) * Rs;
        const dashes = b.dashLines[o];
        for (let d = 0; d < dashes.length; d++) {
          const dash = dashes[d];
          if (dash.skip) continue;
          const a0 = baseAng + dash.a;
          const a1 = a0 + dash.len;
          const c = 0.5 + 0.5 * Math.cos(a0 - hotAng);
          const I = 0.28 + 0.72 * c * c;
          const jit = 1 + en * lv * 0.05 * Math.sin(a0 * 5 + t * 8 + o);
          this.line3(
            Mb,
            Math.cos(a0) * rad * jit, 0, Math.sin(a0) * rad * jit,
            Math.cos(a1) * rad * jit, 0, Math.sin(a1) * rad * jit,
            this.mid, 0.12 + I * 0.55 + dash.h * 0.15 + lv * 0.3, cx, cy, f, R
          );
        }
      }

      for (let i = 0; i < b.dots.length; i++) {
        const d = b.dots[i];
        const a = baseAng + d.a;
        const c = 0.5 + 0.5 * Math.cos(a - hotAng);
        const I = 0.28 + 0.72 * c * c;
        const rad = (b.rad + d.off * w) * Rs * (1 + en * lv * 0.06 * Math.sin(t * 9 + d.ph));
        this.project(Mb, Math.cos(a) * rad, d.y * w * Rs * 0.5, Math.sin(a) * rad, cx, cy, f, R);

        if (d.bokeh) {
          const blur = Math.min(1, Math.abs(this.pz - 0.5) * 2.2);
          this.boke.push(
            this.px, this.py,
            d.sz * 6 * k * this.ps * (1 + blur * 0.8),
            (0.25 + I * 0.5) * capA * (0.6 + 0.4 * Math.sin(t * d.sp + d.ph)) * (1 - blur * 0.4)
          );
        } else {
          this.dots.push(
            this.px, this.py,
            d.sz * k * this.ps * (1 + lv * 0.6),
            this.bk(0.15 + I * 0.7 + d.h * 0.2 + lv * 0.3)
          );
        }
      }
    }

    // The eye
    const Me = mulM(
      rotY(Math.sin(t * 0.23) * 0.22 + this.yaw * 0.05),
      rotX(0.18 + Math.sin(t * 0.31) * 0.1)
    );
    const eyeScale = 1 + lv * en * 0.35;

    for (let i = 0; i < this.eyeRings.length; i++) {
      const ring = this.eyeRings[i];
      const rad = ring.r * R * eyeScale;
      const ang0 = ring.ph + this.swirlT * ring.spd * 2;
      const arr = this[ring.arr];
      const n = ring.segs.length;
      for (let s = 0; s < n; s++) {
        if (ring.segs[s]) continue;
        const a0 = ang0 + (s / n) * TAU;
        const a1 = ang0 + ((s + 1) / n) * TAU;
        this.line3(
          Me,
          Math.cos(a0) * rad, Math.sin(a0) * rad, 0,
          Math.cos(a1) * rad, Math.sin(a1) * rad, 0,
          arr, ring.v + lv * 0.3, cx, cy, f, R
        );
      }
    }

    // Radial hatching around the eye (spectrum-driven)
    const HT = 96;
    for (let i = 0; i < HT; i++) {
      const v = this.tickVals[i % this.SPK];
      const a = this.swirlT * 0.4 + (i / HT) * TAU;
      const r0 = 0.3 * R * eyeScale;
      const r1 = r0 + (0.03 + v * 0.06) * R;
      this.line3(
        Me,
        Math.cos(a) * r0, Math.sin(a) * r0, 0,
        Math.cos(a) * r1, Math.sin(a) * r1, 0,
        this.thin, 0.3 + v * 0.6, cx, cy, f, R
      );
    }

    // Tunnel spiral inside the eye
    let prevX = 0;
    let prevY = 0;
    for (let i = 0; i < this.eyeSpiralN; i++) {
      const t01 = i / this.eyeSpiralN;
      const ang = t01 * 18 + this.swirlT * 3;
      const rr = (0.03 + 0.11 * t01) * R * eyeScale;
      this.project(Me, Math.cos(ang) * rr, Math.sin(ang) * rr, -(1 - t01) * 0.18 * R, cx, cy, f, R);
      this.dots.push(this.px, this.py, (1.1 + t01 * 1.2) * k * this.ps, this.bk(0.7 + t01 * 0.3));
      if (i > 0) {
        this.thin.push(prevX, prevY, this.px, this.py, this.bk(0.65 + t01 * 0.35));
      }
      prevX = this.px;
      prevY = this.py;
    }

    // Circuit traces on the shell
    for (let i = 0; i < this.traces.length; i++) {
      const tr = this.traces[i];
      if (Math.sin(t * tr.sp * 1.7 + tr.ph) < -0.6) continue;

      const grow = Math.min(1, 0.3 + lv * en * 1.2 + 0.25 * (0.5 + 0.5 * Math.sin(t * tr.sp + tr.ph)));
      const n = Math.floor(grow * (tr.pts.length - 1));
      const v = 0.35 + 0.5 * (0.5 + 0.5 * Math.sin(t * 3 + tr.ph));
      const rr = tr.rad * Rs;

      for (let s = 0; s < n; s++) {
        const a = tr.pts[s];
        const b = tr.pts[s + 1];
        this.line3(
          Mg,
          Math.cos(a.lat) * Math.cos(a.lon) * rr, Math.sin(a.lat) * rr, Math.cos(a.lat) * Math.sin(a.lon) * rr,
          Math.cos(b.lat) * Math.cos(b.lon) * rr, Math.sin(b.lat) * rr, Math.cos(b.lat) * Math.sin(b.lon) * rr,
          this.mid, v, cx, cy, f, R
        );
      }
    }

    // Free bokeh
    for (let i = 0; i < this.bokehs.length; i++) {
      const o = this.bokehs[i];
      this.project(Mg, o.x * Rs, o.y * Rs, o.z * Rs, cx, cy, f, R);
      const blur = Math.min(1, Math.abs(this.pz - 0.5) * 2.2);
      this.boke.push(
        this.px, this.py,
        o.size * k * this.ps * (1 + blur * 0.9),
        (0.12 + 0.3 * (0.5 + 0.5 * Math.sin(t * o.sp + o.ph)) + lv * 0.25) * (1 - blur * 0.5) * capA
      );
    }

    // Glitch fragments (flicker faster when there is energy)
    const gStep = Math.floor(t * (6 + lv * 10));
    const gThresh = 0.7 - lv * 0.4;
    for (let i = 0; i < this.glitch.length; i++) {
      const g = this.glitch[i];
      const hv = Math.sin(i * 12.9898 + gStep * 78.233) * 43758.5453;
      const vis = hv - Math.floor(hv);
      if (vis < gThresh) continue;
      this.project(Mg, g.x * Rs, g.y * Rs, g.z * Rs, cx, cy, f, R);
      this.rects.push(this.px, this.py, g.w * k * this.ps, g.h * k * this.ps, this.bk(0.55 + this.pz * 0.3 + lv * 0.2));
    }

    // Sparks flying off while there is energy
    if (!this.reduced && lv > 0.12) {
      this.spawnAcc += dt * lv * p.energy * 90;
      while (this.spawnAcc >= 1 && this.sparks.length < 160) {
        this.spawnAcc -= 1;
        this.sparks.push({
          a: Math.random() * TAU,
          r0: R * (1.05 + Math.random() * 0.35),
          speed: R * (0.5 + Math.random() * 1.4),
          life: 0.5 + Math.random() * 0.7,
          len: R * (0.03 + Math.random() * 0.05),
          age: 0,
        });
      }
      this.spawnAcc = Math.min(this.spawnAcc, 3);
    }

    for (let i = this.sparks.length - 1; i >= 0; i--) {
      const sp = this.sparks[i];
      sp.age += dt;
      if (sp.age >= sp.life) {
        this.sparks[i] = this.sparks[this.sparks.length - 1];
        this.sparks.pop();
        continue;
      }
      const life01 = 1 - sp.age / sp.life;
      const rr = sp.r0 + sp.speed * sp.age;
      const ca = Math.cos(sp.a);
      const sa = Math.sin(sp.a);
      this.thin.push(
        cx + ca * rr, cy + sa * rr,
        cx + ca * (rr + sp.len * (0.5 + life01)), cy + sa * (rr + sp.len * (0.5 + life01)),
        Math.max(0, Math.min(7, Math.floor(life01 * 7.5)))
      );
    }

    // ---------- Flush ----------
    const sprite = this.sprite;
    for (let i = 0; i < this.boke.length; i += 4) {
      const s = this.boke[i + 2];
      ctx.globalAlpha = Math.max(0, Math.min(1, this.boke[i + 3]));
      ctx.drawImage(sprite, this.boke[i] - s, this.boke[i + 1] - s, s * 2, s * 2);
    }
    ctx.globalAlpha = 1;

    this.flushDots();
    this.flushRects();
    this.flushLines(this.thin, 1.0);
    this.flushLines(this.mid, 1.5);
    this.flushLines(this.thick, 2.3);

    ctx.globalCompositeOperation = "source-over";
  }
}

// ============================================================
// Client
// ============================================================

class PMAIWebVoiceClient {
  constructor() {
    // ----------------------------------------------------------
    // UI
    // ----------------------------------------------------------

    this.root = buildMaiUI();
    const $ = (id) => document.getElementById(id);

    this.elStateText = $("mai-state-text");
    this.elPulse = $("mai-pulse");
    this.elConnText = $("mai-conn-text");
    this.elConnDot = $("mai-conn-dot");
    this.elLiveTranscript = $("mai-live-transcript");
    this.elFeed = $("mai-feed");
    this.elError = $("mai-error");
    this.elBtnSpeak = $("mai-btn-speak");
    this.elSpeakLabel = $("mai-speak-label");
    this.elBtnStop = $("mai-btn-stop");
    this.elBtnCancel = $("mai-btn-cancel");
    this.elMetrics = $("mai-metrics");

    this.reducedMotion =
      !!(window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches);

    // ----------------------------------------------------------
    // WebSocket
    // ----------------------------------------------------------

    this.ws = null;
    this.currentState = "DISCONNECTED";

    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 3;
    this.reconnectTimer = null;

    // ----------------------------------------------------------
    // Recording
    // ----------------------------------------------------------

    this.isRecording = false;
    this.micStream = null;
    this.processorNode = null;
    this.sourceNode = null;

    // ----------------------------------------------------------
    // Audio context
    // ----------------------------------------------------------

    this.audioCtx = null;
    this.analyserNode = null; // microphone analyser
    this.outAnalyser = null; // playback analyser

    // ----------------------------------------------------------
    // Playback state
    // ----------------------------------------------------------

    this.isPlayingAudio = false;
    this.currentAudioSource = null;

    /*
     * RAW BYTE CHUNKS received from RESPONSE_CHUNK.
     * We never decode them individually: all chunks of one response
     * are assembled and decoded/played as one complete payload.
     */
    this.audioQueue = [];
    this.responseAudioChunks = this.audioQueue;
    this.responseAudioBytes = 0;
    this.responseAudioContentType = null;
    this.responseHasAudio = false;

    // Invalidates stale asynchronous decode operations.
    this.playbackGeneration = 0;

    // ----------------------------------------------------------
    // Performance metrics
    // ----------------------------------------------------------

    this.turnStart = 0;
    this.timeAudioEnd = 0;
    this.responseStartTime = 0;
    this.responseEndTime = 0;

    // ----------------------------------------------------------
    // Core visual
    // ----------------------------------------------------------

    this.core = new EmberCore(
      $("mai-core-canvas"),
      this,
      $("mai-level-fill")
    );

    // ----------------------------------------------------------
    // Initialize
    // ----------------------------------------------------------

    this.attachEventListeners();
    this.setState("DISCONNECTED");
    this.initWebSocket();
  }

  // ============================================================
  // State handling
  // ============================================================

  setState(stateName, customMsg = null) {
    this.currentState = stateName;

    const info = STATES[stateName] || STATES.IDLE;
    const label = customMsg || info.label;

    this.root.dataset.state = stateName;
    this.root.style.setProperty("--state", info.color);

    if (this.elStateText) this.elStateText.textContent = label;

    if (this.elBtnSpeak) {
      this.elBtnSpeak.classList.toggle("recording", this.isRecording);
      this.elBtnSpeak.setAttribute(
        "aria-label",
        this.isRecording ? "Finish talking" : "Start talking"
      );
    }

    if (this.elSpeakLabel) {
      this.elSpeakLabel.textContent = this.isRecording
        ? "Tap to finish"
        : "Tap or press Space to talk";
    }

    if (this.elError) {
      if (stateName === "ERROR") {
        this.elError.textContent = label;
        this.elError.style.display = "block";
      } else {
        this.elError.style.display = "none";
      }
    }
  }

  setConnectionStatus(connected, text) {
    if (this.elConnDot) {
      const c = connected ? "#ffc94d" : "#ff3b30";
      this.elConnDot.style.background = c;
      this.elConnDot.style.boxShadow = `0 0 10px ${c}`;
    }
    if (this.elConnText) this.elConnText.textContent = text;
  }

  // ============================================================
  // WebSocket
  // ============================================================

  initWebSocket() {
    this.setState("CONNECTING");
    this.setConnectionStatus(false, "Connecting");

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";

    const wsUrl =
      `${protocol}//${host}/voice/realtime` +
      `?conversation_id=web-client&user_id=web-user`;

    try {
      this.ws = new WebSocket(wsUrl);
    } catch (err) {
      this.handleDisconnect(err);
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;

      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }

      this.setConnectionStatus(true, "Connected");
      this.setState("IDLE");
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleServerEvent(msg);
      } catch (err) {
        this.setState("ERROR", "Malformed server response");
      }
    };

    this.ws.onerror = () => {
      this.setState("ERROR", "WebSocket connection error");
    };

    this.ws.onclose = () => {
      this.handleDisconnect();
    };
  }

  handleDisconnect(err = null) {
    this.setConnectionStatus(false, "Disconnected");

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;

      this.setState(
        "CONNECTING",
        `Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );

      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.initWebSocket();
      }, 2000);
    } else {
      this.setState(
        "DISCONNECTED",
        "Connection lost. Please refresh or reconnect."
      );
    }
  }

  // ============================================================
  // Server event handling
  // ============================================================

  handleServerEvent(event) {
    if (!event || typeof event !== "object") {
      this.setState("ERROR", "Invalid server event");
      return;
    }

    switch (event.type) {
      case "SPEECH_START":
        this.setState("LISTENING");
        break;

      case "SPEECH_END":
        this.setState("TRANSCRIBING");
        break;

      case "TRANSCRIPT_PARTIAL":
        if (this.elLiveTranscript) {
          this.elLiveTranscript.textContent = event.transcript || "";
        }
        break;

      case "TRANSCRIPT_FINAL":
        if (this.elLiveTranscript) this.elLiveTranscript.textContent = "";
        if (event.transcript) this.appendMessage("User", event.transcript);
        this.setState("PROCESSING");
        break;

      case "RESPONSE_START":
        this.beginResponseAudio(event);
        this.setState("PROCESSING");
        break;

      case "RESPONSE_CHUNK":
        /*
         * Only buffer raw bytes here. The UI switches to "Speaking"
         * when playback actually starts, so the core does not pulse
         * while nothing is audible yet.
         */
        if (event.audio_base64) {
          this.bufferResponseAudioChunk(event.audio_base64, event);
        }
        break;

      case "RESPONSE_END": {
        this.responseEndTime = performance.now();

        if (event.text_chunk) this.appendMessage("MAI", event.text_chunk);

        if (event.text_response && event.text_response !== event.text_chunk) {
          this.appendMessage("MAI", event.text_response);
        }

        if (event.metadata) {
          const maiMs = Number(event.metadata.mai_latency_ms) || 0;
          const ttsMs = Number(event.metadata.tts_latency_ms) || 0;
          if (this.elMetrics) {
            this.elMetrics.textContent = `MAI ${maiMs} ms · TTS ${ttsMs} ms`;
          }
        }

        /*
         * Playback begins ONLY after the full response audio is
         * available (avoids decodeAudioData() failures on fragments).
         */
        if (this.responseHasAudio) {
          this.playBufferedResponseAudio();
        } else {
          this.setState("IDLE");
        }
        break;
      }

      case "INTERRUPT":
        this.stopPlayback();
        this.setState("INTERRUPTED");
        window.setTimeout(() => {
          if (this.currentState === "INTERRUPTED") this.setState("IDLE");
        }, 800);
        break;

      case "ERROR":
        this.stopPlayback();
        this.setState(
          "ERROR",
          `${event.error_code || "ERROR"}: ${event.error_message || "Server error"}`
        );
        break;

      case "CLOSE":
        this.stopPlayback();
        this.setState("CLOSED");
        break;

      default:
        // Ignore unknown future events safely.
        break;
    }
  }

  // ============================================================
  // Conversation UI
  // ============================================================

  appendMessage(speaker, text) {
    if (!this.elFeed || !text) return;

    const isUser = speaker === "User";

    const card = document.createElement("div");
    card.className = `mai-msg ${isUser ? "mai-msg-user" : "mai-msg-mai"}`;

    const label = document.createElement("div");
    label.className = "mai-msg-label";
    label.textContent = isUser ? "You" : speaker;

    const content = document.createElement("div");
    content.className = "mai-msg-text";

    card.appendChild(label);
    card.appendChild(content);
    this.elFeed.appendChild(card);

    if (isUser || this.reducedMotion) {
      content.textContent = text;
    } else {
      this.revealText(content, text);
    }

    // Keep history bounded to 20 messages.
    while (this.elFeed.children.length > 20) {
      this.elFeed.removeChild(this.elFeed.firstChild);
    }

    this.elFeed.scrollTop = this.elFeed.scrollHeight;
  }

  revealText(el, text) {
    const tokens = text.split(/(\s+)/);
    const perTick = Math.max(2, Math.ceil(tokens.length / 80));
    let i = 0;

    const timer = setInterval(() => {
      if (!el.isConnected) {
        clearInterval(timer);
        return;
      }

      i = Math.min(tokens.length, i + perTick);
      el.textContent = tokens.slice(0, i).join("");

      if (this.elFeed) this.elFeed.scrollTop = this.elFeed.scrollHeight;
      if (i >= tokens.length) clearInterval(timer);
    }, 40);
  }

  // ============================================================
  // Audio context
  // ============================================================

  async ensureAudioContext() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Playback analyser: everything MAI says flows through it,
    // so the core can pulse with the real output level.
    if (!this.outAnalyser) {
      this.outAnalyser = this.audioCtx.createAnalyser();
      this.outAnalyser.fftSize = 512;
      this.outAnalyser.smoothingTimeConstant = 0.75;
      this.outAnalyser.connect(this.audioCtx.destination);
    }

    // Chrome may suspend an AudioContext until a user gesture resumes it.
    if (this.audioCtx.state === "suspended") {
      try {
        await this.audioCtx.resume();
      } catch (err) {
        console.warn("Unable to resume AudioContext:", err);
      }
    }

    return this.audioCtx;
  }

  // ============================================================
  // Recording
  // ============================================================

  async startRecording() {
    // Barge-in: stop local playback first. Does NOT cancel the task.
    if (this.currentState === "SPEAKING" || this.isPlayingAudio) {
      this.stopPlayback();
    }

    // Clear stale response audio before starting a new turn.
    this.resetResponseAudio();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.setState("ERROR", "Microphone not supported on this browser.");
      return;
    }

    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      this.setState(
        "ERROR",
        "Microphone permission denied or device unavailable."
      );
      return;
    }

    try {
      await this.ensureAudioContext();
    } catch (err) {
      this.setState("ERROR", "Web Audio is unavailable in this browser.");
      this.cleanupMicrophone();
      return;
    }

    try {
      this.sourceNode = this.audioCtx.createMediaStreamSource(this.micStream);

      this.analyserNode = this.audioCtx.createAnalyser();
      this.analyserNode.fftSize = 256;
      this.analyserNode.smoothingTimeConstant = 0.7;
      this.sourceNode.connect(this.analyserNode);

      /*
       * ScriptProcessorNode is kept for compatibility with the existing
       * backend contract. Chrome shows a deprecation warning; migrating
       * to AudioWorklet is a good future step.
       */
      this.processorNode = this.audioCtx.createScriptProcessor(4096, 1, 1);

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) return;

        try {
          const inputData = e.inputBuffer.getChannelData(0);
          const pcm16 = this.downsampleAndConvert(
            inputData,
            this.audioCtx.sampleRate,
            16000
          );
          this.sendAudioChunk(pcm16);
        } catch (err) {
          console.warn("Microphone audio processing failed:", err);
        }
      };

      this.sourceNode.connect(this.processorNode);

      // Keep the processing node alive.
      this.processorNode.connect(this.audioCtx.destination);

      this.isRecording = true;
      this.turnStart = performance.now();

      this.setState("LISTENING");

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "AUDIO_START" }));
      }
    } catch (err) {
      this.cleanupMicrophone();
      this.setState("ERROR", "Unable to initialize microphone audio.");
    }
  }

  stopRecording() {
    if (!this.isRecording) return;

    this.isRecording = false;
    this.timeAudioEnd = performance.now();

    this.cleanupMicrophone();
    this.setState("TRANSCRIBING");

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "AUDIO_END" }));
    }
  }

  cleanupMicrophone() {
    if (this.processorNode) {
      try {
        this.processorNode.disconnect();
      } catch (err) {
        // Ignore cleanup failure.
      }
      this.processorNode = null;
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch (err) {
        // Ignore cleanup failure.
      }
      this.sourceNode = null;
    }

    if (this.micStream) {
      this.micStream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (err) {
          // Ignore cleanup failure.
        }
      });
      this.micStream = null;
    }
  }

  toggleRecording() {
    if (this.isRecording) this.stopRecording();
    else this.startRecording();
  }

  // ============================================================
  // Sending audio
  // ============================================================

  sendAudioChunk(pcm16Bytes) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (!pcm16Bytes || pcm16Bytes.byteLength === 0) return;

    const base64Audio = this.arrayBufferToBase64(pcm16Bytes.buffer);

    /*
     * Existing backend contract preserved: raw PCM bytes labelled
     * audio/wav. Do not change unless the backend changes too.
     */
    this.ws.send(
      JSON.stringify({
        type: "AUDIO_CHUNK",
        audio_base64: base64Audio,
        content_type: "audio/wav",
      })
    );
  }

  // ============================================================
  // PCM conversion
  // ============================================================

  downsampleAndConvert(buffer, sampleRate, outSampleRate) {
    if (outSampleRate === sampleRate) return this.floatTo16BitPCM(buffer);

    const ratio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Int16Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);

      let accum = 0;
      let count = 0;

      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }

      const s = Math.max(-1, Math.min(1, count > 0 ? accum / count : 0));
      result[offsetResult] = s < 0 ? s * 0x8000 : s * 0x7fff;

      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }

    return result;
  }

  floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return output;
  }

  // ============================================================
  // Base64 helpers
  // ============================================================

  arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // ============================================================
  // Response audio buffering
  // ============================================================

  beginResponseAudio(event = null) {
    // Invalidate any previous asynchronous decoder.
    this.playbackGeneration++;

    this.resetResponseAudio();

    if (event) {
      this.responseAudioContentType =
        event.audio_content_type ||
        event.content_type ||
        (event.metadata && event.metadata.audio_content_type) ||
        null;
    }

    this.responseStartTime = performance.now();
  }

  resetResponseAudio() {
    // Clears pending response bytes only; playback is stopPlayback()'s job.
    this.audioQueue.length = 0;
    this.responseAudioBytes = 0;
    this.responseAudioContentType = null;
    this.responseHasAudio = false;
  }

  bufferResponseAudioChunk(base64Data, event = null) {
    if (typeof base64Data !== "string" || base64Data.length === 0) return;

    try {
      const arrayBuf = this.base64ToArrayBuffer(base64Data);
      if (!arrayBuf || arrayBuf.byteLength === 0) return;

      if (event) {
        this.responseAudioContentType =
          event.audio_content_type ||
          event.content_type ||
          this.responseAudioContentType ||
          (event.metadata && event.metadata.audio_content_type) ||
          null;
      }

      // Store the bytes. Never decode here.
      this.audioQueue.push(arrayBuf);
      this.responseAudioBytes += arrayBuf.byteLength;
      this.responseHasAudio = true;

      console.debug("[P-MAI audio] received chunk:", {
        bytes: arrayBuf.byteLength,
        totalBytes: this.responseAudioBytes,
        chunks: this.audioQueue.length,
        contentType: this.responseAudioContentType,
      });
    } catch (err) {
      console.warn("[P-MAI audio] base64 decode failed:", err);
    }
  }

  concatArrayBuffers(buffers) {
    const totalLength = buffers.reduce((sum, b) => sum + b.byteLength, 0);
    const result = new Uint8Array(totalLength);

    let offset = 0;
    for (const buffer of buffers) {
      result.set(new Uint8Array(buffer), offset);
      offset += buffer.byteLength;
    }

    return result.buffer;
  }

  isWavBuffer(arrayBuffer) {
    if (!arrayBuffer || arrayBuffer.byteLength < 12) return false;

    const b = new Uint8Array(arrayBuffer);

    return (
      b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 && // RIFF
      b[8] === 0x57 && b[9] === 0x41 && b[10] === 0x56 && b[11] === 0x45 // WAVE
    );
  }

  createWavFromPcm(pcmBuffer, sampleRate = 16000, channels = 1, bitsPerSample = 16) {
    const pcmBytes = new Uint8Array(pcmBuffer);
    const blockAlign = channels * (bitsPerSample / 8);
    const byteRate = sampleRate * blockAlign;
    const dataSize = pcmBytes.byteLength;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeAscii = (offset, text) => {
      for (let i = 0; i < text.length; i++) {
        view.setUint8(offset + i, text.charCodeAt(i));
      }
    };

    writeAscii(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(8, "WAVE");
    writeAscii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeAscii(36, "data");
    view.setUint32(40, dataSize, true);

    new Uint8Array(buffer, 44).set(pcmBytes);

    return buffer;
  }

  // ============================================================
  // Complete buffered response playback
  // ============================================================

  async playBufferedResponseAudio() {
    const generation = this.playbackGeneration;

    if (!this.audioQueue.length || this.responseAudioBytes <= 0) {
      this.setState("IDLE");
      return;
    }

    try {
      const combinedBuffer = this.concatArrayBuffers(this.audioQueue);

      if (!combinedBuffer || combinedBuffer.byteLength === 0) {
        this.setState("IDLE");
        return;
      }

      // Stale if the user already interrupted playback.
      if (generation !== this.playbackGeneration) return;

      // Valid WAV → decode as is. Otherwise assume raw PCM and wrap it.
      let playableBuffer = combinedBuffer;

      if (!this.isWavBuffer(combinedBuffer)) {
        playableBuffer = this.createWavFromPcm(combinedBuffer, 16000, 1, 16);

        console.debug("[P-MAI audio] raw PCM detected; wrapped as WAV", {
          pcmBytes: combinedBuffer.byteLength,
          wavBytes: playableBuffer.byteLength,
        });
      } else {
        console.debug("[P-MAI audio] complete WAV detected", {
          wavBytes: combinedBuffer.byteLength,
        });
      }

      const ctx = await this.ensureAudioContext();
      if (generation !== this.playbackGeneration) return;

      // Decode ONE complete, valid audio payload.
      const audioBuffer = await ctx.decodeAudioData(playableBuffer.slice(0));
      if (generation !== this.playbackGeneration) return;

      this.stopCurrentAudioSourceOnly();

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;

      // Route through the playback analyser (→ destination),
      // which also feeds the core visual.
      source.connect(this.outAnalyser);

      this.currentAudioSource = source;
      this.isPlayingAudio = true;

      this.setState("SPEAKING");

      source.onended = () => {
        if (this.currentAudioSource === source) {
          this.isPlayingAudio = false;
          this.currentAudioSource = null;

          console.debug("[P-MAI audio] [Stage 4/4] playback completed normally", {
            durationSeconds: audioBuffer.duration,
          });

          if (this.currentState === "SPEAKING") this.setState("IDLE");
        }
      };

      source.start(0);

      console.debug("[P-MAI audio] [Stage 3/4] playback started", {
        durationSeconds: audioBuffer.duration,
        sampleRate: audioBuffer.sampleRate,
        channels: audioBuffer.numberOfChannels,
      });
    } catch (err) {
      console.error("[P-MAI audio] playback failed:", err);

      this.isPlayingAudio = false;
      this.currentAudioSource = null;

      this.setState("ERROR", "Unable to play MAI audio in this browser.");
    } finally {
      // Byte buffers are no longer needed once playback is prepared.
      if (generation === this.playbackGeneration) {
        this.resetResponseAudio();
      }
    }
  }

  // ============================================================
  // Playback controls
  // ============================================================

  stopCurrentAudioSourceOnly() {
    if (!this.currentAudioSource) return;

    try {
      this.currentAudioSource.onended = null;
      this.currentAudioSource.stop();
    } catch (err) {
      // Already stopped / naturally ended.
    }

    this.currentAudioSource = null;
    this.isPlayingAudio = false;
  }

  stopPlayback() {
    // Invalidate pending async decoding.
    this.playbackGeneration++;

    this.stopCurrentAudioSourceOnly();
    this.resetResponseAudio();
  }

  // ============================================================
  // Stop speaking (audio only; does not cancel the task)
  // ============================================================

  onStopSpeaking() {
    this.stopPlayback();

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "INTERRUPT",
          reason: "stop_speaking_button",
        })
      );
    }

    this.setState("IDLE");
  }

  // ============================================================
  // Cancel task
  // ============================================================

  onCancelTask() {
    this.stopPlayback();

    // Preserve the existing backend cancellation mechanism.
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "AUDIO_START" }));

      window.setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "AUDIO_END" }));
        }
      }, 50);
    }

    this.appendMessage("User", "Cancel active task");
    this.setState("IDLE");
  }

  // ============================================================
  // Event listeners
  // ============================================================

  attachEventListeners() {
    if (this.elBtnSpeak) {
      this.elBtnSpeak.addEventListener("click", () => this.toggleRecording());
    }

    if (this.elBtnStop) {
      this.elBtnStop.addEventListener("click", () => this.onStopSpeaking());
    }

    if (this.elBtnCancel) {
      this.elBtnCancel.addEventListener("click", () => this.onCancelTask());
    }

    if (this.elError) {
      this.elError.addEventListener("click", () => {
        this.elError.style.display = "none";
      });
    }

    window.addEventListener("keydown", (e) => {
      if (e.repeat) return;

      // Space: start/stop recording.
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        this.toggleRecording();
      }
      // Escape: stop speaking only.
      else if (e.code === "Escape") {
        e.preventDefault();
        this.onStopSpeaking();
      }
    });
  }
}

// ============================================================
// Instantiate on page load
// ============================================================

window.addEventListener("DOMContentLoaded", () => {
  window.pmaiVoiceClient = new PMAIWebVoiceClient();
});