import "@fontsource-variable/inter";
import { invoke } from "@tauri-apps/api/core";
import { emitTo, listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { openPath, openUrl } from "@tauri-apps/plugin-opener";
import { icon } from "./icons";
import { TRAY_ACTION, TRAY_HELLO, TRAY_STATE, type TrayAction, type TrayState } from "./tray-protocol";

type Carrier = "saw" | "chord" | "bell" | "organ" | "choir" | "noise" | "buzz";

interface Params {
  enabled: boolean;
  voices: string[];
  pitch: number;
  formant: number;
  breath: number;
  rasp: number;
  vibrato: number;
  nasal: number;
  vocoder: Carrier | null;
  vocoderMix: number;
  cave: boolean;
  hall: boolean;
  chorus: boolean;
  ghost: boolean;
  echo: boolean;
  room: boolean;
  slap: boolean;
  flange: boolean;
  bottle: boolean;
  boost: number;
  sensitivityDb: number;
  bassDb: number;
  trebleDb: number;
  /** 0..1 mix of the neural noise remover. */
  denoiseAmount: number;
  /** 0..1 steady noise reduction. */
  noiseFloor: number;
  /** 0..1 short click reduction. */
  impact: number;
  /** 0..1 compressor amount. */
  compress: number;
  /** Close the mic when the sound is not the enrolled voice. */
  voiceGate: boolean;
  /** 0 loose, 1 strict. */
  voiceStrict: number;
  voiceBands: number[];
  voicePitch: number;
  voicePitchLo: number;
  voicePitchHi: number;
  voiceLevel: number;
  voiceF1: number;
  voiceF2: number;
  voiceF3: number;
  /** CAM++ enrollment embedding (512). Empty = DSP-only gate. */
  voiceEmbed: number[];
  /** Extra CAM++ templates (N×512). Live score uses the best match. */
  voiceGallery: number[];
  /** Calibrated cosine thresholds (strict 0 → lo, strict 1 → hi). */
  voiceNeuralLo: number;
  voiceNeuralHi: number;
  /** Sustained cosine below this ⇒ CONFIRMED_OTHER. */
  voiceNeuralReject: number;
  /** Kept for diagnostics; both modes share the 30 ms fail-open line. */
  voiceProtect: boolean;
  /** Ten gains in dB, 20 Hz through 16 kHz. */
  eq: number[];
  denoise: boolean;
  appendOriginal: boolean;
  /** Longer pitch window for classic voices. */
  smooth: boolean;
}

type Page = "voices" | "ai" | "effects" | "mic" | "profiles" | "shortcuts" | "music";

interface Track {
  id: string;
  name: string;
  path: string;
}

interface CustomShortcut {
  id: string;
  name: string;
  /** Target id, such as "power", "voice:male" or "music:…". */
  action: string;
}

interface Profile {
  name: string;
  params: Params;
}

type AiPreset = "fast" | "balanced" | "quality" | "custom";

interface Timing {
  blockMs: number;
  contextMs: number;
  crossfadeMs: number;
}

interface AiSettings {
  enabled: boolean;
  voice: string | null;
  preset: AiPreset;
  device: "gpu" | "cpu";
  pitchMethod: "rmvpe" | "yin";
  custom: Timing;
  /** Transposition per voice, in semitones. */
  transpose: Record<string, number>;
}

interface Settings {
  params: Params;
  mixer: boolean;
  category: string;
  input: string | null;
  output: string | null;
  monitor: boolean;
  /** Send the voice to the Windows microphone where KamoVox is installed. */
  windowsMic: boolean;
  page: Page;
  profiles: Profile[];
  ai: AiSettings;
  /** The "how it works" card on the Voices page was dismissed. */
  guideDone: boolean;
  /** Action id → key combo, for example "Ctrl+Alt+1". */
  shortcuts: Record<string, string>;
  tracks: Track[];
  customs: CustomShortcut[];
  /** Hold a key to talk. The voice is silence until the key is down. */
  ptt: { enabled: boolean; inverted: boolean; key: string };
  /** Click-mute while push-to-talk is off. */
  micMuted: boolean;
  /** Launch KamoVox when Windows starts. */
  startWithWindows: boolean;
  /** system follows Windows; otherwise light or dark. */
  theme: "system" | "light" | "dark";
  /** Show experimental features (voice gate, smoother pitch grain, …). */
  beta: boolean;
  /** Last time the support note was closed, in ms. */
  supportAt: number;
  /** Time the voice changer stayed on, in ms. */
  usedMs: number;
  /** Measured speaking pitch from automatic calibration, in Hz. */
  calibration: { hz: number } | null;
  /** User-saved graphic-EQ curves. */
  eqPresets: EqPreset[];
}

interface EqPreset {
  id: string;
  name: string;
  eq: number[];
}

interface Endpoint {
  id: string;
  name: string;
  installed: boolean;
  other: string | null;
}

interface CatalogVoice {
  id: string;
  name: string;
  tags: string[];
  description: string;
  credit: string;
  terms: string;
  source: string;
  pitch: number;
  size: number;
}

interface InstalledVoice {
  id: string;
  name: string;
  tags: string[];
  credit: string;
  terms: string;
  source: string;
  pitch: number;
  imported: boolean;
  version: string;
  sampleRate: number;
  f0: boolean;
  /** Cover picture as a data URL. */
  image?: string | null;
}

interface Progress {
  id: string;
  received: number;
  total: number;
  done: boolean;
  error: string | null;
}

interface Library {
  modelsDir: string;
  baseInstalled: boolean;
  baseSize: number;
  speakerInstalled: boolean;
  speakerSize: number;
  voices: InstalledVoice[];
  catalog: CatalogVoice[];
  downloads: Progress[];
}

interface AiStats {
  inferMs: number;
  blockMs: number;
  load: number;
  latencyMs: number;
}

interface DeviceList {
  inputs: string[];
  outputs: string[];
  defaultInput: string | null;
  defaultOutput: string | null;
}

interface Status {
  running: boolean;
  sampleRate: number;
  input: string | null;
  output: string | null;
  monitoring: boolean;
  denoiseSupported: boolean;
  aiDevice: "gpu" | "cpu" | null;
  apo: boolean;
}

interface GateEventDto {
  utteranceId: number;
  prev: number;
  next: number;
  speechHops: number;
  neural: number;
  thresh: number;
  strict: number;
  protect: boolean;
  reason: number;
  delaySamples: number;
  inferId: number;
}

interface GateSnapshot {
  phase: number;
  reason: number;
  reasonLabel: string;
  phaseLabel: string;
  utteranceId: number;
  neural: number | null;
  thresh: number;
  strict: number;
  protect: boolean;
  delayMs: number;
  bufferedMs: number;
  bufferCapacityMs: number;
  speechHops: number;
  speechMs: number;
  speechEnv: number;
  inferId: number;
  inferMs: number;
  verifyAgeMs: number;
  authorized: boolean;
  events: GateEventDto[];
}

interface Levels {
  input: number;
  output: number;
  running: boolean;
  error: string | null;
  ai: AiStats | null;
  gate: GateSnapshot | null;
  buildId?: string;
}

const MAX_CHAIN = 4;

const defaultParams = (): Params => ({
  enabled: true,
  voices: [],
  pitch: 0,
  formant: 0,
  breath: 0,
  rasp: 0,
  vibrato: 0,
  nasal: 0,
  vocoder: null,
  vocoderMix: 1,
  cave: false,
  hall: false,
  chorus: false,
  ghost: false,
  echo: false,
  room: false,
  slap: false,
  flange: false,
  bottle: false,
  boost: 100,
  sensitivityDb: -90,
  bassDb: 0,
  trebleDb: 0,
  denoiseAmount: 0.75,
  noiseFloor: 0,
  impact: 0,
  compress: 0,
  voiceGate: false,
  voiceStrict: 0.35,
  voiceBands: Array(12).fill(0),
  voicePitch: 0,
  voicePitchLo: 0,
  voicePitchHi: 0,
  voiceLevel: 0,
  voiceF1: 0,
  voiceF2: 0,
  voiceF3: 0,
  voiceEmbed: [],
  voiceGallery: [],
  voiceNeuralLo: 0,
  voiceNeuralHi: 0,
  voiceNeuralReject: 0,
  voiceProtect: true,
  eq: Array(10).fill(0),
  denoise: false,
  appendOriginal: false,
  smooth: false,
});

interface Voice {
  id: string;
  name: string;
  category: string;
}

const EQ_BANDS = [
  { hz: 20, label: "20", color: "#bf5af2" },
  { hz: 50, label: "50", color: "#7d7aff" },
  { hz: 100, label: "100", color: "#0a84ff" },
  { hz: 200, label: "200", color: "#ff375f" },
  { hz: 500, label: "500", color: "#ff9f0a" },
  { hz: 1000, label: "1k", color: "#ffd60a" },
  { hz: 2000, label: "2k", color: "#c8f542" },
  { hz: 5000, label: "5k", color: "#30d158" },
  { hz: 10000, label: "10k", color: "#64d2ff" },
  { hz: 16000, label: "16k", color: "#5ac8fa" },
];
const EQ_GROUPS = [
  { name: "Subgrave", from: 20, to: 70 },
  { name: "Grave", from: 70, to: 250 },
  { name: "Médio grave", from: 250, to: 700 },
  { name: "Médio", from: 700, to: 2500 },
  { name: "Médio agudo", from: 2500, to: 6000 },
  { name: "Agudo", from: 6000, to: 16000 },
];
const EQ_FACTORY: EqPreset[] = [
  { id: "flat", name: "Plano", eq: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { id: "voice", name: "Voz", eq: [-2, -1, 0, 1, 2, 3, 2, 1, 0, -1] },
  { id: "podcast", name: "Podcast", eq: [-4, -3, -1, 1, 2, 3, 2, 0, -1, -2] },
  { id: "radio", name: "Rádio", eq: [-3, -1, 1, 2, 0, -1, 2, 3, 1, 0] },
  { id: "phone", name: "Telefone", eq: [-8, -6, -2, 3, 4, 3, 2, -2, -6, -8] },
  { id: "bass", name: "Grave", eq: [6, 5, 3, 1, 0, 0, 0, 0, 0, 0] },
  { id: "presence", name: "Presença", eq: [0, 0, 0, 0, 1, 3, 4, 2, 0, 0] },
  { id: "vocal", name: "Vocal", eq: [-1, 0, 1, 2, 1, 2, 3, 1, 0, -1] },
  { id: "bright", name: "Brilho", eq: [-1, 0, 0, 0, 0, 1, 2, 3, 4, 3] },
  { id: "warm", name: "Quente", eq: [3, 2, 1, 1, 0, -1, -2, -1, 0, 0] },
  { id: "smooth", name: "Suave", eq: [1, 1, 0, 0, 0, -1, -2, -3, -2, -1] },
  { id: "natural", name: "Voz natural", eq: [0.5, 0, 1, 0.5, 0, -0.5, -1, -2, -2.5, -1.5] },
  { id: "dark", name: "Escuro", eq: [1, 0.5, 0, 0, -1, -2, -4, -8, -10, -12] },
  { id: "deess", name: "Menos sibilância", eq: [0, 0, 0, 0, 0, 0, 1, -4, -3, 0] },
  { id: "clarity", name: "Clareza", eq: [-3, -2, 0, 0, 0, 1, 2, 2, 1, 0] },
];
const EQ_TOOLS: { id: string; name: string; add: number[] }[] = [
  { id: "more-bass", name: "+ Grave", add: [2.5, 2, 1.5, 0.5, 0, 0, 0, 0, 0, 0] },
  { id: "more-presence", name: "+ Presença", add: [0, 0, 0, 0, 0.5, 1.5, 2, 1, 0, 0] },
  { id: "more-air", name: "+ Brilho", add: [0, 0, 0, 0, 0, 0, 0.5, 1.5, 2, 1.5] },
  { id: "less-ess", name: "− Sibilância", add: [0, 0, 0, 0, 0, 0, 0, -2.5, -2, 0] },
  { id: "less-rumble", name: "− Rumble", add: [-3, -2.5, -1, 0, 0, 0, 0, 0, 0, 0] },
];

function clampEqBand(v: number) {
  return Math.max(-EQ_DB, Math.min(EQ_DB, Math.round(v * 2) / 2));
}

function eqCurvesClose(a: number[], b: number[]) {
  if (a.length !== b.length) return false;
  return a.every((v, i) => Math.abs(v - (b[i] ?? 0)) < 0.26);
}

function allEqPresets(): EqPreset[] {
  return [...EQ_FACTORY, ...settings.eqPresets];
}

function currentEqPresetId(): string | null {
  const eq = settings.params.eq;
  return allEqPresets().find((p) => eqCurvesClose(eq, p.eq))?.id ?? null;
}
const EQ_MIN_HZ = 20;
const EQ_MAX_HZ = 16000;
const EQ_DB = 12;

function eqX(hz: number, width: number, pad: number) {
  const t = Math.log(hz / EQ_MIN_HZ) / Math.log(EQ_MAX_HZ / EQ_MIN_HZ);
  return pad + t * (width - pad * 2);
}

function eqY(db: number, height: number, pad: number) {
  const t = (EQ_DB - db) / (EQ_DB * 2);
  return pad + t * (height - pad * 2);
}

function eqSummary() {
  const id = currentEqPresetId();
  const named = allEqPresets().find((p) => p.id === id);
  if (named && named.id !== "flat") return named.name;
  const n = settings.params.eq.filter((v) => Math.abs(v) >= 0.25).length;
  if (!n) return "Plano";
  return n === 1 ? "1 faixa" : `${n} faixas`;
}

function eqMarkup() {
  const w = 720;
  const h = 232;
  const pad = 16;
  const grids = [-12, -6, 0, 6, 12]
    .map((db) => {
      const y = eqY(db, h, pad);
      return `<line class="eq-grid" x1="${pad}" y1="${y}" x2="${w - pad}" y2="${y}" /><text class="eq-db" x="4" y="${y + 3}">${db > 0 ? "+" : ""}${db}</text>`;
    })
    .join("");
  const groups = EQ_GROUPS.map((g) => {
    const x1 = eqX(g.from, w, pad);
    const x2 = eqX(g.to, w, pad);
    return `<span style="left:${(x1 / w) * 100}%;width:${((x2 - x1) / w) * 100}%">${g.name}</span>`;
  }).join("");
  const freqs = EQ_BANDS.map((b) => `<span style="left:${(eqX(b.hz, w, pad) / w) * 100}%">${b.label}</span>`).join("");
  const active = currentEqPresetId();
  const chips = allEqPresets()
    .map((p) => {
      const user = !EQ_FACTORY.some((f) => f.id === p.id);
      return `<button type="button" class="chip ${p.id === active ? "active" : ""}" data-eq-preset="${esc(p.id)}">${esc(p.name)}${
        user ? `<span class="chip-x" data-eq-del="${esc(p.id)}" title="Excluir">×</span>` : ""
      }</button>`;
    })
    .join("");
  const tools = EQ_TOOLS.map((t) => `<button type="button" class="chip" data-eq-tool="${t.id}">${esc(t.name)}</button>`).join("");
  return `<div class="eq">
    <div class="eq-top"><span>Curva do microfone</span><button type="button" class="link" id="eq-reset">${icon("reset")}Zerar</button></div>
    <div class="eq-presets chips" role="list">${chips}</div>
    <div class="eq-tools chips">${tools}</div>
    <div class="eq-groups">${groups}</div>
    <svg id="eq-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="Equalizador gráfico">
      ${grids}
      <path id="eq-area" class="eq-area" />
      <path id="eq-curve" class="eq-curve" />
      <g id="eq-dots"></g>
    </svg>
    <div class="eq-freqs">${freqs}</div>
    <p class="eq-read" id="eq-read">Arraste um ponto. O ganho vai de −12 a +12 dB.</p>
    <div class="eq-save-row">
      <button type="button" class="btn" id="eq-save">${icon("save")}Salvar pré-definição</button>
    </div>
  </div>`;
}

let eqDrag = -1;

function paintEq() {
  const svg = document.getElementById("eq-svg") as SVGSVGElement | null;
  if (!svg) return;
  const w = 720;
  const h = 232;
  const pad = 16;
  const gains = settings.params.eq;
  const pts = EQ_BANDS.map((b, i) => ({ x: eqX(b.hz, w, pad), y: eqY(gains[i] ?? 0, h, pad), color: b.color, db: gains[i] ?? 0, label: b.label }));
  let line = `M${pts[0]!.x.toFixed(1)} ${pts[0]!.y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)]!;
    const p1 = pts[i]!;
    const p2 = pts[i + 1]!;
    const p3 = pts[Math.min(pts.length - 1, i + 2)]!;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    line += ` C${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  const zero = eqY(0, h, pad);
  $("#eq-curve").setAttribute("d", line);
  $("#eq-area").setAttribute("d", `${line} L${pts[pts.length - 1]!.x.toFixed(1)} ${zero} L${pts[0]!.x.toFixed(1)} ${zero} Z`);
  const dots = $("#eq-dots");
  let circles = dots.querySelectorAll("circle");
  if (circles.length !== pts.length) {
    dots.innerHTML = pts
      .map(
        (p, i) =>
          `<circle data-eq="${i}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${eqDrag === i ? 7 : 5.5}" fill="${p.color}" />`,
      )
      .join("");
    circles = dots.querySelectorAll("circle");
  }
  pts.forEach((p, i) => {
    const c = circles[i];
    if (!c) return;
    c.setAttribute("cx", p.x.toFixed(1));
    c.setAttribute("cy", p.y.toFixed(1));
    c.setAttribute("r", eqDrag === i ? "7" : "5.5");
  });
  if (eqDrag >= 0) return;
  const fold = document.querySelector('[data-fold="eq"] small');
  if (fold) fold.textContent = eqSummary();
}

function bindEq() {
  const svg = document.getElementById("eq-svg");
  if (!svg) return;
  paintEq();
  const h = 232;
  const pad = 16;
  const read = document.getElementById("eq-read");
  const curve = document.getElementById("eq-curve");
  const area = document.getElementById("eq-area");
  const circles = [...svg.querySelectorAll("circle")];
  const xs = EQ_BANDS.map((b) => eqX(b.hz, 720, pad));
  let dragRect: DOMRect | null = null;
  let eqFlight = false;
  let eqPending = false;
  let lastMove = 0;
  let worstGap = 0;
  let moves = 0;
  const curvePath = () => {
    const ys = settings.params.eq.map((db) => eqY(db, h, pad));
    let line = `M${xs[0]!.toFixed(1)} ${ys[0]!.toFixed(1)}`;
    for (let i = 0; i < xs.length - 1; i++) {
      const p0x = xs[Math.max(0, i - 1)]!;
      const p0y = ys[Math.max(0, i - 1)]!;
      const p1x = xs[i]!;
      const p1y = ys[i]!;
      const p2x = xs[i + 1]!;
      const p2y = ys[i + 1]!;
      const p3x = xs[Math.min(xs.length - 1, i + 2)]!;
      const p3y = ys[Math.min(ys.length - 1, i + 2)]!;
      const c1x = p1x + (p2x - p0x) / 6;
      const c1y = p1y + (p2y - p0y) / 6;
      const c2x = p2x - (p3x - p1x) / 6;
      const c2y = p2y - (p3y - p1y) / 6;
      line += ` C${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2x.toFixed(1)} ${p2y.toFixed(1)}`;
    }
    return { line, ys };
  };
  const paintDrag = () => {
    const { line, ys } = curvePath();
    const zero = eqY(0, h, pad);
    curve?.setAttribute("d", line);
    area?.setAttribute("d", `${line} L${xs[xs.length - 1]!.toFixed(1)} ${zero} L${xs[0]!.toFixed(1)} ${zero} Z`);
    const dot = circles[eqDrag];
    if (dot) {
      dot.setAttribute("cy", ys[eqDrag]!.toFixed(1));
      dot.setAttribute("r", "7");
    }
  };
  const pumpEq = () => {
    if (eqFlight || !eqPending) return;
    eqPending = false;
    eqFlight = true;
    const eq = settings.params.eq.slice();
    void invoke("set_eq", { eq }).finally(() => {
      eqFlight = false;
      if (eqPending) pumpEq();
    });
  };
  const setFromY = (clientY: number) => {
    if (eqDrag < 0 || !dragRect || dragRect.height <= 0) return;
    const now = performance.now();
    if (lastMove) worstGap = Math.max(worstGap, now - lastMove);
    lastMove = now;
    moves += 1;
    const y = ((clientY - dragRect.top) / dragRect.height) * h;
    const db = Math.max(-EQ_DB, Math.min(EQ_DB, Math.round((EQ_DB - ((y - pad) / (h - pad * 2)) * (EQ_DB * 2)) * 2) / 2));
    if (settings.params.eq[eqDrag] === db) return;
    settings.params.eq[eqDrag] = db;
    if (read) read.textContent = `${EQ_BANDS[eqDrag]!.label} Hz · ${db > 0 ? "+" : ""}${db.toFixed(1)} dB`;
    paintDrag();
    eqPending = true;
    pumpEq();
  };
  svg.addEventListener("pointerdown", (e) => {
    const ev = e as PointerEvent;
    dragRect = svg.getBoundingClientRect();
    const x = ((ev.clientX - dragRect.left) / Math.max(dragRect.width, 1)) * 720;
    let best = 0;
    let dist = Infinity;
    xs.forEach((bandX, i) => {
      const d = Math.abs(x - bandX);
      if (d < dist) {
        dist = d;
        best = i;
      }
    });
    if (dist > 36) {
      dragRect = null;
      return;
    }
    eqDrag = best;
    lastMove = 0;
    worstGap = 0;
    moves = 0;
    svg.setPointerCapture(ev.pointerId);
    if (ev.detail === 2) {
      settings.params.eq[best] = 0;
      paintEq();
      eqPending = true;
      pumpEq();
      pushParams();
    } else setFromY(ev.clientY);
  });
  svg.addEventListener("pointermove", (e) => setFromY((e as PointerEvent).clientY));
  const end = () => {
    if (eqDrag < 0) return;
    const gap = worstGap;
    const n = moves;
    eqDrag = -1;
    dragRect = null;
    paintEq();
    eqPending = true;
    pumpEq();
    pushParams();
    console.info(`eq-drag moves=${n} worstGapMs=${gap.toFixed(1)}`);
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  $("#eq-reset")?.addEventListener("click", () => {
    settings.params.eq = Array(10).fill(0);
    settings.params.bassDb = 0;
    settings.params.trebleDb = 0;
    $("#eq-read").textContent = "Arraste um ponto. O ganho vai de −12 a +12 dB.";
    paintEq();
    pushParams();
    rerender();
  });
  $$<HTMLButtonElement>("[data-eq-preset]", content).forEach((b) =>
    b.addEventListener("click", (e) => {
      const del = (e.target as HTMLElement).closest("[data-eq-del]");
      if (del) {
        e.preventDefault();
        e.stopPropagation();
        const id = (del as HTMLElement).dataset.eqDel!;
        const pr = settings.eqPresets.find((p) => p.id === id);
        if (!pr) return;
        confirmSheet(`Excluir “${pr.name}”?`, "A curva atual do equalizador não muda.", "Excluir", () => {
          settings.eqPresets = settings.eqPresets.filter((p) => p.id !== id);
          save();
          rerender();
          toast(`Pré-definição “${pr.name}” excluída`);
        });
        return;
      }
      const id = b.dataset.eqPreset!;
      const pr = allEqPresets().find((p) => p.id === id);
      if (!pr) return;
      settings.params.eq = pr.eq.slice();
      settings.params.bassDb = 0;
      settings.params.trebleDb = 0;
      $("#eq-read").textContent = `${pr.name}`;
      paintEq();
      pushParams();
      rerender();
    }),
  );
  $$<HTMLButtonElement>("[data-eq-tool]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const tool = EQ_TOOLS.find((t) => t.id === b.dataset.eqTool);
      if (!tool) return;
      settings.params.eq = settings.params.eq.map((v, i) => clampEqBand(v + (tool.add[i] ?? 0)));
      $("#eq-read").textContent = tool.name;
      paintEq();
      pushParams();
      rerender();
    }),
  );
  $("#eq-save")?.addEventListener("click", openEqPresetSheet);
}

function openEqPresetSheet() {
  openSheet(`
    <h3>Salvar equalizador</h3>
    <p>A curva atual vira uma pré-definição só sua, neste computador.</p>
    <input type="text" id="eq-preset-name" placeholder="Nome da pré-definição" maxlength="32" />
    <button class="btn block" id="eq-preset-save">Salvar</button>`);
  const input = $<HTMLInputElement>("#eq-preset-name");
  setTimeout(() => input.focus(), 250);
  const submit = () => {
    const name = input.value.trim();
    if (!name) return input.focus();
    const eq = settings.params.eq.slice();
    const existing = settings.eqPresets.findIndex((p) => p.name.toLowerCase() === name.toLowerCase());
    if (existing >= 0) settings.eqPresets[existing] = { ...settings.eqPresets[existing]!, name, eq };
    else settings.eqPresets.push({ id: `eq-${Date.now().toString(36)}`, name, eq });
    save();
    closeSheet();
    rerender();
    toast(`Pré-definição “${name}” salva`);
  };
  $("#eq-preset-save").addEventListener("click", submit);
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
}

const CATEGORIES = [
  { id: "people", name: "Pessoas" },
  { id: "creatures", name: "Criaturas" },
  { id: "machines", name: "Máquinas" },
  { id: "broadcast", name: "Transmissão" },
  { id: "ambient", name: "Ambiente" },
  { id: "all", name: "Todas" },
];

const VOICES: Voice[] = [
  { id: "male", name: "Masculina", category: "people" },
  { id: "female", name: "Feminina", category: "people" },
  { id: "baby", name: "Criança", category: "people" },
  { id: "giant", name: "Gigante", category: "people" },
  { id: "helium", name: "Hélio", category: "people" },
  { id: "demon", name: "Demônio", category: "creatures" },
  { id: "monster", name: "Monstro", category: "creatures" },
  { id: "alien", name: "Alienígena", category: "creatures" },
  { id: "mutation", name: "Mutação", category: "creatures" },
  { id: "slowMutation", name: "Mutação lenta", category: "creatures" },
  { id: "fastMutation", name: "Mutação rápida", category: "creatures" },
  { id: "robot", name: "Robô", category: "machines" },
  { id: "cyborg", name: "Ciborgue", category: "machines" },
  { id: "atari", name: "8-bit", category: "machines" },
  { id: "clone", name: "Clone", category: "machines" },
  { id: "radio", name: "Rádio", category: "broadcast" },
  { id: "telephone", name: "Telefone", category: "broadcast" },
  { id: "walkie", name: "Walkie-talkie", category: "broadcast" },
  { id: "megaphone", name: "Megafone", category: "broadcast" },
  { id: "underwater", name: "Subaquático", category: "ambient" },
  { id: "whisper", name: "Sussurro", category: "ambient" },
  { id: "silence", name: "Silêncio", category: "ambient" },
];

const CARRIERS: { id: Carrier; name: string }[] = [
  { id: "saw", name: "Sintetizador" },
  { id: "chord", name: "Acorde" },
  { id: "bell", name: "Sino" },
  { id: "organ", name: "Órgão" },
  { id: "choir", name: "Coral" },
  { id: "buzz", name: "Zumbido" },
  { id: "noise", name: "Vento" },
];

type EffectKey = "cave" | "hall" | "room" | "chorus" | "ghost" | "echo" | "slap" | "flange" | "bottle";
const EFFECTS: { key: EffectKey; name: string }[] = [
  { key: "cave", name: "Caverna" },
  { key: "hall", name: "Salão" },
  { key: "room", name: "Sala" },
  { key: "chorus", name: "Coro" },
  { key: "ghost", name: "Fantasma" },
  { key: "echo", name: "Eco" },
  { key: "slap", name: "Reflexo" },
  { key: "flange", name: "Jato" },
  { key: "bottle", name: "Garrafa" },
];

type CharacterKey = "pitch" | "formant" | "breath" | "rasp" | "vibrato" | "nasal";
type Character = Record<CharacterKey, number>;
const CHARACTER_KEYS: CharacterKey[] = ["pitch", "formant", "breath", "rasp", "vibrato", "nasal"];

const VOICE_SOURCES = [
  {
    name: "Hugging Face",
    hint: "Milhares de modelos RVC gratuitos. Procure arquivos .pth e leia a licença de cada um.",
    url: "https://huggingface.co/models?other=rvc&sort=downloads",
  },
  {
    name: "VCTK no Hugging Face",
    hint: "Mais 100 falantes em inglês com licença aberta (CC BY 4.0).",
    url: "https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample/tree/main",
  },
  {
    name: "Tsukuyomi-chan",
    hint: "Outras variações oficiais (forte, suave) no mesmo pacote.",
    url: "https://tyc.rei-yumesaki.net/work/software/rvc/",
  },
  {
    name: "Amitaro",
    hint: "Voz de anime gratuita para uso pessoal; baixe direto da autora.",
    url: "https://amitaro.net/synth/rvc/",
  },
  {
    name: "Weights",
    hint: "Grande acervo da comunidade. Muitos imitam pessoas reais: evite esses.",
    url: "https://www.weights.com/",
  },
];

const CHARACTERS: { name: string; c: Character }[] = [
  { name: "Homem", c: { pitch: -4, formant: -2, breath: 0, rasp: 0, vibrato: 0, nasal: 0 } },
  { name: "Mulher", c: { pitch: 5, formant: 2.5, breath: 0.1, rasp: 0, vibrato: 0, nasal: 0 } },
  { name: "Jovem", c: { pitch: 2, formant: 1.2, breath: 0, rasp: 0, vibrato: 0, nasal: 0 } },
  { name: "Criança", c: { pitch: 7, formant: 4.5, breath: 0, rasp: 0, vibrato: 0, nasal: 0 } },
  { name: "Idoso", c: { pitch: -2, formant: -0.8, breath: 0.25, rasp: 0.35, vibrato: 0.45, nasal: 0.1 } },
  { name: "Narrador", c: { pitch: -1.5, formant: -0.7, breath: 0, rasp: 0.08, vibrato: 0, nasal: 0 } },
  { name: "Vilão", c: { pitch: -3, formant: -1.5, breath: 0.2, rasp: 0.25, vibrato: 0, nasal: 0 } },
  { name: "Gigante", c: { pitch: -8, formant: -5, breath: 0, rasp: 0.2, vibrato: 0, nasal: 0 } },
  { name: "Anão", c: { pitch: -3, formant: 2.5, breath: 0, rasp: 0.3, vibrato: 0, nasal: 0 } },
  { name: "Orc", c: { pitch: -6, formant: -4, breath: 0, rasp: 0.7, vibrato: 0, nasal: 0 } },
  { name: "Bruxa", c: { pitch: 3, formant: 1, breath: 0, rasp: 0.45, vibrato: 0.15, nasal: 0.6 } },
  { name: "Fada", c: { pitch: 9, formant: 5, breath: 0.35, rasp: 0, vibrato: 0.2, nasal: 0 } },
  { name: "Pato", c: { pitch: 4, formant: 7, breath: 0, rasp: 0.55, vibrato: 0, nasal: 1 } },
  { name: "Máscara sombria", c: { pitch: -4, formant: -3, breath: 0.3, rasp: 0.15, vibrato: 0, nasal: 0 } },
  { name: "Sussurro", c: { pitch: 0, formant: 0, breath: 1, rasp: 0, vibrato: 0, nasal: 0 } },
  { name: "Cantor", c: { pitch: 0, formant: 0, breath: 0.1, rasp: 0, vibrato: 0.7, nasal: 0 } },
];

const AI_PRESETS: { id: AiPreset; name: string; timing: Timing; hint: string }[] = [
  { id: "fast", name: "Desempenho", timing: { blockMs: 160, contextMs: 300, crossfadeMs: 40 }, hint: "Menor atraso; um pouco menos natural" },
  { id: "balanced", name: "Equilibrado", timing: { blockMs: 250, contextMs: 600, crossfadeMs: 60 }, hint: "Bom para conversar e jogar" },
  { id: "quality", name: "Qualidade", timing: { blockMs: 400, contextMs: 1200, crossfadeMs: 100 }, hint: "Mais natural; mais atraso" },
  { id: "custom", name: "Personalizado", timing: { blockMs: 250, contextMs: 600, crossfadeMs: 60 }, hint: "Você escolhe cada ajuste" },
];

const defaultAi = (): AiSettings => ({
  enabled: false,
  voice: null,
  preset: "balanced",
  device: "gpu",
  pitchMethod: "rmvpe",
  custom: { ...AI_PRESETS[1].timing },
  transpose: {},
});

const NAV: { id: Page; name: string }[] = [
  { id: "voices", name: "Vozes" },
  { id: "effects", name: "Efeitos" },
  { id: "music", name: "Música" },
  { id: "shortcuts", name: "Atalhos" },
  { id: "mic", name: "Ajustes" },
  { id: "profiles", name: "Perfis" },
];

/** Pages that were merged into others in earlier versions. */
const MOVED_PAGES: Record<string, Page> = { tune: "voices", devices: "mic" };

// ---------- settings ----------

const STORAGE_KEY = "kamovox.settings.v1";
const VOICE_IDS = new Set(VOICES.map((v) => v.id));

function cleanParams(raw: Partial<Params> & { customPitch?: number }): Params {
  const p = { ...defaultParams(), ...raw };
  if (raw.voices?.includes("custom") && raw.pitch === undefined) p.pitch = raw.customPitch ?? 0;
  p.voices = (p.voices ?? []).filter((v) => VOICE_IDS.has(v)).slice(0, MAX_CHAIN);
  delete (p as { customPitch?: number }).customPitch;
  const incoming = Array.isArray(raw.eq) ? raw.eq : null;
  const eq = (incoming ?? []).slice(0, 10).map((v) => Math.max(-12, Math.min(12, Number(v) || 0)));
  while (eq.length < 10) eq.push(0);
  if (!incoming) {
    if (p.bassDb) {
      eq[2] = p.bassDb;
      eq[3] = p.bassDb * 0.5;
    }
    if (p.trebleDb) {
      eq[7] = p.trebleDb * 0.6;
      eq[8] = p.trebleDb;
    }
    p.bassDb = 0;
    p.trebleDb = 0;
  }
  p.eq = eq;
  const unit = (v: unknown, fallback: number) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : fallback;
  };
  p.denoiseAmount = unit(raw.denoiseAmount, 0.75);
  p.noiseFloor = unit(raw.noiseFloor, 0);
  p.impact = unit(raw.impact, 0);
  p.compress = unit(raw.compress, 0);
  p.voiceStrict = unit(raw.voiceStrict, 0.35);
  p.voiceGate = raw.voiceGate === true;
  const bands = Array.isArray(raw.voiceBands) ? raw.voiceBands.slice(0, 12).map((v) => Number(v) || 0) : [];
  while (bands.length < 12) bands.push(0);
  p.voiceBands = bands;
  p.voicePitch = Number(raw.voicePitch) > 60 ? Number(raw.voicePitch) : 0;
  p.voicePitchLo = Number(raw.voicePitchLo) > 60 ? Number(raw.voicePitchLo) : 0;
  p.voicePitchHi = Number(raw.voicePitchHi) > 60 ? Number(raw.voicePitchHi) : 0;
  p.voiceLevel = Number(raw.voiceLevel) > 0 ? Number(raw.voiceLevel) : 0;
  p.voiceF1 = Number(raw.voiceF1) > 80 ? Number(raw.voiceF1) : 0;
  p.voiceF2 = Number(raw.voiceF2) > 200 ? Number(raw.voiceF2) : 0;
  p.voiceF3 = Number(raw.voiceF3) > 400 ? Number(raw.voiceF3) : 0;
  const embed = Array.isArray(raw.voiceEmbed)
    ? raw.voiceEmbed.map((v) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
      })
    : [];
  p.voiceEmbed = embed.length === 512 ? embed : [];
  const gallery = Array.isArray(raw.voiceGallery)
    ? raw.voiceGallery.map((v) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
      })
    : [];
  p.voiceGallery =
    gallery.length >= 512 ? gallery.slice(0, Math.floor(gallery.length / 512) * 512).slice(0, 12 * 512) : [];
  p.voiceNeuralLo = Number(raw.voiceNeuralLo) > 0.05 ? Number(raw.voiceNeuralLo) : 0;
  p.voiceNeuralHi = Number(raw.voiceNeuralHi) > p.voiceNeuralLo ? Number(raw.voiceNeuralHi) : 0;
  p.voiceNeuralReject =
    Number(raw.voiceNeuralReject) > 0.05 && Number(raw.voiceNeuralReject) < p.voiceNeuralLo
      ? Number(raw.voiceNeuralReject)
      : 0;
  p.voiceProtect = raw.voiceProtect !== false;
  if (p.voicePitch <= 0) p.voiceGate = false;
  return p;
}

function loadSettings(): Settings {
  const base: Settings = {
    params: defaultParams(),
    mixer: false,
    category: "people",
    input: null,
    output: null,
    monitor: false,
    windowsMic: true,
    page: "voices",
    profiles: [],
    ai: defaultAi(),
    guideDone: false,
    shortcuts: {},
    tracks: [],
    customs: [],
    ptt: { enabled: false, inverted: false, key: "CapsLock" },
    micMuted: false,
    startWithWindows: false,
    theme: "system",
    beta: false,
    supportAt: 0,
    usedMs: 0,
    calibration: null,
    eqPresets: [],
  };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return base;
    const saved = JSON.parse(raw) as Partial<Settings>;
    const s = { ...base, ...saved, params: cleanParams(saved.params ?? {}) };
    s.category = base.category;
    s.ai = { ...defaultAi(), ...saved.ai };
    s.ai.custom = { ...defaultAi().custom, ...saved.ai?.custom };
    s.profiles = (saved.profiles ?? []).map((pr) => ({ name: pr.name, params: cleanParams(pr.params) }));
    s.shortcuts = saved.shortcuts ?? {};
    s.tracks = saved.tracks ?? [];
    s.customs = saved.customs ?? [];
    s.ptt = { ...base.ptt, ...saved.ptt };
    s.micMuted = saved.micMuted === true;
    s.startWithWindows = saved.startWithWindows ?? false;
    s.theme = saved.theme === "light" || saved.theme === "dark" ? saved.theme : "system";
    s.beta = saved.beta === true;
    s.supportAt = typeof saved.supportAt === "number" ? saved.supportAt : 0;
    if (!s.beta) {
      s.params.voiceGate = false;
      s.params.smooth = false;
    }
    s.usedMs = typeof saved.usedMs === "number" && saved.usedMs > 0 ? saved.usedMs : 0;
    s.calibration = typeof saved.calibration?.hz === "number" && saved.calibration.hz > 50 ? { hz: saved.calibration.hz } : null;
    s.eqPresets = Array.isArray(saved.eqPresets)
      ? saved.eqPresets
          .filter((p) => p && typeof p.name === "string" && Array.isArray(p.eq))
          .map((p, i) => ({
            id: typeof p.id === "string" && p.id ? p.id : `eq-saved-${i}`,
            name: p.name.trim().slice(0, 32) || `Curva ${i + 1}`,
            eq: p.eq.slice(0, 10).map((v) => clampEqBand(Number(v) || 0)).concat(Array(10).fill(0)).slice(0, 10),
          }))
      : [];
    s.page = MOVED_PAGES[s.page] ?? s.page;
    if (s.page !== "ai" && !NAV.some((n) => n.id === s.page)) s.page = "voices";
    return s;
  } catch {
    return base;
  }
}

const settings = loadSettings();

function applyTheme() {
  const root = document.documentElement;
  if (settings.theme === "system") root.removeAttribute("data-theme");
  else root.dataset.theme = settings.theme;
}

applyTheme();
let devices: DeviceList = { inputs: [], outputs: [], defaultInput: null, defaultOutput: null };
let endpoints: Endpoint[] = [];
let micBusy = false;
let status: Status | null = null;
let engineError: string | null = null;

const $ = <T extends HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
const $$ = <T extends HTMLElement>(sel: string, root: ParentNode = document) => [...root.querySelectorAll<T>(sel)];
const content = $("#content");

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch (e) {
    console.warn("kamovox: não deu para gravar as configurações", e);
  }
  scheduleTrayState();
  scheduleHotkeys();
}

let saveTimer: number | undefined;
let paramsFlight = false;
let paramsDirty = false;
function pushParams() {
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(save, 500);
  paramsDirty = true;
  if (paramsFlight) return;
  const send = () => {
    if (!paramsDirty) {
      paramsFlight = false;
      return;
    }
    paramsDirty = false;
    paramsFlight = true;
    void invoke("set_params", { params: settings.params }).finally(() => {
      if (paramsDirty) send();
      else paramsFlight = false;
    });
  };
  send();
}

let toastTimer: number | undefined;
function toast(msg: string) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 2800);
}

const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const fmt = (v: number, digits: number) =>
  v.toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const signed = (v: number, digits: number, unit = "") =>
  `${v > 0 ? "+" : ""}${fmt(v, digits)}${unit ? ` ${unit}` : ""}`;
const percent = (v: number) => `${Math.round(v)}%`;

// ---------- controls ----------

interface SliderDef {
  id: string;
  label: string;
  hint?: string;
  help?: string;
  min: number;
  max: number;
  step: number;
  def: number;
  centered?: boolean;
  stepper?: boolean;
  format: (v: number) => string;
  get: () => number;
  set: (v: number) => void;
  onChange?: () => void;
  /** Heavy work (reload the AI voice) waits until the slider is released. */
  commitOnRelease?: boolean;
  /** Frase curta que acompanha o valor, no mesmo card. */
  feel?: (v: number) => string;
}

let sliders: SliderDef[] = [];

function helpButton(text?: string) {
  return text ? `<button class="help" data-help="${esc(text)}" aria-label="Ajuda">${icon("help")}</button>` : "";
}

function pitchFeel(v: number): string {
  const amount = Math.abs(v);
  if (amount < 0.05) return "A voz fica no tom natural.";
  const dir = v > 0 ? "fina" : "grossa";
  if (amount <= 2) return `A voz fica um pouco mais ${dir}.`;
  if (amount <= 6) return `A voz fica mais ${dir}.`;
  if (amount <= 12) return `A voz fica muito ${dir}.`;
  return `A voz fica extremamente ${dir}.`;
}

function slider(d: SliderDef) {
  sliders.push(d);
  const row = `
    <div class="row slider" data-slider="${d.id}">
      <div class="label">
        <span class="title">${d.label}${helpButton(d.help)}</span>
        ${d.hint ? `<small>${d.hint}</small>` : ""}
      </div>
      <div class="slider-wrap">
        <input type="range" class="${d.centered ? "centered" : ""}" min="${d.min}" max="${d.max}" step="${d.step}" value="${d.get()}" />
      </div>
      <div class="value resettable" title="Clique duas vezes para zerar"></div>
      ${
        d.stepper
          ? `<div class="stepper"><button data-dir="-1" aria-label="Diminuir">${icon("minus")}</button><button data-dir="1" aria-label="Aumentar">${icon("plus")}</button></div>`
          : ""
      }
    </div>`;
  return d.feel ? `${row}<p class="slider-feel" data-feel="${d.id}">${esc(d.feel(d.get()))}</p>` : row;
}

const decimals = (step: number) => (String(step).split(".")[1] ?? "").length;

function bindSliders() {
  for (const d of sliders) {
    const row = $(`[data-slider="${d.id}"]`, content);
    if (!row) continue;
    const input = $<HTMLInputElement>("input", row);
    const out = $(".value", row);
    const digits = decimals(d.step);
    const quantize = (v: number) =>
      Number(Math.min(d.max, Math.max(d.min, Math.round((v - d.min) / d.step) * d.step + d.min)).toFixed(digits));

    const paint = () => {
      const v = d.get();
      input.value = String(v);
      const span = d.max - d.min;
      const pct = ((v - d.min) / span) * 100;
      if (d.centered) {
        const zero = ((Math.min(Math.max(0, d.min), d.max) - d.min) / span) * 100;
        const width = Math.abs(pct - zero);
        // background-position % aligns that % of the image with that % of the box.
        const pos = width >= 100 ? 0 : (Math.min(pct, zero) / (100 - width)) * 100;
        input.style.setProperty("--from", `${pos}%`);
        input.style.setProperty("--span", `${width}%`);
      } else {
        input.style.setProperty("--fill-pct", `${pct}%`);
      }
      out.textContent = d.format(v);
      const feel = row.parentElement?.querySelector(`[data-feel="${d.id}"]`);
      if (feel && d.feel) feel.textContent = d.feel(v);
    };
    const preview = (v: number) => {
      const q = quantize(v);
      if (q === d.get()) return;
      d.set(q);
      paint();
    };
    const publish = () => {
      if (d.commitOnRelease) return;
      if (d.onChange) d.onChange();
      else pushParams();
    };
    const release = () => {
      if (d.commitOnRelease && d.onChange) {
        save();
        d.onChange();
      } else if (!d.onChange) {
        pushParams();
      }
    };

    paint();
    input.addEventListener("input", () => {
      preview(Number(input.value));
      publish();
    });
    input.addEventListener("change", release);
    input.addEventListener("pointerup", release);
    input.addEventListener("keyup", release);
    out.addEventListener("dblclick", () => {
      preview(d.def);
      release();
      publish();
    });

    $$<HTMLButtonElement>(".stepper button", row).forEach((btn) => {
      const dir = Number(btn.dataset.dir);
      let timer: number | undefined;
      const stop = () => {
        window.clearTimeout(timer);
        release();
        publish();
      };
      const tick = (delay: number) => {
        preview(d.get() + dir * d.step);
        publish();
        timer = window.setTimeout(() => tick(Math.max(30, delay * 0.82)), delay);
      };
      btn.addEventListener("pointerdown", (e) => {
        btn.setPointerCapture(e.pointerId);
        tick(380);
      });
      btn.addEventListener("pointerup", stop);
      btn.addEventListener("pointercancel", stop);
      btn.addEventListener("lostpointercapture", stop);
    });
  }
  sliders = [];
}

function betaTag() {
  return `<span class="beta-tag" title="Recurso experimental">Beta</span>`;
}

function switchRow(
  id: string,
  label: string,
  hint: string,
  checked: boolean,
  opts: { disabled?: boolean; help?: string; icon?: string; tone?: string; beta?: boolean } = {},
) {
  return `
    <div class="row ${opts.disabled ? "disabled" : ""}">
      ${opts.icon ? `<span class="glyph-sm ${opts.tone ?? ""}">${icon(opts.icon)}</span>` : ""}
      <div class="label">
        <span class="title">${label}${opts.beta ? betaTag() : ""}${helpButton(opts.help)}</span>
        ${hint ? `<small>${hint}</small>` : ""}
      </div>
      <label class="switch"><input type="checkbox" id="${id}" ${checked ? "checked" : ""} /><span></span></label>
    </div>`;
}

const openSections = new Set<string>();

function disclosure(id: string, title: string, summary: string, body: string, glyph?: string, tone?: string) {
  const open = openSections.has(id);
  return `
    <section class="fold ${open ? "open" : ""}">
      <button type="button" class="fold-btn" data-fold="${id}" aria-expanded="${open}">
        ${glyph ? `<span class="glyph-sm ${tone ?? ""}">${icon(glyph)}</span>` : ""}
        <span class="label"><span class="title">${title}</span>${summary ? `<small>${summary}</small>` : ""}</span>
        <span class="fold-chev">${icon("chevron")}</span>
      </button>
      ${open ? `<div class="fold-body">${body}</div>` : ""}
    </section>`;
}

function bindFolds() {
  $$<HTMLButtonElement>("[data-fold]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const id = b.dataset.fold!;
      if (openSections.has(id)) openSections.delete(id);
      else openSections.add(id);
      rerender();
    }),
  );
}

function bindSwitch(id: string, apply: (v: boolean) => void) {
  $<HTMLInputElement>(`#${id}`)?.addEventListener("change", (e) => apply((e.target as HTMLInputElement).checked));
}

const segmentPos = new Map<string, { x: number; w: number }>();

function segmented(name: string, options: { id: string; label: string }[], active: string, wide = false) {
  return `
    <div class="segmented ${wide ? "wide" : ""}" data-seg="${name}">
      <div class="thumb"></div>
      ${options.map((o) => `<button data-value="${o.id}" class="${o.id === active ? "active" : ""}">${o.label}</button>`).join("")}
    </div>`;
}

function layoutSegmented() {
  $$(".segmented", content).forEach((seg) => {
    const thumb = $(".thumb", seg);
    const btn = $(".active", seg);
    if (!btn) return;
    const target = { x: btn.offsetLeft, w: btn.offsetWidth };
    const prev = segmentPos.get(seg.dataset.seg!);
    const place = (p: { x: number; w: number }) => {
      thumb.style.width = `${p.w}px`;
      thumb.style.transform = `translateX(${p.x}px)`;
    };
    if (prev && (prev.x !== target.x || prev.w !== target.w)) {
      thumb.style.transition = "none";
      place(prev);
      void thumb.offsetWidth;
      thumb.style.transition = "";
    }
    place(target);
    segmentPos.set(seg.dataset.seg!, target);
  });
}

function onSegment(name: string, handler: (value: string) => void) {
  $$<HTMLButtonElement>(`[data-seg="${name}"] button`, content).forEach((b) =>
    b.addEventListener("click", () => handler(b.dataset.value!)),
  );
}

// ---------- page plumbing ----------

let animateNext = true;
let popTile: string | null = null;

function mount(html: string) {
  const top = content.scrollTop;
  content.innerHTML = `<div class="page" style="${animateNext ? "" : "animation:none"}">${html}</div>`;
  if (!animateNext) content.scrollTop = top;
  animateNext = false;
  bindSliders();
  bindFolds();
  bindEq();
  layoutSegmented();
  if (popTile) {
    $(`[data-tile="${popTile}"]`, content)?.classList.add("pop");
    popTile = null;
  }
  $$<HTMLButtonElement>("[data-go]", content).forEach((b) => b.addEventListener("click", () => navigate(b.dataset.go as Page)));
}

function rerender() {
  PAGES[settings.page]();
}

/** Updates the AI page status in place; `false` when the layout must change. */
function refreshAiLive(): boolean {
  const sw = document.getElementById("ai-on") as HTMLInputElement | null;
  if (!sw) return false;
  const hint = sw.closest(".row")?.querySelector(".label small");
  if (hint) hint.textContent = aiStateText();
  sw.checked = settings.ai.enabled && !!library?.baseInstalled && !!settings.ai.voice;
  return true;
}

function head(title: string, subtitle: string, right = "") {
  return `
    <div class="page-head">
      <div><h1>${title}</h1><div class="subtitle">${subtitle}</div></div>
      ${right}
    </div>`;
}

function tile(id: string, iconName: string, name: string, active: boolean, order?: number) {
  return `
    <button class="tile ${active ? "active" : ""}" data-tile="${id}">
      ${order ? `<span class="order">${order}</span>` : ""}
      <div class="glyph">${icon(iconName)}</div>
      <div class="name">${name}</div>
    </button>`;
}

const pitchSlider = (hint?: string): SliderDef => ({
  id: "pitch",
  label: "Tom",
  hint,
  help: "Deixa a voz mais grave ou mais aguda, em semitons. O deslocamento é no domínio do tempo (WSOLA com interpolação cúbica), como o Clownfish em shifts pequenos. Em 0 o tom original passa intacto. Use − e + para 0,1.",
  min: -24,
  max: 24,
  step: 0.1,
  def: 0,
  centered: true,
  stepper: true,
  format: (v) => signed(v, 1, "st"),
  get: () => settings.params.pitch,
  set: (v) => (settings.params.pitch = v),
  feel: pitchFeel,
});

const formantSlider = (hint?: string): SliderDef => ({
  id: "formant",
  label: "Entonação",
  hint,
  help: "Muda o “tamanho” de quem fala (formantes). Em 0 o tom usa só o domínio do tempo — mais limpo, como o Clownfish em shifts pequenos. Longe de 0 ativa o ajuste espectral de timbre.",
  min: -12,
  max: 12,
  step: 0.1,
  def: 0,
  centered: true,
  stepper: true,
  format: (v) => signed(v, 1),
  get: () => settings.params.formant,
  set: (v) => (settings.params.formant = v),
});

// ---------- pages ----------

function guideCard() {
  if (settings.guideDone) return "";
  const step = (n: number, title: string, text: string) =>
    `<div class="step"><span class="num">${n}</span><div><b>${title}</b><small>${text}</small></div></div>`;
  return `
    <div class="guide">
      <div class="guide-head">
        <div><h3>Como funciona</h3><small>Três passos para começar a falar com outra voz.</small></div>
        <button class="btn plain" id="guide-done">Entendi</button>
      </div>
      <div class="steps">
        ${step(1, "Instale no microfone", "Em <b>Ajustes</b>, toque em Instalar no microfone que o Discord usa. Sem isso, a chamada ouve a sua voz normal.")}
        ${step(2, "Escolha uma voz", "Toque numa voz pronta ou abra a aba <b>Voz IA</b>.")}
        ${step(3, "Ligue e fale", "O botão <b>Trocador de voz</b>, no canto inferior esquerdo, liga e desliga tudo.")}
      </div>
      <button class="btn plain" data-go="mic">${icon("gear")}Abrir ajustes</button>
    </div>`;
}

function voiceKindTabs() {
  const on = settings.page === "ai" ? "ai" : "voices";
  const card = (id: "voices" | "ai", title: string, hint: string, glyph: string) => `
    <button type="button" class="kind ${on === id ? "active" : ""}" data-kind="${id}" aria-pressed="${on === id}">
      <span class="kind-ic">${icon(glyph)}</span>
      <span class="kind-copy"><b>${title}</b><small>${hint}</small></span>
    </button>`;
  return `<div class="kind-switch" role="tablist" aria-label="Tipo de voz">
    ${card("voices", "Vozes prontas", "Tom e personagens", "voices")}
    ${card("ai", "Voz IA", "Voz de um personagem", "ai")}
  </div>`;
}

function bindVoiceKindTabs() {
  $$<HTMLButtonElement>("[data-kind]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const page = b.dataset.kind as Page;
      if (page !== settings.page) navigate(page);
    }),
  );
}

function renderVoices() {
  const p = settings.params;
  const activeCharacter = CHARACTERS.find((ch) => sameCharacter(ch.c));
  const list = VOICES.filter((v) => settings.category === "all" || v.category === settings.category);
  const amount = (key: "breath" | "rasp" | "vibrato" | "nasal", label: string, hint: string, help: string): SliderDef => ({
    id: key,
    label,
    hint,
    help,
    min: 0,
    max: 100,
    step: 1,
    def: 0,
    stepper: true,
    format: percent,
    get: () => Math.round(p[key] * 100),
    set: (v) => (p[key] = v / 100),
  });
  mount(`
    ${voiceKindTabs()}
    ${guideCard()}
    <div class="section-head">
      <h2>Vozes prontas</h2>
      ${segmented("mode", [
        { id: "single", label: "Uma voz" },
        { id: "mixer", label: "Combinar" },
      ], settings.mixer ? "mixer" : "single")}
    </div>
    <p class="section-hint">${
      settings.mixer
        ? `Toque em até ${MAX_CHAIN} vozes; elas são aplicadas na ordem dos números.`
        : "Toque numa voz para ativar e toque de novo para voltar à sua voz natural."
    }</p>
    <div class="chips filters">
      ${[...CATEGORIES.filter((c) => c.id === "all"), ...CATEGORIES.filter((c) => c.id !== "all")]
        .map((c) => `<button type="button" class="chip ${settings.category === c.id ? "active" : ""}" data-category="${c.id}">${c.name}</button>`)
        .join("")}
    </div>
    <div class="tiles">
      ${list
        .map((v) => {
          const idx = p.voices.indexOf(v.id);
          return tile(v.id, v.id, v.name, idx >= 0, settings.mixer && idx >= 0 ? idx + 1 : undefined);
        })
        .join("")}
    </div>

    ${disclosure(
      "characters",
      "Personagens",
      activeCharacter ? `Usando ${esc(activeCharacter.name)}` : `${CHARACTERS.length} atalhos`,
      `<p class="section-hint">Ajustam tom, timbre e textura de uma vez.</p>
       <div class="group"><div class="row"><div class="chips">
         ${CHARACTERS.map((ch, i) => `<button class="chip ${sameCharacter(ch.c) ? "active" : ""}" data-character="${i}">${ch.name}</button>`).join("")}
       </div></div></div>`,
      "sparkle",
      "tone-purple",
    )}

    <div class="cols">
      <div>
        <div class="section-head">
          <h2>Tom e timbre</h2>
          ${p.voices.length ? `<button class="link" id="voice-cal">${icon("sparkle")}Calibrar</button>` : ""}
        </div>
        <div class="group">
          ${slider(pitchSlider())}
          ${slider(formantSlider("Corpo maior ↔ menor"))}
        </div>
      </div>
      <div>
        <div class="section-head">
          <h2>Textura</h2>
          <button class="link" id="tune-reset">${icon("reset")}Zerar</button>
        </div>
        <div class="group">
          ${slider(amount("breath", "Respiração", "Voz soprada, com ar", "Mistura ar à voz, como alguém ofegante, cansado ou sussurrando."))}
          ${slider(amount("rasp", "Rouquidão", "Áspera, rasgada", "Deixa a voz rouca e granulada: ótimo para idosos, monstros, piratas e vilões."))}
          ${slider(amount("vibrato", "Vibrato", "Tremor no tom", "Faz o tom oscilar levemente, como voz trêmula de idoso ou de alguém com medo."))}
          ${slider(amount("nasal", "Nasal", "Som pelo nariz", "Realça a ressonância nasal, como voz de bruxa, de duende ou de alguém resfriado."))}
        </div>
      </div>
    </div>`);

  $("#guide-done")?.addEventListener("click", () => {
    settings.guideDone = true;
    save();
    renderVoices();
  });
  $$<HTMLButtonElement>("[data-character]", content).forEach((b) =>
    b.addEventListener("click", () => {
      Object.assign(p, CHARACTERS[Number(b.dataset.character)].c);
      pushParams();
      renderVoices();
    }),
  );
  $("#voice-cal")?.addEventListener("click", () => useSavedCalibration("classic"));
  $("#tune-reset").addEventListener("click", () => {
    for (const k of CHARACTER_KEYS) p[k] = 0;
    pushParams();
    renderVoices();
  });
  onSegment("mode", (v) => {
    settings.mixer = v === "mixer";
    if (!settings.mixer) p.voices = p.voices.slice(0, 1);
    pushParams();
    renderVoices();
  });
  $$<HTMLButtonElement>("[data-category]", content).forEach((b) =>
    b.addEventListener("click", () => {
      settings.category = b.dataset.category!;
      save();
      renderVoices();
    }),
  );
  $$<HTMLButtonElement>("[data-tile]", content).forEach((b) => b.addEventListener("click", () => toggleVoice(b.dataset.tile!)));
  bindVoiceKindTabs();
}

function toggleVoice(id: string) {
  const p = settings.params;
  const idx = p.voices.indexOf(id);
  if (settings.mixer) {
    if (idx >= 0) p.voices.splice(idx, 1);
    else if (p.voices.length < MAX_CHAIN) p.voices.push(id);
    else return toast(`O mixer aceita até ${MAX_CHAIN} vozes.`);
  } else {
    p.voices = idx >= 0 ? [] : [id];
  }
  if (idx < 0) popTile = id;
  pushParams();
  renderVoices();
}

// ---------- AI voice ----------

let library: Library | null = null;
const downloads = new Map<string, Progress>();
let aiLoading = false;
const LIBRARY_PREVIEW = 12;
let libraryExpanded = false;

function avatar(v: InstalledVoice, cls = "avatar") {
  return v.image
    ? `<div class="${cls} photo"><img src="${v.image}" alt="" draggable="false" /></div>`
    : `<div class="${cls}">${esc(v.name.trim().charAt(0).toUpperCase() || "?")}</div>`;
}

function openVoiceEditor(id: string) {
  const voice = () => library?.voices.find((v) => v.id === id);
  const v = voice();
  if (!v) return;
  const update = async (image?: string) => {
    const name = $<HTMLInputElement>("#voice-name").value.trim() || voice()!.name;
    library = await invoke<Library>("ai_update_voice", { id, name, image: image ?? null });
    if (settings.page === "ai") renderAi();
    scheduleTrayState();
  };
  const paint = () => {
    const cur = voice();
    if (!cur) return;
    $("#voice-preview").outerHTML = avatar(cur, "avatar big").replace('class="', 'id="voice-preview" class="');
    $("#voice-image-clear").toggleAttribute("hidden", !cur.image);
  };
  openSheet(`
    <h3>Editar voz</h3>
    <p>Dê um nome fácil de reconhecer e, se quiser, uma imagem.</p>
    <div class="voice-editor">
      ${avatar(v, "avatar big").replace('class="', 'id="voice-preview" class="')}
      <div class="voice-editor-actions">
        <button class="btn plain" id="voice-image">${icon("image")}Escolher imagem</button>
        <button class="btn danger" id="voice-image-clear" ${v.image ? "" : "hidden"}>${icon("trash")}Tirar imagem</button>
      </div>
    </div>
    <label class="field-label" for="voice-name">Nome</label>
    <input type="text" id="voice-name" maxlength="40" value="${esc(v.name)}" />
    <button class="btn block" id="voice-save">Salvar</button>`);
  const input = $<HTMLInputElement>("#voice-name");
  $("#voice-image").addEventListener("click", async () => {
    const file = await openDialog({
      title: "Imagem da voz",
      filters: [{ name: "Imagem", extensions: ["png", "jpg", "jpeg", "webp", "gif"] }],
    });
    if (typeof file !== "string") return;
    try {
      await update(file);
      paint();
    } catch (e) {
      toast(String(e));
    }
  });
  $("#voice-image-clear").addEventListener("click", async () => {
    try {
      await update("");
      paint();
    } catch (e) {
      toast(String(e));
    }
  });
  const submit = async () => {
    if (!input.value.trim()) return input.focus();
    try {
      await update();
      closeSheet();
      toast("Voz atualizada");
    } catch (e) {
      toast(String(e));
    }
  };
  $("#voice-save").addEventListener("click", submit);
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
}

const mb = (bytes: number) => `${fmt(bytes / 1e6, 0)} MB`;

const aiTiming = (): Timing =>
  settings.ai.preset === "custom" ? settings.ai.custom : AI_PRESETS.find((p) => p.id === settings.ai.preset)!.timing;

function aiTranspose(id: string | null): number {
  if (!id) return 0;
  return settings.ai.transpose[id] ?? library?.voices.find((v) => v.id === id)?.pitch ?? 0;
}

function aiRequest() {
  const a = settings.ai;
  if (!a.enabled || !a.voice || !library?.baseInstalled || !library.voices.some((v) => v.id === a.voice)) return null;
  return { voice: a.voice, device: a.device, pitch: a.pitchMethod, timing: aiTiming(), transpose: aiTranspose(a.voice) };
}

async function refreshLibrary() {
  try {
    library = await invoke<Library>("ai_library");
    for (const d of library.downloads) downloads.set(d.id, d);
  } catch (e) {
    toast(String(e));
  }
}

/** The window can load before the backend finishes setting up its state. */
async function waitForBackend() {
  for (let i = 0; i < 100; i++) {
    try {
      library = await invoke<Library>("ai_library");
      for (const d of library.downloads) downloads.set(d.id, d);
      return;
    } catch {
      await sleep(100);
    }
  }
  await refreshLibrary();
}

let aiRestartTimer: number | undefined;
function scheduleAiRestart(delay = 600) {
  window.clearTimeout(aiRestartTimer);
  if (!settings.ai.enabled) return;
  aiRestartTimer = window.setTimeout(() => restartEngine(), delay);
}

function progressBar(id: string) {
  const d = downloads.get(id);
  const pct = d && d.total ? (d.received / d.total) * 100 : 0;
  return `
    <div class="download" data-progress="${id}">
      <div class="bar"><div style="width:${pct}%"></div></div>
      <span class="pct">${Math.floor(pct)}%</span>
      <button class="btn danger" data-cancel="${id}" aria-label="Cancelar">${icon("close")}</button>
    </div>`;
}

const downloading = (id: string) => {
  const d = downloads.get(id);
  return !!d && !d.done;
};

function aiStateText(): string {
  const a = settings.ai;
  if (!library?.baseInstalled) return "Instale os componentes de IA primeiro";
  if (!library.voices.length) return "Baixe ou importe uma voz";
  if (!a.voice) return "Escolha uma voz abaixo";
  if (aiLoading) return "Carregando a voz…";
  if (a.enabled && status?.aiDevice) return status.aiDevice === "gpu" ? "Ativa na placa de vídeo" : "Ativa no processador";
  return "Desligada";
}

function aiTotalText() {
  const t = aiTiming();
  return `≈ ${Math.round(t.blockMs + t.crossfadeMs + 10)} ms + processamento`;
}

async function rescanModels() {
  try {
    const r = await invoke<{ added: number; failed: string[]; library: Library }>("ai_rescan");
    library = r.library;
    const parts = [r.added ? `${r.added} ${r.added === 1 ? "voz adicionada" : "vozes adicionadas"}` : "Nenhuma voz nova na pasta"];
    if (r.failed.length) parts.push(`${r.failed.length} arquivo(s) não são modelos RVC: ${r.failed.join(", ")}`);
    toast(parts.join(" · "));
  } catch (e) {
    toast(String(e));
  }
  renderAi();
}

function renderAi() {
  const a = settings.ai;
  const lib = library;
  if (!lib) {
    mount(`${voiceKindTabs()}`);
    bindVoiceKindTabs();
    return;
  }
  const ready = lib.baseInstalled && lib.voices.length > 0;
  const selected = lib.voices.find((v) => v.id === a.voice) ?? null;
  const available = lib.catalog.filter((c) => !lib.voices.some((v) => v.id === c.id));
  const active = a.enabled && !!status?.aiDevice;

  const setup = lib.baseInstalled
    ? ""
    : `
    <div class="group setup">
      <div class="row">
        <div class="glyph">${icon("ai")}</div>
        <div class="label">
          <span class="title">Componentes de IA</span>
          <small>Necessários uma única vez · ${mb(lib.baseSize)} · salvos em ${esc(lib.modelsDir)}</small>
        </div>
        ${downloading("base") ? progressBar("base") : `<button class="btn" data-download="base">${icon("save")}Baixar</button>`}
      </div>
    </div>`;

  const voiceCard = (v: InstalledVoice) => `
    <div class="voice-card ${v.id === a.voice ? "active" : ""}" data-voice="${esc(v.id)}" role="button" tabindex="0" aria-label="Usar a voz ${esc(v.name)}">
      ${avatar(v)}
      <div class="name">${esc(v.name)}</div>
      ${v.id === a.voice ? `<button type="button" class="voice-cal" data-calibrate="${esc(v.id)}">Calibrar</button>` : ""}
      <div class="card-actions">
        <button class="card-btn" data-edit="${esc(v.id)}" aria-label="Editar nome e imagem" title="Editar">${icon("edit")}</button>
        <button class="card-btn danger" data-remove="${esc(v.id)}" aria-label="Apagar voz" title="Apagar">${icon("trash")}</button>
      </div>
    </div>`;

  const catalogCard = (c: CatalogVoice) => `
    <button type="button" class="lib-card" data-download="${esc(c.id)}" ${downloading(c.id) ? "disabled" : ""}>
      <span class="avatar">${esc(c.name.trim().charAt(0).toUpperCase())}</span>
      <span class="name">${esc(c.name)}</span>
      ${downloading(c.id) ? `<span class="get">baixando</span>` : `<span class="get">Baixar</span>`}
    </button>`;
  // Downloads in progress stay visible even when the list is collapsed.
  const shown = libraryExpanded
    ? available
    : available.filter((c, i) => i < LIBRARY_PREVIEW || downloading(c.id));
  const hidden = available.length - shown.length;

  const timingSlider = (key: keyof Timing, label: string, hint: string, help: string, min: number, max: number, step: number): SliderDef => ({
    id: `ai-${key}`,
    label,
    hint,
    help,
    min,
    max,
    step,
    def: AI_PRESETS[1].timing[key],
    stepper: true,
    format: (v) => `${v} ms`,
    get: () => a.custom[key],
    set: (v) => {
      a.custom[key] = v;
      const total = $("#ai-total");
      if (total) total.textContent = aiTotalText();
    },
    onChange: () => scheduleAiRestart(),
    commitOnRelease: true,
  });
  const preset = AI_PRESETS.find((p) => p.id === a.preset)!;

  if ([...downloads.values()].some((d) => !d.done && d.id !== "base")) openSections.add("library");

  mount(`
    ${voiceKindTabs()}
    ${setup}
    <div class="group ${ready ? "" : "disabled"}">
      ${switchRow("ai-on", "Usar voz IA", aiStateText(), a.enabled && ready && !!a.voice, {
        disabled: !ready || !a.voice,
        icon: "ai",
        tone: "tone-purple",
        help: "Enquanto a voz IA estiver ligada, ela substitui as vozes prontas. Efeitos, textura e os ajustes do microfone continuam valendo.",
      })}
    </div>

    <div class="section-head">
      <h2>Suas vozes</h2>
      <button class="link" id="ai-import-open">${icon("folder")}Importação</button>
    </div>
    <p class="section-hint">${lib.voices.length ? "Toque numa voz para usar." : "Baixe uma voz da biblioteca ou importe um modelo."}</p>
    ${
      lib.voices.length
        ? `<div class="voice-cards">${lib.voices.map(voiceCard).join("")}</div>`
        : `<div class="group"><div class="empty">Nenhuma voz ainda.</div></div>`
    }
    ${selected?.credit ? `<div class="credit">Voz: ${esc(selected.credit)}</div>` : ""}

    <h2>Tom</h2>
    <div class="group ${selected ? "" : "disabled"}">
      ${slider({
        id: "ai-transpose",
        label: "Tom da voz IA",
        help: "Move o tom antes da conversão, em semitons. Homem para voz feminina: cerca de +12. Mulher para voz masculina: cerca de −12. Muda na hora, sem recarregar.",
        min: -24,
        max: 24,
        step: 0.1,
        def: selected?.pitch ?? 0,
        centered: true,
        stepper: true,
        format: (v) => signed(v, 1, "st"),
        get: () => aiTranspose(a.voice),
        set: (v) => {
          if (a.voice) a.transpose[a.voice] = v;
        },
        onChange: () => invoke("set_ai_pitch", { semitones: aiTranspose(a.voice) }).catch(() => {}),
        feel: pitchFeel,
      })}
      ${slider({
        id: "ai-formant",
        label: "Entonação",
        help: "Mexe no corpo da voz depois da conversão, sem trocar o tom. Negativo soa maior e mais grave no timbre. Positivo soa menor e mais brilhante.",
        min: -12,
        max: 12,
        step: 0.1,
        def: 0,
        centered: true,
        stepper: true,
        format: (v) => signed(v, 1),
        get: () => settings.params.formant,
        set: (v) => (settings.params.formant = v),
      })}
    </div>
    ${disclosure(
      "ai-advanced",
      "Qualidade",
      `${preset.name} · <span id="ai-total">${aiTotalText()}</span>`,
      `${segmented("ai-preset", AI_PRESETS.map((p) => ({ id: p.id, label: p.name })), a.preset, true)}
       <div class="group">
         <div class="row"><div class="label"><span class="title">${preset.name}</span><small>${preset.hint}</small></div>
           ${active ? `<div class="ai-stats"><span>Atraso <b id="ai-latency">—</b></span><span>Uso <b id="ai-load">—</b></span></div>` : ""}
         </div>
         ${
           a.preset === "custom"
             ? [
                 slider(timingSlider("blockMs", "Bloco", "Áudio convertido por vez", "Blocos maiores aliviam a placa de vídeo, mas aumentam o atraso na mesma medida.", 100, 800, 10)),
                 slider(timingSlider("contextMs", "Contexto", "Áudio anterior analisado", "Quanto mais contexto, mais natural e estável a voz. Não aumenta o atraso, mas exige mais processamento.", 100, 2500, 50)),
                 slider(timingSlider("crossfadeMs", "Transição", "Suaviza a emenda entre blocos", "Transições maiores evitam estalos entre blocos; somam ao atraso.", 20, 150, 5)),
               ].join("")
             : ""
         }
         <div class="row">
           <div class="label"><span class="title">Processar na</span><small>A placa de vídeo é bem mais rápida</small></div>
           ${segmented("ai-device", [{ id: "gpu", label: "Placa de vídeo" }, { id: "cpu", label: "Processador" }], a.device)}
         </div>
         <div class="row">
           <div class="label"><span class="title">Detecção de tom</span>${helpButton("A detecção precisa (RMVPE) acompanha melhor a entonação. A leve (YIN) usa menos recursos.")}</div>
           ${segmented("ai-method", [{ id: "rmvpe", label: "Precisa" }, { id: "yin", label: "Leve" }], a.pitchMethod)}
         </div>
       </div>`,
      "quality",
      "tone-purple",
    )}

    ${
      available.length
        ? disclosure(
            "library",
            "Biblioteca",
            `${available.length} ${available.length === 1 ? "voz" : "vozes"}`,
            `<p class="section-hint">Toque em Baixar. A voz entra na sua lista sem abrir outra janela.</p>
             <div class="lib-grid">${shown.map(catalogCard).join("")}
               ${
                 available.length > LIBRARY_PREVIEW
                   ? `<button class="row more" id="library-more">${libraryExpanded ? "Ver menos" : `Ver mais (${hidden})`}${icon("chevron")}</button>`
                   : ""
               }
             </div>`,
            "save",
          )
        : ""
    }
    ${disclosure(
      "sources",
      "Onde encontrar mais vozes",
      "Hugging Face, VCTK e outros",
      `<div class="group">${VOICE_SOURCES.map(
        (s) => `<div class="row">
          <span class="glyph-sm">${icon("external")}</span>
          <div class="label"><span class="title">${esc(s.name)}</span><small>${esc(s.hint)}</small></div>
          <button class="btn plain" data-url="${esc(s.url)}">Abrir</button>
        </div>`,
      ).join("")}</div>`,
      "external",
    )}`);

  bindSwitch("ai-on", (on) => {
    a.enabled = on;
    save();
    restartEngine();
  });
  $$<HTMLElement>("[data-voice]", content).forEach((el) => {
    const pick = () => {
      if (a.voice === el.dataset.voice && a.enabled) return;
      a.voice = el.dataset.voice!;
      a.enabled = lib.baseInstalled;
      save();
      restartEngine();
      renderAi();
    };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && pick());
  });
  $$<HTMLButtonElement>("[data-calibrate]", content).forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      useSavedCalibration("ai", b.dataset.calibrate);
    }),
  );
  $$<HTMLButtonElement>("[data-remove]", content).forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = b.dataset.remove!;
      const name = lib.voices.find((v) => v.id === id)?.name ?? id;
      confirmSheet(`Apagar a voz “${name}”?`, "A pasta dessa voz será apagada deste computador. Você pode baixá-la de novo depois.", "Apagar voz", async () => {
        if (a.voice === id) {
          a.voice = null;
          a.enabled = false;
          save();
          await restartEngine();
        }
        try {
          library = await invoke<Library>("ai_remove", { id });
          toast(`Voz “${name}” apagada`);
        } catch (err) {
          toast(String(err));
        }
        renderAi();
      });
    }),
  );
  $$<HTMLButtonElement>("[data-edit]", content).forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      openVoiceEditor(b.dataset.edit!);
    }),
  );
  $("#library-more")?.addEventListener("click", () => {
    libraryExpanded = !libraryExpanded;
    renderAi();
  });
  $$<HTMLButtonElement>("[data-download]", content).forEach((b) =>
    b.addEventListener("click", async () => {
      const id = b.dataset.download!;
      downloads.set(id, { id, received: 0, total: 0, done: false, error: null });
      renderAi();
      invoke("ai_download", { id }).catch((e) => {
        downloads.delete(id);
        toast(String(e));
        renderAi();
      });
    }),
  );
  $$<HTMLButtonElement>("[data-cancel]", content).forEach((b) =>
    b.addEventListener("click", () => invoke("ai_cancel", { id: b.dataset.cancel })),
  );
  $$<HTMLButtonElement>("[data-url]", content).forEach((b) =>
    b.addEventListener("click", () => openUrl(b.dataset.url!).catch((e) => toast(String(e)))),
  );
  $("#ai-import-open").addEventListener("click", openImportSheet);
  bindVoiceKindTabs();
  onSegment("ai-preset", (v) => {
    a.preset = v as AiPreset;
    save();
    renderAi();
    scheduleAiRestart(150);
  });
  onSegment("ai-device", (v) => {
    a.device = v as AiSettings["device"];
    save();
    renderAi();
    scheduleAiRestart(150);
  });
  onSegment("ai-method", (v) => {
    a.pitchMethod = v as AiSettings["pitchMethod"];
    save();
    renderAi();
    scheduleAiRestart(150);
  });
}

function openImportSheet() {
  const lib = library;
  if (!lib) return;
  openSheet(`
    <h3>Importação</h3>
    <p>Coloque arquivos .pth nesta pasta ou importe um agora. Atualizar procura as vozes novas.</p>
    <div class="sheet-path" id="import-path">${esc(lib.modelsDir)}</div>
    <div class="sheet-actions">
      <button class="btn block" id="ai-import">${icon("upload")}Importar .pth</button>
      <button class="btn block plain" id="ai-rescan">${icon("refresh")}Atualizar a pasta</button>
      <button class="btn block plain" id="ai-open-dir">Abrir pasta</button>
      <button class="btn block plain" id="ai-change-dir">Alterar pasta</button>
    </div>`);
  $("#ai-import").addEventListener("click", () => {
    closeSheet();
    importVoice();
  });
  $("#ai-rescan").addEventListener("click", () => rescanModels());
  $("#ai-open-dir").addEventListener("click", () => openPath(lib.modelsDir).catch((e) => toast(String(e))));
  $("#ai-change-dir").addEventListener("click", async () => {
    const dir = await openDialog({ directory: true, title: "Pasta dos modelos de IA" });
    if (typeof dir !== "string") return;
    try {
      library = await invoke<Library>("ai_set_models_dir", { path: dir });
      const path = $("#import-path");
      if (path && library) path.textContent = library.modelsDir;
      toast("Pasta alterada");
      if (settings.ai.enabled) restartEngine();
    } catch (e) {
      toast(String(e));
    }
    if (settings.page === "ai") renderAi();
  });
}

async function importVoice() {
  const file = await openDialog({ title: "Importar voz RVC", filters: [{ name: "Modelo RVC", extensions: ["pth"] }] });
  if (typeof file !== "string") return;
  const before = new Set(library?.voices.map((v) => v.id));
  try {
    toast("Verificando o modelo…");
    library = await invoke<Library>("ai_import", { path: file });
    const added = library.voices.find((v) => !before.has(v.id));
    if (added) {
      toast(`Voz “${added.name}” importada`);
      settings.ai.voice = added.id;
      settings.ai.enabled = library.baseInstalled;
      save();
      restartEngine();
    }
  } catch (e) {
    toast(String(e));
  }
  if (settings.page === "ai") renderAi();
}

function onDownload(p: Progress) {
  downloads.set(p.id, p);
  if (!p.done) {
    const el = $(`[data-progress="${p.id}"]`, content);
    if (el) {
      const pct = p.total ? (p.received / p.total) * 100 : 0;
      $<HTMLElement>(".bar div", el).style.width = `${pct}%`;
      $(".pct", el).textContent = `${Math.floor(pct)}%`;
    }
    return;
  }
  downloads.delete(p.id);
  if (p.error) toast(p.error);
  else
    toast(
      p.id === "base"
        ? "Componentes de IA instalados"
        : p.id === "speaker"
          ? "Biometria neural instalada"
          : "Voz baixada",
    );
  refreshLibrary().then(() => {
    if (!p.error && p.id !== "base" && p.id !== "speaker" && !settings.ai.voice) {
      settings.ai.voice = p.id;
      save();
    }
    if (settings.page === "ai") renderAi();
  });
}

function sameCharacter(c: Character) {
  return CHARACTER_KEYS.every((k) => Math.abs(settings.params[k] - c[k]) < 1e-6);
}

function renderEffects() {
  const p = settings.params;
  mount(`
    ${head("Efeitos", "Ambientes e extras que se somam à voz.")}
    <div class="tiles">
      ${EFFECTS.map((e) => tile(e.key, e.key, e.name, p[e.key])).join("")}
    </div>
    <div class="cols">
    <div>
    <h2>Vocoder</h2>
    <div class="group">
      ${switchRow("vocoder-on", "Vocoder", "Sua fala tocada por um instrumento", p.vocoder !== null, {
        icon: "vocoder",
        tone: "tone-purple",
        help: "O vocoder usa o ritmo e as palavras da sua voz para “tocar” um instrumento, como nas vozes de robô da música eletrônica.",
      })}
      <div class="row ${p.vocoder ? "" : "disabled"}"><div class="chips">
        ${CARRIERS.map((c) => `<button class="chip ${p.vocoder === c.id ? "active" : ""}" data-carrier="${c.id}">${c.name}</button>`).join("")}
      </div></div>
      <div class="${p.vocoder ? "" : "disabled"}">
        ${slider({
          id: "vocoder-mix",
          label: "Intensidade",
          min: 0,
          max: 100,
          step: 1,
          def: 100,
          stepper: true,
          format: percent,
          get: () => Math.round(p.vocoderMix * 100),
          set: (v) => (p.vocoderMix = v / 100),
        })}
      </div>
    </div>
    </div>
    <div>
    <h2>Mistura</h2>
    <div class="group">
      ${switchRow("append", "Misturar voz original", "Sua voz natural toca junto com a alterada", p.appendOriginal, { icon: "voices", tone: "tone-teal" })}
    </div>
    </div>
    </div>`);

  $$<HTMLButtonElement>("[data-tile]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const key = b.dataset.tile as EffectKey;
      p[key] = !p[key];
      if (p[key]) popTile = key;
      pushParams();
      renderEffects();
    }),
  );
  $$<HTMLButtonElement>("[data-carrier]", content).forEach((b) =>
    b.addEventListener("click", () => {
      p.vocoder = b.dataset.carrier as Carrier;
      pushParams();
      renderEffects();
    }),
  );
  bindSwitch("vocoder-on", (on) => {
    p.vocoder = on ? (p.vocoder ?? "saw") : null;
    pushParams();
    renderEffects();
  });
  bindSwitch("append", (on) => {
    p.appendOriginal = on;
    pushParams();
  });
}

function micSoundControls() {
  const p = settings.params;
  const denoiseOk = status?.denoiseSupported ?? true;
  const beta = settings.beta;
  const voiceGateBlock = beta
    ? `
    <h2>Só a minha voz ${betaTag()}</h2>
    <p class="section-hint">A sua voz sai na hora (~30 ms). O CAM++ só fecha se outra pessoa falar de verdade. Regrave nesta versão — a biometria guarda até 5 impressões (normal, grossa, fina, risada).</p>
    <div class="group">
      ${switchRow("voice-gate", "Usar só a minha voz", p.voicePitch > 0
        ? (p.voiceEmbed.length === 512
          ? `Neural · cos ${p.voiceNeuralLo > 0 ? p.voiceNeuralLo.toFixed(2) : "—"}–${p.voiceNeuralHi > 0 ? p.voiceNeuralHi.toFixed(2) : "—"} · ${p.voiceProtect ? "Protegido" : "Rápido"}`
          : p.voicePitchLo > 0 && p.voicePitchHi > p.voicePitchLo
            ? `Faixa gravada · ${Math.round(p.voicePitchLo)}–${Math.round(p.voicePitchHi)} Hz`
            : `Voz gravada · cerca de ${Math.round(p.voicePitch)} Hz`)
        : "Ainda não há uma gravação", p.voiceGate, {
        icon: "voices",
        tone: "tone-teal",
        beta: true,
        help: "Abre na hora quando você fala no microfone. Palavras curtas, risadas e mudanças de tom não são cortadas. Outra voz (vídeo, TV) é vetada pelo CAM++ depois de ~0,5 s. Regrave a voz nesta versão.",
      })}
      <div class="row">
        <div class="label"><span class="title">Gravação</span><small>15 s: fale normal, depois grosso/fino e uma risada · regrave nesta versão</small></div>
        <button type="button" class="btn" id="voice-print">${icon("mic")}Gravar minha voz</button>
      </div>
      <div class="${p.voicePitch > 0 && p.voiceEmbed.length === 512 ? "" : "disabled"}">
        ${switchRow(
          "voice-protect",
          "Modo Protegido",
          p.voiceProtect ? "Veto CAM++ ativo · delay fixo 30 ms" : "Mesmo delay 30 ms · veto igual",
          p.voiceProtect,
          {
            icon: "mic",
            tone: "tone-teal",
            beta: true,
            help: "Os dois modos agora usam delay fixo de 30 ms e abrem na hora. O CAM++ só veta outra voz após ~0,5 s e duas leituras baixas.",
          },
        )}
      </div>
      <div class="${p.voicePitch > 0 ? "" : "disabled"}">
        ${slider({
          id: "voice-strict",
          label: "Exigência da biometria",
          hint: "0% folga para a sua voz · 100% limiar ~0,50 contra vídeo/TV",
          help: "0% aceita a sua voz com folga. 100% sobe o cosseno até ~0,50 para o veto fechar YouTube ou outra pessoa. Regrave se mudar de microfone ou de sala.",
          min: 0,
          max: 100,
          step: 1,
          def: 35,
          stepper: true,
          format: percent,
          get: () => Math.round(p.voiceStrict * 100),
          set: (v) => (p.voiceStrict = v / 100),
        })}
      </div>
      <details class="gate-diag" id="gate-diag" ${p.voiceGate ? "open" : ""}>
        <summary>Diagnóstico do Speaker Gate</summary>
        <p class="section-hint">Ao vivo · sem gravar áudio nem embedding. Use isto para ver por que o mic abre ou fica mudo.</p>
        <div class="gate-diag-grid" id="gate-diag-live">
          <div><span>Estado</span><strong id="gd-phase">—</strong></div>
          <div><span>Motivo</span><strong id="gd-reason">—</strong></div>
          <div><span>Cosseno</span><strong id="gd-neural">—</strong></div>
          <div><span>Limiar</span><strong id="gd-thresh">—</strong></div>
          <div><span>Exigência</span><strong id="gd-strict">—</strong></div>
          <div><span>Modo</span><strong id="gd-mode">—</strong></div>
          <div><span>Fala (ms)</span><strong id="gd-speech">—</strong></div>
          <div><span>Buffer usado</span><strong id="gd-buf">—</strong></div>
          <div><span>Buffer cap</span><strong id="gd-bufcap">—</strong></div>
          <div><span>Delay linha</span><strong id="gd-delay">—</strong></div>
          <div><span>ONNX (última)</span><strong id="gd-infer">—</strong></div>
          <div><span>Espera verify</span><strong id="gd-vage">—</strong></div>
          <div><span>Utterance</span><strong id="gd-utt">—</strong></div>
          <div><span>Autorizado</span><strong id="gd-auth">—</strong></div>
          <div><span>VAD env</span><strong id="gd-env">—</strong></div>
          <div><span>Build ID</span><strong id="gd-build">—</strong></div>
        </div>
        <pre class="gate-diag-log" id="gd-log" aria-label="Últimos eventos"></pre>
      </details>
    </div>`
    : "";
  return `
    ${voiceGateBlock}
    <div class="section-head">
      <h2>Ganho e ruído</h2>
      <button class="link" id="mic-reset">${icon("reset")}Restaurar</button>
    </div>
    <div class="group">
      ${switchRow(
        "denoise",
        "Cancelamento de ruído por IA",
        denoiseOk ? "Tira o que não é voz: ventilador, chiado e sala" : "Disponível só com microfone em 48 kHz",
        p.denoise,
        {
          disabled: !denoiseOk,
          icon: "whisper",
          tone: "tone-teal",
          help: "O modelo sempre limpa no máximo. O percentual mistura esse áudio com o original: 80% = 80% limpo + 20% cru. Não é o portão da Limitação de ruído.",
        },
      )}
      <div class="${p.denoise && denoiseOk ? "" : "disabled"}">
        ${slider({
          id: "denoise-amount",
          label: "Intensidade",
          hint: "Mistura com o original · 80% ainda deixa 20% cru",
          min: 0,
          max: 100,
          step: 1,
          def: 75,
          stepper: true,
          format: percent,
          get: () => Math.round(p.denoiseAmount * 100),
          set: (v) => (p.denoiseAmount = v / 100),
        })}
      </div>
    </div>
    <h2>Redução de ruído</h2>
    <p class="section-hint">${p.denoise ? "Pausada enquanto o cancelamento por IA estiver ligado." : "Para ruído constante e cliques, sem o modelo de IA."}</p>
    <div class="group ${p.denoise ? "disabled" : ""}">
      ${slider({
        id: "noise-floor",
        label: "Som de fundo",
        hint: "Ventilador e chiado parado",
        help: "Abaixa o som que fica no mesmo volume o tempo todo, como a ventoinha do PC ou um chiado fixo. A fala, que varia, passa.",
        min: 0,
        max: 100,
        step: 1,
        def: 0,
        stepper: true,
        format: percent,
        get: () => Math.round(p.noiseFloor * 100),
        set: (v) => (p.noiseFloor = v / 100),
      })}
      ${slider({
        id: "impact",
        label: "Impacto",
        hint: "Teclado e cliques",
        help: "Segura estalos bem curtos e agudos, como tecla e clique. Uma sílaba da fala dura mais e não entra nesse corte.",
        min: 0,
        max: 100,
        step: 1,
        def: 0,
        stepper: true,
        format: percent,
        get: () => Math.round(p.impact * 100),
        set: (v) => (p.impact = v / 100),
      })}
    </div>
    <h2>Dinâmica</h2>
    <div class="group">
      ${slider({
        id: "gate",
        label: "Limitação de ruído",
        hint: "Corta o que ficar abaixo do limite",
        help: "Abaixo desse volume o microfone fecha de vez. À esquerda está desligado. Suba só até o ruído sumir e a fala continuar.",
        min: -90,
        max: -20,
        step: 1,
        def: -90,
        stepper: true,
        format: (v) => (v <= -90 ? "Desligado" : `${v} dB`),
        get: () => p.sensitivityDb,
        set: (v) => (p.sensitivityDb = v),
      })}
      ${slider({
        id: "compress",
        label: "Compressor",
        hint: "Deixa a voz num volume estável",
        help: "Em 100% a razão fica perto de 7:1, ataque de 4 ms e volta de 90 ms. Zero deixa o volume do microfone. Não é um segundo limitador.",
        min: 0,
        max: 100,
        step: 1,
        def: 0,
        stepper: true,
        format: percent,
        get: () => Math.round(p.compress * 100),
        set: (v) => (p.compress = v / 100),
      })}
      ${
        beta
          ? switchRow("smooth", "Voz mais suave", p.smooth ? "Grão longo · menos emenda no tom" : "Grão curto", p.smooth, {
              icon: "sparkle",
              tone: "tone-purple",
              beta: true,
              help: "Só muda o grão do tom clássico (2560 amostras, sobreposição 1024). Não equaliza. Corpo e agudos ficam em Grave, Agudo e no equalizador. A troca entra no próximo bloco de áudio, com crossfade.",
            })
          : ""
      }
      ${slider({
        id: "bass-shelf",
        label: "Grave",
        hint: "0 dB neutro · prateleira em 180 Hz",
        help: "Sobe ou desce o corpo da voz, separado do equalizador gráfico. 0 dB não altera nada.",
        min: -30,
        max: 12,
        step: 0.5,
        def: 0,
        centered: true,
        format: (v) => signed(v, 1, "dB"),
        get: () => p.bassDb,
        set: (v) => (p.bassDb = v),
      })}
      ${slider({
        id: "treble-shelf",
        label: "Agudo",
        hint: "0 dB neutro · prateleira em 4 kHz · até −30 dB",
        help: "Tira ou põe brilho acima de 4 kHz. O gráfico do equalizador só chega a −12 dB; este controle desce até −30 dB. 0 dB é o padrão.",
        min: -30,
        max: 12,
        step: 0.5,
        def: 0,
        centered: true,
        format: (v) => signed(v, 1, "dB"),
        get: () => p.trebleDb,
        set: (v) => (p.trebleDb = v),
      })}
      ${slider({
        id: "boost",
        label: "Ganho",
        hint: "100% = volume do microfone · 250% ≈ +8 dB",
        min: 0,
        max: 400,
        step: 1,
        def: 100,
        stepper: true,
        format: percent,
        get: () => p.boost,
        set: (v) => (p.boost = v),
      })}
    </div>`;
}

function bindMicSoundControls() {
  const p = settings.params;
  bindSwitch("denoise", (on) => {
    p.denoise = on;
    pushParams();
    renderMic();
  });
  bindSwitch("voice-gate", (on) => {
    if (on && p.voicePitch <= 0) {
      p.voiceGate = false;
      openVoicePrint();
      renderMic();
      return;
    }
    p.voiceGate = on;
    pushParams();
  });
  bindSwitch("voice-protect", (on) => {
    p.voiceProtect = on;
    pushParams();
    renderMic();
  });
  $("#voice-print")?.addEventListener("click", openVoicePrint);
  bindSwitch("smooth", (on) => {
    p.smooth = on;
    pushParams();
    syncPower();
  });
  $("#mic-reset")?.addEventListener("click", () => {
    const d = defaultParams();
    Object.assign(p, {
      boost: d.boost,
      sensitivityDb: d.sensitivityDb,
      bassDb: d.bassDb,
      trebleDb: d.trebleDb,
      eq: d.eq.slice(),
      denoise: d.denoise,
      denoiseAmount: d.denoiseAmount,
      noiseFloor: d.noiseFloor,
      impact: d.impact,
      compress: d.compress,
    });
    pushParams();
    renderMic();
  });
}

function describe(p: Params): string {
  const names = new Map(VOICES.map((v) => [v.id, v.name] as const));
  const parts = p.voices.map((v) => names.get(v) ?? v);
  const ch = CHARACTERS.find((c) => CHARACTER_KEYS.every((k) => Math.abs(p[k] - c.c[k]) < 1e-6));
  if (ch) parts.push(ch.name);
  else if (CHARACTER_KEYS.some((k) => p[k] !== 0)) parts.push("Personalizada");
  if (p.vocoder) parts.push(`Vocoder ${CARRIERS.find((c) => c.id === p.vocoder)?.name ?? ""}`);
  for (const e of EFFECTS) if (p[e.key]) parts.push(e.name);
  return parts.length ? parts.join(" · ") : "Voz natural";
}

function renderProfiles() {
  mount(`
    ${head(
      "Perfis",
      "Salve combinações de voz, efeitos e microfone para trocar com um clique.",
      `<button class="btn" id="profile-new">${icon("save")}Salvar atual</button>`,
    )}
    <div class="group">
      ${
        settings.profiles.length === 0
          ? `<div class="empty">Nenhum perfil salvo ainda.</div>`
          : settings.profiles
              .map(
                (pr, i) => `
          <div class="row">
            <div class="label"><span class="title">${esc(pr.name)}</span><small>${esc(describe(pr.params))}</small></div>
            ${bindKey(profileShortcut(pr.name))}
            <button class="btn plain" data-apply="${i}">Aplicar</button>
            <button class="btn danger" data-delete="${i}" aria-label="Excluir perfil" title="Excluir perfil">${icon("trash")}</button>
          </div>`,
              )
              .join("")
      }
    </div>`);

  bindKeyButtons();
  $("#profile-new").addEventListener("click", openProfileSheet);
  $$<HTMLButtonElement>("[data-apply]", content).forEach((b) =>
    b.addEventListener("click", () => applyProfile(Number(b.dataset.apply))),
  );
  $$<HTMLButtonElement>("[data-delete]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const i = Number(b.dataset.delete);
      const pr = settings.profiles[i];
      confirmSheet(`Excluir o perfil “${pr.name}”?`, "A voz atual não muda; só o atalho salvo é apagado.", "Excluir perfil", () => {
        settings.profiles.splice(settings.profiles.indexOf(pr), 1);
        delete settings.shortcuts[profileShortcut(pr.name)];
        save();
        renderProfiles();
        toast(`Perfil “${pr.name}” excluído`);
      });
    }),
  );
}

function profileShortcut(name: string) {
  return `profile:${encodeURIComponent(name)}`;
}

function profileKey(p: Params) {
  const copy = cleanParams(structuredClone(p));
  copy.enabled = true;
  return JSON.stringify(copy);
}

function currentProfileName() {
  const key = profileKey(settings.params);
  return settings.profiles.find((p) => profileKey(p.params) === key)?.name ?? null;
}

function classicVoiceOn() {
  const p = settings.params;
  return p.voices.length > 0 || CHARACTER_KEYS.some((k) => Math.abs(p[k]) > 1e-6);
}

function applyProfile(i: number) {
  const pr = settings.profiles[i];
  if (!pr) return;
  settings.params = cleanParams(structuredClone(pr.params));
  settings.mixer = settings.params.voices.length > 1;
  syncPower();
  pushParams();
  rerender();
  toast(`Perfil “${pr.name}” aplicado`);
}

function openProfileSheet() {
  openSheet(`
    <h3>Novo perfil</h3>
    <p>${esc(describe(settings.params))}</p>
    <input type="text" id="profile-name" placeholder="Nome do perfil" maxlength="40" />
    <button class="btn block" id="profile-save">Salvar</button>`);
  const input = $<HTMLInputElement>("#profile-name");
  setTimeout(() => input.focus(), 250);
  const submit = () => {
    const name = input.value.trim();
    if (!name) return input.focus();
    const entry = { name, params: structuredClone(settings.params) };
    const existing = settings.profiles.findIndex((p) => p.name === name);
    if (existing >= 0) settings.profiles[existing] = entry;
    else settings.profiles.push(entry);
    save();
    closeSheet();
    renderProfiles();
    toast(`Perfil “${name}” salvo`);
  };
  $("#profile-save").addEventListener("click", submit);
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
}

function picker(id: string, list: string[], selected: string | null, fallback: string | null, disabled: boolean) {
  const current = selected && list.includes(selected) ? selected : "";
  const label = current || `Padrão do Windows${fallback ? ` (${fallback})` : ""}`;
  const option = (value: string, text: string) =>
    `<button type="button" class="picker-opt ${value === current ? "active" : ""}" data-value="${esc(value)}" role="option" aria-selected="${value === current}">${esc(text)}</button>`;
  return `
    <div class="picker ${disabled ? "disabled" : ""}" data-picker="${id}">
      <button type="button" class="picker-btn" ${disabled ? "disabled" : ""} aria-haspopup="listbox" aria-expanded="false">${esc(label)}</button>
      <div class="picker-menu" role="listbox" hidden>
        ${option("", `Padrão do Windows${fallback ? ` (${fallback})` : ""}`)}
        ${list.map((d) => option(d, d)).join("")}
      </div>
    </div>`;
}

function closePickers() {
  $$(".picker-menu").forEach((menu) => {
    menu.hidden = true;
    menu.parentElement?.querySelector(".picker-btn")?.setAttribute("aria-expanded", "false");
  });
}

let pickerDismissBound = false;

function bindPickers(onPick: (id: string, value: string) => void) {
  if (!pickerDismissBound) {
    pickerDismissBound = true;
    document.addEventListener("click", () => closePickers());
  }
  $$<HTMLElement>(".picker", content).forEach((box) => {
    const btn = $<HTMLButtonElement>(".picker-btn", box);
    const menu = $<HTMLElement>(".picker-menu", box);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const willOpen = menu.hidden;
      closePickers();
      if (!willOpen || btn.disabled) return;
      menu.hidden = false;
      btn.setAttribute("aria-expanded", "true");
      const room = window.innerHeight - btn.getBoundingClientRect().bottom;
      menu.classList.toggle("up", room < 120);
    });
    $$<HTMLButtonElement>(".picker-opt", menu).forEach((opt) =>
      opt.addEventListener("click", (e) => {
        e.stopPropagation();
        closePickers();
        onPick(box.dataset.picker!, opt.dataset.value ?? "");
      }),
    );
  });
}

const CAL_SECONDS = 8;
const PRINT_SECONDS = 15;
let calStop = 0;
let calListening = false;
let calPurpose: "pitch" | "print" = "pitch";
let calPending: { kind: "ai" | "classic"; id?: string } | null = null;

function prepareCalibration() {
  stopCalibration();
  const err = $("#cal-error");
  err.hidden = true;
  $("#cal-read").hidden = false;
  $("#cal-result").hidden = true;
  $("#cal-time").textContent = "00:00";
  const start = $<HTMLButtonElement>("#cal-start");
  start.disabled = false;
  start.textContent = "Iniciar";
  $$(".cal-slider").forEach((el) => {
    el.hidden = calPurpose === "print";
  });
  const meter = $("#cal-meter");
  if (!meter.childElementCount) {
    for (let i = 0; i < 28; i++) meter.appendChild(document.createElement("i"));
  }
  $("#cal-backdrop").hidden = false;
}

function openCalibration() {
  calPurpose = "pitch";
  $("#cal-title").textContent = "Calibração automática";
  $(".cal-kicker").textContent = "Leia em voz natural";
  $(".cal-script").textContent = "Olá, eu estou calibrando o KamoVox. Vou falar do meu jeito, sem sussurrar e sem gritar, para o programa encontrar o meu tom.";
  prepareCalibration();
}

async function ensureSpeakerModel(): Promise<boolean> {
  try {
    library = await invoke<Library>("ai_library");
  } catch {
    /* keep cached */
  }
  if (library?.speakerInstalled) return true;
  const id = "speaker";
  if (!downloading(id)) {
    downloads.set(id, { id, received: 0, total: library?.speakerSize ?? 0, done: false, error: null });
    try {
      await invoke("ai_download", { id });
    } catch (e) {
      downloads.delete(id);
      throw e;
    }
  }
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 400));
    const job = downloads.get(id);
    if (job?.done && !job.error) {
      downloads.delete(id);
      library = await invoke<Library>("ai_library");
      return !!library?.speakerInstalled;
    }
    if (job?.error) {
      const err = job.error;
      downloads.delete(id);
      throw new Error(err);
    }
    if (!job && library?.speakerInstalled) return true;
    try {
      library = await invoke<Library>("ai_library");
      if (library.speakerInstalled) {
        downloads.delete(id);
        return true;
      }
    } catch {
      /* retry */
    }
  }
  throw new Error("Tempo esgotado ao baixar a biometria neural (~28 MB).");
}

function openVoicePrint() {
  calPurpose = "print";
  $("#cal-title").textContent = "Gravar a minha voz";
  $(".cal-kicker").textContent = "15 segundos: voz normal, depois grossa ou fina, e uma risada";
  $(".cal-script").textContent =
    "Vou falar do meu dia com a voz normal, depois repetir uma frase mais grossa e outra mais fina, e no fim dar uma risada curta, para o programa gravar vários jeitos da minha voz. Quem estiver por perto fica em silêncio o tempo todo.";
  prepareCalibration();
  const err = $("#cal-error");
  const start = $<HTMLButtonElement>("#cal-start");
  void (async () => {
    try {
      if (!library?.speakerInstalled) {
        start.disabled = true;
        err.hidden = false;
        err.textContent = "Baixando biometria neural (CAM++ · ~28 MB)…";
        await ensureSpeakerModel();
        err.hidden = true;
        start.disabled = false;
      }
    } catch (e) {
      start.disabled = false;
      err.hidden = false;
      err.textContent = String(e);
    }
  })();
}

function closeCalibration() {
  stopCalibration();
  $("#cal-backdrop").hidden = true;
  if (settings.page === "ai" || settings.page === "voices") rerender();
}

function stopCalibration() {
  window.clearInterval(calStop);
  if (!calListening) return;
  calListening = false;
  void invoke("calibration_end").catch(() => {});
}

async function finishVoicePrint() {
  const button = $<HTMLButtonElement>("#cal-start");
  const err = $("#cal-error");
  button.disabled = true;
  button.textContent = "Processando…";
  try {
    const print = await invoke<{
      bands: number[];
      pitch: number;
      pitchLo: number;
      pitchHi: number;
      level: number;
      f1: number;
      f2: number;
      f3: number;
      embed: number[];
      gallery: number[];
      neural: boolean;
      neuralLo: number;
      neuralHi: number;
      neuralReject: number;
      selfP05: number;
      selfP10: number;
      selfMedian: number;
      selfLive: number;
    }>("voice_print_end");
    settings.params.voiceBands = print.bands;
    settings.params.voicePitch = print.pitch;
    settings.params.voicePitchLo = print.pitchLo;
    settings.params.voicePitchHi = print.pitchHi;
    settings.params.voiceLevel = print.level;
    settings.params.voiceF1 = print.f1;
    settings.params.voiceF2 = print.f2;
    settings.params.voiceF3 = print.f3;
    settings.params.voiceEmbed = print.neural && print.embed?.length === 512 ? print.embed : [];
    settings.params.voiceGallery =
      print.neural && Array.isArray(print.gallery) && print.gallery.length >= 512 && print.gallery.length % 512 === 0
        ? print.gallery.slice(0, 12 * 512)
        : [];
    settings.params.voiceNeuralLo = print.neural ? print.neuralLo : 0;
    settings.params.voiceNeuralHi = print.neural ? print.neuralHi : 0;
    settings.params.voiceNeuralReject = print.neural ? print.neuralReject : 0;
    settings.params.voiceGate = true;
    save();
    await invoke("set_params", { params: settings.params });
    const galleryN = Math.floor((settings.params.voiceGallery.length || 0) / 512);
    const protoN = Math.max(galleryN, print.embed?.length === 512 ? 1 : 0);
    const liveNote =
      Number.isFinite(print.selfLive) && print.selfLive > 0
        ? ` · replay ao vivo ${print.selfLive.toFixed(2)}`
        : "";
    const neuralNote = print.neural
      ? `Biometria calibrada · ${protoN} impressões${liveNote} · p05 ${print.selfP05.toFixed(2)} p10 ${print.selfP10.toFixed(2)} · aceita ${print.neuralLo.toFixed(2)}–${print.neuralHi.toFixed(2)} · rejeita <${print.neuralReject.toFixed(2)}.`
      : "Modelo neural ausente — usando timbre/formantes.";
    $("#cal-heard").textContent = `Impressão vocal gravada · ${Math.round(print.pitchLo)}–${Math.round(print.pitchHi)} Hz · formantes ${Math.round(print.f1)}/${Math.round(print.f2)}/${Math.round(print.f3)} Hz. ${neuralNote}`;
    $("#cal-read").hidden = true;
    $("#cal-result").hidden = false;
    $$(".cal-slider").forEach((el) => {
      el.hidden = true;
    });
    if (print.neural && status?.running) {
      await restartEngine();
    }
    if (settings.page === "mic") renderMic();
  } catch (e) {
    button.disabled = false;
    button.textContent = "Iniciar";
    err.hidden = false;
    err.textContent = String(e);
  }
}

function paintCentered(input: HTMLInputElement) {
  const v = Number(input.value);
  const min = Number(input.min);
  const max = Number(input.max);
  if (!Number.isFinite(v) || !(max > min)) return;
  const pct = ((v - min) / (max - min)) * 100;
  const zero = ((0 - min) / (max - min)) * 100;
  const width = Math.abs(pct - zero);
  const pos = width >= 100 ? 0 : (Math.min(pct, zero) / (100 - width)) * 100;
  input.style.setProperty("--from", `${pos}%`);
  input.style.setProperty("--span", `${width}%`);
}

function paintCalValues() {
  const tom = $<HTMLInputElement>("#cal-tom");
  const ent = $<HTMLInputElement>("#cal-ent");
  paintCentered(tom);
  paintCentered(ent);
  $("#cal-tom-val").textContent = signed(Number(tom.value), 1, "st");
  $("#cal-ent-val").textContent = signed(Number(ent.value), 1);
}

/** Speaking pitch the classic voices were balanced against. */
const CAL_REF_HZ = 150;
/** Extra semitones a feminine classic voice keeps when the speaker is exactly on that pitch. */
const CAL_FEMININE_AT_REF = 2;

function calibrationBase(kind: "ai" | "classic", id?: string) {
  if (kind === "ai") return library?.voices.find((v) => v.id === id)?.pitch ?? 0;
  return settings.params.voices.includes("female") ? CAL_FEMININE_AT_REF : 0;
}

function suggestionFromVoice(hz: number, base: number) {
  const fromRef = 12 * Math.log2(hz / CAL_REF_HZ);
  const tom = Math.min(16, Math.max(-16, Math.round((base - fromRef) * 10) / 10));
  const ent = Math.min(6, Math.max(-6, Math.round(-tom * 0.35 * 10) / 10));
  return { tom, ent };
}

function applyCalibration(tom: number, ent: number, voiceId?: string) {
  const voice = voiceId ?? settings.ai.voice;
  if (voice) {
    settings.ai.transpose[voice] = tom;
    invoke("set_ai_pitch", { semitones: tom }).catch(() => {});
  } else {
    settings.params.pitch = tom;
  }
  settings.params.formant = ent;
  save();
  pushParams();
}

function useSavedCalibration(kind: "ai" | "classic", id?: string) {
  const hz = settings.calibration?.hz;
  if (!hz) {
    calPending = { kind, id };
    if (settings.page !== "mic") navigate("mic");
    openCalibration();
    return;
  }
  const { tom, ent } = suggestionFromVoice(hz, calibrationBase(kind, id));
  if (kind === "ai" && id) {
    const same = settings.ai.voice === id && settings.ai.enabled;
    settings.ai.voice = id;
    settings.ai.enabled = !!library?.baseInstalled;
    settings.ai.transpose[id] = tom;
    save();
    if (same) invoke("set_ai_pitch", { semitones: tom }).catch(() => {});
    else void restartEngine();
  } else {
    settings.params.pitch = tom;
  }
  settings.params.formant = ent;
  save();
  pushParams();
  toast(`Calibrado · tom ${signed(tom, 1, "st")} · entonação ${signed(ent, 1)}`);
  if (settings.page === "ai" || settings.page === "voices") rerender();
}

function medianPitch(samples: Float32Array, sr: number): number | null {
  const size = 2048;
  const minLag = Math.floor(sr / 400);
  const maxLag = Math.floor(sr / 70);
  const found: number[] = [];
  for (let start = 0; start + size < samples.length; start += 1024) {
    let energy = 0;
    for (let i = 0; i < size; i++) energy += samples[start + i]! * samples[start + i]!;
    if (energy / size < 0.00008) continue;
    const diff = new Float32Array(maxLag + 1);
    for (let lag = 1; lag <= maxLag; lag++) {
      let sum = 0;
      const step = 2;
      for (let i = 0; i < size - lag; i += step) {
        const delta = samples[start + i]! - samples[start + i + lag]!;
        sum += delta * delta;
      }
      diff[lag] = sum;
    }
    let running = 0;
    let lag = -1;
    for (let i = 1; i <= maxLag; i++) {
      running += diff[i]!;
      const norm = running > 0 ? (diff[i]! * i) / running : 1;
      diff[i] = norm;
      if (lag < 0 && i >= minLag && norm < 0.15) lag = i;
    }
    if (lag > 0) {
      while (lag + 1 <= maxLag && diff[lag + 1]! < diff[lag]!) lag += 1;
      found.push(sr / lag);
    }
  }
  if (found.length < 3) return null;
  found.sort((a, b) => a - b);
  return found[Math.floor(found.length / 2)]!;
}

async function startCalibration() {
  const err = $("#cal-error");
  err.hidden = true;
  const button = $<HTMLButtonElement>("#cal-start");
  button.disabled = true;
  button.textContent = "Ouvindo…";
  try {
    await invoke("calibration_begin");
  } catch (e) {
    button.disabled = false;
    button.textContent = "Iniciar";
    err.hidden = false;
    err.textContent = String(e);
    return;
  }
  calListening = true;
  const bars = [...$("#cal-meter").children] as HTMLElement[];
  const started = performance.now();
  const limit = (calPurpose === "print" ? PRINT_SECONDS : CAL_SECONDS) * 1000;
  calStop = window.setInterval(() => {
    const seconds = Math.min(limit / 1000, (performance.now() - started) / 1000);
    $("#cal-time").textContent = `00:${String(Math.floor(seconds)).padStart(2, "0")}`;
    void invoke<Levels>("get_levels")
      .then((levels) => {
        const level = Math.min(1, levels.input * 6);
        bars.forEach((bar, index) => {
          const wobble = 0.65 + 0.35 * Math.sin(index * 0.7 + seconds * 8);
          bar.style.transform = `scaleY(${0.18 + level * wobble})`;
        });
      })
      .catch(() => {});
    if (performance.now() - started >= limit) {
      window.clearInterval(calStop);
      void pullCalibration();
    }
  }, 80);
}

async function pullCalibration() {
  calListening = false;
  if (calPurpose === "print") {
    await finishVoicePrint();
    return;
  }
  let captured: { samples: number[]; rate: number };
  try {
    captured = await invoke<{ samples: number[]; rate: number }>("calibration_end");
  } catch (e) {
    const button = $<HTMLButtonElement>("#cal-start");
    button.disabled = false;
    button.textContent = "Iniciar";
    const err = $("#cal-error");
    err.hidden = false;
    err.textContent = String(e);
    return;
  }
  const hz = captured.samples.length < 1000 ? null : medianPitch(Float32Array.from(captured.samples), captured.rate);
  if (hz == null) {
    const button = $<HTMLButtonElement>("#cal-start");
    button.disabled = false;
    button.textContent = "Iniciar";
    const err = $("#cal-error");
    err.hidden = false;
    err.textContent = captured.samples.length < 1000
      ? "O trocador de voz precisa estar ligado. Ele usa o microfone que você já escolheu."
      : "Não ouvi voz suficiente. Fale mais perto do microfone e grave de novo.";
    return;
  }
  settings.calibration = { hz };
  save();
  const pending = calPending;
  calPending = null;
  const voiceId = pending?.kind === "ai" ? pending.id : settings.ai.voice ?? undefined;
  const kind = pending?.kind ?? (voiceId ? "ai" : "classic");
  const { tom, ent } = suggestionFromVoice(hz, calibrationBase(kind, voiceId));
  if (pending) useSavedCalibration(pending.kind, pending.id);
  else applyCalibration(tom, ent, voiceId);
  $("#cal-heard").textContent = `Sua voz ficou em torno de ${Math.round(hz)} Hz. Essa medida fica salva para calibrar outras vozes.`;
  $("#cal-tom-hint").textContent = `Sugerido: ${signed(tom, 1, "st")}`;
  $("#cal-ent-hint").textContent = `Sugerido: ${signed(ent, 1)}`;
  $<HTMLInputElement>("#cal-tom").value = String(tom);
  $<HTMLInputElement>("#cal-ent").value = String(ent);
  paintCalValues();
  $("#cal-read").hidden = true;
  $("#cal-result").hidden = false;
}

function renderMic() {
  const integrated = installedEndpoint();
  const usingWindowsMic = !!integrated && settings.windowsMic;
  const endpointRow = (ep: Endpoint) => {
    const detail = ep.installed
      ? "KamoVox instalado"
      : ep.other
        ? `Já tem outro efeito de áudio (${ep.other}); ele volta se você remover o KamoVox`
        : "Disponível";
    const action = ep.installed
      ? `<button class="btn plain" data-mic-remove="${esc(ep.id)}" ${micBusy ? "disabled" : ""}>Remover</button>`
      : `<button class="btn" data-mic-install="${esc(ep.id)}" ${micBusy || integrated ? "disabled" : ""}>Instalar</button>`;
    return `<div class="row">
      <span class="glyph-sm ${ep.installed ? "tone-green" : ""}">${icon("mic")}</span>
      <div class="label"><span class="title">${esc(ep.name)}</span><small>${esc(detail)}</small></div>
      ${action}
    </div>`;
  };

  const renderMicAudio = () => `
    <h2>Entrada</h2>
    <div class="group">
      <div class="row ${usingWindowsMic ? "disabled" : ""}">
        <div class="glyph-sm">${icon("mic")}</div>
        <div class="label"><span class="title">Microfone</span><small>${
          usingWindowsMic ? `Usando ${esc(integrated!.name)}, configurado abaixo` : "Onde você fala"
        }</small></div>
        <div class="picker-slot">${picker("input", devices.inputs, settings.input, devices.defaultInput, usingWindowsMic)}</div>
      </div>
      <div class="row cal-row">
        <button type="button" class="cal-open" id="cal-open">${icon("sparkle")}Calibração automática</button>
      </div>
    </div>
    <h2>Ouvir a própria voz</h2>
    <div class="group">
      ${switchRow("monitor", "Ouvir no fone", "Use fone para não dar eco", settings.monitor, { icon: "headphones", tone: "tone-purple" })}
      <div class="row">
        <div class="label"><span class="title">Saída</span><small>Fone ou cabo virtual</small></div>
        <div class="picker-slot">${picker("output", devices.outputs, settings.output, devices.defaultOutput, false)}</div>
      </div>
    </div>
    <h2 id="programs">Usar em outros programas</h2>
    ${
      integrated
        ? `<p class="section-hint">Instalado em ${esc(integrated.name)}. Chamadas e jogos ouvem a voz alterada.</p>`
        : `<div class="teach">
            <span class="teach-n">1</span>
            <div>
              <b>Instale no microfone do Discord</b>
              <small>Sem esta instalação, o Discord continua ouvindo a sua voz normal. Toque em <b>Instalar</b> no microfone da chamada. O Windows pede permissão e o som reinicia por alguns segundos.</small>
            </div>
          </div>`
    }
    <div class="group ${integrated ? "" : "attention"}">
      ${
        integrated
          ? switchRow(
              "windows-mic",
              "Voz alterada nos programas",
              status?.apo ? `Ativa em ${integrated.name}` : integrated.name,
              settings.windowsMic,
              { icon: "link", tone: "tone-green" },
            )
          : ""
      }
      ${
        endpoints.length
          ? (integrated ? endpoints.filter((e) => e.installed) : endpoints).map(endpointRow).join("")
          : `<div class="empty">Nenhum microfone do Windows apareceu. Conecte o microfone, fale nele e toque em Atualizar.</div>`
      }
    </div>
    ${
      integrated && endpoints.some((e) => !e.installed)
        ? disclosure(
            "other-mics",
            "Outros microfones",
            `${endpoints.filter((e) => !e.installed).length} disponíveis`,
            `<div class="group">${endpoints.filter((e) => !e.installed).map(endpointRow).join("")}</div>
             <div class="note">${icon("sparkle")}<div>Trocar o microfone pede administrador e reinicia o som do Windows por alguns segundos.</div></div>`,
            "mic",
          )
        : ""
    }
    ${renderMicSound()}`;

  const renderMicTalk = () => `
    <p class="section-hint">Segure a tecla para falar. Ao soltar, o Discord fica em silêncio.</p>
    <div class="group">
      ${switchRow("ptt-on", "Usar push-to-talk", pttOpen && settings.ptt.enabled ? "Aberto — pode falar" : settings.ptt.enabled ? "Mutado" : "Desligado", settings.ptt.enabled, {
        icon: "mic",
        tone: settings.ptt.enabled && pttOpen ? "tone-green" : "",
        help: "Igual ao push-to-talk clássico: o microfone começa fechado. Segurar a tecla abre a voz. Soltar fecha de novo. O modo invertido faz o contrário.",
      })}
      <div class="row ${settings.ptt.enabled ? "" : "disabled"}">
        <div class="label"><span class="title">Tecla</span><small>O Caps Lock continua trocando maiúsculo e minúsculo</small></div>
        <button type="button" class="keycap ${capturing === "ptt" ? "arm" : ""}" id="ptt-key">${esc(settings.ptt.key === " " ? "Space" : settings.ptt.key)}</button>
      </div>
      <div class="row ${settings.ptt.enabled ? "" : "disabled"}">
        <div class="label"><span class="title">Ao segurar</span></div>
        ${segmented("ptt-mode", [{ id: "talk", label: "Falar" }, { id: "mute", label: "Mutar" }], settings.ptt.inverted ? "mute" : "talk")}
      </div>
    </div>`;

  const renderMicApp = () => `
    <p class="section-hint">O que o KamoVox faz quando você liga o computador.</p>
    <div class="group">
      ${switchRow("startup", "Iniciar com o Windows", "Abre o KamoVox quando você iniciar o computador.", settings.startWithWindows)}
      <div class="row">
        <div class="label"><span class="title">Aparência</span><small>Claro, escuro ou o mesmo do Windows</small></div>
        ${segmented("theme", [
          { id: "system", label: "Sistema" },
          { id: "light", label: "Claro" },
          { id: "dark", label: "Escuro" },
        ], settings.theme)}
      </div>
    </div>
    <h2>Recursos experimentais</h2>
    <p class="section-hint">Funções em teste. Podem mudar ou falhar. Desligar oculta os controles Beta no Microfone.</p>
    <div class="group">
      ${switchRow(
        "beta",
        "Beta",
        settings.beta ? "Recursos experimentais visíveis" : "Só funções estáveis",
        settings.beta,
        {
          icon: "sparkle",
          tone: "tone-purple",
          help: "Com Beta ligado, aparecem “Usar só a minha voz” e “Voz mais suave”. Ao desligar, esses recursos saem da tela e são desativados.",
        },
      )}
    </div>`;

  const renderMicSound = () => `
    ${micSoundControls()}
    ${disclosure("eq", "Equalizador", eqSummary(), eqMarkup(), "tune", "tone-orange")}`;

  mount(`
    ${head(
      "Ajustes",
      status?.running ? `Áudio funcionando a ${fmt(status.sampleRate / 1000, 1)} kHz` : "O áudio está parado",
      `<button class="btn plain" id="refresh">${icon("refresh")}Atualizar</button>`,
    )}
    <div class="page-tabs">
      ${segmented("mic-tab", [
        { id: "mic", label: "Microfone" },
        { id: "talk", label: "Push-to-talk" },
        { id: "app", label: "Programa" },
      ], micTab, true)}
    </div>
    ${micTab === "mic" ? renderMicAudio() : micTab === "talk" ? renderMicTalk() : renderMicApp()}`);
  onSegment("mic-tab", (v) => {
    if (v === micTab) return;
    micTab = v as typeof micTab;
    content.scrollTop = 0;
    animateNext = false;
    renderMic();
  });
  bindMicSoundControls();
  $("#cal-open")?.addEventListener("click", openCalibration);
  bindSwitch("ptt-on", (on) => {
    settings.ptt.enabled = on;
    save();
    syncPtt();
    renderMic();
  });
  $("#ptt-key")?.addEventListener("click", () => {
    const btn = $<HTMLButtonElement>("#ptt-key");
    btn.textContent = "Pressione…";
    btn.classList.add("arm");
    const onKey = (e: KeyboardEvent) => {
    e.preventDefault();
      e.stopPropagation();
      if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;
      window.removeEventListener("keydown", onKey, true);
      settings.ptt.key = e.key === " " ? "Space" : e.key.length === 1 ? e.key.toUpperCase() : e.key;
      save();
      syncPtt();
      renderMic();
    };
    window.addEventListener("keydown", onKey, true);
  });
  onSegment("ptt-mode", (v) => {
    settings.ptt.inverted = v === "mute";
    const seg = document.querySelector('[data-seg="ptt-mode"]');
    seg?.querySelectorAll("button").forEach((b) => b.classList.toggle("active", (b as HTMLButtonElement).dataset.value === v));
    layoutSegmented();
    save();
    syncPtt();
  });
  bindSwitch("startup", (on) => {
    settings.startWithWindows = on;
    save();
    invoke("set_startup", { enabled: on }).catch((e) => toast(String(e)));
  });
  bindSwitch("beta", (on) => {
    settings.beta = on;
    if (!on) {
      settings.params.voiceGate = false;
      settings.params.smooth = false;
      pushParams();
    }
    save();
    renderMic();
  });
  onSegment("theme", (v) => {
    settings.theme = v as Settings["theme"];
    applyTheme();
    save();
    renderMic();
  });
  bindPickers((id, value) => {
    if (id === "input") settings.input = value || null;
    else settings.output = value || null;
    save();
    restartEngine();
    renderMic();
  });

  if (integrated) {
    bindSwitch("windows-mic", (on) => {
      settings.windowsMic = on;
      save();
      restartEngine();
    });
  }
  $$<HTMLButtonElement>("[data-mic-install]", content).forEach((b) =>
    b.addEventListener("click", () => confirmMicInstall(endpoints.find((e) => e.id === b.dataset.micInstall)!)),
  );
  $$<HTMLButtonElement>("[data-mic-remove]", content).forEach((b) =>
    b.addEventListener("click", () => changeMicIntegration(b.dataset.micRemove!, false)),
  );
  bindSwitch("monitor", (on) => {
    settings.monitor = on;
    save();
    restartEngine();
  });
  $("#refresh").addEventListener("click", async () => {
    await refreshDevices();
    renderMic();
    toast("Lista de dispositivos atualizada");
  });
}

function installedEndpoint(): Endpoint | undefined {
  return endpoints.find((e) => e.installed);
}

function confirmMicInstall(ep: Endpoint) {
  openSheet(`
    <h3>Instalar no ${esc(ep.name)}</h3>
    <p>O Windows vai pedir permissão de administrador e o som do computador vai reiniciar por alguns segundos.
    ${ep.other ? `O ${esc(ep.other)} deixa de atuar neste microfone; ao remover o KamoVox, ele volta.` : ""}</p>
    <button class="btn block" id="mic-confirm">Instalar</button>`);
  $("#mic-confirm").addEventListener("click", () => {
    closeSheet();
    changeMicIntegration(ep.id, true);
  });
}

async function changeMicIntegration(endpoint: string, install: boolean) {
  micBusy = true;
  if (settings.page === "mic") renderMic();
  try {
    endpoints = await invoke<Endpoint[]>("mic_integration", { endpoint, install });
    if (install) settings.windowsMic = true;
    save();
    await refreshDevices();
    toast(install ? "KamoVox instalado no microfone do Windows" : "Microfone do Windows restaurado");
  } catch (e) {
    toast(String(e));
  }
  micBusy = false;
  await restartEngine();
  if (settings.page === "mic") renderMic();
}

const PAGES: Record<Page, () => void> = {
  voices: renderVoices,
  ai: renderAi,
  effects: renderEffects,
  music: renderMusic,
  shortcuts: renderShortcuts,
  mic: renderMic,
  profiles: renderProfiles,
};

// ---------- shortcuts and music ----------

let hotkeyTimer: number | undefined;
let capturing: string | null = null;
/** Aba de Ajustes: microfone, push-to-talk ou programa. */
let micTab: "mic" | "talk" | "app" = "mic";
let pttOpen = false;

function scheduleHotkeys() {
  window.clearTimeout(hotkeyTimer);
  hotkeyTimer = window.setTimeout(syncHotkeys, 200);
}

function shortcutRows(): { id: string; name: string; hint: string; glyph: string; tone?: string }[] {
  const rows = [
    { id: "power", name: "Ligar o modificador", hint: "Liga ou desliga toda a alteração", glyph: "power", tone: "" },
    { id: "ai", name: "Ligar a Voz IA", hint: "Liga ou desliga a voz de inteligência artificial", glyph: "ai", tone: "tone-purple" },
    { id: "windowsMic", name: "Voz nos programas", hint: "Discord, jogos e chamadas", glyph: "link", tone: "tone-green" },
    { id: "monitor", name: "Ouvir no fone", hint: "A sua voz alterada", glyph: "headphones", tone: "tone-purple" },
  ];
  return rows;
}

function collectBinds(): { id: string; combo: string }[] {
  const ids = [
    ...shortcutRows().map((r) => r.id),
    ...VOICES.map((v) => `voice:${v.id}`),
    ...(library?.voices.map((v) => `ai:${v.id}`) ?? []),
    ...EFFECTS.map((e) => `effect:${e.key}`),
    ...settings.tracks.map((t) => `music:${t.id}`),
    ...settings.customs.map((c) => `custom:${c.id}`),
    ...settings.profiles.map((p) => profileShortcut(p.name)),
  ];
  return ids.filter((id) => settings.shortcuts[id]).map((id) => ({ id, combo: settings.shortcuts[id] }));
}

async function syncHotkeys() {
  try {
    const failed = await invoke<string[]>("set_hotkeys", { binds: collectBinds() });
    if (failed.length) toast(failed[0]);
  } catch {
    /* backend still starting */
  }
}

function comboFromEvent(e: KeyboardEvent): string | null {
  if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return null;
  const name = e.key === " " ? "Space" : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  const fn = /^F\d{1,2}$/.test(name);
  if (!fn && !e.ctrlKey && !e.altKey && !e.shiftKey && !e.metaKey) return null;
  const parts = [e.ctrlKey ? "Ctrl" : "", e.altKey ? "Alt" : "", e.shiftKey ? "Shift" : "", e.metaKey ? "Win" : "", name].filter(Boolean);
  return parts.join("+");
}

function runHotkey(id: string) {
  if (id.startsWith("custom:")) {
    const custom = settings.customs.find((c) => `custom:${c.id}` === id);
    if (custom) runHotkey(custom.action);
    return;
  }
  if (id === "power") {
    settings.params.enabled = !settings.params.enabled;
    syncPower();
    pushParams();
  } else if (id === "ai") {
    if (!aiAvailable()) {
      navigate("ai");
      return;
    }
    settings.ai.enabled = !settings.ai.enabled;
    save();
    restartEngine();
  } else if (id === "windowsMic") {
    settings.windowsMic = !settings.windowsMic;
    save();
    restartEngine();
  } else if (id === "monitor") {
    settings.monitor = !settings.monitor;
    save();
    restartEngine();
  } else if (id.startsWith("voice:")) {
    settings.ai.enabled = false;
    settings.params.enabled = true;
    settings.params.voices = [id.slice(6)];
    settings.mixer = false;
    syncPower();
    pushParams();
    save();
    restartEngine();
  } else if (id.startsWith("ai:")) {
    settings.ai.voice = id.slice(3);
    settings.ai.enabled = true;
    settings.params.enabled = true;
    save();
    syncPower();
    pushParams();
    restartEngine();
  } else if (id.startsWith("effect:")) {
    const key = id.slice(7) as EffectKey;
    settings.params[key] = !settings.params[key];
    settings.params.enabled = true;
    pushParams();
    syncPower();
  } else if (id.startsWith("profile:")) {
    const name = decodeURIComponent(id.slice("profile:".length));
    const index = settings.profiles.findIndex((p) => p.name === name);
    if (index >= 0) applyProfile(index);
    return;
  } else if (id.startsWith("music:")) {
    const track = settings.tracks.find((t) => `music:${t.id}` === id);
    if (!track) return;
    invoke("play_sound", { id: track.id, path: track.path }).catch((e) => toast(String(e)));
  }
  if (settings.page === "shortcuts" || settings.page === "music" || settings.page === "effects" || settings.page === "voices" || settings.page === "profiles") rerender();
}

function bindKey(id: string) {
  return `<button type="button" class="keycap ${capturing === id ? "arm" : ""}" data-bind="${esc(id)}">${esc(settings.shortcuts[id] || (capturing === id ? "Pressione…" : "Definir"))}</button>${
    settings.shortcuts[id] ? `<button type="button" class="keycap clear" data-clear="${esc(id)}" aria-label="Limpar atalho">${icon("close")}</button>` : ""
  }`;
}

function bindKeyButtons() {
  $$<HTMLButtonElement>("[data-bind]", content).forEach((b) =>
    b.addEventListener("click", () => {
      capturing = b.dataset.bind!;
      rerender();
      const onKey = (e: KeyboardEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === "Escape") {
          delete settings.shortcuts[capturing!];
        } else {
          const combo = comboFromEvent(e);
          if (!combo) return;
          const owner = Object.entries(settings.shortcuts).find(([k, v]) => v === combo && k !== capturing);
          if (owner) {
            toast("Essa tecla já está em outro atalho");
            return;
          }
          settings.shortcuts[capturing!] = combo;
        }
        window.removeEventListener("keydown", onKey, true);
        capturing = null;
        save();
        rerender();
      };
      window.addEventListener("keydown", onKey, true);
    }),
  );
  $$<HTMLButtonElement>("[data-clear]", content).forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      delete settings.shortcuts[b.dataset.clear!];
      save();
      rerender();
    }),
  );
}

function renderShortcuts() {
  const voiceRows = VOICES.map((v) => ({ id: `voice:${v.id}`, name: v.name, hint: "Ativa esta voz pronta e desliga a IA", glyph: v.id }));
  const aiRows = (library?.voices ?? []).map((v) => ({ id: `ai:${v.id}`, name: v.name, hint: "Ativa esta voz de IA", glyph: "ai", tone: "tone-purple" }));
  const effectRows = EFFECTS.map((e) => ({ id: `effect:${e.key}`, name: e.name, hint: "Liga ou desliga o efeito", glyph: e.key }));
  const group = (title: string, rows: { id: string; name: string; hint: string; glyph: string; tone?: string }[]) =>
    rows.length
      ? `${title ? `<h2>${title}</h2>` : ""}<div class="group">${rows
          .map(
            (r) => `<div class="row slim">
              <span class="glyph-sm ${r.tone ?? ""}">${icon(r.glyph)}</span>
              <div class="label"><span class="title">${esc(r.name)}</span></div>
              ${bindKey(r.id)}
            </div>`,
          )
          .join("")}</div>`
      : "";
  const customBody = settings.customs
    .map(
      (c) => `<div class="row">
        <span class="glyph-sm">${icon("shortcuts")}</span>
        <div class="label"><span class="title">${esc(c.name)}</span><small>${esc(actionLabel(c.action))}</small></div>
        ${bindKey(`custom:${c.id}`)}
        <button class="btn danger" data-custom-del="${esc(c.id)}" aria-label="Apagar atalho">${icon("trash")}</button>
      </div>`,
    )
    .join("");
  mount(`
    ${head("Atalhos", "Uma tecla para cada ação. Letras precisam de Ctrl, Alt ou Shift. F1 a F12 podem ir sozinhas.", `<button class="btn" id="custom-new">${icon("plus")}Personalizado</button>`)}
    ${group("Geral", shortcutRows())}
    ${disclosure("sc-voices", "Vozes prontas", `${VOICES.length} vozes`, group("", voiceRows), "voices")}
    ${disclosure("sc-ai", "Vozes IA", aiRows.length ? `${aiRows.length} vozes` : "Nenhuma instalada", group("", aiRows), "ai", "tone-purple")}
    ${disclosure("sc-fx", "Efeitos", `${EFFECTS.length} efeitos`, group("", effectRows), "effects")}
    ${disclosure("sc-music", "Músicas", settings.tracks.length ? `${settings.tracks.length} faixas` : "Nenhuma", group("", settings.tracks.map((t) => ({ id: `music:${t.id}`, name: t.name, hint: "Toca no microfone. De novo, para.", glyph: "music", tone: "tone-orange" }))), "music", "tone-orange")}
    <h2>Personalizados</h2>
    <div class="group">${customBody || `<div class="empty">Nenhum atalho personalizado. Crie um com o botão acima.</div>`}</div>`);
  bindKeyButtons();
  $("#custom-new").addEventListener("click", openCustomShortcut);
  $$<HTMLButtonElement>("[data-custom-del]", content).forEach((b) =>
    b.addEventListener("click", () => {
      settings.customs = settings.customs.filter((c) => c.id !== b.dataset.customDel);
      delete settings.shortcuts[`custom:${b.dataset.customDel}`];
      save();
      renderShortcuts();
    }),
  );
}

function actionLabel(id: string) {
  return shortcutRows().find((r) => r.id === id)?.name
    ?? VOICES.find((v) => `voice:${v.id}` === id)?.name
    ?? library?.voices.find((v) => `ai:${v.id}` === id)?.name
    ?? EFFECTS.find((e) => `effect:${e.key}` === id)?.name
    ?? settings.tracks.find((t) => `music:${t.id}` === id)?.name
    ?? (id.startsWith("profile:") ? decodeURIComponent(id.slice("profile:".length)) : id);
}

function openCustomShortcut() {
  const choices = [
    ...shortcutRows().map((r) => ({ id: r.id, label: r.name })),
    ...VOICES.map((v) => ({ id: `voice:${v.id}`, label: `Voz · ${v.name}` })),
    ...(library?.voices.map((v) => ({ id: `ai:${v.id}`, label: `IA · ${v.name}` })) ?? []),
    ...EFFECTS.map((e) => ({ id: `effect:${e.key}`, label: `Efeito · ${e.name}` })),
    ...settings.tracks.map((t) => ({ id: `music:${t.id}`, label: `Música · ${t.name}` })),
    ...settings.profiles.map((p) => ({ id: profileShortcut(p.name), label: `Perfil · ${p.name}` })),
  ];
  let action = choices[0]?.id ?? "";
  const current = choices.find((c) => c.id === action);
  openSheet(`
    <h3>Atalho personalizado</h3>
    <p>Dê um nome e escolha o que a tecla faz. A tecla em si você define na lista.</p>
    <label class="field-label" for="custom-name">Nome</label>
    <input type="text" id="custom-name" maxlength="40" placeholder="Ex.: Voz de jogo" />
    <label class="field-label">Ação</label>
    <div class="picker sheet-picker" data-picker="custom-action">
      <button type="button" class="picker-btn" id="custom-action-btn" aria-haspopup="listbox" aria-expanded="false">${esc(current?.label ?? "")}</button>
      <div class="picker-menu up" role="listbox" hidden>
        ${choices.map((c) => `<button type="button" class="picker-opt ${c.id === action ? "active" : ""}" data-value="${esc(c.id)}">${esc(c.label)}</button>`).join("")}
      </div>
    </div>
    <button class="btn block" id="custom-save">Adicionar</button>`);
  const box = $<HTMLElement>(".sheet-picker", $("#sheet"));
  const btn = $<HTMLButtonElement>("#custom-action-btn");
  const menu = $<HTMLElement>(".picker-menu", box);
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = menu.hidden;
    menu.hidden = !willOpen;
    btn.setAttribute("aria-expanded", String(willOpen));
  });
  $$<HTMLButtonElement>(".picker-opt", menu).forEach((opt) =>
    opt.addEventListener("click", (e) => {
      e.stopPropagation();
      action = opt.dataset.value ?? action;
      btn.textContent = opt.textContent;
      menu.hidden = true;
      btn.setAttribute("aria-expanded", "false");
      $$(".picker-opt", menu).forEach((el) => el.classList.toggle("active", el === opt));
    }),
  );
  $("#custom-save").addEventListener("click", () => {
    const name = $<HTMLInputElement>("#custom-name").value.trim();
    if (!name) return $<HTMLInputElement>("#custom-name").focus();
    settings.customs.push({ id: crypto.randomUUID(), name, action });
    save();
    closeSheet();
    renderShortcuts();
  });
}

function renderMusic() {
  mount(`
    ${head("Música", "Toque no som para ouvir. A tecla manda o áudio para o microfone.", `<button class="btn" id="track-add">${icon("plus")}Adicionar</button>`)}
    ${
      settings.tracks.length
        ? `<div class="sound-grid">${settings.tracks
            .map(
              (t) => `<article class="sound">
                <button type="button" class="sound-play" data-track-play="${esc(t.id)}" aria-label="Tocar ${esc(t.name)}">${icon("music")}</button>
                <div class="sound-name" title="${esc(t.path)}">${esc(t.name)}</div>
                <div class="sound-keys">${bindKey("music:" + t.id)}</div>
                <button type="button" class="sound-remove" data-track-del="${esc(t.id)}" aria-label="Remover ${esc(t.name)}">${icon("close")}</button>
              </article>`,
            )
            .join("")}</div>`
        : `<div class="group"><div class="empty">Nenhuma música ainda.</div></div>`
    }`);
  bindKeyButtons();
  $("#track-add").addEventListener("click", async () => {
    const file = await openDialog({
      title: "Adicionar música",
      filters: [{ name: "Áudio", extensions: ["mp3", "wav", "ogg", "flac", "m4a", "aac", "mp4"] }],
    });
    if (typeof file !== "string") return;
    const name = file.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "Música";
    settings.tracks.push({ id: crypto.randomUUID(), name, path: file });
    save();
    renderMusic();
  });
  $$<HTMLButtonElement>("[data-track-play]", content).forEach((b) =>
    b.addEventListener("click", () => runHotkey(`music:${b.dataset.trackPlay}`)),
  );
  $$<HTMLButtonElement>("[data-track-del]", content).forEach((b) =>
    b.addEventListener("click", () => {
      const id = b.dataset.trackDel!;
      settings.tracks = settings.tracks.filter((t) => t.id !== id);
      delete settings.shortcuts[`music:${id}`];
      save();
      renderMusic();
    }),
  );
}

function navIndex(page: Page) {
  const id = page === "ai" ? "voices" : page;
  return NAV.findIndex((n) => n.id === id);
}

function placeNavPill(animate = true) {
  const item = $$<HTMLButtonElement>(".nav-item")[navIndex(settings.page)];
  const pill = document.getElementById("nav-pill");
  if (!item || !pill || !item.offsetHeight) return;
  const nav = item.offsetParent instanceof HTMLElement ? item.offsetParent : null;
  if (nav) {
    const top = item.offsetTop;
    const bottom = top + item.offsetHeight;
    if (top < nav.scrollTop) nav.scrollTop = top;
    else if (bottom > nav.scrollTop + nav.clientHeight) nav.scrollTop = bottom - nav.clientHeight;
  }
  if (!animate) pill.style.transition = "none";
  pill.style.height = `${item.offsetHeight}px`;
  pill.style.transform = `translateY(${item.offsetTop}px)`;
  if (!animate) {
    void pill.offsetWidth;
    pill.style.transition = "";
  }
}

function navigate(page: Page) {
  const changed = page !== settings.page;
  settings.page = page;
  save();
  const idx = navIndex(page);
  $$(".nav-item").forEach((el, i) => el.classList.toggle("active", i === idx));
  placeNavPill();
  $("#compact-title").textContent = page === "ai" ? "Voz IA" : (NAV[idx]?.name ?? "");
  animateNext = true;
  hidePopover();
  if (changed) content.scrollTop = 0;
  PAGES[page]();
}

function renderNav() {
  $("#nav").insertAdjacentHTML(
    "beforeend",
    NAV.map((n) => `<button class="nav-item" data-page="${n.id}">${icon(n.id === "mic" ? "gear" : n.id)}${n.name}</button>`).join(""),
  );
  $$<HTMLButtonElement>(".nav-item").forEach((b) => b.addEventListener("click", () => navigate(b.dataset.page as Page)));
  window.addEventListener("resize", () => placeNavPill(false));
}

// ---------- sheet & popover ----------

function openSheet(html: string) {
  const sheet = $("#sheet");
  sheet.innerHTML = `<div class="handle"></div>${html}`;
  sheet.style.transform = "";
  sheet.classList.add("show");
  $("#sheet-backdrop").classList.add("show");

  const handle = $(".handle", sheet);
  handle.addEventListener("pointerdown", (e) => {
    const startY = e.clientY;
    handle.setPointerCapture(e.pointerId);
    sheet.style.transition = "none";
    const move = (ev: PointerEvent) => {
      const dy = Math.max(0, ev.clientY - startY);
      sheet.style.transform = `translate(-50%, ${dy}px)`;
    };
    const up = (ev: PointerEvent) => {
      handle.removeEventListener("pointermove", move);
      sheet.style.transition = "";
      sheet.style.transform = "";
      if (ev.clientY - startY > 70) closeSheet();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up, { once: true });
  });
}

function closeSheet() {
  $("#sheet").classList.remove("show");
  $("#sheet-backdrop").classList.remove("show");
}

/** iOS-style action sheet for destructive actions. */
function confirmSheet(title: string, message: string, action: string, onConfirm: () => void) {
  openSheet(`
    <h3>${esc(title)}</h3>
    <p>${esc(message)}</p>
    <div class="sheet-actions">
      <button class="btn block destructive" id="sheet-confirm">${icon("trash")}${esc(action)}</button>
      <button class="btn block plain" id="sheet-cancel">Cancelar</button>
    </div>`);
  $("#sheet-cancel").addEventListener("click", closeSheet);
  $("#sheet-confirm").addEventListener("click", () => {
    closeSheet();
    onConfirm();
  });
  setTimeout(() => $("#sheet-cancel")?.focus(), 250);
}

let popoverFor: HTMLElement | null = null;
let hoverTimer: number | undefined;

function showPopover(trigger: HTMLElement) {
  const pop = $("#popover");
  pop.textContent = trigger.dataset.help ?? "";
  pop.style.left = "0px";
  pop.style.top = "0px";
  const r = trigger.getBoundingClientRect();
  const w = pop.offsetWidth;
  const h = pop.offsetHeight;
  const left = Math.min(window.innerWidth - w - 10, Math.max(10, r.left + r.width / 2 - w / 2));
  const above = r.top - h - 8 > 8;
  const top = above ? r.top - h - 8 : r.bottom + 8;
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
  pop.style.setProperty("--origin", `${r.left + r.width / 2 - left}px ${above ? "100%" : "0%"}`);
  pop.classList.add("show");
  popoverFor = trigger;
}

function hidePopover() {
  window.clearTimeout(hoverTimer);
  $("#popover").classList.remove("show");
  popoverFor = null;
}

function setupPopovers() {
  document.addEventListener("click", (e) => {
    const trigger = (e.target as HTMLElement).closest<HTMLElement>("[data-help]");
    if (trigger) {
      e.preventDefault();
      if (popoverFor === trigger) hidePopover();
      else showPopover(trigger);
    } else if (popoverFor) {
      hidePopover();
    }
  });
  document.addEventListener("mouseover", (e) => {
    const trigger = (e.target as HTMLElement).closest<HTMLElement>("[data-help]");
    if (!trigger || popoverFor === trigger) return;
    window.clearTimeout(hoverTimer);
    hoverTimer = window.setTimeout(() => showPopover(trigger), 350);
  });
  document.addEventListener("mouseout", (e) => {
    const trigger = (e.target as HTMLElement).closest<HTMLElement>("[data-help]");
    if (trigger) hidePopover();
  });
  content.addEventListener("scroll", hidePopover, { passive: true });
  window.addEventListener("resize", hidePopover);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hidePopover();
      closeSheet();
    }
  });
  $("#sheet-backdrop").addEventListener("click", closeSheet);
}

// ---------- engine ----------

async function refreshDevices() {
  try {
    [devices, endpoints] = await Promise.all([invoke<DeviceList>("list_devices"), invoke<Endpoint[]>("mic_endpoints")]);
  } catch (e) {
    toast(String(e));
  }
  devicesReady = true;
  paintNotice();
}

let engineQueue: Promise<void> = Promise.resolve();

/** Restarts are serialized: loading an AI voice takes a few seconds. */
function restartEngine(): Promise<void> {
  engineQueue = engineQueue.then(startEngineNow);
  return engineQueue;
}

async function startEngineNow() {
  const windowsMic = settings.windowsMic ? installedEndpoint() : undefined;
  const config = {
    input: windowsMic ? windowsMic.name : settings.input,
    output: settings.output,
    monitor: settings.monitor,
    apo: !!windowsMic,
  };
  const ai = aiRequest();
  aiLoading = !!ai;
  syncPower();
  if (settings.page === "ai" && !refreshAiLive()) rerender();
  try {
    status = await invoke<Status>("start_engine", { config, ai });
    engineError = null;
  } catch (e) {
    const msg = String(e);
    if (config.apo) {
      // Keep KamoVox usable locally even if the Windows microphone is unavailable.
      toast(msg);
      try {
        status = await invoke<Status>("start_engine", { config: { ...config, input: settings.input, apo: false }, ai });
        engineError = null;
      } catch (e2) {
        status = null;
        engineError = String(e2);
      }
    } else if (ai) {
      // Keep the microphone working without the AI voice.
      settings.ai.enabled = false;
      save();
      toast(msg);
      try {
        status = await invoke<Status>("start_engine", { config, ai: null });
        engineError = null;
      } catch (e2) {
        status = null;
        engineError = String(e2);
      }
    } else {
      status = null;
      engineError = msg;
      toast(msg);
    }
  }
  aiLoading = false;
  syncPower();
  if (settings.page === "mic" || (settings.page === "ai" && !refreshAiLive())) rerender();
}

const meters = { input: 0, output: 0 };
const meterWidth = (peak: number) => (peak <= 0 ? 0 : Math.max(0, Math.min(1, (20 * Math.log10(peak) + 60) / 60)));

async function pollLevels() {
  try {
    const lv = await invoke<Levels>("get_levels");
    meters.input = Math.max(meterWidth(lv.input), meters.input * 0.82);
    meters.output = Math.max(meterWidth(lv.output), meters.output * 0.82);
    $("#meter-in").style.width = `${meters.input * 100}%`;
    $("#meter-out").style.width = `${meters.output * 100}%`;
    const speaking = meters.input > 0.04;
    if (!status?.running || !settings.params.enabled || speaking) quietSince = 0;
    else if (!quietSince) quietSince = performance.now();
    const silent = quietSince > 0 && performance.now() - quietSince > 2500;
    if (silent !== inputSilent) {
      inputSilent = silent;
      paintNotice();
    }
    document.querySelector(".meter-line")?.classList.toggle("hot", meters.input > 0.82);
    if (lv.ai) {
      for (const [lat, load] of [["#ai-latency", "#ai-load"], ["#power-latency", "#power-load"]]) {
        const latEl = document.querySelector(lat);
        const loadEl = document.querySelector(load);
        if (!latEl || !loadEl) continue;
        latEl.textContent = `${Math.round(lv.ai.latencyMs)} ms`;
        loadEl.textContent = percent(lv.ai.load * 100);
        loadEl.classList.toggle("warn", lv.ai.load > 0.85);
      }
    }
    if (lv.error) {
      engineError = lv.error;
      toast(lv.error);
      syncPower();
    }
    if (eqDrag < 0) paintGateDiag(lv.gate ?? null, lv.buildId);
  } catch {
    /* backend busy; next tick retries */
  }
  window.setTimeout(pollLevels, 50);
}

function paintGateDiag(g: GateSnapshot | null, buildId?: string) {
  const root = document.getElementById("gate-diag-live");
  if (!root) return;
  const set = (id: string, text: string, ok?: boolean | null) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("ok", ok === true);
    el.classList.toggle("bad", ok === false);
  };
  set("gd-build", buildId && buildId.length > 0 ? buildId : "—");
  if (!g) {
    set("gd-phase", settings.params.voiceGate ? "aguardando motor…" : "gate off");
    for (const id of [
      "gd-reason",
      "gd-neural",
      "gd-thresh",
      "gd-strict",
      "gd-mode",
      "gd-speech",
      "gd-buf",
      "gd-bufcap",
      "gd-delay",
      "gd-infer",
      "gd-vage",
      "gd-utt",
      "gd-auth",
      "gd-env",
    ]) {
      set(id, "—");
    }
    const log = document.getElementById("gd-log");
    if (log) log.textContent = "";
    return;
  }
  const open = g.authorized || g.phaseLabel === "Locked" || g.phaseLabel === "Holding";
  const blocked = g.phaseLabel === "Rejected" || g.reasonLabel === "OTHER_SPEAKER_REJECTED";
  set("gd-phase", g.phaseLabel, open ? true : blocked ? false : null);
  set("gd-reason", g.reasonLabel, open ? true : blocked ? false : null);
  const neural = g.neural;
  const pass = neural != null && neural >= g.thresh;
  set(
    "gd-neural",
    neural == null || Number.isNaN(neural) ? "sem score" : neural.toFixed(3),
    neural == null ? null : pass,
  );
  set("gd-thresh", g.thresh.toFixed(3), pass);
  set("gd-strict", `${Math.round(g.strict * 100)}%`);
  set("gd-mode", g.protect ? "Protegido" : "Rápido");
  set("gd-speech", `${Math.round(g.speechMs)} ms`);
  set("gd-buf", `${Math.round(g.bufferedMs)} ms (retido)`);
  set("gd-bufcap", `${Math.round(g.bufferCapacityMs)} ms`);
  set("gd-delay", `${Math.round(g.delayMs)} ms`);
  set("gd-infer", g.inferMs > 0 ? `${g.inferMs.toFixed(0)} ms · #${g.inferId}` : `— · #${g.inferId}`);
  set("gd-vage", `${Math.round(g.verifyAgeMs)} ms`);
  set("gd-utt", String(g.utteranceId));
  set("gd-auth", g.authorized ? "sim · transmitindo" : "não · mudo", g.authorized);
  set("gd-env", g.speechEnv.toFixed(2));
  const log = document.getElementById("gd-log");
  if (log) {
    const lines = (g.events ?? [])
      .slice(-8)
      .map((e) => {
        const n = Number.isFinite(e.neural) ? e.neural.toFixed(2) : "—";
        return `#${e.utteranceId} ${e.prev}→${e.next} r${e.reason} cos=${n}/${e.thresh.toFixed(2)} buf=${e.delaySamples}`;
      });
    log.textContent = lines.join("\n");
  }
}

function trayStatsLine() {
  if (aiRunning()) {
    const where = status?.aiDevice === "gpu" ? "GPU" : "CPU";
    const load = document.getElementById("power-load")?.textContent ?? "—";
    const latency = document.getElementById("power-latency")?.textContent ?? "—";
    return `${where} ${load} · Atraso ${latency}`;
  }
  return settings.params.smooth ? "Voz pronta · tom suave" : "Voz pronta · atraso baixo";
}

function aiRunning() {
  return settings.ai.enabled && !!status?.aiDevice;
}

function powerStats() {
  const ai = aiRunning();
  const cell = (label: string, value: string, id = "") =>
    `<div class="stat"><small>${label}</small><b${id ? ` id="${id}"` : ""}>${value}</b></div>`;
  return [
    cell(ai ? (status?.aiDevice === "gpu" ? "Uso da GPU" : "Uso da CPU") : "Processamento", ai ? "—" : "Voz pronta", "power-load"),
    cell("Atraso", ai ? "—" : settings.params.smooth ? "Suave" : "Baixo", "power-latency"),
  ].join("");
}

let inputSilent = false;
let quietSince = 0;
let devicesReady = false;

function noticeState(): { text: string; kind: "" | "err" | "warn" | "load"; go: string } {
  if (aiLoading) return { text: "Carregando…", kind: "load", go: "" };
  if (engineError) return { text: "Erro", kind: "err", go: "" };
  if (inputSilent) return { text: "Sem som", kind: "warn", go: "" };
  if (devicesReady && !installedEndpoint()) return { text: "Instalar", kind: "warn", go: "mic" };
  return { text: "", kind: "", go: "" };
}

function paintNotice() {
  const notice = noticeState();
  const sub = $<HTMLButtonElement>("#power-sub");
  const card = $("#power-card");
  sub.hidden = !notice.text;
  sub.classList.toggle("err", notice.kind === "err");
  sub.classList.toggle("warn", notice.kind === "warn");
  sub.dataset.go = notice.go;
  sub.title = notice.kind === "err" ? "Erro no áudio" : notice.text;
  sub.querySelector("span")!.textContent = notice.text;
  card.classList.toggle("alert", notice.kind === "err");
  document.querySelector(".meter-line")?.classList.toggle("quiet", inputSilent && !engineError);
}

function syncPtt() {
  pttOpen = !settings.ptt.enabled || settings.ptt.inverted;
  paintPtt();
  invoke("set_ptt", { enabled: settings.ptt.enabled, inverted: settings.ptt.inverted, key: settings.ptt.key }).catch((e) => toast(String(e)));
  invoke("set_mic_mute", { muted: !settings.ptt.enabled && settings.micMuted }).catch(() => {});
}

/** Green mic: the voice is going out. Push-to-talk follows the key. Otherwise it stays open until a click mutes it. */
function micLive() {
  if (settings.ptt.enabled) return settings.params.enabled && pttOpen;
  return !settings.micMuted;
}

function pttTalking() {
  return micLive();
}

let lastTalking: boolean | null = null;

function paintPtt() {
  const talking = pttTalking();
  const muted = !settings.ptt.enabled && settings.micMuted;
  const mic = document.getElementById("ptt-mic");
  mic?.classList.toggle("live", talking);
  mic?.classList.toggle("muted", muted);
  mic?.setAttribute("aria-pressed", String(muted));
  mic?.setAttribute("aria-label", muted ? "Desmutar o microfone" : "Mutar o microfone");
  mic?.setAttribute("title", settings.ptt.enabled ? "Push-to-talk" : muted ? "Microfone mutado" : "Microfone aberto");
  if (talking !== lastTalking) {
    lastTalking = talking;
    invoke("set_tray_talking", { talking }).catch(() => {});
    scheduleTrayState();
  }
  const text = !settings.ptt.enabled ? "Desligado" : pttOpen ? "Aberto — pode falar" : "Mutado";
  const status = document.getElementById("ptt-status");
  if (status) status.textContent = settings.ptt.enabled ? text : settings.micMuted ? "Microfone mutado" : "A voz passa o tempo todo";
  const hint = document.getElementById("ptt-on")?.closest(".row")?.querySelector("small");
  if (hint) hint.textContent = text;
}

function syncPower() {
  const on = settings.params.enabled;
  const power = $<HTMLButtonElement>("#power");
  power.classList.toggle("on", on);
  power.setAttribute("aria-pressed", String(on));
  power.setAttribute("aria-label", on ? "Desligar o trocador de voz" : "Ligar o trocador de voz");
  const hear = $<HTMLButtonElement>("#hear");
  hear.classList.toggle("on", settings.monitor);
  hear.setAttribute("aria-pressed", String(settings.monitor));
  hear.setAttribute("aria-label", settings.monitor ? "Parar de ouvir a própria voz" : "Ouvir a própria voz");
  const card = $("#power-card");
  card.classList.toggle("off", !on);
  card.classList.toggle("ai", on && aiRunning());
  paintNotice();

  const voice = library?.voices.find((v) => v.id === settings.ai.voice) ?? null;
  const ai = aiRunning() && !!voice;
  $("#power-art").innerHTML = ai && voice?.image ? `<img src="${voice.image}" alt="" />` : icon(ai ? "ai" : "voices");
  $("#power-now-hint").textContent = on ? "Você está falando como" : "Trocador desligado";
  $("#power-voice").textContent = !on ? "Voz natural" : ai ? voice!.name : describe(settings.params);
  paintPtt();
  $("#power-stats").innerHTML = powerStats();
  scheduleTrayState();
}

// ---------- tray menu ----------

let trayTimer: number | undefined;

function scheduleTrayState() {
  window.clearTimeout(trayTimer);
  trayTimer = window.setTimeout(sendTrayState, 40);
}

function aiAvailable() {
  const a = settings.ai;
  return !!library?.baseInstalled && !!a.voice && library.voices.some((v) => v.id === a.voice);
}

function trayState(): TrayState {
  const voice = library?.voices.find((v) => v.id === settings.ai.voice) ?? null;
  return {
    enabled: settings.params.enabled,
    status: aiLoading ? "Carregando voz IA…" : engineError ?? "",
    error: !!engineError,
    voice: settings.ai.enabled && voice ? voice.name : describe(settings.params),
    stats: trayStatsLine(),
    ai: { available: aiAvailable(), on: settings.ai.enabled && aiAvailable(), name: voice?.name ?? null, image: voice?.image ?? null },
    windowsMic: { available: !!installedEndpoint(), on: settings.windowsMic },
    monitor: settings.monitor,
    talking: micLive(),
    micMuted: !settings.ptt.enabled && settings.micMuted,
    ptt: settings.ptt.enabled,
    profile: currentProfileName(),
    classic: { on: classicVoiceOn(), summary: describe(settings.params) },
    profiles: settings.profiles.map((p) => p.name),
  };
}

function sendTrayState() {
  emitTo("tray", TRAY_STATE, trayState()).catch(() => {});
}

function onTrayAction(action: TrayAction) {
  switch (action.kind) {
    case "power":
      settings.params.enabled = action.on;
      syncPower();
      pushParams();
      break;
    case "ai":
      if (action.on && !aiAvailable()) {
        invoke("show_main");
        navigate("ai");
        return;
      }
      settings.ai.enabled = action.on;
      save();
      restartEngine();
      break;
    case "windowsMic":
      settings.windowsMic = action.on;
      save();
      restartEngine();
      break;
    case "monitor":
      settings.monitor = action.on;
      save();
      restartEngine();
      break;
    case "micMute":
      if (settings.ptt.enabled) return;
      settings.micMuted = !settings.micMuted;
      save();
      syncPtt();
      break;
    case "profile":
      applyProfile(action.index);
      return;
  }
  rerender();
}

// ---------- boot ----------

function paintAppVersion() {
  const v = `v${__APP_VERSION__}`;
  const footer = document.getElementById("app-version");
  const log = document.getElementById("log-ver");
  if (footer) footer.textContent = v;
  if (log) log.textContent = v;
}

function setupWindow() {
  try {
    const win = getCurrentWindow();
    // Minimizing and closing both keep KamoVox running in the notification area.
    $("#win-min").addEventListener("click", () => win.hide());
    $("#win-max").addEventListener("click", () => win.toggleMaximize());
    $("#win-close").addEventListener("click", () => win.hide());
  } catch {
    /* A janela nativa só existe dentro do aplicativo. */
  }
  content.addEventListener(
    "scroll",
    () => $("#topbar").classList.toggle("scrolled", content.scrollTop > 34),
    { passive: true },
  );
  window.addEventListener("resize", () => layoutSegmented());
}

const bootStart = performance.now();
const SPLASH_MIN_MS = 1600;
const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

function splashStatus(text: string) {
  const el = document.getElementById("splash-status");
  if (el) el.textContent = text;
}

function splashProgress(pct: number, text: string) {
  splashStatus(text);
  const bar = document.getElementById("splash-fill");
  if (bar) bar.style.width = `${Math.max(8, Math.min(100, pct))}%`;
}

interface UpdateInfo {
  current: string;
  latest: string;
  notes: string;
  available: boolean;
  download: string | null;
  page: string;
}

let pendingUpdate: UpdateInfo | null = null;

function paintUpdateButton() {
  const btn = $<HTMLButtonElement>("#check-update");
  const label = btn.querySelector(".update-label");
  btn.classList.toggle("pending", !!pendingUpdate);
  if (label && !btn.classList.contains("busy")) {
    label.textContent = pendingUpdate ? "Atualização pendente" : "Verificar atualizações";
  }
}

function askToInstall(info: UpdateInfo) {
  openSheet(`
    <h3>Atualização disponível</h3>
    <p>A versão ${esc(info.latest)} está pronta. O KamoVox fecha, troca os arquivos e abre de novo, sem o assistente de instalação.</p>
    <div class="sheet-actions">
      <button class="btn block" id="sheet-confirm">Atualizar agora</button>
      <button class="btn block plain" id="sheet-cancel">Agora não</button>
    </div>`);
  $("#sheet-cancel").addEventListener("click", closeSheet);
  $("#sheet-confirm").addEventListener("click", () => {
    closeSheet();
    void installUpdate(info);
  });
}

async function installUpdate(info: UpdateInfo) {
  const btn = $<HTMLButtonElement>("#check-update");
  const label = btn.querySelector(".update-label");
  btn.disabled = true;
  btn.classList.add("busy");
  if (label) label.textContent = `Baixando ${info.latest}…`;
  try {
    await invoke("install_update", { url: info.download });
  } catch (e) {
    toast(String(e));
    btn.disabled = false;
    btn.classList.remove("busy");
    paintUpdateButton();
  }
}

async function lookForUpdate(interactive: boolean) {
  const btn = $<HTMLButtonElement>("#check-update");
  const label = btn.querySelector(".update-label");
  if (btn.disabled) return;
  btn.disabled = true;
  btn.classList.add("busy");
  if (label) label.textContent = "Verificando…";
  try {
    const info = await invoke<UpdateInfo>("check_update");
    if (info.available && info.download) {
      pendingUpdate = info;
      paintUpdateButton();
      if (interactive) askToInstall(info);
      return;
    }
    pendingUpdate = null;
    if (interactive) {
      if (info.available && !info.download) {
        toast("Há uma versão nova. Abrindo a página para baixar.");
        await openUrl(info.page);
      } else {
        showUpToDate(info.current);
      }
    }
  } catch (e) {
    if (interactive) toast(String(e));
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    paintUpdateButton();
  }
}

const CREDIT_NAME = "Konnak";
const CREDIT_GLYPHS = "K#9x@4n_ABCDEFGHJKLMNPQRSTUVWXYZ0123456789";
let creditFrame = 0;

function scrambleCredit() {
  const el = document.getElementById("credit-name");
  if (!el) return;
  cancelAnimationFrame(creditFrame);
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.textContent = CREDIT_NAME;
    return;
  }
  const start = performance.now();
  const step = (now: number) => {
    const t = Math.min(1, (now - start) / 260);
    const locked = Math.floor(t * CREDIT_NAME.length);
    let text = "";
    for (let i = 0; i < CREDIT_NAME.length; i++) {
      text += i < locked ? CREDIT_NAME[i]! : CREDIT_GLYPHS[Math.floor(Math.random() * CREDIT_GLYPHS.length)]!;
    }
    el.textContent = t === 1 ? CREDIT_NAME : text;
    if (t < 1) creditFrame = requestAnimationFrame(step);
  };
  creditFrame = requestAnimationFrame(step);
}

function resetCredit() {
  cancelAnimationFrame(creditFrame);
  const el = document.getElementById("credit-name");
  if (el) el.textContent = CREDIT_NAME;
}

function openChangelog() {
  $("#log-title").textContent = "Notas da versão";
  $("#log-ver").textContent = $("#app-version").textContent || "";
  $("#log-lead").hidden = true;
  $("#log-notes").hidden = false;
  $("#changelog-backdrop").hidden = false;
}

function showUpToDate(version: string) {
  $("#log-title").textContent = "Você está em dia";
  $("#log-ver").textContent = version.startsWith("v") ? version : `v${version}`;
  const lead = $("#log-lead");
  lead.hidden = false;
  lead.textContent = "Esta já é a versão mais recente do KamoVox.";
  $("#log-notes").hidden = true;
  $("#changelog-backdrop").hidden = false;
}

function closeChangelog() {
  $("#changelog-backdrop").hidden = true;
}

const PIX_KEY = "35b26296-c096-4bf9-b455-7ef0cf6e7f65";
const PIX_EMV = "00020126580014BR.GOV.BCB.PIX013635b26296-c096-4bf9-b455-7ef0cf6e7f655204000053039865802BR5901N6001C62070503***63049E0E";
const SUPPORT_GAP = 3 * 24 * 60 * 60 * 1000;

function usedLine() {
  const minutes = Math.floor(settings.usedMs / 60000);
  const ask = "Que tal retribuir com uma ajuda ao projeto?";
  if (minutes < 1) return `Você acabou de começar a usar o modulador. ${ask}`;
  if (minutes < 60) {
    const unit = minutes === 1 ? "minuto" : "minutos";
    return `Você já usou o modulador por <b>${minutes} ${unit}</b>. ${ask}`;
  }
  const hours = Math.floor(minutes / 60);
  const unit = hours === 1 ? "hora" : "horas";
  return `Você já usou o modulador por <b>${hours} ${unit}</b>. ${ask}`;
}

function openSupport() {
  const used = $("#support-used");
  if (used) used.innerHTML = usedLine();
  const later = $<HTMLButtonElement>("#support-later");
  later.textContent = "Agora não";
  const card = $(".support-card");
  card.classList.remove("peek");
  $<HTMLVideoElement>("#support-peek").pause();
  const copy = $<HTMLButtonElement>("#pix-copy");
  copy.classList.remove("thanks");
  copy.textContent = "Copiar chave Pix";
  $("#support-backdrop").hidden = false;
}

function videoBusy(video: HTMLVideoElement) {
  return !video.paused && !video.ended && video.currentTime > 0.05;
}

function playVideo(video: HTMLVideoElement) {
  if (videoBusy(video)) return;
  video.currentTime = 0;
  video.play().catch(() => {});
}

function setPeek(on: boolean) {
  $(".support-card").classList.toggle("peek", on);
  if (on) playVideo($<HTMLVideoElement>("#support-peek"));
}

function rainConfetti() {
  const layer = document.createElement("div");
  layer.className = "confetti-rain";
  const colors = ["#0a84ff", "#30d158", "#ffd60a", "#ff9f0a", "#bf5af2", "#ff453a", "#64d2ff"];
  for (let i = 0; i < 70; i++) {
    const bit = document.createElement("i");
    bit.style.left = `${Math.random() * 100}%`;
    bit.style.background = colors[i % colors.length]!;
    bit.style.animationDelay = `${Math.random() * 0.25}s`;
    bit.style.animationDuration = `${1.6 + Math.random() * 1.4}s`;
    layer.appendChild(bit);
  }
  document.body.appendChild(layer);
  window.setTimeout(() => layer.remove(), 3400);
}

let mascotTimer = 0;

function shineSupport() {
  const btn = $("#support-open");
  btn.classList.remove("shine");
  void btn.offsetWidth;
  btn.classList.add("shine");
}

function playMascot() {
  const video = $<HTMLVideoElement>("#support-mascot");
  if (videoBusy(video)) return;
  window.clearTimeout(mascotTimer);
  shineSupport();
  video.currentTime = 0;
  video.play().catch(() => {});
}

function scheduleMascot() {
  window.clearTimeout(mascotTimer);
  mascotTimer = window.setTimeout(playMascot, 35000 + Math.random() * 40000);
}

function closeSupport() {
  $("#support-backdrop").hidden = true;
  settings.supportAt = Date.now();
  save();
}

const LATER_LINES = ["Deixar o camaleão com fome", "Tudo bem, ele sobrevive de vento"];
let laterLine = 0;

async function copyPix(button: HTMLButtonElement, text: string, done: string, again: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    toast("Não consegui copiar. O texto está na tela.");
    return;
  }
  rainConfetti();
  button.classList.remove("urge");
  button.classList.add("thanks");
  button.textContent = done;
  window.setTimeout(() => {
    button.classList.remove("thanks");
    button.textContent = again;
  }, 2400);
}

async function copyPixKey() {
  await copyPix($<HTMLButtonElement>("#pix-copy"), PIX_KEY, "Chave copiada! O camaleão agradece", "Copiar chave Pix");
}

function maybeSupport() {
  if (Date.now() - settings.supportAt < SUPPORT_GAP) return;
  openSupport();
}

async function hideSplash() {
  const splash = document.getElementById("splash");
  if (!splash) return;
  await sleep(SPLASH_MIN_MS - (performance.now() - bootStart));
  splash.classList.add("done");
  window.setTimeout(() => {
    splash.remove();
    maybeSupport();
  }, 700);
}

window.addEventListener("DOMContentLoaded", async () => {
  paintAppVersion();
  setupWindow();
  setupPopovers();
  renderNav();
  syncPower();
  $("#power").addEventListener("click", () => {
    settings.params.enabled = !settings.params.enabled;
    syncPower();
    pushParams();
  });
  $("#hear").addEventListener("click", () => {
    settings.monitor = !settings.monitor;
    save();
    syncPower();
    restartEngine();
    if (settings.page === "mic") rerender();
  });
  $("#power-sub").addEventListener("click", () => {
    if ($<HTMLButtonElement>("#power-sub").dataset.go !== "mic") return;
    micTab = "mic";
    if (settings.page === "mic") renderMic();
    else navigate("mic");
    requestAnimationFrame(() => document.getElementById("programs")?.scrollIntoView({ block: "center" }));
  });
  $("#support-open").addEventListener("click", openSupport);
  $("#support-open").addEventListener("mouseenter", () => {
    shineSupport();
    playMascot();
  });
  $$<HTMLButtonElement>(".support-actions .btn").forEach((btn) => {
    btn.addEventListener("mouseenter", () => {
      btn.classList.add("charge");
      if (btn.id !== "support-later") {
        window.clearTimeout(Number(btn.dataset.urge));
        btn.dataset.urge = String(window.setTimeout(() => btn.classList.add("urge"), 1400));
      }
    });
    btn.addEventListener("mouseleave", () => {
      window.clearTimeout(Number(btn.dataset.urge));
      btn.classList.remove("charge", "urge");
    });
  });
  const later = $<HTMLButtonElement>("#support-later");
  const card = $(".support-card");
  later.addEventListener("click", closeSupport);
  later.addEventListener("mouseenter", () => {
    setPeek(true);
    later.textContent = LATER_LINES[laterLine % LATER_LINES.length]!;
    laterLine += 1;
  });
  later.addEventListener("mouseleave", () => {
    later.textContent = "Agora não";
    if (card.matches(":hover")) setPeek(false);
  });
  card.addEventListener("mouseleave", () => setPeek(true));
  card.addEventListener("mouseenter", () => {
    if (!later.matches(":hover")) setPeek(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || $("#support-backdrop").hidden) return;
    e.preventDefault();
    setPeek(true);
    window.setTimeout(() => {
      if (card.matches(":hover") && !later.matches(":hover")) setPeek(false);
    }, 900);
  });
  $("#pix-copy").addEventListener("click", copyPixKey);
  $("#pix-payload").addEventListener("click", () =>
    copyPix($<HTMLButtonElement>("#pix-payload"), PIX_EMV, "Pix copiado! O camaleão agradece", "Copiar Pix copia e cola"),
  );
  const mascot = $<HTMLVideoElement>("#support-mascot");
  mascot.addEventListener("ended", () => scheduleMascot());
  window.setTimeout(() => playMascot(), 1200);
  let usedMark = Date.now();
  window.setInterval(() => {
    const now = Date.now();
    const delta = now - usedMark;
    usedMark = now;
    if (document.hidden || !settings.params.enabled) return;
    settings.usedMs += delta;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, 30000);
  $("#power-now").addEventListener("click", () => navigate(aiRunning() ? "ai" : "voices"));
  $("#ptt-mic").addEventListener("click", () => {
    if (settings.ptt.enabled) return;
    settings.micMuted = !settings.micMuted;
    save();
    syncPtt();
  });

  listen<Progress>("ai-download", (e) => onDownload(e.payload));
  listen<TrayAction>(TRAY_ACTION, (e) => onTrayAction(e.payload));
  listen<string>("hotkey", (e) => runHotkey(e.payload));
  listen<boolean>("ptt-state", (e) => {
    pttOpen = e.payload;
    paintPtt();
  });
  syncPtt();
  invoke("set_startup", { enabled: settings.startWithWindows }).catch(() => {});
  listen(TRAY_HELLO, () => sendTrayState());

  const openGithub = () => openUrl("https://github.com/Konnak").catch((e) => toast(String(e)));
  $("#credit-link").addEventListener("mouseenter", scrambleCredit);
  $("#credit-link").addEventListener("mouseleave", resetCredit);
  $("#credit-open").addEventListener("click", openGithub);
  $("#credit-github").addEventListener("click", openGithub);
  $("#app-version").addEventListener("click", openChangelog);
  $("#changelog-close").addEventListener("click", closeChangelog);
  $("#changelog-backdrop").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeChangelog();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || $("#changelog-backdrop").hidden) return;
    e.preventDefault();
    closeChangelog();
  });
  $("#cal-close").addEventListener("click", closeCalibration);
  $("#cal-backdrop").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeCalibration();
  });
  $("#cal-start").addEventListener("click", () => void startCalibration());
  $("#cal-again").addEventListener("click", () => {
    stopCalibration();
    $("#cal-result").hidden = true;
    $("#cal-read").hidden = false;
    $("#cal-time").textContent = "00:00";
    const start = $<HTMLButtonElement>("#cal-start");
    start.disabled = false;
    start.textContent = "Iniciar";
    $("#cal-error").hidden = true;
  });
  const onCalSlider = () => {
    applyCalibration(Number($<HTMLInputElement>("#cal-tom").value), Number($<HTMLInputElement>("#cal-ent").value));
    paintCalValues();
  };
  $("#cal-tom").addEventListener("input", onCalSlider);
  $("#cal-ent").addEventListener("input", onCalSlider);
  $("#check-update").addEventListener("click", () => {
    if (pendingUpdate?.download) askToInstall(pendingUpdate);
    else void lookForUpdate(true);
  });

  splashProgress(18, "Carregando suas vozes…");
  await waitForBackend();
  navigate(settings.page);
  await invoke("set_params", { params: settings.params }).catch(() => {});
  splashProgress(52, "Procurando microfones…");
  await refreshDevices();
  splashProgress(78, aiRequest() ? "Preparando a voz IA…" : "Ligando o áudio…");
  const engine = restartEngine();
  pollLevels();
  await engine;
  splashProgress(100, "Pronto");
  void lookForUpdate(false);
  await hideSplash();
});
