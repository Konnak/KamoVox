import "@fontsource-variable/inter";
import { invoke } from "@tauri-apps/api/core";
import { emitTo, listen } from "@tauri-apps/api/event";
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import { icon } from "./icons";
import { TRAY_ACTION, TRAY_HELLO, TRAY_STATE, type TrayAction, type TrayState } from "./tray-protocol";

const WIDTH = 312;
const win = getCurrentWindow();
const menu = document.getElementById("menu")!;
let state: TrayState | null = null;
let height = 0;

const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

function send(action: TrayAction) {
  emitTo("main", TRAY_ACTION, action).catch(() => {});
}

function toggleRow(kind: "ai" | "windowsMic" | "monitor", iconName: string, title: string, hint: string, on: boolean) {
  return `
    <button class="item" role="menuitemcheckbox" aria-checked="${on}" data-toggle="${kind}">
      <span class="badge">${icon(iconName)}</span>
      <span class="text"><b>${title}</b><small>${esc(hint)}</small></span>
      <span class="switch ${on ? "on" : ""}"><span></span></span>
    </button>`;
}

function render() {
  if (!state) {
    menu.innerHTML = `<div class="loading">Carregando…</div>`;
    return fit();
  }
  const s = state;
  const voiceCard = (kind: "ai" | "classic" | "plain", art: string, hint: string, title: string) => `
    <div class="now ${s.enabled ? "" : "muted"} ${kind}">
      <span class="art">${art}</span>
      <span class="text"><small>${esc(hint)}</small><b>${esc(title)}</b></span>
    </div>`;
  const aiArt = s.ai.image ? `<img src="${s.ai.image}" alt="" />` : icon("ai");
  const cards = [
    s.ai.on ? voiceCard("ai", aiArt, "Voz IA ligada", s.ai.name ?? "Voz IA") : "",
    s.classic.on ? voiceCard("classic", icon("voices"), "Vozes prontas", s.classic.summary) : "",
    !s.ai.on && !s.classic.on ? voiceCard("plain", icon("voices"), s.enabled ? "Você está falando como" : "Trocador desligado", "Voz natural") : "",
  ].join("");
  menu.innerHTML = `
    <header class="head card-head">
      <div class="brand">
        <b>Trocador de voz</b>
        <small class="profile-now">${s.profile ? `Perfil · ${esc(s.profile)}` : "Nenhum perfil aplicado"}</small>
        ${s.status ? `<small class="status ${s.error ? "err" : ""}"><i></i>${esc(s.status)}</small>` : ""}
      </div>
      <div class="powers">
        <button class="round hear ${s.monitor ? "on" : ""}" id="hear" title="Ouvir a própria voz" aria-pressed="${s.monitor}">${icon("headphones")}</button>
        <button class="round mic ${s.talking ? "live" : ""} ${s.micMuted ? "muted" : ""}" id="mic" title="${s.ptt ? "Push-to-talk" : s.micMuted ? "Microfone mutado" : "Microfone aberto"}" aria-pressed="${s.micMuted}">${icon("mic")}</button>
        <button class="round power ${s.enabled ? "on" : ""}" id="power" role="menuitemcheckbox" aria-checked="${s.enabled}"
          aria-label="${s.enabled ? "Desligar" : "Ligar"} o trocador de voz" title="${s.enabled ? "Desligar" : "Ligar"} o trocador de voz">
          ${icon("power")}
        </button>
      </div>
    </header>

    ${cards}
    ${s.stats ? `<div class="stats">${esc(s.stats)}</div>` : ""}
    <div class="meter" title="Voz que os programas ouvem"><div id="level"></div></div>

    <div class="list">
      ${toggleRow("ai", "ai", "Voz IA", s.ai.on ? (s.ai.name ?? "Ligada") : s.ai.available ? "Desligada" : "Escolha uma voz no app", s.ai.on)}
      ${s.windowsMic.available ? toggleRow("windowsMic", "link", "Voz nos programas", "Chamadas e jogos ouvem a voz alterada", s.windowsMic.on) : ""}
    </div>

    ${
      s.profiles.length
        ? `<div class="section">Perfis</div>
           <div class="chips">${s.profiles.map((p, i) => `<button class="chip ${p === s.profile ? "picked" : ""}" data-profile="${i}">${esc(p)}</button>`).join("")}</div>`
        : ""
    }

    <footer class="foot">
      <button class="action primary" id="open" role="menuitem">${icon("window")}Abrir KamoVox</button>
      <button class="action quit" id="quit" role="menuitem" title="Fecha o KamoVox e volta o microfone ao normal">${icon("exit")}Sair</button>
    </footer>`;

  document.getElementById("hear")!.addEventListener("click", () => {
    s.monitor = !s.monitor;
    send({ kind: "monitor", on: s.monitor });
    render();
  });
  document.getElementById("mic")!.addEventListener("click", () => {
    if (s.ptt) return;
    s.micMuted = !s.micMuted;
    s.talking = !s.micMuted;
    send({ kind: "micMute" });
    render();
  });
  document.getElementById("power")!.addEventListener("click", () => {
    s.enabled = !s.enabled;
    send({ kind: "power", on: s.enabled });
    render();
  });
  menu.querySelectorAll<HTMLButtonElement>("[data-toggle]").forEach((b) =>
    b.addEventListener("click", () => {
      const kind = b.dataset.toggle as "ai" | "windowsMic" | "monitor";
      const next = kind === "ai" ? !s.ai.on : kind === "windowsMic" ? !s.windowsMic.on : !s.monitor;
      if (kind === "ai") {
        if (!s.ai.available) return void send({ kind, on: true });
        s.ai.on = next;
      } else if (kind === "windowsMic") s.windowsMic.on = next;
      else s.monitor = next;
      send({ kind, on: next });
      render();
    }),
  );
  menu.querySelectorAll<HTMLButtonElement>("[data-profile]").forEach((b) =>
    b.addEventListener("click", () => {
      send({ kind: "profile", index: Number(b.dataset.profile) });
      b.classList.add("picked");
    }),
  );
  document.getElementById("open")!.addEventListener("click", () => invoke("show_main"));
  document.getElementById("quit")!.addEventListener("click", () => invoke("quit_app"));
  fit();
}

/** Resizes the window to the menu so the tray positions it correctly. */
function fit() {
  const h = Math.ceil(menu.getBoundingClientRect().height);
  if (h && h !== height) {
    height = h;
    win.setSize(new LogicalSize(WIDTH, h)).catch(() => {});
  }
}

let visible = false;
async function pollLevel() {
  if (!visible) return;
  try {
    const lv = await invoke<{ output: number }>("get_levels");
    const db = lv.output <= 0 ? -60 : 20 * Math.log10(lv.output);
    const el = document.getElementById("level");
    if (el) el.style.width = `${Math.max(0, Math.min(1, (db + 60) / 60)) * 100}%`;
  } catch {
    /* engine busy */
  }
  window.setTimeout(pollLevel, 70);
}

function opened() {
  menu.classList.remove("enter");
  void menu.offsetWidth;
  menu.classList.add("enter");
  emitTo("main", TRAY_HELLO).catch(() => {});
  if (!visible) {
    visible = true;
    pollLevel();
  }
}

function close() {
  visible = false;
  invoke("hide_tray_menu");
}

listen<TrayState>(TRAY_STATE, (e) => {
  state = e.payload;
  render();
});
listen("tray-open", opened);
win.onFocusChanged(({ payload: focused }) => {
  if (!focused) close();
});
document.addEventListener("keydown", (e) => e.key === "Escape" && close());
document.addEventListener("contextmenu", (e) => e.preventDefault());

render();
emitTo("main", TRAY_HELLO).catch(() => {});
