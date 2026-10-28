/** Messages between the main window and the tray menu window. */

export interface TrayState {
  enabled: boolean;
  /** Shown only for an error or while a voice is loading. */
  status: string;
  error: boolean;
  /** Short description of the current voice, e.g. "Robô · Eco". */
  voice: string;
  /** One line, same numbers as the sidebar card. */
  stats: string;
  ai: { available: boolean; on: boolean; name: string | null; image: string | null };
  /** Ready voices or tom/timbre are in use, even while the AI voice is on. */
  classic: { on: boolean; summary: string };
  windowsMic: { available: boolean; on: boolean };
  monitor: boolean;
  /** Push-to-talk owns the microphone button. */
  ptt: boolean;
  /** Voice is going out right now. */
  talking: boolean;
  /** Click-mute, only while push-to-talk is off. */
  micMuted: boolean;
  /** Name of the saved profile that matches the current voice, if any. */
  profile: string | null;
  profiles: string[];
}

export type TrayAction =
  | { kind: "power" | "ai" | "windowsMic" | "monitor"; on: boolean }
  | { kind: "micMute" }
  | { kind: "profile"; index: number };

export const TRAY_STATE = "tray-state";
export const TRAY_ACTION = "tray-action";
/** Sent by the tray window when it needs a fresh state. */
export const TRAY_HELLO = "tray-hello";
