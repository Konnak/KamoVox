//! Global shortcuts. A private thread owns the Windows hotkey registrations.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Emitter};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    RegisterHotKey, UnregisterHotKey, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, MOD_WIN,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{PeekMessageW, MSG, PM_REMOVE, WM_HOTKEY};

pub struct Hotkeys {
    tx: mpsc::Sender<Job>,
}

struct Job {
    binds: Vec<(String, String)>,
    reply: mpsc::Sender<Vec<String>>,
}

impl Hotkeys {
    pub fn start(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        thread::Builder::new()
            .name("kamovox-hotkey".into())
            .spawn(move || run(app, rx))
            .expect("hotkey thread");
        Self { tx }
    }

    pub fn set(&self, binds: Vec<(String, String)>) -> Result<Vec<String>, String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx.send(Job { binds, reply: reply_tx }).map_err(|e| e.to_string())?;
        reply_rx.recv_timeout(Duration::from_secs(2)).map_err(|_| "O registro de atalhos não respondeu".to_string())
    }
}

fn run(app: AppHandle, rx: mpsc::Receiver<Job>) {
    let mut registered: Vec<i32> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    loop {
        let job = match registered.is_empty() {
            true => match rx.recv() {
                Ok(j) => j,
                Err(_) => break,
            },
            false => match rx.try_recv() {
                Ok(j) => j,
                Err(mpsc::TryRecvError::Empty) => {
                    pump(&app, &names);
                    thread::sleep(Duration::from_millis(12));
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            },
        };
        for id in &registered {
            unsafe { UnregisterHotKey(std::ptr::null_mut(), *id) };
        }
        registered.clear();
        names.clear();
        let mut failed = Vec::new();
        let mut seen = Vec::new();
        for (name, combo) in job.binds {
            if combo.is_empty() || seen.iter().any(|c| c == &combo) {
                if !combo.is_empty() {
                    failed.push(format!("{combo} está em mais de um atalho"));
                }
                continue;
            }
            let Ok((mods, vk)) = parse(&combo) else {
                failed.push(format!("{combo} não é uma tecla válida"));
                continue;
            };
            let id = (registered.len() + 1) as i32;
            let ok = unsafe { RegisterHotKey(std::ptr::null_mut(), id, mods, vk) } != 0;
            if ok {
                registered.push(id);
                names.push(name);
                seen.push(combo);
            } else {
                failed.push(format!("{combo} já está em uso por outro programa"));
            }
        }
        let _ = job.reply.send(failed);
    }
    for id in &registered {
        unsafe { UnregisterHotKey(std::ptr::null_mut(), *id) };
    }
}

fn pump(app: &AppHandle, names: &[String]) {
    let mut msg = unsafe { std::mem::zeroed::<MSG>() };
    while unsafe { PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) } != 0 {
        if msg.message == WM_HOTKEY {
            let id = msg.wParam as usize;
            if id >= 1 && id <= names.len() {
                let _ = app.emit("hotkey", &names[id - 1]);
            }
        }
    }
}

fn parse(combo: &str) -> Result<(u32, u32), ()> {
    let mut mods = MOD_NOREPEAT;
    let mut vk = 0u32;
    for part in combo.split('+').map(str::trim).filter(|p| !p.is_empty()) {
        match part {
            "Ctrl" => mods |= MOD_CONTROL,
            "Alt" => mods |= MOD_ALT,
            "Shift" => mods |= MOD_SHIFT,
            "Win" => mods |= MOD_WIN,
            key => vk = virtual_key(key).ok_or(())?,
        }
    }
    if vk == 0 { Err(()) } else { Ok((mods, vk)) }
}

pub(crate) fn virtual_key(key: &str) -> Option<u32> {
    let bytes = key.as_bytes();
    if bytes.len() == 1 {
        let c = bytes[0].to_ascii_uppercase();
        if c.is_ascii_alphanumeric() {
            return Some(c as u32);
        }
    }
    if let Some(n) = key.strip_prefix('F').and_then(|n| n.parse::<u32>().ok()) {
        if (1..=24).contains(&n) {
            return Some(0x70 + n - 1);
        }
    }
    Some(match key {
        "CapsLock" | "caps_lock" | "Capital" => 0x14,
        "Space" | " " => 0x20,
        "Enter" => 0x0D,
        "Tab" => 0x09,
        "Esc" | "Escape" => 0x1B,
        "Left" | "ArrowLeft" => 0x25,
        "Up" | "ArrowUp" => 0x26,
        "Right" | "ArrowRight" => 0x27,
        "Down" | "ArrowDown" => 0x28,
        "Insert" => 0x2D,
        "Delete" => 0x2E,
        "Home" => 0x24,
        "End" => 0x23,
        "PageUp" => 0x21,
        "PageDown" => 0x22,
        _ => return None,
    })
}
