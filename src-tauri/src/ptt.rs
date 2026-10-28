//! Push-to-talk that works even when KamoVox is not the focused window.
//!
//! The chosen key is only observed. Caps Lock keeps its normal job of
//! switching upper and lower case; KamoVox only needs to know if it is held.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Emitter};
use windows_sys::Win32::Foundation::{GetLastError, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RIDEV_INPUTSINK,
    RID_INPUT, RIM_TYPEKEYBOARD,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, PeekMessageW, RegisterClassW,
    SetWindowsHookExW, TranslateMessage, HWND_MESSAGE, KBDLLHOOKSTRUCT, PM_NOREMOVE, WH_KEYBOARD_LL, WINDOWS_HOOK_ID,
    WM_INPUT, WNDCLASSW,
};

use crate::audio::Engine;
use crate::hotkey::virtual_key;

const LLKHF_UP: u32 = 0x80;
const RI_KEY_BREAK: u16 = 1;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INVERTED: AtomicBool = AtomicBool::new(false);
static VK: AtomicU32 = AtomicU32::new(0x14);
static DOWN: AtomicBool = AtomicBool::new(false);
static OPEN: AtomicBool = AtomicBool::new(true);
/// Click-mute used only while push-to-talk is off. The key owns the gate otherwise.
static HELD_MUTE: AtomicBool = AtomicBool::new(false);
static ENGINE: OnceLock<Engine> = OnceLock::new();
static APP: OnceLock<AppHandle> = OnceLock::new();

fn trace(msg: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(r"C:\Users\Cleison Carvalho\AppData\Local\Temp\kamovox-ptt.log") {
        let _ = writeln!(file, "{msg}");
    }
}

pub fn start(app: AppHandle, engine: Engine) {
    let _ = ENGINE.set(engine);
    let _ = APP.set(app);
    let _ = std::fs::remove_file(r"C:\Users\Cleison Carvalho\AppData\Local\Temp\kamovox-ptt.log");
    trace("ptt start");
    thread::Builder::new().name("kamovox-ptt-hook".into()).spawn(hook_thread).expect("ptt hook");
    thread::Builder::new().name("kamovox-ptt-raw".into()).spawn(input_thread).expect("ptt raw");
    thread::Builder::new().name("kamovox-ptt".into()).spawn(apply_thread).expect("ptt apply");
}

/// `key` is a name such as "CapsLock" or "Space".
pub fn configure(enabled: bool, inverted: bool, key: &str) -> Result<(), String> {
    let vk = virtual_key(key).ok_or_else(|| format!("Tecla não reconhecida: {key}"))?;
    VK.store(vk, Ordering::Relaxed);
    INVERTED.store(inverted, Ordering::Relaxed);
    ENABLED.store(enabled, Ordering::Relaxed);
    DOWN.store(false, Ordering::Relaxed);
    trace(&format!("configure enabled={enabled} inverted={inverted} key={key} vk={vk}"));
    let open = if enabled { inverted } else { !HELD_MUTE.load(Ordering::Relaxed) };
    publish(open);
    Ok(())
}

pub fn set_held_mute(muted: bool) {
    HELD_MUTE.store(muted, Ordering::Relaxed);
    if !ENABLED.load(Ordering::Relaxed) {
        publish(!muted);
    }
}

fn publish(open: bool) {
    if OPEN.swap(open, Ordering::AcqRel) == open {
        return;
    }
    if let Some(engine) = ENGINE.get() {
        engine.set_ptt_transmit(open);
    }
    if open {
        crate::endpoint::unmute_captures();
    }
    if let Some(app) = APP.get() {
        let _ = app.emit("ptt-state", open);
    }
    trace(&format!("state open={open}"));
}

fn hook_thread() {
    unsafe {
        let mut msg = std::mem::zeroed();
        PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_NOREMOVE);
        let hook = SetWindowsHookExW(WH_KEYBOARD_LL as WINDOWS_HOOK_ID, Some(hook_proc), std::ptr::null_mut(), 0);
        trace(&format!("hook={hook:?} err={}", GetLastError()));
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

fn input_thread() {
    unsafe {
        let class = wide("KamoVoxPtt");
        let instance = GetModuleHandleW(std::ptr::null());
        let wc = WNDCLASSW {
            style: 0,
            lpfnWndProc: Some(wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: instance,
            hIcon: std::ptr::null_mut(),
            hCursor: std::ptr::null_mut(),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: class.as_ptr(),
        };
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(0, class.as_ptr(), std::ptr::null(), 0, 0, 0, 0, 0, HWND_MESSAGE, std::ptr::null_mut(), instance, std::ptr::null());
        if !hwnd.is_null() {
            let device = RAWINPUTDEVICE {
                usUsagePage: 0x01,
                usUsage: 0x06,
                dwFlags: RIDEV_INPUTSINK,
                hwndTarget: hwnd,
            };
            let ok = RegisterRawInputDevices(&device, 1, std::mem::size_of::<RAWINPUTDEVICE>() as u32);
            trace(&format!("raw hwnd={hwnd:?} ok={ok} err={}", GetLastError()));
        } else {
            trace("raw window failed");
        }
        let mut msg = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

fn note_key(vk: u32, up: bool) {
    if ENABLED.load(Ordering::Relaxed) && vk == VK.load(Ordering::Relaxed) {
        DOWN.store(!up, Ordering::Relaxed);
    }
}

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_INPUT {
        unsafe { read_raw(lparam) };
        return 0;
    }
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

unsafe fn read_raw(lparam: LPARAM) {
    unsafe {
        let handle = lparam as HRAWINPUT;
        let mut size = 0u32;
        let header = std::mem::size_of::<RAWINPUTHEADER>() as u32;
        GetRawInputData(handle, RID_INPUT, std::ptr::null_mut(), &mut size, header);
        if size == 0 {
            return;
        }
        let mut buf = vec![0u8; size as usize];
        if GetRawInputData(handle, RID_INPUT, buf.as_mut_ptr().cast(), &mut size, header) == u32::MAX {
            return;
        }
        let raw = &*(buf.as_ptr() as *const RAWINPUT);
        if raw.header.dwType != RIM_TYPEKEYBOARD {
            return;
        }
        let key = raw.data.keyboard;
        if key.VKey == 0 || key.VKey == 0xFF {
            return;
        }
        note_key(key.VKey as u32, key.Flags & RI_KEY_BREAK != 0);
    }
}

/// Observes the key and always lets it continue, so Caps Lock still changes case.
unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code >= 0 {
        let info = unsafe { &*(lparam as *const KBDLLHOOKSTRUCT) };
        let up = wparam == 0x0101 || wparam == 0x0105 || (info.flags & LLKHF_UP) != 0;
        note_key(info.vkCode, up);
    }
    unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) }
}

fn apply_thread() {
    let mut ticks = 0u32;
    loop {
        let enabled = ENABLED.load(Ordering::Relaxed);
        let inverted = INVERTED.load(Ordering::Relaxed);
        let down = DOWN.load(Ordering::Relaxed);
        let next = if enabled {
            if down { !inverted } else { inverted }
        } else {
            !HELD_MUTE.load(Ordering::Relaxed)
        };
        publish(next);
        if next {
            ticks += 1;
            if ticks == 1 || ticks % 60 == 0 {
                crate::endpoint::unmute_captures();
            }
        } else {
            ticks = 0;
        }
        thread::sleep(Duration::from_millis(8));
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}
