//! Notification-area icon. Left click opens KamoVox; right click opens the
//! quick menu, a small glass window (`tray.html`) placed next to the icon.

use std::sync::OnceLock;

use tauri::image::Image;
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition};

struct BaseIcon {
    px: Vec<u8>,
    w: u32,
    h: u32,
}

static BASE: OnceLock<BaseIcon> = OnceLock::new();

pub fn setup(app: &tauri::App) -> tauri::Result<()> {
    if let Some(icon) = app.default_window_icon() {
        let _ = BASE.set(BaseIcon { px: icon.rgba().to_vec(), w: icon.width(), h: icon.height() });
    }
    let mut builder = TrayIconBuilder::with_id("kamovox").tooltip("KamoVox").show_menu_on_left_click(false);
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder
        .on_tray_icon_event(|tray, event| {
            let app = tray.app_handle();
            if let TrayIconEvent::Click { button, button_state: MouseButtonState::Up, position, .. } = event {
                match button {
                    MouseButton::Left => show_main(app),
                    MouseButton::Right => show_menu(app, position),
                    _ => {}
                }
            }
        })
        .build(app)?;
    Ok(())
}

pub fn show_main(app: &AppHandle) {
    hide_menu(app);
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

pub fn hide_menu(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("tray") {
        let _ = w.hide();
    }
}

fn show_menu(app: &AppHandle, cursor: PhysicalPosition<f64>) {
    let Some(w) = app.get_webview_window("tray") else { return };
    let size = w.outer_size().unwrap_or_default();
    let (width, height) = (size.width as i32, size.height as i32);
    // Open above and to the left of the cursor (taskbar at the bottom), kept inside the work area.
    let mut x = cursor.x as i32 - width + 12;
    let mut y = cursor.y as i32 - height - 8;
    if let Ok(Some(monitor)) = app.monitor_from_point(cursor.x, cursor.y) {
        let area = monitor.work_area();
        let (left, top) = (area.position.x, area.position.y);
        let (right, bottom) = (left + area.size.width as i32, top + area.size.height as i32);
        x = x.clamp(left + 8, (right - width - 8).max(left + 8));
        y = y.clamp(top + 8, (bottom - height - 8).max(top + 8));
    }
    let _ = w.set_position(PhysicalPosition::new(x, y));
    let _ = w.show();
    let _ = w.set_focus();
    let _ = app.emit_to("tray", "tray-open", ());
}

/// Green ring around the notification icon while push-to-talk is open.
pub fn set_talking(app: &AppHandle, talking: bool) {
    let Some(tray) = app.tray_by_id("kamovox") else { return };
    let Some(base) = BASE.get() else { return };
    let px = if talking { ring(&base.px, base.w, base.h) } else { base.px.clone() };
    let _ = tray.set_icon(Some(Image::new_owned(px, base.w, base.h)));
}

fn ring(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut px = src.to_vec();
    let (wf, hf) = (w as f32, h as f32);
    let (cx, cy) = (wf / 2.0, hf / 2.0);
    let outer = cx.min(cy) - 0.5;
    let inner = (outer - (wf.max(32.0) / 16.0)).max(1.0);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= outer && d >= inner {
                let i = ((y * w + x) * 4) as usize;
                if i + 3 < px.len() {
                    px[i] = 48;
                    px[i + 1] = 209;
                    px[i + 2] = 88;
                    px[i + 3] = 255;
                }
            }
        }
    }
    px
}
