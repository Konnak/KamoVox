mod apo;
mod audio;
mod endpoint;
mod hotkey;
mod library;
mod priority;
mod ptt;
mod sound;
mod tray;
mod update;

/// Elevated helper mode (`--apo-install` / `--apo-uninstall`); `None` for a normal launch.
pub fn apo_cli() -> Option<i32> {
    apo::run_cli()
}

use std::path::PathBuf;
use std::sync::Arc;

use audio::{AiRun, DeviceList, Engine, Levels, StartConfig, Status};
use kamovox_ai::{Device, LoadKey, PitchMethod, Timing};
use kamovox_dsp::Params;
use library::{Library, LibraryState};
use serde::Deserialize;
use tauri::{AppHandle, Manager, State};

struct AppState {
    engine: Engine,
    library: Arc<Library>,
    hotkeys: hotkey::Hotkeys,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiRequest {
    voice: String,
    #[serde(default)]
    speaker: usize,
    device: Device,
    pitch: PitchMethod,
    timing: Timing,
    transpose: f32,
}

#[tauri::command]
fn list_devices() -> Result<DeviceList, String> {
    audio::list_devices()
}

fn resources_dir(app: &AppHandle) -> PathBuf {
    let bundled = app.path().resource_dir().map(|d| d.join("resources").join("rvc")).ok();
    match bundled {
        Some(d) if d.exists() => d,
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("rvc"),
    }
}

#[tauri::command]
async fn start_engine(app: AppHandle, state: State<'_, AppState>, config: StartConfig, ai: Option<AiRequest>) -> Result<Status, String> {
    let ai = match ai {
        Some(r) => {
            let lib = &state.library;
            let voice = lib.voice_path(&r.voice).ok_or("Voz IA não encontrada; baixe ou importe novamente")?;
            Some(AiRun {
                base: lib.base_dir(),
                resources: resources_dir(&app),
                key: LoadKey { voice, speaker: r.speaker, device: r.device, pitch: r.pitch },
                timing: r.timing,
                transpose: r.transpose,
            })
        }
        None => None,
    };
    state.engine.set_speaker_model(state.library.speaker_path());
    let engine = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || engine.start(config, ai)).await.map_err(|e| e.to_string())?
}

#[tauri::command]
async fn stop_engine(state: State<'_, AppState>) -> Result<Status, String> {
    let engine = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || engine.stop()).await.map_err(|e| e.to_string())
}

#[tauri::command]
fn set_params(state: State<AppState>, params: Params) {
    state.engine.set_params(params);
}

#[tauri::command]
fn set_ai_pitch(state: State<AppState>, semitones: f32) {
    state.engine.set_transpose(semitones);
}

#[derive(serde::Serialize)]
struct CalibrationCapture {
    samples: Vec<f32>,
    rate: u32,
}

#[tauri::command]
fn calibration_begin(state: State<AppState>) -> Result<(), String> {
    state.engine.begin_calibration()
}

#[tauri::command]
fn calibration_end(state: State<AppState>) -> CalibrationCapture {
    let (samples, rate) = state.engine.end_calibration();
    CalibrationCapture { samples, rate }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct VoicePrintCapture {
    bands: Vec<f32>,
    pitch: f32,
    pitch_lo: f32,
    pitch_hi: f32,
    level: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    frames: u32,
    /// CAM++ enrollment embedding when the speaker model is installed.
    embed: Vec<f32>,
    /// Extra CAM++ templates (N×512).
    gallery: Vec<f32>,
    neural: bool,
    neural_lo: f32,
    neural_hi: f32,
    neural_reject: f32,
    self_p05: f32,
    self_p10: f32,
    self_median: f32,
}

#[tauri::command]
async fn voice_print_end(state: State<'_, AppState>) -> Result<VoicePrintCapture, String> {
    let engine = state.engine.clone();
    let speaker = state.library.speaker_path();
    tauri::async_runtime::spawn_blocking(move || voice_print_end_blocking(engine, speaker))
        .await
        .map_err(|e| format!("Biometria interrompida: {e}"))?
}

fn voice_print_end_blocking(engine: Engine, speaker: Option<PathBuf>) -> Result<VoicePrintCapture, String> {
    let (samples, rate) = engine.end_calibration();
    let print = kamovox_dsp::build_voice_print(&samples, rate as f32).ok_or(
        "Não ouvi voz suficiente. Fale por uns 15 segundos, perto do microfone, no volume normal, com o ambiente em silêncio.",
    )?;
    struct ResumeNeural(Engine);
    impl Drop for ResumeNeural {
        fn drop(&mut self) {
            self.0.pause_neural(false);
        }
    }
    let (embed, gallery, neural, neural_lo, neural_hi, neural_reject, self_p05, self_p10, self_median) =
        match speaker {
            Some(path) => {
                engine.pause_neural(true);
                let _resume = ResumeNeural(engine.clone());
                let wav16 = kamovox_ai::to_16k(&samples, rate);
                match kamovox_ai::SpeakerEmbedder::load(&path).and_then(|mut e| e.embed_enroll(&wav16)) {
                    Ok(r) => (
                        r.embed,
                        kamovox_ai::flatten_gallery(&r.gallery),
                        true,
                        r.thresh_lo,
                        r.thresh_hi,
                        r.thresh_reject,
                        r.self_p05,
                        r.self_p10,
                        r.self_median,
                    ),
                    Err(e) => return Err(format!("Biometria neural: {e}")),
                }
            }
            None => (Vec::new(), Vec::new(), false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
    Ok(VoicePrintCapture {
        bands: print.bands.to_vec(),
        pitch: print.pitch,
        pitch_lo: print.pitch_lo,
        pitch_hi: print.pitch_hi,
        level: print.level,
        f1: print.f1,
        f2: print.f2,
        f3: print.f3,
        frames: print.frames,
        embed,
        gallery,
        neural,
        neural_lo,
        neural_hi,
        neural_reject,
        self_p05,
        self_p10,
        self_median,
    })
}

#[tauri::command]
fn get_levels(state: State<AppState>) -> Levels {
    state.engine.levels()
}

#[tauri::command]
fn ai_library(state: State<AppState>) -> LibraryState {
    state.library.state()
}

#[tauri::command]
fn ai_download(app: AppHandle, state: State<AppState>, id: String) -> Result<(), String> {
    state.library.download(app, id)
}

#[tauri::command]
fn ai_cancel(state: State<AppState>, id: String) {
    state.library.cancel(&id);
}

#[tauri::command]
async fn ai_import(state: State<'_, AppState>, path: String) -> Result<LibraryState, String> {
    let lib = state.library.clone();
    tauri::async_runtime::spawn_blocking(move || {
        lib.import(&PathBuf::from(path))?;
        Ok(lib.state())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(serde::Serialize)]
struct ScanResult {
    added: usize,
    failed: Vec<String>,
    library: LibraryState,
}

#[tauri::command]
async fn ai_rescan(state: State<'_, AppState>) -> Result<ScanResult, String> {
    let lib = state.library.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (added, failed) = lib.scan();
        ScanResult { added, failed, library: lib.state() }
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn ai_remove(state: State<AppState>, id: String) -> Result<LibraryState, String> {
    state.library.remove(&id)?;
    Ok(state.library.state())
}

#[tauri::command]
fn ai_set_models_dir(state: State<AppState>, path: String) -> Result<LibraryState, String> {
    state.library.set_dir(PathBuf::from(path))?;
    Ok(state.library.state())
}

#[tauri::command]
fn ai_update_voice(state: State<AppState>, id: String, name: String, image: Option<String>) -> Result<LibraryState, String> {
    state.library.update(&id, &name, image.as_deref())?;
    Ok(state.library.state())
}

#[tauri::command]
fn show_main(app: AppHandle) {
    tray::show_main(&app);
}

#[tauri::command]
fn hide_tray_menu(app: AppHandle) {
    tray::hide_menu(&app);
}

#[tauri::command]
fn quit_app(app: AppHandle, state: State<AppState>) {
    state.engine.stop();
    app.exit(0);
}

#[tauri::command]
fn mic_endpoints() -> Vec<apo::Endpoint> {
    apo::endpoints()
}

/// Installs (or removes) KamoVox on a Windows microphone. The engine must be
/// stopped first: the Windows audio service restarts during the change.
#[tauri::command]
async fn mic_integration(app: AppHandle, state: State<'_, AppState>, endpoint: String, install: bool) -> Result<Vec<apo::Endpoint>, String> {
    let engine = state.engine.clone();
    let dll = if install {
        Some(apo::find_dll(app.path().resource_dir().ok()).ok_or("O filtro de áudio do KamoVox não foi encontrado na instalação")?)
    } else {
        None
    };
    tauri::async_runtime::spawn_blocking(move || {
        engine.stop();
        match &dll {
            Some(dll) => apo::elevate(&["--apo-install", &endpoint, &dll.to_string_lossy()])?,
            None => apo::elevate(&["--apo-uninstall", &endpoint])?,
        }
        Ok(apo::endpoints())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Deserialize)]
struct HotkeyBind {
    id: String,
    combo: String,
}

#[tauri::command]
fn set_hotkeys(state: State<AppState>, binds: Vec<HotkeyBind>) -> Result<Vec<String>, String> {
    state.hotkeys.set(binds.into_iter().map(|b| (b.id, b.combo)).collect())
}

#[tauri::command]
fn play_sound(state: State<AppState>, id: String, path: String) -> Result<(), String> {
    state.engine.play_sound(&id, std::path::Path::new(&path))
}

#[tauri::command]
fn set_ptt(enabled: bool, inverted: bool, key: String) -> Result<(), String> {
    ptt::configure(enabled, inverted, &key)
}

#[tauri::command]
fn set_mic_mute(muted: bool) {
    ptt::set_held_mute(muted);
}

#[tauri::command]
fn set_tray_talking(app: tauri::AppHandle, talking: bool) {
    tray::set_talking(&app, talking);
}

#[tauri::command]
fn check_update() -> Result<update::UpdateInfo, String> {
    update::check()
}

#[tauri::command]
fn install_update(app: AppHandle, url: String) -> Result<(), String> {
    update::install(app, url)
}

#[tauri::command]
fn set_startup(enabled: bool) -> Result<(), String> {
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegSetValueExW, HKEY_CURRENT_USER, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE,
        REG_SZ,
    };
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let path = wide(r"Software\Microsoft\Windows\CurrentVersion\Run");
    let name = wide("KamoVox");
    unsafe {
        let mut key = std::ptr::null_mut();
        let created = RegCreateKeyExW(
            HKEY_CURRENT_USER,
            path.as_ptr(),
            0,
            std::ptr::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_SET_VALUE,
            std::ptr::null(),
            &mut key,
            std::ptr::null_mut(),
        );
        if created != 0 {
            return Err("Não foi possível abrir a inicialização do Windows".into());
        }
        let status = if enabled {
            let value = wide(&exe.display().to_string());
            RegSetValueExW(key, name.as_ptr(), 0, REG_SZ, value.as_ptr() as *const u8, (value.len() * 2) as u32)
        } else {
            let removed = RegDeleteValueW(key, name.as_ptr());
            if removed == 2 { 0 } else { removed }
        };
        RegCloseKey(key);
        if status != 0 {
            return Err(format!("Não foi possível salvar a inicialização ({status})"));
        }
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let library = Arc::new(Library::new(app.handle()));
            let hotkeys = hotkey::Hotkeys::start(app.handle().clone());
            let engine = Engine::new();
            ptt::start(app.handle().clone(), engine.clone());
            app.manage(AppState { engine, library, hotkeys });
            priority::keep_process_responsive();
            tray::setup(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the main window keeps KamoVox running in the tray.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            list_devices,
            start_engine,
            stop_engine,
            set_params,
            set_ai_pitch,
            calibration_begin,
            calibration_end,
            voice_print_end,
            get_levels,
            ai_library,
            ai_download,
            ai_cancel,
            ai_import,
            ai_remove,
            ai_set_models_dir,
            ai_rescan,
            mic_endpoints,
            mic_integration,
            ai_update_voice,
            show_main,
            hide_tray_menu,
            quit_app,
            set_hotkeys,
            play_sound,
            set_ptt,
            set_mic_mute,
            set_tray_talking,
            set_startup,
            check_update,
            install_update
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
