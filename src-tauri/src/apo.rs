//! Installs the KamoVox APO on a Windows microphone so every program that
//! records from it hears the KamoVox voice.
//!
//! Registry writes need administrator rights, so the app relaunches itself
//! elevated with `--apo-install` / `--apo-uninstall` (see [`run_cli`]).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;
use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
use windows_sys::Win32::System::Registry::*;

pub const CLSID: &str = "{7A3E5C21-4B9D-4E6F-A1C8-3D2F9B7E6A54}";
const NAME: &str = "KamoVox";
const CAPTURE: &str = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture";
const BACKUP: &str = r"SOFTWARE\KamoVox\Backup";
const INSTALL_DIR: &str = r"C:\Program Files\KamoVox";
const LOG_DIR: &str = r"C:\ProgramData\KamoVox";
const KEY_GFX: &str = "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},2";
const KEY_FX_NAME: &str = "{b725f130-47ef-101a-a5f1-02608c9eebac},10";
/// Effect slots that would take precedence over a GFX or that belong to
/// another effect; they are saved and cleared while KamoVox is installed.
const FX_KEYS: &[&str] = &[
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},0",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},1",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},2",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},3",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},5",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},6",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},7",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},13",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},14",
    "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},15",
    "{d3993a3f-99c2-4402-b5ec-a92a0367664b},5",
    "{d3993a3f-99c2-4402-b5ec-a92a0367664b},6",
    "{d3993a3f-99c2-4402-b5ec-a92a0367664b},7",
    "{b725f130-47ef-101a-a5f1-02608c9eebac},10",
];
const ABSENT: &str = "!KEY";

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Endpoint {
    pub id: String,
    /// Same text cpal shows for the device.
    pub name: String,
    pub installed: bool,
    /// Another effect currently on this microphone (e.g. "Clownfish APO").
    pub other: Option<String>,
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

struct Key(HKEY);

impl Drop for Key {
    fn drop(&mut self) {
        unsafe { RegCloseKey(self.0) };
    }
}

#[derive(Clone)]
enum Value {
    Text(u32, Vec<u16>),
    Dword(u32),
}

impl Key {
    fn open(path: &str, access: REG_SAM_FLAGS) -> Result<Key, u32> {
        let mut h = std::ptr::null_mut();
        let err = unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, wide(path).as_ptr(), 0, access | KEY_WOW64_64KEY, &mut h) };
        if err == ERROR_SUCCESS { Ok(Key(h)) } else { Err(err) }
    }

    fn create(path: &str) -> Result<Key, String> {
        let mut h = std::ptr::null_mut();
        let err = unsafe {
            RegCreateKeyExW(
                HKEY_LOCAL_MACHINE,
                wide(path).as_ptr(),
                0,
                std::ptr::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_ALL_ACCESS | KEY_WOW64_64KEY,
                std::ptr::null(),
                &mut h,
                std::ptr::null_mut(),
            )
        };
        if err == ERROR_SUCCESS { Ok(Key(h)) } else { Err(format!("registro {path}: erro {err}")) }
    }

    fn get(&self, name: &str) -> Option<Value> {
        let name = wide(name);
        let mut kind = 0;
        let mut size = 0u32;
        unsafe {
            if RegQueryValueExW(self.0, name.as_ptr(), std::ptr::null(), &mut kind, std::ptr::null_mut(), &mut size) != ERROR_SUCCESS {
                return None;
            }
            let mut buf = vec![0u8; size as usize + 2];
            if RegQueryValueExW(self.0, name.as_ptr(), std::ptr::null(), &mut kind, buf.as_mut_ptr(), &mut size) != ERROR_SUCCESS {
                return None;
            }
            buf.truncate(size as usize);
            if kind == REG_DWORD && buf.len() >= 4 {
                return Some(Value::Dword(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])));
            }
            let text = buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Some(Value::Text(kind, text))
        }
    }

    fn get_string(&self, name: &str) -> Option<String> {
        match self.get(name)? {
            Value::Text(_, t) => Some(String::from_utf16_lossy(&t).trim_end_matches('\0').to_string()),
            Value::Dword(_) => None,
        }
    }

    fn get_dword(&self, name: &str) -> Option<u32> {
        match self.get(name)? {
            Value::Dword(v) => Some(v),
            Value::Text(..) => None,
        }
    }

    fn set(&self, name: &str, value: &Value) -> Result<(), String> {
        let n = wide(name);
        let err = unsafe {
            match value {
                Value::Dword(v) => RegSetValueExW(self.0, n.as_ptr(), 0, REG_DWORD, v.to_le_bytes().as_ptr(), 4),
                Value::Text(kind, t) => {
                    let mut t = t.clone();
                    if t.last() != Some(&0) {
                        t.push(0);
                    }
                    RegSetValueExW(self.0, n.as_ptr(), 0, *kind, t.as_ptr() as *const u8, (t.len() * 2) as u32)
                }
            }
        };
        if err == ERROR_SUCCESS { Ok(()) } else { Err(format!("valor {name}: erro {err}")) }
    }

    fn set_str(&self, name: &str, value: &str) -> Result<(), String> {
        self.set(name, &Value::Text(REG_SZ, value.encode_utf16().collect()))
    }

    fn delete(&self, name: &str) -> Result<(), String> {
        let err = unsafe { RegDeleteValueW(self.0, wide(name).as_ptr()) };
        if err == ERROR_SUCCESS || err == ERROR_FILE_NOT_FOUND { Ok(()) } else { Err(format!("valor {name}: erro {err}")) }
    }

    fn subkeys(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut buf = [0u16; 256];
        for i in 0.. {
            let mut len = buf.len() as u32;
            let err = unsafe {
                RegEnumKeyExW(self.0, i, buf.as_mut_ptr(), &mut len, std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
            };
            if err != ERROR_SUCCESS {
                break;
            }
            out.push(String::from_utf16_lossy(&buf[..len as usize]));
        }
        out
    }
}

/// Active microphones, with whether KamoVox (or another effect) is on them.
pub fn endpoints() -> Vec<Endpoint> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let Ok(root) = Key::open(CAPTURE, KEY_READ) else { return Vec::new() };
    let inputs: Vec<String> = cpal::default_host()
        .input_devices()
        .map(|d| d.filter_map(|x| x.name().ok()).collect())
        .unwrap_or_default();
    // Windows may number duplicate devices ("Microfone (3- HyperX SoloCast)").
    let device_name = |desc: &str, iface: &str| {
        let exact = format!("{desc} ({iface})");
        if inputs.contains(&exact) {
            return exact;
        }
        let head = format!("{desc} (");
        let tail = format!("{iface})");
        let mut found = inputs.iter().filter(|n| n.starts_with(&head) && n.ends_with(&tail));
        match (found.next(), found.next()) {
            (Some(n), None) => n.clone(),
            _ => exact,
        }
    };
    let mut out: Vec<Endpoint> = root
        .subkeys()
        .into_iter()
        .filter_map(|id| {
            let ep = Key::open(&format!(r"{CAPTURE}\{id}"), KEY_READ).ok()?;
            (ep.get_dword("DeviceState")? == 1).then_some(())?;
            let props = Key::open(&format!(r"{CAPTURE}\{id}\Properties"), KEY_READ).ok()?;
            let desc = props.get_string("{a45c254e-df1c-4efd-8020-67d146a850e0},2")?;
            let iface = props.get_string("{b3f8fa53-0004-438e-9003-51a46e139bfc},6").unwrap_or_default();
            let name = if iface.is_empty() { desc } else { device_name(&desc, &iface) };
            let fx = Key::open(&format!(r"{CAPTURE}\{id}\FxProperties"), KEY_READ).ok();
            let gfx = fx.as_ref().and_then(|k| k.get_string(KEY_GFX)).unwrap_or_default();
            let installed = gfx.eq_ignore_ascii_case(CLSID);
            let other = (!installed && !gfx.is_empty())
                .then(|| fx.as_ref().and_then(|k| k.get_string(KEY_FX_NAME)).unwrap_or_else(|| "outro efeito".into()));
            Some(Endpoint { id, name, installed, other })
        })
        .collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

/// The APO DLL shipped with the app (bundled resource, or the build output in development).
pub fn find_dll(resource_dir: Option<PathBuf>) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        resource_dir.map(|d| d.join("resources").join("apo").join("kamovox_apo.dll")),
        Some(manifest.join("resources").join("apo").join("kamovox_apo.dll")),
        Some(manifest.join("..").join("target").join("release").join("kamovox_apo.dll")),
        Some(manifest.join("..").join("target").join("debug").join("kamovox_apo.dll")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use cpal::traits::{DeviceTrait, HostTrait};

    #[test]
    fn endpoint_names_match_cpal() {
        let cpal: Vec<String> = cpal::default_host().input_devices().unwrap().filter_map(|d| d.name().ok()).collect();
        for ep in super::endpoints() {
            println!("{} {} instalado={} outro={:?} cpal={}", ep.id, ep.name, ep.installed, ep.other, cpal.contains(&ep.name));
            assert!(cpal.contains(&ep.name), "{} não aparece no cpal: {cpal:?}", ep.name);
        }
    }
}

fn valid_endpoint(id: &str) -> bool {
    id.len() == 38 && id.starts_with('{') && id.ends_with('}') && id[1..37].chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

/// Runs the elevated copy of the app and waits for it. `Ok` once finished.
pub fn elevate(args: &[&str]) -> Result<(), String> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject, INFINITE};
    use windows_sys::Win32::UI::Shell::{ShellExecuteExW, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW};

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let result_file = std::env::temp_dir().join(format!("kamovox-apo-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&result_file);
    let mut params: Vec<String> = args.iter().map(|a| format!("\"{a}\"")).collect();
    params.push(format!("\"{}\"", result_file.display()));
    let verb = wide("runas");
    let file = wide(&exe.to_string_lossy());
    let params = wide(&params.join(" "));
    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb = verb.as_ptr();
    info.lpFile = file.as_ptr();
    info.lpParameters = params.as_ptr();
    info.nShow = 0;
    unsafe {
        if ShellExecuteExW(&mut info) == 0 || info.hProcess.is_null() {
            return Err("É preciso permitir o acesso de administrador para alterar o microfone do Windows".into());
        }
        WaitForSingleObject(info.hProcess, INFINITE);
        let mut code = 1u32;
        GetExitCodeProcess(info.hProcess, &mut code);
        CloseHandle(info.hProcess);
        let msg = std::fs::read_to_string(&result_file).unwrap_or_default();
        let _ = std::fs::remove_file(&result_file);
        if code == 0 { Ok(()) } else { Err(if msg.is_empty() { format!("Falha ao alterar o microfone (código {code})") } else { msg }) }
    }
}

/// Handles the elevated command line. Returns `None` for a normal launch.
pub fn run_cli() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--apo-install") if args.len() == 5 => install(&args[2], Path::new(&args[3])),
        Some("--apo-uninstall") if args.len() == 4 => uninstall(&args[2]),
        _ => return None,
    };
    let report = args.last().unwrap();
    match result {
        Ok(()) => Some(0),
        Err(e) => {
            let _ = std::fs::write(report, e);
            Some(1)
        }
    }
}

fn audio_service(start: bool) {
    let verb = if start { "start" } else { "stop" };
    let mut cmd = Command::new("net");
    cmd.args([verb, "audiosrv"]);
    if !start {
        cmd.arg("/y");
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let _ = cmd.output();
}

fn install(endpoint: &str, dll: &Path) -> Result<(), String> {
    if !valid_endpoint(endpoint) {
        return Err("Microfone inválido".into());
    }
    let fx = Key::open(&format!(r"{CAPTURE}\{endpoint}\FxProperties"), KEY_READ | KEY_SET_VALUE)
        .map_err(|e| format!("Este microfone não aceita efeitos do Windows (erro {e})"))?;

    audio_service(false);
    let result = (|| {
        std::fs::create_dir_all(INSTALL_DIR).map_err(|e| format!("Não foi possível criar {INSTALL_DIR}: {e}"))?;
        let target = Path::new(INSTALL_DIR).join("KamoVoxAPO.dll");
        std::fs::copy(dll, &target).map_err(|e| format!("Não foi possível copiar o filtro de áudio: {e}"))?;
        let target = target.to_string_lossy().to_string();

        let clsid = Key::create(&format!(r"SOFTWARE\Classes\CLSID\{CLSID}"))?;
        clsid.set_str("", "KamoVox APO")?;
        let server = Key::create(&format!(r"SOFTWARE\Classes\CLSID\{CLSID}\InprocServer32"))?;
        server.set_str("", &target)?;
        server.set_str("ThreadingModel", "Both")?;

        let reg = Key::create(&format!(r"SOFTWARE\Classes\AudioEngine\AudioProcessingObjects\{CLSID}"))?;
        reg.set_str("FriendlyName", NAME)?;
        reg.set_str("Copyright", NAME)?;
        for (name, v) in [
            ("MajorVersion", 1),
            ("MinorVersion", 0),
            ("Flags", 0xf),
            ("MinInputConnections", 1),
            ("MaxInputConnections", 1),
            ("MinOutputConnections", 1),
            ("MaxOutputConnections", 1),
            ("MaxInstances", u32::MAX),
            ("NumAPOInterfaces", 1),
        ] {
            reg.set(name, &Value::Dword(v))?;
        }
        reg.set_str("APOInterface0", "{FD7F2B29-24D0-4B5C-B177-592C39F9CA10}")?;

        // The apo.log diagnostics are written by the audio service (Local Service).
        let _ = std::fs::create_dir_all(LOG_DIR);
        let _ = Command::new("icacls").args([LOG_DIR, "/grant", "*S-1-5-19:(OI)(CI)M"]).output();

        let backup_path = format!(r"{BACKUP}\{endpoint}");
        if Key::open(&backup_path, KEY_READ).is_err() {
            let backup = Key::create(&backup_path)?;
            for name in FX_KEYS {
                match fx.get(name) {
                    Some(v) => backup.set(name, &v)?,
                    None => backup.set_str(name, ABSENT)?,
                }
            }
        }
        for name in FX_KEYS {
            fx.delete(name)?;
        }
        fx.set_str(KEY_GFX, CLSID)?;
        fx.set_str(KEY_FX_NAME, NAME)?;
        Ok(())
    })();
    audio_service(true);
    result
}

fn uninstall(endpoint: &str) -> Result<(), String> {
    if !valid_endpoint(endpoint) {
        return Err("Microfone inválido".into());
    }
    let fx = Key::open(&format!(r"{CAPTURE}\{endpoint}\FxProperties"), KEY_READ | KEY_SET_VALUE)
        .map_err(|e| format!("Não foi possível abrir o microfone (erro {e})"))?;
    let backup_path = format!(r"{BACKUP}\{endpoint}");
    let backup = Key::open(&backup_path, KEY_READ).ok();

    audio_service(false);
    let result = (|| {
        match &backup {
            Some(b) => {
                for name in FX_KEYS {
                    match b.get(name) {
                        Some(Value::Text(_, t)) if String::from_utf16_lossy(&t).trim_end_matches('\0') == ABSENT => fx.delete(name)?,
                        Some(v) => fx.set(name, &v)?,
                        None => {}
                    }
                }
            }
            None => {
                if fx.get_string(KEY_GFX).is_some_and(|g| g.eq_ignore_ascii_case(CLSID)) {
                    fx.delete(KEY_GFX)?;
                    fx.delete(KEY_FX_NAME)?;
                }
            }
        }
        drop(backup);
        let path = wide(&backup_path);
        unsafe {
            RegDeleteTreeW(HKEY_LOCAL_MACHINE, path.as_ptr());
            RegDeleteKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), KEY_WOW64_64KEY, 0);
        }
        Ok(())
    })();
    audio_service(true);
    result
}
