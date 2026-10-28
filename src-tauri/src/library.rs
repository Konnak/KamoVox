//! AI voice library: models folder, curated catalog, verified downloads and imports.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};

const CATALOG: &str = include_str!("catalog.json");
const BASE_ID: &str = "base";
/// CAM++ speaker embedding (~28 MB) for "only my voice".
pub const SPEAKER_ID: &str = "speaker";
const SPEAKER_FILE: &str = "campplus_sv_en_voxceleb_16k.onnx";
const SPEAKER_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
const SPEAKER_SIZE: u64 = 29_596_978;
const SPEAKER_SHA: &str = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b";

#[derive(Deserialize)]
struct BaseFile {
    file: String,
    url: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogVoice {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub description: String,
    pub credit: String,
    pub terms: String,
    pub source: String,
    pub pitch: f32,
    pub url: String,
    pub size: u64,
    pub sha256: String,
    /// When `url` is a zip archive: index of the `.pth` entry to install.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<usize>,
}

#[derive(Deserialize)]
struct Catalog {
    base: Vec<BaseFile>,
    voices: Vec<CatalogVoice>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct InstalledVoice {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub credit: String,
    #[serde(default)]
    pub terms: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub pitch: f32,
    #[serde(default)]
    pub imported: bool,
    pub version: String,
    pub sample_rate: u32,
    pub f0: bool,
    pub speakers: usize,
    /// Cover picture file inside the voice folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<String>,
    /// Cover as a data URL, filled in for the UI only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

const MAX_COVER_BYTES: u64 = 6 * 1024 * 1024;

fn image_mime(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

fn bundled_voices_dir(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join("voices"));
        candidates.push(dir.join("resources").join("voices"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("resources").join("voices"));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("voices"));
    candidates.into_iter().find(|p| p.is_dir())
}

fn stock_cover(id: &str) -> Option<PathBuf> {
    let name = format!("{id}/cover.png");
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("resources").join("voices").join(&name));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("voices").join(name));
    candidates.into_iter().find(|p| p.is_file())
}

#[derive(Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub id: String,
    pub received: u64,
    pub total: u64,
    pub done: bool,
    pub error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryState {
    pub models_dir: String,
    pub base_installed: bool,
    pub base_size: u64,
    pub speaker_installed: bool,
    pub speaker_size: u64,
    pub voices: Vec<InstalledVoice>,
    pub catalog: Vec<CatalogVoice>,
    pub downloads: Vec<Progress>,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct Config {
    models_dir: Option<PathBuf>,
}

struct Job {
    progress: Progress,
    cancel: Arc<AtomicBool>,
}

pub struct Library {
    catalog: Catalog,
    dir: Mutex<PathBuf>,
    jobs: Arc<Mutex<HashMap<String, Job>>>,
    config_path: PathBuf,
}

/// Fixed drive with the most free space, preferring one that already holds KamoVox models.
fn default_models_dir() -> PathBuf {
    let mut best: Option<(u64, PathBuf)> = None;
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::Storage::FileSystem::{GetDiskFreeSpaceExW, GetDriveTypeW, GetLogicalDrives};
        const DRIVE_FIXED: u32 = 3;
        let mask = GetLogicalDrives();
        for i in 0..26u32 {
            if mask & (1 << i) == 0 {
                continue;
            }
            let root = format!("{}:\\", (b'A' + i as u8) as char);
            let wide: Vec<u16> = root.encode_utf16().chain(Some(0)).collect();
            if GetDriveTypeW(wide.as_ptr()) != DRIVE_FIXED {
                continue;
            }
            let dir = PathBuf::from(&root).join("KamoVox").join("Modelos");
            if dir.exists() {
                return dir;
            }
            let mut free = 0u64;
            if GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) != 0
                && best.as_ref().map(|b| free > b.0).unwrap_or(true)
            {
                best = Some((free, dir));
            }
        }
    }
    best.map(|b| b.1).unwrap_or_else(|| PathBuf::from("KamoVox").join("Modelos"))
}

fn slug(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "voz".into() } else { s }
}

impl Library {
    pub fn new(app: &AppHandle) -> Self {
        let config_path = app.path().app_config_dir().unwrap_or_else(|_| PathBuf::from(".")).join("ai.json");
        let config: Config = fs::read(&config_path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let dir = config.models_dir.unwrap_or_else(default_models_dir);
        let lib = Self {
            catalog: serde_json::from_str(CATALOG).expect("catalog.json"),
            dir: Mutex::new(dir),
            jobs: Arc::default(),
            config_path,
        };
        lib.seed_bundled(app);
        lib
    }

    /// Copies the three voices shipped with the program, once, into the library.
    fn seed_bundled(&self, app: &AppHandle) {
        let Some(root) = bundled_voices_dir(app) else { return };
        let _ = fs::create_dir_all(self.voices_dir());
        let Ok(entries) = fs::read_dir(root) else { return };
        for entry in entries.flatten() {
            let src = entry.path();
            let model = src.join("model.pth");
            if !model.is_file() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().to_string();
            let dest = self.voices_dir().join(&id);
            if dest.join("model.pth").is_file() {
                continue;
            }
            let Some(v) = self.catalog.voices.iter().find(|v| v.id == id).cloned() else { continue };
            if fs::create_dir_all(&dest).is_err() || fs::copy(&model, dest.join("model.pth")).is_err() {
                continue;
            }
            let cover = if src.join("cover.png").is_file() && fs::copy(src.join("cover.png"), dest.join("cover.png")).is_ok() {
                Some("cover.png".to_string())
            } else {
                None
            };
            let Ok(info) = kamovox_ai::inspect(&dest.join("model.pth")) else { continue };
            let meta = InstalledVoice {
                id: v.id,
                name: v.name,
                tags: v.tags,
                credit: v.credit,
                terms: v.terms,
                source: v.source,
                pitch: v.pitch,
                imported: false,
                version: info.version,
                sample_rate: info.sample_rate,
                f0: info.f0,
                speakers: info.speakers,
                cover,
                image: None,
            };
            let _ = fs::write(dest.join("voice.json"), serde_json::to_vec_pretty(&meta).unwrap());
        }
    }

    pub fn dir(&self) -> PathBuf {
        self.dir.lock().unwrap().clone()
    }

    pub fn base_dir(&self) -> PathBuf {
        self.dir().join("base")
    }

    pub fn speaker_dir(&self) -> PathBuf {
        self.dir().join("speaker")
    }

    pub fn speaker_path(&self) -> Option<PathBuf> {
        let p = self.speaker_dir().join(SPEAKER_FILE);
        fs::metadata(&p).ok().filter(|m| m.len() == SPEAKER_SIZE).map(|_| p)
    }

    pub fn speaker_installed(&self) -> bool {
        self.speaker_path().is_some()
    }

    fn voices_dir(&self) -> PathBuf {
        self.dir().join("voices")
    }

    pub fn voice_path(&self, id: &str) -> Option<PathBuf> {
        let p = self.voices_dir().join(slug(id)).join("model.pth");
        p.exists().then_some(p)
    }

    pub fn set_dir(&self, dir: PathBuf) -> Result<(), String> {
        fs::create_dir_all(&dir).map_err(|e| format!("Não foi possível usar a pasta: {e}"))?;
        *self.dir.lock().unwrap() = dir.clone();
        if let Some(parent) = self.config_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let cfg = Config { models_dir: Some(dir) };
        fs::write(&self.config_path, serde_json::to_vec_pretty(&cfg).unwrap()).map_err(|e| e.to_string())
    }

    fn base_installed(&self) -> bool {
        let dir = self.base_dir();
        self.catalog.base.iter().all(|b| fs::metadata(dir.join(&b.file)).map(|m| m.len() == b.size).unwrap_or(false))
    }

    fn installed(&self) -> Vec<InstalledVoice> {
        let mut out: Vec<InstalledVoice> = fs::read_dir(self.voices_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let meta = fs::read(e.path().join("voice.json")).ok()?;
                e.path().join("model.pth").exists().then_some(())?;
                let mut v: InstalledVoice = serde_json::from_slice(&meta).ok()?;
                v.image = v.cover.as_deref().and_then(|c| {
                    use base64::Engine;
                    let mime = image_mime(Path::new(c).extension()?.to_str()?)?;
                    let bytes = fs::read(e.path().join(c)).ok()?;
                    Some(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
                });
                Some(v)
            })
            .collect();
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        out
    }

    pub fn state(&self) -> LibraryState {
        LibraryState {
            models_dir: self.dir().display().to_string(),
            base_installed: self.base_installed(),
            base_size: self.catalog.base.iter().map(|b| b.size).sum(),
            speaker_installed: self.speaker_installed(),
            speaker_size: SPEAKER_SIZE,
            voices: self.installed(),
            catalog: self.catalog.voices.clone(),
            downloads: self.jobs.lock().unwrap().values().map(|j| j.progress.clone()).collect(),
        }
    }

    fn write_voice(&self, src: &Path, meta: InstalledVoice, copy: bool) -> Result<(), String> {
        let dir = self.voices_dir().join(&meta.id);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let dest = dir.join("model.pth");
        if copy {
            fs::copy(src, &dest).map_err(|e| format!("Não foi possível copiar: {e}"))?;
        } else {
            fs::rename(src, &dest).map_err(|e| e.to_string())?;
        }
        fs::write(dir.join("voice.json"), serde_json::to_vec_pretty(&meta).unwrap()).map_err(|e| e.to_string())
    }

    pub fn import(&self, path: &Path) -> Result<InstalledVoice, String> {
        self.add_model(path, true)
    }

    /// Adds `.pth` files placed directly in the models folder, or in a voice
    /// subfolder without KamoVox metadata. They are moved into the library.
    /// Returns how many were added and the names that are not valid RVC models.
    pub fn scan(&self) -> (usize, Vec<String>) {
        let is_pth = |p: &Path| p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("pth"));
        let mut found: Vec<PathBuf> = fs::read_dir(self.dir())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_pth(p))
            .collect();
        for entry in fs::read_dir(self.voices_dir()).into_iter().flatten().flatten() {
            let dir = entry.path();
            if dir.is_dir() && !dir.join("voice.json").exists() {
                found.extend(fs::read_dir(&dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_file() && is_pth(p)));
            }
        }
        let mut added = 0;
        let mut failed = Vec::new();
        for path in found {
            let parent = path.parent().map(Path::to_path_buf);
            match self.add_model(&path, false) {
                Ok(_) => {
                    added += 1;
                    if let Some(p) = parent.filter(|p| p.starts_with(self.voices_dir())) {
                        let _ = fs::remove_dir(p);
                    }
                }
                Err(_) => failed.push(path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()),
            }
        }
        (added, failed)
    }

    fn add_model(&self, path: &Path, copy: bool) -> Result<InstalledVoice, String> {
        let info = kamovox_ai::inspect(path)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Voz").to_string();
        let base = slug(&stem);
        let mut id = base.clone();
        let mut n = 2;
        while self.voices_dir().join(&id).exists() {
            id = format!("{base}-{n}");
            n += 1;
        }
        let meta = InstalledVoice {
            id,
            name: stem,
            tags: vec!["Importada".into()],
            credit: String::new(),
            terms: String::new(),
            source: String::new(),
            pitch: 0.0,
            imported: true,
            version: info.version,
            sample_rate: info.sample_rate,
            f0: info.f0,
            speakers: info.speakers,
            cover: None,
            image: None,
        };
        self.write_voice(path, meta.clone(), copy)?;
        Ok(meta)
    }

    /// Renames a voice and sets (`Some(path)`), keeps (`None`) or clears (`Some("")`) its cover.
    pub fn update(&self, id: &str, name: &str, image: Option<&str>) -> Result<(), String> {
        let dir = self.voices_dir().join(slug(id));
        let meta_path = dir.join("voice.json");
        let mut meta: InstalledVoice = fs::read(&meta_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or("Voz não encontrada")?;
        let name = name.trim();
        if !name.is_empty() {
            meta.name = name.chars().take(40).collect();
        }
        if let Some(src) = image {
            let next = if src.is_empty() {
                None
            } else {
                let src = Path::new(src);
                let ext = src.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
                image_mime(&ext).ok_or("Use uma imagem PNG, JPG, WEBP ou GIF")?;
                let size = fs::metadata(src).map_err(|e| e.to_string())?.len();
                if size > MAX_COVER_BYTES {
                    return Err("A imagem é grande demais (máximo 6 MB)".into());
                }
                let bytes = fs::read(src).map_err(|e| format!("Não foi possível ler a imagem: {e}"))?;
                Some((format!("cover.{ext}"), bytes))
            };
            if let Some(old) = meta.cover.take() {
                let _ = fs::remove_file(dir.join(old));
            }
            if let Some((file, bytes)) = next {
                fs::write(dir.join(&file), bytes).map_err(|e| format!("Não foi possível salvar a imagem: {e}"))?;
                meta.cover = Some(file);
            }
        }
        meta.image = None;
        fs::write(meta_path, serde_json::to_vec_pretty(&meta).unwrap()).map_err(|e| e.to_string())
    }

    pub fn remove(&self, id: &str) -> Result<(), String> {
        let dir = self.voices_dir().join(slug(id));
        if dir.exists() {
            fs::remove_dir_all(dir).map_err(|e| format!("Não foi possível remover: {e}"))?;
        }
        Ok(())
    }

    pub fn cancel(&self, id: &str) {
        if let Some(j) = self.jobs.lock().unwrap().get(id) {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Starts a background download of base components, speaker biometrics, or a catalog voice.
    pub fn download(self: &Arc<Self>, app: AppHandle, id: String) -> Result<(), String> {
        let files: Vec<(String, PathBuf, u64, String)> = if id == BASE_ID {
            let dir = self.base_dir();
            self.catalog.base.iter().map(|b| (b.url.clone(), dir.join(&b.file), b.size, b.sha256.clone())).collect()
        } else if id == SPEAKER_ID {
            let dir = self.speaker_dir();
            let _ = fs::create_dir_all(&dir);
            vec![(SPEAKER_URL.into(), dir.join(SPEAKER_FILE), SPEAKER_SIZE, SPEAKER_SHA.into())]
        } else {
            let v = self.catalog.voices.iter().find(|v| v.id == id).ok_or("Voz não encontrada no catálogo")?;
            let ext = if v.url.to_ascii_lowercase().ends_with(".zip") || v.entry.is_some() { "zip" } else { "pth" };
            let tmp = self.dir().join("downloads").join(format!("{}.{ext}", v.id));
            vec![(v.url.clone(), tmp, v.size, v.sha256.clone())]
        };
        let total = files.iter().map(|f| f.2).sum();
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = self.jobs.lock().unwrap();
            if jobs.get(&id).map(|j| !j.progress.done).unwrap_or(false) {
                return Ok(());
            }
            let progress = Progress { id: id.clone(), total, ..Default::default() };
            jobs.insert(id.clone(), Job { progress, cancel: cancel.clone() });
        }
        let lib = self.clone();
        std::thread::spawn(move || {
            let jobs = lib.jobs.clone();
            let emit = |p: &Progress| {
                if let Some(j) = jobs.lock().unwrap().get_mut(&p.id) {
                    j.progress = p.clone();
                }
                let _ = app.emit("ai-download", p.clone());
            };
            let mut p = Progress { id: id.clone(), total, ..Default::default() };
            let mut result = Ok(());
            for (url, dest, size, sha) in &files {
                if fs::metadata(dest).map(|m| m.len() == *size).unwrap_or(false) {
                    p.received += size;
                    continue;
                }
                let offset = p.received;
                let mut last = Instant::now();
                result = fetch(url, dest, sha, &cancel, |n| {
                    p.received = offset + n;
                    if last.elapsed().as_millis() >= 150 {
                        last = Instant::now();
                        emit(&p);
                    }
                });
                if result.is_err() {
                    break;
                }
                p.received = offset + size;
            }
            if result.is_ok() && id != BASE_ID && id != SPEAKER_ID {
                result = lib.install_catalog_voice(&id, &files[0].1);
            }
            p.done = true;
            p.error = result.err();
            emit(&p);
            lib.jobs.lock().unwrap().remove(&id);
        });
        Ok(())
    }

    fn install_catalog_voice(&self, id: &str, file: &Path) -> Result<(), String> {
        let v = self.catalog.voices.iter().find(|v| v.id == id).ok_or("Voz não encontrada")?;
        let zip = file.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("zip"));
        let (model, cover) = if zip { self.unpack_voice(file)? } else { (file.to_path_buf(), None) };
        let info = kamovox_ai::inspect(&model)?;
        let meta = InstalledVoice {
            id: v.id.clone(),
            name: v.name.clone(),
            tags: v.tags.clone(),
            credit: v.credit.clone(),
            terms: v.terms.clone(),
            source: v.source.clone(),
            pitch: v.pitch,
            imported: false,
            version: info.version,
            sample_rate: info.sample_rate,
            f0: info.f0,
            speakers: info.speakers,
            cover: cover.as_ref().map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("cover.png").to_string()),
            image: None,
        };
        self.write_voice(&model, meta, false)?;
        if let Some(image) = cover {
            let dir = self.voices_dir().join(&v.id);
            let name = image.file_name().unwrap_or_default();
            let _ = fs::rename(&image, dir.join(name));
        } else if let Some(stock) = stock_cover(&v.id) {
            let _ = fs::copy(&stock, self.voices_dir().join(&v.id).join("cover.png"));
            let meta_path = self.voices_dir().join(&v.id).join("voice.json");
            if let Ok(bytes) = fs::read(&meta_path) {
                if let Ok(mut saved) = serde_json::from_slice::<InstalledVoice>(&bytes) {
                    saved.cover = Some("cover.png".into());
                    saved.image = None;
                    let _ = fs::write(meta_path, serde_json::to_vec_pretty(&saved).unwrap());
                }
            }
        }
        let _ = fs::remove_file(file);
        Ok(())
    }

    /// Pulls the `.pth` and, when present, a png/jpg/jpeg out of a voice pack.
    fn unpack_voice(&self, zip_path: &Path) -> Result<(PathBuf, Option<PathBuf>), String> {
        let bad = |e: zip::result::ZipError| format!("Pacote da voz inválido: {e}");
        let mut archive = zip::ZipArchive::new(File::open(zip_path).map_err(|e| e.to_string())?).map_err(bad)?;
        let mut model_at = None;
        let mut image_at: Option<(usize, String)> = None;
        for i in 0..archive.len() {
            let name = archive.by_index(i).map_err(bad)?.name().replace('\\', "/");
            let lower = name.to_ascii_lowercase();
            if lower.contains("__macosx") || lower.ends_with('/') {
                continue;
            }
            if lower.ends_with(".pth") {
                model_at = Some(i);
            } else if let Some(ext) = Path::new(&lower).extension().and_then(|e| e.to_str()) {
                if image_mime(ext).is_some() {
                    image_at = Some((i, ext.to_string()));
                }
            }
        }
        let model_at = model_at.ok_or("O pacote não contém um modelo .pth")?;
        let model_dest = zip_path.with_extension("pth");
        {
            let mut entry = archive.by_index(model_at).map_err(bad)?;
            let mut out = File::create(&model_dest).map_err(|e| format!("Não foi possível gravar: {e}"))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| format!("Não foi possível extrair: {e}"))?;
        }
        let image = if let Some((index, ext)) = image_at {
            let dest = zip_path.with_file_name(format!("cover.{ext}"));
            let mut entry = archive.by_index(index).map_err(bad)?;
            let mut out = File::create(&dest).map_err(|e| format!("Não foi possível gravar a imagem: {e}"))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| format!("Não foi possível extrair a imagem: {e}"))?;
            Some(dest)
        } else {
            None
        };
        Ok((model_dest, image))
    }
}

fn fetch(url: &str, dest: &Path, sha256: &str, cancel: &AtomicBool, mut progress: impl FnMut(u64)) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Não foi possível criar a pasta: {e}"))?;
    }
    let part = dest.with_extension("part");
    let resp = ureq::get(url).call().map_err(|e| format!("Falha no download: {e}"))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = File::create(&part).map_err(|e| format!("Não foi possível gravar: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 18];
    let mut received = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = fs::remove_file(&part);
            return Err("Download cancelado".into());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("Falha no download: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("Não foi possível gravar: {e}"))?;
        hasher.update(&buf[..n]);
        received += n as u64;
        progress(received);
    }
    drop(file);
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if digest != sha256 {
        let _ = fs::remove_file(&part);
        return Err("O arquivo baixado está corrompido; tente novamente".into());
    }
    fs::rename(&part, dest).map_err(|e| e.to_string())
}
