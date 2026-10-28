//! Looks for a newer installer on GitHub Releases. The repository only needs
//! the published setup file; the source stays private.

use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use tauri::AppHandle;

const REPO: &str = "Konnak/KamoVox";
const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub notes: String,
    pub available: bool,
    pub download: Option<String>,
    pub page: String,
}

pub fn check() -> Result<UpdateInfo, String> {
    let page = format!("https://github.com/{REPO}/releases/latest");
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .build()
        .into();
    let response = agent
        .get(&url)
        .header("User-Agent", "KamoVox")
        .header("Accept", "application/vnd.github+json")
        .call();
    let response = match response {
        Ok(r) => r,
        Err(ureq::Error::StatusCode(404)) => {
            return Ok(none(page, "Ainda não há uma versão publicada."));
        }
        Err(e) => return Err(format!("Não foi possível verificar: {e}")),
    };
    let body = response.into_body().read_to_string().map_err(|e| format!("Resposta inválida: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("Resposta inválida: {e}"))?;
    let latest = json.get("tag_name").and_then(|v| v.as_str()).unwrap_or(CURRENT).trim_start_matches('v').to_string();
    let notes = json.get("body").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let download = json
        .get("assets")
        .and_then(|v| v.as_array())
        .and_then(|assets| {
            assets.iter().find(|a| {
                let name = a.get("name").and_then(|n| n.as_str()).unwrap_or("").to_ascii_lowercase();
                name.ends_with(".exe") && (name.contains("setup") || name.contains("x64"))
            })
        })
        .and_then(|a| a.get("browser_download_url").and_then(|u| u.as_str()).map(str::to_string));
    let available = newer(&latest, CURRENT);
    Ok(UpdateInfo {
        current: CURRENT.into(),
        latest,
        notes,
        available,
        download,
        page,
    })
}

pub fn install(app: AppHandle, url: String) -> Result<(), String> {
    if !url.starts_with("https://github.com/") && !url.starts_with("https://objects.githubusercontent.com/") && !url.starts_with("https://release-assets.githubusercontent.com/")
    {
        return Err("Endereço de download não permitido".into());
    }
    let dest = std::env::temp_dir().join("KamoVox-setup.exe");
    download(&url, &dest)?;
    let size = std::fs::metadata(&dest).map_err(|e| format!("Instalador não foi gravado: {e}"))?.len();
    if size < 100_000 {
        return Err("O download da atualização veio incompleto.".into());
    }
    // A .cmd file keeps the path quoted. Passing it through `cmd /C` directly
    // strips quotes and Windows then tries to open "\\".
    let script = std::env::temp_dir().join("KamoVox-update.cmd");
    let exe = dest.display().to_string();
    std::fs::write(
        &script,
        // /S = NSIS silent: no wizard and no progress window. /UPDATE skips the
        // reinstall choice. /R relaunches the app when the files are in place.
        format!("@echo off\r\ntimeout /t 2 /nobreak >nul\r\n\"{exe}\" /S /UPDATE /R\r\n"),
    )
    .map_err(|e| format!("Não foi possível preparar a atualização: {e}"))?;
    spawn_updater(&script)?;
    app.exit(0);
    Ok(())
}

fn spawn_updater(script: &std::path::Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    // `/C` removes one pair of quotes. The extra pair survives, so a user folder
    // with spaces (for example "Cleison Carvalho") still reaches the script.
    std::process::Command::new("cmd.exe")
        .raw_arg(format!("/D /C \"\"{}\"\"", script.display()))
        .creation_flags(0x0800_0000)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Não foi possível abrir o instalador: {e}"))
}

fn download(url: &str, dest: &PathBuf) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(180))).build().into();
    let response = agent.get(url).header("User-Agent", "KamoVox").call().map_err(|e| format!("Falha no download: {e}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(dest).map_err(|e| format!("Não foi possível gravar: {e}"))?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Falha no download: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("Não foi possível gravar: {e}"))?;
    }
    Ok(())
}

fn none(page: String, notes: &str) -> UpdateInfo {
    UpdateInfo {
        current: CURRENT.into(),
        latest: CURRENT.into(),
        notes: notes.into(),
        available: false,
        download: None,
        page,
    }
}

fn newer(latest: &str, current: &str) -> bool {
    let parts = |v: &str| -> Vec<u32> {
        v.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect()
    };
    parts(latest) > parts(current)
}
