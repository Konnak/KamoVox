//! Loads ONNX Runtime sessions: the content embedder, the pitch network and an RVC
//! voice injected into a weightless synthesizer template.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ort::environment::Environment;
use ort::ep;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use crate::onnx_patch;
use crate::pth::{Checkpoint, TensorRef, Value};

pub const EMBEDDER: &str = "contentvec_768l12.onnx";
pub const RMVPE: &str = "rmvpe.onnx";
const FINAL_PROJ: &str = "contentvec_final_proj.bin";
const LAYER9: &str = "/hubert/encoder/layers.8/final_layer_norm/Add_1_output_0";

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub enum Device {
    Gpu,
    Cpu,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub enum PitchMethod {
    Rmvpe,
    Yin,
}

/// Everything that requires rebuilding sessions when it changes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoadKey {
    pub voice: PathBuf,
    pub speaker: usize,
    pub device: Device,
    pub pitch: PitchMethod,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInfo {
    pub version: String,
    pub sample_rate: u32,
    pub f0: bool,
    pub speakers: usize,
    pub info: String,
}

fn env() -> Result<&'static Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("kamovox").build().map_err(|e| e.to_string())).as_ref().map_err(Clone::clone)
}

fn cpu_threads() -> usize {
    (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 2).clamp(1, 8)
}

pub(crate) fn session_builder(device: Device) -> Result<SessionBuilder, String> {
    builder(device)
}

fn builder(device: Device) -> Result<SessionBuilder, String> {
    let s = |e: ort::Error<SessionBuilder>| e.to_string();
    let b = Session::builder(env()?).map_err(|e| e.to_string())?.with_optimization_level(GraphOptimizationLevel::Level3).map_err(s)?;
    match device {
        Device::Gpu => b
            .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
            .map_err(s)?
            .with_memory_pattern(false)
            .map_err(s)?
            .with_parallel_execution(false)
            .map_err(s),
        Device::Cpu => b
            .with_execution_providers([ep::CPU::default().build()])
            .map_err(s)?
            .with_intra_threads(cpu_threads())
            .map_err(s),
    }
}

pub struct Embedder {
    pub session: Session,
    pub output: String,
    /// RVC v1 projects layer 9 (768) down to 256 with ContentVec's final_proj.
    pub proj: Option<(usize, usize, Vec<f32>, Vec<f32>)>,
}

impl Embedder {
    pub fn dim(&self) -> usize {
        self.proj.as_ref().map(|p| p.0).unwrap_or(768)
    }
}

pub enum Pitch {
    Rmvpe(Session, String),
    Yin,
}

pub struct Synth {
    pub session: Session,
    pub inputs: Vec<String>,
    pub output: String,
    pub sample_rate: u32,
    pub upp: usize,
    pub f0: bool,
    /// Speaker embedding `[1, 256, 1]`.
    pub g: Vec<f32>,
}

pub struct Models {
    pub key: LoadKey,
    pub embedder: Embedder,
    pub pitch: Option<Pitch>,
    pub synth: Synth,
    /// Execution provider that actually runs the networks.
    pub device: Device,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TemplateMeta {
    sample_rate: u32,
    f0: bool,
    emb_channels: usize,
    upp: usize,
    weights: HashMap<String, Vec<i64>>,
    inputs: Vec<String>,
}

struct Parsed {
    ckpt: Checkpoint,
    tensors: HashMap<String, TensorRef>,
    version: String,
    sample_rate: u32,
    f0: bool,
    info: String,
}

fn parse_voice(path: &Path) -> Result<Parsed, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Não foi possível ler a voz: {e}"))?;
    let ckpt = Checkpoint::open(bytes)?;
    let weights = ckpt.root.get("weight").ok_or("o arquivo não é um modelo RVC (sem 'weight')")?;
    let tensors: HashMap<String, TensorRef> = weights
        .items()
        .into_iter()
        .filter_map(|(k, v)| match (k, v) {
            (Value::Str(k), Value::Tensor(t)) => Some((k, t)),
            _ => None,
        })
        .collect();
    let emb = tensors.get("enc_p.emb_phone.weight").ok_or("o arquivo não é um modelo RVC (sem enc_p)")?;
    let version = match emb.shape.get(1) {
        Some(768) => "v2",
        Some(256) => "v1",
        _ => return Err("versão de modelo RVC não suportada".into()),
    };
    let sample_rate = ckpt
        .root
        .get("config")
        .and_then(|c| c.elements().last().and_then(Value::as_int))
        .or_else(|| {
            ckpt.root.get("sr").and_then(|s| match s {
                Value::Str(s) => s.trim_end_matches('k').parse::<i64>().ok().map(|k| k * 1000),
                v => v.as_int(),
            })
        })
        .ok_or("taxa de amostragem do modelo desconhecida")? as u32;
    let f0 = ckpt.root.get("f0").and_then(|v| v.as_int()).unwrap_or(1) != 0;
    let info = ckpt.root.get("info").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    Ok(Parsed { ckpt, tensors, version: version.into(), sample_rate, f0, info })
}

pub fn inspect(path: &Path) -> Result<VoiceInfo, String> {
    let p = parse_voice(path)?;
    let speakers = p.tensors.get("emb_g.weight").and_then(|t| t.shape.first().copied()).unwrap_or(1);
    Ok(VoiceInfo { version: p.version, sample_rate: p.sample_rate, f0: p.f0, speakers, info: p.info })
}

/// Fetches a template parameter from the checkpoint, accepting both weight-norm layouts.
fn resolve(p: &mut Parsed, name: &str) -> Result<Vec<f32>, String> {
    if let Some(t) = p.tensors.get(name).cloned() {
        return p.ckpt.read_f32(&t);
    }
    let missing = || format!("peso ausente no modelo: {name}");
    let (base, is_g) = if let Some(b) = name.strip_suffix(".weight_g") {
        (b, true)
    } else if let Some(b) = name.strip_suffix(".weight_v") {
        (b, false)
    } else {
        return Err(missing());
    };
    let alias = format!("{base}.parametrizations.weight.original{}", if is_g { 0 } else { 1 });
    if let Some(t) = p.tensors.get(&alias).cloned() {
        return p.ckpt.read_f32(&t);
    }
    let plain = p.tensors.get(&format!("{base}.weight")).cloned().ok_or_else(missing)?;
    let w = p.ckpt.read_f32(&plain)?;
    if !is_g {
        return Ok(w);
    }
    let rows = plain.shape.first().copied().unwrap_or(1).max(1);
    let per = w.len() / rows;
    Ok(w.chunks(per).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect())
}

fn load_synth(resources: &Path, voice: &Path, speaker: usize, device: Device) -> Result<(Synth, usize), String> {
    let mut p = parse_voice(voice)?;
    let tag = if p.f0 { "f0" } else { "nof0" };
    let name = format!("rvc_{}_{}k_{tag}", p.version, p.sample_rate / 1000);
    let meta_path = resources.join(format!("{name}.json"));
    let meta: TemplateMeta = serde_json::from_slice(
        &std::fs::read(&meta_path).map_err(|_| format!("Modelo {} {} Hz não suportado", p.version, p.sample_rate))?,
    )
    .map_err(|e| e.to_string())?;

    let emb = p.tensors.get("emb_g.weight").cloned().ok_or("modelo sem embedding de locutor")?;
    let all = p.ckpt.read_f32(&emb)?;
    let width = emb.shape.get(1).copied().unwrap_or(256);
    let speakers = emb.shape.first().copied().unwrap_or(1);
    let sid = speaker.min(speakers.saturating_sub(1));
    let g = all[sid * width..(sid + 1) * width].to_vec();

    let mut b = builder(device)?;
    for (wname, dims) in &meta.weights {
        let data = resolve(&mut p, wname)?;
        let numel: i64 = dims.iter().product();
        if data.len() as i64 != numel {
            return Err(format!("formato inesperado em {wname}"));
        }
        let t = Tensor::from_array((dims.clone(), data)).map_err(|e| e.to_string())?;
        b = b.with_external_initializer(wname, Arc::new(t.into_dyn())).map_err(|e| e.to_string())?;
    }
    let session = b.commit_from_file(resources.join(format!("{name}.onnx"))).map_err(|e| e.to_string())?;
    let output = session.outputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "audio".into());
    Ok((
        Synth { session, inputs: meta.inputs, output, sample_rate: meta.sample_rate, upp: meta.upp, f0: meta.f0, g },
        meta.emb_channels,
    ))
}

fn load_embedder(base: &Path, resources: &Path, emb: usize, device: Device) -> Result<Embedder, String> {
    let file = base.join(EMBEDDER);
    let bytes = std::fs::read(&file).map_err(|_| "Componentes de IA não instalados (ContentVec)".to_string())?;
    if emb == 768 {
        let session = builder(device)?.commit_from_memory(&bytes).map_err(|e| e.to_string())?;
        return Ok(Embedder { session, output: "hidden_states".into(), proj: None });
    }
    let patched = onnx_patch::replace_outputs(&bytes, LAYER9)?;
    drop(bytes);
    let session = builder(device)?.commit_from_memory(&patched).map_err(|e| e.to_string())?;
    let raw = std::fs::read(resources.join(FINAL_PROJ)).map_err(|e| e.to_string())?;
    let u = |i: usize| u32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) as usize;
    let (out, inp) = (u(0), u(4));
    let floats: Vec<f32> = raw[8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    if out != emb || floats.len() != out * inp + out {
        return Err("projeção do ContentVec inválida".into());
    }
    let (w, bias) = floats.split_at(out * inp);
    Ok(Embedder { session, output: LAYER9.into(), proj: Some((out, inp, w.to_vec(), bias.to_vec())) })
}

fn load_pitch(base: &Path, method: PitchMethod, device: Device) -> Result<Pitch, String> {
    match method {
        PitchMethod::Yin => Ok(Pitch::Yin),
        PitchMethod::Rmvpe => {
            let file = base.join(RMVPE);
            if !file.exists() {
                return Err("Componentes de IA não instalados (RMVPE)".into());
            }
            let session = builder(device)?.commit_from_file(file).map_err(|e| e.to_string())?;
            let output = session.outputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "output".into());
            Ok(Pitch::Rmvpe(session, output))
        }
    }
}

impl Models {
    /// Loads everything for `key`. When the GPU can't be used, falls back to the CPU.
    pub fn load(base: &Path, resources: &Path, key: LoadKey) -> Result<Self, String> {
        match Self::load_on(base, resources, &key, key.device) {
            Err(e) if key.device == Device::Gpu && !e.contains("não instalados") => {
                Self::load_on(base, resources, &key, Device::Cpu).map_err(|_| e)
            }
            r => r,
        }
    }

    fn load_on(base: &Path, resources: &Path, key: &LoadKey, device: Device) -> Result<Self, String> {
        let (synth, emb) = load_synth(resources, &key.voice, key.speaker, device)?;
        let embedder = load_embedder(base, resources, emb, device)?;
        let pitch = if synth.f0 { Some(load_pitch(base, key.pitch, device)?) } else { None };
        Ok(Self { key: key.clone(), embedder, pitch, synth, device })
    }
}
