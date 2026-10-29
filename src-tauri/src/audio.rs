use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use kamovox_ai::{
    max_cosine_flat, to_16k, LoadKey, Models, Realtime, SpeakerEmbedder, SpeechBuffer, Stats, Timing,
    EMBED_DIM, FIRST_SPEECH_MS, LIVE_PERIOD_MS, LIVE_SPEECH_MS,
};
use kamovox_dsp::{Params, Processor};
use kamovox_link::{alive, now_ms, Link};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DeviceList {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub default_input: Option<String>,
    pub default_output: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct StartConfig {
    pub input: Option<String>,
    pub output: Option<String>,
    pub monitor: bool,
    /// Process the microphone through the KamoVox APO installed on `input`,
    /// so other programs recording from it hear the voice.
    #[serde(default)]
    pub apo: bool,
}

/// Resolved AI voice run, built from the UI request and the library.
#[derive(Clone, Debug)]
pub struct AiRun {
    pub base: PathBuf,
    pub resources: PathBuf,
    pub key: LoadKey,
    pub timing: Timing,
    pub transpose: f32,
}

#[derive(Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub running: bool,
    pub sample_rate: u32,
    pub input: Option<String>,
    pub output: Option<String>,
    pub monitoring: bool,
    pub denoise_supported: bool,
    /// Device running the AI voice ("gpu" / "cpu"), when active.
    pub ai_device: Option<kamovox_ai::Device>,
    /// The voice is being sent to the Windows microphone.
    pub apo: bool,
}

/// Visible in diagnostics so an old 0.1.16 binary cannot be mistaken for a new build.
pub const BUILD_ID: &str = "0.1.25-gate-recheck-20261029";

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Levels {
    pub input: f32,
    pub output: f32,
    pub running: bool,
    pub error: Option<String>,
    pub ai: Option<Stats>,
    /// Speaker Gate snapshot (no PCM / embeddings).
    pub gate: Option<GateSnapshot>,
    /// Internal build stamp (version alone stayed 0.1.16 across many fixes).
    pub build_id: String,
}

/// Compact live view of why the mic opens or stays closed.
#[derive(Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GateSnapshot {
    pub phase: u8,
    pub reason: u8,
    pub reason_label: String,
    pub phase_label: String,
    pub utterance_id: u64,
    pub neural: Option<f32>,
    pub thresh: f32,
    pub strict: f32,
    pub protect: bool,
    pub delay_ms: f32,
    pub buffered_ms: f32,
    pub buffer_capacity_ms: f32,
    pub speech_hops: u32,
    pub speech_ms: f32,
    pub speech_env: f32,
    pub infer_id: u64,
    pub infer_ms: f32,
    pub verify_age_ms: f32,
    pub authorized: bool,
    pub events: Vec<GateEventDto>,
}

fn gate_reason_label(code: u8) -> &'static str {
    match code {
        1 => "OWNER_CONFIRMED",
        2 => "PENDING",
        3 => "OTHER_SPEAKER_REJECTED",
        4 => "PENDING_VERIFICATION",
        5 => "SILENCE",
        7 => "UTTERANCE_FINISHED",
        8 => "SPEAKER_CHANGE",
        9 => "INSUFFICIENT_EVIDENCE",
        10 => "LOW_CONFIDENCE",
        11 => "WAITING_SCORE",
        12 => "REJECT_COOLDOWN",
        _ => "NONE",
    }
}

fn gate_phase_label(phase: u8) -> &'static str {
    match phase {
        0 => "Off",
        1 => "Closed",
        2 => "Candidate",
        3 => "Verifying",
        4 => "Locked",
        5 => "Holding",
        6 => "Releasing",
        7 => "Rejected",
        _ => "?",
    }
}

#[derive(Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GateEventDto {
    pub utterance_id: u64,
    pub prev: u8,
    pub next: u8,
    pub speech_hops: u32,
    pub neural: f32,
    pub thresh: f32,
    pub strict: f32,
    pub protect: bool,
    pub reason: u8,
    pub delay_samples: u32,
    pub infer_id: u64,
}

struct Shared {
    params: Mutex<Params>,
    generation: AtomicU64,
    level_in: AtomicU32,
    level_out: AtomicU32,
    error: Mutex<Option<String>>,
    transpose: AtomicU32,
    ai_stats: Mutex<Option<Stats>>,
    /// Recent peak time (in samples) the AI took to turn a block into voice.
    ai_work: AtomicU32,
    rate: AtomicU32,
    board: Mutex<Board>,
    /// When false, the voice sent to programs and headphones is silence.
    ptt_transmit: AtomicBool,
    /// Microphone samples collected for calibration. None means not recording.
    calib: Mutex<Option<Vec<f32>>>,
    /// ONNX path for CAM++ biometrics (set before start).
    speaker_model: Mutex<Option<PathBuf>>,
    /// Latest owner cosine; `NaN` bits mean "no neural score".
    neural_score: AtomicU32,
    /// SpeechBuffer utterance id that produced `neural_score` (0 = none).
    neural_utt: AtomicU64,
    /// Params generation when score was computed.
    neural_cfg: AtomicU64,
    /// Monotonic inference id — gate rejects scores with id ≤ accept_after.
    neural_infer_id: AtomicU64,
    /// Last CAM++ inference duration (ms bits).
    neural_infer_ms: AtomicU32,
    /// Last Speaker Gate snapshot for diagnostics.
    gate_snap: Mutex<Option<GateSnapshot>>,
    /// Enrollment pauses live CAM++ so we do not load a second session mid-run.
    neural_pause: AtomicBool,
    neural_busy: AtomicBool,
}

impl Shared {
    fn neural_score(&self) -> Option<f32> {
        let v = f32::from_bits(self.neural_score.load(Ordering::Relaxed));
        v.is_finite().then_some(v.clamp(-1.0, 1.0))
    }

    fn neural_infer_id(&self) -> u64 {
        self.neural_infer_id.load(Ordering::Relaxed)
    }

    fn clear_neural(&self) {
        self.neural_score.store(f32::NAN.to_bits(), Ordering::Relaxed);
        self.neural_utt.store(0, Ordering::Relaxed);
    }

    fn publish_neural(&self, score: f32, utt: u64, cfg: u64, infer_ms: f32) -> u64 {
        let id = self.neural_infer_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.neural_score.store(score.to_bits(), Ordering::Relaxed);
        self.neural_utt.store(utt, Ordering::Relaxed);
        self.neural_cfg.store(cfg, Ordering::Relaxed);
        self.neural_infer_ms.store(infer_ms.to_bits(), Ordering::Relaxed);
        id
    }
}

struct Playing {
    id: String,
    pcm: Vec<f32>,
    pos: usize,
}

struct Board {
    items: Vec<Playing>,
}

impl Board {
    fn toggle(&mut self, id: &str, pcm: Vec<f32>) {
        if let Some(i) = self.items.iter().position(|p| p.id == id) {
            self.items.remove(i);
        } else {
            self.items.push(Playing { id: id.to_string(), pcm, pos: 0 });
        }
    }

    fn mix(&mut self, out: &mut [f32]) {
        if self.items.is_empty() {
            return;
        }
        for s in out.iter_mut() {
            let mut add = 0.0;
            for p in &mut self.items {
                if p.pos < p.pcm.len() {
                    add += p.pcm[p.pos];
                    p.pos += 1;
                }
            }
            *s = (*s + add * 0.8).clamp(-1.0, 1.0);
        }
        self.items.retain(|p| p.pos < p.pcm.len());
    }
}

fn mix_board(shared: &Shared, buf: &mut [f32]) {
    if let Ok(mut board) = shared.board.try_lock() {
        board.mix(buf);
    }
}

fn silence_if_muted(shared: &Shared, buf: &mut [f32]) {
    if !shared.ptt_transmit.load(Ordering::Relaxed) {
        buf.fill(0.0);
    }
}

impl Shared {
    fn report(&self, msg: String) {
        if let Ok(mut e) = self.error.lock() {
            *e = Some(msg);
        }
    }

    /// Applies new params to `proc` when they changed since `seen`, without blocking.
    fn sync(&self, seen: &mut u64, proc: &mut Processor, map: impl Fn(Params) -> Params) {
        let gen = self.generation.load(Ordering::Acquire);
        if gen != *seen {
            if let Ok(p) = self.params.try_lock() {
                proc.set_params(map(p.clone()));
                *seen = gen;
            }
        }
    }
}

enum Command {
    Start(StartConfig, Option<AiRun>, mpsc::Sender<Result<Status, String>>),
    Stop(mpsc::Sender<Status>),
}

/// Owns the audio thread. cpal streams are not `Send`, so they live on that thread.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Command>,
    shared: Arc<Shared>,
    status: Arc<Mutex<Status>>,
}

impl Engine {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            params: Mutex::new(Params::default()),
            generation: AtomicU64::new(1),
            level_in: AtomicU32::new(0),
            level_out: AtomicU32::new(0),
            error: Mutex::new(None),
            transpose: AtomicU32::new(0),
            ai_stats: Mutex::new(None),
            ai_work: AtomicU32::new(0),
            rate: AtomicU32::new(48_000),
            board: Mutex::new(Board { items: Vec::new() }),
            ptt_transmit: AtomicBool::new(true),
            calib: Mutex::new(None),
            speaker_model: Mutex::new(None),
            neural_score: AtomicU32::new(f32::NAN.to_bits()),
            neural_utt: AtomicU64::new(0),
            neural_cfg: AtomicU64::new(0),
            neural_infer_id: AtomicU64::new(0),
            neural_infer_ms: AtomicU32::new(0),
            gate_snap: Mutex::new(None),
            neural_pause: AtomicBool::new(false),
            neural_busy: AtomicBool::new(false),
        });
        shared.clear_neural();
        let status = Arc::new(Mutex::new(Status::default()));
        let (tx, rx) = mpsc::channel::<Command>();
        let thread_shared = shared.clone();
        let thread_status = status.clone();
        thread::Builder::new()
            .name("kamovox-audio".into())
            .spawn(move || audio_thread(rx, thread_shared, thread_status))
            .expect("failed to spawn audio thread");
        Self { tx, shared, status }
    }

    /// CAM++ ONNX used by the neural biometric worker (call before `start`).
    pub fn set_speaker_model(&self, path: Option<PathBuf>) {
        if let Ok(mut slot) = self.shared.speaker_model.lock() {
            *slot = path;
        }
    }

    pub fn start(&self, cfg: StartConfig, ai: Option<AiRun>) -> Result<Status, String> {
        let (reply, rx) = mpsc::channel();
        self.tx.send(Command::Start(cfg, ai, reply)).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?
    }

    pub fn set_ptt_transmit(&self, on: bool) {
        self.shared.ptt_transmit.store(on, Ordering::Relaxed);
    }

    /// Plays a clip into the microphone. The same id stops it if it is already playing.
    pub fn play_sound(&self, id: &str, path: &std::path::Path) -> Result<(), String> {
        let rate = self.shared.rate.load(Ordering::Relaxed).max(8_000);
        let pcm = crate::sound::decode(path, rate)?;
        self.shared.board.lock().map_err(|e| e.to_string())?.toggle(id, pcm);
        Ok(())
    }

    pub fn stop(&self) -> Status {
        let (reply, rx) = mpsc::channel();
        if self.tx.send(Command::Stop(reply)).is_err() {
            return Status::default();
        }
        rx.recv().unwrap_or_default()
    }

    pub fn set_params(&self, params: Params) {
        if let Ok(mut p) = self.shared.params.lock() {
            *p = params;
        }
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
    }

    pub fn set_transpose(&self, semitones: f32) {
        self.shared.transpose.store(semitones.clamp(-24.0, 24.0).to_bits(), Ordering::Relaxed);
    }

    /// Starts keeping the microphone audio the program is already hearing.
    pub fn begin_calibration(&self) -> Result<(), String> {
        let running = self.status.lock().map(|s| s.running).unwrap_or(false);
        if !running {
            return Err("Ligue o trocador de voz. A calibração usa o microfone que o programa já está ouvindo.".into());
        }
        if let Ok(mut slot) = self.shared.calib.lock() {
            *slot = Some(Vec::with_capacity(48_000 * 8));
        }
        Ok(())
    }

    pub fn end_calibration(&self) -> (Vec<f32>, u32) {
        let samples = self.shared.calib.lock().ok().and_then(|mut slot| slot.take()).unwrap_or_default();
        let rate = self.shared.rate.load(Ordering::Relaxed).max(8_000);
        (samples, rate)
    }

    /// Pause live CAM++ while enrollment loads its own session (ORT is not happy
    /// with two `campplus.onnx` sessions inferring at once).
    pub fn pause_neural(&self, pause: bool) {
        self.shared.neural_pause.store(pause, Ordering::Release);
        if !pause {
            return;
        }
        let deadline = std::time::Instant::now() + Duration::from_millis(1500);
        while self.shared.neural_busy.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
    }

    pub fn levels(&self) -> Levels {
        let running = self.status.lock().map(|s| s.running).unwrap_or(false);
        Levels {
            input: f32::from_bits(self.shared.level_in.swap(0, Ordering::Relaxed)),
            output: f32::from_bits(self.shared.level_out.swap(0, Ordering::Relaxed)),
            running,
            error: self.shared.error.lock().ok().and_then(|mut e| e.take()),
            ai: self.shared.ai_stats.lock().ok().and_then(|s| *s),
            gate: self.shared.gate_snap.lock().ok().and_then(|g| g.clone()),
            build_id: BUILD_ID.to_string(),
        }
    }
}

pub fn list_devices() -> Result<DeviceList, String> {
    let host = cpal::default_host();
    let inputs = host.input_devices().map(|d| d.filter_map(|x| x.name().ok()).collect()).unwrap_or_default();
    let outputs = host.output_devices().map(|d| d.filter_map(|x| x.name().ok()).collect()).unwrap_or_default();
    Ok(DeviceList {
        inputs,
        outputs,
        default_input: host.default_input_device().and_then(|d| d.name().ok()),
        default_output: host.default_output_device().and_then(|d| d.name().ok()),
    })
}

struct AiWorker {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<Models>,
}

struct LinkWorker {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

struct NeuralWorker {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl LinkWorker {
    fn stop(self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.handle.join();
    }
}

impl NeuralWorker {
    fn stop(self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.handle.join();
    }
}

struct Running {
    _input: Stream,
    _output: Option<Stream>,
    worker: Option<AiWorker>,
    link: Option<LinkWorker>,
    neural: Option<NeuralWorker>,
}

/// Stops the AI worker (if any) and keeps its models for a quick restart.
fn shut_down(running: &mut Option<Running>, cache: &mut Option<Models>, shared: &Shared) {
    if let Some(mut r) = running.take() {
        if let Some(n) = r.neural.take() {
            n.stop();
        }
        if let Some(l) = r.link.take() {
            l.stop();
        }
        if let Some(w) = r.worker.take() {
            w.stop.store(true, Ordering::Release);
            if let Ok(models) = w.handle.join() {
                *cache = Some(models);
            }
        }
    }
    shared.clear_neural();
    if let Ok(mut s) = shared.ai_stats.lock() {
        *s = None;
    }
}

fn audio_thread(rx: mpsc::Receiver<Command>, shared: Arc<Shared>, status: Arc<Mutex<Status>>) {
    let mut running: Option<Running> = None;
    let mut cache: Option<Models> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::Start(cfg, ai, reply) => {
                shut_down(&mut running, &mut cache, &shared);
                if ai.is_none() {
                    cache = None;
                }
                let answer = match open_streams(&cfg, ai, &shared, &mut cache) {
                    Ok((r, st)) => {
                        running = Some(r);
                        Ok(st)
                    }
                    Err(e) => Err(e),
                };
                if let Ok(mut s) = status.lock() {
                    *s = answer.clone().unwrap_or_default();
                }
                let _ = reply.send(answer);
            }
            Command::Stop(reply) => {
                shut_down(&mut running, &mut cache, &shared);
                let st = Status::default();
                if let Ok(mut s) = status.lock() {
                    *s = st.clone();
                }
                let _ = reply.send(st);
            }
        }
    }
}

fn find_device(devices: Result<impl Iterator<Item = Device>, cpal::DevicesError>, name: &str) -> Option<Device> {
    devices.ok()?.find(|d| d.name().map(|n| n == name).unwrap_or(false))
}

fn prepare_ai(ai: &AiRun, sample_rate: u32, cache: &mut Option<Models>) -> Result<Realtime, String> {
    let models = match cache.take() {
        Some(m) if m.key == ai.key => m,
        _ => Models::load(&ai.base, &ai.resources, ai.key.clone())?,
    };
    let mut rt = Realtime::new(models, sample_rate, ai.timing);
    rt.transpose = ai.transpose;
    if let Err(e) = rt.warm_up() {
        *cache = Some(rt.into_models());
        return Err(format!("Voz IA: {e}"));
    }
    Ok(rt)
}

fn open_streams(
    cfg: &StartConfig,
    ai: Option<AiRun>,
    shared: &Arc<Shared>,
    cache: &mut Option<Models>,
) -> Result<(Running, Status), String> {
    let host = cpal::default_host();
    let input = match &cfg.input {
        Some(name) => find_device(host.input_devices(), name),
        None => host.default_input_device(),
    }
    .ok_or("Microfone não encontrado")?;

    let in_supported = input.default_input_config().map_err(|e| format!("Microfone: {e}"))?;
    let in_format = in_supported.sample_format();
    let in_config: StreamConfig = in_supported.config();

    // Windows only runs the APO while something records from the microphone,
    // so keep a capture open; the raw microphone itself comes from the link.
    let (keeper, link) = if cfg.apo {
        let keeper = match in_format {
            SampleFormat::F32 => build_keeper::<f32>(&input, &in_config, shared.clone()),
            SampleFormat::I16 => build_keeper::<i16>(&input, &in_config, shared.clone()),
            SampleFormat::U16 => build_keeper::<u16>(&input, &in_config, shared.clone()),
            SampleFormat::I32 => build_keeper::<i32>(&input, &in_config, shared.clone()),
            f => Err(format!("Formato de microfone não suportado: {f:?}")),
        }?;
        keeper.play().map_err(|e| format!("Microfone: {e}"))?;
        crate::endpoint::unmute_captures();
        (Some(keeper), Some(Arc::new(wait_for_link()?)))
    } else {
        (None, None)
    };
    let sample_rate = match &link {
        Some(l) => l.header().sample_rate.load(Ordering::Relaxed),
        None => in_config.sample_rate.0,
    };
    shared.rate.store(sample_rate.max(8_000), Ordering::Relaxed);

    let realtime = match &ai {
        Some(a) => {
            shared.transpose.store(a.transpose.to_bits(), Ordering::Relaxed);
            Some(prepare_ai(a, sample_rate, cache)?)
        }
        None => None,
    };
    let ai_device = realtime.as_ref().map(|r| r.models().device);
    let block = realtime.as_ref().map(|r| r.block_len()).unwrap_or(0);

    let mut processor = Processor::new(sample_rate);
    let mut seen = 0;
    shared.sync(&mut seen, &mut processor, |p| p);
    let denoise_supported = processor.denoise_supported();

    let mut out_name = None;
    let (producer, output_stream) = if cfg.monitor {
        let output = match &cfg.output {
            Some(name) => find_device(host.output_devices(), name),
            None => host.default_output_device(),
        }
        .ok_or("Saída de áudio não encontrada")?;
        out_name = output.name().ok();
        let out_supported = output.default_output_config().map_err(|e| format!("Saída: {e}"))?;
        let out_format = out_supported.sample_format();
        let out_config: StreamConfig = out_supported.config();

        let rb = HeapRb::<f32>::new(sample_rate as usize * 2 + block * 2);
        let (prod, cons) = rb.split();
        let ms = |v: f32| (sample_rate as f32 * v / 1000.0) as usize;
        // AI audio arrives in bursts of one block; keep a cushion against inference jitter.
        let (cushion, max) = if block > 0 { (ms(40.0), block * 2 + ms(80.0)) } else { (ms(10.0), ms(120.0)) };
        let state = OutputState::new(cons, sample_rate, out_config.sample_rate.0, cushion, max, shared.clone());
        let stream = match out_format {
            SampleFormat::F32 => build_output::<f32>(&output, &out_config, state),
            SampleFormat::I16 => build_output::<i16>(&output, &out_config, state),
            SampleFormat::U16 => build_output::<u16>(&output, &out_config, state),
            SampleFormat::I32 => build_output::<i32>(&output, &out_config, state),
            f => Err(format!("Formato de saída não suportado: {f:?}")),
        }?;
        (Some(prod), Some(stream))
    } else {
        (None, None)
    };

    // AI finishes a whole block at once (up to ~800 ms). The APO ring is only
    // ~680 ms, so a dump would wrap and eat syllables in Discord. The monitor
    // has its own 2 s buffer, which is why it still sounded fine. Pace the
    // APO at the microphone clock from a large app-side buffer instead.
    let mut monitor_out = producer;
    let (paced_prod, paced_cons) = if link.is_some() && block > 0 {
        let rb = HeapRb::<f32>::new(sample_rate as usize * 4 + block * 2);
        let (prod, cons) = rb.split();
        (
            Some(prod),
            Some(Pace {
                cons,
                block,
                arm_at: None,
                started: false,
                last: 0.0,
                monitor: monitor_out.take(),
                shared: shared.clone(),
            }),
        )
    } else {
        (None, None)
    };
    let out = Out {
        monitor: monitor_out,
        link: if paced_prod.is_some() { None } else { link.clone() },
        paced: paced_prod,
        shared: shared.clone(),
    };
    let (sink, worker) = match realtime {
        Some(rt) => {
            let (ai_prod, ai_cons) = HeapRb::<f32>::new(sample_rate as usize * 4).split();
            let stop = Arc::new(AtomicBool::new(false));
            let w = AiLoop { rt, input: ai_cons, output: out, shared: shared.clone(), stop: stop.clone(), sample_rate };
            let handle = thread::Builder::new().name("kamovox-ai".into()).spawn(move || w.run()).map_err(|e| e.to_string())?;
            (Sink::Ai(ai_prod), Some(AiWorker { stop, handle }))
        }
        None => (Sink::Direct(out), None),
    };

    let (neural_prod, neural) = match shared.speaker_model.lock().ok().and_then(|p| p.clone()) {
        Some(path) if path.is_file() => {
            let (prod, cons) = HeapRb::<f32>::new(sample_rate as usize * 3).split();
            let stop = Arc::new(AtomicBool::new(false));
            let thread_shared = shared.clone();
            let thread_stop = stop.clone();
            let handle = thread::Builder::new()
                .name("kamovox-speaker".into())
                .spawn(move || neural_loop(cons, thread_shared, thread_stop, path, sample_rate))
                .map_err(|e| e.to_string())?;
            (Some(prod), Some(NeuralWorker { stop, handle }))
        }
        _ => {
            shared.clear_neural();
            (None, None)
        }
    };

    let stop_worker = |worker: Option<AiWorker>, neural: Option<NeuralWorker>, cache: &mut Option<Models>| {
        if let Some(n) = neural {
            n.stop();
        }
        if let Some(w) = worker {
            w.stop.store(true, Ordering::Release);
            if let Ok(m) = w.handle.join() {
                *cache = Some(m);
            }
        }
    };

    let state = InputState {
        processor,
        seen,
        mono: Vec::with_capacity(8192),
        sink,
        neural: neural_prod,
        shared: shared.clone(),
    };
    let (input_stream, link_worker) = match (keeper, link) {
        (Some(keeper), Some(link)) => {
            let ms = |v: f32| (sample_rate as f32 * v / 1000.0) as u32;
            let h = link.header();
            let cushion = if block > 0 {
                shared.ai_work.store(0, Ordering::Relaxed);
                Cushion { now: ms(20.0), min: ms(20.0), max: ms(60.0), step: ms(10.0), ai: true }
            } else {
                Cushion { now: ms(10.0), min: ms(10.0), max: ms(60.0), step: ms(10.0), ai: false }
            };
            h.cushion.store(cushion.now, Ordering::Relaxed);
            // One audio period, not the AI block: the pacer writes in lockstep with the mic.
            h.burst.store(ms(20.0), Ordering::Relaxed);
            let stop = Arc::new(AtomicBool::new(false));
            let thread_stop = stop.clone();
            let spawned = thread::Builder::new()
                .name("kamovox-link".into())
                .spawn(move || run_link(link, state, thread_stop, cushion, paced_cons));
            match spawned {
                Ok(handle) => (keeper, Some(LinkWorker { stop, handle })),
                Err(e) => {
                    stop_worker(worker, neural, cache);
                    return Err(e.to_string());
                }
            }
        }
        _ => {
            drop(paced_cons);
            let built = match in_format {
                SampleFormat::F32 => build_input::<f32>(&input, &in_config, state),
                SampleFormat::I16 => build_input::<i16>(&input, &in_config, state),
                SampleFormat::U16 => build_input::<u16>(&input, &in_config, state),
                SampleFormat::I32 => build_input::<i32>(&input, &in_config, state),
                f => Err(format!("Formato de microfone não suportado: {f:?}")),
            };
            let stream = match built {
                Ok(s) => s,
                Err(e) => {
                    stop_worker(worker, neural, cache);
                    return Err(e);
                }
            };
            if let Err(e) = stream.play() {
                stop_worker(worker, neural, cache);
                return Err(format!("Microfone: {e}"));
            }
            (stream, None)
        }
    };

    if let Some(s) = &output_stream {
        if let Err(e) = s.play() {
            if let Some(l) = link_worker {
                l.stop();
            }
            stop_worker(worker, neural, cache);
            return Err(format!("Saída: {e}"));
        }
    }

    let status = Status {
        running: true,
        sample_rate,
        input: input.name().ok(),
        output: out_name,
        monitoring: cfg.monitor,
        denoise_supported,
        ai_device,
        apo: link_worker.is_some(),
    };
    Ok((Running { _input: input_stream, _output: output_stream, worker, link: link_worker, neural }, status))
}

fn wait_for_link() -> Result<Link, String> {
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_millis(2500) {
        if let Some(l) = Link::open() {
            let h = l.header();
            if alive(&h.apo_beat) && h.sample_rate.load(Ordering::Relaxed) > 0 {
                return Ok(l);
            }
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err("O KamoVox não está ativo neste microfone. Instale-o em Dispositivos › Microfone do Windows.".into())
}

/// Feeds the raw microphone from the APO through the voice chain.
/// Safety margin (samples) the APO keeps queued. It grows after every dropout
/// and slowly returns to `min` once playback has been stable for a while.
#[derive(Clone, Copy)]
struct Cushion {
    now: u32,
    min: u32,
    max: u32,
    step: u32,
    ai: bool,
}

const CUSHION_RELAX_MS: u64 = 30_000;

fn run_link(link: Arc<Link>, mut state: InputState, stop: Arc<AtomicBool>, mut cushion: Cushion, mut pace: Option<Pace>) {
    crate::priority::pro_audio_thread();
    let h = link.header();
    let mut pos = link.mic_position();
    let mut buf = vec![0f32; 4096];
    let mut voice = vec![0f32; 4096];
    let mut underruns = h.underruns.load(Ordering::Relaxed);
    let mut calm_since = now_ms();
    while !stop.load(Ordering::Acquire) {
        let now = now_ms();
        h.app_beat.store(now, Ordering::Release);
        h.app_active.store(1, Ordering::Release);

        if !cushion.ai {
            let before = cushion.now;
            let seen = h.underruns.load(Ordering::Relaxed);
            if seen != underruns {
                underruns = seen;
                calm_since = now;
                cushion.now = (cushion.now + cushion.step).min(cushion.max);
            } else if cushion.now > cushion.min && now - calm_since > CUSHION_RELAX_MS {
                calm_since = now;
                cushion.now = cushion.now.saturating_sub(cushion.step / 2).max(cushion.min);
            }
            if cushion.now != before {
                h.cushion.store(cushion.now, Ordering::Relaxed);
            }
        }

        let n = link.pull_mic(&mut pos, &mut buf);
        if n == 0 {
            thread::sleep(Duration::from_millis(1));
            continue;
        }
        state.handle_mono(&buf[..n]);
        if let Some(pace) = &mut pace {
            pace.feed(&link, n, &mut voice);
        }
    }
    h.app_active.store(0, Ordering::Release);
}

fn build_keeper<T: SizedSample>(device: &Device, config: &StreamConfig, shared: Arc<Shared>) -> Result<Stream, String> {
    device
        .build_input_stream(config, |_: &[T], _: &cpal::InputCallbackInfo| {}, move |err| shared.report(format!("Microfone: {err}")), None)
        .map_err(|e| format!("Microfone: {e}"))
}

/// Destinations of the finished voice: the monitor and/or the Windows microphone.
struct Out {
    monitor: Option<HeapProd<f32>>,
    link: Option<Arc<Link>>,
    paced: Option<HeapProd<f32>>,
    shared: Arc<Shared>,
}

impl Out {
    /// Mix / mute in-place — no heap alloc on the realtime path.
    fn push(&mut self, samples: &mut [f32]) {
        if self.paced.is_some() {
            if let Some(p) = &mut self.paced {
                p.push_slice(samples);
            }
            return;
        }
        mix_board(&self.shared, samples);
        silence_if_muted(&self.shared, samples);
        if let Some(p) = &mut self.monitor {
            p.push_slice(samples);
        }
        if let Some(l) = &self.link {
            l.push_voice(samples);
        }
    }
}

/// Turns AI bursts into a microphone-clock stream for the APO.
struct Pace {
    cons: HeapCons<f32>,
    block: usize,
    /// When the first block arrived, plus a spare so the next block can be late.
    arm_at: Option<u64>,
    started: bool,
    last: f32,
    monitor: Option<HeapProd<f32>>,
    shared: Arc<Shared>,
}

impl Pace {
    fn feed(&mut self, link: &Link, n: usize, tmp: &mut [f32]) {
        let tmp = &mut tmp[..n];
        let queued = self.cons.occupied_len();
        if !self.shared.ptt_transmit.load(Ordering::Relaxed) {
            if queued > 0 {
                let _ = self.cons.pop_slice(tmp);
            }
            tmp.fill(0.0);
            link.push_voice(tmp);
            if let Some(m) = &mut self.monitor {
                m.push_slice(tmp);
            }
            return;
        }
        // The next AI block arrives just as the previous one finishes, so a
        // single late block used to empty the buffer and then wait for a whole
        // new block of silence. Hold a spare 150 ms instead.
        if !self.started {
            if queued == 0 {
                return;
            }
            let now = now_ms();
            let arm = *self.arm_at.get_or_insert(now + 150);
            if now < arm {
                return;
            }
            self.started = true;
        }
        let cap = self.block.saturating_mul(2) + n;
        if queued > cap {
            self.cons.skip(queued - self.block);
        }
        let got = self.cons.pop_slice(tmp);
        if got > 0 {
            self.last = tmp[got - 1];
        }
        if got < n {
            // Fade to silence instead of holding a DC step (click on refill).
            for s in &mut tmp[got..] {
                self.last *= 0.95;
                *s = self.last;
            }
        }
        mix_board(&self.shared, tmp);
        silence_if_muted(&self.shared, tmp);
        link.push_voice(tmp);
        if let Some(m) = &mut self.monitor {
            m.push_slice(tmp);
        }
    }
}

fn store_peak(slot: &AtomicU32, peak: f32) {
    let prev = f32::from_bits(slot.load(Ordering::Relaxed));
    if peak > prev {
        slot.store(peak.to_bits(), Ordering::Relaxed);
    }
}

fn peak(buf: &[f32]) -> f32 {
    buf.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// After the AI voice only effects and the character extras apply; the classic voices
/// and pitch would fight the model.
fn ai_post_params(mut p: Params) -> Params {
    p.voices.clear();
    p.vocoder = None;
    p.pitch = 0.0;
    p.denoise = false;
    p.sensitivity_db = -90.0;
    p.boost = 100.0;
    p.bass_db = 0.0;
    p.treble_db = 0.0;
    p
}

struct AiLoop {
    rt: Realtime,
    input: HeapCons<f32>,
    output: Out,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    sample_rate: u32,
}

impl AiLoop {
    fn run(mut self) -> Models {
        crate::priority::pro_audio_thread();
        let n = self.rt.block_len();
        let mut block = vec![0f32; n];
        let mut out = vec![0f32; n];
        let mut post = Processor::new(self.sample_rate);
        let mut seen = 0;
        let mut failures = 0u32;
        let mut work_peak = 0f32;
        while !self.stop.load(Ordering::Acquire) {
            let queued = self.input.occupied_len();
            if queued < n {
                thread::sleep(Duration::from_millis(2));
                continue;
            }
            let started = std::time::Instant::now();
            // Falling behind: drop the oldest audio instead of letting latency grow.
            if queued > n * 3 {
                self.input.skip(queued - n);
            }
            self.input.pop_slice(&mut block);
            self.shared.sync(&mut seen, &mut post, ai_post_params);
            self.rt.transpose = f32::from_bits(self.shared.transpose.load(Ordering::Relaxed));

            if !post.params().enabled {
                out.copy_from_slice(&block);
            } else {
                match self.rt.process(&block, &mut out) {
                    Ok(()) => failures = 0,
                    Err(e) => {
                        out.fill(0.0);
                        failures += 1;
                        if failures == 1 {
                            self.shared.report(format!("Voz IA: {e}"));
                        }
                    }
                }
                post.process_voice(&mut out);
            }
            store_peak(&self.shared.level_out, peak(&out));
            let work = started.elapsed().as_secs_f32() * self.sample_rate as f32 + 2.0 * self.sample_rate as f32 / 1000.0;
            work_peak = work.max(work_peak * 0.98);
            self.shared.ai_work.store(work_peak as u32, Ordering::Relaxed);
            self.output.push(&mut out);
            if let Ok(mut s) = self.shared.ai_stats.lock() {
                *s = Some(self.rt.stats());
            }
        }
        self.rt.into_models()
    }
}

enum Sink {
    /// Classic voices: processed audio goes straight to the output.
    Direct(Out),
    /// AI voice: cleaned-up microphone audio goes to the AI worker.
    Ai(HeapProd<f32>),
}

struct InputState {
    processor: Processor,
    seen: u64,
    mono: Vec<f32>,
    sink: Sink,
    neural: Option<HeapProd<f32>>,
    shared: Arc<Shared>,
}

impl InputState {
    fn handle<T>(&mut self, data: &[T], channels: usize)
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        self.mono.clear();
        let scale = 1.0 / channels as f32;
        for frame in data.chunks(channels) {
            let v = frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() * scale;
            self.mono.push(v);
        }
        self.run();
    }

    fn handle_mono(&mut self, data: &[f32]) {
        self.mono.clear();
        self.mono.extend_from_slice(data);
        self.run();
    }

    fn run(&mut self) {
        self.shared.sync(&mut self.seen, &mut self.processor, |p| p);
        self.processor
            .set_neural_score(self.shared.neural_score(), self.shared.neural_infer_id());
        store_peak(&self.shared.level_in, peak(&self.mono));
        if let Ok(mut slot) = self.shared.calib.try_lock() {
            if let Some(buf) = slot.as_mut() {
                let cap = 48_000 * 20;
                if buf.len() < cap {
                    let room = cap - buf.len();
                    let n = self.mono.len().min(room);
                    buf.extend_from_slice(&self.mono[..n]);
                }
            }
        }
        if let Some(prod) = &mut self.neural {
            let _ = prod.push_slice(&self.mono);
        }

        match &mut self.sink {
            Sink::Direct(out) => {
                self.processor.process(&mut self.mono);
                store_peak(&self.shared.level_out, peak(&self.mono));
                out.push(&mut self.mono);
            }
            Sink::Ai(p) => {
                self.processor.process_input(&mut self.mono);
                p.push_slice(&self.mono);
            }
        }
        self.publish_gate();
    }

    fn publish_gate(&mut self) {
        if !self.processor.params().voice_gate {
            if let Ok(mut slot) = self.shared.gate_snap.try_lock() {
                *slot = None;
            }
            return;
        }
        let events = self
            .processor
            .gate_events()
            .into_iter()
            .map(|e| GateEventDto {
                utterance_id: e.utterance_id,
                prev: e.prev,
                next: e.next,
                speech_hops: e.speech_hops,
                neural: e.neural,
                thresh: e.thresh,
                strict: e.strict,
                protect: e.protect,
                reason: e.reason,
                delay_samples: e.delay_samples,
                infer_id: e.infer_id,
            })
            .collect();
        let phase = self.processor.gate_phase() as u8;
        let reason = self.processor.gate_reason();
        let hops = self.processor.gate_speech_hops();
        // Ears hop ≈ 20 ms.
        let speech_ms = hops as f32 * 20.0;
        let snap = GateSnapshot {
            phase,
            reason,
            reason_label: gate_reason_label(reason).to_string(),
            phase_label: gate_phase_label(phase).to_string(),
            utterance_id: self.processor.gate_utterance_id(),
            neural: self.processor.gate_display_score(),
            thresh: self.processor.gate_thresh(),
            strict: self.processor.gate_strict(),
            protect: self.processor.gate_protect(),
            delay_ms: self.processor.gate_delay_ms(),
            buffered_ms: self.processor.gate_buffered_ms(),
            buffer_capacity_ms: self.processor.gate_buffer_capacity_ms(),
            speech_hops: hops,
            speech_ms,
            speech_env: self.processor.gate_speech_env(),
            infer_id: self.processor.gate_infer_id(),
            infer_ms: f32::from_bits(self.shared.neural_infer_ms.load(Ordering::Relaxed)),
            verify_age_ms: self.processor.gate_verify_age_ms(),
            authorized: self.processor.gate_authorized(),
            events,
        };
        if let Ok(mut slot) = self.shared.gate_snap.try_lock() {
            *slot = Some(snap);
        }
    }
}

fn neural_loop(
    mut cons: HeapCons<f32>,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    model: PathBuf,
    sample_rate: u32,
) {
    let mut embedder = match SpeakerEmbedder::load(&model) {
        Ok(e) => e,
        Err(e) => {
            shared.report(format!("Biometria neural: {e}"));
            shared.clear_neural();
            return;
        }
    };
    let mut speech = SpeechBuffer::new(sample_rate);
    let win = ((sample_rate as u64 * LIVE_SPEECH_MS as u64) / 1000) as usize;
    let mut scratch = vec![0f32; 4096];
    let period = Duration::from_millis(LIVE_PERIOD_MS as u64);
    let mut last = std::time::Instant::now();
    let mut scored_utt = 0u64;
    while !stop.load(Ordering::Acquire) {
        loop {
            let n = cons.pop_slice(&mut scratch);
            if n == 0 {
                break;
            }
            speech.push(&scratch[..n]);
        }
        if shared.neural_pause.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(8));
            continue;
        }
        let (centroid, gallery, cfg) = match shared.params.lock() {
            Ok(p) if p.voice_gate && p.voice_embed.len() == EMBED_DIM => {
                (
                    Some(p.voice_embed.clone()),
                    p.voice_gallery.clone(),
                    shared.generation.load(Ordering::Relaxed),
                )
            }
            _ => (None, Vec::new(), 0),
        };
        let Some(centroid) = centroid else {
            shared.clear_neural();
            scored_utt = 0;
            thread::sleep(Duration::from_millis(8));
            continue;
        };

        // First score as soon as ~280 ms of speech exists; then a sliding window
        // of the last ~1.8 s every LIVE_PERIOD_MS. Never freeze on the first hop.
        let can_score = speech.ready_first()
            || (speech.ready_short() && (speech.frozen() || speech.speech_ms() >= FIRST_SPEECH_MS as f32));
        if !can_score {
            thread::sleep(Duration::from_millis(8));
            continue;
        }

        let utt = speech.utterance_id;
        let due = scored_utt != utt || last.elapsed() >= period;
        if !due {
            thread::sleep(Duration::from_millis(8));
            continue;
        }

        let window = speech.snapshot(win);
        let t0 = std::time::Instant::now();
        shared.neural_busy.store(true, Ordering::Release);
        if shared.neural_pause.load(Ordering::Acquire) {
            shared.neural_busy.store(false, Ordering::Release);
            continue;
        }
        // One sliding window — do not max() a lucky 400 ms tail with the full
        // phrase (that is how a YouTube clip hit 0.41 and opened Locked).
        let wav = to_16k(&window, sample_rate);
        let best = embedder
            .embed(&wav)
            .ok()
            .map(|live| max_cosine_flat(&live, &centroid, &gallery));
        shared.neural_busy.store(false, Ordering::Release);
        if let Some(score) = best {
            let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
            if speech.utterance_id == utt {
                let infer = shared.publish_neural(score, utt, cfg, infer_ms);
                scored_utt = utt;
                last = std::time::Instant::now();
                dump_biometry_capture(utt, infer, sample_rate, &window, &wav, score, &centroid);
            }
        } else {
            last = std::time::Instant::now();
        }
    }
    shared.neural_busy.store(false, Ordering::Release);
    shared.clear_neural();
}

/// Opt-in live↔offline equivalence dump.
/// Set env `KAMOVOX_BIOMETRY_DUMP` to a directory; each inference writes PCM + JSON.
/// Replay: `cargo run -p kamovox-ai --example speaker_diagnostic -- --model … --wav dump.wav`
fn dump_biometry_capture(
    utt: u64,
    infer: u64,
    raw_sr: u32,
    raw_window: &[f32],
    wav16: &[f32],
    score: f32,
    centroid: &[f32],
) {
    let Ok(dir) = std::env::var("KAMOVOX_BIOMETRY_DUMP") else {
        return;
    };
    let path = std::path::PathBuf::from(&dir);
    if dir.is_empty() || (!path.exists() && std::fs::create_dir_all(&path).is_err()) {
        return;
    }
    let stem = path.join(format!("utt{utt}_inf{infer}"));
    let rms = |s: &[f32]| {
        if s.is_empty() {
            0.0
        } else {
            (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
        }
    };
    let mut cent_norm = 0.0f32;
    for &x in centroid {
        cent_norm += x * x;
    }
    cent_norm = cent_norm.sqrt();
    // Raw f32 LE PCM @ 16 kHz for bit-exact offline replay of the live tensor input.
    let mut bytes = Vec::with_capacity(wav16.len() * 4);
    for s in wav16 {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let _ = std::fs::write(stem.with_extension("f32"), bytes);
    let _ = write_wav_f32(stem.with_extension("wav"), wav16, 16_000);
    let meta = format!(
        "{{\n  \"build_id\": \"{BUILD_ID}\",\n  \"utterance_id\": {utt},\n  \"inference_id\": {infer},\n  \
         \"raw_pcm_sample_rate\": {raw_sr},\n  \"raw_pcm_samples\": {},\n  \"raw_pcm_rms\": {:.6},\n  \
         \"resampled_pcm_sample_rate\": 16000,\n  \"resampled_pcm_samples\": {},\n  \"resampled_pcm_rms\": {:.6},\n  \
         \"embedding_dimension\": {},\n  \"centroid_l2_norm\": {:.6},\n  \"cosine_similarity\": {:.6},\n  \
         \"speech_window_ms\": {:.1}\n}}\n",
        raw_window.len(),
        rms(raw_window),
        wav16.len(),
        rms(wav16),
        centroid.len(),
        cent_norm,
        score,
        wav16.len() as f32 * 1000.0 / 16_000.0,
    );
    let _ = std::fs::write(stem.with_extension("json"), meta);
}

fn write_wav_f32(path: std::path::PathBuf, samples: &[f32], sr: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let n = samples.len() as u32;
    let data_bytes = n * 2; // i16
    let mut hdr = Vec::with_capacity(44);
    hdr.extend_from_slice(b"RIFF");
    hdr.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    hdr.extend_from_slice(b"WAVEfmt ");
    hdr.extend_from_slice(&16u32.to_le_bytes());
    hdr.extend_from_slice(&1u16.to_le_bytes()); // PCM
    hdr.extend_from_slice(&1u16.to_le_bytes()); // mono
    hdr.extend_from_slice(&sr.to_le_bytes());
    hdr.extend_from_slice(&(sr * 2).to_le_bytes());
    hdr.extend_from_slice(&2u16.to_le_bytes());
    hdr.extend_from_slice(&16u16.to_le_bytes());
    hdr.extend_from_slice(b"data");
    hdr.extend_from_slice(&data_bytes.to_le_bytes());
    f.write_all(&hdr)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn build_input<T>(device: &Device, config: &StreamConfig, mut state: InputState) -> Result<Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let shared = state.shared.clone();
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| state.handle(data, channels),
            move |err| shared.report(format!("Microfone: {err}")),
            None,
        )
        .map_err(|e| format!("Microfone: {e}"))
}

/// Pulls processed mono audio and linearly resamples it to the output rate.
struct OutputState {
    consumer: HeapCons<f32>,
    step: f64,
    pos: f64,
    cur: f32,
    next: f32,
    /// Audio to accumulate before playing again after running dry.
    cushion: usize,
    max: usize,
    buffering: bool,
    shared: Arc<Shared>,
}

impl OutputState {
    fn new(consumer: HeapCons<f32>, in_rate: u32, out_rate: u32, cushion: usize, max: usize, shared: Arc<Shared>) -> Self {
        Self {
            consumer,
            step: in_rate as f64 / out_rate as f64,
            pos: 0.0,
            cur: 0.0,
            next: 0.0,
            cushion,
            max,
            buffering: true,
            shared,
        }
    }

    fn fill<T>(&mut self, data: &mut [T], channels: usize)
    where
        T: SizedSample + FromSample<f32>,
    {
        let queued = self.consumer.occupied_len();
        if queued > self.max {
            self.consumer.skip(queued - self.cushion);
        }
        if self.buffering && self.consumer.occupied_len() >= self.cushion {
            self.buffering = false;
        }
        for frame in data.chunks_mut(channels) {
            while self.pos >= 1.0 {
                self.cur = self.next;
                self.next = if self.buffering {
                    0.0
                } else {
                    match self.consumer.try_pop() {
                        Some(v) => v,
                        None => {
                            self.buffering = true;
                            0.0
                        }
                    }
                };
                self.pos -= 1.0;
            }
            let v = self.cur + (self.next - self.cur) * self.pos as f32;
            self.pos += self.step;
            let s = T::from_sample(v);
            for out in frame.iter_mut() {
                *out = s;
            }
        }
    }
}

fn build_output<T>(device: &Device, config: &StreamConfig, mut state: OutputState) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let shared = state.shared.clone();
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| state.fill(data, channels),
            move |err| shared.report(format!("Saída: {err}")),
            None,
        )
        .map_err(|e| format!("Saída: {e}"))
}
