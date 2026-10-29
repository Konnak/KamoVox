//! "Only my voice" — zero-delay fail-open gate with CAM++ as impostor veto.
//!
//! Architecture:
//! - Sidechain analysis; wet PCM delayed a **fixed 30 ms** (fade/onset only).
//! - Speech + mic proximity opens immediately (`gain = 1`, bit-exact).
//! - CAM++ never holds the buffer: it only **vetoes** after ≥500 ms of speech
//!   and two low scores in a row.
//! - Laughs, pitch swings and short words stay open when the mic is close
//!   or the owner spoke in the last 2 s.

use std::collections::VecDeque;

use crate::filters::{coef, Biquad};

pub const PRINT_BANDS: usize = 12;
/// Fixed lookahead: enough for a 5 ms fade without eating the first consonant.
pub const LOOKAHEAD_MS: f32 = 30.0;
/// Back-compat aliases — both modes share the 30 ms line.
pub const LOOKAHEAD_FAST_MS: f32 = LOOKAHEAD_MS;
pub const LOOKAHEAD_PROTECTED_MS: f32 = LOOKAHEAD_MS;

const CENTERS: [f32; PRINT_BANDS] = [
    140.0, 200.0, 280.0, 390.0, 540.0, 750.0, 1_040.0, 1_450.0, 2_000.0, 2_700.0, 3_500.0, 4_500.0,
];

const SILENCE: f32 = 0.00010;
/// ~260 ms of inactivity clears a veto / ends an utterance.
const PAUSE_HOPS: u32 = 13;
const HOLD_EXIT_HOPS: u32 = 13;
const REJECT_COOLDOWN_HOPS: u32 = 13;
/// CAM++ (or DSP) may veto only after this much speech (~500 ms).
const VETO_SPEECH_HOPS: u32 = 25;
/// Short words stay fail-open (~600 ms).
const SHORT_WORD_HOPS: u32 = 30;
/// Owner spoke recently — keep expressions open (~2 s).
const OWNER_HOLD_HOPS: u32 = 100;
/// Two consecutive CAM++ misses trigger the impostor veto.
const VETO_FAILS: u32 = 2;

/// Public phase for UI / diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GatePhase {
    Off = 0,
    Closed = 1,
    Candidate = 2,
    Verifying = 3,
    Locked = 4,
    Holding = 5,
    Releasing = 6,
    Rejected = 7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
enum GateState {
    Closed = 1,
    Confirming = 2,
    /// Speech started; buffering. Never audible in Protected.
    Candidate = 3,
    /// Waiting for CAM++. Audible only in Fast mode.
    Verifying = 4,
    Locked = 5,
    Holding = 6,
    Releasing = 7,
    Rejected = 8,
}

#[derive(Clone, Copy, Debug)]
pub struct VoicePrint {
    pub bands: [f32; PRINT_BANDS],
    pub pitch: f32,
    pub pitch_lo: f32,
    pub pitch_hi: f32,
    pub level: f32,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub frames: u32,
}

pub fn build_voice_print(samples: &[f32], sr: f32) -> Option<VoicePrint> {
    if sr < 8_000.0 || samples.len() < (sr as usize) * 2 {
        return None;
    }
    let mut ears = Ears::new(sr);
    let mut spectra = Vec::new();
    let mut pitches = Vec::new();
    let mut levels = Vec::new();
    let mut f1s = Vec::new();
    let mut f2s = Vec::new();
    let mut f3s = Vec::new();
    for &x in samples {
        if let Some(frame) = ears.push(x) {
            if frame.voiced {
                if let Some(pitch) = frame.pitch {
                    spectra.push(frame.bands);
                    pitches.push(pitch);
                    levels.push(frame.energy.sqrt());
                    f1s.push(frame.f1);
                    f2s.push(frame.f2);
                    f3s.push(frame.f3);
                }
            }
        }
    }
    if pitches.len() < 60 {
        return None;
    }
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = pitches.len();
    let pitch = pitches[n / 2];
    let mut lo = pitches[n / 10] * 0.88;
    let mut hi = pitches[n * 9 / 10] * 1.12;
    if hi < lo + 5.0 {
        lo = pitch * 0.82;
        hi = pitch * 1.22;
    }
    let level = levels[n / 2].max(1e-4);
    let f1 = median(&mut f1s);
    let f2 = median(&mut f2s);
    let f3 = median(&mut f3s);

    let mut mean = [0.0; PRINT_BANDS];
    for spec in &spectra {
        for (m, s) in mean.iter_mut().zip(spec) {
            *m += *s;
        }
    }
    let sn = spectra.len() as f32;
    for m in &mut mean {
        *m /= sn;
    }
    normalize(&mut mean);
    Some(VoicePrint {
        bands: mean,
        pitch,
        pitch_lo: lo,
        pitch_hi: hi,
        level,
        f1,
        f2,
        f3,
        frames: pitches.len() as u32,
    })
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

pub struct VoiceGate {
    on: bool,
    print: [f32; PRINT_BANDS],
    pitch_lo: f32,
    pitch_hi: f32,
    level: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    strict: f32,
    neural_lo: f32,
    neural_hi: f32,
    /// Sustained cosine below this ⇒ CONFIRMED_OTHER (not mere non-accept).
    neural_reject: f32,
    neural_enrolled: bool,
    /// Protected = confirm-before-transmit; Fast = may soft-open.
    protect: bool,
    sr: f32,
    ears: Ears,
    state: GateState,
    confidence: f32,
    gain: f32,
    target: f32,
    open: f32,
    close: f32,
    silence_hops: u32,
    foreign_hops: u32,
    confirm_hops: u32,
    speech_env: f32,
    neural: Option<f32>,
    utterance_vetoed: bool,
    delay: VecDeque<f32>,
    delay_len: usize,
    /// Last decision reason code for diagnostics.
    reason: u8,
    /// Speech hops accumulated in the current Candidate/Verifying utterance.
    speech_hops: u32,
    /// Monotonic id for the open utterance (diagnostics / neural sync).
    utterance_id: u64,
    /// Last inference id observed from the neural worker.
    last_infer_id: u64,
    /// Only accept scores with `infer_id > accept_after` (drops stale CAM++ results).
    accept_after: u64,
    /// Compact ring of recent gate decisions (no PCM / embeddings).
    diag: [GateEvent; 16],
    diag_i: usize,
    diag_n: usize,
    /// Samples removed from the delay line (emitted or discarded) since arm.
    popped: u64,
    /// Input index of the sample emitted by the last `process` call (MAX = none).
    last_src: u64,
    /// Score that authorized the current Locked/Holding utterance (diagnostics).
    confirmed_score: Option<f32>,
    /// Hops since the last *new* CAM++ infer (Locked included — must not freeze).
    verify_hops: u32,
    /// Last raw cosine (not EMA) from a new infer_id.
    neural_raw: Option<f32>,
    pass_streak: u32,
    fail_streak: u32,
    applied_infer: u64,
    last_pitch: f32,
    rms_ema: f32,
    owner_hold_hops: u32,
    express_hops: u32,
}

/// One Speaker Gate decision for the diagnostic report (no private audio).
#[derive(Clone, Copy, Debug, Default)]
pub struct GateEvent {
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

impl VoiceGate {
    pub fn new(sr: f32) -> Self {
        let delay_len = ((sr * LOOKAHEAD_MS / 1000.0) as usize).max(1);
        Self {
            on: false,
            print: [0.0; PRINT_BANDS],
            pitch_lo: 0.0,
            pitch_hi: 0.0,
            level: 0.05,
            f1: 500.0,
            f2: 1_500.0,
            f3: 2_500.0,
            strict: 0.55,
            neural_lo: 0.18,
            neural_hi: 0.40,
            neural_reject: 0.08,
            neural_enrolled: false,
            protect: true,
            sr,
            ears: Ears::new(sr),
            state: GateState::Closed,
            confidence: 0.0,
            gain: 0.0,
            target: 0.0,
            open: coef(sr, 90.0),
            close: coef(sr, 4.0),
            silence_hops: 0,
            foreign_hops: 0,
            confirm_hops: 0,
            speech_env: 0.0,
            neural: None,
            utterance_vetoed: false,
            delay: VecDeque::with_capacity(delay_len + 8),
            delay_len,
            reason: 0,
            speech_hops: 0,
            utterance_id: 0,
            last_infer_id: 0,
            accept_after: 0,
            diag: [GateEvent::default(); 16],
            diag_i: 0,
            diag_n: 0,
            popped: 0,
            last_src: u64::MAX,
            confirmed_score: None,
            verify_hops: 0,
            neural_raw: None,
            pass_streak: 0,
            fail_streak: 0,
            applied_infer: 0,
            last_pitch: 0.0,
            rms_ema: 0.0,
            owner_hold_hops: 0,
            express_hops: 0,
        }
    }

    /// Input sample index (counted from the last hard reset) emitted by the
    /// most recent `process` call, if any. Used for offline alignment tests.
    pub fn last_source_index(&self) -> Option<u64> {
        (self.last_src != u64::MAX).then_some(self.last_src)
    }

    pub fn buffered_samples(&self) -> usize {
        self.delay.len()
    }

    pub fn buffered_ms(&self) -> f32 {
        self.delay.len() as f32 * 1000.0 / self.sr.max(1.0)
    }

    pub fn buffer_capacity_ms(&self) -> f32 {
        self.delay_ms()
    }

    pub fn speech_hops(&self) -> u32 {
        self.speech_hops
    }

    pub fn speech_env(&self) -> f32 {
        self.speech_env
    }

    pub fn utterance_id(&self) -> u64 {
        self.utterance_id
    }

    /// Live score if present, else the score that opened Locked (stable in UI).
    pub fn display_score(&self) -> Option<f32> {
        self.neural.or(self.confirmed_score)
    }

    pub fn authorized(&self) -> bool {
        matches!(self.state, GateState::Locked | GateState::Holding)
            && self.gain >= 1.0 - 1e-6
    }

    pub fn verify_age_ms(&self) -> f32 {
        self.verify_hops as f32 * 20.0
    }

    /// Apply a CAM++ cosine tagged with a monotonic `infer_id`.
    /// Scores with `infer_id <= accept_after` are ignored (stale / wrong utterance).
    #[cfg(test)]
    pub fn set_neural(&mut self, score: Option<f32>) {
        self.set_neural_scored(score, self.last_infer_id.saturating_add(1));
    }

    pub fn set_neural_scored(&mut self, score: Option<f32>, infer_id: u64) {
        if infer_id > 0 {
            self.last_infer_id = self.last_infer_id.max(infer_id);
        }
        if infer_id <= self.accept_after {
            return;
        }
        let Some(s) = score.map(|v| v.clamp(-1.0, 1.0)) else {
            return;
        };
        self.neural = Some(match self.neural {
            Some(prev) => prev * 0.40 + s * 0.60,
            None => s,
        });
        self.neural_raw = Some(s);
        if infer_id != self.applied_infer {
            self.applied_infer = infer_id;
            self.verify_hops = 0;
            let need = self.neural_need();
            if s >= need {
                self.pass_streak = self.pass_streak.saturating_add(1);
                self.fail_streak = 0;
            } else {
                self.pass_streak = 0;
                self.fail_streak = self.fail_streak.saturating_add(1);
            }
        }
    }

    fn invalidate_neural(&mut self) {
        self.accept_after = self.last_infer_id;
        self.neural = None;
        self.neural_raw = None;
        self.pass_streak = 0;
        self.fail_streak = 0;
    }

    fn neural_below(&self) -> bool {
        let Some(raw) = self.neural_raw.or(self.neural) else {
            return false;
        };
        raw < self.neural_need()
    }

    fn push_diag(&mut self, prev: GateState, next: GateState) {
        if prev == next {
            return;
        }
        let ev = GateEvent {
            utterance_id: self.utterance_id,
            prev: prev as u8,
            next: next as u8,
            speech_hops: self.speech_hops,
            neural: self.neural.unwrap_or(f32::NAN),
            thresh: self.neural_need(),
            strict: self.strict,
            protect: self.protect,
            reason: self.reason,
            delay_samples: self.delay.len() as u32,
            infer_id: self.last_infer_id,
        };
        self.diag[self.diag_i % 16] = ev;
        self.diag_i = self.diag_i.wrapping_add(1);
        self.diag_n = (self.diag_n + 1).min(16);
    }

    /// Recent gate transitions (oldest→newest), for UI / logs.
    pub fn diagnostic_events(&self) -> Vec<GateEvent> {
        let n = self.diag_n.min(16);
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        let start = self.diag_i.wrapping_sub(n);
        for i in 0..n {
            out.push(self.diag[(start + i) % 16]);
        }
        out
    }

    pub fn phase(&self) -> GatePhase {
        if !self.on {
            return GatePhase::Off;
        }
        match self.state {
            GateState::Closed => GatePhase::Closed,
            GateState::Confirming | GateState::Candidate => GatePhase::Candidate,
            GateState::Verifying => GatePhase::Verifying,
            GateState::Locked => GatePhase::Locked,
            GateState::Holding => GatePhase::Holding,
            GateState::Releasing => GatePhase::Releasing,
            GateState::Rejected => GatePhase::Rejected,
        }
    }

    #[allow(dead_code)]
    pub fn reason_code(&self) -> u8 {
        self.reason
    }

    #[allow(dead_code)]
    pub fn neural_score(&self) -> Option<f32> {
        self.neural
    }

    pub fn delay_ms(&self) -> f32 {
        self.delay_len as f32 * 1000.0 / self.sr.max(1.0)
    }

    pub fn neural_thresh(&self) -> f32 {
        self.neural_need()
    }

    pub fn is_protect(&self) -> bool {
        self.protect
    }

    pub fn last_infer_id(&self) -> u64 {
        self.last_infer_id
    }

    pub fn set(
        &mut self,
        on: bool,
        print: [f32; PRINT_BANDS],
        pitch: f32,
        pitch_lo: f32,
        pitch_hi: f32,
        level: f32,
        f1: f32,
        f2: f32,
        f3: f32,
        strict: f32,
        neural_enrolled: bool,
        neural_lo: f32,
        neural_hi: f32,
        neural_reject: f32,
        protect: bool,
    ) {
        let ready = print.iter().any(|v| v.abs() > 1e-4) && pitch > 60.0;
        let new_on = on && ready;
        let new_protect = neural_enrolled && protect;
        let (lo, hi) = if pitch_lo > 60.0 && pitch_hi > pitch_lo {
            (pitch_lo, pitch_hi)
        } else {
            (pitch * 0.82, pitch * 1.22)
        };
        let new_lo = if neural_lo > 0.05 { neural_lo } else { 0.18 };
        let new_hi = {
            let h = if neural_hi > new_lo { neural_hi } else { 0.50 };
            h.clamp((new_lo + 0.14).max(0.48), 0.55)
        };
        let new_reject = if neural_reject > 0.04 && neural_reject < new_lo {
            neural_reject
        } else {
            (new_lo - 0.10).clamp(0.05, 0.18)
        };

        // Hard reset only when identity / mode / enrollment changes — NOT when
        // the user only moves the exigência slider (that was killing short phrases).
        let hard = new_on != self.on
            || neural_enrolled != self.neural_enrolled
            || new_protect != self.protect
            || (pitch - self.pitch_lo).abs() > 80.0 && self.pitch_lo > 0.0
            || print
                .iter()
                .zip(self.print.iter())
                .any(|(a, b)| (a - b).abs() > 0.15);

        self.on = new_on;
        self.print = print;
        self.pitch_lo = lo;
        self.pitch_hi = hi;
        self.level = if level > 1e-4 { level } else { 0.05 };
        self.f1 = if f1 > 80.0 { f1 } else { 500.0 };
        self.f2 = if f2 > 200.0 { f2 } else { 1_500.0 };
        self.f3 = if f3 > 400.0 { f3 } else { 2_500.0 };
        self.strict = strict.clamp(0.0, 1.0);
        self.neural_enrolled = neural_enrolled;
        self.neural_lo = new_lo;
        self.neural_hi = new_hi;
        self.neural_reject = new_reject;
        self.protect = new_protect;

        let want = ((self.sr * LOOKAHEAD_MS / 1000.0) as usize).max(1);
        if want != self.delay_len {
            self.delay_len = want;
            if hard {
                self.delay = VecDeque::with_capacity(want + 8);
            }
        }

        if hard {
            self.state = GateState::Closed;
            self.confidence = 0.0;
            self.silence_hops = 0;
            self.foreign_hops = 0;
            self.confirm_hops = 0;
            self.speech_env = 0.0;
            self.utterance_vetoed = false;
            self.reason = 0;
            self.speech_hops = 0;
            self.invalidate_neural();
            self.confirmed_score = None;
            self.verify_hops = 0;
            self.pass_streak = 0;
            self.fail_streak = 0;
            self.applied_infer = 0;
            self.last_pitch = 0.0;
            self.rms_ema = 0.0;
            self.owner_hold_hops = 0;
            self.express_hops = 0;
            self.delay.clear();
            self.popped = 0;
            self.last_src = u64::MAX;
            if self.on {
                self.gain = 0.0;
                self.target = 0.0;
            } else {
                self.gain = 1.0;
                self.target = 1.0;
            }
        } else if !self.on {
            self.gain = 1.0;
            self.target = 1.0;
            self.state = GateState::Closed;
        }
    }

    fn neural_need(&self) -> f32 {
        self.neural_lo + self.strict * (self.neural_hi - self.neural_lo)
    }

    fn open_exact(&mut self) {
        self.target = 1.0;
        self.gain = 1.0;
    }

    fn mute(&mut self) {
        self.target = 0.0;
    }

    fn emit_sample(&mut self, out: f32) -> f32 {
        self.last_src = self.popped;
        self.popped = self.popped.wrapping_add(1);
        if self.gain >= 1.0 - 1e-6 {
            out
        } else if self.gain <= 1e-6 {
            0.0
        } else {
            out * self.gain
        }
    }

    /// Sidechain analyze `raw`; emit delayed `wet` (original path, never FFT'd).
    pub fn process(&mut self, raw: f32, wet: f32) -> f32 {
        if !self.on {
            return wet;
        }
        self.last_src = u64::MAX;
        self.observe(raw);
        while self.delay.len() < self.delay_len {
            self.delay.push_back(0.0);
        }
        self.delay.push_back(wet);
        let out = self.delay.pop_front().unwrap_or(0.0);
        self.emit_sample(out)
    }

    fn observe(&mut self, x: f32) {
        if let Some(frame) = self.ears.push(x) {
            self.step_frame(&frame);
        }
        let open = matches!(self.state, GateState::Locked | GateState::Holding);
        if open && self.target >= 1.0 - 1e-6 {
            self.gain = 1.0;
            return;
        }
        let k = if self.target > self.gain { self.open } else { self.close };
        self.gain += (self.target - self.gain) * k;
        self.gain = self.gain.clamp(0.0, 1.0);
    }

    fn step_frame(&mut self, frame: &Frame) {
        let quiet = frame.energy < SILENCE;
        self.speech_env += ((if quiet { 0.0 } else { 1.0 }) - self.speech_env) * 0.18;
        let inactive = self.speech_env < 0.15;
        let rms = frame.energy.sqrt();
        self.rms_ema = if self.rms_ema <= 0.0 {
            rms
        } else {
            self.rms_ema * 0.92 + rms * 0.08
        };

        let mut frame_alike = self.confidence;
        if !quiet && frame.voiced {
            frame_alike = match_alike(
                &self.print,
                self.pitch_lo,
                self.pitch_hi,
                self.level,
                self.f1,
                self.f2,
                self.f3,
                rms,
                frame,
            );
            self.confidence = self.confidence * 0.72 + frame_alike * 0.28;
        }

        let speech = !quiet && frame.energy > SILENCE * 3.0;
        let proximity = speech && rms >= self.level * 0.22;
        let pitch_jump = match frame.pitch {
            Some(hz) if self.last_pitch > 60.0 => {
                let ratio = (hz / self.last_pitch).max(self.last_pitch / hz);
                ratio > 1.55
            }
            _ => false,
        };
        if let Some(hz) = frame.pitch {
            self.last_pitch = hz;
        }
        let laugh = speech && rms >= self.level * 0.85 && !frame.voiced;
        if laugh || pitch_jump {
            self.express_hops = 15;
        } else if self.express_hops > 0 {
            self.express_hops -= 1;
        }
        let expressive = laugh || self.express_hops > 0 || self.speech_hops < SHORT_WORD_HOPS;
        let fail_open = (proximity || self.owner_hold_hops > 0) && expressive;
        let energy_spike = speech && rms >= self.level * 1.6 && rms > self.rms_ema * 2.0;

        if speech {
            self.speech_hops = self.speech_hops.saturating_add(1);
            self.silence_hops = 0;
        } else {
            self.silence_hops = self.silence_hops.saturating_add(1);
        }
        self.verify_hops = self.verify_hops.saturating_add(1);

        let long_enough = self.speech_hops >= VETO_SPEECH_HOPS;
        let neural_veto = self.neural_enrolled
            && long_enough
            && self.fail_streak >= VETO_FAILS
            && self.neural_below()
            && !fail_open;
        let dsp_veto = !self.neural_enrolled
            && long_enough
            && !proximity
            && frame.voiced
            && frame_alike < 0.42
            && !fail_open;

        let prev_state = self.state;
        match self.state {
            GateState::Closed | GateState::Confirming | GateState::Candidate | GateState::Verifying => {
                if speech && proximity {
                    self.state = GateState::Locked;
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                    self.reason = 1;
                    self.utterance_vetoed = false;
                    self.confirmed_score = self.neural_raw.or(self.neural);
                    self.open_exact();
                } else {
                    self.state = GateState::Closed;
                    self.mute();
                    if inactive {
                        self.reason = 5;
                    }
                }
            }
            GateState::Locked | GateState::Holding => {
                if neural_veto || dsp_veto {
                    self.state = GateState::Rejected;
                    self.reason = 3;
                    self.utterance_vetoed = true;
                    self.confirmed_score = None;
                    self.owner_hold_hops = 0;
                    self.express_hops = 0;
                    self.mute();
                } else if self.silence_hops >= HOLD_EXIT_HOPS {
                    self.state = GateState::Releasing;
                    self.reason = 7;
                    self.mute();
                } else if self.silence_hops >= PAUSE_HOPS {
                    self.state = GateState::Holding;
                    self.reason = 7;
                    self.open_exact();
                } else {
                    self.state = GateState::Locked;
                    self.reason = 1;
                    self.utterance_vetoed = false;
                    self.confirmed_score = self.neural_raw.or(self.neural);
                    self.open_exact();
                }
            }
            GateState::Rejected => {
                self.mute();
                self.utterance_vetoed = true;
                self.confirmed_score = None;
                if energy_spike {
                    self.state = GateState::Locked;
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                    self.reason = 1;
                    self.utterance_vetoed = false;
                    self.speech_hops = 1;
                    self.fail_streak = 0;
                    self.pass_streak = 0;
                    self.open_exact();
                } else if self.silence_hops >= REJECT_COOLDOWN_HOPS {
                    self.state = GateState::Closed;
                    self.reason = 12;
                    self.speech_hops = 0;
                    self.fail_streak = 0;
                } else {
                    self.reason = 3;
                }
            }
            GateState::Releasing => {
                if speech && (proximity || energy_spike) {
                    self.state = GateState::Locked;
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                    self.reason = 1;
                    self.utterance_vetoed = false;
                    self.open_exact();
                } else {
                    self.mute();
                    if self.gain < 0.02 {
                        self.state = GateState::Closed;
                        self.reason = 5;
                    }
                }
            }
        }

        if matches!(self.state, GateState::Locked | GateState::Holding) {
            self.owner_hold_hops = OWNER_HOLD_HOPS;
        } else if self.owner_hold_hops > 0 {
            self.owner_hold_hops -= 1;
        }
        if self.silence_hops >= PAUSE_HOPS {
            self.speech_hops = 0;
            self.fail_streak = 0;
            self.pass_streak = 0;
            self.utterance_vetoed = false;
        }
        self.push_diag(prev_state, self.state);
    }
}

fn match_alike(
    print: &[f32; PRINT_BANDS],
    pitch_lo: f32,
    pitch_hi: f32,
    enroll_level: f32,
    e_f1: f32,
    e_f2: f32,
    e_f3: f32,
    live_rms: f32,
    frame: &Frame,
) -> f32 {
    let timbre = cosine(print, &frame.bands).clamp(-1.0, 1.0);
    let timbre_match = ((timbre + 0.05) / 1.05).clamp(0.0, 1.0);

    let pitch_match = match frame.pitch {
        Some(heard) if heard > 60.0 && pitch_lo > 60.0 => {
            if heard >= pitch_lo && heard <= pitch_hi {
                1.0
            } else {
                let edge = if heard < pitch_lo { pitch_lo } else { pitch_hi };
                let st = (12.0 * (heard / edge).ln() / std::f32::consts::LN_2).abs();
                (1.0 - st / 3.5).clamp(0.0, 1.0)
            }
        }
        _ => 0.0,
    };

    // Formant biometric: same f0 males diverge here.
    let formant_match = formant_score(e_f1, e_f2, e_f3, frame.f1, frame.f2, frame.f3);

    // Timbre + formants dominate; pitch alone must never open the lock.
    let mut alike = pitch_match * 0.20 + timbre_match * 0.35 + formant_match * 0.45;
    // Both spectral cues must agree enough — blocks same-f0 / different tract.
    alike = alike.min(timbre_match * 0.55 + formant_match * 0.45 + pitch_match * 0.15);

    // Level: far-field quieter → reject. Near-field same level ≠ automatic pass.
    let ratio = (live_rms / enroll_level.max(1e-4)).clamp(0.0, 2.5);
    if ratio < 0.28 {
        let lm = (ratio / 0.28).clamp(0.0, 1.0);
        alike *= lm * lm;
    }

    alike.clamp(0.0, 1.0)
}

fn formant_score(e1: f32, e2: f32, e3: f32, f1: f32, f2: f32, f3: f32) -> f32 {
    let d = |a: f32, b: f32| ((a - b).abs() / a.max(80.0)).min(1.5);
    let err = (d(e1, f1) + d(e2, f2) + d(e3, f3)) / 3.0;
    (1.0 - err / 0.85).clamp(0.0, 1.0)
}

fn cosine(a: &[f32; PRINT_BANDS], b: &[f32; PRINT_BANDS]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Frame {
    bands: [f32; PRINT_BANDS],
    pitch: Option<f32>,
    energy: f32,
    voiced: bool,
    f1: f32,
    f2: f32,
    f3: f32,
}

struct Ears {
    sr: f32,
    filters: [Biquad; PRINT_BANDS],
    hop: usize,
    seen: usize,
    energy: [f32; PRINT_BANDS],
    window: Vec<f32>,
    pos: usize,
    filled: bool,
    prev: f32,
    zcr_count: u32,
    high_e: f32,
    total_e: f32,
}

impl Ears {
    fn new(sr: f32) -> Self {
        Self {
            sr,
            filters: CENTERS.map(|hz| Biquad::bandpass(sr, hz, 2.0)),
            hop: ((sr * 0.02) as usize).max(64),
            seen: 0,
            energy: [0.0; PRINT_BANDS],
            window: vec![0.0; 2048],
            pos: 0,
            filled: false,
            prev: 0.0,
            zcr_count: 0,
            high_e: 0.0,
            total_e: 0.0,
        }
    }

    fn push(&mut self, x: f32) -> Option<Frame> {
        self.window[self.pos] = x;
        self.pos += 1;
        if self.pos == self.window.len() {
            self.pos = 0;
            self.filled = true;
        }
        if self.prev * x < 0.0 {
            self.zcr_count += 1;
        }
        self.prev = x;
        self.total_e += x * x;
        for (i, (band, slot)) in self.filters.iter_mut().zip(self.energy.iter_mut()).enumerate() {
            let y = band.process(x);
            let e = y * y;
            *slot += e;
            if i >= 8 {
                self.high_e += e;
            }
        }
        self.seen += 1;
        if self.seen < self.hop {
            return None;
        }
        let n = self.seen as f32;
        self.seen = 0;
        let raw = self.energy;
        let mut bands = [0.0; PRINT_BANDS];
        for (b, e) in bands.iter_mut().zip(raw) {
            *b = (e / n + 1e-8).ln();
        }
        self.energy = [0.0; PRINT_BANDS];
        normalize(&mut bands);
        let (f1, f2, f3) = estimate_formants(&raw);
        let energy = window_energy(&self.window);
        let zcr = self.zcr_count as f32 / n;
        let high_ratio = self.high_e / (self.high_e + self.total_e * 0.15 + 1e-12);
        self.zcr_count = 0;
        self.high_e = 0.0;
        self.total_e = 0.0;

        let pitch = if self.filled { frame_pitch(&self.window, self.pos, self.sr) } else { None };
        let voiced = matches!(pitch, Some(_)) && zcr < 0.35 && high_ratio < 0.85;

        Some(Frame { bands, pitch, energy, voiced, f1, f2, f3 })
    }
}

/// Energy-weighted centroid in each formant band (shifts with spectral tilt).
fn estimate_formants(raw: &[f32; PRINT_BANDS]) -> (f32, f32, f32) {
    let centroid = |lo_hz: f32, hi_hz: f32| -> f32 {
        let mut wsum = 0.0f32;
        let mut esum = 0.0f32;
        for (i, &e) in raw.iter().enumerate() {
            let hz = CENTERS[i];
            if hz >= lo_hz && hz <= hi_hz {
                wsum += hz * e;
                esum += e;
            }
        }
        if esum > 1e-12 {
            wsum / esum
        } else {
            (lo_hz + hi_hz) * 0.5
        }
    };
    (centroid(200.0, 1_000.0), centroid(800.0, 2_500.0), centroid(1_800.0, 4_000.0))
}

fn window_energy(window: &[f32]) -> f32 {
    let sum: f32 = window.iter().map(|s| s * s).sum();
    sum / window.len() as f32
}

fn normalize(v: &mut [f32; PRINT_BANDS]) {
    let mean = v.iter().sum::<f32>() / PRINT_BANDS as f32;
    for x in v.iter_mut() {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn frame_pitch(window: &[f32], pos: usize, sr: f32) -> Option<f32> {
    let n = window.len();
    let mut energy = 0.0;
    for &s in window {
        energy += s * s;
    }
    if energy / (n as f32) < 0.00012 {
        return None;
    }
    let min_lag = (sr / 400.0) as usize;
    let max_lag = ((sr / 70.0) as usize).min(n / 2);
    if max_lag <= min_lag + 2 {
        return None;
    }
    let at = |i: usize| window[(pos + i) % n];
    let mut running = 0.0;
    let mut lag = 0;
    let mut best = 1.0;
    for candidate in 1..=max_lag {
        let mut sum = 0.0;
        let mut i = 0;
        while i + candidate < n {
            let d = at(i) - at(i + candidate);
            sum += d * d;
            i += 2;
        }
        running += sum;
        if candidate < min_lag {
            continue;
        }
        let norm = if running > 0.0 { sum * candidate as f32 / running } else { 1.0 };
        if lag == 0 && norm < 0.2 {
            best = norm;
            lag = candidate;
        }
        if lag > 0 && candidate <= lag + 4 && norm < best {
            best = norm;
            lag = candidate;
        }
        if lag > 0 && candidate > lag + 4 {
            break;
        }
    }
    if lag == 0 {
        return None;
    }
    Some(sr / lag as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(sr: f32, hz: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (0.55 * (std::f32::consts::TAU * hz * t).sin()
                    + 0.28 * (std::f32::consts::TAU * hz * 2.0 * t).sin()
                    + 0.12 * (std::f32::consts::TAU * hz * 3.0 * t).sin())
                    * 0.4
            })
            .collect()
    }

    /// Same f0, energy pushed into higher formants (another male tract).
    fn other_male(sr: f32, hz: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let glottal = (0..12)
                    .map(|h| {
                        let k = (h + 1) as f32;
                        // Emphasize 2–4 kHz region (higher F2/F3).
                        let w = if k * hz > 1_800.0 { 0.55 } else if k * hz > 900.0 { 0.35 } else { 0.08 };
                        w / k * (std::f32::consts::TAU * hz * k * t).sin()
                    })
                    .sum::<f32>();
                glottal * 0.45
            })
            .collect()
    }

    fn quiet_voice(sr: f32, hz: f32, seconds: f32, amp: f32) -> Vec<f32> {
        voice(sr, hz, seconds).into_iter().map(|x| x * amp).collect()
    }

    fn word_with_fricative_ending(sr: f32, hz: f32) -> Vec<f32> {
        let mut v = voice(sr, hz, 0.45);
        let n = (sr * 0.18) as usize;
        let mut state = 0x1234_5678u32;
        for i in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let fade = 1.0 - (i as f32 / n as f32);
            let t = i as f32 / sr;
            let vow = 0.2 * fade * (std::f32::consts::TAU * hz * t).sin();
            v.push(vow + noise * 0.12 * (0.3 + 0.7 * (1.0 - fade)));
        }
        v
    }

    fn spoken(sr: f32, center: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let hz = center * (1.0 + 0.08 * (std::f32::consts::TAU * 3.5 * t).sin());
            phase += std::f32::consts::TAU * hz / sr;
            let y = 0.55 * phase.sin() + 0.28 * (phase * 2.0).sin() + 0.12 * (phase * 3.0).sin();
            out.push(y * 0.4);
        }
        out
    }

    fn arm(gate: &mut VoiceGate, print: &VoicePrint, strict: f32) {
        gate.set(
            true,
            print.bands,
            print.pitch,
            print.pitch_lo,
            print.pitch_hi,
            print.level,
            print.f1,
            print.f2,
            print.f3,
            strict,
            false,
            0.0,
            0.0,
            0.0,
            false,
        );
    }

    fn kept_ratio(gate: &mut VoiceGate, samples: &[f32]) -> f32 {
        let skip = samples.len() / 3;
        let mut kept = 0.0;
        let mut dry = 0.0;
        for (i, &x) in samples.iter().enumerate() {
            let y = gate.process(x, x);
            if i >= skip {
                kept += y * y;
                dry += x * x;
            }
        }
        kept / dry.max(1e-12)
    }

    #[test]
    fn print_finds_the_spoken_pitch() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 210.0, 2.5), sr).expect("print");
        assert!((print.pitch - 210.0).abs() < 25.0);
        assert!(print.f1 > 80.0 && print.f2 > print.f1);
    }

    #[test]
    fn locked_speech_is_bit_exact_passthrough() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.4);
        let samples = voice(sr, 160.0, 1.8);
        let delay = gate.delay_len;
        // Warm until locked.
        for &x in &samples[..samples.len() / 2] {
            let _ = gate.process(x, x);
        }
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding),
            "expected lock, got {:?}",
            gate.state
        );
        // Remaining samples: out[i] must equal in[i - delay] exactly.
        let start = samples.len() / 2;
        for i in start..samples.len() {
            let y = gate.process(samples[i], samples[i]);
            if matches!(gate.state, GateState::Locked | GateState::Holding) && gate.gain >= 1.0 - 1e-6 {
                let src = samples[i - delay];
                assert!((y - src).abs() < 1e-6, "not bit-exact at {i}: {y} vs {src}");
            }
        }
    }

    #[test]
    fn far_field_other_male_stays_closed() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 130.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.85);
        let far: Vec<f32> = other_male(sr, 130.0, 1.5).into_iter().map(|x| x * 0.12).collect();
        let ratio = kept_ratio(&mut gate, &far);
        assert!(ratio < 0.15, "far-field other male leaked {ratio}");
    }

    #[test]
    fn delay_line_is_about_thirty_ms() {
        let sr = 48_000.0;
        let gate = VoiceGate::new(sr);
        let ms = gate.delay_ms();
        assert!((ms - 30.0).abs() < 1.5, "lookahead {ms} ms");
        assert!((gate.buffer_capacity_ms() - 30.0).abs() < 1.5);
    }

    #[test]
    fn mid_distance_peaks_do_not_open() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 150.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.7);
        // Short bursts (~40 ms) of quiet other speech — must stay closed.
        let mut samples = Vec::new();
        for _ in 0..8 {
            samples.extend(quiet_voice(sr, 150.0, 0.04, 0.12));
            samples.extend(std::iter::repeat(0.0).take((sr * 0.2) as usize));
        }
        let ratio = kept_ratio(&mut gate, &samples);
        assert!(ratio < 0.08, "mid-distance peaks leaked {ratio}");
    }

    #[test]
    fn own_voice_stays_open() {
        let sr = 48_000.0;
        let print = build_voice_print(&spoken(sr, 180.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.45);
        assert!(kept_ratio(&mut gate, &spoken(sr, 180.0, 1.5)) > 0.7);
    }

    #[test]
    fn fricative_word_ending_keeps_gain_near_one() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.5);
        let word = word_with_fricative_ending(sr, 160.0);
        let mut min_g = 1.0f32;
        let warmup = (sr * 0.35) as usize;
        for (i, &x) in word.iter().enumerate() {
            let _ = gate.process(x, x);
            if i >= warmup {
                min_g = min_g.min(gate.gain);
            }
        }
        assert!(min_g > 0.85, "ending gain {min_g}");
    }

    #[test]
    fn quiet_background_voice_is_blocked() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.7);
        assert!(kept_ratio(&mut gate, &quiet_voice(sr, 160.0, 1.5, 0.08)) < 0.15);
    }

    #[test]
    fn close_mic_low_pitch_stays_open() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 220.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.85);
        let ratio = kept_ratio(&mut gate, &voice(sr, 110.0, 1.2));
        assert!(ratio > 0.55, "voz mais grave no mic próximo cortada ({ratio})");
    }

    #[test]
    fn portuguese_like_phrase_stays_locked_across_words() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 155.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.45);
        let gap = |ms: f32| vec![0.0f32; (sr * ms / 1000.0) as usize];
        let mut phrase = Vec::new();
        phrase.extend(word_with_fricative_ending(sr, 155.0));
        phrase.extend(gap(120.0));
        phrase.extend(word_with_fricative_ending(sr, 155.0));
        phrase.extend(gap(90.0));
        phrase.extend(voice(sr, 155.0, 0.5));
        let warmup = (sr * 0.6) as usize;
        let mut speech_gains = Vec::new();
        let hop = 960usize;
        for (i, &x) in phrase.iter().enumerate() {
            let _ = gate.process(x, x);
            if i < warmup {
                continue;
            }
            if i % hop == hop - 1 {
                let chunk = &phrase[i + 1 - hop..=i];
                let e: f32 = chunk.iter().map(|s| s * s).sum::<f32>() / hop as f32;
                if e > 0.0004 {
                    speech_gains.push(gate.gain);
                }
            }
        }
        assert!(!speech_gains.is_empty());
        let min_g = speech_gains.iter().cloned().fold(1.0f32, f32::min);
        assert!(min_g > 0.85, "pipoca {min_g}");
    }

    #[test]
    fn recording_151_quiet_blocked_loud_passes() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/bg_leak.f32");
        let Ok(raw) = std::fs::read(path) else {
            return;
        };
        let samples: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sr = 48_000.0f32;
        let sr_i = sr as usize;
        let enroll = &samples[sr_i * 185 / 10..sr_i * 230 / 10];
        let Some(print) = build_voice_print(enroll, sr) else {
            return;
        };
        let mut g1 = VoiceGate::new(sr);
        arm(&mut g1, &print, 0.7);
        let quiet = kept_ratio(&mut g1, &samples[sr_i / 2..sr_i * 35 / 10]);
        let mut g2 = VoiceGate::new(sr);
        arm(&mut g2, &print, 0.55);
        let loud = kept_ratio(&mut g2, &samples[sr_i * 315 / 10..sr_i * 332 / 10]);
        assert!(quiet < 0.25, "quiet {quiet}");
        assert!(loud > 0.45, "loud {loud}");
    }

    fn arm_neural(gate: &mut VoiceGate, print: &VoicePrint, strict: f32, protect: bool) {
        gate.set(
            true,
            print.bands,
            print.pitch,
            print.pitch_lo,
            print.pitch_hi,
            print.level,
            print.f1,
            print.f2,
            print.f3,
            strict,
            true,
            0.30,
            0.48,
            0.12,
            protect,
        );
    }

    fn feed(gate: &mut VoiceGate, samples: &[f32], score: Option<f32>) -> u32 {
        let mut audible = 0u32;
        for &x in samples {
            if let Some(s) = score {
                gate.set_neural(Some(s));
            }
            if gate.process(x, x).abs() > 1e-4 {
                audible += 1;
            }
        }
        audible
    }

    /// Fail-open: close-mic speech is audible in <40 ms without waiting for CAM++.
    #[test]
    fn fail_open_on_proximity_without_neural() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        gate.set_neural(None);
        let phrase = voice(sr, 160.0, 0.5);
        let mut first_open = None;
        for (i, &x) in phrase.iter().enumerate() {
            let y = gate.process(x, x);
            if y.abs() > 1e-4 && first_open.is_none() {
                first_open = Some(i);
            }
        }
        let first = first_open.expect("gate must open on proximity");
        assert!(
            first as f32 / sr < 0.04,
            "open took {:.1} ms",
            first as f32 * 1000.0 / sr
        );
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        assert!((gate.delay_ms() - 30.0).abs() < 1.5);
    }

    #[test]
    fn locked_output_is_bit_exact_with_thirty_ms_line() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let phrase = voice(sr, 160.0, 1.2);
        let delay = gate.delay_len;
        for &x in &phrase[..phrase.len() / 3] {
            gate.set_neural(Some(0.55));
            let _ = gate.process(x, x);
        }
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        for i in phrase.len() / 3..phrase.len() {
            gate.set_neural(Some(0.55));
            let y = gate.process(phrase[i], phrase[i]);
            if gate.authorized() {
                let src = phrase[i - delay];
                assert!((y - src).abs() < 1e-6, "not bit-exact at {i}");
            }
        }
    }

    #[test]
    fn short_oi_is_audible() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let oi = voice(sr, 160.0, 0.28);
        let opened = feed(&mut gate, &oi, None);
        assert!(
            opened > (sr * 0.12) as u32,
            "short Oi was eaten ({opened} samples)"
        );
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        assert!(
            gate.delay.len() <= gate.delay_len + 8,
            "hold-grow still active: {}",
            gate.delay.len()
        );
    }

    #[test]
    fn short_utt_without_score_is_not_reject() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.25), None);
        let silence = vec![0.0f32; (sr * 0.8) as usize];
        let _ = feed(&mut gate, &silence, None);
        assert_eq!(gate.state, GateState::Closed);
        assert!(!gate.utterance_vetoed);
    }

    #[test]
    fn slider_only_does_not_hard_reset_lock() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.3, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.4), Some(0.55));
        let before = gate.state;
        assert!(matches!(before, GateState::Locked | GateState::Holding));
        arm_neural(&mut gate, &print, 0.9, true);
        assert_eq!(gate.state, before, "slider must not reset utterance");
    }

    #[test]
    fn two_low_scores_after_500ms_veto() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 1.0, true);
        let opened = feed(&mut gate, &voice(sr, 160.0, 0.35), None);
        assert!(opened > 0, "must fail-open before CAM++");
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.90), Some(0.12));
        assert_eq!(gate.state, GateState::Rejected, "got {:?}", gate.state);
        let tail = feed(&mut gate, &voice(sr, 160.0, 0.4), Some(0.12));
        assert_eq!(tail, 0, "veto must stay closed on the same talker");
        assert!(!matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn pause_clears_veto_so_owner_can_return() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.9), Some(0.05));
        assert_eq!(gate.state, GateState::Rejected);
        let silence = vec![0.0f32; (sr * 0.30) as usize];
        let _ = feed(&mut gate, &silence, None);
        assert!(!gate.utterance_vetoed);
        let opened = feed(&mut gate, &voice(sr, 160.0, 0.4), Some(0.58));
        assert!(opened > (sr * 0.08) as u32, "owner stayed muted after pause");
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn energy_spike_reopens_over_impostor() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let quiet: Vec<f32> = voice(sr, 160.0, 0.9).into_iter().map(|x| x * 0.35).collect();
        let _ = feed(&mut gate, &quiet, Some(0.05));
        assert_eq!(gate.state, GateState::Rejected, "got {:?}", gate.state);
        let loud: Vec<f32> = voice(sr, 160.0, 0.5).into_iter().map(|x| x * 3.0).collect();
        let opened = feed(&mut gate, &loud, Some(0.58));
        assert!(opened > (sr * 0.05) as u32, "barge-in stayed muted ({opened})");
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn rejected_same_energy_does_not_reopen_on_high_score() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, false);
        let _ = feed(&mut gate, &voice(sr, 160.0, 1.0), Some(0.05));
        assert_eq!(gate.state, GateState::Rejected, "got {:?}", gate.state);
        let utt = gate.utterance_id;
        let leaked = feed(&mut gate, &voice(sr, 160.0, 0.5), Some(0.55));
        assert_eq!(leaked, 0, "same-energy impostor reopened");
        assert_ne!(gate.state, GateState::Locked);
        assert_eq!(gate.utterance_id, utt);
    }

    #[test]
    fn locked_survives_brief_score_dip_with_hysteresis() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.5), Some(0.55));
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        let dip_id = gate.last_infer_id;
        for &x in &voice(sr, 160.0, 0.25) {
            gate.set_neural_scored(Some(0.271), dip_id);
            let _ = gate.process(x, x);
        }
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding),
            "hysteresis failed: {:?}",
            gate.state
        );
        assert!(gate.display_score().is_some());
    }

    #[test]
    fn waiting_score_stays_open() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let opened = feed(&mut gate, &voice(sr, 160.0, 1.0), None);
        assert!(opened > (sr * 0.4) as u32, "waiting for CAM++ muted the owner");
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        assert_ne!(gate.reason, 3);
    }

    #[test]
    fn session_keeps_owner_open_across_pause() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.3, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.8), Some(0.62));
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        let silence = vec![0.0f32; (sr * 0.12) as usize];
        let _ = feed(&mut gate, &silence, None);
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding),
            "brief pause should keep session, got {:?}",
            gate.state
        );
        let opened = feed(&mut gate, &voice(sr, 160.0, 1.0), Some(0.58));
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        assert!(opened > (sr * 0.08) as u32, "next phrase was muted ({opened} samples)");
    }

    #[test]
    fn owner_high_score_stays_locked() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.5), Some(0.55));
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn locked_revokes_on_two_low_inferences() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.65), Some(0.55));
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        let base = gate.last_infer_id;
        for &x in &voice(sr, 160.0, 0.12) {
            gate.set_neural_scored(Some(0.22), base + 1);
            let _ = gate.process(x, x);
        }
        for &x in &voice(sr, 160.0, 0.12) {
            gate.set_neural_scored(Some(0.20), base + 2);
            let _ = gate.process(x, x);
        }
        assert!(
            !matches!(gate.state, GateState::Locked | GateState::Holding),
            "two low scores must revoke, got {:?}",
            gate.state
        );
    }

    #[test]
    fn long_pause_then_low_score_is_vetoed() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.6), Some(0.62));
        let silence = vec![0.0f32; (sr * 0.55) as usize];
        let _ = feed(&mut gate, &silence, None);
        assert!(
            !matches!(gate.state, GateState::Locked | GateState::Holding),
            "0.55 s pause must drop hold, got {:?}",
            gate.state
        );
        let phrase = voice(sr, 160.0, 0.9);
        let mut late = 0u32;
        let cut = (sr * 0.70) as usize;
        for (i, &x) in phrase.iter().enumerate() {
            gate.set_neural(Some(0.12));
            let y = gate.process(x, x);
            if i >= cut && y.abs() > 1e-4 {
                late += 1;
            }
        }
        assert_eq!(late, 0, "impostor still open after two CAM++ fails");
        assert!(!matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn laugh_unvoiced_burst_stays_open() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let _ = feed(&mut gate, &voice(sr, 160.0, 0.4), Some(0.55));
        let n = (sr * 0.5) as usize;
        let mut laugh = Vec::with_capacity(n);
        let mut state = 0xC0FFEEu32;
        for i in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let t = i as f32 / sr;
            laugh.push(noise * 0.55 * (0.6 + 0.4 * (std::f32::consts::TAU * 8.0 * t).sin()));
        }
        let opened = feed(&mut gate, &laugh, Some(0.08));
        assert!(opened > (sr * 0.20) as u32, "laugh was gated ({opened})");
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
    }
}
