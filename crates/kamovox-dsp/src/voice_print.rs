//! "Only my voice" — sidechain gate with utterance lock and bit-exact passthrough.
//!
//! Architecture:
//! - Analysis runs on a sidechain; wet PCM is delayed bit-exactly.
//! - **Protected** (default with CAM++): hold audio until neural confirm; delay
//!   ≈ 620 ms so the phrase start is still in the buffer when we open.
//! - **Fast**: may soft-open into Verifying (audible) before CAM++ — lower
//!   latency, higher risk of early third-party leak.
//! - States: Closed → Candidate → Verifying → Locked → Holding → Releasing / Rejected.

use std::collections::VecDeque;

use crate::filters::{coef, Biquad};

pub const PRINT_BANDS: usize = 12;
/// Fast mode delay (covers soft hops + small margin).
pub const LOOKAHEAD_FAST_MS: f32 = 280.0;
/// Protected: ≥400 ms speech + inference + margin so confirm-before-transmit works.
pub const LOOKAHEAD_PROTECTED_MS: f32 = 620.0;
/// Back-compat alias (fast).
#[allow(dead_code)]
pub const LOOKAHEAD_MS: f32 = LOOKAHEAD_FAST_MS;

const CENTERS: [f32; PRINT_BANDS] = [
    140.0, 200.0, 280.0, 390.0, 540.0, 750.0, 1_040.0, 1_450.0, 2_000.0, 2_700.0, 3_500.0, 4_500.0,
];

const SILENCE: f32 = 0.00010;
const SILENCE_EXIT_HOPS: u32 = 25;
const HOLD_EXIT_HOPS: u32 = 22; // ~440 ms hold for word endings
const FOREIGN_EXIT_HOPS: u32 = 18;
const OPEN_CONFIRM_HOPS: u32 = 8;
const CANDIDATE_HOPS: u32 = 2;
const REJECT_COOLDOWN_HOPS: u32 = 20;
/// Wait for CAM++ after speech ends (~1 s) before giving up on a short phrase.
const VERIFY_GRACE_HOPS: u32 = 50;
/// Need roughly this much speech (~500 ms) before CONFIRMED_OTHER is allowed.
const MIN_SPEECH_HOPS_REJECT: u32 = 25;
/// Sustained low scores required to enter Rejected from Verifying.
const VERIFY_FOREIGN_HOPS: u32 = 8;
/// Once Locked, require this many confirmed-other hops to leave.
const LOCKED_FOREIGN_HOPS: u32 = 12;

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
    /// Hops spent in Candidate/Verifying since last enter (for WAITING_SCORE age).
    verify_hops: u32,
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
        let delay_len = ((sr * LOOKAHEAD_PROTECTED_MS / 1000.0) as usize).max(1);
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
            neural_lo: 0.30,
            neural_hi: 0.48,
            neural_reject: 0.12,
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
        let cap = self.delay_len + ((self.sr * 1.25) as usize);
        cap as f32 * 1000.0 / self.sr.max(1.0)
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
        if infer_id > self.accept_after {
            self.neural = score.map(|s| s.clamp(-1.0, 1.0));
        } else {
            self.neural = None;
        }
    }

    fn invalidate_neural(&mut self) {
        self.accept_after = self.last_infer_id;
        self.neural = None;
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
        let new_lo = if neural_lo > 0.05 { neural_lo } else { 0.22 };
        let new_hi = if neural_hi > new_lo { neural_hi.min(0.55) } else { (new_lo + 0.12).min(0.48) };
        let new_reject = if neural_reject > 0.05 && neural_reject < new_lo {
            neural_reject
        } else {
            (new_lo - 0.12).clamp(0.08, 0.22)
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

        let want_ms = if self.neural_enrolled && self.protect {
            LOOKAHEAD_PROTECTED_MS
        } else {
            LOOKAHEAD_FAST_MS
        };
        let want = ((self.sr * want_ms / 1000.0) as usize).max(1);
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
        self.delay.push_back(wet);

        // Protected: grow only while Candidate/Verifying (or Closed with confirm
        // hops — onset). Do NOT grow on bare speech_env in Closed: that filled
        // the 1.25 s cap while vetoed/silent-reason and looked like a 1870 ms bug.
        let holding_utt = self.protect
            && !self.utterance_vetoed
            && (matches!(self.state, GateState::Candidate | GateState::Verifying)
                || (matches!(self.state, GateState::Closed) && self.confirm_hops > 0));
        if holding_utt {
            let cap = self.delay_len + ((self.sr * 1.25) as usize);
            while self.delay.len() > cap {
                self.delay.pop_front();
                self.popped = self.popped.wrapping_add(1);
            }
            return 0.0;
        }

        // Recover latency only when fully closed — not in Holding, or we would
        // discard the still-unplayed tail of the phrase.
        if matches!(self.state, GateState::Closed | GateState::Rejected) {
            while self.delay.len() > self.delay_len {
                self.delay.pop_front();
                self.popped = self.popped.wrapping_add(1);
            }
        }

        if self.delay.len() <= self.delay_len {
            return 0.0;
        }
        let out = self.delay.pop_front().unwrap_or(0.0);
        self.emit_sample(out)
    }

    fn observe(&mut self, x: f32) {
        if let Some(frame) = self.ears.push(x) {
            self.step_frame(&frame);
        }
        let audible = matches!(
            self.state,
            GateState::Locked | GateState::Holding | GateState::Verifying
        ) && !(self.state == GateState::Verifying && self.protect);
        if audible && self.target >= 1.0 - 1e-6 {
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
        let lock_need = 0.60 + self.strict * 0.12;
        let neural_need = self.neural_need();

        let mut frame_alike = self.confidence;
        if !quiet && frame.voiced {
            let rms = frame.energy.sqrt();
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

        let pitch_in_range = frame
            .pitch
            .map(|h| h >= self.pitch_lo * 0.96 && h <= self.pitch_hi * 1.04)
            .unwrap_or(false);
        let fmatch = formant_score(self.f1, self.f2, self.f3, frame.f1, frame.f2, frame.f3);
        let speech = !quiet && frame.energy > SILENCE * 3.0;
        let near_level = {
            let rms = frame.energy.sqrt();
            rms >= self.level * 0.22 && rms <= self.level * 4.0
        };

        // Fast soft-open: stricter than v0.1.16 (require pitch + formant agree).
        let soft_owner = speech
            && near_level
            && pitch_in_range
            && fmatch >= 0.45
            && frame_alike >= 0.48;

        let speech_start = speech && near_level; // Candidate trigger (not identity)

        let dsp_owner = frame.voiced
            && !quiet
            && pitch_in_range
            && fmatch >= 0.52
            && self.confidence >= lock_need
            && frame_alike >= lock_need - 0.05;
        let dsp_foreign =
            frame.voiced && !quiet && (!pitch_in_range || fmatch < 0.42 || frame_alike < lock_need - 0.18);

        let neural_ok = self.neural.map(|s| s >= neural_need);
        // Three zones: OWNER (>= need), UNCERTAIN (reject..need), OTHER (< reject).
        // Absence of accept is NOT proof of another speaker.
        let enough_for_reject = self.speech_hops >= MIN_SPEECH_HOPS_REJECT;
        let reject_line = self.neural_reject.max(0.08).min((neural_need - 0.05).max(0.08));
        let neural_bad = self
            .neural
            .map(|s| speech && enough_for_reject && s < reject_line);
        if let Some(score) = self.neural {
            if speech {
                self.confidence = self.confidence * 0.55 + score.max(0.0) * 0.45;
            }
        }

        let foreign_voiced = if self.neural_enrolled {
            neural_bad == Some(true)
        } else {
            dsp_foreign
        };
        // Leave Locked only on CONFIRMED_OTHER evidence, not mid-band dips.
        let leave_voiced = foreign_voiced;

        // Speech duration belongs to the active candidate/verify/lock cycle only.
        if speech
            && matches!(
                self.state,
                GateState::Candidate
                    | GateState::Verifying
                    | GateState::Locked
                    | GateState::Holding
                    | GateState::Confirming
            )
        {
            self.speech_hops = self.speech_hops.saturating_add(1);
        }

        let prev_state = self.state;
        match self.state {
            GateState::Closed => {
                self.foreign_hops = 0;
                self.verify_hops = 0;
                self.mute();
                if inactive {
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    if self.silence_hops >= SILENCE_EXIT_HOPS {
                        self.utterance_vetoed = false;
                        self.speech_hops = 0;
                        self.confirmed_score = None;
                        self.reason = 5; // SILENCE — only when truly inactive
                    }
                } else {
                    self.silence_hops = 0;
                    if self.utterance_vetoed {
                        self.reason = 12; // REJECT_COOLDOWN (still speaking, vetoed)
                    }
                }
                if self.neural_enrolled {
                    // Never open Closed→Locked on a score alone — that reused stale
                    // CAM++ results across short utterances. Always re-buffer.
                    if speech_start && !self.utterance_vetoed {
                        self.confirm_hops = self.confirm_hops.saturating_add(1);
                        if self.confirm_hops >= CANDIDATE_HOPS {
                            self.state = GateState::Candidate;
                            self.confirm_hops = 0;
                            self.speech_hops = 0;
                            self.verify_hops = 0;
                            self.confirmed_score = None;
                            self.utterance_id = self.utterance_id.wrapping_add(1);
                            self.invalidate_neural();
                            self.reason = 2; // PENDING / buffering
                            self.mute();
                        }
                    } else {
                        self.confirm_hops = 0;
                    }
                } else if dsp_owner {
                    self.state = GateState::Confirming;
                    self.confirm_hops = 1;
                } else {
                    self.confirm_hops = 0;
                }
            }
            GateState::Confirming => {
                self.mute();
                if dsp_owner {
                    self.confirm_hops = self.confirm_hops.saturating_add(1);
                    if self.confirm_hops >= OPEN_CONFIRM_HOPS {
                        self.state = GateState::Locked;
                        self.confirm_hops = 0;
                        self.open_exact();
                    }
                } else if !quiet && !frame.voiced {
                    // freeze
                } else {
                    self.state = GateState::Closed;
                    self.confirm_hops = 0;
                }
            }
            GateState::Candidate => {
                // Buffering — never audible. Keep alive after speech ends until Verifying.
                self.mute();
                self.verify_hops = self.verify_hops.saturating_add(1);
                if neural_ok == Some(true) {
                    self.state = GateState::Locked;
                    self.reason = 1;
                    self.foreign_hops = 0;
                    self.silence_hops = 0;
                    self.confirmed_score = self.neural;
                    self.verify_hops = 0;
                    self.open_exact();
                } else if foreign_voiced {
                    self.state = GateState::Rejected;
                    self.reason = 3; // OTHER_SPEAKER_REJECTED
                    self.utterance_vetoed = true;
                    self.confirm_hops = 0;
                    self.confirmed_score = None;
                } else if speech {
                    self.silence_hops = 0;
                    self.confirm_hops = self.confirm_hops.saturating_add(1);
                    if self.confirm_hops >= 2 {
                        self.state = GateState::Verifying;
                        self.confirm_hops = 0;
                        self.reason = 4; // PENDING_VERIFICATION
                        if !self.protect && soft_owner {
                            self.open_exact();
                        } else {
                            self.mute();
                        }
                    }
                } else if inactive {
                    // Phrase ended early ("Oi") — still wait for CAM++, don't drop.
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    self.state = GateState::Verifying;
                    self.reason = 4;
                    self.mute();
                }
            }
            GateState::Verifying => {
                self.verify_hops = self.verify_hops.saturating_add(1);
                if self.protect {
                    self.mute();
                } else if soft_owner || neural_ok == Some(true) {
                    self.open_exact();
                } else {
                    self.mute();
                }
                // Confirm even after the phrase ended — delay line still holds PCM.
                if neural_ok == Some(true) {
                    self.state = GateState::Locked;
                    self.reason = 1;
                    self.foreign_hops = 0;
                    self.silence_hops = 0;
                    self.utterance_vetoed = false;
                    self.confirmed_score = self.neural;
                    self.verify_hops = 0;
                    self.open_exact();
                } else if foreign_voiced {
                    self.foreign_hops = self.foreign_hops.saturating_add(1);
                    if self.foreign_hops >= VERIFY_FOREIGN_HOPS {
                        self.state = GateState::Rejected;
                        self.reason = 3; // CONFIRMED_OTHER only
                        self.utterance_vetoed = true;
                        self.foreign_hops = 0;
                        self.confirmed_score = None;
                        self.mute();
                    }
                } else if inactive {
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    self.foreign_hops = 0;
                    if self.silence_hops >= VERIFY_GRACE_HOPS {
                        // Not "other speaker" — not enough evidence for a short utt.
                        self.state = GateState::Closed;
                        self.reason = 9; // INSUFFICIENT_EVIDENCE
                        self.silence_hops = 0;
                        self.speech_hops = 0;
                        self.verify_hops = 0;
                        self.invalidate_neural();
                        self.mute();
                        // Do NOT set utterance_vetoed — owner may retry immediately.
                    }
                } else {
                    // Still speaking: mid-band stays VERIFYING (uncertain ≠ other).
                    self.foreign_hops = 0;
                    self.silence_hops = 0;
                    if self.verify_hops >= VERIFY_GRACE_HOPS {
                        match self.neural {
                            Some(s) if s >= neural_need - 0.03 => {
                                self.state = GateState::Locked;
                                self.reason = 1;
                                self.confirmed_score = Some(s);
                                self.verify_hops = 0;
                                self.open_exact();
                            }
                            Some(s) if s < reject_line => {
                                // Wait for VERIFY_FOREIGN_HOPS via foreign_voiced path.
                                self.reason = 10;
                            }
                            Some(_) => {
                                self.reason = 10; // LOW_CONFIDENCE / UNCERTAIN
                            }
                            None => {
                                self.reason = 11; // WAITING_SCORE
                            }
                        }
                    } else if self.neural.is_none() {
                        self.reason = 4; // still PENDING_VERIFICATION
                    } else if self.neural.map(|s| s < neural_need && s >= reject_line) == Some(true) {
                        self.reason = 10; // UNCERTAIN — keep collecting
                    }
                }
            }
            GateState::Locked => {
                self.confirm_hops = 0;
                self.verify_hops = 0;
                self.open_exact();
                if inactive {
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    self.foreign_hops = 0;
                    if self.silence_hops >= SILENCE_EXIT_HOPS {
                        self.state = GateState::Holding;
                        self.reason = 7; // UTTERANCE_FINISHED → hold
                        self.silence_hops = 0;
                    }
                } else if leave_voiced {
                    self.silence_hops = 0;
                    self.foreign_hops = self.foreign_hops.saturating_add(1);
                    let need = if self.neural_enrolled {
                        LOCKED_FOREIGN_HOPS
                    } else {
                        FOREIGN_EXIT_HOPS
                    };
                    if self.foreign_hops >= need {
                        self.state = GateState::Rejected;
                        self.reason = 8; // SPEAKER_CHANGE
                        self.foreign_hops = 0;
                        self.utterance_vetoed = true;
                        self.confirmed_score = None;
                        self.mute();
                    }
                } else {
                    self.silence_hops = 0;
                    self.foreign_hops = 0;
                }
            }
            GateState::Holding => {
                self.open_exact();
                // Stay authorized through short gaps using the latched confirm —
                // do NOT re-open via a fresh neural_ok (that reused scores across
                // Rejected→Locked on the same utterance id).
                if self.neural_enrolled && speech_start {
                    // New phrase after short pause — always new utterance_id.
                    self.state = GateState::Candidate;
                    self.silence_hops = 0;
                    self.speech_hops = 0;
                    self.verify_hops = 0;
                    self.confirmed_score = None;
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                    self.invalidate_neural();
                    self.reason = 2;
                    self.mute();
                } else if !self.neural_enrolled && dsp_owner {
                    self.state = GateState::Locked;
                    self.silence_hops = 0;
                } else if inactive {
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    if self.silence_hops >= HOLD_EXIT_HOPS {
                        self.state = GateState::Releasing;
                        self.reason = 7;
                        self.mute();
                    }
                } else if leave_voiced {
                    self.foreign_hops = self.foreign_hops.saturating_add(1);
                    if self.foreign_hops >= 5 {
                        self.state = GateState::Rejected;
                        self.utterance_vetoed = true;
                        self.confirmed_score = None;
                        self.reason = 8;
                        self.mute();
                    }
                } else {
                    // Unvoiced continuation of same utterance.
                    self.state = GateState::Locked;
                    self.silence_hops = 0;
                    self.foreign_hops = 0;
                    self.reason = 1;
                }
            }
            GateState::Rejected => {
                self.mute();
                self.utterance_vetoed = true;
                self.confirmed_score = None;
                // NEVER Rejected→Locked on the same utterance. Cooldown → Closed,
                // then a new Candidate bumps utterance_id.
                if inactive {
                    self.silence_hops = self.silence_hops.saturating_add(1);
                    if self.silence_hops >= REJECT_COOLDOWN_HOPS {
                        self.state = GateState::Closed;
                        self.silence_hops = 0;
                        self.reason = 12; // REJECT_COOLDOWN done → Closed
                    }
                } else {
                    self.silence_hops = 0;
                    self.reason = 3;
                }
            }
            GateState::Releasing => {
                self.mute();
                // Neural enrolled: finish release to Closed; next speech starts
                // a new utterance. Do not jump back to Locked mid-release.
                if !self.neural_enrolled && dsp_owner {
                    self.state = GateState::Confirming;
                    self.confirm_hops = 1;
                } else if inactive {
                    if self.gain < 0.02 {
                        self.state = GateState::Closed;
                        self.confidence *= 0.85;
                        self.confirmed_score = None;
                        self.reason = 5;
                    }
                } else if self.neural_enrolled {
                    // Still energy during release — finish closing with cooldown label.
                    self.reason = 7;
                    if self.gain < 0.02 {
                        self.state = GateState::Closed;
                        self.confirmed_score = None;
                        self.reason = 5;
                    }
                }
            }
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
    fn same_f0_different_timbre_is_blocked() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 130.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.85);
        let ratio = kept_ratio(&mut gate, &other_male(sr, 130.0, 1.5));
        assert!(ratio < 0.35, "other male at same f0 kept {ratio}");
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
            samples.extend(quiet_voice(sr, 150.0, 0.04, 0.35));
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
    fn a_much_lower_voice_is_closed() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 220.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm(&mut gate, &print, 0.85);
        assert!(kept_ratio(&mut gate, &voice(sr, 110.0, 1.2)) < 0.35);
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

    /// Protected: silence until CAM++ confirms — no provisional leak.
    #[test]
    fn protected_stays_silent_until_neural() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        gate.set_neural(None);
        let phrase = voice(sr, 160.0, 0.5);
        let mut leaked = 0;
        for &x in &phrase {
            let y = gate.process(x, x);
            if y.abs() > 1e-4 {
                leaked += 1;
            }
        }
        assert_eq!(leaked, 0, "protected leaked {leaked} samples before neural");
        assert!(matches!(
            gate.state,
            GateState::Candidate | GateState::Verifying | GateState::Closed
        ));
    }

    /// After neural confirm, delayed PCM is bit-exact.
    #[test]
    fn protected_opens_bit_exact_after_confirm() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let phrase = voice(sr, 160.0, 1.4);
        let delay = gate.delay_len;
        for &x in &phrase[..phrase.len() / 4] {
            gate.set_neural(None);
            let _ = gate.process(x, x);
        }
        for &x in &phrase[phrase.len() / 4..phrase.len() / 2] {
            gate.set_neural(Some(0.55));
            let _ = gate.process(x, x);
        }
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding),
            "got {:?}",
            gate.state
        );
        let start = phrase.len() / 2;
        for i in start..phrase.len() {
            gate.set_neural(Some(0.55));
            let y = gate.process(phrase[i], phrase[i]);
            if matches!(gate.state, GateState::Locked | GateState::Holding) && gate.gain >= 1.0 - 1e-6 {
                let src = phrase[i - delay];
                assert!((y - src).abs() < 1e-6, "not bit-exact at {i}");
            }
        }
    }

    #[test]
    fn neural_veto_rejects_and_stays_closed() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.8, true);
        for &x in &voice(sr, 160.0, 0.2) {
            gate.set_neural(None);
            let _ = gate.process(x, x);
        }
        for &x in &voice(sr, 160.0, 0.6) {
            gate.set_neural(Some(0.05));
            let y = gate.process(x, x);
            assert!(y.abs() < 1e-4, "leak during reject");
        }
        assert!(
            matches!(gate.state, GateState::Rejected | GateState::Closed | GateState::Verifying | GateState::Candidate),
            "got {:?}",
            gate.state
        );
        // Soft speech must not reopen without good score.
        for &x in &voice(sr, 160.0, 0.3) {
            gate.set_neural(Some(0.05));
            let y = gate.process(x, x);
            assert!(y.abs() < 1e-4);
        }
        assert!(!matches!(gate.state, GateState::Locked | GateState::Holding));
    }

    #[test]
    fn fast_mode_may_soft_open_before_score() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.4, false);
        gate.set_neural(None);
        let phrase = voice(sr, 160.0, 0.7);
        let mut first_open = None;
        for (i, &x) in phrase.iter().enumerate() {
            let y = gate.process(x, x);
            if y.abs() > 1e-4 && first_open.is_none() {
                first_open = Some(i);
            }
        }
        // Fast can open; Protected must not — this is Fast.
        assert!(first_open.is_some(), "fast mode should soft-open on own voice");
    }

    /// Short "Oi" (~300 ms): hold-grow until late CAM++ confirm, then emit start.
    #[test]
    fn protected_short_utt_preserved_after_late_confirm() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let oi = voice(sr, 160.0, 0.30);
        let silence = vec![0.0f32; (sr * 0.45) as usize];
        let mut leaked = 0u32;
        for &x in oi.iter().chain(silence.iter()) {
            gate.set_neural(None);
            let y = gate.process(x, x);
            if y.abs() > 1e-4 {
                leaked += 1;
            }
        }
        assert_eq!(leaked, 0, "short utt leaked before confirm");
        assert!(
            matches!(gate.state, GateState::Verifying | GateState::Candidate),
            "expected verifying short utt, got {:?}",
            gate.state
        );
        let held = gate.delay.len();
        // From empty delay, hold-grow keeps the whole oi+tail (not a primed 620 ms line).
        assert!(
            held >= oi.len() && held > (sr * 0.55) as usize,
            "delay should retain short phrase: held={held} oi={}",
            oi.len()
        );
        // Late confirm while still silent — must open and flush held PCM.
        let mut opened = 0u32;
        let flush = vec![0.0f32; held + gate.delay_len];
        for &x in &flush {
            gate.set_neural(Some(0.55));
            let y = gate.process(x, x);
            if y.abs() > 1e-4 {
                opened += 1;
            }
        }
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding | GateState::Releasing | GateState::Closed),
            "got {:?}",
            gate.state
        );
        assert!(
            opened > (sr * 0.12) as u32,
            "expected flushed short phrase energy, opened={opened}"
        );
        // After flush + silence the gate may already be Closed (reason SILENCE/FINISHED);
        // the important part is that authorized audio was emitted.
        assert!(opened > 0);
    }

    #[test]
    fn short_utt_insufficient_is_not_reject() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        let oi = voice(sr, 160.0, 0.25);
        for &x in &oi {
            gate.set_neural(None);
            let _ = gate.process(x, x);
        }
        // ~1.2 s silence → VERIFY_GRACE → INSUFFICIENT_EVIDENCE (reason 9), not Rejected.
        let silence = vec![0.0f32; (sr * 1.3) as usize];
        for &x in &silence {
            gate.set_neural(None);
            let y = gate.process(x, x);
            assert!(y.abs() < 1e-4);
        }
        assert_eq!(gate.state, GateState::Closed);
        assert_eq!(gate.reason, 9, "INSUFFICIENT_EVIDENCE");
        assert!(!gate.utterance_vetoed);
    }

    #[test]
    fn stale_infer_cannot_open_new_utterance() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        // Fake a previous high score.
        gate.set_neural_scored(Some(0.90), 1);
        let oi = voice(sr, 160.0, 0.35);
        let mut leaked = 0u32;
        for &x in &oi {
            // Re-apply the same stale infer id after Candidate invalidates.
            gate.set_neural_scored(Some(0.90), 1);
            let y = gate.process(x, x);
            if y.abs() > 1e-4 {
                leaked += 1;
            }
        }
        assert_eq!(leaked, 0, "stale infer opened gate");
        assert!(
            !matches!(gate.state, GateState::Locked | GateState::Holding),
            "stale score locked: {:?}",
            gate.state
        );
        // Fresh inference must be allowed to confirm.
        for &x in &voice(sr, 160.0, 0.2) {
            gate.set_neural_scored(Some(0.55), 2);
            let _ = gate.process(x, x);
        }
        assert!(
            matches!(gate.state, GateState::Locked | GateState::Holding),
            "fresh infer should lock, got {:?}",
            gate.state
        );
    }

    #[test]
    fn slider_only_does_not_hard_reset_verifying() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.3, true);
        for &x in &voice(sr, 160.0, 0.4) {
            gate.set_neural(None);
            let _ = gate.process(x, x);
        }
        let before = gate.state;
        assert!(matches!(before, GateState::Candidate | GateState::Verifying));
        arm_neural(&mut gate, &print, 0.9, true); // exigência only
        assert_eq!(gate.state, before, "slider must not reset utterance");
    }

    /// Video utt 18: Rejected must not jump to Locked on a later high score.
    #[test]
    fn rejected_cannot_reopen_same_utterance() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, false); // Fast 0%
        // Need sustained CONFIRMED_OTHER (score << reject) for ≥500 ms + foreign hops.
        for &x in &voice(sr, 160.0, 1.2) {
            gate.set_neural(Some(0.05));
            let _ = gate.process(x, x);
        }
        assert_eq!(gate.state, GateState::Rejected, "got {:?}", gate.state);
        let utt = gate.utterance_id;
        let hops_at_reject = gate.speech_hops;
        for &x in &voice(sr, 160.0, 0.6) {
            gate.set_neural(Some(0.55));
            let y = gate.process(x, x);
            assert!(y.abs() < 1e-4, "rejected utterance must stay muted");
            assert_ne!(gate.state, GateState::Locked);
            assert_eq!(gate.utterance_id, utt);
        }
        assert!(
            matches!(gate.state, GateState::Rejected | GateState::Closed),
            "got {:?}",
            gate.state
        );
        assert!(
            gate.speech_hops <= hops_at_reject + 2,
            "speech_hops must freeze while Rejected: {} → {}",
            hops_at_reject,
            gate.speech_hops
        );
    }

    #[test]
    fn locked_survives_brief_score_dip_with_hysteresis() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, true);
        for &x in &voice(sr, 160.0, 0.5) {
            gate.set_neural(Some(0.55));
            let _ = gate.process(x, x);
        }
        assert!(matches!(gate.state, GateState::Locked | GateState::Holding));
        // Dip just under open thresh (0.28) but above leave thresh — must stay Locked.
        for &x in &voice(sr, 160.0, 0.25) {
            gate.set_neural(Some(0.271));
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
    fn waiting_score_is_not_other_speaker() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.5, true);
        for &x in &voice(sr, 160.0, 1.2) {
            gate.set_neural(None);
            let y = gate.process(x, x);
            assert!(y.abs() < 1e-4);
        }
        assert!(
            matches!(gate.state, GateState::Verifying | GateState::Candidate),
            "got {:?}",
            gate.state
        );
        assert!(
            matches!(gate.reason, 4 | 11),
            "expected PENDING/WAITING_SCORE, got {}",
            gate.reason
        );
        assert_ne!(gate.reason, 3);
    }

    /// Mid-band cosine (below accept, above reject) must NOT become OTHER_SPEAKER.
    #[test]
    fn uncertain_mid_band_does_not_reject_owner() {
        let sr = 48_000.0;
        let print = build_voice_print(&voice(sr, 160.0, 2.5), sr).expect("print");
        let mut gate = VoiceGate::new(sr);
        arm_neural(&mut gate, &print, 0.0, true); // need≈0.30, reject≈0.12
        for &x in &voice(sr, 160.0, 1.5) {
            gate.set_neural(Some(0.22)); // classic false-"other" band under old 0.28 floor
            let _ = gate.process(x, x);
        }
        assert!(
            !matches!(gate.state, GateState::Rejected),
            "0.22 must be UNCERTAIN not OTHER, got {:?} r{}",
            gate.state,
            gate.reason
        );
        assert!(
            matches!(gate.state, GateState::Verifying | GateState::Candidate | GateState::Locked),
            "got {:?}",
            gate.state
        );
    }
}
