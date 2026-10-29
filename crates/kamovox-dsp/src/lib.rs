//! KamoVox voice processing. Everything runs on mono `f32` blocks in place and
//! never blocks, so the same engine can live in the app or in an audio APO.

mod filters;
mod voice_print;
mod formant;
mod fx;
mod pitch;
mod tuner;
mod vocoder;
mod voices;

use std::collections::VecDeque;

use nnnoiseless::DenoiseState;
use serde::{Deserialize, Serialize};

use filters::{coef, Biquad, Envelope};
use voice_print::VoiceGate;
use fx::{Chorus, Echo, Reverb};
use tuner::{Character, Tuner};
pub use vocoder::Carrier;
use vocoder::Vocoder;
pub use voice_print::{
    build_voice_print, GateEvent, GatePhase, VoicePrint, LOOKAHEAD_FAST_MS, LOOKAHEAD_MS,
    LOOKAHEAD_PROTECTED_MS, PRINT_BANDS,
};
pub use voices::Preset;
use voices::VoiceStage;

pub const MAX_CHAIN: usize = 4;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Params {
    /// When false the voice passes through untouched.
    pub enabled: bool,
    /// Voice stages applied in order (up to `MAX_CHAIN`).
    pub voices: Vec<Preset>,
    /// Global pitch in semitones, applied on top of any voice.
    pub pitch: f32,
    /// Timbre (formants) in semitones; 0 = time-domain pitch only (clean).
    /// Non-zero engages the STFT formant stage.
    pub formant: f32,
    /// 0..1 voice-character amounts.
    pub breath: f32,
    pub rasp: f32,
    pub vibrato: f32,
    pub nasal: f32,
    pub vocoder: Option<Carrier>,
    pub vocoder_mix: f32,
    pub cave: bool,
    pub hall: bool,
    pub chorus: bool,
    pub ghost: bool,
    pub echo: bool,
    /// Smaller room than the hall.
    pub room: bool,
    /// Short slapback delay.
    pub slap: bool,
    /// Modulated jet, without the ghost reverb.
    pub flange: bool,
    /// Hollow, muffled resonance, like talking into a bottle.
    pub bottle: bool,
    /// Input gain in percent (100 = unchanged).
    pub boost: f32,
    /// Noise-gate threshold in dBFS; -90 or lower disables the gate.
    pub sensitivity_db: f32,
    pub bass_db: f32,
    pub treble_db: f32,
    /// Ten graphic-EQ gains in dB, from 20 Hz to 16 kHz. Missing data stays flat.
    #[serde(default = "flat_eq")]
    pub eq: [f32; 10],
    pub denoise: bool,
    /// 0..1 wet mix of the neural noise remover. 1 replaces the mic with the cleaned voice.
    #[serde(default = "full_mix")]
    pub denoise_amount: f32,
    /// 0..1 reduction of steady noise (fans, hiss). Idle while neural denoise is on.
    #[serde(default)]
    pub noise_floor: f32,
    /// 0..1 reduction of short clicks (keyboard). Idle while neural denoise is on.
    #[serde(default)]
    pub impact: f32,
    /// 0..1 voice compressor. 0 is off.
    #[serde(default)]
    pub compress: f32,
    /// Close the mic when the sound is not the enrolled owner.
    #[serde(default)]
    pub voice_gate: bool,
    /// 0 loose, 1 strict.
    #[serde(default = "voice_strict_default")]
    pub voice_strict: f32,
    #[serde(default = "empty_print")]
    pub voice_bands: [f32; voice_print::PRINT_BANDS],
    #[serde(default)]
    pub voice_pitch: f32,
    /// Low/high of the enrolled pitch range. 0 = derive from `voice_pitch`.
    #[serde(default)]
    pub voice_pitch_lo: f32,
    #[serde(default)]
    pub voice_pitch_hi: f32,
    /// Median near-field RMS from enrollment. 0 = default.
    #[serde(default)]
    pub voice_level: f32,
    /// Enrolled formant peaks (Hz). 0 = unset.
    #[serde(default)]
    pub voice_f1: f32,
    #[serde(default)]
    pub voice_f2: f32,
    #[serde(default)]
    pub voice_f3: f32,
    /// CAM++ enrollment embedding (512). Empty = DSP-only gate.
    #[serde(default)]
    pub voice_embed: Vec<f32>,
    /// Extra CAM++ templates (N×512, N≤16). Live score = max cosine vs centroid+gallery.
    #[serde(default)]
    pub voice_gallery: Vec<f32>,
    /// Calibrated cosine at strict=0 / strict=1 (from enrollment self-scores).
    #[serde(default)]
    pub voice_neural_lo: f32,
    #[serde(default)]
    pub voice_neural_hi: f32,
    /// Below this (sustained) → CONFIRMED_OTHER. Mid-band is uncertain, not reject.
    #[serde(default)]
    pub voice_neural_reject: f32,
    /// Confirm-before-transmit (true) vs Fast soft-open (false). Default protected.
    #[serde(default = "voice_protect_default")]
    pub voice_protect: bool,
    pub append_original: bool,
    /// Longer WSOLA grain (~44 ms vs ~38 ms). Classic voices only.
    pub smooth: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            enabled: true,
            voices: Vec::new(),
            pitch: 0.0,
            formant: 0.0,
            breath: 0.0,
            rasp: 0.0,
            vibrato: 0.0,
            nasal: 0.0,
            vocoder: None,
            vocoder_mix: 1.0,
            cave: false,
            hall: false,
            chorus: false,
            ghost: false,
            echo: false,
            room: false,
            slap: false,
            flange: false,
            bottle: false,
            boost: 100.0,
            sensitivity_db: -90.0,
            bass_db: 0.0,
            treble_db: 0.0,
            eq: flat_eq(),
            denoise: false,
            denoise_amount: 0.75,
            noise_floor: 0.0,
            impact: 0.0,
            compress: 0.0,
            voice_gate: false,
            voice_strict: 0.35, // keep in sync with voice_strict_default
            voice_bands: empty_print(),
            voice_pitch: 0.0,
            voice_pitch_lo: 0.0,
            voice_pitch_hi: 0.0,
            voice_level: 0.0,
            voice_f1: 0.0,
            voice_f2: 0.0,
            voice_f3: 0.0,
            voice_embed: Vec::new(),
            voice_gallery: Vec::new(),
            voice_neural_lo: 0.0,
            voice_neural_hi: 0.0,
            voice_neural_reject: 0.0,
            voice_protect: true,
            append_original: false,
            smooth: false,
        }
    }
}

fn flat_eq() -> [f32; 10] {
    [0.0; 10]
}

fn full_mix() -> f32 {
    0.75
}

fn voice_strict_default() -> f32 {
    0.35
}

fn voice_protect_default() -> bool {
    true
}

fn empty_print() -> [f32; voice_print::PRINT_BANDS] {
    [0.0; voice_print::PRINT_BANDS]
}

/// Centers of the graphic EQ. The first is a low shelf and the last a high shelf.
const EQ_HZ: [f32; 10] = [20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 16_000.0];

fn eq_band(sr: f32, index: usize, db: f32) -> Biquad {
    let db = db.clamp(-12.0, 12.0);
    if db.abs() < 0.05 {
        return Biquad::identity();
    }
    match index {
        0 => Biquad::low_shelf(sr, 40.0, db),
        9 => Biquad::high_shelf(sr, 10_000.0, db),
        _ => Biquad::peaking(sr, EQ_HZ[index], 1.1, db),
    }
}

const DENOISE_FRAME: usize = DenoiseState::FRAME_SIZE;

/// RNNoise plus a voice-activity gate driven by its own speech probability:
/// RNNoise alone leaves broadband noise around -12 dB, the gate removes it
/// between words.
struct Denoiser {
    state: Box<DenoiseState<'static>>,
    input: Vec<f32>,
    output: VecDeque<f32>,
    frame: [f32; DENOISE_FRAME],
    dry: VecDeque<f32>,
    gain: f32,
    target: f32,
    attack: f32,
    release: f32,
    hold: u32,
    amount: f32,
}

/// Residual gain when RNNoise VAD says “not speech”. Must stay audible enough
/// that unvoiced consonants (s, f, t) are not chopped — RNNoise already cleaned
/// the spectrum; this only ducks leftover hiss between phrases.
const DENOISE_FLOOR: f32 = 0.18;
const DENOISE_HOLD_FRAMES: u32 = 36;

impl Denoiser {
    fn new() -> Self {
        let mut output = VecDeque::with_capacity(DENOISE_FRAME * 3);
        output.extend(std::iter::repeat(0.0).take(DENOISE_FRAME));
        let mut dry = VecDeque::with_capacity(DENOISE_FRAME * 3);
        dry.extend(std::iter::repeat(0.0).take(DENOISE_FRAME));
        Self {
            state: DenoiseState::new(),
            input: Vec::with_capacity(DENOISE_FRAME),
            output,
            frame: [0.0; DENOISE_FRAME],
            dry,
            gain: 1.0,
            target: 1.0,
            attack: coef(48_000.0, 5.0),
            release: coef(48_000.0, 80.0),
            hold: 0,
            amount: 1.0,
        }
    }

    fn set_amount(&mut self, amount: f32) {
        self.amount = amount.clamp(0.0, 1.0);
    }

    fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            self.dry.push_back(*s);
            self.input.push(*s * 32_767.0);
            if self.input.len() == DENOISE_FRAME {
                let vad = self.state.process_frame(&mut self.frame, &self.input);
                // Do not hard-gate on VAD: loud hiss scores ~0.5, the same band as
                // unvoiced consonants. RNNoise already cleaned the spectrum; only
                // duck true quiet so the noise floor between phrases drops a bit.
                if vad > 0.28 {
                    self.hold = DENOISE_HOLD_FRAMES;
                } else if self.hold > 0 {
                    self.hold -= 1;
                }
                self.target = if self.hold > 0 {
                    1.0
                } else {
                    DENOISE_FLOOR + 0.35 * vad
                };
                for v in &self.frame {
                    self.output.push_back(v / 32_767.0);
                }
                self.input.clear();
            }
            let wet = self.output.pop_front().unwrap_or(0.0);
            let dry = self.dry.pop_front().unwrap_or(0.0);
            let k = if self.target > self.gain { self.attack } else { self.release };
            self.gain += (self.target - self.gain) * k;
            let wet = wet * self.gain;
            *s = dry * (1.0 - self.amount) + wet * self.amount;
        }
    }
}

/// Turns down sound that stays near the recent noise floor: fans and steady hiss.
struct NoiseFloor {
    env: Envelope,
    floor: f32,
    gain: f32,
    fall: f32,
    rise: f32,
    smooth: f32,
}

impl NoiseFloor {
    fn new(sr: f32) -> Self {
        Self {
            env: Envelope::new(sr, 8.0, 80.0),
            floor: 0.01,
            gain: 1.0,
            fall: coef(sr, 40.0),
            rise: coef(sr, 2500.0),
            smooth: coef(sr, 15.0),
        }
    }

    fn process(&mut self, x: f32, amount: f32) -> f32 {
        if amount < 0.01 {
            return x;
        }
        let e = self.env.process(x);
        if e < self.floor {
            self.floor += (e - self.floor) * self.fall;
        } else {
            self.floor += (e - self.floor) * self.rise;
        }
        let open = ((e / (self.floor * 1.6 + 1e-5) - 1.0) / 3.0).clamp(0.0, 1.0);
        let target = 1.0 - amount * (1.0 - open);
        self.gain += (target - self.gain) * self.smooth;
        x * self.gain
    }
}

/// Ducks very short high-frequency spikes, such as a key click, and lets vowels through.
struct Impact {
    band: Biquad,
    fast: Envelope,
    slow: Envelope,
    gain: f32,
    duck: f32,
    release: f32,
}

impl Impact {
    fn new(sr: f32) -> Self {
        Self {
            band: Biquad::highpass(sr, 2_500.0, 0.7),
            fast: Envelope::new(sr, 0.4, 12.0),
            slow: Envelope::new(sr, 25.0, 80.0),
            gain: 1.0,
            duck: coef(sr, 1.0),
            release: coef(sr, 35.0),
        }
    }

    fn process(&mut self, x: f32, amount: f32) -> f32 {
        if amount < 0.01 {
            return x;
        }
        let hf = self.band.process(x);
        let fast = self.fast.process(hf);
        let slow = self.slow.process(hf);
        let click = fast > 0.015 && fast > slow * (2.2 + amount);
        let target = if click { 1.0 - amount * 0.9 } else { 1.0 };
        let k = if target < self.gain { self.duck } else { self.release };
        self.gain += (target - self.gain) * k;
        x * self.gain
    }
}

/// Keeps speech in a narrower loudness range: quieter syllables come up, peaks come down.
struct Compressor {
    env: Envelope,
    gain: f32,
    attack: f32,
    release: f32,
}

impl Compressor {
    fn new(sr: f32) -> Self {
        Self { env: Envelope::new(sr, 3.0, 100.0), gain: 1.0, attack: coef(sr, 4.0), release: coef(sr, 90.0) }
    }

    fn process(&mut self, x: f32, amount: f32) -> f32 {
        if amount < 0.01 {
            return x;
        }
        let e = self.env.process(x).max(1e-6);
        let thresh = 0.12;
        let ratio = 1.0 + amount * 6.0;
        let compressed = if e > thresh {
            thresh * (e / thresh).powf(1.0 / ratio) / e
        } else {
            1.0
        };
        let norm = (e / thresh).clamp(0.0, 1.0);
        let presence = (e / 0.02).clamp(0.0, 1.0);
        let desired = compressed + amount * 0.45 * (1.0 - norm) * presence;
        let k = if desired < self.gain { self.attack } else { self.release };
        self.gain += (desired - self.gain) * k;
        (x * self.gain).tanh()
    }
}

struct Gate {
    env: Envelope,
    gain: f32,
    open: f32,
    close: f32,
}

impl Gate {
    fn new(sr: f32) -> Self {
        Self { env: Envelope::new(sr, 1.0, 120.0), gain: 1.0, open: coef(sr, 4.0), close: coef(sr, 90.0) }
    }

    #[inline]
    fn process(&mut self, x: f32, threshold: f32) -> f32 {
        let target = if self.env.process(x) > threshold { 1.0 } else { 0.0 };
        let k = if target > self.gain { self.open } else { self.close };
        self.gain += (target - self.gain) * k;
        x * self.gain
    }
}

struct Sfx {
    cave_echo: Echo,
    cave_verb: Reverb,
    hall: Reverb,
    chorus: Chorus,
    ghost_flanger: Chorus,
    ghost_verb: Reverb,
    echo: Echo,
    room: Reverb,
    slap: Echo,
    flange: Chorus,
}

impl Sfx {
    fn new(sr: f32) -> Self {
        Self {
            cave_echo: Echo::new(sr, 280.0, 0.42, 0.5),
            cave_verb: Reverb::new(sr, 0.85, 0.5, 0.25),
            hall: Reverb::new(sr, 0.92, 0.3, 0.45),
            chorus: Chorus::chorus(sr),
            ghost_flanger: Chorus::flanger(sr),
            ghost_verb: Reverb::new(sr, 0.96, 0.2, 0.6),
            echo: Echo::new(sr, 420.0, 0.38, 0.45),
            room: Reverb::new(sr, 0.45, 0.55, 0.28),
            slap: Echo::new(sr, 95.0, 0.12, 0.4),
            flange: Chorus::flanger(sr),
        }
    }
}

/// Broadcast polish for “Voz mais suave”: HPF, de-esser, body, soft saturation.
struct StudioPolish {
    on: bool,
    hp: Biquad,
    body: Biquad,
    nasal: Biquad,
    presence: Biquad,
    air: Biquad,
    ess: Biquad,
    ess_env: Envelope,
    full_env: Envelope,
    duck: f32,
    duck_atk: f32,
    duck_rel: f32,
}

impl StudioPolish {
    fn new(sr: f32) -> Self {
        let mut p = Self {
            on: false,
            hp: Biquad::identity(),
            body: Biquad::identity(),
            nasal: Biquad::identity(),
            presence: Biquad::identity(),
            air: Biquad::identity(),
            ess: Biquad::identity(),
            ess_env: Envelope::new(sr, 1.5, 40.0),
            full_env: Envelope::new(sr, 3.0, 50.0),
            duck: 0.0,
            duck_atk: coef(sr, 3.0),
            duck_rel: coef(sr, 45.0),
        };
        p.tune(sr);
        p
    }

    fn tune(&mut self, sr: f32) {
        self.hp = Biquad::highpass(sr, 75.0, 0.707);
        self.body = Biquad::peaking(sr, 160.0, 0.8, 1.5);
        self.nasal = Biquad::peaking(sr, 450.0, 1.2, -1.5);
        self.presence = Biquad::peaking(sr, 3_200.0, 0.9, 1.2);
        self.air = Biquad::lowpass(sr, 14_000.0, 0.707);
        self.ess = Biquad::bandpass(sr, 6_800.0, 2.2);
    }

    fn set(&mut self, on: bool) {
        if on && !self.on {
            self.duck = 0.0;
        }
        self.on = on;
    }

    fn process(&mut self, x: f32) -> f32 {
        if !self.on {
            return x;
        }
        let mut y = self.hp.process(x);
        y = self.body.process(y);
        y = self.nasal.process(y);
        y = self.presence.process(y);
        y = self.air.process(y);
        let band = self.ess.process(y);
        let be = self.ess_env.process(band);
        let fe = self.full_env.process(y);
        let ratio = be / (fe * 0.42 + 1e-4);
        let target = ((ratio - 1.15) * 0.45).clamp(0.0, 0.37);
        let k = if target > self.duck { self.duck_atk } else { self.duck_rel };
        self.duck += (target - self.duck) * k;
        y -= band * self.duck;
        let drive = 0.9;
        (y * drive).tanh() / drive
    }
}

pub struct Processor {
    sr: f32,
    params: Params,
    voices: Vec<VoiceStage>,
    tuner: Tuner,
    vocoder: Option<Vocoder>,
    sfx: Sfx,
    bottle_bp: Biquad,
    bottle_peak: Biquad,
    bottle_lp: Biquad,
    bass: Biquad,
    treble: Biquad,
    eq: [Biquad; 10],
    gate: Gate,
    noise_floor: NoiseFloor,
    impact: Impact,
    compressor: Compressor,
    owner: VoiceGate,
    denoiser: Option<Denoiser>,
    polish: StudioPolish,
    limiter: f32,
    limiter_attack: f32,
    limiter_release: f32,
}

impl Processor {
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        let mut p = Self {
            sr,
            params: Params::default(),
            voices: Vec::new(),
            tuner: Tuner::new(sr),
            vocoder: None,
            sfx: Sfx::new(sr),
            bottle_bp: Biquad::bandpass(sr, 520.0, 1.6),
            bottle_peak: Biquad::peaking(sr, 780.0, 5.0, 9.0),
            bottle_lp: Biquad::lowpass(sr, 1600.0, 0.7),
            bass: Biquad::identity(),
            treble: Biquad::identity(),
            eq: [Biquad::identity(); 10],
            gate: Gate::new(sr),
            noise_floor: NoiseFloor::new(sr),
            impact: Impact::new(sr),
            compressor: Compressor::new(sr),
            owner: VoiceGate::new(sr),
            denoiser: None,
            polish: StudioPolish::new(sr),
            limiter: 1.0,
            limiter_attack: coef(sr, 1.2),
            limiter_release: coef(sr, 150.0),
        };
        p.set_params(Params::default());
        p
    }

    pub fn sample_rate(&self) -> u32 {
        self.sr as u32
    }

    /// RNNoise only runs at 48 kHz.
    pub fn denoise_supported(&self) -> bool {
        self.sample_rate() == 48_000
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn set_params(&mut self, mut p: Params) {
        p.voices.truncate(MAX_CHAIN);
        let old = std::mem::replace(&mut self.params, p);
        let p = &self.params;

        let kinds: Vec<Preset> = self.voices.iter().map(|v| v.kind()).collect();
        if kinds != p.voices {
            self.voices = p.voices.iter().map(|&k| VoiceStage::new(self.sr, k)).collect();
        }

        // Smooth only lengthens the pitch grain. The hidden studio EQ used to
        // ride this switch, so “EQ plano” was not actually flat.
        let window_ms = if p.smooth { 44.0 } else { 38.0 };
        self.polish.set(false);
        self.tuner.set_window_ms(window_ms);
        for v in &mut self.voices {
            v.set_window_ms(window_ms);
        }

        self.tuner.set(Character {
            pitch: p.pitch.clamp(-24.0, 24.0),
            formant: p.formant.clamp(-12.0, 12.0),
            breath: p.breath.clamp(0.0, 1.0),
            rasp: p.rasp.clamp(0.0, 1.0),
            vibrato: p.vibrato.clamp(0.0, 1.0),
            nasal: p.nasal.clamp(0.0, 1.0),
        });

        match p.vocoder {
            Some(c) if self.vocoder.as_ref().map(|v| v.carrier()) != Some(c) => {
                self.vocoder = Some(Vocoder::new(self.sr, c));
            }
            None => self.vocoder = None,
            _ => {}
        }

        if p.cave && !old.cave {
            self.sfx.cave_echo.clear();
            self.sfx.cave_verb.clear();
        }
        if p.hall && !old.hall {
            self.sfx.hall.clear();
        }
        if p.chorus && !old.chorus {
            self.sfx.chorus.clear();
        }
        if p.ghost && !old.ghost {
            self.sfx.ghost_flanger.clear();
            self.sfx.ghost_verb.clear();
        }
        if p.echo && !old.echo {
            self.sfx.echo.clear();
        }
        if p.room && !old.room {
            self.sfx.room.clear();
        }
        if p.slap && !old.slap {
            self.sfx.slap.clear();
        }
        if p.flange && !old.flange {
            self.sfx.flange.clear();
        }

        self.bass.retune(Biquad::low_shelf(self.sr, 180.0, p.bass_db.clamp(-30.0, 15.0)));
        self.treble.retune(Biquad::high_shelf(self.sr, 4_000.0, p.treble_db.clamp(-30.0, 15.0)));
        let sr = self.sr;
        for (i, band) in self.eq.iter_mut().enumerate() {
            band.retune(eq_band(sr, i, p.eq[i]));
        }

        self.owner.set(
            p.voice_gate,
            p.voice_bands,
            p.voice_pitch,
            p.voice_pitch_lo,
            p.voice_pitch_hi,
            p.voice_level,
            p.voice_f1,
            p.voice_f2,
            p.voice_f3,
            p.voice_strict,
            p.voice_embed.len() == 512,
            p.voice_neural_lo,
            p.voice_neural_hi,
            p.voice_neural_reject,
            p.voice_protect,
        );

        if p.denoise && self.denoise_supported() {
            if self.denoiser.is_none() {
                self.denoiser = Some(Denoiser::new());
            }
            if let Some(d) = &mut self.denoiser {
                d.set_amount(p.denoise_amount);
            }
        } else {
            self.denoiser = None;
        }
    }

    /// Processes one mono block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        if !self.params.enabled {
            if self.params.voice_gate {
                self.apply_owner_gate(buf);
            }
            return;
        }
        self.process_input(buf);
        self.process_voice(buf);
    }

    fn apply_owner_gate(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let raw = *s;
            *s = self.owner.process(raw, raw);
        }
    }

    /// Latest CAM++ cosine for the owner gate (`None` = no fresh score).
    /// `infer_id` must be the monotonic id from the neural worker.
    pub fn set_neural_score(&mut self, score: Option<f32>, infer_id: u64) {
        self.owner.set_neural_scored(score, infer_id);
    }

    pub fn gate_phase(&self) -> voice_print::GatePhase {
        self.owner.phase()
    }

    pub fn gate_delay_ms(&self) -> f32 {
        self.owner.delay_ms()
    }

    pub fn gate_utterance_id(&self) -> u64 {
        self.owner.utterance_id()
    }

    pub fn gate_reason(&self) -> u8 {
        self.owner.reason_code()
    }

    pub fn gate_events(&self) -> Vec<voice_print::GateEvent> {
        self.owner.diagnostic_events()
    }

    pub fn gate_neural(&self) -> Option<f32> {
        self.owner.neural_score()
    }

    pub fn gate_thresh(&self) -> f32 {
        self.owner.neural_thresh()
    }

    pub fn gate_protect(&self) -> bool {
        self.owner.is_protect()
    }

    pub fn gate_infer_id(&self) -> u64 {
        self.owner.last_infer_id()
    }

    pub fn gate_buffered_ms(&self) -> f32 {
        self.owner.buffered_ms()
    }

    pub fn gate_speech_hops(&self) -> u32 {
        self.owner.speech_hops()
    }

    pub fn gate_speech_env(&self) -> f32 {
        self.owner.speech_env()
    }

    pub fn gate_strict(&self) -> f32 {
        self.params.voice_strict
    }

    pub fn gate_display_score(&self) -> Option<f32> {
        self.owner.display_score()
    }

    pub fn gate_authorized(&self) -> bool {
        self.owner.authorized()
    }

    pub fn gate_verify_age_ms(&self) -> f32 {
        self.owner.verify_age_ms()
    }

    pub fn gate_buffer_capacity_ms(&self) -> f32 {
        self.owner.buffer_capacity_ms()
    }

    /// Input index of the sample the gate emitted on the last call (offline tests).
    pub fn gate_last_source(&self) -> Option<u64> {
        self.owner.last_source_index()
    }

    pub fn gate_buffered_samples(&self) -> usize {
        self.owner.buffered_samples()
    }

    /// Microphone clean-up only: denoise, gate, EQ and boost.
    pub fn process_input(&mut self, buf: &mut [f32]) {
        if !self.params.enabled {
            return;
        }
        if let Some(d) = &mut self.denoiser {
            d.process(buf);
        }
        let gain = (self.params.boost / 100.0).clamp(0.0, 6.0);
        let gate_on = self.params.sensitivity_db > -89.0;
        let threshold = 10f32.powf(self.params.sensitivity_db / 20.0);
        let ai = self.params.denoise && self.denoiser.is_some();
        let floor = if ai { 0.0 } else { self.params.noise_floor.clamp(0.0, 1.0) };
        let impact = if ai { 0.0 } else { self.params.impact.clamp(0.0, 1.0) };
        let compress = self.params.compress.clamp(0.0, 1.0);
        for s in buf.iter_mut() {
            let raw = *s;
            let mut x = raw;
            x = self.noise_floor.process(x, floor);
            x = self.impact.process(x, impact);
            if gate_on {
                x = self.gate.process(x, threshold);
            }
            x = self.compressor.process(x, compress);
            let mut y = self.treble.process(self.bass.process(x));
            for band in &mut self.eq {
                y = band.process(y);
            }
            // Lookahead gate: decide on raw mic, apply to the delayed wet sample.
            *s = self.owner.process(raw, y * gain);
        }
    }

    /// Voice transformation, effects and the output limiter.
    pub fn process_voice(&mut self, buf: &mut [f32]) {
        if !self.params.enabled {
            return;
        }
        let mix = self.params.vocoder_mix.clamp(0.0, 1.0);
        let chorus = self.params.chorus;
        let flange = self.params.flange;
        let bottle = self.params.bottle;
        let slap = self.params.slap;
        let echo = self.params.echo;
        let cave = self.params.cave;
        let room = self.params.room;
        let hall = self.params.hall;
        let ghost = self.params.ghost;
        let append_original = self.params.append_original;
        let tune = self.tuner.is_active();

        for s in buf.iter_mut() {
            let x = *s;
            let dry = x;

            let mut y = if tune { self.tuner.process(x) } else { x };
            for v in &mut self.voices {
                y = v.process(y);
            }
            if let Some(voc) = &mut self.vocoder {
                y = voc.process(y) * mix + y * (1.0 - mix);
            }
            y = self.polish.process(y);
            if chorus {
                y = self.sfx.chorus.process(y);
            }
            if flange {
                y = self.sfx.flange.process(y);
            }
            if bottle {
                let wet = self.bottle_lp.process(self.bottle_peak.process(self.bottle_bp.process(y)));
                y = wet * 0.82 + y * 0.18;
            }
            if slap {
                y = self.sfx.slap.process(y);
            }
            if echo {
                y = self.sfx.echo.process(y);
            }
            if cave {
                y = self.sfx.cave_verb.process(self.sfx.cave_echo.process(y));
            }
            if room {
                y = self.sfx.room.process(y);
            }
            if hall {
                y = self.sfx.hall.process(y);
            }
            if ghost {
                y = self.sfx.ghost_verb.process(self.sfx.ghost_flanger.process(y));
            }
            if append_original {
                y = 0.5 * (y + dry);
            }

            let desired = if y.abs() > 0.95 { 0.95 / y.abs() } else { 1.0 };
            let k = if desired < self.limiter {
                self.limiter_attack
            } else {
                self.limiter_release
            };
            self.limiter += (desired - self.limiter) * k;
            self.limiter = self.limiter.clamp(0.05, 1.0);
            *s = (y * self.limiter).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / sr).sin() * 0.5).collect()
    }

    #[test]
    fn every_preset_stays_finite_and_bounded() {
        let all = [
            Preset::Male, Preset::Female, Preset::Helium, Preset::Baby, Preset::Giant,
            Preset::Demon, Preset::Monster, Preset::Alien, Preset::Mutation, Preset::SlowMutation,
            Preset::FastMutation, Preset::Clone, Preset::Robot, Preset::Cyborg, Preset::Atari,
            Preset::Radio, Preset::Telephone, Preset::Walkie, Preset::Megaphone, Preset::Underwater,
            Preset::Whisper, Preset::Silence,
        ];
        let carriers = [None, Some(Carrier::Saw), Some(Carrier::Bell), Some(Carrier::Noise)];
        for (i, &preset) in all.iter().enumerate() {
            let mut proc = Processor::new(48_000);
            proc.set_params(Params {
                voices: vec![preset],
                pitch: if i % 2 == 0 { 2.3 } else { -4.1 },
                formant: if i % 3 == 0 { -1.5 } else { 0.0 },
                breath: (i % 4) as f32 * 0.3,
                rasp: (i % 5) as f32 * 0.25,
                vibrato: (i % 3) as f32 * 0.5,
                nasal: (i % 2) as f32 * 0.7,
                vocoder: carriers[i % carriers.len()],
                cave: i % 2 == 0,
                hall: i % 3 == 0,
                chorus: i % 4 == 0,
                ghost: i % 5 == 0,
                echo: i % 6 == 0,
                room: i % 7 == 0,
                slap: i % 8 == 0,
                flange: i % 9 == 0,
                bottle: i % 10 == 0,
                denoise: i % 2 == 1,
                sensitivity_db: -50.0,
                bass_db: 6.0,
                treble_db: -6.0,
                boost: 250.0,
                ..Params::default()
            });
            let mut buf = sine(48_000.0, 48_000);
            for chunk in buf.chunks_mut(480) {
                proc.process(chunk);
            }
            assert!(buf.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{preset:?}");
        }
    }

    #[test]
    fn formant_shifter_at_unity_reconstructs_input() {
        let input = sine(48_000.0, 8_192);
        let mut f = formant::FormantShifter::new(48_000.0);
        f.set_ratio(1.0);
        let out: Vec<f32> = input.iter().map(|&x| f.process(x)).collect();
        let delay = 1_024; // n - hop @ 48 kHz with n=1024
        let err = (4_000..8_000).map(|i| (out[i] - input[i - delay]).abs()).fold(0f32, f32::max);
        assert!(err < 1e-3, "max error {err}");
    }

    #[test]
    fn denoise_removes_steady_noise() {
        let mut noise = filters::Noise::default();
        let input: Vec<f32> = (0..48_000 * 2).map(|_| noise.next() * 0.05).collect();
        let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();

        let mut proc = Processor::new(48_000);
        proc.set_params(Params { denoise: true, denoise_amount: 1.0, ..Params::default() });
        let mut buf = input.clone();
        for chunk in buf.chunks_mut(441) {
            proc.process(chunk);
        }
        let tail = 48_000;
        let ratio = rms(&buf[tail..]) / rms(&input[tail..]);
        // RNNoise residual, not a VAD mute (that mute ate consonants).
        assert!(ratio < 0.35, "noise only reduced to {ratio}");
    }

    #[test]
    fn denoise_keeps_voiced_speech() {
        let sr = 48_000.0;
        let input: Vec<f32> = (0..48_000 * 2)
            .map(|i| {
                let t = i as f32 / sr;
                let mut x = 0.0f32;
                for h in 1..=6 {
                    x += (std::f32::consts::TAU * 150.0 * h as f32 * t).sin() / h as f32;
                }
                x * 0.18
            })
            .collect();
        let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        let mut proc = Processor::new(48_000);
        proc.set_params(Params { denoise: true, denoise_amount: 0.75, ..Params::default() });
        let mut buf = input.clone();
        for chunk in buf.chunks_mut(480) {
            proc.process(chunk);
        }
        let gain = rms(&buf[24_000..]) / rms(&input[24_000..]);
        assert!(gain > 0.40, "denoise ate the voice ({gain})");
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smooth_polish_changes_a_bright_voice() {
        let sr = 48_000.0;
        let n = 48_000;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let body = (std::f32::consts::TAU * 140.0 * t).sin() * 0.2;
                let ess = (std::f32::consts::TAU * 7_000.0 * t).sin() * 0.08;
                body + ess
            })
            .collect();
        let run = |smooth: bool| {
            let mut proc = Processor::new(48_000);
            proc.set_params(Params { pitch: -1.7, smooth, ..Params::default() });
            let mut buf = input.clone();
            for chunk in buf.chunks_mut(480) {
                proc.process(chunk);
            }
            buf
        };
        let plain = run(false);
        let soft = run(true);
        assert!(plain.iter().all(|v| v.is_finite()) && soft.iter().all(|v| v.is_finite()));
        let skip = 16_000;
        let err = plain[skip..]
            .iter()
            .zip(&soft[skip..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err > 0.01, "Voz mais suave did not change the signal ({err})");
    }

    #[test]
    fn textures_stay_silent_on_silence_and_keep_level() {
        let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        let textures = [
            Params { breath: 1.0, ..Params::default() },
            Params { rasp: 1.0, ..Params::default() },
            Params { nasal: 1.0, ..Params::default() },
            Params { vibrato: 1.0, ..Params::default() },
        ];
        for params in textures {
            let mut proc = Processor::new(48_000);
            proc.set_params(params.clone());
            let mut silence = vec![0.0; 48_000];
            for chunk in silence.chunks_mut(480) {
                proc.process(chunk);
            }
            assert!(rms(&silence) < 1e-5, "{params:?} makes noise on silence");

            let input = sine(48_000.0, 48_000);
            let mut buf = input.clone();
            for chunk in buf.chunks_mut(480) {
                proc.process(chunk);
            }
            let gain = rms(&buf[8_000..]) / rms(&input[8_000..]);
            assert!((0.35..2.2).contains(&gain), "{params:?} level changed by {gain}");
        }
    }

    #[test]
    fn breath_is_clearly_audible() {
        let sr = 48_000.0;
        // Harmonic-rich "voice": 150 Hz sawtooth.
        let input: Vec<f32> = (0..48_000).map(|i| ((i as f32 * 150.0 / sr).fract() * 2.0 - 1.0) * 0.3).collect();
        let run = |breath: f32| {
            let mut f = formant::FormantShifter::new(sr);
            f.set_breath(breath);
            input.iter().map(|&x| f.process(x)).collect::<Vec<f32>>()
        };
        let plain = run(0.0);
        let breathy = run(1.0);
        let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        let diff: Vec<f32> = plain.iter().zip(&breathy).skip(4_000).map(|(a, b)| a - b).collect();
        let ratio = rms(&diff) / rms(&plain[4_000..]);
        assert!(ratio > 0.25, "breath changes the signal by only {ratio}");
        assert!(rms(&breathy[4_000..]) < rms(&plain[4_000..]) * 1.6, "breath too loud");
    }

    #[test]
    fn moderate_breath_adds_audible_air() {
        let sr = 48_000.0;
        let input: Vec<f32> = (0..48_000).map(|i| ((i as f32 * 150.0 / sr).fract() * 2.0 - 1.0) * 0.3).collect();
        let render = |breath: f32| {
            let mut proc = Processor::new(48_000);
            proc.set_params(Params { breath, ..Params::default() });
            let mut buf = input.clone();
            for chunk in buf.chunks_mut(480) {
                proc.process(chunk);
            }
            buf
        };
        let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        let band = |buf: &[f32], hp_hz: f32, lp_hz: f32| {
            let mut hp = filters::Biquad::highpass(sr, hp_hz, 0.7);
            let mut lp = filters::Biquad::lowpass(sr, lp_hz, 0.7);
            let samples: Vec<f32> = buf[8_000..].iter().map(|&x| lp.process(hp.process(x))).collect();
            rms(&samples)
        };
        let plain = render(0.0);
        let breathy = render(0.3);
        let changed = plain[8_000..]
            .iter()
            .zip(&breathy[8_000..])
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let change = rms(&changed) / rms(&plain[8_000..]);
        assert!(change > 0.04, "30 % breath barely moves the voice ({change})");
        let hiss_db = 20.0 * (band(&breathy, 7_000.0, 16_000.0) / band(&plain, 7_000.0, 16_000.0).max(1e-8)).log10();
        assert!(hiss_db < 3.0, "30 % breath adds {hiss_db:.1} dB of hiss above 7 kHz");
        let quiet = render(0.06);
        let quiet_change = plain[8_000..]
            .iter()
            .zip(&quiet[8_000..])
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let quiet_ratio = rms(&quiet_change) / rms(&plain[8_000..]);
        assert!(quiet_ratio < 0.08, "6 % breath already shifts the voice by {quiet_ratio}");
    }

    #[test]
    fn disabled_is_passthrough() {
        let mut proc = Processor::new(44_100);
        proc.set_params(Params { enabled: false, voices: vec![Preset::Robot], ..Params::default() });
        let input = sine(44_100.0, 1024);
        let mut buf = input.clone();
        proc.process(&mut buf);
        assert_eq!(buf, input);
    }

    fn voice_bandlimited(sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let mut x = 0.0f32;
                for h in 1..=8 {
                    let f = 140.0 * h as f32;
                    if f > 3_500.0 {
                        break;
                    }
                    x += (std::f32::consts::TAU * f * t).sin() / h as f32;
                }
                x * 0.12
            })
            .collect()
    }

    fn render(params: Params, input: &[f32]) -> Vec<f32> {
        let mut proc = Processor::new(48_000);
        proc.set_params(params);
        let mut buf = input.to_vec();
        for chunk in buf.chunks_mut(480) {
            proc.process(chunk);
        }
        buf
    }

    fn band_rms(buf: &[f32], hp_hz: f32, lp_hz: f32) -> f32 {
        let mut hp = filters::Biquad::highpass(48_000.0, hp_hz, 0.707);
        let mut lp = filters::Biquad::lowpass(48_000.0, lp_hz, 0.707);
        let mut e = 0.0f32;
        let mut n = 0usize;
        for &x in buf.iter().skip(16_000) {
            let y = lp.process(hp.process(x));
            e += y * y;
            n += 1;
        }
        (e / n.max(1) as f32).sqrt()
    }

    fn spike_count(buf: &[f32]) -> usize {
        let mut n = 0usize;
        let mut prev = 0.0f32;
        let mut ema = 0.0f32;
        for &s in buf.iter().skip(20_000) {
            let d = (s - prev).abs();
            if ema > 1e-5 && d > ema * 14.0 && d > 0.03 {
                n += 1;
            }
            ema = ema * 0.999 + d * 0.001;
            prev = s;
        }
        n
    }

    #[test]
    fn audit_pitch_does_not_invent_clicks_or_highs() {
        let sr = 48_000.0;
        let input = voice_bandlimited(sr, sr as usize * 2);
        let dry_hi = band_rms(&input, 4_000.0, 8_000.0);
        let dry_mid = band_rms(&input, 400.0, 1_500.0);
        let pitched = render(Params { pitch: -1.7, ..Params::default() }, &input);
        let hi = band_rms(&pitched, 4_000.0, 8_000.0);
        let mid = band_rms(&pitched, 400.0, 1_500.0);
        let dry_rel = 20.0 * (dry_hi / dry_mid.max(1e-8)).log10();
        let rel = 20.0 * (hi / mid.max(1e-8)).log10();
        let spikes = spike_count(&pitched);
        let zeros = pitched.iter().skip(20_000).filter(|s| s.abs() < 1e-8).count();
        eprintln!(
            "AUDIT pitch-only  rel_4k {rel:.1} dB (dry {dry_rel:.1}) spikes {spikes} near-zeros {zeros}"
        );
        assert!(spikes < 8, "pitch-only spikes {spikes}");
        assert!(zeros < 8, "pitch-only dropouts {zeros}");
        assert!(rel < dry_rel + 6.0, "pitch added highs: {rel:.1} vs dry {dry_rel:.1}");

        let user = render(
            Params {
                pitch: -1.7,
                compress: 1.0,
                smooth: true,
                denoise: true,
                denoise_amount: 0.8,
                ..Params::default()
            },
            &input,
        );
        let user_rel = 20.0 * (band_rms(&user, 4_000.0, 8_000.0) / band_rms(&user, 400.0, 1_500.0).max(1e-8)).log10();
        let user_spikes = spike_count(&user);
        eprintln!("AUDIT user-chain rel_4k {user_rel:.1} dB spikes {user_spikes}");

        let mut proc = Processor::new(48_000);
        proc.set_params(Params { pitch: -1.7, ..Params::default() });
        let mut live = input[..24_000].to_vec();
        proc.process(&mut live);
        proc.set_params(Params { pitch: -1.7, smooth: true, ..Params::default() });
        let mut tail = input[24_000..28_000].to_vec();
        let before = live[live.len() - 1];
        proc.process(&mut tail);
        let jump = (tail[0] - before).abs();
        eprintln!("AUDIT smooth-toggle first-sample jump {jump:.4}");
        assert!(jump < 0.08, "smooth toggle clicks ({jump})");
    }
}
