use std::f32::consts::TAU;

use crate::filters::{Biquad, Envelope, Noise};
use crate::formant::FormantShifter;
use crate::pitch::PitchShifter;

/// Fine voice-character controls shared by every preset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Character {
    /// Semitones.
    pub pitch: f32,
    /// Formant offset in semitones; 0 keeps the speaker's own timbre even when
    /// the pitch moves.
    pub formant: f32,
    /// The following are 0..1.
    pub breath: f32,
    pub rasp: f32,
    pub vibrato: f32,
    pub nasal: f32,
}

impl Character {
    fn is_neutral(&self) -> bool {
        *self == Character::default()
    }
}

/// Natural vibrato sits at 5–7 Hz with up to about one semitone of extent.
const VIBRATO_HZ: f32 = 5.6;
const VIBRATO_SEMITONES: f32 = 0.8;
/// Growl "macro-period" amplitude modulation range.
const GROWL_HZ: (f32, f32) = (30.0, 50.0);

/// Slowly wandering value in -1..1 (low-passed noise), for natural irregularity.
struct Drift {
    value: f32,
    target: f32,
    coef: f32,
    countdown: u32,
    every: u32,
}

impl Drift {
    fn new(sr: f32, changes_per_sec: f32) -> Self {
        let every = (sr / changes_per_sec).max(1.0) as u32;
        Self { value: 0.0, target: 0.0, coef: 1.0 - (-1.0 / every as f32).exp(), countdown: 0, every }
    }

    #[inline]
    fn next(&mut self, noise: &mut Noise) -> f32 {
        if self.countdown == 0 {
            self.target = noise.next();
            self.countdown = self.every;
        }
        self.countdown -= 1;
        self.value += (self.target - self.value) * self.coef;
        self.value
    }
}

pub struct Tuner {
    sr: f32,
    c: Character,
    shifter: PitchShifter,
    formant: FormantShifter,
    sub: PitchShifter,
    sub_band: [Biquad; 2],
    rasp_tone: Biquad,
    nasal: [Biquad; 4],
    /// Upper part of the voice that breath softens, and the noise that replaces it.
    edge: Biquad,
    air_band: [Biquad; 2],
    air_env: Envelope,
    noise: Noise,
    growl_phase: f32,
    growl_drift: Drift,
    shimmer: Drift,
    air_phase: f32,
    vib_phase: f32,
    vib_drift: Drift,
    tick: u32,
}

impl Tuner {
    pub fn new(sr: f32) -> Self {
        Self {
            sr,
            c: Character::default(),
            shifter: PitchShifter::new(sr),
            formant: FormantShifter::new(sr),
            sub: PitchShifter::new(sr),
            sub_band: [Biquad::highpass(sr, 70.0, 0.7), Biquad::lowpass(sr, 1_400.0, 0.7)],
            rasp_tone: Biquad::lowpass(sr, 6_500.0, 0.7),
            nasal: [Biquad::identity(); 4],
            edge: Biquad::highpass(sr, 1_400.0, 0.7),
            air_band: [Biquad::highpass(sr, 1_000.0, 0.7), Biquad::lowpass(sr, 3_200.0, 0.7)],
            air_env: Envelope::new(sr, 8.0, 90.0),
            noise: Noise::default(),
            growl_phase: 0.0,
            growl_drift: Drift::new(sr, 6.0),
            shimmer: Drift::new(sr, 90.0),
            air_phase: 0.0,
            vib_phase: 0.0,
            vib_drift: Drift::new(sr, 1.5),
            tick: 0,
        }
    }

    pub fn set_window_ms(&mut self, ms: f32) {
        self.shifter.set_window_ms(self.sr, ms);
        self.sub.set_window_ms(self.sr, ms);
    }

    pub fn set(&mut self, c: Character) {
        let sr = self.sr;
        self.c = c;
        self.shifter.set_semitones(c.pitch);
        // Formant STFT only when the user moves the formant control.
        // Auto-compensating every pitch change via STFT (envelope warp + OLA)
        // was the metallic / "PVC pipe" path — Clownfish stays clean on ±2 st
        // because it stays in the time domain for small shifts.
        if c.formant.abs() > 0.05 {
            self.formant.set_ratio(2f32.powf((c.formant - c.pitch) / 12.0));
        } else {
            self.formant.set_ratio(1.0);
        }
        // Breath is time-domain aspiration below. The spectral "whisper" path
        // randomizes phase and comes out as television hiss.
        self.formant.set_breath(0.0);
        self.sub.set_semitones(-12.0);

        // Nasal resonance (~250 Hz), F1 anti-resonance (~600 Hz), the "honk"
        // (~1.3 kHz) and the second nasal resonance (~2.5 kHz).
        let n = c.nasal;
        let targets = [
            Biquad::peaking(sr, 260.0, 1.2, 4.0 * n),
            Biquad::peaking(sr, 600.0, 1.4, -8.0 * n),
            Biquad::peaking(sr, 1_300.0, 1.6, 9.0 * n),
            Biquad::peaking(sr, 2_500.0, 2.0, 4.0 * n),
        ];
        for (f, t) in self.nasal.iter_mut().zip(targets) {
            f.retune(t);
        }
    }

    pub fn is_active(&self) -> bool {
        !self.c.is_neutral()
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let c = self.c;
        let mut y = x;

        if c.pitch != 0.0 || c.vibrato > 0.0 {
            if c.vibrato > 0.0 {
                let rate = VIBRATO_HZ + 0.6 * self.vib_drift.next(&mut self.noise);
                self.vib_phase = (self.vib_phase + rate / self.sr) % 1.0;
                self.tick = self.tick.wrapping_add(1);
                if self.tick % 32 == 0 {
                    let wobble = (TAU * self.vib_phase).sin() * c.vibrato * VIBRATO_SEMITONES;
                    self.shifter.set_semitones(c.pitch + wobble);
                }
            }
            y = self.shifter.process(y);
            if c.vibrato > 0.0 {
                // A little loudness tremor rides along with the pitch, as in real voices.
                y *= 1.0 - 0.1 * c.vibrato * (0.5 + 0.5 * (TAU * self.vib_phase).sin());
            }
        }
        if !self.formant.is_neutral() {
            y = self.formant.process(y);
        }
        if c.nasal > 0.0 {
            for f in &mut self.nasal {
                y = f.process(y);
            }
        }
        if c.rasp > 0.0 {
            y = self.growl(y, c.rasp);
        }
        if c.breath > 0.0 {
            y = self.aspirate(y, c.breath);
        }
        y
    }

    /// Breath: the upper harmonics lose some edge and that slot is filled with
    /// noise in the same band, following the voice. The body of the voice stays.
    #[inline]
    fn aspirate(&mut self, x: f32, amount: f32) -> f32 {
        let edge = self.edge.process(x);
        let env = self.air_env.process(edge);
        let t = ((env - 0.003) / 0.02).clamp(0.0, 1.0);
        let gate = t * t * (3.0 - 2.0 * t);
        let n = self.air_band[0].process(self.noise.next());
        let n = self.air_band[1].process(n);
        self.air_phase = (self.air_phase + 4.5 / self.sr) % 1.0;
        let flutter = 0.7 + 0.3 * (TAU * self.air_phase).sin();
        let air = n * env * 3.2 * gate * flutter;
        x - edge * (0.8 * amount) + air * amount
    }

    /// Growl: an octave-down subharmonic plus an irregular 30–50 Hz amplitude
    /// modulation and shimmer, then gentle saturation and a low-pass so it
    /// stays rough rather than harsh.
    #[inline]
    fn growl(&mut self, x: f32, amount: f32) -> f32 {
        let sub = self.sub.process(x);
        let sub = self.sub_band[0].process(sub);
        let sub = self.sub_band[1].process(sub);
        let mut y = x + sub * 0.45 * amount;

        let drift = self.growl_drift.next(&mut self.noise);
        let hz = GROWL_HZ.0 + (GROWL_HZ.1 - GROWL_HZ.0) * (0.5 + 0.5 * drift);
        self.growl_phase = (self.growl_phase + hz / self.sr) % 1.0;
        let am = 1.0 - 0.6 * amount * (0.5 + 0.5 * (TAU * self.growl_phase).sin());
        let shimmer = 1.0 + 0.25 * amount * self.shimmer.next(&mut self.noise);
        y *= am * shimmer;

        let drive = 1.0 + 1.8 * amount;
        y = (y * drive).tanh() / drive;
        let toned = self.rasp_tone.process(y);
        toned * amount + y * (1.0 - amount)
    }
}
