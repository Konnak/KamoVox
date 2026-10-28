use std::f32::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::filters::{Biquad, DelayLine, Envelope, Noise};
use crate::formant::FormantShifter;
use crate::pitch::PitchShifter;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Preset {
    Male,
    Female,
    Helium,
    Baby,
    Giant,
    Demon,
    Monster,
    Alien,
    Mutation,
    SlowMutation,
    FastMutation,
    Clone,
    Robot,
    Cyborg,
    Atari,
    Radio,
    Telephone,
    Walkie,
    Megaphone,
    Underwater,
    Whisper,
    Silence,
}

/// One voice-transforming stage. Several stages can be chained (effect mixer).
pub struct VoiceStage {
    kind: Preset,
    sr: f32,
    p1: PitchShifter,
    p2: PitchShifter,
    formant: Option<FormantShifter>,
    f: [Biquad; 3],
    comb: DelayLine,
    env: Envelope,
    noise: Noise,
    lfo: f32,
    osc: f32,
    hold: f32,
    hold_count: u32,
    tick: u32,
}

impl VoiceStage {
    pub fn new(sr: f32, kind: Preset) -> Self {
        let mut s = Self {
            kind,
            sr,
            p1: PitchShifter::new(sr),
            p2: PitchShifter::new(sr),
            formant: None,
            f: [Biquad::identity(); 3],
            comb: DelayLine::new((sr * 0.02) as usize),
            env: Envelope::new(sr, 2.0, 60.0),
            noise: Noise::default(),
            lfo: 0.0,
            osc: 0.0,
            hold: 0.0,
            hold_count: 0,
            tick: 0,
        };
        s.configure();
        s
    }

    pub fn kind(&self) -> Preset {
        self.kind
    }

    pub fn set_window_ms(&mut self, ms: f32) {
        self.p1.set_window_ms(self.sr, ms);
        self.p2.set_window_ms(self.sr, ms);
    }

    /// Pitch shift whose formants land at `formant` semitones instead of
    /// following the pitch, which is what makes it sound like another person.
    fn natural(&mut self, pitch: f32, formant: f32) {
        self.p1.set_semitones(pitch);
        let mut f = FormantShifter::new(self.sr);
        f.set_ratio(2f32.powf((formant - pitch) / 12.0));
        self.formant = Some(f);
    }

    fn configure(&mut self) {
        let sr = self.sr;
        use Preset::*;
        match self.kind {
            Male => self.natural(-4.0, -2.0),
            Female => self.natural(5.0, 2.5),
            Helium => self.p1.set_semitones(8.0),
            Baby => self.natural(9.0, 6.0),
            Giant => {
                self.natural(-7.0, -5.0);
                self.f[0] = Biquad::low_shelf(sr, 160.0, 5.0);
            }
            Demon => {
                self.p1.set_semitones(-7.0);
                self.p2.set_semitones(-12.0);
                self.f[0] = Biquad::low_shelf(sr, 200.0, 6.0);
            }
            Monster => self.p1.set_semitones(-10.0),
            Cyborg => self.p1.set_semitones(-2.0),
            Clone => self.p1.set_semitones(0.18),
            Radio => {
                self.f[0] = Biquad::highpass(sr, 450.0, 0.8);
                self.f[1] = Biquad::lowpass(sr, 2_800.0, 0.8);
            }
            Telephone => {
                self.f[0] = Biquad::highpass(sr, 320.0, 0.9);
                self.f[1] = Biquad::lowpass(sr, 3_300.0, 0.9);
            }
            Walkie => {
                self.f[0] = Biquad::highpass(sr, 550.0, 1.0);
                self.f[1] = Biquad::lowpass(sr, 2_400.0, 1.0);
            }
            Megaphone => {
                self.f[0] = Biquad::highpass(sr, 650.0, 0.8);
                self.f[1] = Biquad::bandpass(sr, 1_600.0, 0.7);
            }
            Underwater => self.f[0] = Biquad::lowpass(sr, 700.0, 2.0),
            Whisper => {
                self.f[0] = Biquad::bandpass(sr, 2_600.0, 0.6);
                self.f[1] = Biquad::highpass(sr, 1_500.0, 0.7);
            }
            Alien | Mutation | SlowMutation | FastMutation | Robot | Atari | Silence => {}
        }
    }

    #[inline]
    fn advance_lfo(&mut self, hz: f32) -> f32 {
        self.lfo = (self.lfo + hz / self.sr) % 1.0;
        self.lfo
    }

    #[inline]
    fn ring(&mut self, hz: f32) -> f32 {
        self.osc = (self.osc + hz / self.sr) % 1.0;
        (TAU * self.osc).sin()
    }

    #[inline]
    fn crush(&mut self, x: f32, every: u32, bits: f32) -> f32 {
        if self.hold_count == 0 {
            let steps = 2f32.powf(bits);
            self.hold = (x * steps).round() / steps;
        }
        self.hold_count = (self.hold_count + 1) % every.max(1);
        self.hold
    }

    #[inline]
    fn band(&mut self, x: f32) -> f32 {
        let a = self.f[0].process(x);
        self.f[1].process(a)
    }

    /// Triangle sweep between `lo` and `hi` semitones, retuned every 32 samples.
    #[inline]
    fn sweep_pitch(&mut self, hz: f32, lo: f32, hi: f32) {
        let ph = self.advance_lfo(hz);
        self.tick = self.tick.wrapping_add(1);
        if self.tick % 32 == 0 {
            let tri = 1.0 - (2.0 * ph - 1.0).abs();
            self.p1.set_semitones(lo + (hi - lo) * tri);
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        use Preset::*;
        match self.kind {
            Helium => self.p1.process(x),
            Male | Female | Baby => {
                let y = self.p1.process(x);
                self.formant.as_mut().map_or(y, |f| f.process(y))
            }
            Giant => {
                let y = self.p1.process(x);
                let y = self.formant.as_mut().map_or(y, |f| f.process(y));
                self.f[0].process(y)
            }
            Demon => {
                let y = 0.65 * self.p1.process(x) + 0.65 * self.p2.process(x);
                self.f[0].process((1.6 * y).tanh() * 0.8)
            }
            Monster => {
                let y = self.p1.process(x);
                let growl = 0.7 + 0.3 * self.ring(38.0);
                (1.8 * y * growl).tanh() * 0.75
            }
            Alien => {
                let ph = self.advance_lfo(3.0);
                self.tick = self.tick.wrapping_add(1);
                if self.tick % 32 == 0 {
                    let s = if ph < 0.5 { -3.0 } else { 6.0 };
                    self.p1.set_semitones(s);
                }
                let y = self.p1.process(x);
                y * (0.8 + 0.2 * self.ring(31.0))
            }
            Mutation => {
                self.sweep_pitch(0.45, -4.0, 13.0);
                self.p1.process(x)
            }
            SlowMutation => {
                self.sweep_pitch(0.12, -4.0, 13.0);
                self.p1.process(x)
            }
            FastMutation => {
                self.sweep_pitch(1.6, -4.0, 13.0);
                self.p1.process(x)
            }
            Clone => 0.55 * x + 0.55 * self.p1.process(x),
            Robot => {
                let d = self.comb.read(self.sr * 0.0085);
                let y = x + d * 0.62;
                self.comb.write(y);
                y * (0.55 + 0.45 * self.ring(72.0)) * 0.8
            }
            Cyborg => {
                let y = self.p1.process(x);
                let d = self.comb.read(self.sr * 0.003);
                let z = y + d * 0.55;
                self.comb.write(z);
                0.5 * z + 0.5 * z * self.ring(185.0)
            }
            Atari => {
                let r = x.max(0.0) * 2.2;
                let every = (self.sr / 11_025.0).round() as u32;
                self.crush(r, every, 4.0)
            }
            Radio => {
                let y = self.band(x);
                (3.0 * y).tanh() * 0.45 + self.noise.next() * 0.004
            }
            Telephone => {
                let y = self.band(x);
                let every = (self.sr / 8_000.0).round() as u32;
                self.crush((2.0 * y).tanh(), every, 7.0)
            }
            Walkie => {
                let y = self.band(x * 3.0);
                y.clamp(-0.35, 0.35) * 1.6 + self.noise.next() * 0.006
            }
            Megaphone => {
                let y = self.band(x);
                (7.0 * y).tanh() * 0.5
            }
            Underwater => {
                let ph = self.advance_lfo(0.7);
                self.tick = self.tick.wrapping_add(1);
                if self.tick % 64 == 0 {
                    let cutoff = 450.0 + 400.0 * (0.5 + 0.5 * (TAU * ph).sin());
                    self.f[0].retune(Biquad::lowpass(self.sr, cutoff, 3.0));
                    self.p1.set_semitones(0.6 * (TAU * ph * 1.7).sin());
                }
                self.f[0].process(self.p1.process(x)) * 1.4
            }
            Whisper => {
                let e = self.env.process(x);
                let n = self.f[0].process(self.noise.next()) * e * 6.0;
                n + self.f[1].process(x) * 0.3
            }
            Silence => 0.0,
        }
    }
}
