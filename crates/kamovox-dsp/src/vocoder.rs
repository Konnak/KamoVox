use std::f32::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::filters::{coef, Biquad, Noise};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Carrier {
    Saw,
    Chord,
    Bell,
    Organ,
    Choir,
    Noise,
    Buzz,
}

const BANDS: usize = 18;

/// Channel vocoder: the voice's band envelopes shape a synthesized carrier.
pub struct Vocoder {
    carrier: Carrier,
    sr: f32,
    mod_bands: Vec<Biquad>,
    car_bands: Vec<Biquad>,
    envs: Vec<f32>,
    attack: f32,
    release: f32,
    phases: [f32; 6],
    noise: Noise,
    sibilance: Biquad,
}

impl Vocoder {
    pub fn new(sr: f32, carrier: Carrier) -> Self {
        let lo: f32 = 110.0;
        let hi: f32 = 7_500.0f32.min(sr * 0.42);
        let mut mod_bands = Vec::with_capacity(BANDS);
        let mut car_bands = Vec::with_capacity(BANDS);
        for i in 0..BANDS {
            let f = lo * (hi / lo).powf(i as f32 / (BANDS - 1) as f32);
            mod_bands.push(Biquad::bandpass(sr, f, 5.0));
            car_bands.push(Biquad::bandpass(sr, f, 5.0));
        }
        Self {
            carrier,
            sr,
            mod_bands,
            car_bands,
            envs: vec![0.0; BANDS],
            attack: coef(sr, 3.0),
            release: coef(sr, 40.0),
            phases: [0.0; 6],
            noise: Noise::default(),
            sibilance: Biquad::highpass(sr, 5_000.0, 0.7),
        }
    }

    pub fn carrier(&self) -> Carrier {
        self.carrier
    }

    #[inline]
    fn osc(&mut self, i: usize, freq: f32) -> f32 {
        let p = self.phases[i];
        self.phases[i] = (p + freq / self.sr) % 1.0;
        p
    }

    #[inline]
    fn carrier_sample(&mut self) -> f32 {
        let saw = |p: f32| 2.0 * p - 1.0;
        match self.carrier {
            Carrier::Saw => saw(self.osc(0, 110.0)),
            Carrier::Chord => {
                (saw(self.osc(0, 110.0)) + saw(self.osc(1, 138.6)) + saw(self.osc(2, 164.8))) / 3.0
            }
            Carrier::Bell => {
                let mut s = 0.0;
                for (i, (m, a)) in [(1.0, 1.0), (2.76, 0.6), (5.4, 0.4), (8.93, 0.25)].iter().enumerate() {
                    s += (TAU * self.osc(i, 220.0 * m)).sin() * a;
                }
                s * 0.45
            }
            Carrier::Organ => {
                let mut s = 0.0;
                for (i, h) in [1.0f32, 2.0, 3.0, 4.0, 6.0, 8.0].iter().enumerate() {
                    s += (TAU * self.osc(i, 98.0 * h)).sin() / h;
                }
                s * 0.5
            }
            Carrier::Choir => {
                let mut s = 0.0;
                for (i, f) in [130.8f32, 131.6, 164.8, 165.5, 196.0, 196.9].iter().enumerate() {
                    s += saw(self.osc(i, *f));
                }
                s / 6.0
            }
            Carrier::Noise => self.noise.next() * 0.8,
            Carrier::Buzz => {
                if self.osc(0, 80.0) < 0.12 {
                    1.0
                } else {
                    -0.14
                }
            }
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let c = self.carrier_sample();
        let mut sum = 0.0;
        for i in 0..BANDS {
            let m = self.mod_bands[i].process(x).abs();
            let e = &mut self.envs[i];
            let k = if m > *e { self.attack } else { self.release };
            *e += (m - *e) * k;
            sum += self.car_bands[i].process(c) * *e;
        }
        sum * 14.0 + self.sibilance.process(x) * 0.35
    }
}
