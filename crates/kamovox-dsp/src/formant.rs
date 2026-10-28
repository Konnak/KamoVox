use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::filters::Noise;

/// Moves the spectral envelope (formants, i.e. the timbre) without touching the
/// pitch. The envelope is estimated per STFT frame by cepstral smoothing; each
/// bin is rescaled by `env(f / ratio) / env(f)`.
///
/// It can also add breathiness by "whisperization": part of each upper bin is
/// replaced by the same magnitude with a random phase, so the added air follows
/// the voice's own spectrum and disappears when you stop talking.
pub struct FormantShifter {
    n: usize,
    hop: usize,
    lifter: usize,
    ratio: f32,
    breath: f32,
    /// Per-bin share of the breath effect (0 below ~700 Hz, 1 from ~2.5 kHz).
    air: Vec<f32>,
    noise: Noise,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    in_fifo: Vec<f32>,
    out_fifo: Vec<f32>,
    accum: Vec<f32>,
    rover: usize,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
    cep: Vec<f32>,
    cep_spec: Vec<Complex<f32>>,
    env: Vec<f32>,
    /// Log envelope after the formant shift.
    target: Vec<f32>,
    scratch_fwd: Vec<Complex<f32>>,
    scratch_inv: Vec<Complex<f32>>,
}

impl FormantShifter {
    pub fn new(sr: f32) -> Self {
        // ~21 ms @ 48 kHz (1024). Used only when the formant slider is moved;
        // pitch-only stays in the time-domain shifter.
        let n = ((sr * 0.0213) as usize).next_power_of_two().max(256);
        let hop = n / 4;
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let bins = n / 2 + 1;
        let bin_hz = sr / n as f32;
        let smooth = |lo: f32, hi: f32, f: f32| {
            let t = ((f - lo) / (hi - lo)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        Self {
            n,
            hop,
            lifter: ((sr * 0.0007) as usize).max(8),
            ratio: 1.0,
            breath: 0.0,
            air: (0..bins)
                .map(|k| {
                    let f = k as f32 * bin_hz;
                    smooth(400.0, 2_000.0, f) * (1.0 - 0.5 * smooth(9_000.0, 14_000.0, f))
                })
                .collect(),
            noise: Noise::default(),
            window: (0..n).map(|i| 0.5 - 0.5 * (TAU * i as f32 / n as f32).cos()).collect(),
            in_fifo: vec![0.0; n],
            out_fifo: vec![0.0; n],
            accum: vec![0.0; 2 * n],
            rover: n - hop,
            time: vec![0.0; n],
            spec: vec![Complex::default(); bins],
            cep: vec![0.0; n],
            cep_spec: vec![Complex::default(); bins],
            env: vec![0.0; bins],
            target: vec![0.0; bins],
            scratch_fwd: fwd.make_scratch_vec(),
            scratch_inv: inv.make_scratch_vec(),
            fwd,
            inv,
        }
    }

    /// `ratio` > 1 moves formants up (smaller/brighter), < 1 down (bigger/darker).
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.clamp(0.5, 2.0);
    }

    /// 0..1 share of the upper spectrum turned into breath noise.
    pub fn set_breath(&mut self, amount: f32) {
        self.breath = amount.clamp(0.0, 1.0);
    }

    pub fn is_neutral(&self) -> bool {
        (self.ratio - 1.0).abs() < 1e-4 && self.breath == 0.0
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let latency = self.n - self.hop;
        self.in_fifo[self.rover] = x;
        let y = self.out_fifo[self.rover - latency];
        self.rover += 1;
        if self.rover >= self.n {
            self.rover = latency;
            self.frame();
        }
        y
    }

    fn frame(&mut self) {
        let n = self.n;
        let bins = n / 2 + 1;
        for i in 0..n {
            self.time[i] = self.in_fifo[i] * self.window[i];
        }
        let _ = self.fwd.process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_fwd);
        let warp = (self.ratio - 1.0).abs() >= 1e-4;
        if warp || self.breath > 0.0 {
            self.estimate_envelope();
        }
        if warp {
            self.warp();
        } else {
            self.target.copy_from_slice(&self.env);
        }
        if self.breath > 0.0 {
            // Aspiration noise shaped by the (shifted) envelope fills the gaps
            // between harmonics, while the harmonics themselves soften.
            for k in 1..bins - 1 {
                let a = self.breath * self.air[k];
                if a <= 0.0 {
                    continue;
                }
                let mag = self.target[k].exp() * a * 2.6;
                let ph = self.noise.next() * PI;
                self.spec[k] = self.spec[k] * (1.0 - 0.55 * a) + Complex::from_polar(mag, ph);
            }
        }
        self.spec[0].im = 0.0;
        self.spec[bins - 1].im = 0.0;
        let _ = self.inv.process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_inv);

        // Hann analysis + synthesis at 75 % overlap sums to 1.5.
        let norm = 1.0 / (n as f32 * 1.5);
        for i in 0..n {
            self.accum[i] += self.time[i] * self.window[i] * norm;
        }
        self.out_fifo[..self.hop].copy_from_slice(&self.accum[..self.hop]);
        self.accum.copy_within(self.hop.., 0);
        let len = self.accum.len();
        self.accum[len - self.hop..].fill(0.0);
        self.in_fifo.copy_within(self.hop.., 0);
    }

    fn estimate_envelope(&mut self) {
        let n = self.n;
        let bins = n / 2 + 1;
        for k in 0..bins {
            self.cep_spec[k] = Complex::new((self.spec[k].norm() + 1e-6).ln(), 0.0);
        }
        let _ = self.inv.process_with_scratch(&mut self.cep_spec, &mut self.cep, &mut self.scratch_inv);
        let scale = 1.0 / n as f32;
        for (i, c) in self.cep.iter_mut().enumerate() {
            let keep = i < self.lifter || i > n - self.lifter;
            *c = if keep { *c * scale } else { 0.0 };
        }
        let _ = self.fwd.process_with_scratch(&mut self.cep, &mut self.cep_spec, &mut self.scratch_fwd);
        for k in 0..bins {
            self.env[k] = self.cep_spec[k].re;
        }
    }

    fn warp(&mut self) {
        let bins = self.n / 2 + 1;
        let inv_ratio = 1.0 / self.ratio;
        let last = (bins - 1) as f32;
        for k in 0..bins {
            let src = (k as f32 * inv_ratio).min(last);
            let i0 = src as usize;
            let i1 = (i0 + 1).min(bins - 1);
            let f = src - i0 as f32;
            let target = self.env[i0] + (self.env[i1] - self.env[i0]) * f;
            let delta = (target - self.env[k]).clamp(-7.0, 2.5);
            self.target[k] = self.env[k] + delta;
            self.spec[k] *= delta.exp();
        }
    }
}
