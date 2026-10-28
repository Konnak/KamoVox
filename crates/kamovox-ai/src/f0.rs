//! Pitch tracking at 16 kHz with one frame every 10 ms (hop 160), centred frames.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const SR: usize = 16_000;
pub const HOP: usize = 160;
pub const F0_MIN: f32 = 50.0;
pub const F0_MAX: f32 = 1_100.0;

/// Number of centred frames produced for `len` samples.
pub fn frames_for(len: usize) -> usize {
    len / HOP + 1
}

/// RVC's coarse pitch: mel-scaled f0 quantised to 1..=255 (1 = unvoiced).
pub fn coarse(f0: f32) -> i64 {
    let mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
    let (lo, hi) = (mel(F0_MIN), mel(F0_MAX));
    let m = mel(f0);
    let v = if m > 0.0 { (m - lo) * 254.0 / (hi - lo) + 1.0 } else { m };
    v.clamp(1.0, 255.0).round() as i64
}

fn reflect(i: isize, len: usize) -> usize {
    let n = len as isize;
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = i.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}

const N_FFT: usize = 1024;
const N_MELS: usize = 128;
pub const RMVPE_BINS: usize = 360;

/// Log-mel front end of RMVPE (n_fft 1024, hop 160, 128 HTK mels 30..8000 Hz, Slaney norm).
pub struct MelFrontEnd {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Sparse filters: (first bin, weights).
    filters: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<realfft::num_complex::Complex<f32>>,
    scratch: Vec<realfft::num_complex::Complex<f32>>,
    mag: Vec<f32>,
}

impl MelFrontEnd {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let window = (0..N_FFT).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N_FFT as f32).cos()).collect();
        let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
        let (lo, hi) = (hz_to_mel(30.0), hz_to_mel(8_000.0));
        let pts: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
        let bins = N_FFT / 2 + 1;
        let freqs: Vec<f64> = (0..bins).map(|k| k as f64 * SR as f64 / N_FFT as f64).collect();
        let filters = (0..N_MELS)
            .map(|m| {
                let enorm = 2.0 / (pts[m + 2] - pts[m]);
                let w: Vec<f32> = freqs
                    .iter()
                    .map(|&f| {
                        let lower = (f - pts[m]) / (pts[m + 1] - pts[m]);
                        let upper = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
                        (lower.min(upper).max(0.0) * enorm) as f32
                    })
                    .collect();
                let first = w.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = w.iter().rposition(|&v| v > 0.0).unwrap_or(first);
                (first, w[first..=last].to_vec())
            })
            .collect();
        Self {
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            frame: fft.make_input_vec(),
            fft,
            window,
            filters,
            mag: vec![0.0; bins],
        }
    }

    /// Returns `[128, frames]` row-major log-mel, with frames padded (reflect) to a multiple of 32.
    pub fn compute(&mut self, audio: &[f32]) -> (Vec<f32>, usize, usize) {
        let frames = frames_for(audio.len());
        let padded = frames.div_ceil(32) * 32;
        let mut mel = vec![0f32; N_MELS * padded];
        for t in 0..frames {
            let start = (t * HOP) as isize - (N_FFT / 2) as isize;
            for i in 0..N_FFT {
                self.frame[i] = audio[reflect(start + i as isize, audio.len())] * self.window[i];
            }
            let _ = self.fft.process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch);
            for (m, c) in self.mag.iter_mut().zip(&self.spectrum) {
                *m = c.norm();
            }
            for (m, (first, w)) in self.filters.iter().enumerate() {
                let e: f32 = w.iter().zip(&self.mag[*first..]).map(|(a, b)| a * b).sum();
                mel[m * padded + t] = e.max(1e-5).ln();
            }
        }
        for t in frames..padded {
            let src = reflect(t as isize, frames);
            for m in 0..N_MELS {
                mel[m * padded + t] = mel[m * padded + src];
            }
        }
        (mel, frames, padded)
    }
}

/// Turns RMVPE salience `[frames, 360]` into Hz (0 = unvoiced).
pub fn rmvpe_decode(salience: &[f32], frames: usize, threshold: f32) -> Vec<f32> {
    let cents = |i: usize| 20.0 * i as f32 + 1997.379_4;
    (0..frames)
        .map(|t| {
            let row = &salience[t * RMVPE_BINS..(t + 1) * RMVPE_BINS];
            let (center, &peak) = row.iter().enumerate().fold((0, &f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
            if peak <= threshold {
                return 0.0;
            }
            let lo = center.saturating_sub(4);
            let hi = (center + 5).min(RMVPE_BINS);
            let (mut num, mut den) = (0f32, 0f32);
            for (i, &s) in row.iter().enumerate().take(hi).skip(lo) {
                num += s * cents(i);
                den += s;
            }
            if den <= 0.0 {
                return 0.0;
            }
            10.0 * 2f32.powf(num / den / 1200.0)
        })
        .collect()
}

/// Lightweight CPU tracker (YIN, de Cheveigné & Kawahara 2002).
pub struct Yin {
    frame: Vec<f32>,
    diff: Vec<f32>,
}

const YIN_WINDOW: usize = 1024;
const YIN_THRESHOLD: f32 = 0.15;

impl Yin {
    pub fn new() -> Self {
        Self { frame: vec![0.0; YIN_WINDOW], diff: vec![0.0; YIN_WINDOW / 2] }
    }

    pub fn track(&mut self, audio: &[f32]) -> Vec<f32> {
        let frames = frames_for(audio.len());
        let tau_min = (SR as f32 / F0_MAX) as usize;
        let tau_max = ((SR as f32 / F0_MIN) as usize).min(YIN_WINDOW / 2 - 1);
        let w = YIN_WINDOW - tau_max;
        (0..frames)
            .map(|t| {
                let start = (t * HOP) as isize - (YIN_WINDOW / 2) as isize;
                let mut energy = 0f32;
                for i in 0..YIN_WINDOW {
                    let v = audio[reflect(start + i as isize, audio.len())];
                    self.frame[i] = v;
                    energy += v * v;
                }
                if energy / (YIN_WINDOW as f32) < 1e-6 {
                    return 0.0;
                }
                let x = &self.frame;
                self.diff[0] = 1.0;
                let mut running = 0f32;
                for tau in 1..=tau_max {
                    let d: f32 = (0..w).map(|i| {
                        let e = x[i] - x[i + tau];
                        e * e
                    }).sum();
                    running += d;
                    self.diff[tau] = if running > 0.0 { d * tau as f32 / running } else { 1.0 };
                }
                let mut best = None;
                let mut tau = tau_min;
                while tau < tau_max {
                    if self.diff[tau] < YIN_THRESHOLD {
                        while tau + 1 < tau_max && self.diff[tau + 1] < self.diff[tau] {
                            tau += 1;
                        }
                        best = Some(tau);
                        break;
                    }
                    tau += 1;
                }
                let Some(tau) = best else { return 0.0 };
                let (a, b, c) = (self.diff[tau - 1], self.diff[tau], self.diff[tau + 1]);
                let denom = a - 2.0 * b + c;
                let shift = if denom.abs() > 1e-9 { 0.5 * (a - c) / denom } else { 0.0 };
                SR as f32 / (tau as f32 + shift.clamp(-1.0, 1.0))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(f0: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / SR as f32;
                (1..8).map(|h| (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32).sum::<f32>() * 0.2
            })
            .collect()
    }

    #[test]
    fn yin_finds_the_fundamental() {
        let f = Yin::new().track(&voice(180.0, 8_000));
        let mid = &f[5..f.len() - 5];
        assert!(mid.iter().all(|&v| (v - 180.0).abs() < 2.0), "{mid:?}");
    }

    #[test]
    fn coarse_matches_rvc_bounds() {
        assert_eq!(coarse(0.0), 1);
        assert_eq!(coarse(50.0), 1);
        assert_eq!(coarse(1_100.0), 255);
        assert_eq!(coarse(5_000.0), 255);
    }

    #[test]
    fn mel_frames_are_padded() {
        let (mel, frames, padded) = MelFrontEnd::new().compute(&voice(200.0, 5_120 - 160));
        assert_eq!(frames, 32);
        assert_eq!(padded, 32);
        assert!(mel.iter().all(|v| v.is_finite()));
    }
}
