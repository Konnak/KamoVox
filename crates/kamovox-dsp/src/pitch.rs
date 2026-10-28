//! Independent-pointer WSOLA pitch shifter (SoundTouch-style grain stream).
//!
//! Solo most of each grain (gain 1.0) → short Hann crossfade after a single
//! normalized correlation lock. Heads never retarget while audible.

const GRAIN: usize = 2048; // ~42.7 ms @ 48 kHz
const OVERLAP: usize = 512; // ~10.7 ms
const SOLO: usize = GRAIN - OVERLAP; // 1536
const SEEK: i32 = 384;
const CORR_LEN: usize = 256;
const MIN_BEHIND: usize = 768;
const MAX_BEHIND: usize = 6144;

#[derive(Clone, Debug)]
pub struct PitchShifter {
    buf: Vec<f32>,
    write: usize,
    /// Absolute write counter (monotonic). Next write index.
    write_abs: u64,
    read_a: f64,
    read_b: f64,
    step: f64,
    /// Samples into the current grain (0..GRAIN).
    grain_i: usize,
    /// True while overlapping A→B.
    overlapping: bool,
    primed: bool,
}

impl PitchShifter {
    pub fn new(sr: f32) -> Self {
        let _ = sr;
        let cap = MAX_BEHIND + GRAIN + SEEK as usize * 2 + 128;
        Self {
            buf: vec![0.0; cap],
            write: 0,
            write_abs: 0,
            read_a: 0.0,
            read_b: 0.0,
            step: 1.0,
            grain_i: 0,
            overlapping: false,
            primed: false,
        }
    }

    pub fn set_semitones(&mut self, semitones: f32) {
        self.step = 2f64.powf(f64::from(semitones.clamp(-24.0, 24.0)) / 12.0);
    }

    /// API compat — grain sizes are fixed (Haas-safe).
    pub fn set_window_ms(&mut self, _sr: f32, _ms: f32) {}

    /// Fractional absolute index → linear interpolate.
    /// `delay = write_abs - abs`: 1.0 = newest sample.
    #[inline]
    fn at_abs(&self, abs: f64) -> f32 {
        let delay = self.write_abs as f64 - abs;
        if !(1.0..(self.buf.len() - 1) as f64).contains(&delay) {
            return 0.0;
        }
        let d_newer = delay.floor() as usize;
        let d_older = d_newer + 1;
        let frac_delay = (delay - d_newer as f64) as f32;

        let len = self.buf.len();
        let i_newer = (self.write + len - d_newer) % len;
        let i_older = (self.write + len - d_older) % len;

        let s_newer = self.buf[i_newer];
        let s_older = self.buf[i_older];
        s_newer + (s_older - s_newer) * frac_delay
    }

    fn clamp_read(abs: f64, write_abs: u64) -> f64 {
        let w = write_abs as f64;
        let min_abs = w - MAX_BEHIND as f64;
        let max_abs = w - MIN_BEHIND as f64;
        abs.clamp(min_abs.max(0.0), max_abs.max(0.0))
    }

    /// Pitch-up needs more runway behind write; pitch-down stays closer.
    fn ideal_target(&self) -> f64 {
        let w = self.write_abs as f64;
        let offset = if self.step > 1.0 { 3072.0 } else { 1792.0 };
        Self::clamp_read(w - offset, self.write_abs)
    }

    fn ensure_primed(&mut self) {
        if self.primed {
            return;
        }
        if self.write_abs < (MIN_BEHIND + GRAIN + 1024) as u64 {
            return;
        }
        self.read_a = self.ideal_target();
        self.read_b = self.read_a;
        self.grain_i = 0;
        self.overlapping = false;
        self.primed = true;
    }

    /// Place `read_b` once at overlap start (normalized correlation @ `step`).
    fn place_b(&mut self) {
        let target = self.ideal_target();
        let mut best = target;
        let mut best_score = f32::MIN;
        let mut off = -SEEK;
        while off <= SEEK {
            let cand = Self::clamp_read(target + f64::from(off), self.write_abs);
            let mut dot = 0.0f32;
            let mut norm_b = 0.0f32;
            let mut i = 0usize;
            while i < CORR_LEN {
                let step_off = (i as f64) * self.step;
                let a = self.at_abs(self.read_a + step_off);
                let b = self.at_abs(cand + step_off);
                dot += a * b;
                norm_b += b * b;
                i += 2;
            }
            let score = dot / (norm_b.sqrt() + 1e-6);
            if score > best_score {
                best_score = score;
                best = cand;
            }
            off += 2;
        }
        self.read_b = best;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.buf[self.write] = x;
        self.write = (self.write + 1) % self.buf.len();
        self.write_abs = self.write_abs.wrapping_add(1);

        self.ensure_primed();
        if !self.primed {
            return 0.0;
        }

        // Unity: single head, fixed 1:1 — no overlap machinery.
        if (self.step - 1.0).abs() < 1e-6 {
            let y = self.at_abs(self.read_a);
            self.read_a = Self::clamp_read(self.read_a + 1.0, self.write_abs);
            self.grain_i = 0;
            self.overlapping = false;
            return y;
        }

        // Start overlap exactly once per grain.
        if !self.overlapping && self.grain_i >= SOLO {
            self.place_b();
            self.overlapping = true;
        }

        let y = if self.overlapping {
            let k = self.grain_i - SOLO;
            // Raised-cosine fade — continuous derivative at edges.
            let phase = (k as f32 / OVERLAP as f32) * std::f32::consts::PI;
            let t = 0.5 * (1.0 - phase.cos());
            let a = self.at_abs(self.read_a);
            let b = self.at_abs(self.read_b);
            a * (1.0 - t) + b * t
        } else {
            self.at_abs(self.read_a)
        };

        self.read_a = Self::clamp_read(self.read_a + self.step, self.write_abs);
        if self.overlapping {
            self.read_b = Self::clamp_read(self.read_b + self.step, self.write_abs);
        }

        self.grain_i += 1;
        if self.grain_i >= GRAIN {
            if self.overlapping {
                self.read_a = self.read_b;
            }
            self.grain_i = 0;
            self.overlapping = false;
        }

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|x| x * x).sum::<f32>() / buf.len().max(1) as f32).sqrt()
    }

    fn voice(sr: f32, hz: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let mut x = 0.0f32;
                for h in 1..=6 {
                    x += (std::f32::consts::TAU * hz * h as f32 * t).sin() / h as f32;
                }
                x * 0.15
            })
            .collect()
    }

    #[test]
    fn unity_keeps_level() {
        let mut p = PitchShifter::new(48_000.0);
        p.set_semitones(0.0);
        let input = voice(48_000.0, 140.0, 48_000);
        let out: Vec<f32> = input.iter().map(|&x| p.process(x)).collect();
        let g = rms(&out[12_000..]) / rms(&input[12_000..]);
        assert!((0.85..1.15).contains(&g), "unity gain {g}");
    }

    #[test]
    fn small_shift_intelligible_level() {
        let mut p = PitchShifter::new(48_000.0);
        p.set_semitones(-1.7);
        let input = voice(48_000.0, 140.0, 48_000);
        let out: Vec<f32> = input.iter().map(|&x| p.process(x)).collect();
        assert!(out.iter().all(|v| v.is_finite()));
        let g = rms(&out[12_000..]) / rms(&input[12_000..]);
        assert!((0.45..1.6).contains(&g), "gain {g}");
        let body = &out[12_000..];
        let peak = body.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let crest = peak / rms(body).max(1e-6);
        assert!(crest < 10.0, "crest {crest}");
    }

    #[test]
    fn pitch_up_does_not_clamp_mid_grain() {
        let mut p = PitchShifter::new(48_000.0);
        p.set_semitones(6.0);
        let input = voice(48_000.0, 140.0, 48_000);
        let out: Vec<f32> = input.iter().map(|&x| p.process(x)).collect();
        assert!(out.iter().all(|v| v.is_finite()));
        let g = rms(&out[16_000..]) / rms(&input[16_000..]);
        assert!((0.35..1.8).contains(&g), "pitch-up gain {g}");
    }

    #[test]
    fn chunked_matches_stream() {
        let mut a = PitchShifter::new(48_000.0);
        let mut b = PitchShifter::new(48_000.0);
        a.set_semitones(2.0);
        b.set_semitones(2.0);
        let src = voice(48_000.0, 160.0, 16_000);
        let xa: Vec<f32> = src.iter().map(|&x| a.process(x)).collect();
        let mut xb = Vec::new();
        for chunk in src.chunks(256) {
            for &x in chunk {
                xb.push(b.process(x));
            }
        }
        let err = xa.iter().zip(&xb).map(|(u, v)| (u - v).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-4, "diverge {err}");
    }

    #[test]
    fn heads_stay_in_safe_zone() {
        let mut p = PitchShifter::new(48_000.0);
        p.set_semitones(-5.0);
        for &x in &voice(48_000.0, 120.0, 24_000) {
            let _ = p.process(x);
            if p.primed {
                let behind = p.write_abs as f64 - p.read_a;
                assert!(behind >= MIN_BEHIND as f64 - 1.0, "behind {behind}");
                assert!(behind <= MAX_BEHIND as f64 + 1.0, "behind {behind}");
            }
        }
    }
}
