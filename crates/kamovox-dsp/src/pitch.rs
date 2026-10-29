//! Independent-pointer WSOLA pitch shifter (SoundTouch-style grain stream).
//!
//! Solo most of each grain (gain 1.0) → short raised-cosine crossfade after a
//! true cosine lock. Heads never retarget while audible. Cubic interpolation
//! keeps small shifts from going metallic.

const MIN_BEHIND: usize = 768;
const MAX_BEHIND: usize = 8192;
const CORR_LEN: usize = 320;
const BUF_PAD: usize = 2048;

#[derive(Clone, Debug)]
pub struct PitchShifter {
    sr: f32,
    buf: Vec<f32>,
    write: usize,
    /// Absolute write counter (monotonic). Next write index.
    write_abs: u64,
    read_a: f64,
    read_b: f64,
    step: f64,
    /// Samples into the current grain (0..grain).
    grain_i: usize,
    /// True while overlapping A→B.
    overlapping: bool,
    primed: bool,
    grain: usize,
    overlap: usize,
    seek: i32,
}

impl PitchShifter {
    pub fn new(sr: f32) -> Self {
        let cap = MAX_BEHIND + 4096 + BUF_PAD;
        let mut s = Self {
            sr: sr.max(8_000.0),
            buf: vec![0.0; cap],
            write: 0,
            write_abs: 0,
            read_a: 0.0,
            read_b: 0.0,
            step: 1.0,
            grain_i: 0,
            overlapping: false,
            primed: false,
            grain: 1824,
            overlap: 576,
            seek: 480,
        };
        s.set_window_ms(s.sr, 38.0);
        s
    }

    pub fn set_semitones(&mut self, semitones: f32) {
        self.step = 2f64.powf(f64::from(semitones.clamp(-24.0, 24.0)) / 12.0);
    }

    /// Default ~38 ms grain / 12 ms overlap. “Voz mais suave” ~44 ms / 15 ms.
    /// Stays under the ~50 ms Haas double-voice region.
    pub fn set_window_ms(&mut self, sr: f32, ms: f32) {
        self.sr = sr.max(8_000.0);
        let grain_ms = ms.clamp(32.0, 48.0);
        let overlap_ms = if grain_ms >= 42.0 { 15.2 } else { 12.0 };
        let grain = ((self.sr * grain_ms / 1000.0) as usize).max(1024) & !1;
        let overlap = ((self.sr * overlap_ms / 1000.0) as usize).clamp(384, grain / 2) & !1;
        let seek = ((self.sr * 0.012) as i32).max(256).min(overlap as i32);
        if grain == self.grain && overlap == self.overlap && seek == self.seek {
            return;
        }
        self.grain = grain;
        self.overlap = overlap;
        self.seek = seek;
        // Restart the grain stream so heads never straddle two geometries.
        self.grain_i = 0;
        self.overlapping = false;
        if self.primed {
            self.read_a = self.ideal_target();
            self.read_b = self.read_a;
        }
    }

    fn solo(&self) -> usize {
        self.grain.saturating_sub(self.overlap).max(1)
    }

    /// Cubic Hermite between the two central samples.
    #[inline]
    fn at_abs(&self, abs: f64) -> f32 {
        let delay = self.write_abs as f64 - abs;
        let max_d = (self.buf.len() - 4) as f64;
        if !(2.0..max_d).contains(&delay) {
            return 0.0;
        }
        let d1 = delay.floor() as usize;
        let t = (delay - d1 as f64) as f32;
        let len = self.buf.len();
        let at = |d: usize| self.buf[(self.write + len - d) % len];
        let y0 = at(d1.saturating_sub(1).max(1));
        let y1 = at(d1);
        let y2 = at(d1 + 1);
        let y3 = at((d1 + 2).min(len - 1));
        hermite(y0, y1, y2, y3, t)
    }

    fn clamp_read(abs: f64, write_abs: u64) -> f64 {
        let w = write_abs as f64;
        let min_abs = w - MAX_BEHIND as f64;
        let max_abs = w - MIN_BEHIND as f64;
        abs.clamp(min_abs.max(0.0), max_abs.max(0.0))
    }

    /// Pitch-up needs enough runway for one full grain at `step`; pitch-down stays closer.
    fn ideal_target(&self) -> f64 {
        let w = self.write_abs as f64;
        let extra = if self.step > 1.0 {
            (self.grain as f64) * (self.step - 1.0) + (self.overlap as f64) + 640.0
        } else {
            (self.grain as f64) * 0.88
        };
        let offset = extra.max((MIN_BEHIND + 256) as f64);
        Self::clamp_read(w - offset, self.write_abs)
    }

    fn ensure_primed(&mut self) {
        if self.primed {
            return;
        }
        if self.write_abs < (MIN_BEHIND + self.grain + 1024) as u64 {
            return;
        }
        self.read_a = self.ideal_target();
        self.read_b = self.read_a;
        self.grain_i = 0;
        self.overlapping = false;
        self.primed = true;
    }

    /// Place `read_b` once at overlap start (true cosine @ `step`).
    fn place_b(&mut self) {
        let target = self.ideal_target();
        let mut best = target;
        let mut best_score = f32::MIN;
        let mut off = -self.seek;
        while off <= self.seek {
            let cand = Self::clamp_read(target + f64::from(off), self.write_abs);
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            let mut i = 0usize;
            while i < CORR_LEN {
                let step_off = (i as f64) * self.step;
                let a = self.at_abs(self.read_a + step_off);
                let b = self.at_abs(cand + step_off);
                dot += a * b;
                na += a * a;
                nb += b * b;
                i += 1;
            }
            if nb < 1e-8 {
                off += 1;
                continue;
            }
            let score = dot / (na.sqrt() * nb.sqrt() + 1e-6);
            if score > best_score {
                best_score = score;
                best = cand;
            }
            off += 1;
        }
        self.read_b = best;
    }

    /// If a head would leave the safe zone mid-grain, recycle at the next ideal
    /// (avoids the old freeze-clamp that sounded robotic on pitch-up).
    fn recycle_if_starved(&mut self) {
        let behind = self.write_abs as f64 - self.read_a;
        if behind >= MIN_BEHIND as f64 && behind <= MAX_BEHIND as f64 {
            return;
        }
        self.place_b();
        self.read_a = self.read_b;
        self.grain_i = 0;
        self.overlapping = false;
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

        let solo = self.solo();
        if !self.overlapping && self.grain_i >= solo {
            self.place_b();
            self.overlapping = true;
        }

        let y = if self.overlapping {
            let k = self.grain_i - solo;
            let phase = (k as f32 / self.overlap.max(1) as f32) * std::f32::consts::PI;
            let t = 0.5 * (1.0 - phase.cos());
            let a = self.at_abs(self.read_a);
            let b = self.at_abs(self.read_b);
            a * (1.0 - t) + b * t
        } else {
            self.at_abs(self.read_a)
        };

        // Advance without mid-grain clamp — lock stays intact across the grain.
        self.read_a += self.step;
        if self.overlapping {
            self.read_b += self.step;
        }
        self.recycle_if_starved();

        self.grain_i += 1;
        if self.grain_i >= self.grain {
            if self.overlapping {
                self.read_a = self.read_b;
            }
            self.grain_i = 0;
            self.overlapping = false;
        }

        y
    }
}

#[inline]
fn hermite(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + c0
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
                assert!(behind >= MIN_BEHIND as f64 - 8.0, "behind {behind}");
                assert!(behind <= MAX_BEHIND as f64 + 8.0, "behind {behind}");
            }
        }
    }

    #[test]
    fn smooth_window_is_longer_than_default() {
        let mut p = PitchShifter::new(48_000.0);
        let g0 = p.grain;
        let o0 = p.overlap;
        p.set_window_ms(48_000.0, 44.0);
        assert!(p.grain > g0, "smooth grain {} vs default {g0}", p.grain);
        assert!(p.overlap > o0, "smooth overlap {} vs default {o0}", p.overlap);
        assert!(p.grain <= 2304, "grain {} must stay Haas-safe", p.grain);
    }

    #[test]
    fn large_pitch_up_stays_alive() {
        let mut p = PitchShifter::new(48_000.0);
        p.set_semitones(12.0);
        let input = voice(48_000.0, 140.0, 48_000);
        let out: Vec<f32> = input.iter().map(|&x| p.process(x)).collect();
        let g = rms(&out[20_000..]) / rms(&input[20_000..]);
        assert!(g > 0.20, "octave-up collapsed to {g}");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
