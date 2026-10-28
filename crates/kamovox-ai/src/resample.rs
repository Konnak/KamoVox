//! Stateless polyphase windowed-sinc resampler for rational rate pairs. Every call
//! treats its input as an isolated block; callers overlap blocks themselves.

const ZERO_CROSSINGS: f64 = 12.0;
const KAISER_BETA: f64 = 8.0;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > sum * 1e-12 {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

pub struct Resampler {
    up: usize,
    down: usize,
    half: usize,
    /// `table[phase * 2 * half + j]` weights input sample `i + j + 1 - half`.
    table: Vec<f32>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        let g = gcd(from, to);
        let (up, down) = ((to / g) as usize, (from / g) as usize);
        let cutoff = (to as f64 / from as f64).min(1.0) * 0.97;
        let half = (ZERO_CROSSINGS / cutoff).ceil() as usize;
        let mut table = Vec::with_capacity(up * 2 * half);
        let i0b = bessel_i0(KAISER_BETA);
        for p in 0..up {
            let frac = p as f64 / up as f64;
            for j in 0..2 * half {
                let d = frac - (j as f64 + 1.0 - half as f64);
                let x = d * cutoff;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let r = d / half as f64;
                let w = if r.abs() >= 1.0 { 0.0 } else { bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / i0b };
                table.push((cutoff * sinc * w) as f32);
            }
        }
        Self { up, down, half, table }
    }

    pub fn is_identity(&self) -> bool {
        self.up == self.down
    }

    /// Writes `out.len()` samples starting at input position 0.
    pub fn process(&self, input: &[f32], out: &mut [f32]) {
        if self.is_identity() {
            for (o, i) in out.iter_mut().zip(input.iter().chain(std::iter::repeat(&0.0))) {
                *o = *i;
            }
            return;
        }
        let taps = 2 * self.half;
        let len = input.len() as isize;
        for (n, o) in out.iter_mut().enumerate() {
            let pos = n * self.down;
            let (i, phase) = ((pos / self.up) as isize, pos % self.up);
            let h = &self.table[phase * taps..(phase + 1) * taps];
            let start = i + 1 - self.half as isize;
            let mut acc = 0f32;
            if start >= 0 && start + taps as isize <= len {
                let x = &input[start as usize..start as usize + taps];
                for (a, b) in h.iter().zip(x) {
                    acc += a * b;
                }
            } else {
                for (j, a) in h.iter().enumerate() {
                    let k = start + j as isize;
                    if k >= 0 && k < len {
                        acc += a * input[k as usize];
                    }
                }
            }
            *o = acc;
        }
    }

    pub fn output_len(&self, input_len: usize) -> usize {
        input_len * self.up / self.down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (std::f32::consts::TAU * f * i as f32 / sr as f32).sin()).collect()
    }

    #[test]
    fn keeps_a_tone_through_common_ratios() {
        for (from, to) in [(48_000, 16_000), (44_100, 16_000), (40_000, 48_000), (32_000, 44_100), (48_000, 44_100)] {
            let r = Resampler::new(from, to);
            let input = tone(440.0, from, from as usize / 10);
            let mut out = vec![0.0; r.output_len(input.len())];
            r.process(&input, &mut out);
            let expect = tone(440.0, to, out.len());
            let err = (out.len() / 4..out.len() * 3 / 4).map(|i| (out[i] - expect[i]).abs()).fold(0f32, f32::max);
            assert!(err < 2e-3, "{from}->{to}: {err}");
        }
    }
}
