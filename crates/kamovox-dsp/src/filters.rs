use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Default for Biquad {
    fn default() -> Self {
        Self::identity()
    }
}

impl Biquad {
    pub fn identity() -> Self {
        Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, z1: 0.0, z2: 0.0 }
    }

    fn normalized(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: 0.0, z2: 0.0 }
    }

    fn omega(sr: f32, freq: f32) -> (f32, f32) {
        let f = freq.clamp(10.0, sr * 0.45);
        let w0 = 2.0 * PI * f / sr;
        (w0.cos(), w0.sin())
    }

    pub fn lowpass(sr: f32, freq: f32, q: f32) -> Self {
        let (c, s) = Self::omega(sr, freq);
        let alpha = s / (2.0 * q);
        Self::normalized((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    pub fn highpass(sr: f32, freq: f32, q: f32) -> Self {
        let (c, s) = Self::omega(sr, freq);
        let alpha = s / (2.0 * q);
        Self::normalized((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// Band-pass with 0 dB peak gain.
    pub fn bandpass(sr: f32, freq: f32, q: f32) -> Self {
        let (c, s) = Self::omega(sr, freq);
        let alpha = s / (2.0 * q);
        Self::normalized(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// Peaking EQ (boost or cut around `freq`).
    pub fn peaking(sr: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let (c, s) = Self::omega(sr, freq);
        let alpha = s / (2.0 * q);
        Self::normalized(1.0 + alpha * a, -2.0 * c, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * c, 1.0 - alpha / a)
    }

    pub fn low_shelf(sr: f32, freq: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let (c, s) = Self::omega(sr, freq);
        let k = 2.0 * a.sqrt() * (s / 2.0 * 2f32.sqrt());
        Self::normalized(
            a * ((a + 1.0) - (a - 1.0) * c + k),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * ((a + 1.0) - (a - 1.0) * c - k),
            (a + 1.0) + (a - 1.0) * c + k,
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            (a + 1.0) + (a - 1.0) * c - k,
        )
    }

    pub fn high_shelf(sr: f32, freq: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let (c, s) = Self::omega(sr, freq);
        let k = 2.0 * a.sqrt() * (s / 2.0 * 2f32.sqrt());
        Self::normalized(
            a * ((a + 1.0) + (a - 1.0) * c + k),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - k),
            (a + 1.0) - (a - 1.0) * c + k,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - k,
        )
    }

    /// Replaces the coefficients while keeping the filter state, so sweeps stay click-free.
    pub fn retune(&mut self, other: Biquad) {
        self.b0 = other.b0;
        self.b1 = other.b1;
        self.b2 = other.b2;
        self.a1 = other.a1;
        self.a2 = other.a2;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Peak envelope follower with separate attack and release times.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    attack: f32,
    release: f32,
    pub value: f32,
}

impl Envelope {
    pub fn new(sr: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self { attack: coef(sr, attack_ms), release: coef(sr, release_ms), value: 0.0 }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let target = x.abs();
        let k = if target > self.value { self.attack } else { self.release };
        self.value += (target - self.value) * k;
        self.value
    }
}

pub fn coef(sr: f32, ms: f32) -> f32 {
    1.0 - (-1.0 / (sr * ms.max(0.01) / 1000.0)).exp()
}

/// Circular delay line with fractional (linear) reads.
#[derive(Clone, Debug)]
pub struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    pub fn new(len: usize) -> Self {
        Self { buf: vec![0.0; len.max(4)], pos: 0 }
    }

    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }

    #[inline]
    pub fn write(&mut self, x: f32) {
        self.buf[self.pos] = x;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
    }

    /// Reads the sample written `delay` samples ago (1.0 = previous write).
    #[inline]
    pub fn read(&self, delay: f32) -> f32 {
        let len = self.buf.len();
        let d = delay.clamp(1.0, (len - 2) as f32);
        let mut p = self.pos as f32 - d;
        if p < 0.0 {
            p += len as f32;
        }
        let i0 = p as usize % len;
        let i1 = (i0 + 1) % len;
        let frac = p - p.floor();
        self.buf[i0] + (self.buf[i1] - self.buf[i0]) * frac
    }
}

/// Small xorshift noise source; deterministic and allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct Noise(u32);

impl Default for Noise {
    fn default() -> Self {
        Noise(0x9E37_79B9)
    }
}

impl Noise {
    #[inline]
    pub fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
