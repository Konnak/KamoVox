use std::f32::consts::TAU;

use crate::filters::DelayLine;

struct Comb {
    line: Vec<f32>,
    idx: usize,
    store: f32,
}

impl Comb {
    #[inline]
    fn process(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let out = self.line[self.idx];
        self.store = out * (1.0 - damp) + self.store * damp;
        self.line[self.idx] = x + self.store * feedback;
        self.idx = (self.idx + 1) % self.line.len();
        out
    }
}

struct Allpass {
    line: Vec<f32>,
    idx: usize,
}

impl Allpass {
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let buf = self.line[self.idx];
        self.line[self.idx] = x + buf * 0.5;
        self.idx = (self.idx + 1) % self.line.len();
        buf - x
    }
}

/// Mono Schroeder/Moorer reverb (Freeverb topology).
pub struct Reverb {
    combs: Vec<Comb>,
    allpasses: Vec<Allpass>,
    feedback: f32,
    damp: f32,
    pub wet: f32,
}

impl Reverb {
    pub fn new(sr: f32, room: f32, damp: f32, wet: f32) -> Self {
        let scale = sr / 44_100.0;
        let size = |n: usize| ((n as f32 * scale) as usize).max(8);
        let combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
            .iter()
            .map(|&n| Comb { line: vec![0.0; size(n)], idx: 0, store: 0.0 })
            .collect();
        let allpasses = [556, 441, 341, 225]
            .iter()
            .map(|&n| Allpass { line: vec![0.0; size(n)], idx: 0 })
            .collect();
        Self { combs, allpasses, feedback: room * 0.28 + 0.7, damp: damp * 0.4, wet }
    }

    pub fn clear(&mut self) {
        for c in &mut self.combs {
            c.line.fill(0.0);
            c.store = 0.0;
        }
        for a in &mut self.allpasses {
            a.line.fill(0.0);
        }
    }

    /// Returns only the reverberated signal.
    #[inline]
    pub fn tail(&mut self, x: f32) -> f32 {
        let input = x * 0.015;
        let mut out = 0.0;
        for c in &mut self.combs {
            out += c.process(input, self.feedback, self.damp);
        }
        for a in &mut self.allpasses {
            out = a.process(out);
        }
        out
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        x + self.tail(x) * self.wet * 3.0
    }
}

pub struct Echo {
    line: DelayLine,
    delay: f32,
    feedback: f32,
    mix: f32,
}

impl Echo {
    pub fn new(sr: f32, ms: f32, feedback: f32, mix: f32) -> Self {
        let delay = sr * ms / 1000.0;
        Self { line: DelayLine::new(delay as usize + 4), delay, feedback, mix }
    }

    pub fn clear(&mut self) {
        self.line.clear();
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let d = self.line.read(self.delay);
        self.line.write(x + d * self.feedback);
        x + d * self.mix
    }
}

/// Modulated short delays: `voices` copies with spread LFO phases.
pub struct Chorus {
    line: DelayLine,
    sr: f32,
    phase: f32,
    rate: f32,
    base_ms: f32,
    depth_ms: f32,
    voices: usize,
    feedback: f32,
    mix: f32,
    last: f32,
}

impl Chorus {
    pub fn chorus(sr: f32) -> Self {
        Self::new(sr, 0.7, 18.0, 4.0, 3, 0.0, 0.55)
    }

    pub fn flanger(sr: f32) -> Self {
        Self::new(sr, 0.25, 2.5, 2.0, 1, 0.6, 0.6)
    }

    fn new(sr: f32, rate: f32, base_ms: f32, depth_ms: f32, voices: usize, feedback: f32, mix: f32) -> Self {
        let len = (sr * (base_ms + depth_ms + 5.0) / 1000.0) as usize;
        Self { line: DelayLine::new(len), sr, phase: 0.0, rate, base_ms, depth_ms, voices, feedback, mix, last: 0.0 }
    }

    pub fn clear(&mut self) {
        self.line.clear();
        self.last = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.line.write(x + self.last * self.feedback);
        let mut sum = 0.0;
        for v in 0..self.voices {
            let ph = self.phase + v as f32 / self.voices as f32;
            let ms = self.base_ms + self.depth_ms * (0.5 + 0.5 * (TAU * ph).sin());
            sum += self.line.read(ms * self.sr / 1000.0);
        }
        let wet = sum / self.voices as f32;
        self.last = wet;
        self.phase = (self.phase + self.rate / self.sr) % 1.0;
        x * (1.0 - self.mix * 0.5) + wet * self.mix
    }
}
