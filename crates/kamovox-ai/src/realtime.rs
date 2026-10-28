//! Block-based real-time conversion. Each block re-runs the networks over a window of
//! recent audio (context + crossfade + search + block) and stitches the result with
//! SOLA so consecutive blocks join without clicks.

use std::borrow::Cow;
use std::time::Instant;

use ort::session::SessionInputValue;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use crate::f0::{self, MelFrontEnd, Yin, HOP, RMVPE_BINS};
use crate::models::{Models, Pitch};
use crate::resample::Resampler;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Timing {
    /// Audio converted per inference.
    pub block_ms: f32,
    /// Past audio fed as context (quality, costs compute, not latency).
    pub context_ms: f32,
    /// Overlap blended between blocks.
    pub crossfade_ms: f32,
}

impl Default for Timing {
    fn default() -> Self {
        Self { block_ms: 250.0, context_ms: 600.0, crossfade_ms: 60.0 }
    }
}

#[derive(Serialize, Clone, Copy, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub infer_ms: f32,
    pub block_ms: f32,
    /// Inference time over block duration; above 1 the GPU/CPU can't keep up.
    pub load: f32,
    /// Microphone-to-output delay added by the conversion.
    pub latency_ms: f32,
    pub pitch_ms: f32,
    pub embed_ms: f32,
    pub synth_ms: f32,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    fn gauss(&mut self, out: &mut [f32]) {
        for pair in out.chunks_mut(2) {
            let (u, v) = (self.next(), self.next());
            let r = (-2.0 * u.ln()).sqrt();
            let a = std::f32::consts::TAU * v;
            pair[0] = r * a.cos();
            if pair.len() > 1 {
                pair[1] = r * a.sin();
            }
        }
    }
}

pub struct Realtime {
    models: Models,
    zc: usize,
    block: usize,
    sola: usize,
    search: usize,
    skip_head: usize,
    ret: usize,
    input: Vec<f32>,
    input16: Vec<f32>,
    to16: Resampler,
    from_model: Resampler,
    cache_f0: Vec<f32>,
    sola_buffer: Vec<f32>,
    fade_in: Vec<f32>,
    mel: MelFrontEnd,
    yin: Yin,
    rng: Rng,
    model_out: Vec<f32>,
    infer: Vec<f32>,
    stats: Stats,
    /// Transposition in semitones applied to the detected pitch.
    pub transpose: f32,
}

fn smooth(old: f32, new: f32) -> f32 {
    if old == 0.0 { new } else { old * 0.8 + new * 0.2 }
}

fn ms_to(sr: usize, zc: usize, ms: f32) -> usize {
    ((ms / 1000.0 * sr as f32 / zc as f32).round() as usize) * zc
}

impl Realtime {
    pub fn new(models: Models, sample_rate: u32, timing: Timing) -> Self {
        let sr = sample_rate as usize;
        let zc = sr / 100;
        let block = ms_to(sr, zc, timing.block_ms.clamp(50.0, 1_000.0)).max(zc * 5);
        let sola = ms_to(sr, zc, timing.crossfade_ms.clamp(10.0, 150.0)).max(zc).min(block);
        let search = zc;
        let extra = ms_to(sr, zc, timing.context_ms.clamp(100.0, 5_000.0));
        let total = extra + sola + search + block;
        let frames = total / zc;
        let synth_sr = models.synth.sample_rate;
        let fade_in = (0..sola)
            .map(|i| (0.5 * std::f32::consts::PI * i as f32 / sola as f32).sin().powi(2))
            .collect();
        Self {
            zc,
            block,
            sola,
            search,
            skip_head: extra / zc,
            ret: (block + sola + search) / zc,
            input: vec![0.0; total],
            input16: vec![0.0; frames * HOP],
            to16: Resampler::new(sample_rate, f0::SR as u32),
            from_model: Resampler::new(synth_sr, sample_rate),
            cache_f0: vec![0.0; frames],
            sola_buffer: vec![0.0; sola],
            fade_in,
            mel: MelFrontEnd::new(),
            yin: Yin::new(),
            rng: Rng(0x9e37_79b9_7f4a_7c15),
            model_out: Vec::new(),
            infer: vec![0.0; block + sola + search],
            stats: Stats::default(),
            transpose: 0.0,
            models,
        }
    }

    pub fn block_len(&self) -> usize {
        self.block
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn models(&self) -> &Models {
        &self.models
    }

    pub fn into_models(self) -> Models {
        self.models
    }

    /// Converts one block of `block_len()` samples.
    pub fn process(&mut self, block: &[f32], out: &mut [f32]) -> Result<(), String> {
        let started = Instant::now();
        let n = self.block;
        debug_assert!(block.len() == n && out.len() == n);
        let total = self.input.len();
        self.input.copy_within(n.., 0);
        self.input[total - n..].copy_from_slice(block);

        let block16 = n / self.zc * HOP;
        let len16 = self.input16.len();
        self.input16.copy_within(block16.., 0);
        let src = &self.input[total - n - 2 * self.zc..];
        let mut res = vec![0.0; (n / self.zc + 2) * HOP];
        self.to16.process(src, &mut res);
        self.input16[len16 - block16 - HOP..].copy_from_slice(&res[HOP..]);

        let shift = block16 / HOP;
        self.cache_f0.copy_within(shift.., 0);
        let frames = self.cache_f0.len();
        self.cache_f0[frames - shift..].fill(0.0);

        let peak = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak < 1e-3 {
            for (i, o) in out.iter_mut().enumerate() {
                *o = if i < self.sola { self.sola_buffer[i] * (1.0 - self.fade_in[i]) } else { 0.0 };
            }
            self.sola_buffer.fill(0.0);
            self.finish(started);
            return Ok(());
        }

        let ms = |t: Instant| t.elapsed().as_secs_f32() * 1000.0;
        let t = Instant::now();
        if self.models.synth.f0 {
            self.track_pitch(block16)?;
        }
        let pitch_ms = ms(t);
        let t = Instant::now();
        let phone = self.embed()?;
        let embed_ms = ms(t);
        let t = Instant::now();
        self.synthesize(phone)?;
        let synth_ms = ms(t);
        self.stitch(out);
        self.stats.pitch_ms = smooth(self.stats.pitch_ms, pitch_ms);
        self.stats.embed_ms = smooth(self.stats.embed_ms, embed_ms);
        self.stats.synth_ms = smooth(self.stats.synth_ms, synth_ms);
        self.finish(started);
        Ok(())
    }

    /// Runs the networks a few times so GPU kernels are compiled before audio flows.
    pub fn warm_up(&mut self) -> Result<(), String> {
        let mut rng = Rng(7);
        let mut block = vec![0f32; self.block];
        let mut out = vec![0f32; self.block];
        for _ in 0..3 {
            rng.gauss(&mut block);
            block.iter_mut().for_each(|v| *v *= 0.05);
            self.process(&block, &mut out)?;
        }
        self.input.fill(0.0);
        self.input16.fill(0.0);
        self.cache_f0.fill(0.0);
        self.sola_buffer.fill(0.0);
        self.stats = Stats::default();
        Ok(())
    }

    fn finish(&mut self, started: Instant) {
        let sr = (self.zc * 100) as f32;
        let block_ms = self.block as f32 / sr * 1000.0;
        let infer_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.stats.infer_ms = smooth(self.stats.infer_ms, infer_ms);
        self.stats.block_ms = block_ms;
        self.stats.load = self.stats.infer_ms / block_ms;
        self.stats.latency_ms = block_ms + (self.sola + self.search) as f32 / sr * 1000.0 + self.stats.infer_ms;
    }

    fn track_pitch(&mut self, block16: usize) -> Result<(), String> {
        let mut want = block16 + 800;
        let rmvpe = matches!(self.models.pitch, Some(Pitch::Rmvpe(..)));
        if rmvpe {
            want = 5_120 * ((want - 1) / 5_120 + 1) - HOP;
        }
        let want = want.min(self.input16.len());
        let audio = &self.input16[self.input16.len() - want..];
        let f0 = match &mut self.models.pitch {
            Some(Pitch::Rmvpe(session, output)) => {
                let (mel, frames, padded) = self.mel.compute(audio);
                let t = Tensor::from_array(([1usize, 128, padded], mel)).map_err(|e| e.to_string())?;
                let outs = session.run(ort::inputs!["input" => t]).map_err(|e| e.to_string())?;
                let (_, sal) = outs[output.as_str()].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                if sal.len() < frames * RMVPE_BINS {
                    return Err("saída inesperada do RMVPE".into());
                }
                f0::rmvpe_decode(sal, frames, 0.03)
            }
            _ => self.yin.track(audio),
        };
        let k = 2f32.powf(self.transpose / 12.0);
        let fresh = f0.len().saturating_sub(4);
        let frames = self.cache_f0.len();
        let fresh = fresh.min(frames);
        for (dst, &v) in self.cache_f0[frames - fresh..].iter_mut().zip(&f0[3..3 + fresh]) {
            *dst = v * k;
        }
        Ok(())
    }

    fn embed(&mut self) -> Result<Vec<f32>, String> {
        let len = self.input16.len();
        let wav = Tensor::from_array(([1usize, len], self.input16.clone())).map_err(|e| e.to_string())?;
        let mask = Tensor::from_array(([1usize, len], vec![1i64; len])).map_err(|e| e.to_string())?;
        let e = &mut self.models.embedder;
        let outs = e.session.run(ort::inputs!["input_values" => wav, "attention_mask" => mask]).map_err(|e| e.to_string())?;
        let (shape, feats) = outs[e.output.as_str()].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        let width = *shape.last().unwrap_or(&768) as usize;
        let rows = feats.len() / width;

        let dim = e.proj.as_ref().map(|p| p.0).unwrap_or(width);
        let mut projected = vec![0f32; rows * dim];
        match &e.proj {
            Some((out, inp, w, b)) => {
                for r in 0..rows {
                    let x = &feats[r * width..r * width + inp];
                    for o in 0..*out {
                        let row = &w[o * inp..(o + 1) * inp];
                        projected[r * dim + o] = b[o] + row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
                    }
                }
            }
            None => projected.copy_from_slice(&feats[..rows * dim]),
        }

        // Repeat the last frame, then upsample 50 → 100 fps by repetition.
        let frames = self.cache_f0.len();
        let mut phone = vec![0f32; frames * dim];
        for t in 0..frames {
            let r = (t / 2).min(rows.saturating_sub(1));
            phone[t * dim..(t + 1) * dim].copy_from_slice(&projected[r * dim..(r + 1) * dim]);
        }
        Ok(phone)
    }

    fn synthesize(&mut self, phone: Vec<f32>) -> Result<(), String> {
        let frames = self.cache_f0.len();
        let s = &mut self.models.synth;
        let dim = phone.len() / frames;
        let flow_head = self.skip_head.saturating_sub(24);
        let dec_head = self.skip_head - flow_head;
        let upp = s.upp;

        let mut rnd = vec![0f32; 192 * frames];
        self.rng.gauss(&mut rnd);
        let mut noise = vec![0f32; self.ret * upp];
        self.rng.gauss(&mut noise);

        let err = |e: ort::Error| e.to_string();
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(9);
        for name in &s.inputs {
            let v: SessionInputValue = match name.as_str() {
                "phone" => Tensor::from_array(([1usize, frames, dim], phone.clone())).map_err(err)?.into(),
                "pitch" => Tensor::from_array(([1usize, frames], self.cache_f0.iter().map(|&f| f0::coarse(f)).collect::<Vec<i64>>()))
                    .map_err(err)?
                    .into(),
                "pitchf" => Tensor::from_array(([1usize, frames], self.cache_f0.clone())).map_err(err)?.into(),
                "g" => Tensor::from_array(([1usize, 256, 1], s.g.clone())).map_err(err)?.into(),
                "rnd" => Tensor::from_array(([1usize, 192, frames], std::mem::take(&mut rnd))).map_err(err)?.into(),
                "noise" => Tensor::from_array(([1usize, self.ret * upp, 1], std::mem::take(&mut noise))).map_err(err)?.into(),
                "flow_head" => Tensor::from_array(([1usize], vec![flow_head as i64])).map_err(err)?.into(),
                "dec_head" => Tensor::from_array(([1usize], vec![dec_head as i64])).map_err(err)?.into(),
                "return_length" => Tensor::from_array(([1usize], vec![self.ret as i64])).map_err(err)?.into(),
                other => return Err(format!("entrada desconhecida no modelo: {other}")),
            };
            inputs.push((Cow::Owned(name.clone()), v));
        }
        let outs = s.session.run(inputs).map_err(err)?;
        let (_, audio) = outs[s.output.as_str()].try_extract_tensor::<f32>().map_err(err)?;
        self.model_out.clear();
        self.model_out.extend_from_slice(audio);
        drop(outs);
        self.from_model.process(&self.model_out, &mut self.infer);
        Ok(())
    }

    fn stitch(&mut self, out: &mut [f32]) {
        let (sola, search) = (self.sola, self.search);
        let infer = &self.infer;
        let mut best = (0, f32::MIN);
        for off in 0..=search {
            let seg = &infer[off..off + sola];
            let num: f32 = seg.iter().zip(&self.sola_buffer).map(|(a, b)| a * b).sum();
            let den = (seg.iter().map(|v| v * v).sum::<f32>() + 1e-8).sqrt();
            let score = num / den;
            if score > best.1 {
                best = (off, score);
            }
        }
        let y = &infer[best.0..];
        for i in 0..self.block {
            out[i] = if i < sola { y[i] * self.fade_in[i] + self.sola_buffer[i] * (1.0 - self.fade_in[i]) } else { y[i] };
        }
        self.sola_buffer.copy_from_slice(&y[self.block..self.block + sola]);
    }
}
