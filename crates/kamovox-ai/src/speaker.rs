//! CAM++ / 3D-Speaker embedding for "only my voice" biometrics.
//!
//! Front-end matches sherpa-onnx `scripts/3dspeaker/test-onnx.py`:
//! Kaldi fbank @ 16 kHz, 80 mel bins, snip_edges, then per-bin global mean.
//! Live path feeds **speech-only** audio (VAD compaction) so silence never
//! pollutes CMVN / the embedding.

use std::path::Path;

use kaldi_fbank::{FbankOptions, OnlineFbank};
use ort::value::Tensor;

use crate::models::{session_builder, Device};

pub const SPEAKER_SR: u32 = 16_000;
pub const EMBED_DIM: usize = 512;
/// Preferred speech before first live score (~400 ms compacted).
pub const FIRST_SPEECH_MS: u32 = 400;
/// Absolute minimum for a short-utterance attempt ("Oi" / "Olá").
pub const SHORT_SPEECH_MS: u32 = 280;
/// Sliding window of compacted speech for live scores.
pub const LIVE_SPEECH_MS: u32 = 1_200;
/// How often to refresh the live cosine (ms).
pub const LIVE_PERIOD_MS: u32 = 80;
/// Keep frozen speech after silence so short phrases can still be scored.
pub const SILENCE_RESET_MS: u32 = 900;
/// Quiet frames shorter than this are holes inside the same utterance.
/// Freezing/clearing on every 10 ms dip was shredding live embeddings vs enroll.
pub const SPEECH_HANGOVER_MS: u32 = 220;

const MEL_BINS: usize = 80;
/// Minimum PCM for embedding (~220 ms @ 16 kHz) — short utterances.
const MIN_EMBED_SAMPLES: usize = SPEAKER_SR as usize * 22 / 100;

pub struct SpeakerEmbedder {
    session: ort::session::Session,
    input: String,
    output: String,
}

/// Enrollment centroid + slider calibration from short self-scores.
#[derive(Clone, Debug)]
pub struct EnrollResult {
    pub embed: Vec<f32>,
    /// Cosine at strict=0 (accept owner).
    pub thresh_lo: f32,
    /// Cosine at strict=1 (still passes own voice).
    pub thresh_hi: f32,
    /// Below this with sustained evidence → CONFIRMED_OTHER (not mere non-accept).
    pub thresh_reject: f32,
    pub self_p05: f32,
    pub self_p10: f32,
    pub self_median: f32,
}

impl SpeakerEmbedder {
    pub fn load(path: &Path) -> Result<Self, String> {
        // CPU only: ~28 MB CAM++ is fast and must not fight DirectML with RVC.
        let session = session_builder(Device::Cpu)?
            .commit_from_file(path)
            .map_err(|e| format!("biometria neural: {e}"))?;
        let input = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".into());
        let output = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "embedding".into());
        Ok(Self { session, input, output })
    }

    /// Embedding from mono PCM already at 16 kHz, samples in [-1, 1].
    /// Prefer speech-compacted input (no long silence).
    pub fn embed(&mut self, samples_16k: &[f32]) -> Result<Vec<f32>, String> {
        if samples_16k.len() < MIN_EMBED_SAMPLES {
            return Err("áudio curto demais para biometria neural".into());
        }
        let mut feats = fbank(samples_16k)?;
        global_mean(&mut feats);
        let t = feats.len() / MEL_BINS;
        if t < 8 {
            return Err("poucos frames de fbank".into());
        }
        let tensor = Tensor::from_array(([1usize, t, MEL_BINS], feats)).map_err(|e| e.to_string())?;
        let outs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .map_err(|e| e.to_string())?;
        let (_shape, data) = outs[self.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let mut emb = data.to_vec();
        if emb.len() > EMBED_DIM {
            emb.truncate(EMBED_DIM);
        }
        if emb.len() != EMBED_DIM {
            return Err(format!("embedding inesperado: {} dims", emb.len()));
        }
        l2_normalize(&mut emb);
        Ok(emb)
    }

    /// Centroid from speech-only chunks + leave-one-out slider calibration.
    ///
    /// Accept thresholds follow the **short-window** genuine distribution (what
    /// live CAM++ actually sees). Reject threshold is far below that — a score
    /// between reject and accept is UNCERTAIN, never CONFIRMED_OTHER.
    pub fn embed_enroll(&mut self, samples_16k: &[f32]) -> Result<EnrollResult, String> {
        let speech = compact_speech(samples_16k, SPEAKER_SR);
        if speech.len() < SPEAKER_SR as usize {
            return Err("pouca fala ativa na gravação para biometria neural".into());
        }

        let win = (SPEAKER_SR as usize * 3) / 2; // 1.5 s
        let hop = SPEAKER_SR as usize * 3 / 4; // 0.75 s
        let mut chunks = Vec::new();
        let mut start = 0;
        while start + win <= speech.len() {
            chunks.push(self.embed(&speech[start..start + win])?);
            start += hop;
        }
        if chunks.is_empty() {
            chunks.push(self.embed(&speech)?);
        }

        let mut acc = vec![0.0f32; EMBED_DIM];
        for emb in &chunks {
            for (a, b) in acc.iter_mut().zip(emb) {
                *a += *b;
            }
        }
        let inv = 1.0 / chunks.len() as f32;
        for a in &mut acc {
            *a *= inv;
        }
        l2_normalize(&mut acc);

        // Leave-one-out on long chunks (stable self scores).
        let mut long_scores = Vec::new();
        if chunks.len() >= 2 {
            for i in 0..chunks.len() {
                let mut cent = vec![0.0f32; EMBED_DIM];
                let mut n = 0u32;
                for (j, emb) in chunks.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    for (a, b) in cent.iter_mut().zip(emb) {
                        *a += *b;
                    }
                    n += 1;
                }
                let inv = 1.0 / n.max(1) as f32;
                for a in &mut cent {
                    *a *= inv;
                }
                l2_normalize(&mut cent);
                long_scores.push(cosine(&cent, &chunks[i]));
            }
        }

        // Live-like short probes dominate calibration (this is what FRR was hitting).
        let mut short_scores = Vec::new();
        for &ms in &[300u32, 500, 800, 1_000] {
            let probe = ((SPEAKER_SR as u64 * ms as u64) / 1000) as usize;
            let probe_hop = (probe / 2).max(SPEAKER_SR as usize / 10);
            let mut p = 0;
            while p + probe <= speech.len() {
                if let Ok(live) = self.embed(&speech[p..p + probe]) {
                    short_scores.push(cosine(&acc, &live));
                }
                p += probe_hop;
            }
        }

        let mut scores = short_scores;
        scores.extend(long_scores.iter().copied());
        if scores.is_empty() {
            scores.push(1.0);
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let self_p05 = percentile(&scores, 0.05);
        let self_p10 = percentile(&scores, 0.10);
        let self_median = percentile(&scores, 0.50);

        // NO artificial 0.28 floor — that forced FRR when short genuine windows
        // landed at 0.22–0.27 (exactly the bug in the user's video at 0%).
        let thresh_lo = (self_p05 - 0.04).clamp(0.15, 0.42);
        let thresh_hi = (self_p10 - 0.01).clamp(thresh_lo + 0.06, 0.55);
        // Confirmed-other only well below the genuine floor.
        let thresh_reject = (self_p05 - 0.18).clamp(0.08, thresh_lo - 0.06);

        Ok(EnrollResult {
            embed: acc,
            thresh_lo,
            thresh_hi,
            thresh_reject,
            self_p05,
            self_p10,
            self_median,
        })
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / d).clamp(-1.0, 1.0)
}

/// Map UI strict 0..1 onto enrolled cosine thresholds.
pub fn thresh_for_strict(strict: f32, lo: f32, hi: f32) -> f32 {
    let lo = if lo > 0.05 { lo } else { 0.18 };
    let hi = if hi > lo { hi.min(0.55) } else { (lo + 0.12).min(0.48) };
    lo + strict.clamp(0.0, 1.0) * (hi - lo)
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.40;
    }
    let i = ((sorted.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Keep only frames with speech energy (drops silence that ruins CMVN).
pub fn compact_speech(samples: &[f32], sr: u32) -> Vec<f32> {
    if samples.is_empty() || sr == 0 {
        return Vec::new();
    }
    let frame = (sr as usize / 100).max(80); // 10 ms
    let mut energies = Vec::with_capacity(samples.len() / frame + 1);
    for chunk in samples.chunks(frame) {
        let e = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32;
        energies.push(e);
    }
    let mut ranked = energies.clone();
    ranked.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Threshold above noise floor; keep upper ~55% of energetic frames when mixed.
    let floor = ranked.get(ranked.len() / 10).copied().unwrap_or(0.0);
    let mid = ranked.get(ranked.len() / 2).copied().unwrap_or(0.0);
    let thresh = (floor * 8.0).max(mid * 0.15).max(1e-6);

    let mut out = Vec::with_capacity(samples.len());
    for (i, chunk) in samples.chunks(frame).enumerate() {
        if energies[i] >= thresh {
            out.extend_from_slice(chunk);
        }
    }
    if out.len() < samples.len() / 8 {
        // Almost everything was "silence" — fall back to raw (better than empty).
        return samples.to_vec();
    }
    out
}

fn fbank(samples: &[f32]) -> Result<Vec<f32>, String> {
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.samp_freq = SPEAKER_SR as f32;
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = MEL_BINS as i32;
    let mut fbank = OnlineFbank::new(opts);
    fbank.accept_waveform(SPEAKER_SR as f32, samples);
    fbank.input_finished();
    let n = fbank.num_frames_ready();
    if n <= 0 {
        return Err("fbank sem frames".into());
    }
    let mut out = Vec::with_capacity(n as usize * MEL_BINS);
    for i in 0..n {
        let frame = fbank.get_frame(i);
        if frame.len() != MEL_BINS {
            return Err(format!("fbank dim {}", frame.len()));
        }
        out.extend_from_slice(frame);
    }
    Ok(out)
}

fn global_mean(feats: &mut [f32]) {
    let t = feats.len() / MEL_BINS;
    if t == 0 {
        return;
    }
    let mut mean = [0.0f32; MEL_BINS];
    for frame in feats.chunks_exact(MEL_BINS) {
        for (m, v) in mean.iter_mut().zip(frame) {
            *m += *v;
        }
    }
    let inv = 1.0 / t as f32;
    for m in &mut mean {
        *m *= inv;
    }
    for frame in feats.chunks_exact_mut(MEL_BINS) {
        for (v, m) in frame.iter_mut().zip(mean) {
            *v -= m;
        }
    }
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

/// Downsample (or upsample) mono PCM to 16 kHz with linear interpolation.
pub fn to_16k(samples: &[f32], sr: u32) -> Vec<f32> {
    if sr == 0 || samples.is_empty() {
        return Vec::new();
    }
    if sr == SPEAKER_SR {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * SPEAKER_SR as u64 / sr as u64) as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; out_len];
    let scale = sr as f64 / SPEAKER_SR as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 * scale;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = samples[j.min(samples.len() - 1)];
        let b = samples[(j + 1).min(samples.len() - 1)];
        *o = a + (b - a) * frac;
    }
    out
}

/// Streaming speech accumulator at an arbitrary sample rate.
///
/// On silence the buffer **freezes** (keeps last utterance) until
/// `SILENCE_RESET_MS` so short phrases can still be scored after they end.
pub struct SpeechBuffer {
    sr: u32,
    frame: usize,
    pending: Vec<f32>,
    speech: std::collections::VecDeque<f32>,
    silence_run: usize,
    energy_ema: f32,
    cap: usize,
    /// True after speech→silence until cleared — do not append more silence.
    frozen: bool,
    /// Bumped each time a new speech segment starts after a freeze/clear.
    pub utterance_id: u64,
}

impl SpeechBuffer {
    pub fn new(sr: u32) -> Self {
        let frame = (sr as usize / 100).max(80);
        let cap = ((sr as u64 * LIVE_SPEECH_MS as u64) / 1000) as usize + frame;
        Self {
            sr,
            frame,
            pending: Vec::with_capacity(frame * 2),
            speech: std::collections::VecDeque::with_capacity(cap + 8),
            silence_run: 0,
            energy_ema: 0.0,
            cap,
            frozen: false,
            utterance_id: 0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= self.frame {
            let frame: Vec<f32> = self.pending.drain(..self.frame).collect();
            let e = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            self.energy_ema = if self.energy_ema <= 0.0 {
                e
            } else {
                self.energy_ema * 0.95 + e * 0.05
            };
            let thresh = (self.energy_ema * 0.35).max(2e-6);
            let hangover = ((self.sr as u64 * SPEECH_HANGOVER_MS as u64) / 1000) as usize;
            let reset = ((self.sr as u64 * SILENCE_RESET_MS as u64) / 1000) as usize;
            if e >= thresh {
                if self.frozen {
                    // Real pause (≥ hangover) ended → new utterance.
                    self.speech.clear();
                    self.frozen = false;
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                } else if self.speech.is_empty() {
                    self.utterance_id = self.utterance_id.wrapping_add(1);
                }
                // Brief mid-phrase dips never freeze, so we keep appending here.
                self.silence_run = 0;
                self.speech.extend(frame);
                while self.speech.len() > self.cap {
                    self.speech.pop_front();
                }
            } else if !self.speech.is_empty() {
                self.silence_run = self.silence_run.saturating_add(frame.len());
                // Only freeze after sustained quiet — not on a single soft consonant.
                if !self.frozen && self.silence_run >= hangover {
                    self.frozen = true;
                }
                if self.silence_run >= reset {
                    self.clear();
                }
            }
        }
    }

    pub fn speech_len(&self) -> usize {
        self.speech.len()
    }

    pub fn speech_ms(&self) -> f32 {
        if self.sr == 0 {
            return 0.0;
        }
        self.speech.len() as f32 * 1000.0 / self.sr as f32
    }

    pub fn frozen(&self) -> bool {
        self.frozen
    }

    pub fn clear(&mut self) {
        self.speech.clear();
        self.pending.clear();
        self.silence_run = 0;
        self.frozen = false;
    }

    /// Copy up to `max` most recent speech samples.
    pub fn snapshot(&self, max: usize) -> Vec<f32> {
        let n = self.speech.len().min(max);
        let start = self.speech.len() - n;
        self.speech.iter().skip(start).copied().collect()
    }

    pub fn ready_first(&self) -> bool {
        let need = ((self.sr as u64 * FIRST_SPEECH_MS as u64) / 1000) as usize;
        self.speech.len() >= need
    }

    /// Enough for a short-utterance attempt (may be lower confidence).
    pub fn ready_short(&self) -> bool {
        let need = ((self.sr as u64 * SHORT_SPEECH_MS as u64) / 1000) as usize;
        self.speech.len() >= need
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![0.0, 1.0, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fbank_produces_frames() {
        let sr = SPEAKER_SR;
        let n = sr as usize;
        let mut wav = vec![0.0f32; n];
        for (i, s) in wav.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            *s = (std::f32::consts::TAU * 180.0 * t).sin() * 0.2;
        }
        let feats = fbank(&wav).expect("fbank");
        assert!(feats.len() >= 80 * 50);
        assert_eq!(feats.len() % 80, 0);
    }

    #[test]
    fn compact_drops_leading_silence() {
        let sr = SPEAKER_SR;
        let mut wav = vec![0.0f32; sr as usize];
        for (i, s) in wav[sr as usize / 2..].iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            *s = (std::f32::consts::TAU * 180.0 * t).sin() * 0.3;
        }
        let c = compact_speech(&wav, sr);
        assert!(c.len() < wav.len());
        assert!(c.len() > sr as usize / 4);
    }

    #[test]
    fn thresh_for_strict_is_monotonic() {
        let a = thresh_for_strict(0.0, 0.18, 0.42);
        let b = thresh_for_strict(0.5, 0.18, 0.42);
        let c = thresh_for_strict(1.0, 0.18, 0.42);
        assert!(a < b && b < c);
        assert!(c <= 0.48);
    }

    fn speech_like(sr: u32, hz: f32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // Harmonic stack ≈ voiced speech better than a pure sine for CAM++.
                let mut s = 0.0f32;
                for h in 1..=6 {
                    s += (std::f32::consts::TAU * hz * h as f32 * t).sin() / h as f32;
                }
                s * 0.12
            })
            .collect()
    }

    #[test]
    fn speech_buffer_freezes_short_utt_after_silence() {
        let sr = 48_000;
        let mut buf = SpeechBuffer::new(sr);
        let oi = speech_like(sr, 160.0, 0.28);
        buf.push(&oi);
        assert!(buf.ready_short(), "short oi should be scoreable");
        let utt = buf.utterance_id;
        let n = buf.speech_len();
        // Push silence — must freeze, not clear immediately.
        buf.push(&vec![0.0f32; sr as usize / 2]);
        assert!(buf.frozen());
        assert_eq!(buf.utterance_id, utt);
        assert_eq!(buf.speech_len(), n);
        assert!(buf.ready_short());
    }

    #[test]
    fn speech_buffer_new_utt_after_freeze_clears() {
        let sr = 48_000;
        let mut buf = SpeechBuffer::new(sr);
        buf.push(&speech_like(sr, 160.0, 0.3));
        let first = buf.utterance_id;
        // Must exceed SPEECH_HANGOVER_MS to freeze, then resume = new utt.
        buf.push(&vec![0.0f32; sr as usize * 3 / 10]);
        assert!(buf.frozen());
        buf.push(&speech_like(sr, 160.0, 0.3));
        assert!(!buf.frozen());
        assert_ne!(buf.utterance_id, first);
    }

    /// Root cause of live vs offline score gap: a soft mid-phrase dip must NOT
    /// wipe the speech buffer / bump utterance_id (that fed CAM++ tiny fragments).
    #[test]
    fn speech_buffer_survives_brief_amplitude_dip() {
        let sr = 48_000;
        let mut buf = SpeechBuffer::new(sr);
        buf.push(&speech_like(sr, 160.0, 0.6));
        let utt = buf.utterance_id;
        let before = buf.speech_len();
        // ~40 ms near-silence (typical soft consonant / breath) — below hangover.
        buf.push(&vec![1e-5f32; sr as usize * 40 / 1000]);
        assert!(!buf.frozen(), "brief dip must not freeze");
        buf.push(&speech_like(sr, 160.0, 0.5));
        assert_eq!(buf.utterance_id, utt, "must stay same utterance");
        assert!(
            buf.speech_len() > before,
            "must keep prior speech (had {before}, now {})",
            buf.speech_len()
        );
        assert!(buf.speech_ms() > 900.0, "window must stay long, got {} ms", buf.speech_ms());
    }

    /// Old VAD behaviour shreds continuous speech into short windows.
    #[test]
    fn speech_buffer_old_frag_vs_hangover_length() {
        let sr = 48_000u32;
        // Continuous speech with periodic 30 ms dips (every 200 ms).
        let mut pcm = Vec::new();
        for i in 0..8 {
            pcm.extend(speech_like(sr, 160.0, 0.17));
            if i < 7 {
                pcm.extend(std::iter::repeat(1e-5f32).take(sr as usize * 30 / 1000));
            }
        }
        let mut buf = SpeechBuffer::new(sr);
        buf.push(&pcm);
        assert!(
            buf.speech_ms() >= 1000.0,
            "hangover must keep ~1 s+ of speech across dips, got {} ms (utt={})",
            buf.speech_ms(),
            buf.utterance_id
        );
        assert_eq!(buf.utterance_id, 1);
    }

    /// Reproduces live FRR: old freeze-on-any-dip fed CAM++ short tails; hangover keeps window.
    #[test]
    #[ignore]
    fn campplus_fragmented_vad_vs_hangover_scores() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/speaker/campplus.onnx");
        assert!(path.is_file(), "missing {}", path.display());
        let mut e = SpeakerEmbedder::load(&path).expect("load");
        let sr = 48_000u32;
        let enroll_pcm = speech_like(sr, 145.0, 8.0);
        let enroll = e.embed_enroll(&to_16k(&enroll_pcm, sr)).expect("enroll");

        // Same continuous phrase with 30 ms dips every ~200 ms (natural soft spots).
        let mut pcm = Vec::new();
        for i in 0..10 {
            pcm.extend(speech_like(sr, 145.0, 0.17));
            if i < 9 {
                pcm.extend(std::iter::repeat(1e-5f32).take(sr as usize * 30 / 1000));
            }
        }

        let mut fixed = SpeechBuffer::new(sr);
        fixed.push(&pcm);
        let win_fixed = fixed.snapshot(((sr as u64 * LIVE_SPEECH_MS as u64) / 1000) as usize);
        let emb_fixed = e.embed(&to_16k(&win_fixed, sr)).expect("fixed");
        let score_fixed = cosine(&enroll.embed, &emb_fixed);

        // Emulate pre-fix VAD: freeze on first quiet frame, clear on resume.
        let frame = (sr as usize / 100).max(80);
        let mut old_speech: std::collections::VecDeque<f32> = std::collections::VecDeque::new();
        let mut pending = Vec::new();
        let mut frozen = false;
        let mut energy_ema = 0.0f32;
        let cap = ((sr as u64 * LIVE_SPEECH_MS as u64) / 1000) as usize + frame;
        pending.extend_from_slice(&pcm);
        while pending.len() >= frame {
            let fr: Vec<f32> = pending.drain(..frame).collect();
            let en = fr.iter().map(|x| x * x).sum::<f32>() / fr.len() as f32;
            energy_ema = if energy_ema <= 0.0 { en } else { energy_ema * 0.95 + en * 0.05 };
            let thresh = (energy_ema * 0.35).max(2e-6);
            if en >= thresh {
                if frozen || old_speech.is_empty() {
                    if frozen {
                        old_speech.clear();
                        frozen = false;
                    }
                }
                old_speech.extend(fr);
                while old_speech.len() > cap {
                    old_speech.pop_front();
                }
            } else if !old_speech.is_empty() {
                frozen = true;
            }
        }
        let win_old: Vec<f32> = old_speech.iter().copied().collect();
        let old_ms = win_old.len() as f32 * 1000.0 / sr as f32;
        let score_old = match e.embed(&to_16k(&win_old, sr)) {
            Ok(emb) => cosine(&enroll.embed, &emb),
            Err(err) => {
                println!("old_frag too short for CAM++ ({err}) — exact live FRR mechanism");
                f32::NAN
            }
        };

        println!(
            "hangover_ms={:.0} score={:.3} | old_frag_ms={:.0} score={:.3} | thresh_lo={:.3}",
            win_fixed.len() as f32 * 1000.0 / sr as f32,
            score_fixed,
            old_ms,
            score_old,
            enroll.thresh_lo
        );
        assert!(
            win_fixed.len() > win_old.len().saturating_mul(2).max(1) || score_old.is_nan()
                || score_fixed + 0.05 >= score_old,
            "hangover must keep longer/higher-quality window than old frag"
        );
        assert!(
            score_fixed >= enroll.thresh_lo - 0.05,
            "fixed live window should near-pass enroll lo ({} vs {})",
            score_fixed,
            enroll.thresh_lo
        );
        // Pre-fix path: either cannot embed or sits well below accept.
        assert!(
            score_old.is_nan() || score_old < enroll.thresh_lo,
            "old fragmentation should fail accept (score={score_old})"
        );
    }

    /// Needs `app/target/speaker/campplus.onnx`.
    #[test]
    #[ignore]
    fn campplus_enroll_and_live_windows() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/speaker/campplus.onnx");
        assert!(path.is_file(), "missing {}", path.display());
        let mut e = SpeakerEmbedder::load(&path).expect("load");

        let enroll = speech_like(SPEAKER_SR, 145.0, 8.0);
        let result = e.embed_enroll(&enroll).expect("enroll");
        println!(
            "p10={:.3} median={:.3} lo={:.3} hi={:.3}",
            result.self_p10, result.self_median, result.thresh_lo, result.thresh_hi
        );
        assert!(result.self_p10 > 0.20, "self_p10 {}", result.self_p10);
        assert!(result.thresh_hi <= 0.55);
        assert!(result.thresh_lo >= 0.15);
        assert!(result.thresh_reject < result.thresh_lo);
        assert!(result.thresh_lo < result.thresh_hi);

        // Live: silence then short utterance (speech-compacted).
        let mut live = vec![0.0f32; SPEAKER_SR as usize]; // 1 s silence
        live.extend(speech_like(SPEAKER_SR, 145.0, 1.2));
        let compacted = compact_speech(&live, SPEAKER_SR);
        let emb = e.embed(&compacted).expect("live");
        let score = cosine(&result.embed, &emb);
        println!("live compacted cosine={score:.3}");
        let need50 = thresh_for_strict(0.5, result.thresh_lo, result.thresh_hi);
        let need100 = thresh_for_strict(1.0, result.thresh_lo, result.thresh_hi);
        assert!(score >= need50, "score {score} < 50% thresh {need50}");
        assert!(score >= need100 - 0.05, "score {score} vs 100% {need100}");

        // Different spectral envelope (noise burst + high resonances), not just F0.
        let mut other = vec![0.0f32; SPEAKER_SR as usize * 12 / 10];
        let mut state = 0xC0FFEEu32;
        for (i, s) in other.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let t = i as f32 / SPEAKER_SR as f32;
            *s = noise * 0.2 + 0.08 * (std::f32::consts::TAU * 900.0 * t).sin()
                + 0.06 * (std::f32::consts::TAU * 2_400.0 * t).sin();
        }
        let other_e = e.embed(&other).expect("other");
        let foreign = cosine(&result.embed, &other_e);
        println!("foreign cosine={foreign:.3}");
        // Synthetic noise is still somewhat speaker-like to CAM++; real speech
        // separates more. Require a clear gap vs the live self window.
        assert!(
            foreign < score - 0.05,
            "foreign {foreign} should be below live {score}"
        );
        let _ = need50; // used above for self-pass asserts
    }

    /// Short phrases vs enrollment — needs campplus.onnx.
    #[test]
    #[ignore]
    fn campplus_short_utterances_vs_thresholds() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/speaker/campplus.onnx");
        assert!(path.is_file(), "missing {}", path.display());
        let mut e = SpeakerEmbedder::load(&path).expect("load");
        let enroll = speech_like(SPEAKER_SR, 145.0, 8.0);
        let result = e.embed_enroll(&enroll).expect("enroll");
        let shorts_ms = [220u32, 350, 500, 700, 900];
        let stricts = [0.0f32, 0.25, 0.5, 0.7, 0.9, 1.0];
        println!(
            "enroll lo={:.3} hi={:.3} median={:.3}",
            result.thresh_lo, result.thresh_hi, result.self_median
        );
        for &ms in &shorts_ms {
            let secs = ms as f32 / 1000.0;
            let wav = speech_like(SPEAKER_SR, 145.0, secs);
            let emb = match e.embed(&wav) {
                Ok(v) => v,
                Err(err) => {
                    println!("{ms}ms: embed err {err}");
                    continue;
                }
            };
            let score = cosine(&result.embed, &emb);
            print!("{ms}ms score={score:.3}");
            for &s in &stricts {
                let need = thresh_for_strict(s, result.thresh_lo, result.thresh_hi);
                let pass = score >= need;
                print!("  {:>3}%:{}", (s * 100.0) as u32, if pass { "OK" } else { "no" });
            }
            println!();
            // Absolute minimum: at least the 0% thresh on ≥350 ms synthetic speech.
            if ms >= 350 {
                let need0 = thresh_for_strict(0.0, result.thresh_lo, result.thresh_hi);
                assert!(
                    score >= need0 - 0.03,
                    "{ms}ms score {score} << lo {need0}"
                );
            }
        }
    }
}
