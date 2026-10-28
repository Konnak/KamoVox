//! Isolate CAM++ from the Speaker Gate.
//!
//! ```text
//! cargo run -p kamovox-ai --example speaker_diagnostic --release -- \
//!   --model path/to/campplus.onnx \
//!   --wav path/to/owner.wav
//! ```
//!
//! Enrolls on the first half of speech-compacted audio, scores the second half
//! with live-like windows. Reports FRR at 0/50/100% with the *new* calibration
//! (and the old 0.28-floor policy for comparison).

use std::path::PathBuf;

use kamovox_ai::{
    compact_speech, cosine, thresh_for_strict, to_16k, SpeakerEmbedder, EMBED_DIM, SPEAKER_SR,
};

fn main() {
    let mut model = PathBuf::from("target/speaker/campplus.onnx");
    let mut wav: Option<PathBuf> = None;
    let mut sr_hint = 48_000u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(args.next().expect("--model path")),
            "--wav" => wav = Some(PathBuf::from(args.next().expect("--wav path"))),
            "--sr" => sr_hint = args.next().expect("--sr").parse().expect("sr"),
            "--help" | "-h" => {
                eprintln!(
                    "speaker_diagnostic --model campplus.onnx --wav owner.wav [--sr 48000]\n\
                     Without --wav: synthetic self-split stress of the calibrator."
                );
                return;
            }
            other => panic!("unknown arg {other}"),
        }
    }

    assert!(model.is_file(), "missing model {}", model.display());
    let model_bytes = std::fs::read(&model).expect("read model");
    let model_sha = {
        // lightweight fingerprint without extra deps
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &model_bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    };

    let mut embedder = SpeakerEmbedder::load(&model).expect("load CAM++");

    let (pcm, sr) = match &wav {
        Some(p) => {
            let (s, r) = read_wav_mono(p).unwrap_or_else(|e| panic!("wav {}: {e}", p.display()));
            (s, r)
        }
        None => {
            eprintln!("NOTE: no --wav; using harmonic synthetic speech for calibrator stress only.");
            (synth_speech(SPEAKER_SR, 145.0, 12.0), SPEAKER_SR)
        }
    };

    let wav16 = to_16k(&pcm, sr);
    let speech = compact_speech(&wav16, SPEAKER_SR);
    println!("MODEL: {}", model.display());
    println!("MODEL_FNV: {model_sha}");
    println!("EMBEDDING_DIMENSION: {EMBED_DIM}");
    println!("TEST_SAMPLE_RATE_IN: {sr}");
    println!("SPEECH_COMPACTED_S: {:.2}", speech.len() as f32 / SPEAKER_SR as f32);
    if speech.len() < SPEAKER_SR as usize * 2 {
        panic!("need ≥2 s compacted speech");
    }

    let mid = speech.len() / 2;
    let enroll_pcm = &speech[..mid];
    let test_pcm = &speech[mid..];

    let enroll = embedder.embed_enroll(enroll_pcm).expect("enroll");
    println!();
    println!(
        "ENROLL thresh_lo={:.3} thresh_hi={:.3} thresh_reject={:.3}",
        enroll.thresh_lo, enroll.thresh_hi, enroll.thresh_reject
    );
    println!(
        "ENROLL self_p05={:.3} self_p10={:.3} self_median={:.3}",
        enroll.self_p05, enroll.self_p10, enroll.self_median
    );

    // Old (buggy) policy for comparison.
    let old_lo = (enroll.self_p10 - 0.10).clamp(0.28, 0.40);
    let old_hi = (enroll.self_p10 - 0.02).clamp(old_lo + 0.08, 0.55);

    let windows_ms = [300u32, 500, 800, 1_000, 1_200];
    let mut all_scores = Vec::new();
    println!();
    for &ms in &windows_ms {
        let win = ((SPEAKER_SR as u64 * ms as u64) / 1000) as usize;
        let hop = win / 2;
        let mut i = 0usize;
        let mut n = 0u32;
        while i + win <= test_pcm.len() {
            let emb = embedder.embed(&test_pcm[i..i + win]).expect("embed");
            let score = cosine(&enroll.embed, &emb);
            n += 1;
            all_scores.push((ms, score));
            println!("Window {n:03} | Duration: {ms}ms | Cosine: {score:.3}");
            i += hop;
        }
    }

    let mut scores: Vec<f32> = all_scores.iter().map(|(_, s)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = percentile(&scores, 0.50);
    let p05 = percentile(&scores, 0.05);
    let p10 = percentile(&scores, 0.10);
    let p90 = percentile(&scores, 0.90);
    let min = scores.first().copied().unwrap_or(0.0);
    let max = scores.last().copied().unwrap_or(0.0);
    println!();
    println!("Genuine median: {median:.3}");
    println!("Genuine p05: {p05:.3}");
    println!("Genuine p10: {p10:.3}");
    println!("Genuine p90: {p90:.3}");
    println!("Genuine minimum: {min:.3}");
    println!("Genuine maximum: {max:.3}");

    println!();
    println!("=== NEW calibration (three-zone) ===");
    for &pct in &[0.0f32, 0.5, 1.0] {
        let need = thresh_for_strict(pct, enroll.thresh_lo, enroll.thresh_hi);
        let frr = scores.iter().filter(|&&s| s < need).count();
        let other = scores
            .iter()
            .filter(|&&s| s < enroll.thresh_reject)
            .count();
        let uncertain = scores
            .iter()
            .filter(|&&s| s >= enroll.thresh_reject && s < need)
            .count();
        println!(
            "{:>3}% need={need:.3} reject<{:.3} | FRR(accept)={frr}/{} | would_OTHER={other} | UNCERTAIN={uncertain}",
            (pct * 100.0) as u32,
            enroll.thresh_reject,
            scores.len()
        );
    }

    println!();
    println!("=== OLD calibration (0.28 floor, binary reject≈need-0.12) ===");
    for &pct in &[0.0f32, 0.5, 1.0] {
        let need = thresh_for_strict(pct, old_lo, old_hi);
        let foreign = (need - 0.12).max(old_lo * 0.55).max(0.12);
        let frr = scores.iter().filter(|&&s| s < need).count();
        let other = scores.iter().filter(|&&s| s < foreign).count();
        println!(
            "{:>3}% need={need:.3} foreign<{foreign:.3} | FRR={frr}/{} | would_OTHER={other}",
            (pct * 100.0) as u32,
            scores.len()
        );
    }

    let _ = sr_hint;
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn synth_speech(sr: u32, hz: f32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let mut s = 0.0f32;
            for h in 1..=6 {
                s += (std::f32::consts::TAU * hz * h as f32 * t).sin() / h as f32;
            }
            // Amplitude modulation ≈ syllables.
            let env = 0.55 + 0.45 * (std::f32::consts::TAU * 3.5 * t).sin().abs();
            s * 0.12 * env
        })
        .collect()
}

/// Minimal PCM16/float WAV reader (mono or downmix).
fn read_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let mut fmt = None;
    let mut data = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let start = pos + 8;
        let end = (start + size).min(bytes.len());
        if id == b"fmt " && size >= 16 {
            let audio_format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
            let channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
            fmt = Some((audio_format, channels, sample_rate, bits));
        } else if id == b"data" {
            data = Some(&bytes[start..end]);
        }
        pos = start + size + (size % 2);
    }
    let (audio_format, channels, sample_rate, bits) = fmt.ok_or("missing fmt")?;
    let data = data.ok_or("missing data")?;
    let ch = channels.max(1) as usize;
    let mut mono = Vec::new();
    match (audio_format, bits) {
        (1, 16) => {
            for frame in data.chunks_exact(2 * ch) {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    let s = i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]);
                    acc += s as f32 / 32768.0;
                }
                mono.push(acc / ch as f32);
            }
        }
        (3, 32) => {
            for frame in data.chunks_exact(4 * ch) {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    let s = f32::from_le_bytes(frame[4 * c..4 * c + 4].try_into().unwrap());
                    acc += s;
                }
                mono.push(acc / ch as f32);
            }
        }
        _ => return Err(format!("unsupported wav format={audio_format} bits={bits}")),
    }
    Ok((mono, sample_rate))
}
