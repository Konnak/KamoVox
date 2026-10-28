//! Prove the live SpeechBuffer VAD gap vs contiguous offline windows.
//!
//! ```text
//! cargo run -p kamovox-ai --example live_vs_offline --release -- \
//!   --model target/speaker/campplus.onnx [--wav owner.wav]
//! ```

use std::path::PathBuf;

use kamovox_ai::{
    compact_speech, cosine, to_16k, SpeakerEmbedder, SpeechBuffer, LIVE_SPEECH_MS, SPEAKER_SR,
};

fn main() {
    let mut model = PathBuf::from("target/speaker/campplus.onnx");
    let mut wav: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(args.next().expect("--model")),
            "--wav" => wav = Some(PathBuf::from(args.next().expect("--wav"))),
            other => panic!("unknown {other}"),
        }
    }
    assert!(model.is_file(), "missing {}", model.display());
    let mut emb = SpeakerEmbedder::load(&model).expect("load");

    let (pcm, sr) = match wav {
        Some(p) => read_wav_mono(&p).unwrap_or_else(|e| panic!("{e}")),
        None => {
            eprintln!("NOTE: synthetic speech with 30 ms dips (reproduces live FRR mechanism).");
            let sr = 48_000u32;
            let mut pcm = Vec::new();
            for i in 0..12 {
                pcm.extend(synth(sr, 145.0, 0.17));
                if i < 11 {
                    pcm.extend(std::iter::repeat(1e-5f32).take(sr as usize * 30 / 1000));
                }
            }
            // Prepend a clean enroll segment without dips.
            let mut full = synth(sr, 145.0, 6.0);
            full.extend(pcm);
            (full, sr)
        }
    };

    let wav16 = to_16k(&pcm, sr);
    let speech = compact_speech(&wav16, SPEAKER_SR);
    let mid = speech.len() / 2;
    let enroll = emb.embed_enroll(&speech[..mid]).expect("enroll");
    let test = &speech[mid..];

    println!("ENROLL lo={:.3} reject={:.3} p05={:.3} median={:.3}", enroll.thresh_lo, enroll.thresh_reject, enroll.self_p05, enroll.self_median);

    // Offline contiguous 800 ms windows on compacted speech.
    let win = SPEAKER_SR as usize * 800 / 1000;
    let mut offline = Vec::new();
    let mut i = 0;
    while i + win <= test.len() {
        let e = emb.embed(&test[i..i + win]).expect("off");
        offline.push(cosine(&enroll.embed, &e));
        i += win / 2;
    }

    // Live path: push original-rate audio (with dips) through SpeechBuffer.
    let mut buf = SpeechBuffer::new(sr);
    // Rebuild a dipped test at original rate from second half of raw pcm.
    let raw_mid = pcm.len() / 2;
    buf.push(&pcm[raw_mid..]);
    let snap = buf.snapshot(((sr as u64 * LIVE_SPEECH_MS as u64) / 1000) as usize);
    let live16 = to_16k(&snap, sr);
    let live_score = if live16.len() >= SPEAKER_SR as usize * 22 / 100 {
        let e = emb.embed(&live16).expect("live");
        cosine(&enroll.embed, &e)
    } else {
        f32::NAN
    };

    offline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = offline[offline.len() / 2];
    let p10 = offline[offline.len() / 10];
    println!("OFFLINE contiguous windows: n={} median={:.3} p10={:.3} min={:.3}", offline.len(), med, p10, offline[0]);
    println!(
        "LIVE SpeechBuffer snapshot: ms={:.0} utt={} cosine={:.3}",
        buf.speech_ms(),
        buf.utterance_id,
        live_score
    );
    println!(
        "DELTA (offline_median - live) = {:.3}",
        med - if live_score.is_finite() { live_score } else { 0.0 }
    );
    if live_score.is_finite() && live_score >= enroll.thresh_lo {
        println!("RESULT: live window PASSES thresh_lo (hangover OK)");
    } else if live_score.is_finite() {
        println!("RESULT: live window BELOW thresh_lo — inspect buffer duration");
    }
}

fn synth(sr: u32, hz: f32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let mut s = 0.0f32;
            for h in 1..=6 {
                s += (std::f32::consts::TAU * hz * h as f32 * t).sin() / h as f32;
            }
            s * 0.12
        })
        .collect()
}

fn read_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" {
        return Err("not wav".into());
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    let mut data_off = 12usize;
    while data_off + 8 < bytes.len() {
        let id = &bytes[data_off..data_off + 4];
        let sz = u32::from_le_bytes([
            bytes[data_off + 4],
            bytes[data_off + 5],
            bytes[data_off + 6],
            bytes[data_off + 7],
        ]) as usize;
        if id == b"data" {
            data_off += 8;
            let slice = &bytes[data_off..data_off + sz.min(bytes.len() - data_off)];
            let mut mono = Vec::new();
            if bits == 16 {
                for frame in slice.chunks_exact(2 * channels) {
                    let mut acc = 0.0f32;
                    for c in 0..channels {
                        let i = c * 2;
                        let s = i16::from_le_bytes([frame[i], frame[i + 1]]);
                        acc += s as f32 / 32768.0;
                    }
                    mono.push(acc / channels as f32);
                }
            } else if bits == 32 {
                for frame in slice.chunks_exact(4 * channels) {
                    let mut acc = 0.0f32;
                    for c in 0..channels {
                        let i = c * 4;
                        let s = f32::from_le_bytes([frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]);
                        acc += s;
                    }
                    mono.push(acc / channels as f32);
                }
            } else {
                return Err(format!("bits {bits}"));
            }
            return Ok((mono, sr));
        }
        data_off += 8 + sz + (sz % 2);
    }
    Err("no data chunk".into())
}
