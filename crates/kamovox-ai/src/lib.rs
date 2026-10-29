//! Real-time AI voice conversion (RVC) on ONNX Runtime with DirectML or CPU.
//! Voices are plain RVC `.pth` files injected into bundled weightless templates.

pub mod f0;
mod models;
mod onnx_patch;
mod pth;
mod realtime;
pub mod resample;
pub mod speaker;

pub use models::{inspect, Device, LoadKey, Models, PitchMethod, VoiceInfo, EMBEDDER, RMVPE};
pub use realtime::{Realtime, Stats, Timing};
pub use speaker::{
    cosine, flatten_gallery, max_cosine, max_cosine_flat, thresh_for_strict, to_16k, compact_speech,
    EnrollResult, SpeakerEmbedder, SpeechBuffer, EMBED_DIM, FIRST_SPEECH_MS, LIVE_PERIOD_MS,
    LIVE_SCORE_MS, LIVE_SPEECH_MS, MAX_GALLERY, MAX_PROTOTYPES, SHORT_SPEECH_MS, SILENCE_RESET_MS,
    SPEAKER_SR, SPEECH_HANGOVER_MS,
};

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    fn voice() -> Option<PathBuf> {
        let p = PathBuf::from(r"D:\DevTools\rvc-export\test\Shigure.pth");
        p.exists().then_some(p)
    }

    #[test]
    fn reads_rvc_checkpoint() {
        let Some(v) = voice() else { return };
        let info = inspect(&v).unwrap();
        assert_eq!((info.version.as_str(), info.sample_rate, info.f0), ("v1", 40_000, true));
    }

    /// Full pipeline on real models; run with `cargo test -p kamovox-ai -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn converts_in_real_time() {
        let Some(v) = voice() else { return };
        let base = Path::new(r"D:\KamoVox\Modelos\base");
        let res = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src-tauri/resources/rvc");
        let fast = Timing { block_ms: 160.0, context_ms: 300.0, crossfade_ms: 40.0 };
        let cpu = Timing { block_ms: 400.0, context_ms: 300.0, crossfade_ms: 40.0 };
        let quality = Timing { block_ms: 400.0, context_ms: 1_200.0, crossfade_ms: 100.0 };
        let cases = [
            (Device::Gpu, PitchMethod::Rmvpe, fast),
            (Device::Gpu, PitchMethod::Rmvpe, quality),
            (Device::Cpu, PitchMethod::Yin, cpu),
        ];
        for (device, pitch, timing) in cases {
            let key = LoadKey { voice: v.clone(), speaker: 0, device, pitch };
            let t = std::time::Instant::now();
            let models = Models::load(base, &res, key).unwrap();
            println!("{device:?} (ran on {:?}) {pitch:?}: loaded in {:?}", models.device, t.elapsed());
            let mut rt = Realtime::new(models, 48_000, timing);
            rt.warm_up().unwrap();
            let n = rt.block_len();
            let sr = 48_000.0;
            let mut out = vec![0.0; n];
            let mut energy = 0.0;
            for b in 0..16 {
                let block: Vec<f32> = (0..n)
                    .map(|i| {
                        let t = (b * n + i) as f32 / sr;
                        let f = 160.0 + 30.0 * (t * 2.0).sin();
                        (1..10).map(|h| (std::f32::consts::TAU * f * h as f32 * t).sin() / h as f32).sum::<f32>() * 0.15
                    })
                    .collect();
                rt.process(&block, &mut out).unwrap();
                assert!(out.iter().all(|v| v.is_finite()));
                if b > 4 {
                    energy += out.iter().map(|v| v * v).sum::<f32>();
                }
            }
            println!("  {:?}", rt.stats());
            assert!(energy > 1.0, "silent output");
        }
    }
}
