//! Decodes a short clip into mono f32 at the engine sample rate.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const MAX_SECONDS: u32 = 180;

pub fn decode(path: &Path, target_rate: u32) -> Result<Vec<f32>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let file = File::open(path).map_err(|e| format!("Não foi possível abrir o áudio: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if !ext.is_empty() {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|_| "Esse arquivo não é um áudio reconhecido. Use MP3, WAV, OGG, FLAC ou M4A.".to_string())?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("Esse arquivo não tem faixa de áudio.")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.unwrap_or(target_rate).max(8_000);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Não foi possível ler o áudio: {e}"))?;

    let mut mono = Vec::<f32>::new();
    let max = target_rate as usize * MAX_SECONDS as usize;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(Error::DecodeError(_)) => continue,
            Err(_) => break,
        };
        append_mono(&decoded, &mut mono);
        if mono.len() > max {
            return Err("O áudio passa de 3 minutos. Use um trecho mais curto.".into());
        }
    }
    if mono.is_empty() {
        return Err("Não encontrei áudio nesse arquivo.".into());
    }
    Ok(resample(&mono, rate, target_rate.max(8_000)))
}

fn append_mono(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => mix(b, out, |s| *s),
        AudioBufferRef::S16(b) => mix(b, out, |s| *s as f32 / 32768.0),
        AudioBufferRef::S32(b) => mix(b, out, |s| *s as f32 / 2_147_483_648.0),
        AudioBufferRef::U8(b) => mix(b, out, |s| (*s as f32 - 128.0) / 128.0),
        AudioBufferRef::F64(b) => mix(b, out, |s| *s as f32),
        _ => {}
    }
}

fn mix<T: symphonia::core::sample::Sample>(buf: &symphonia::core::audio::AudioBuffer<T>, out: &mut Vec<f32>, map: impl Fn(&T) -> f32) {
    let ch = buf.spec().channels.count().max(1);
    let frames = buf.frames();
    for i in 0..frames {
        let mut s = 0.0;
        for c in 0..ch {
            s += map(&buf.chan(c)[i]);
        }
        out.push(s / ch as f32);
    }
}

fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let step = from as f64 / to as f64;
    let n = ((input.len() as f64) / step) as usize;
    let mut out = Vec::with_capacity(n);
    let mut pos = 0.0;
    while (pos as usize) + 1 < input.len() {
        let i = pos as usize;
        let f = (pos - i as f64) as f32;
        out.push(input[i] * (1.0 - f) + input[i + 1] * f);
        pos += step;
    }
    out
}
