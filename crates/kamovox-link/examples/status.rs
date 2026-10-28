//! `cargo run -p kamovox-link --example status`: live state of the APO link.

use std::sync::atomic::Ordering;

use kamovox_link::{now_ms, Link};

fn main() {
    let Some(link) = Link::open() else {
        println!("Link não existe: o APO ainda não rodou desde que o áudio reiniciou.");
        return;
    };
    let h = link.header();
    let count: usize = std::env::args().nth(1).and_then(|n| n.parse().ok()).unwrap_or(5);
    for _ in 0..count {
        let now = now_ms();
        let voice = link.voice_position();
        let mic = link.mic_position();
        let mut peak = 0f32;
        let mut mic_peak = 0f32;
        for i in 0..480 {
            peak = peak.max(link.voice_at(voice.saturating_sub(i + 1)).abs());
            mic_peak = mic_peak.max(link.mic_at(mic.saturating_sub(i + 1)).abs());
        }
        println!(
            "APO há {} ms ({} Hz, {} canais, período {}), app há {} ms ativo={} | mic={} voz={} pico_mic={:.4} pico_voz={:.4} falhas={} folga={} bloco={}",
            now.saturating_sub(h.apo_beat.load(Ordering::Relaxed)),
            h.sample_rate.load(Ordering::Relaxed),
            h.channels.load(Ordering::Relaxed),
            h.period.load(Ordering::Relaxed),
            now.saturating_sub(h.app_beat.load(Ordering::Relaxed)),
            h.app_active.load(Ordering::Relaxed),
            mic,
            voice,
            mic_peak,
            peak,
            h.underruns.load(Ordering::Relaxed),
            h.cushion.load(Ordering::Relaxed),
            h.burst.load(Ordering::Relaxed),
        );
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}
