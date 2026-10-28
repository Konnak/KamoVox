fn main() {
    use kamovox_dsp::{Params, Processor};
    let sr = 48_000u32;
    let input: Vec<f32> = (0..sr as usize)
        .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / sr as f32).sin() * 0.5)
        .collect();
    let rms = |b: &[f32]| (b.iter().map(|v| v * v).sum::<f32>() / b.len().max(1) as f32).sqrt();
    for (name, p) in [
        ("pitch2", Params { pitch: 2.0, ..Params::default() }),
        ("vib1", Params { vibrato: 1.0, ..Params::default() }),
        ("plain", Params::default()),
    ] {
        let mut proc = Processor::new(sr);
        proc.set_params(p);
        let mut buf = input.clone();
        for chunk in buf.chunks_mut(480) {
            proc.process(chunk);
        }
        println!("{name}: gain={}", rms(&buf[8000..]) / rms(&input[8000..]));
    }
}
