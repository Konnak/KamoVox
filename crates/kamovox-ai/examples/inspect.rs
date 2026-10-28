//! `cargo run -p kamovox-ai --example inspect -- <model.pth | pack.zip> [entry]`

use std::io::copy;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("path"));
    let path = match args.next() {
        Some(entry) => {
            let mut zip = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
            let mut file = zip.by_index(entry.parse().unwrap()).unwrap();
            let out = std::env::temp_dir().join("kamovox-inspect.pth");
            copy(&mut file, &mut std::fs::File::create(&out).unwrap()).unwrap();
            out
        }
        None => path,
    };
    match kamovox_ai::inspect(&path) {
        Ok(info) => println!("{} {} Hz f0={} speakers={}", info.version, info.sample_rate, info.f0, info.speakers),
        Err(e) => println!("ERRO: {e}"),
    }
}
