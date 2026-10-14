//! The detector and orientation models, native vs ONNX Runtime:
//!   INDRA_ENGINE_PROFILE=1 cargo run --release --example bench_det -- PAGE.npy
use std::time::Instant;

use indra_ocr::backend::Backend;
use indra_ocr::imgproc::Gray;

fn load_npy(p: &str) -> Gray {
    let raw = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen]).unwrap();
    let s = &header[header.find("'shape': (").unwrap() + 10..];
    let dims: Vec<usize> = s[..s.find(')').unwrap()].split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    Gray::from_vec(dims[1], dims[0], raw[10 + hlen..].to_vec())
}

fn main() {
    let g = load_npy(&std::env::args().nth(1).expect("PAGE.npy"));
    let dir = indra_ocr::models::model_dir(None, None, false).unwrap();
    let mut native = indra_ocr::native_backend::NativeBackend::new(&dir, 1, false).unwrap();
    let mut ort = indra_ocr::ort_backend::OrtBackend::new(&dir, 1, false).unwrap();
    let (x, _) = indra_ocr::det_post::page_to_input(&g, 1024);
    let oi = indra_ocr::pipeline::orient_input(&g, 768);
    println!("det input {}x{}", x.w, x.h);
    let reps = 5;
    for (name, b) in [("ort", &mut ort as &mut dyn Backend), ("native", &mut native as &mut dyn Backend)] {
        b.det(&x).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            b.det(&x).unwrap();
        }
        let det = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            b.orient(&oi).unwrap();
        }
        println!("{name}: det {det:.1} ms, orient {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    }
    print!("{}", indra_engine::profile::report());
}
