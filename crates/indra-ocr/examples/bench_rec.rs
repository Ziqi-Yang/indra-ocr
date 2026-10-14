//! The recognize stage of one page, step by step:
//!   INDRA_ENGINE_PROFILE=1 cargo run --release --example bench_rec -- PAGE.npy
use std::time::Instant;

use indra_ocr::backend::{Backend, RecModel};
use indra_ocr::imgproc::{Gray, ImageF};
use indra_ocr::pipeline::{Ocr, OcrOptions};

fn load_npy(p: &str) -> Gray {
    let raw = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen]).unwrap();
    let s = &header[header.find("'shape': (").unwrap() + 10..];
    let dims: Vec<usize> =
        s[..s.find(')').unwrap()].split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    Gray::from_vec(dims[1], dims[0], raw[10 + hlen..].to_vec())
}

fn main() {
    let g = load_npy(&std::env::args().nth(1).expect("PAGE.npy"));
    let dir = indra_ocr::models::model_dir(None, None, false).unwrap();
    let mut ocr = Ocr::with_ort(&dir, 1, OcrOptions::default(), false).unwrap();
    let mut native = indra_ocr::native_backend::NativeBackend::new(&dir, 1, false).unwrap();
    let mut ort = indra_ocr::ort_backend::OrtBackend::new(&dir, 1, false).unwrap();
    let (up, _, _) = ocr.upright(&g, true).unwrap();
    let boxes: Vec<[f32; 4]> = ocr.detect(&up).unwrap().iter().map(|b| b.xyxy()).collect();
    let reps = 5;
    let t0 = Instant::now();
    let mut crops = vec![];
    for _ in 0..reps {
        crops = ocr.crop_lines(&up, &boxes, 32).unwrap();
    }
    println!("{} lines, {} columns", crops.len(), crops.iter().map(|c| c.w).sum::<usize>());
    println!("crops: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    let refs: Vec<&ImageF> = crops.iter().collect();
    let t0 = Instant::now();
    for _ in 0..reps {
        ort.script(&refs).unwrap();
    }
    println!("script (ort): {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    for (name, b) in [("ort", &mut ort as &mut dyn Backend), ("native", &mut native as &mut dyn Backend)] {
        b.rec(RecModel::Main, &refs).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            b.rec(RecModel::Main, &refs).unwrap();
        }
        println!("rec {name}: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    }
    print!("{}", indra_engine::profile::report());
}
