//! Time the CPU-side stages of one page: cargo run --release --example bench_stages -- PAGE.npy
use std::time::Instant;

use indra_ocr::imgproc::{Gray, Interp, resize_u8};
use indra_ocr::pipeline::{estimate_skew, orient_input};

fn load_npy(p: &str) -> Gray {
    let raw = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen]).unwrap();
    let s = &header[header.find("'shape': (").unwrap() + 10..];
    let dims: Vec<usize> =
        s[..s.find(')').unwrap()].split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    Gray::from_vec(dims[1], dims[0], raw[10 + hlen..].to_vec())
}

fn time<T>(name: &str, n: usize, mut f: impl FnMut() -> T) {
    f();
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(f());
    }
    println!("{name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0 / n as f64);
}

fn main() {
    let g = load_npy(&std::env::args().nth(1).expect("PAGE.npy"));
    println!("page {}x{}", g.w, g.h);
    time("estimate_skew", 5, || estimate_skew(&g));
    time("orient_input", 5, || orient_input(&g, 768));
    time("resize area 800", 5, || resize_u8(&g, 800 * g.w / g.h, 800, Interp::Area));
    time("rotate_page", 5, || indra_ocr::pipeline::rotate_page(&g, 1.3));
    let (x, _) = indra_ocr::det_post::page_to_input(&g, 1024);
    time("page_to_input 1024", 5, || indra_ocr::det_post::page_to_input(&g, 1024));
    let dir = indra_ocr::models::model_dir(None, None, false).unwrap();
    let mut nb = indra_ocr::native_backend::NativeBackend::new(&dir, 1, false).unwrap();
    use indra_ocr::backend::Backend;
    let out = nb.det(&x).unwrap();
    time("det net", 5, || nb.det(&x).unwrap());
    time("det decode", 5, || indra_ocr::det_post::decode(&out.data, out.h, out.w, 0.5, 2));
    let boxes = indra_ocr::det_post::decode(&out.data, out.h, out.w, 0.5, 2);
    let mh = indra_ocr::py::median_f32(&boxes.iter().map(|b| b.y1 - b.y0).collect::<Vec<_>>());
    println!("{} boxes, median height {mh:.1} (second pass below 9)", boxes.len());
}
