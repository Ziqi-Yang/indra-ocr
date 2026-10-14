//! The order model on a page's lines, native vs ONNX Runtime.
use std::time::Instant;

use indra_ocr::backend::Backend;

fn main() {
    let dir = indra_ocr::models::model_dir(None, None, false).unwrap();
    let mut native = indra_ocr::native_backend::NativeBackend::new(&dir, 1, false).unwrap();
    let mut ort = indra_ocr::ort_backend::OrtBackend::new(&dir, 1, false).unwrap();
    for n in [30usize, 100, 300] {
        let boxes: Vec<[f32; 4]> = (0..n)
            .map(|i| [10.0 + (i % 2) as f32 * 400.0, 20.0 * (i / 2) as f32, 380.0 + (i % 2) as f32 * 400.0, 20.0 * (i / 2) as f32 + 14.0])
            .collect();
        let f = indra_ocr::order_post::line_features(&boxes, (800, 1100), None);
        for (name, b) in [("ort", &mut ort as &mut dyn Backend), ("native", &mut native as &mut dyn Backend)] {
            b.order(&f, n).unwrap();
            let t0 = Instant::now();
            for _ in 0..20 {
                b.order(&f, n).unwrap();
            }
            println!("n={n} {name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0 / 20.0);
        }
    }
    print!("{}", indra_engine::profile::report());
}
