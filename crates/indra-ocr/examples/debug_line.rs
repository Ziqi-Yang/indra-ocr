//! Both engines on one line box of a PDF page:
//!   cargo run --release --example debug_line -- FILE.pdf PAGE x0 y0 x1 y1
use indra_ocr::backend::{Backend, MathBackend};
use indra_ocr::imgproc::ImageF;
use indra_ocr::pipeline::{Ocr, OcrOptions};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, page) = (std::path::PathBuf::from(&a[1]), a[2].parse::<usize>().unwrap());
    let b: [f32; 4] = [a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap(), a[6].parse().unwrap()];
    let doc = indra_ocr::pdfium::Document::open(&path, None).unwrap();
    let g = doc.with_page(page, |api, p| p.render(api, 200.0 / 72.0)).unwrap();
    let dir = indra_ocr::models::model_dir(None, None, true).unwrap();
    for (name, mut o) in [("ort", Ocr::with_ort(&dir, 1, OcrOptions::default(), true).unwrap()), ("native", Ocr::with_native(&dir, 1, OcrOptions::default(), true).unwrap())] {
        let crops = o.crop_lines(&g, &[b], 32).unwrap();
        let refs: Vec<&ImageF> = crops.iter().collect();
        let rec = o.recognize(&g, &[b]).unwrap();
        let p = o.backend.math().unwrap().classify(&refs).unwrap();
        let read = o.math_read(&g, &[b]).unwrap();
        let s = o.backend.script(&refs).unwrap_or_default();
        println!("{name}: rec {:?} conf {:.5}; script {:?}; math p {:?}; read {:?}", rec[0].text, rec[0].conf, s, p, read);
        let _ = o.backend.has_cjk();
    }
}
