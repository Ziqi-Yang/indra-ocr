//! Int8 vs float recognizer on page crops: text agreement and speed.
//!   cargo run --release --example eval_int8 -- PAGE.npy...
use std::time::Instant;

use indra_engine::rec::Recognizer;
use indra_ocr::imgproc::{Gray, ImageF};
use indra_ocr::pipeline::{Ocr, OcrOptions};

fn load_npy(p: &str) -> Gray {
    let raw = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen]).unwrap();
    let s = &header[header.find("'shape': (").unwrap() + 10..];
    let dims: Vec<usize> = s[..s.find(')').unwrap()].split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    Gray::from_vec(dims[1], dims[0], raw[10 + hlen..].to_vec())
}

fn text(ocr: &Ocr, ids: &[u32]) -> String {
    let keep = (0..ids.len()).filter(|&t| ids[t] != 0 && (t == 0 || ids[t] != ids[t - 1])).map(|t| ids[t]);
    ocr.vocab.decode(keep)
}

fn main() {
    let dir = indra_ocr::models::model_dir(None, None, false).unwrap();
    let mut ocr = Ocr::with_native(&dir, 1, OcrOptions::default(), false).unwrap();
    let f = Recognizer::load(&dir.join("rec.onnx")).unwrap();
    let q = Recognizer::load_int8(&dir.join("rec.onnx")).unwrap();
    let par = indra_engine::par::serial();
    let (mut lines, mut same, mut chars, mut cdiff) = (0, 0, 0usize, 0usize);
    let (mut tf, mut tq) = (0.0, 0.0);
    for p in std::env::args().skip(1) {
        let g = load_npy(&p);
        let (up, _, _) = ocr.upright(&g, true).unwrap();
        let boxes: Vec<[f32; 4]> = ocr.detect(&up).unwrap().iter().map(|b| b.xyxy()).collect();
        let crops: Vec<ImageF> = ocr.crop_lines(&up, &boxes, 32).unwrap();
        let input: Vec<(&[f32], usize)> = crops.iter().map(|c| (c.data.as_slice(), c.w)).collect();
        let t0 = Instant::now();
        let a = f.run(&input, par);
        tf += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let b = q.run(&input, par);
        tq += t0.elapsed().as_secs_f64();
        for (x, y) in a.iter().zip(&b) {
            let (s, t) = (text(&ocr, &x.ids), text(&ocr, &y.ids));
            lines += 1;
            chars += s.chars().count();
            if s == t {
                same += 1;
            } else {
                // character edit distance
                let (u, v): (Vec<char>, Vec<char>) = (s.chars().collect(), t.chars().collect());
                let mut d: Vec<usize> = (0..=v.len()).collect();
                for i in 1..=u.len() {
                    let mut prev = d[0];
                    d[0] = i;
                    for j in 1..=v.len() {
                        let cur = d[j];
                        d[j] = (d[j] + 1).min(d[j - 1] + 1).min(prev + (u[i - 1] != v[j - 1]) as usize);
                        prev = cur;
                    }
                }
                cdiff += d[v.len()];
                if lines - same <= 8 {
                    println!("  {s:?}\n  {t:?}");
                }
            }
        }
    }
    println!("int8: {same}/{lines} lines identical, {cdiff} character edits in {chars} characters ({:.3}%)", 100.0 * cdiff as f64 / chars.max(1) as f64);
    println!("recognizer time: float {:.0} ms, int8 {:.0} ms", tf * 1000.0, tq * 1000.0);
}
