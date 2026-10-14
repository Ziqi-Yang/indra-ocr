//! The native engine's models against ONNX Runtime's on the fixture pages
//! (tools/dump_golden.py inputs): same classes per timestep, same text.

#![cfg(all(feature = "ort", feature = "native"))]

mod common;

use common::{Npy, golden};
use indra_ocr::backend::{Backend, RecModel};
use indra_ocr::imgproc::{Image, ImageF};
use indra_ocr::native_backend::NativeBackend;
use indra_ocr::ort_backend::OrtBackend;
use indra_ocr::pipeline::{Ocr, OcrOptions};

#[test]
fn recognizer_matches_ort() {
    let Some(root) = golden("pages") else { return };
    let dir = indra_ocr::models::model_dir(None, None, false).expect("models");
    let mut ocr = Ocr::with_ort(&dir, 1, OcrOptions::default(), false).unwrap();
    let mut ort = OrtBackend::new(&dir, 1, false).unwrap();
    let mut native = NativeBackend::new(&dir, 1, false).unwrap();
    let (mut steps, mut same, mut lines, mut same_lines) = (0usize, 0usize, 0usize, 0usize);
    let mut worst_conf = 0f32;
    let (mut det_worst, mut orient_worst, mut box_total, mut box_same) = (0f32, 0f32, 0usize, 0usize);
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect();
    dirs.sort();
    for d in dirs.iter().filter(|d| !d.to_string_lossy().ends_with("_math")) {
        let n = Npy::read(&d.join("upright.npy"));
        let (h, w) = n.hw();
        let g = Image::from_vec(w, h, n.u8());
        // detector and orientation outputs
        let (x, _) = indra_ocr::det_post::page_to_input(&g, 1024);
        let (da, db) = (ort.det(&x).unwrap(), native.det(&x).unwrap());
        assert_eq!((da.h, da.w), (db.h, db.w), "det output size");
        let dmax = da.data.iter().zip(&db.data).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        det_worst = det_worst.max(dmax);
        let ba = indra_ocr::det_post::decode(&da.data, da.h, da.w, 0.5, 2);
        let bb = indra_ocr::det_post::decode(&db.data, db.h, db.w, 0.5, 2);
        box_total += ba.len();
        box_same += (ba.len() == bb.len()) as usize * ba.iter().zip(&bb).filter(|(p, q)| p.xyxy().iter().zip(q.xyxy()).all(|(u, v)| (u - v).abs() < 0.01)).count();
        let oi = indra_ocr::pipeline::orient_input(&g, 768);
        let (oa, ob) = (ort.orient(&oi).unwrap(), native.orient(&oi).unwrap());
        orient_worst = orient_worst.max(oa.iter().zip(&ob).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max));
        let boxes: Vec<[f32; 4]> = ocr.detect(&g).unwrap().iter().map(|b| b.xyxy()).collect();
        let crops = ocr.crop_lines(&g, &boxes, 32).unwrap();
        let refs: Vec<&ImageF> = crops.iter().collect();
        for model in [RecModel::Main, RecModel::Cjk] {
            let a = ort.rec(model, &refs).unwrap();
            let b = native.rec(model, &refs).unwrap();
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.ids.len(), y.ids.len(), "timesteps");
                steps += x.ids.len();
                same += x.ids.iter().zip(&y.ids).filter(|(p, q)| p == q).count();
                lines += 1;
                same_lines += (x.ids == y.ids) as usize;
                for t in 0..x.ids.len() {
                    let keep = x.ids[t] != 0 && (t == 0 || x.ids[t] != x.ids[t - 1]);
                    if keep && x.ids == y.ids {
                        worst_conf = worst_conf.max((x.conf[t] - y.conf[t]).abs());
                    }
                }
            }
        }
    }
    eprintln!("detector: max output diff {det_worst:.2e}, {box_same}/{box_total} boxes within 0.01 px; orientation: max diff {orient_worst:.2e}");
    assert!(box_same * 1000 >= box_total * 995, "detector boxes differ");
    assert!(orient_worst < 1e-3, "orientation differs by {orient_worst}");
    let frac = same as f64 / steps.max(1) as f64;
    eprintln!("recognizer: {same}/{steps} timesteps ({:.4}%), {same_lines}/{lines} lines identical, max conf diff {worst_conf:.2e}", 100.0 * frac);
    assert!(frac >= 0.999, "only {:.4}% of timesteps agree", 100.0 * frac);
    assert!(worst_conf < 1e-3, "confidence differs by {worst_conf}");
}
