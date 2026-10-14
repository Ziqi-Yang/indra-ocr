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
    let (mut script_worst, mut script_flips) = (0f32, 0usize);
    let (mut order_worst, mut order_sign) = (0f32, 0usize);
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
        box_same += (ba.len() == bb.len()) as usize
            * ba.iter().zip(&bb).filter(|(p, q)| p.xyxy().iter().zip(q.xyxy()).all(|(u, v)| (u - v).abs() < 0.01)).count();
        let oi = indra_ocr::pipeline::orient_input(&g, 768);
        let (oa, ob) = (ort.orient(&oi).unwrap(), native.orient(&oi).unwrap());
        orient_worst = orient_worst.max(oa.iter().zip(&ob).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max));
        let boxes: Vec<[f32; 4]> = ocr.detect(&g).unwrap().iter().map(|b| b.xyxy()).collect();
        if boxes.len() >= 2 {
            let f = indra_ocr::order_post::line_features(&boxes, (g.w, g.h), None);
            let (oa, ob) = (ort.order(&f, boxes.len()).unwrap(), native.order(&f, boxes.len()).unwrap());
            let diff = |x: &indra_ocr::order_post::Mat, y: &indra_ocr::order_post::Mat| {
                x.data.iter().zip(&y.data).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max)
            };
            order_worst = order_worst
                .max(diff(&oa.a, &ob.a))
                .max(diff(oa.same.as_ref().unwrap(), ob.same.as_ref().unwrap()))
                .max(diff(oa.cls.as_ref().unwrap(), ob.cls.as_ref().unwrap()));
            for k in 0..3 {
                order_worst = order_worst.max(diff(&oa.tab.as_ref().unwrap()[k], &ob.tab.as_ref().unwrap()[k]));
            }
            order_sign += oa.a.data.iter().zip(&ob.a.data).filter(|(p, q)| (**p > 0.0) != (**q > 0.0)).count();
        }
        let crops = ocr.crop_lines(&g, &boxes, 32).unwrap();
        let refs: Vec<&ImageF> = crops.iter().collect();
        let (sa, sb) = (ort.script(&refs).unwrap(), native.script(&refs).unwrap());
        script_worst = script_worst.max(sa.iter().zip(&sb).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max));
        script_flips += sa.iter().zip(&sb).filter(|(p, q)| (**p > 0.5) != (**q > 0.5)).count();
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
    eprintln!(
        "detector: max output diff {det_worst:.2e}, {box_same}/{box_total} boxes within 0.01 px; orientation: max diff {orient_worst:.2e}"
    );
    eprintln!("order: max logit diff {order_worst:.2e}, {order_sign} pair signs differ");
    assert!(order_worst < 1e-2, "order model differs by {order_worst}");
    eprintln!("script: max diff {script_worst:.2e}, {script_flips} decisions differ");
    assert!(script_flips == 0 && script_worst < 1e-4, "script classifier differs");
    assert!(box_same * 1000 >= box_total * 995, "detector boxes differ");
    assert!(orient_worst < 1e-3, "orientation differs by {orient_worst}");
    let frac = same as f64 / steps.max(1) as f64;
    eprintln!(
        "recognizer: {same}/{steps} timesteps ({:.4}%), {same_lines}/{lines} lines identical, max conf diff {worst_conf:.2e}",
        100.0 * frac
    );
    assert!(frac >= 0.999, "only {:.4}% of timesteps agree", 100.0 * frac);
    assert!(worst_conf < 1e-3, "confidence differs by {worst_conf}");
}

#[test]
fn math_matches_ort() {
    let Some(root) = golden("pages") else { return };
    let dir = indra_ocr::models::model_dir(None, None, true).expect("math models");
    let mut ocr_o = Ocr::with_ort(&dir, 1, OcrOptions::default(), true).unwrap();
    let mut ocr_n = Ocr::with_native(&dir, 1, OcrOptions::default(), true).unwrap();
    let (mut lines, mut same, mut flags, mut flags_same) = (0usize, 0usize, 0usize, 0usize);
    for name in ["2609_03923v1_p5_math", "2412_09840v2_p3_math"] {
        let d = root.join(name);
        if !d.exists() {
            continue;
        }
        let n = Npy::read(&d.join("upright.npy"));
        let (h, w) = n.hw();
        let g = Image::from_vec(w, h, n.u8());
        let boxes: Vec<[f32; 4]> = ocr_o.detect(&g).unwrap().iter().map(|b| b.xyxy()).collect();
        let crops = ocr_o.crop_lines(&g, &boxes, 32).unwrap();
        let refs: Vec<&ImageF> = crops.iter().collect();
        let (fa, fb) = (ocr_o.math_flags(&refs).unwrap(), ocr_n.math_flags(&refs).unwrap());
        flags += fa.len();
        flags_same += fa.iter().zip(&fb).filter(|(a, b)| a == b).count();
        let idx: Vec<usize> = (0..boxes.len()).filter(|&j| fa[j]).collect();
        let mb: Vec<[f32; 4]> = idx.iter().map(|&j| boxes[j]).collect();
        let (ra, rb) = (ocr_o.math_read(&g, &mb).unwrap(), ocr_n.math_read(&g, &mb).unwrap());
        for (a, b) in ra.iter().zip(&rb) {
            lines += 1;
            if a.0 == b.0 && (a.1 - b.1).abs() < 1e-3 {
                same += 1;
            } else {
                eprintln!("math differs: {:?} ({:.4}) vs {:?} ({:.4})", a.0, a.1, b.0, b.1);
            }
        }
    }
    eprintln!("math: {flags_same}/{flags} flags, {same}/{lines} readings identical");
    assert_eq!(flags_same, flags, "math classifier flags differ");
    assert!(same * 100 >= lines * 98, "math readings differ");
}

#[test]
fn classifiers_match_on_narrow_lines() {
    let dir = indra_ocr::models::model_dir(None, None, false).expect("models");
    let mut ort = OrtBackend::new(&dir, 1, false).unwrap();
    let mut native = NativeBackend::new(&dir, 1, false).unwrap();
    if !ort.has_cjk() {
        return;
    }
    let mut seed = 3u32;
    let lines: Vec<ImageF> = (1..70)
        .map(|w| {
            let data = (0..32 * w)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32
                })
                .collect();
            ImageF::from_vec(w, 32, data)
        })
        .collect();
    // one line per batch, so each runs at its own padded width
    for l in &lines {
        let (a, b) = (ort.script(&[l]).unwrap()[0], native.script(&[l]).unwrap()[0]);
        assert!((a - b).abs() < 1e-4, "width {}: script {a} vs {b}", l.w);
    }
}
