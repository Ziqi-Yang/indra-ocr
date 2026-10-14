//! The OpenCV ports against cv2's own output (tools/gen_cv_golden.py).

mod common;

use common::{Npy, golden};
use indra_ocr::imgproc::*;

fn load_case(dir: &std::path::Path, name: &str, arr: &str) -> Npy {
    Npy::read(&dir.join(name).join(format!("{arr}.npy")))
}

/// (pixels that differ, largest difference)
fn diff_u8(a: &[u8], b: &[u8]) -> (usize, i32) {
    let mut n = 0;
    let mut m = 0;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as i32 - *y as i32).abs();
        if d > 0 {
            n += 1;
            m = m.max(d);
        }
    }
    (n, m)
}

#[test]
fn opencv_parity() {
    let Some(dir) = golden("cv") else { return };
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut report = vec![];
    let mut worst = 0i32;
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let src = load_case(&dir, name, "src");
        let (h, w) = src.hw();
        match kind {
            "resize_u8" => {
                let dst = load_case(&dir, name, "dst");
                let (dh, dw) = dst.hw();
                let interp = if case["interp"] == "area" { Interp::Area } else { Interp::Linear };
                let out = resize_u8(&Image::from_vec(w, h, src.u8()), dw, dh, interp);
                let (n, m) = diff_u8(&out.data, &dst.u8());
                worst = worst.max(m);
                if n > 0 {
                    report.push(format!("{name} {interp:?} {w}x{h}->{dw}x{dh}: {n}/{} px differ, max {m}", dw * dh));
                }
            }
            "resize_f32" => {
                let dst = load_case(&dir, name, "dst");
                let (dh, dw) = dst.hw();
                let out = resize_linear_f32(&Image::from_vec(w, h, src.f32()), dw, dh);
                let m = out.data.iter().zip(dst.f32()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(m < 1e-4, "{name}: max diff {m}");
            }
            "warp_u8" | "warp_f32" => {
                let dst = load_case(&dir, name, "dst");
                let m: [f64; 6] = load_case(&dir, name, "M").f64().try_into().unwrap();
                let a = case["angle"].as_f64().unwrap();
                let mine = rotation_matrix((w as f64 / 2.0, h as f64 / 2.0), a, 1.0);
                for k in 0..6 {
                    assert!((mine[k] - m[k]).abs() < 1e-9, "{name}: rotation matrix {mine:?} vs {m:?}");
                }
                if kind == "warp_u8" {
                    let out = warp_affine_u8(&Image::from_vec(w, h, src.u8()), &m, w, h, Border::Replicate, 0);
                    let (n, mx) = diff_u8(&out.data, &dst.u8());
                    worst = worst.max(mx);
                    if n > 0 {
                        report.push(format!("{name} {w}x{h} {a}deg: {n}/{} px differ, max {mx}", w * h));
                    }
                } else {
                    let out = warp_affine_f32(&Image::from_vec(w, h, src.f32()), &m, w, h, Border::Constant, 0.0);
                    let d = dst.f32();
                    let mx = out.data.iter().zip(&d).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let n = out.data.iter().zip(&d).filter(|(a, b)| (*a - *b).abs() > 1e-4).count();
                    if n > 0 {
                        report.push(format!("{name} f32 {w}x{h} {a}deg: {n}/{} px differ, max {mx}", w * h));
                    }
                }
            }
            "cc" => {
                let (labels, stats) = connected_components(&Image::from_vec(w, h, src.u8()));
                let want = load_case(&dir, name, "labels").i32();
                assert_eq!(labels, want, "{name}: labels");
                let st = load_case(&dir, name, "stats").i32();
                assert_eq!(stats.len(), case["n"].as_u64().unwrap() as usize);
                for (i, s) in stats.iter().enumerate() {
                    assert_eq!([s.left, s.top, s.width, s.height, s.area], st[i * 5..i * 5 + 5], "{name}: stats {i}");
                }
            }
            _ => panic!("unknown case kind {kind}"),
        }
    }
    for r in &report {
        eprintln!("{r}");
    }
    assert!(report.is_empty(), "{} cases differ from OpenCV (max diff {worst})", report.len());
}
