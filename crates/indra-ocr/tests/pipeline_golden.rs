//! The pipeline on ONNX Runtime against the Python package, stage by stage
//! (tools/dump_golden.py). Needs the models (HF cache or INDRA_OCR_MODEL_DIR)
//! and ONNX Runtime (ORT_DYLIB_PATH).

#![cfg(feature = "ort")]

mod common;

use common::{Npy, golden};
use indra_ocr::imgproc::Image;
use indra_ocr::pipeline::{Ocr, OcrOptions, TokVal};
use serde_json::Value;

fn ocr(math: bool) -> Ocr {
    let dir = indra_ocr::models::model_dir(None, None, math).expect("models");
    Ocr::with_ort(&dir, 1, OcrOptions::default(), math).expect("ort backend")
}

fn gray(p: &std::path::Path) -> Image<u8> {
    let n = Npy::read(p);
    let (h, w) = n.hw();
    Image::from_vec(w, h, n.u8())
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn stages_match_python() {
    let Some(root) = golden("pages") else { return };
    let mut ocr_plain = ocr(false);
    let mut ocr_math: Option<Ocr> = None;
    let mut failures = vec![];
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect();
    dirs.sort();
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let stages: Value = serde_json::from_str(&std::fs::read_to_string(d.join("stages.json")).unwrap()).unwrap();
        let g = gray(&d.join("gray.npy"));
        let o = if name.ends_with("_math") { ocr_math.get_or_insert_with(|| ocr(true)) } else { &mut ocr_plain };
        // orientation and deskew
        let (up, k, skew) = o.upright(&g, true).unwrap();
        let want_up = gray(&d.join("upright.npy"));
        assert_eq!(k as i64, stages["orientation"].as_i64().unwrap(), "{name}: orientation");
        assert!(close(skew, stages["skew"].as_f64().unwrap(), 1e-9), "{name}: skew {skew} vs {}", stages["skew"]);
        let diff = up.data.iter().zip(&want_up.data).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{name}: upright page differs in {diff} px");
        // detection
        let boxes = o.detect(&up).unwrap();
        let want: Vec<Vec<f64>> = serde_json::from_value(stages["boxes"].clone()).unwrap();
        if boxes.len() != want.len() {
            failures.push(format!("{name}: {} boxes vs {}", boxes.len(), want.len()));
            continue;
        }
        let mut worst = 0f64;
        for (b, w) in boxes.iter().zip(&want) {
            for (x, y) in b.xyxy().iter().zip(w) {
                worst = worst.max((*x as f64 - y).abs());
            }
        }
        if worst > 0.05 {
            failures.push(format!("{name}: boxes differ by up to {worst:.3} px"));
        }
        // recognition
        let bx: Vec<[f32; 4]> = want.iter().map(|w| [w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32]).collect();
        let rec = o.recognize(&up, &bx).unwrap();
        let want_rec = stages["rec"].as_array().unwrap();
        let mut text_diff = 0;
        for (i, (r, w)) in rec.iter().zip(want_rec).enumerate() {
            let wt = w[0].as_str().unwrap();
            if r.text != wt {
                text_diff += 1;
                if text_diff <= 5 {
                    failures.push(format!("{name} line {i}: {:?} vs {:?}", r.text, wt));
                }
            }
            if !close(r.conf, w[1].as_f64().unwrap(), 1e-4) {
                failures.push(format!("{name} line {i}: conf {} vs {}", r.conf, w[1]));
            }
            let wt = w[2].as_array().unwrap();
            if r.toks.len() != wt.len() {
                failures.push(format!("{name} line {i}: {} tokens vs {}", r.toks.len(), wt.len()));
                continue;
            }
            for (t, u) in r.toks.iter().zip(wt) {
                let same_v = match &t.v {
                    TokVal::Id(id) => u[0].as_u64() == Some(*id as u64),
                    TokVal::Str(s) => u[0].as_str() == Some(s),
                };
                if !same_v || !close(t.x, u[1].as_f64().unwrap(), 1e-6) {
                    failures.push(format!("{name} line {i}: token {:?} vs {u}", t));
                    break;
                }
            }
        }
        // the whole page
        let page = o.ocr(&g, true).unwrap();
        let want_page: Value = serde_json::from_str(&std::fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
        let got = page.to_json();
        for key in ["source", "rotation", "skew", "note", "markdown", "blocks"] {
            if got[key] != want_page[key] {
                failures.push(format!("{name}: {key} differs"));
                if key == "markdown" {
                    let (a, b) = (got[key].as_str().unwrap_or(""), want_page[key].as_str().unwrap_or(""));
                    for (la, lb) in a.lines().zip(b.lines()) {
                        if la != lb {
                            failures.push(format!("   rust:   {la}\n   python: {lb}"));
                            break;
                        }
                    }
                }
            }
        }
        let (gl, wl) = (got["lines"].as_array().unwrap(), want_page["lines"].as_array().unwrap());
        if gl.len() != wl.len() {
            failures.push(format!("{name}: {} lines vs {}", gl.len(), wl.len()));
        } else {
            for (i, (a, b)) in gl.iter().zip(wl).enumerate() {
                if a["text"] != b["text"] || a["box"] != b["box"] || !close(a["conf"].as_f64().unwrap(), b["conf"].as_f64().unwrap(), 2e-4)
                {
                    failures.push(format!("{name}: page line {i}: {a} vs {b}"));
                    break;
                }
            }
        }
        eprintln!("{name}: checked ({} lines)", gl.len());
    }
    for f in &failures {
        eprintln!("{f}");
    }
    assert!(failures.is_empty(), "{} differences from Python", failures.len());
}
