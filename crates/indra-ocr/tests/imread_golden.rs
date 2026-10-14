//! Image decoding against cv2.imread(..., IMREAD_GRAYSCALE) (tools/gen_imread_golden.py).

mod common;

use common::{Npy, golden};

#[test]
fn decodes_like_opencv() {
    let Some(dir) = golden("imread") else { return };
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut report = vec![];
    for (f, n) in manifest.as_object().unwrap() {
        let frames = match indra_ocr::io::read_images(&dir.join(f)) {
            Ok(fr) => fr,
            Err(e) => {
                report.push(format!("{f}: {e}"));
                continue;
            }
        };
        if frames.len() as u64 != n.as_u64().unwrap() {
            report.push(format!("{f}: {} frames vs {n}", frames.len()));
            continue;
        }
        for (i, fr) in frames.iter().enumerate() {
            let want = Npy::read(&dir.join(format!("{f}.{i}.npy")));
            let (h, w) = want.hw();
            if (fr.h, fr.w) != (h, w) {
                report.push(format!("{f}[{i}]: {}x{} vs {w}x{h}", fr.w, fr.h));
                continue;
            }
            let wd = want.u8();
            let diff: Vec<i32> = fr.data.iter().zip(&wd).map(|(a, b)| (*a as i32 - *b as i32).abs()).collect();
            let n = diff.iter().filter(|&&d| d > 0).count();
            // JPEG: zune-jpeg's IDCT differs from libjpeg-turbo's by one grey
            // level on a few percent of the pixels.
            let jpeg = f.ends_with(".jpg");
            if jpeg && diff.iter().all(|&d| d <= 1) && n * 20 < w * h {
                eprintln!("{f}[{i}]: {n}/{} px differ by 1 (JPEG IDCT)", w * h);
                continue;
            }
            if n > 0 {
                report.push(format!("{f}[{i}]: {n}/{} px differ, max {}", w * h, diff.iter().max().unwrap()));
            }
        }
    }
    for r in &report {
        eprintln!("{r}");
    }
    assert!(report.is_empty(), "{} files decode differently from OpenCV", report.len());
}
