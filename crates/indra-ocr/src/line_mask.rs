//! CenterMask (line_mask.py): before the recognizer crops a line, the ink of
//! neighbouring lines in its window is whitened, using mask.onnx when the
//! model folder has it.

use crate::backend::{Backend, Result};
use crate::imgproc::{Gray, ImageF, Interp, resize_linear_f32, resize_u8};
use crate::py::{percentile, round_int};

pub const WINDOW_H: usize = 96;
pub const WINDOW_ABOVE: f64 = 1.0;
pub const WINDOW_SIDE: f64 = 0.5;
pub const KEEP: f32 = 0.5;

/// The window around a line box (page pixels), clipped to the page.
pub fn window_box(b: [f32; 4], w_page: f64, h_page: f64) -> [f64; 4] {
    let [x0, y0, x1, y1] = b.map(|v| v as f64);
    let h = (y1 - y0).max(1.0);
    [
        (x0 - WINDOW_SIDE * h).max(0.0),
        (y0 - WINDOW_ABOVE * h).max(0.0),
        (x1 + WINDOW_SIDE * h).min(w_page),
        (y1 + WINDOW_ABOVE * h).min(h_page),
    ]
}

fn resize_window(a: &Gray, w: usize, h: usize) -> Gray {
    let shrink = h as f64 / a.h.max(1) as f64;
    resize_u8(a, w, h, if shrink >= 0.5 { Interp::Linear } else { Interp::Area })
}

/// For each line box: (window with foreign ink whitened, the box inside it).
pub fn masked_windows(gray: &Gray, boxes: &[[f32; 4]], backend: &mut dyn Backend) -> Result<Vec<(Gray, [f32; 4])>> {
    let mut wins = vec![];
    for b in boxes {
        let wb = window_box(*b, gray.w as f64, gray.h as f64).map(|v| round_int(v).max(0) as usize);
        let win = gray.crop(wb[0], wb[1], wb[2], wb[3]);
        let (ox, oy) = (wb[0] as f32, wb[1] as f32);
        wins.push((win, [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]));
    }
    // Ink maps at the mask model's height, batched 16 at a time by width.
    let xs: Vec<ImageF> = wins
        .iter()
        .map(|(w, _)| {
            let wd = (round_int(w.w as f64 * WINDOW_H as f64 / w.h.max(1) as f64 / 8.0) * 8).max(32) as usize;
            let r = resize_window(w, wd, WINDOW_H);
            ImageF::from_vec(r.w, r.h, r.data.iter().map(|&v| 1.0 - v as f32 / 255.0).collect())
        })
        .collect();
    let mut keep: Vec<Option<Vec<bool>>> = vec![None; wins.len()];
    let order = crate::py::argsort_by(&xs.iter().map(|x| x.w).collect::<Vec<_>>(), |w| w as f64);
    for idx in order.chunks(16) {
        let wm = idx.iter().map(|&i| xs[i].w).max().unwrap();
        let mut batch = vec![0f32; idx.len() * WINDOW_H * wm];
        for (j, &i) in idx.iter().enumerate() {
            for y in 0..WINDOW_H {
                let d = (j * WINDOW_H + y) * wm;
                batch[d..d + xs[i].w].copy_from_slice(xs[i].row(y));
            }
        }
        let p = backend.mask(&batch, idx.len(), WINDOW_H, wm)?;
        for (j, &i) in idx.iter().enumerate() {
            let mut pm = ImageF::new(xs[i].w, WINDOW_H);
            for y in 0..WINDOW_H {
                let s = (j * WINDOW_H + y) * wm;
                pm.row_mut(y).copy_from_slice(&p[s..s + xs[i].w]);
            }
            let (ww, wh) = (wins[i].0.w, wins[i].0.h);
            // The probabilities are scaled back bilinearly before the threshold.
            let up = resize_linear_f32(&pm, ww.max(1), wh.max(1));
            keep[i] = Some(up.data.iter().map(|&v| v >= KEEP).collect());
        }
    }
    let mut out = vec![];
    for ((w, b), k) in wins.into_iter().zip(keep) {
        let mut m = w.clone();
        // paper level: 90th percentile grey on a 1/16 subsample of the window
        let paper = if w.data.is_empty() {
            255
        } else {
            let sub: Vec<f64> =
                (0..w.h).step_by(4).flat_map(|y| (0..w.w).step_by(4).map(move |x| (x, y))).map(|(x, y)| w.at(x, y) as f64).collect();
            percentile(&sub, 90.0) as u8
        };
        if let Some(k) = k {
            for (v, keep) in m.data.iter_mut().zip(k) {
                if !keep {
                    *v = paper;
                }
            }
        }
        out.push((m, b));
    }
    Ok(out)
}
