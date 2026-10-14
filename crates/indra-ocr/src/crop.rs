//! Recognizer line crops (rec_preproc.py): pad the box, crop from the page
//! (white outside it), scale to the line height; and the width batches.

use crate::imgproc::{Gray, ImageF, Interp, resize_u8};
use crate::py::round_int;

pub const LINE_HEIGHT: usize = 32;
pub const PAD_Y: f64 = 0.2;
pub const PAD_X: f64 = 0.3;
pub const WIDTH_MULTIPLE: usize = 8;
pub const MAX_WIDTH: usize = 4096;

/// The page region of a line box with its padding, white outside the page.
pub fn padded_region(gray: &Gray, b: [f32; 4], pad_x: f64, pad_y: f64) -> Gray {
    let [x0, y0, x1, y1] = b.map(|v| v as f64);
    let h = (y1 - y0).max(1.0);
    let (ix0, iy0) = ((x0 - pad_x * h).floor() as i64, (y0 - pad_y * h).floor() as i64);
    let (ix1, iy1) = ((x1 + pad_x * h).ceil() as i64, (y1 + pad_y * h).ceil() as i64);
    let (cw, ch) = ((ix1 - ix0).max(1) as usize, (iy1 - iy0).max(1) as usize);
    let mut crop = Gray::filled(cw, ch, 255);
    let (sx0, sy0) = (ix0.max(0), iy0.max(0));
    let (sx1, sy1) = (ix1.min(gray.w as i64), iy1.min(gray.h as i64));
    if sx1 > sx0 && sy1 > sy0 {
        for y in sy0..sy1 {
            let src = &gray.row(y as usize)[sx0 as usize..sx1 as usize];
            let dy = (y - iy0) as usize;
            let dx = (sx0 - ix0) as usize;
            crop.data[dy * cw + dx..dy * cw + dx + src.len()].copy_from_slice(src);
        }
    }
    crop
}

/// The recognizer input of one line: an ink map (0 = paper) ``height`` high.
pub fn crop_line(gray: &Gray, b: [f32; 4], height: usize) -> ImageF {
    let crop = padded_region(gray, b, PAD_X, PAD_Y);
    let (cw, ch) = (crop.w, crop.h);
    let out_w = (round_int(cw as f64 * height as f64 / ch as f64) as usize).clamp(WIDTH_MULTIPLE, MAX_WIDTH).max(WIDTH_MULTIPLE);
    let interp = if ch > height { Interp::Area } else { Interp::Linear };
    let r = resize_u8(&crop, out_w, height, interp);
    ImageF::from_vec(r.w, r.h, r.data.iter().map(|&v| 1.0 - v as f32 / 255.0).collect())
}

/// Line indices grouped into batches of similar width (narrowest first).
pub fn width_batches(widths: &[usize], max_batch: usize, max_ratio: f64, min_width: usize, max_area: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]);
    let mut batches: Vec<Vec<usize>> = vec![];
    for i in order {
        let wi = widths[i].max(min_width);
        if let Some(last) = batches.last_mut() {
            let w0 = widths[last[0]].max(min_width);
            if last.len() < max_batch && wi as f64 <= max_ratio * w0 as f64 && (last.len() + 1) * wi <= max_area {
                last.push(i);
                continue;
            }
        }
        batches.push(vec![i]);
    }
    batches
}

/// width_batches with the recognizer's defaults.
pub fn rec_batches(widths: &[usize], max_batch: usize) -> Vec<Vec<usize>> {
    width_batches(widths, max_batch, 1.3, 64, 64 * 640)
}

/// Lines right-padded to one width (a multiple of ``multiple``):
/// ([B, H, W] row-major, W, widths).
pub fn pad_batch(lines: &[&ImageF], multiple: usize) -> (Vec<f32>, usize, Vec<i64>) {
    let h = lines[0].h;
    let wm = lines.iter().map(|l| l.w).max().unwrap().div_ceil(multiple) * multiple;
    let mut out = vec![0f32; lines.len() * h * wm];
    for (b, l) in lines.iter().enumerate() {
        for y in 0..h {
            out[(b * h + y) * wm..(b * h + y) * wm + l.w].copy_from_slice(l.row(y));
        }
    }
    (out, wm, lines.iter().map(|l| l.w as i64).collect())
}
