//! LineNet input and output (det_post.py): the page as a padded ink map, and
//! line boxes from the core / top / bottom maps.

use crate::imgproc::{Gray, ImageF, Interp, connected_components, resize_u8};
use crate::py::round_int;

pub const CORE_X_SHRINK: f64 = 0.15;
pub const OUT_STRIDE: usize = 2;
pub const INPUT_MULTIPLE: usize = 32;

/// A line box in pixels, with the detector's mean core probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl DetBox {
    pub fn xyxy(&self) -> [f32; 4] {
        [self.x0, self.y0, self.x1, self.y1]
    }
}

/// The page with its long side scaled to ``long_side`` and padded with white to a
/// multiple of 32: (ink map, 0 = paper; scale).
pub fn page_to_input(gray: &Gray, long_side: usize) -> (ImageF, f64) {
    let (h, w) = (gray.h, gray.w);
    let scale = long_side as f64 / h.max(w) as f64;
    let nh = round_int(h as f64 * scale).max(1) as usize;
    let nw = round_int(w as f64 * scale).max(1) as usize;
    let resized;
    let g = if (nh, nw) != (h, w) {
        resized = resize_u8(gray, nw, nh, if scale < 1.0 { Interp::Area } else { Interp::Linear });
        &resized
    } else {
        gray
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = ImageF::new(pw, ph);
    for y in 0..nh {
        let (s, d) = (g.row(y), &mut out.data[y * pw..y * pw + nw]);
        for (o, &v) in d.iter_mut().zip(s) {
            *o = 1.0 - v as f32 / 255.0;
        }
    }
    (out, scale)
}

/// Line boxes from the detector output [3, h, w] (core logit, top distance,
/// bottom distance), in input pixels.
pub fn decode(out: &[f32], h: usize, w: usize, thresh: f32, min_area: i32) -> Vec<DetBox> {
    let n_px = h * w;
    let (core, top_d, bot_d) = (&out[..n_px], &out[n_px..2 * n_px], &out[2 * n_px..3 * n_px]);
    let prob: Vec<f32> = core.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let mask = crate::imgproc::Image::from_vec(w, h, prob.iter().map(|&p| (p > thresh) as u8).collect());
    let (labels, stats) = connected_components(&mask);
    let n = stats.len();
    if n <= 1 {
        return vec![];
    }
    let s = OUT_STRIDE as f32;
    let mut cnt = vec![0f64; n];
    let (mut y0, mut y1, mut sc) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let l = l as usize;
        let ys = ((i / w) as f32 + 0.5) * s;
        cnt[l] += 1.0;
        y0[l] += (ys - top_d[i]) as f64;
        y1[l] += (ys + bot_d[i]) as f64;
        sc[l] += prob[i] as f64;
    }
    let mut boxes = Vec::new();
    for i in 1..n {
        if stats[i].area < min_area {
            continue;
        }
        let c = (cnt[i] as f32).max(1.0) as f64;
        let (a, b, score) = (y0[i] / c, y1[i] / c, sc[i] / c);
        let lh = (b - a).max(1.0);
        let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
        let x0 = (stats[i].left * OUT_STRIDE as i32) as f64 - ex;
        let x1 = ((stats[i].left + stats[i].width) * OUT_STRIDE as i32) as f64 + ex;
        boxes.push(DetBox { x0: x0 as f32, y0: a as f32, x1: x1 as f32, y1: b as f32, score: score as f32 });
    }
    boxes
}
