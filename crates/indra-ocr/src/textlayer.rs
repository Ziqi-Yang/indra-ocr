//! The PDF text layer (textlayer.py): runs of characters from pdfium, the checks
//! that decide whether to trust them, and whether the content order reads.

use crate::order_post::Mat;
use crate::py::median_f32;

pub const MAX_INVISIBLE_FRAC: f64 = 0.05;
pub const MAX_BAD_CHAR_FRAC: f64 = 0.005;
pub const MAX_NO_INK_LINE_FRAC: f64 = 0.03;
/// Grey level below which a pixel counts as ink.
pub const DARK: u8 = 160;
pub const MIN_MODEL_AGREEMENT: f64 = 0.85;
/// TextLayer::reason when the layer's codes are not text.
pub const BROKEN_FONT_MAP: &str = "broken font map";

/// Characters that mark a broken font map.
pub fn bad_char_count(text: &str) -> usize {
    text.chars().filter(|&c| crate::py::is_bad_char(c)).count()
}

/// Moves upward in content order that do not go to a column on the right.
pub fn order_backtracks(boxes: &[[f32; 4]]) -> usize {
    if boxes.len() < 3 {
        return 0;
    }
    let h = median_f32(&boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>()) as f64;
    boxes
        .windows(2)
        .filter(|w| {
            let (a, b) = (w[0], w[1]);
            b[1] < a[1] - (0.5 * h) as f32 && b[0] < a[2] - h as f32
        })
        .count()
}

/// Is the content order a reading order? Few upward moves that are not column
/// changes, and the order model agrees with most consecutive pairs.
pub fn pdf_order_coherent(boxes: &[[f32; 4]], pair_logits: Option<&Mat>) -> bool {
    if order_backtracks(boxes) > 1usize.max((0.02 * boxes.len() as f64) as usize) {
        return false;
    }
    let Some(l) = pair_logits else { return true };
    if boxes.len() < 3 {
        return true;
    }
    let n = boxes.len() - 1;
    let agree = (0..n).filter(|&i| l.at(i, i + 1) > 0.0).count();
    agree as f64 / n as f64 >= MIN_MODEL_AGREEMENT
}
