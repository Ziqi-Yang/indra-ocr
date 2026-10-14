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

/// One text-layer run: its box, text, and characters with boxes (points, in
/// the rendered frame, top-left origin), in PDF order.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLine {
    pub bbox: [f64; 4],
    pub text: String,
    pub chars: Vec<(char, [f64; 4])>,
}

/// Counts from the text layer: characters, characters not upright on the
/// rendered page, bad characters, characters off the page.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerStats {
    pub chars: usize,
    pub rotated: usize,
    pub bad: usize,
    pub off_page: usize,
    pub no_ink: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextLayer {
    /// runs in PDF content order
    pub lines: Vec<TextLine>,
    /// page width, height in points (as rendered)
    pub size: (f64, f64),
    pub trusted: bool,
    /// why the layer is not trusted
    pub reason: String,
    pub stats: LayerStats,
}

#[cfg(feature = "pdf")]
mod pdf {
    use super::*;
    use crate::imgproc::Gray;
    use crate::io::InputError;
    use crate::pdfium::{Api, PageRef};
    use crate::py::{is_space, squash_spaces};

    /// Upright text runs of a page, in content order.
    pub fn extract_runs(api: &Api, page: PageRef) -> Result<(Vec<TextLine>, LayerStats), InputError> {
        let raw_chars = page.chars(api)?;
        let a = page.device_affine(api);
        let l = [[a[0][0], a[0][1]], [a[1][0], a[1][1]]];
        let mut upright_cache: Vec<(u32, bool)> = vec![];
        let (mut chars, mut raw, mut breaks): (Vec<char>, Vec<[f64; 4]>, Vec<usize>) = (vec![], vec![], vec![]);
        let mut st = LayerStats::default();
        for (u, angle, b) in raw_chars {
            let mut c = char::from_u32(u).unwrap_or('\u{FFFD}');
            if c == '\r' || c == '\n' {
                breaks.push(chars.len());
                continue;
            }
            if c == '\u{2}' {
                c = '-'; // pdfium's mark for a hyphen at a line break
            }
            if !is_space(c) {
                st.chars += 1;
                st.bad += crate::py::is_bad_char(c) as usize;
                // A character is upright on the rendered page when its baseline,
                // mapped through the page-to-render transform, runs left to right.
                let key = angle.to_bits();
                let up = match upright_cache.iter().find(|e| e.0 == key) {
                    Some(e) => e.1,
                    None => {
                        let (ca, sa) = ((angle as f64).cos(), (angle as f64).sin());
                        let d = [l[0][0] * ca + l[0][1] * sa, l[1][0] * ca + l[1][1] * sa];
                        let up = !(d[0] <= 0.0 || d[1].abs() > 1e-3 * d[0].abs().max(1e-9));
                        upright_cache.push((key, up));
                        up
                    }
                };
                if !up {
                    st.rotated += 1;
                    breaks.push(chars.len());
                    continue;
                }
            }
            if b[0].is_nan() {
                return Err(InputError("Failed to get charbox.".into()));
            }
            chars.push(c);
            raw.push([b[0], b[2], b[1], b[3]]); // l, b, r, t
        }
        if chars.is_empty() {
            return Ok((vec![], st));
        }
        // Page space -> rendered frame.
        let boxes: Vec<[f64; 4]> = raw
            .iter()
            .map(|r| {
                let xs = [r[0], r[2], r[0], r[2]];
                let ys = [r[1], r[1], r[3], r[3]];
                let px: Vec<f64> = (0..4).map(|k| a[0][0] * xs[k] + a[0][1] * ys[k] + a[0][2]).collect();
                let py: Vec<f64> = (0..4).map(|k| a[1][0] * xs[k] + a[1][1] * ys[k] + a[1][2]).collect();
                let mn = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
                let mx = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                [mn(&px), mn(&py), mx(&px), mx(&py)]
            })
            .collect();
        // Characters outside the rendered frame (beyond a crop box) are dropped.
        let (w, h) = page.size(api);
        let visible: Vec<bool> = boxes
            .iter()
            .map(|b| {
                let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
                cx >= 0.0 && cx <= w && cy >= 0.0 && cy <= h
            })
            .collect();
        st.off_page = visible.iter().filter(|v| !**v).count();
        let mut cuts = breaks.clone();
        cuts.push(chars.len());
        cuts.sort_unstable();
        cuts.dedup();
        let mut runs = vec![];
        let mut start = 0;
        for end in cuts {
            let seg: Vec<(char, [f64; 4])> = (start..end).filter(|&i| visible[i]).map(|i| (chars[i], boxes[i])).collect();
            start = end;
            let ink: Vec<[f64; 4]> = seg.iter().filter(|(c, _)| !is_space(*c)).map(|e| e.1).collect();
            if ink.is_empty() {
                continue;
            }
            let bbox = [
                ink.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min),
                ink.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min),
                ink.iter().map(|v| v[2]).fold(f64::NEG_INFINITY, f64::max),
                ink.iter().map(|v| v[3]).fold(f64::NEG_INFINITY, f64::max),
            ];
            let text = squash_spaces(&seg.iter().map(|e| e.0).collect::<String>());
            runs.push(TextLine { bbox, text, chars: seg });
        }
        Ok((runs, st))
    }

    /// Share of runs whose box holds no dark pixel in ``gray`` (``scale`` px/pt).
    pub fn no_ink_frac(gray: &Gray, runs: &[TextLine], scale: f64) -> f64 {
        let (h, w) = (gray.h as i64, gray.w as i64);
        let mut empty = 0;
        for ln in runs {
            let [x0, y0, x1, y1] = ln.bbox;
            let (r0, r1) = (((y0 * scale) as i64).max(0), (((y1 * scale) as i64) + 1).min(h));
            let (c0, c1) = (((x0 * scale) as i64).max(0), (((x1 * scale) as i64) + 1).min(w));
            let mut any = false;
            'rows: for y in r0..r1 {
                for x in c0..c1 {
                    if gray.at(x as usize, y as usize) < DARK {
                        any = true;
                        break 'rows;
                    }
                }
            }
            if !any {
                empty += 1;
            }
        }
        empty as f64 / runs.len().max(1) as f64
    }

    /// Text-layer runs of a page and the verdict on them. ``gray`` is a render
    /// at ``scale`` pixels per point (a low resolution is enough for the ink check).
    pub fn read_text_layer(api: &Api, page: PageRef, gray: &Gray, scale: f64) -> Result<TextLayer, InputError> {
        let size = page.size(api);
        let (n, inv) = page.text_objects(api, 4);
        let inv = if n > 0 { inv as f64 / n as f64 } else { 0.0 };
        let layer = |lines, trusted, reason: String, stats| TextLayer { lines, size, trusted, reason, stats };
        if inv > MAX_INVISIBLE_FRAC {
            return Ok(layer(vec![], false, format!("invisible text {}% (OCR layer)", pct(inv)), LayerStats::default()));
        }
        let (runs, mut st) = extract_runs(api, page)?;
        if runs.is_empty() {
            return Ok(layer(vec![], false, "no text layer".into(), st));
        }
        if st.bad as f64 > MAX_BAD_CHAR_FRAC * st.chars.max(1) as f64 {
            return Ok(layer(runs, false, BROKEN_FONT_MAP.into(), st));
        }
        let empty = no_ink_frac(gray, &runs, scale);
        st.no_ink = Some(crate::py::round_n(empty, 4));
        if empty > MAX_NO_INK_LINE_FRAC {
            return Ok(layer(runs, false, format!("{}% of text runs have no ink", pct(empty)), st));
        }
        Ok(layer(runs, true, String::new(), st))
    }

    /// Python's f"{x:.0%}".
    fn pct(x: f64) -> String {
        format!("{:.0}", x * 100.0)
    }
}

#[cfg(feature = "pdf")]
pub use pdf::{extract_runs, no_ink_frac, read_text_layer};
