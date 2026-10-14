//! The OCR pipeline (pipeline.py): orientation and deskew, line detection, CTC
//! line recognition (with script routing and math), line splitting, reading
//! order, layout blocks and markdown.

use std::time::Instant;

use serde::Serialize;

use crate::backend::{Backend, BackendError, RecModel};
use crate::crop::{PAD_X, crop_line};
use crate::det_post::{DetBox, decode, page_to_input};
use crate::imgproc::{Border, Gray, ImageF, Interp, resize_u8, rot90_cw, rotation_matrix, warp_affine_u8, warp_ink_row_sums};
use crate::layout::{Block, Cells, page_blocks, page_markdown};
use crate::math::MathReader;
use crate::models::{Meta, ModelError};
use crate::order_post::{Mat, class_id, decode_blocks_order, decode_order, line_features, order_xycut};
use crate::py::{argmax, median_f32, pairwise_sum_f32, round_int, round_n, round1, squash_spaces};
use crate::textlayer::pdf_order_coherent;
use crate::vocab::Vocab;

/// Median line height (input px) that triggers a second detection pass.
pub const ADAPTIVE_MIN_H: f64 = 9.0;
pub const ADAPTIVE_TARGET_H: f64 = 12.0;
pub const ADAPTIVE_MAX_SIDE: f64 = 2048.0;
/// Gaps between CTC character positions, in line heights.
pub const SPLIT_GAP: f64 = 3.0;
pub const CELL_GAP: f64 = 1.8;
pub const GUTTER_GAP: f64 = 1.6;
pub const GUTTER_EDGES: usize = 3;
pub const ORIENT_SIDE: usize = 768;
pub const ORIENT_MIN: f32 = 0.95;
/// A recognized line under this confidence gets no text.
pub const MIN_CONF: f64 = 0.5;
pub const DESKEW_MAX: f64 = 5.0;
pub const DESKEW_MIN: f64 = 0.3;

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, OcrError>;

/// A token's value: a recognizer class id, or a string (a CJK recognizer
/// character, a text-layer character, or a whole math reading).
#[derive(Clone, Debug, PartialEq)]
pub enum TokVal {
    Id(u32),
    Str(String),
}

/// One token of a line with its page x position (and, for text-layer
/// characters, their index in the PDF content order).
#[derive(Clone, Debug, PartialEq)]
pub struct Tok {
    pub v: TokVal,
    pub x: f64,
    pub idx: Option<usize>,
}

/// A recognized (or text-layer) line: text, confidence, tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct RecLine {
    pub text: String,
    pub conf: f64,
    pub toks: Vec<Tok>,
}

impl RecLine {
    pub fn empty(conf: f64) -> Self {
        RecLine { text: String::new(), conf, toks: vec![] }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Line {
    /// x0, y0, x1, y1 in image pixels
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
    pub text: String,
    pub conf: f64,
}

/// Per-step milliseconds, in the order the steps ran.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timings(pub Vec<(String, f64)>);

impl Timings {
    pub fn set(&mut self, k: &str, v: f64) {
        match self.0.iter_mut().find(|(n, _)| n == k) {
            Some(e) => e.1 = v,
            None => self.0.push((k.to_string(), v)),
        }
    }
    pub fn get(&self, k: &str) -> Option<f64> {
        self.0.iter().find(|(n, _)| n == k).map(|e| e.1)
    }
    /// total_ms: the sum of every other step.
    pub fn total(&mut self) {
        let s: f64 = self.0.iter().filter(|(k, _)| k != "total_ms").map(|e| e.1).sum();
        self.set("total_ms", s);
    }
    pub fn since(&mut self, k: &str, t0: Instant) {
        self.set(k, t0.elapsed().as_secs_f64() * 1000.0);
    }
}

impl Serialize for Timings {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    /// in reading order
    pub lines: Vec<Line>,
    pub timings: Timings,
    pub markdown: Option<String>,
    /// ocr | text (PDF text layer) | hybrid (text layer + OCR) | error
    pub source: String,
    /// why the page took this path
    pub note: String,
    /// layout blocks in reading order
    pub blocks: Vec<Block>,
    /// degrees counter-clockwise the input was turned; boxes are upright
    pub rotation: i32,
    /// degrees of small tilt undone after the turn
    pub skew: f64,
}

impl Page {
    pub fn new(lines: Vec<Line>) -> Self {
        Page {
            lines,
            timings: Timings::default(),
            markdown: None,
            source: "ocr".into(),
            note: String::new(),
            blocks: vec![],
            rotation: 0,
            skew: 0.0,
        }
    }
    /// One line per detected line, in reading order.
    pub fn text(&self) -> String {
        self.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n")
    }
    /// The page as JSON (the fields `indra-ocr --format json` writes, after file and page).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "source": self.source,
            "rotation": self.rotation,
            "skew": self.skew,
            "note": self.note,
            "lines": self.lines,
            "markdown": self.markdown,
            "blocks": self.blocks,
            "timings": self.timings,
        })
    }

    /// The page that a failure leaves.
    pub fn error(note: String) -> Self {
        let mut p = Page::new(vec![]);
        p.timings.set("total_ms", 0.0);
        p.markdown = Some(String::new());
        p.source = "error".into();
        p.note = note;
        p
    }
}

/// Options of the OCR pipeline.
#[derive(Clone, Debug)]
pub struct OcrOptions {
    /// Run the detector again at a higher resolution on pages of small text.
    pub adaptive: bool,
    /// Build tables from the order model's table heads (when it has them).
    pub table_heads: bool,
    /// Keep page headers and footers in the markdown.
    pub furniture: bool,
}

impl Default for OcrOptions {
    fn default() -> Self {
        OcrOptions { adaptive: true, table_heads: false, furniture: false }
    }
}

/// Degrees (counter-clockwise) that straighten the text rows of a page: the
/// angle whose horizontal ink profile changes most sharply row to row.
pub fn estimate_skew(gray: &Gray) -> f64 {
    estimate_skew_on(gray, None)
}

/// ``estimate_skew``, the angles evaluated on ``pool`` when given.
pub fn estimate_skew_on(gray: &Gray, pool: Option<&rayon::ThreadPool>) -> f64 {
    let side = 800.0;
    let s = side / gray.h.max(gray.w) as f64;
    let sw = round_int(gray.w as f64 * s).max(1) as usize;
    let sh = round_int(gray.h as f64 * s).max(1) as usize;
    let small = resize_u8(gray, sw, sh, Interp::Area);
    let ink = ImageF::from_vec(sw, sh, small.data.iter().map(|&v| (255.0 - v as f32) / 255.0).collect());
    // a blank page has no rows to straighten
    if (pairwise_sum_f32(&ink.data) / ink.data.len() as f32) < 0.003 {
        return 0.0;
    }
    let c = (sw as f64 / 2.0, sh as f64 / 2.0);
    let sharpness = |a: f64| -> f32 {
        let p = warp_ink_row_sums(&small, &rotation_matrix(c, a, 1.0), sw, sh);
        let d: Vec<f32> = p.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).collect();
        pairwise_sum_f32(&d)
    };
    let best = py_max(&arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 0.5), &sharpness, pool);
    py_max(&arange(best - 0.5, best + 0.5 + 1e-6, 0.1), &sharpness, pool)
}

/// np.arange(start, stop, step): start + i * (second - first).
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..n).map(|i| if i == 0 { start } else { start + i as f64 * delta }).collect()
}

/// Python max(values, key=f): the first value with the largest key.
fn py_max(values: &[f64], f: &(impl Fn(f64) -> f32 + Sync), pool: Option<&rayon::ThreadPool>) -> f64 {
    use rayon::prelude::*;
    let keys: Vec<f32> = match pool {
        Some(p) => p.install(|| values.par_iter().map(|&v| f(v)).collect()),
        None => values.iter().map(|&v| f(v)).collect(),
    };
    let mut best = 0;
    for i in 1..keys.len() {
        if keys[i] > keys[best] {
            best = i;
        }
    }
    values[best]
}

/// The page rotated by ``degrees`` counter-clockwise about its centre (same size).
pub fn rotate_page(gray: &Gray, degrees: f64) -> Gray {
    let m = rotation_matrix((gray.w as f64 / 2.0, gray.h as f64 / 2.0), degrees, 1.0);
    warp_affine_u8(gray, &m, gray.w, gray.h, Border::Replicate, 0)
}

/// A box of the page that rotate_page(.., degrees) made, mapped back to the page
/// before it: the bounding box of the four mapped corners, clipped to it.
pub fn unrotate_box(b: [f64; 4], degrees: f64, wh: (usize, usize)) -> [f64; 4] {
    let (w, h) = (wh.0 as f64, wh.1 as f64);
    let m = rotation_matrix((w / 2.0, h / 2.0), -degrees, 1.0);
    let [x0, y0, x1, y1] = b;
    let pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
    let xs = pts.map(|(x, y)| m[0] * x + m[1] * y + m[2]);
    let ys = pts.map(|(x, y)| m[3] * x + m[4] * y + m[5]);
    let min = |v: [f64; 4]| v.into_iter().fold(f64::INFINITY, f64::min);
    let max = |v: [f64; 4]| v.into_iter().fold(f64::NEG_INFINITY, f64::max);
    [round1(min(xs).clamp(0.0, w)), round1(min(ys).clamp(0.0, h)), round1(max(xs).clamp(0.0, w)), round1(max(ys).clamp(0.0, h))]
}

/// [side, side] ink map (1 = ink): long side scaled to ``side``, padded.
pub fn orient_input(gray: &Gray, side: usize) -> ImageF {
    let s = side as f64 / gray.h.max(gray.w) as f64;
    let sw = round_int(gray.w as f64 * s).max(1) as usize;
    let sh = round_int(gray.h as f64 * s).max(1) as usize;
    let small = resize_u8(gray, sw, sh, Interp::Area);
    let mut out = ImageF::new(side, side);
    for y in 0..sh.min(side) {
        for x in 0..sw.min(side) {
            out.data[y * side + x] = 1.0 - small.at(x, y) as f32 / 255.0;
        }
    }
    out
}

/// Reading order plus, with layout heads, the per-line block classes,
/// same-block logits and (when asked for) table heads.
pub struct Layout {
    pub cls: Mat,
    pub same: Mat,
    pub tab: Option<[Mat; 3]>,
}

pub struct Ocr {
    pub backend: Box<dyn Backend>,
    /// for the CPU-side steps of one page (deskew search, line crops)
    pub pool: Option<rayon::ThreadPool>,
    pub vocab: Vocab,
    pub vocab_cjk: Option<Vocab>,
    pub long_side: usize,
    pub height: usize,
    pub opts: OcrOptions,
    pub math: Option<MathReader>,
}

impl Ocr {
    pub fn new(backend: Box<dyn Backend>, meta: Meta, opts: OcrOptions, math: Option<MathReader>) -> Self {
        Ocr {
            backend,
            pool: None,
            vocab: meta.vocab,
            vocab_cjk: meta.vocab_cjk,
            long_side: meta.det_long_side,
            height: meta.rec_height,
            opts,
            math,
        }
    }

    /// The pipeline on ONNX Runtime with the models in ``dir`` (``threads``
    /// intra-op threads per model).
    #[cfg(feature = "ort")]
    pub fn with_ort(dir: &std::path::Path, threads: usize, opts: OcrOptions, math: bool) -> Result<Self> {
        let meta = Meta::load(dir)?;
        let backend = crate::ort_backend::OrtBackend::new(dir, threads, math)?;
        let reader = if math { Some(MathReader::load(dir)?) } else { None };
        Ok(Ocr::new(Box::new(backend), meta, opts, reader).with_threads(threads))
    }

    /// Run the CPU-side steps of a page on ``threads`` threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.pool = (threads > 1).then(|| rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok()).flatten();
        self
    }

    /// The pipeline on the native engine with the models in ``dir``.
    #[cfg(feature = "native")]
    pub fn with_native(dir: &std::path::Path, threads: usize, opts: OcrOptions, math: bool) -> Result<Self> {
        Self::with_native_int8(dir, threads, opts, math, false)
    }

    /// ``with_native``, the recognizer in int8 when ``int8`` (and the CPU has
    /// AVX-512 VNNI).
    #[cfg(feature = "native")]
    pub fn with_native_int8(dir: &std::path::Path, threads: usize, opts: OcrOptions, math: bool, int8: bool) -> Result<Self> {
        let meta = Meta::load(dir)?;
        let backend = crate::native_backend::NativeBackend::with_int8(dir, threads, math, int8)?;
        let reader = if math { Some(MathReader::load(dir)?) } else { None };
        Ok(Ocr::new(Box::new(backend), meta, opts, reader).with_threads(threads))
    }

    fn detect_at(&mut self, gray: &Gray, long_side: usize) -> Result<(Vec<DetBox>, f64)> {
        let (x, s) = page_to_input(gray, long_side);
        let out = self.backend.det(&x)?;
        Ok((decode(&out.data, out.h, out.w, 0.5, 2), s))
    }

    /// Line boxes in page pixels (a second pass at a higher resolution on pages of small text).
    pub fn detect(&mut self, gray: &Gray) -> Result<Vec<DetBox>> {
        let (mut boxes, mut s) = self.detect_at(gray, self.long_side)?;
        if self.opts.adaptive && boxes.len() >= 8 {
            let med_h = median_f32(&boxes.iter().map(|b| b.y1 - b.y0).collect::<Vec<_>>()) as f64;
            if med_h < ADAPTIVE_MIN_H {
                let ls = (ADAPTIVE_MAX_SIDE.min(self.long_side as f64 * ADAPTIVE_TARGET_H / med_h.max(1.0)) as usize) / 32 * 32;
                if ls > self.long_side {
                    (boxes, s) = self.detect_at(gray, ls)?;
                }
            }
        }
        let (w, h) = (gray.w as f32, gray.h as f32);
        for b in boxes.iter_mut() {
            b.x0 = (b.x0 / s as f32).clamp(0.0, w);
            b.x1 = (b.x1 / s as f32).clamp(0.0, w);
            b.y0 = (b.y0 / s as f32).clamp(0.0, h);
            b.y1 = (b.y1 / s as f32).clamp(0.0, h);
        }
        Ok(boxes)
    }

    /// Recognizer inputs of the line boxes (after CenterMask when the models have it).
    pub fn crop_lines(&mut self, gray: &Gray, boxes: &[[f32; 4]], height: usize) -> Result<Vec<ImageF>> {
        if !self.backend.has_mask() {
            use rayon::prelude::*;
            return Ok(match &self.pool {
                Some(p) => p.install(|| boxes.par_iter().map(|b| crop_line(gray, *b, height)).collect()),
                None => boxes.iter().map(|b| crop_line(gray, *b, height)).collect(),
            });
        }
        let wins = crate::line_mask::masked_windows(gray, boxes, self.backend.as_mut())?;
        Ok(wins.iter().map(|(w, b)| crop_line(w, *b, height)).collect())
    }

    /// (text, confidence, tokens) per line; a line under MIN_CONF comes back empty.
    pub fn recognize(&mut self, gray: &Gray, boxes: &[[f32; 4]]) -> Result<Vec<RecLine>> {
        let crops = self.crop_lines(gray, boxes, self.height)?;
        let mut out: Vec<RecLine> = vec![RecLine::empty(0.0); crops.len()];
        let all: Vec<usize> = (0..crops.len()).collect();
        let mut groups = vec![(RecModel::Main, all.clone())];
        if self.backend.has_cjk() && !crops.is_empty() {
            let refs: Vec<&ImageF> = crops.iter().collect();
            let p = self.backend.script(&refs)?;
            let hanzi: Vec<bool> = p.iter().map(|&v| v > 0.5).collect();
            groups = vec![
                (RecModel::Main, all.iter().copied().filter(|&j| !hanzi[j]).collect()),
                (RecModel::Cjk, all.iter().copied().filter(|&j| hanzi[j]).collect()),
            ];
        }
        for (model, sel) in groups {
            if sel.is_empty() {
                continue;
            }
            let refs: Vec<&ImageF> = sel.iter().map(|&j| &crops[j]).collect();
            let res = self.backend.rec(model, &refs)?;
            let vocab = match model {
                RecModel::Main => &self.vocab,
                RecModel::Cjk => self.vocab_cjk.as_ref().ok_or_else(|| OcrError::Other("meta.json has no vocab_cjk".into()))?,
            };
            for (&j, r) in sel.iter().zip(res) {
                let keep: Vec<usize> = (0..r.ids.len()).filter(|&t| r.ids[t] != 0 && (t == 0 || r.ids[t] != r.ids[t - 1])).collect();
                let [x0, _, x1, _] = boxes[j].map(|v| v as f64);
                let (y0, y1) = (boxes[j][1] as f64, boxes[j][3] as f64);
                let h = (y1 - y0).max(1.0);
                let ix0 = (x0 - PAD_X * h).floor();
                let cw = (x1 + PAD_X * h).ceil() - ix0;
                let f = cw / crops[j].w as f64;
                let toks: Vec<Tok> = keep
                    .iter()
                    .map(|&t| {
                        let x = ix0 + (t as f64 + 0.5) * r.stride * f;
                        let id = r.ids[t];
                        // The CJK recognizer's ids mean nothing in self.vocab: its
                        // tokens are characters, like the text-layer tokens.
                        let v = match model {
                            RecModel::Main => TokVal::Id(id),
                            RecModel::Cjk => TokVal::Str(vocab.chars[id as usize - 1].clone()),
                        };
                        Tok { v, x, idx: None }
                    })
                    .collect();
                let conf = if keep.is_empty() {
                    0.0
                } else {
                    let pm: Vec<f32> = keep.iter().map(|&t| r.conf[t]).collect();
                    (pairwise_sum_f32(&pm) / pm.len() as f32) as f64
                };
                out[j] = if conf >= MIN_CONF {
                    RecLine { text: vocab.decode(keep.iter().map(|&t| r.ids[t])), conf, toks }
                } else {
                    RecLine::empty(conf)
                };
            }
        }
        if self.math.is_some() && !crops.is_empty() {
            let refs: Vec<&ImageF> = crops.iter().collect();
            let flags = self.math_flags(&refs)?;
            let idx: Vec<usize> = (0..crops.len()).filter(|&j| flags[j]).collect();
            if !idx.is_empty() {
                let mb: Vec<[f32; 4]> = idx.iter().map(|&j| boxes[j]).collect();
                let reads = self.math_read(gray, &mb)?;
                for (&j, (text, conf)) in idx.iter().zip(reads) {
                    // Only a confident reading with math replaces the line (one
                    // token holds the whole line: it is never split).
                    if text.contains('$') && conf >= crate::math::MATH_MIN_CONF {
                        let xc = (boxes[j][0] + boxes[j][2]) as f64 / 2.0;
                        out[j] = RecLine { text: text.clone(), conf: 1.0, toks: vec![Tok { v: TokVal::Str(text), x: xc, idx: None }] };
                    }
                }
            }
        }
        Ok(out)
    }

    /// Per recognizer crop: does it hold math?
    pub fn math_flags(&mut self, crops: &[&ImageF]) -> Result<Vec<bool>> {
        let mb = self.backend.math().ok_or_else(|| OcrError::Other("this engine has no math models".into()))?;
        Ok(mb.classify(crops)?.into_iter().map(|p| p > crate::math::MATH_MIN).collect())
    }

    /// (text with inline LaTeX, mean token probability) for each box.
    pub fn math_read(&mut self, gray: &Gray, boxes: &[[f32; 4]]) -> Result<Vec<(String, f64)>> {
        let reader = self.math.as_ref().ok_or_else(|| OcrError::Other("math is off".into()))?;
        let crops: Vec<ImageF> = boxes.iter().map(|b| crop_line(gray, *b, reader.height)).collect();
        let mb = self.backend.math().ok_or_else(|| OcrError::Other("this engine has no math models".into()))?;
        Ok(reader.read(mb, &crops)?)
    }

    /// Per line, the x positions of gaps that sit on a column gutter.
    pub fn gutter_cuts(&self, boxes: &[[f32; 4]], rec: &[RecLine]) -> Vec<Vec<f64>> {
        let mut cuts = Vec::with_capacity(boxes.len());
        for (i, (b, r)) in boxes.iter().zip(rec).enumerate() {
            let h = ((b[3] - b[1]) as f64).max(1.0);
            let xs: Vec<f64> = r.toks.iter().filter(|t| !self.is_space(&t.v)).map(|t| t.x).collect();
            let mut mine = vec![];
            for w in xs.windows(2) {
                let (a, c) = (w[0], w[1]);
                if c - a <= GUTTER_GAP * h {
                    continue;
                }
                let (gl, gr) = ((a + 0.35 * h) as f32, (c - 0.35 * h) as f32);
                let tol = (0.5 * h) as f32;
                let near =
                    boxes.iter().enumerate().filter(|(k, o)| *k != i && ((o[2] - gl).abs() < tol || (o[0] - gr).abs() < tol)).count();
                if near >= GUTTER_EDGES {
                    mine.push((a + c) / 2.0);
                }
            }
            cuts.push(mine);
        }
        cuts
    }

    fn is_space(&self, t: &TokVal) -> bool {
        match t {
            TokVal::Str(s) => s == " ",
            TokVal::Id(i) => self.vocab.index.get(" ") == Some(i),
        }
    }

    /// Text of tokens: recognizer ids, or characters (whitespace squashed).
    pub fn decode_toks(&self, toks: &[Tok]) -> String {
        match toks.first().map(|t| &t.v) {
            Some(TokVal::Str(_)) => {
                let s: String = toks
                    .iter()
                    .map(|t| match &t.v {
                        TokVal::Str(s) => s.as_str(),
                        TokVal::Id(_) => "",
                    })
                    .collect();
                squash_spaces(&s)
            }
            _ => self.vocab.decode(toks.iter().filter_map(|t| match t.v {
                TokVal::Id(i) => Some(i),
                TokVal::Str(_) => None,
            })),
        }
    }

    /// A line cut where neighbouring characters are more than ``gap`` line heights
    /// apart (or at the gutter positions ``at``): [(box, text, tokens)].
    pub fn split_line(&self, b: [f32; 4], toks: &[Tok], gap: f64, at: &[f64]) -> Vec<([f32; 4], String, Vec<Tok>)> {
        let h = ((b[3] - b[1]) as f64).max(1.0);
        let mut at = at.to_vec();
        at.sort_by(|a, b| a.total_cmp(b));
        let mut segs: Vec<Vec<Tok>> = vec![vec![]];
        let mut prev: Option<f64> = None;
        for tk in toks {
            let x = tk.x;
            if !self.is_space(&tk.v) {
                if let Some(p) = prev
                    && (x - p > gap * h || at.iter().any(|&c| p < c && c < x))
                {
                    segs.push(vec![]);
                }
                prev = Some(x);
            }
            segs.last_mut().unwrap().push(tk.clone());
        }
        if segs.len() == 1 {
            return vec![(b, self.decode_toks(toks), toks.to_vec())];
        }
        let mut out = vec![];
        for sg in segs {
            let text = crate::py::strip(&self.decode_toks(&sg)).to_string();
            let xs: Vec<f64> = sg.iter().filter(|t| !self.is_space(&t.v)).map(|t| t.x).collect();
            if !text.is_empty() && !xs.is_empty() {
                let lo = xs.iter().copied().fold(f64::INFINITY, f64::min) - 0.35 * h;
                let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 0.35 * h;
                let bx = [(b[0] as f64).max(lo) as f32, b[1], (b[2] as f64).min(hi) as f32, b[3]];
                out.push((bx, text, sg));
            }
        }
        out
    }

    /// Reading order, the layout (with layout heads), and the pairwise order logits.
    pub fn reading_order(
        &mut self,
        boxes: &[[f32; 4]],
        texts: &[String],
        wh: (usize, usize),
    ) -> Result<(Vec<usize>, Option<Layout>, Option<Mat>)> {
        if boxes.len() < 2 || !self.backend.has_order() {
            return Ok((order_xycut(boxes), None, None));
        }
        let f = line_features(boxes, wh, Some(texts));
        let out = self.backend.order(&f, boxes.len())?;
        if let (Some(cls), Some(same)) = (out.cls, out.same) {
            let tab = if self.opts.table_heads { out.tab } else { None };
            let order = decode_blocks_order(&out.a, &same, boxes, Some(&cls));
            return Ok((order, Some(Layout { cls, same, tab }), Some(out.a)));
        }
        Ok((decode_order(&out.a, 4), None, Some(out.a)))
    }

    /// Quarter turns (counter-clockwise) of the page content, from orient.onnx.
    pub fn orientation(&mut self, gray: &Gray) -> Result<u32> {
        if !self.backend.has_orient() {
            return Ok(0);
        }
        let p = self.backend.orient(&orient_input(gray, ORIENT_SIDE))?;
        let k = argmax(&p);
        Ok(if p[k] > ORIENT_MIN { k as u32 } else { 0 })
    }

    /// The page turned upright and straightened: (page, quarter turns, skew degrees).
    pub fn upright(&mut self, gray: &Gray, turn: bool) -> Result<(Gray, u32, f64)> {
        let k = if turn { self.orientation(gray)? } else { 0 };
        let g = if k > 0 { rot90_cw(gray, k) } else { gray.clone() };
        let a = estimate_skew_on(&g, self.pool.as_ref());
        if a.abs() >= DESKEW_MIN { Ok((rotate_page(&g, a), k, a)) } else { Ok((g, k, 0.0)) }
    }

    /// OCR of a page image (``turn``: ask orient.onnx whether the page is turned).
    pub fn ocr(&mut self, gray: &Gray, turn: bool) -> Result<Page> {
        let mut t = Timings::default();
        let t0 = Instant::now();
        let (g, k, skew) = self.upright(gray, turn)?;
        t.since("orient_ms", t0);
        let t0 = Instant::now();
        let boxes = self.detect(&g)?;
        t.since("det_ms", t0);
        let t0 = Instant::now();
        let bx: Vec<[f32; 4]> = boxes.iter().map(|b| b.xyxy()).collect();
        let rec = self.recognize(&g, &bx)?;
        t.since("rec_ms", t0);
        let mut page = self.assemble((g.w, g.h), bx, rec, t, false)?;
        page.rotation = 90 * k as i32;
        page.skew = round_n(skew, 2);
        if skew != 0.0 {
            let wh = (g.w, g.h);
            for ln in page.lines.iter_mut() {
                ln.bbox = unrotate_box(ln.bbox, skew, wh);
            }
            for b in page.blocks.iter_mut() {
                b.bbox = unrotate_box(b.bbox, skew, wh);
            }
        }
        Ok(page)
    }

    /// Lines -> split at wide gaps -> reading order -> markdown. With
    /// ``content_order`` and text-layer tokens on every line, the PDF content
    /// order is kept when it is coherent.
    pub fn assemble(
        &mut self,
        wh: (usize, usize),
        boxes: Vec<[f32; 4]>,
        rec: Vec<RecLine>,
        mut t: Timings,
        content_order: bool,
    ) -> Result<Page> {
        let t0 = Instant::now();
        let cuts = self.gutter_cuts(&boxes, &rec);
        let (mut nb, mut nrec) = (vec![], vec![]);
        for ((b, r), at) in boxes.iter().zip(&rec).zip(&cuts) {
            for (bx, tx, tk) in self.split_line(*b, &r.toks, SPLIT_GAP, at) {
                nb.push(bx);
                nrec.push(RecLine { text: tx, conf: r.conf, toks: tk });
            }
        }
        let (mut boxes, mut rec) = (nb, nrec);
        let mut texts: Vec<String> = rec.iter().map(|r| r.text.clone()).collect();
        let (mut order, mut layout, mut logits) = self.reading_order(&boxes, &texts, wh)?;
        if let Some(l) = &layout
            && l.tab.is_some()
        {
            // The table heads learned one line per cell, but the detector can join
            // cells: split table lines into cells, then run the order model again.
            let table = class_id("Table");
            let is_table: Vec<bool> = (0..boxes.len()).map(|i| argmax(l.cls.row(i)) == table).collect();
            if is_table.iter().any(|&v| v) {
                let (mut nb, mut nrec) = (vec![], vec![]);
                for (i, (b, r)) in boxes.iter().zip(&rec).enumerate() {
                    if is_table[i] {
                        for (bx, tx, tk) in self.split_line(*b, &r.toks, CELL_GAP, &[]) {
                            nb.push(bx);
                            nrec.push(RecLine { text: tx, conf: r.conf, toks: tk });
                        }
                    } else {
                        nb.push(*b);
                        nrec.push(r.clone());
                    }
                }
                if nb.len() != boxes.len() {
                    boxes = nb;
                    rec = nrec;
                    texts = rec.iter().map(|r| r.text.clone()).collect();
                    (order, layout, logits) = self.reading_order(&boxes, &texts, wh)?;
                }
            }
        }
        let mut note = "order=model".to_string();
        if content_order && boxes.len() > 1 {
            let keys: Vec<Option<usize>> = rec.iter().map(|r| r.toks.iter().filter_map(|t| t.idx).min()).collect();
            if keys.iter().all(|k| k.is_some()) {
                let mut pdf: Vec<usize> = (0..keys.len()).collect();
                pdf.sort_by_key(|&i| keys[i]);
                let pb: Vec<[f32; 4]> = pdf.iter().map(|&i| boxes[i]).collect();
                let pl = logits.as_ref().map(|l| l.select(&pdf, &pdf));
                if pdf_order_coherent(&pb, pl.as_ref()) {
                    order = pdf;
                    note = "order=pdf".into();
                }
            }
        }
        t.since("order_ms", t0);
        t.total();
        let lines = order
            .iter()
            .map(|&i| Line { bbox: boxes[i].map(|v| round1(v as f64)), text: rec[i].text.clone(), conf: round_n(rec[i].conf, 4) })
            .collect();
        let mut page = Page::new(lines);
        page.timings = t;
        page.note = note;
        if let Some(l) = &layout {
            page.blocks = page_blocks(&boxes, &texts, &order, &l.cls, &l.same);
            // Inside tables, labels split cells at smaller gaps (CELL_GAP).
            let cells: Cells = (0..boxes.len())
                .map(|i| (i, self.split_line(boxes[i], &rec[i].toks, CELL_GAP, &[]).into_iter().map(|(b, t, _)| (b, t)).collect()))
                .collect();
            page.markdown = Some(page_markdown(&boxes, &texts, &order, &l.cls, &l.same, self.opts.furniture, Some(&cells), l.tab.as_ref()));
        } else if boxes.len() < 2 {
            page.markdown = Some(texts.join("\n"));
        }
        Ok(page)
    }
}
