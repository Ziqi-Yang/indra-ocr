//! Whole documents (document.py): the PDF text layer where it can be trusted,
//! OCR elsewhere; image files frame by frame.

use std::path::Path;
use std::time::Instant;

use crate::imgproc::Gray;
use crate::io::{InputError, is_pdf, read_images};
use crate::pipeline::{Ocr, Page, RecLine, Timings, Tok, TokVal};
use crate::py::{bidi_ltr, bidi_rtl, is_alpha, is_space, strip};
use crate::textlayer::TextLine;

/// Resolution of the render for the text-layer ink checks.
pub const CHECK_DPI: f64 = 72.0;
/// A page larger than this at the set dpi renders at a lower dpi.
pub const MAX_PAGE_PIXELS: f64 = 100e6;
/// Share of the layer's letters the recognizer must know.
pub const READABLE_MIN: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// the text layer where it passes the checks, OCR elsewhere
    Auto,
    /// never read the text layer
    Ocr,
    /// the text layer even when the checks fail
    Text,
}

impl std::str::FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Mode::Auto),
            "ocr" => Ok(Mode::Ocr),
            "text" => Ok(Mode::Text),
            _ => Err(format!("mode must be auto, ocr or text, not {s:?}")),
        }
    }
}

/// Whether the recognizer can read the script of these text-layer runs.
pub fn readable(ocr: &Ocr, runs: &[TextLine]) -> bool {
    let letters: Vec<char> = runs.iter().flat_map(|r| r.text.chars()).filter(|&c| is_alpha(c)).collect();
    if letters.is_empty() {
        return true;
    }
    let known = letters
        .iter()
        .filter(|&&c| ocr.vocab.has_char(c) || ocr.vocab_cjk.as_ref().is_some_and(|v| v.has_char(c)))
        .count();
    known as f64 >= READABLE_MIN * letters.len() as f64
}

/// Unicode bidi base direction: most strong characters are right-to-left.
pub fn right_to_left(tokens: &[Tok]) -> bool {
    let (mut rtl, mut ltr) = (0, 0);
    for t in tokens {
        if let TokVal::Str(s) = &t.v {
            let mut it = s.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                rtl += bidi_rtl(c) as usize;
                ltr += bidi_ltr(c) as usize;
            }
        }
    }
    rtl > ltr
}

/// Tokens (char, x, content index) per detected line: each character goes to
/// the smallest detected box that contains its center; inside a line each run
/// keeps its PDF order and runs are ordered by x. Returns (tokens per line,
/// characters outside every box).
pub fn assign_runs(runs: &[TextLine], boxes: &[[f32; 4]], k: f64) -> (Vec<Vec<Tok>>, usize) {
    if boxes.is_empty() {
        let lost = runs.iter().flat_map(|r| r.chars.iter()).filter(|(c, _)| !is_space(*c)).count();
        return (vec![], lost);
    }
    // line -> [(run index, tokens)] in first-seen order
    let mut parts: Vec<Vec<(usize, Vec<Tok>)>> = vec![vec![]; boxes.len()];
    let area: Vec<f32> = boxes.iter().map(|b| (b[2] - b[0]) * (b[3] - b[1])).collect();
    let mut lost = 0;
    let tok = |c: char, x: f64, idx: usize| Tok { v: TokVal::Str(c.to_string()), x, idx: Some(idx) };
    for (idx, run) in runs.iter().enumerate() {
        let mut j: Option<usize> = None;
        for &(c, b) in &run.chars {
            if !is_space(c) {
                let (cx, cy) = ((b[0] + b[2]) / 2.0 * k, (b[1] + b[3]) / 2.0 * k);
                let mut best: Option<usize> = None;
                for (i, bx) in boxes.iter().enumerate() {
                    let inside = bx[0] as f64 <= cx && cx <= bx[2] as f64 && bx[1] as f64 <= cy && cy <= bx[3] as f64;
                    if inside && best.is_none_or(|b| area[i] < area[b]) {
                        best = Some(i);
                    }
                }
                let Some(i) = best else {
                    lost += 1;
                    j = None;
                    continue;
                };
                j = Some(i);
                let line = &mut parts[i];
                match line.iter_mut().find(|p| p.0 == idx) {
                    Some(p) => p.1.push(tok(c, cx, idx)),
                    None => line.push((idx, vec![tok(c, cx, idx)])),
                }
            } else if let Some(i) = j {
                let p = parts[i].iter_mut().find(|p| p.0 == idx).expect("the run has a part in its line");
                let x = p.1.last().unwrap().x;
                p.1.push(tok(c, x, idx));
            }
        }
    }
    let toks = parts
        .into_iter()
        .map(|mut line| {
            let all: Vec<Tok> = line.iter().flat_map(|p| p.1.iter().cloned()).collect();
            if right_to_left(&all) {
                let key = |p: &(usize, Vec<Tok>)| -p.1.iter().map(|t| t.x).fold(f64::NEG_INFINITY, f64::max);
                line.sort_by(|a, b| key(a).total_cmp(&key(b)));
            } else {
                let key = |p: &(usize, Vec<Tok>)| p.1.iter().map(|t| t.x).fold(f64::INFINITY, f64::min);
                line.sort_by(|a, b| key(a).total_cmp(&key(b)));
            }
            let mut tk: Vec<Tok> = vec![];
            for (idx, part) in line {
                if let Some(last) = tk.last() {
                    let x = last.x;
                    tk.push(Tok { v: TokVal::Str(" ".into()), x, idx: Some(idx) }); // parts of different runs
                }
                tk.extend(part);
            }
            tk
        })
        .collect();
    (toks, lost)
}

/// Converter options (document.py ``Converter``).
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub dpi: f64,
    pub mode: Mode,
    /// Keep the PDF content order when it is coherent (false: the order model
    /// always orders the lines).
    pub pdf_order: bool,
    /// OCR detected lines that hold no text-layer characters.
    pub image_ocr: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions { dpi: 200.0, mode: Mode::Auto, pdf_order: true, image_ocr: true }
    }
}

/// A source of pages: one image frame, or one page of an open PDF.
pub struct Converter {
    pub ocr: Ocr,
    pub opts: ConvertOptions,
}

/// A page that fails becomes an empty page whose source is "error".
fn safe(r: Result<Page, String>) -> Page {
    r.unwrap_or_else(Page::error)
}

impl Converter {
    pub fn new(ocr: Ocr, opts: ConvertOptions) -> Self {
        Converter { ocr, opts }
    }

    /// OCR of one image page.
    pub fn image_page(&mut self, gray: &Gray) -> Page {
        safe(self.ocr.ocr(gray, true).map_err(|e| format!("{}: {e}", error_kind(&e))))
    }

    /// Every page of a PDF or an image file: (page index, Page). A page that
    /// fails is an "error" page; a file that cannot be opened is an InputError.
    pub fn convert(&mut self, path: &Path, max_pages: usize, password: Option<&str>) -> Result<Vec<(usize, Page)>, InputError> {
        let mut out = vec![];
        self.convert_each(path, max_pages, password, |i, p| out.push((i, p)))?;
        Ok(out)
    }

    /// As ``convert``, handing each page to ``f`` as it is read.
    pub fn convert_each(
        &mut self,
        path: &Path,
        max_pages: usize,
        password: Option<&str>,
        mut f: impl FnMut(usize, Page),
    ) -> Result<(), InputError> {
        if !path.is_file() {
            return Err(InputError(format!("{}: no such file", path.display())));
        }
        if !is_pdf(path) {
            let frames = read_images(path)?;
            let n = if max_pages > 0 { max_pages.min(frames.len()) } else { frames.len() };
            for (i, g) in frames.iter().take(n).enumerate() {
                f(i, self.image_page(g));
            }
            return Ok(());
        }
        #[cfg(feature = "pdf")]
        {
            let doc = crate::pdfium::Document::open(path, password)?;
            let n = doc.page_count();
            let n = if max_pages > 0 { n.min(max_pages) } else { n };
            for i in 0..n {
                f(i, self.pdf_page(&doc, i));
            }
            Ok(())
        }
        #[cfg(not(feature = "pdf"))]
        {
            let _ = password;
            Err(InputError(format!("{}: this build reads no PDFs (feature \"pdf\")", path.display())))
        }
    }
}

fn error_kind(e: &crate::pipeline::OcrError) -> &'static str {
    match e {
        crate::pipeline::OcrError::Backend(_) => "RuntimeError",
        crate::pipeline::OcrError::Model(_) => "ModelError",
        crate::pipeline::OcrError::Other(_) => "RuntimeError",
    }
}

#[cfg(feature = "pdf")]
mod pdf_pages {
    use super::*;
    use crate::pdfium::{Api, Document, PageRef};
    use crate::textlayer::{BROKEN_FONT_MAP, TextLayer, read_text_layer};

    impl Converter {
        /// Pixels per point for a page: the set dpi, lowered for pages so large
        /// that the render would pass MAX_PAGE_PIXELS.
        pub fn page_scale(&self, size: (f64, f64)) -> f64 {
            let k = self.opts.dpi / 72.0;
            k.min((MAX_PAGE_PIXELS / (size.0 * size.1).max(1.0)).sqrt())
        }

        /// One page of an open PDF.
        pub fn pdf_page(&mut self, doc: &Document, index: usize) -> Page {
            safe(self.pdf_page_inner(doc, index))
        }

        fn pdf_page_inner(&mut self, doc: &Document, index: usize) -> Result<Page, String> {
            let mut t = Timings::default();
            let mut note = "mode=ocr".to_string();
            let mut turn = true;
            if self.opts.mode != Mode::Ocr {
                let t0 = Instant::now();
                let s = CHECK_DPI / 72.0;
                let tl = doc
                    .with_page(index, |api, page| {
                        let g = page.render(api, s);
                        read_text_layer(api, page, &g, s)
                    })
                    .and_then(|r| r)
                    .map_err(|e| format!("PdfiumError: {e}"))?;
                t.since("text_ms", t0);
                // A layer the recognizer cannot read beats OCR, unless its codes are junk.
                let unreadable = tl.reason != BROKEN_FONT_MAP && !readable(&self.ocr, &tl.lines);
                if !tl.lines.is_empty() && (tl.trusted || self.opts.mode == Mode::Text || unreadable) {
                    return self.text_page(doc, index, &tl, t);
                }
                note = tl.reason.clone();
                // A layer with a broken font map still places its glyphs: when most
                // run upright, the page is upright (orient.onnx is not asked).
                turn = !(tl.reason == BROKEN_FONT_MAP && (tl.stats.rotated as f64) < 0.5 * tl.stats.chars as f64);
            }
            let t0 = Instant::now();
            let gray = self.render(doc, index)?;
            t.since("render_ms", t0);
            let mut out = self.ocr.ocr(&gray, turn).map_err(|e| format!("{}: {e}", error_kind(&e)))?;
            let mut tm = t;
            for (k, v) in out.timings.0.drain(..) {
                tm.set(&k, v);
            }
            tm.total();
            out.timings = tm;
            out.note = format!("{note}; {}", out.note);
            Ok(out)
        }

        fn render(&self, doc: &Document, index: usize) -> Result<Gray, String> {
            doc.with_page(index, |api: &Api, page: PageRef| {
                let k = self.page_scale(page.size(api));
                page.render(api, k)
            })
            .map_err(|e| format!("PdfiumError: {e}"))
        }

        fn text_page(&mut self, doc: &Document, index: usize, tl: &TextLayer, mut t: Timings) -> Result<Page, String> {
            let t0 = Instant::now();
            let (gray, k) = doc
                .with_page(index, |api, page| {
                    let k = self.page_scale(page.size(api));
                    (page.render(api, k), k)
                })
                .map_err(|e| format!("PdfiumError: {e}"))?;
            t.since("render_ms", t0);
            let ek = |e: crate::pipeline::OcrError| format!("{}: {e}", error_kind(&e));
            let t0 = Instant::now();
            let boxes: Vec<[f32; 4]> = self.ocr.detect(&gray).map_err(ek)?.iter().map(|b| b.xyxy()).collect();
            t.since("det_ms", t0);
            let (toks, lost) = assign_runs(&tl.lines, &boxes, k);
            let mut rec: Vec<Option<RecLine>> = vec![None; boxes.len()];
            let mut empty = vec![];
            for (i, tk) in toks.into_iter().enumerate() {
                let text = if tk.is_empty() { String::new() } else { self.ocr.decode_toks(&tk) };
                if text.is_empty() {
                    empty.push(i);
                } else {
                    rec[i] = Some(RecLine { text, conf: 1.0, toks: tk });
                }
            }
            let mut source = "text";
            if !empty.is_empty() && self.opts.image_ocr && readable(&self.ocr, &tl.lines) {
                let t0 = Instant::now();
                let eb: Vec<[f32; 4]> = empty.iter().map(|&i| boxes[i]).collect();
                for (&i, r) in empty.iter().zip(self.ocr.recognize(&gray, &eb).map_err(ek)?) {
                    if !strip(&r.text).is_empty() {
                        rec[i] = Some(r);
                        source = "hybrid";
                    }
                }
                t.since("rec_ms", t0);
            }
            if self.ocr.math.is_some() {
                let t0 = Instant::now();
                self.read_math(&gray, &boxes, &mut rec).map_err(ek)?;
                t.since("math_ms", t0);
            }
            let keep: Vec<usize> = (0..rec.len()).filter(|&i| rec[i].is_some()).collect();
            let kb: Vec<[f32; 4]> = keep.iter().map(|&i| boxes[i]).collect();
            let kr: Vec<RecLine> = keep.iter().map(|&i| rec[i].clone().unwrap()).collect();
            let mut out = self.ocr.assemble((gray.w, gray.h), kb, kr, t, self.opts.pdf_order).map_err(ek)?;
            out.source = source.into();
            if lost > 0 {
                out.note += &format!("; {lost} chars outside detected lines");
            }
            Ok(out)
        }

        /// Text-layer lines that hold math are read by MathRec instead (the layer
        /// has the symbols but not the LaTeX); a line keeps its place in the PDF
        /// content order (its first character's index).
        fn read_math(&mut self, gray: &Gray, boxes: &[[f32; 4]], rec: &mut [Option<RecLine>]) -> crate::pipeline::Result<()> {
            let idx: Vec<usize> = (0..rec.len())
                .filter(|&i| rec[i].as_ref().is_some_and(|r| r.toks.first().is_some_and(|t| matches!(t.v, TokVal::Str(_)) && t.idx.is_some())))
                .collect();
            if idx.is_empty() {
                return Ok(());
            }
            let h = self.ocr.height;
            let crops: Vec<crate::imgproc::ImageF> = idx.iter().map(|&i| crate::crop::crop_line(gray, boxes[i], h)).collect();
            let refs: Vec<&crate::imgproc::ImageF> = crops.iter().collect();
            let flags = self.ocr.math_flags(&refs)?;
            let math: Vec<usize> = idx.iter().zip(flags).filter(|(_, f)| *f).map(|(i, _)| *i).collect();
            if math.is_empty() {
                return Ok(());
            }
            let mb: Vec<[f32; 4]> = math.iter().map(|&i| boxes[i]).collect();
            for (&i, (text, conf)) in math.iter().zip(self.ocr.math_read(gray, &mb)?) {
                // A reading without math, or not confident, leaves the layer's text.
                if text.contains('$') && conf >= crate::math::MATH_MIN_CONF {
                    let r = rec[i].as_ref().unwrap();
                    let first = r.toks.iter().filter_map(|t| t.idx).min();
                    let xc = (boxes[i][0] + boxes[i][2]) as f64 / 2.0;
                    rec[i] = Some(RecLine { text: text.clone(), conf: 1.0, toks: vec![Tok { v: TokVal::Str(text), x: xc, idx: first }] });
                }
            }
            Ok(())
        }
    }
}
