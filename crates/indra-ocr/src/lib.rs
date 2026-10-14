//! Indra OCR in Rust: line detection, CTC recognition, reading order, layout
//! and markdown, for images and PDFs.

pub mod backend;
pub mod crop;
pub mod det_post;
pub mod document;
pub mod imgproc;
pub mod io;
pub mod layout;
pub mod line_mask;
pub mod math;
pub mod models;
#[cfg(feature = "native")]
pub mod native_backend;
pub mod order_post;
#[cfg(feature = "ort")]
pub mod ort_backend;
pub mod parallel;
#[cfg(feature = "pdf")]
pub mod pdfium;
pub mod pipeline;
pub mod py;
pub mod pyjson;
pub mod textlayer;
mod unitab;
pub mod vocab;

pub use backend::Backend;

/// Lines per recognizer / classifier batch (pipeline.py: ``batch=64``).
pub const REC_BATCH: usize = 64;

pub(crate) fn ort_backend_batch() -> usize {
    REC_BATCH
}
pub use document::{ConvertOptions, Converter, Mode};
pub use io::InputError;
pub use parallel::{Engine, LoadOptions, convert_many, load, physical_cores, usable_cores};
pub use pipeline::{Line, Ocr, OcrOptions, Page};
