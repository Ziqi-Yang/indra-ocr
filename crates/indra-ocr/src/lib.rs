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
pub mod order_post;
#[cfg(feature = "pdf")]
pub mod pdfium;
#[cfg(feature = "ort")]
pub mod ort_backend;
pub mod parallel;
pub mod pipeline;
pub mod py;
pub mod pyjson;
pub mod textlayer;
mod unitab;
pub mod vocab;

pub use backend::Backend;
pub use document::{ConvertOptions, Converter, Mode};
pub use io::InputError;
pub use parallel::{Engine, LoadOptions, convert_many, load, usable_cores};
pub use pipeline::{Line, Ocr, OcrOptions, Page};
