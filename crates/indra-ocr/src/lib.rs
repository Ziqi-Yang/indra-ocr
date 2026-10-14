//! Indra OCR in Rust: line detection, CTC recognition, reading order, layout
//! and markdown, for images and PDFs.

pub mod backend;
pub mod crop;
pub mod det_post;
pub mod imgproc;
pub mod layout;
pub mod line_mask;
pub mod math;
pub mod models;
pub mod order_post;
#[cfg(feature = "ort")]
pub mod ort_backend;
pub mod pipeline;
pub mod py;
pub mod textlayer;
mod unitab;
pub mod vocab;

pub use backend::Backend;
pub use pipeline::{Line, Ocr, OcrOptions, Page};
