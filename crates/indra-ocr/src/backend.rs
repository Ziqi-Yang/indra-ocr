//! What the pipeline needs from an inference engine: one call per model, on
//! ragged inputs (the engine decides how to batch and pad).

use crate::imgproc::ImageF;
use crate::order_post::Mat;

pub type Result<T> = std::result::Result<T, BackendError>;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct BackendError(pub String);

impl BackendError {
    pub fn new(e: impl std::fmt::Display) -> Self {
        BackendError(e.to_string())
    }
}

/// Detector output: [3, h, w] (core logit, top distance, bottom distance).
#[derive(Clone, Debug)]
pub struct DetOut {
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

/// Recognizer output of one line: the best class and its softmax probability
/// at each valid timestep, and the input columns per timestep.
#[derive(Clone, Debug, Default)]
pub struct RecOut {
    pub ids: Vec<u32>,
    pub conf: Vec<f32>,
    pub stride: f64,
}

/// Order model output for N lines: pairwise order logits [N, N], and with
/// layout heads the block classes [N, 10], same-block logits [N, N] and table
/// heads (same row / column / cell) [3][N, N].
#[derive(Clone, Debug)]
pub struct OrderOut {
    pub a: Mat,
    pub cls: Option<Mat>,
    pub same: Option<Mat>,
    pub tab: Option<[Mat; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecModel {
    /// rec.onnx: Latin, Cyrillic, Greek, Devanagari, Bengali
    Main,
    /// rec_cjk.onnx: lines with Chinese characters
    Cjk,
}

/// Math models (math.py): a line classifier, and an encoder with an
/// autoregressive decoder that reads one batch of line crops at a time.
pub trait MathBackend: Send {
    /// Math probability per recognizer line crop.
    fn classify(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>>;
    /// Encode a batch of math crops (``height`` high) and reset the decoder caches.
    fn begin(&mut self, lines: &[&ImageF]) -> Result<()>;
    /// One greedy decoder step for the current batch: logits [B, n_tokens] for
    /// ``tokens`` at position ``pos``.
    fn step(&mut self, tokens: &[i64], pos: i64) -> Result<Vec<f32>>;
}

/// An inference engine for the Indra OCR models.
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    /// The detector on one padded page ink map.
    fn det(&mut self, x: &ImageF) -> Result<DetOut>;
    /// A recognizer on line crops (``rec_height`` high, any widths).
    fn rec(&mut self, model: RecModel, lines: &[&ImageF]) -> Result<Vec<RecOut>>;
    fn has_cjk(&self) -> bool;
    /// Chinese-character probability per line crop (script.onnx).
    fn script(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>>;
    fn has_orient(&self) -> bool;
    /// Quarter-turn probabilities of a page thumbnail (orient.onnx).
    fn orient(&mut self, x: &ImageF) -> Result<[f32; 4]>;
    fn has_order(&self) -> bool;
    /// The order model on [N, 16] line features.
    fn order(&mut self, feats: &[f32], n: usize) -> Result<OrderOut>;
    /// CenterMask keep probabilities of line windows (mask.onnx), [B, 1, H, W] in and out.
    fn has_mask(&self) -> bool {
        false
    }
    fn mask(&mut self, _x: &[f32], _b: usize, _h: usize, _w: usize) -> Result<Vec<f32>> {
        Err(BackendError("this engine has no mask model".into()))
    }
    fn math(&mut self) -> Option<&mut dyn MathBackend> {
        None
    }
}
