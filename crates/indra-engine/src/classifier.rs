//! The line classifiers (script.onnx: does a line hold Chinese characters;
//! mathcls.onnx: does it hold math): 2 x 2 average pool, then 3 x 3 conv +
//! ReLU + 2 x 2 max pool stages, a mean over height, a mean over the line's
//! valid width, and a logistic unit.
//!
//! The convolutions see the zero padding to the right of a line, so a
//! probability depends on how far the line was padded: ``run`` takes that
//! width (the padded batch width ONNX Runtime ran it at).

use std::path::Path;

use crate::cnn::{ConvKind, Feat, conv, conv_kind, pool2};
use crate::gemm::Act;
use crate::onnx::Graph;
use crate::{EngineError, Result};

pub struct LineClassifier {
    avg_first: bool,
    stages: Vec<ConvKind>,
    fc_w: Vec<f32>,
    fc_b: f32,
    /// valid output columns = max(width / div, 1)
    div: i64,
    height: usize,
}

fn bad(m: &str) -> EngineError {
    EngineError::Model(format!("line classifier: {m}"))
}

impl LineClassifier {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Graph::load(path)?;
        Self::from_graph(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn from_graph(g: &Graph) -> Result<Self> {
        let ops: Vec<&str> = g.nodes.iter().map(|n| n.op.as_str()).collect();
        let avg_first = ops.first() == Some(&"AveragePool");
        // Conv, Relu, MaxPool repeated
        let mut stages = vec![];
        for (i, n) in g.nodes.iter().enumerate() {
            if n.op == "Conv" {
                if ops.get(i + 1) != Some(&"Relu") || ops.get(i + 2) != Some(&"MaxPool") {
                    return Err(bad("a conv stage is not Conv, Relu, MaxPool"));
                }
                stages.push(conv_kind(g, n)?);
            }
        }
        if stages.is_empty() {
            return Err(bad("no conv stages"));
        }
        let mean = g.ops("ReduceMean").next().ok_or_else(|| bad("no mean over height"))?;
        if mean.ints("axes") != Some(&[2][..]) {
            return Err(bad("the mean is not over height"));
        }
        let gemm = g.ops("Gemm").next().ok_or_else(|| bad("no Gemm"))?;
        let (w, b) = (g.arg(gemm, 1)?, g.arg(gemm, 2)?);
        if w.dims.first() != Some(&1) {
            return Err(bad("more than one output"));
        }
        let div = g
            .nodes
            .iter()
            .find(|n| n.op == "Div" && n.inputs.iter().any(|i| i.contains("widths") || g.nodes.iter().any(|m| m.outputs[0] == *i && m.inputs.iter().any(|x| x == "widths"))))
            .ok_or_else(|| bad("no widths division"))?;
        let d = g.arg(div, 1)?;
        let div = d.i64.first().copied().or(d.f32.first().map(|&v| v as i64)).ok_or_else(|| bad("widths divisor"))?;
        let height = 1usize << (stages.len() + avg_first as usize);
        Ok(LineClassifier { avg_first, stages, fc_w: w.f32.clone(), fc_b: b.f32[0], div, height })
    }

    /// Input height the stages reduce to one row.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The probability for one line: ``x`` [H, w] row-major, run at ``padded``
    /// columns (zeros right of the line).
    pub fn run(&self, x: &[f32], h: usize, w: usize, padded: usize) -> f32 {
        let mut f = Feat::new(h, padded, 1);
        for y in 0..h {
            f.data[y * padded..y * padded + w].copy_from_slice(&x[y * w..(y + 1) * w]);
        }
        if self.avg_first {
            f = pool2(&f, false);
        }
        for k in &self.stages {
            f = pool2(&conv(&f, k, Act::Relu, None, crate::par::serial()), true);
        }
        // mean over height, then over the valid columns
        let valid = ((w as i64).div_euclid(self.div).max(1) as usize).min(f.w);
        let c = f.c;
        let mut m = vec![0f32; c];
        for t in 0..valid {
            for y in 0..f.h {
                let px = &f.data[(y * f.w + t) * c..(y * f.w + t + 1) * c];
                for (a, &v) in m.iter_mut().zip(px) {
                    *a += v / f.h as f32;
                }
            }
        }
        let z = self.fc_b + m.iter().zip(&self.fc_w).map(|(a, w)| a / valid.max(1) as f32 * w).sum::<f32>();
        crate::simd::sigmoid(z)
    }
}
