//! Math lines (math.py): a classifier finds lines that hold math, and MathRec
//! reads them as text with inline LaTeX ("we have $x _ { i } \leq 1$"), one
//! token per greedy decoder step.

use std::path::Path;

use crate::backend::{MathBackend, Result};
use crate::crop::width_batches;
use crate::imgproc::ImageF;
use crate::models::ModelError;
use crate::py::{pairwise_sum_f32, pairwise_sum_f64, squash_spaces};

/// Classifier probability above which a line is read as math.
pub const MATH_MIN: f32 = 0.5;
/// A math reading replaces a line only at this mean token probability or more.
pub const MATH_MIN_CONF: f64 = 0.9;
/// Decoding stops when the mean token probability is this far below
/// MATH_MIN_CONF after 8 tokens.
pub const EARLY_STOP_MARGIN: f64 = 0.1;
pub const BOS: i64 = 1;
pub const EOS: i64 = 2;
pub const N_SPECIAL: i64 = 4;

/// Tokens -> text; inside math (between "$" tokens) they are joined by spaces.
pub fn detokenize(tokens: &[&str]) -> String {
    let (mut out, mut math): (Vec<String>, Vec<&str>) = (vec![], vec![]);
    let mut in_math = false;
    for &t in tokens {
        if t == "$" {
            if in_math {
                out.push(format!("${}$", math.join(" ")));
                math.clear();
            }
            in_math = !in_math;
        } else if in_math {
            math.push(t);
        } else {
            out.push(t.to_string());
        }
    }
    if in_math && !math.is_empty() {
        out.push(format!("${}$", math.join(" ")));
    }
    squash_spaces(&out.concat())
}

/// The decoding settings and token table of the math models (math_meta.json).
#[derive(Clone, Debug)]
pub struct MathReader {
    pub height: usize,
    pub max_tokens: usize,
    pub itos: Vec<String>,
}

impl MathReader {
    pub fn load(dir: &Path) -> std::result::Result<Self, ModelError> {
        let p = dir.join("math_meta.json");
        let s = std::fs::read_to_string(&p).map_err(|e| ModelError(format!("{}: {e}", p.display())))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| ModelError(format!("{}: {e}", p.display())))?;
        let cfg = &v["config"];
        let get = |k: &str| cfg[k].as_u64().map(|x| x as usize).ok_or_else(|| ModelError(format!("{}: no config.{k}", p.display())));
        let mut itos = vec![String::new(); N_SPECIAL as usize];
        for t in v["tokens"].as_array().ok_or_else(|| ModelError(format!("{}: no tokens", p.display())))? {
            itos.push(t.as_str().unwrap_or_default().to_string());
        }
        Ok(MathReader { height: get("height")?, max_tokens: get("max_tokens")?, itos })
    }

    /// (text with inline LaTeX, mean token probability) for each crop
    /// (``height`` high).
    pub fn read(&self, mb: &mut dyn MathBackend, crops: &[ImageF]) -> Result<Vec<(String, f64)>> {
        let mut out = vec![(String::new(), 0.0); crops.len()];
        // Few, wide batches: each batch is one decoding loop of up to max_tokens
        // steps, and a step costs about the same for 8 lines as for 40.
        let widths: Vec<usize> = crops.iter().map(|c| c.w).collect();
        for idx in width_batches(&widths, 64, 4.0, 64, 16 * 64 * 640) {
            let part: Vec<&ImageF> = idx.iter().map(|&j| &crops[j]).collect();
            mb.begin(&part)?;
            let b = idx.len();
            let mut tok = vec![BOS; b];
            let mut seqs: Vec<Vec<i64>> = vec![vec![]; b];
            let mut probs: Vec<Vec<f64>> = vec![vec![]; b];
            let mut done = vec![false; b];
            for t in 0..self.max_tokens - 1 {
                let logits = mb.step(&tok, t as i64)?;
                let v = logits.len() / b;
                for k in 0..b {
                    let row = &logits[k * v..(k + 1) * v];
                    let am = crate::py::argmax(row);
                    let mx = row[am];
                    let e: Vec<f32> = row.iter().map(|&l| (l - mx).exp()).collect();
                    let top = 1.0f32 / pairwise_sum_f32(&e);
                    tok[k] = am as i64;
                    if done[k] {
                        continue;
                    }
                    probs[k].push(top as f64);
                    let n = probs[k].len();
                    if tok[k] == EOS {
                        done[k] = true;
                    } else if n >= 8 && (pairwise_sum_f64(&probs[k]) / n as f64) < MATH_MIN_CONF - EARLY_STOP_MARGIN {
                        done[k] = true; // rejected: keeps a low mean probability
                    } else if tok[k] >= N_SPECIAL {
                        seqs[k].push(tok[k]);
                    }
                }
                if done.iter().all(|&d| d) {
                    break;
                }
            }
            for (k, &j) in idx.iter().enumerate() {
                let toks: Vec<&str> = seqs[k].iter().map(|&i| self.itos.get(i as usize).map_or("", |s| s.as_str())).collect();
                let conf = if probs[k].is_empty() { 0.0 } else { pairwise_sum_f64(&probs[k]) / probs[k].len() as f64 };
                out[j] = (detokenize(&toks), conf);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokenize_math_spans() {
        assert_eq!(detokenize(&["w", "e", " ", "$", "x", "_", "{", "i", "}", "$"]), "we $x _ { i }$");
        assert_eq!(detokenize(&["$", "\\alpha"]), "$\\alpha$");
    }
}
