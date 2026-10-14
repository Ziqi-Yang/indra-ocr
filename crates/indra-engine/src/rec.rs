//! The line recognizer (rec.onnx, rec_cjk.onnx): a 32 x 4 patch stem, then
//! blocks of depthwise conv (k 9) -> 1x1 expand + HardSwish -> squeeze-excite
//! (masked mean) -> 1x1 project + residual, then a 1x1 classifier with CTC
//! greedy decoding (argmax and its softmax probability per timestep).
//!
//! The ONNX graph runs a padded batch and masks every block's output beyond
//! each line's length, so a line's result does not depend on its batch. Here
//! each line runs at its own length (no padding), time-major ([T, C]), with
//! lines packed into chunks so the products are large and stay in cache.

use std::path::Path;

use crate::gemm::{Act, Out, PackedB, gemm};
use crate::onnx::{Graph, Node};
use crate::simd::{argmax_row, col_sums, dwconv1d, scale_cols, sigmoid, sum_exp};
use crate::{EngineError, Result};

pub(crate) struct Block {
    dw_w: Vec<f32>, // [K, C]
    dw_b: Vec<f32>,
    expand: PackedB,
    se1_w: Vec<f32>, // [R, E]
    se1_b: Vec<f32>,
    se2_w: Vec<f32>, // [E, R]
    se2_b: Vec<f32>,
    project: PackedB,
}

/// A recognizer's output for one line: per timestep, the best class and its
/// softmax probability (computed only where the CTC path keeps the class, 0
/// elsewhere, unless ``all_conf``).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LineOut {
    pub ids: Vec<u32>,
    pub conf: Vec<f32>,
}

pub struct Recognizer {
    height: usize,
    pw: usize, // patch width (stem stride)
    c: usize,
    e: usize,
    stem: PackedB,
    blocks: Vec<Block>,
    head: PackedB,
    classes: usize,
    // lengths = floor((w + add - sub) / div)
    len_add: i64,
    len_sub: i64,
    len_div: i64,
    /// Softmax probability at every timestep (ORT's ``conf`` output), not
    /// only where the CTC path keeps a class.
    pub all_conf: bool,
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::Model(format!("rec: {}", msg.into()))
}

fn conv_wb<'a>(g: &'a Graph, n: &Node) -> Result<(&'a crate::onnx::Tensor, &'a crate::onnx::Tensor)> {
    Ok((g.arg(n, 1)?, g.arg(n, 2)?))
}

/// The scalar constant input ``i`` of a node.
fn scalar_i64(g: &Graph, n: &Node, i: usize) -> Result<i64> {
    let t = g.arg(n, i)?;
    t.i64.first().copied().or(t.f32.first().map(|&v| v as i64)).ok_or_else(|| bad("constant without a value"))
}

impl Recognizer {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Graph::load(path)?;
        Self::from_graph(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn from_graph(g: &Graph) -> Result<Self> {
        let convs: Vec<&Node> = g.ops("Conv").collect();
        if convs.len() < 7 || (convs.len() - 2) % 5 != 0 {
            return Err(bad(format!("{} Conv nodes: not the recognizer architecture", convs.len())));
        }
        // stem: Conv [C, 1, H, PW] stride [H, PW], then BatchNormalization
        let (sw, sb) = conv_wb(g, convs[0])?;
        let (c, height, pw) = match sw.dims[..] {
            [c, 1, h, pw] => (c, h, pw),
            _ => return Err(bad("stem weight is not [C, 1, H, W]")),
        };
        if convs[0].ints("strides") != Some(&[height as i64, pw as i64][..]) {
            return Err(bad("stem stride is not its kernel size"));
        }
        let bn = g.ops("BatchNormalization").next().ok_or_else(|| bad("no BatchNormalization after the stem"))?;
        let eps = bn.float("epsilon").unwrap_or(1e-5);
        let (gamma, beta, mean, var) = (g.arg(bn, 1)?, g.arg(bn, 2)?, g.arg(bn, 3)?, g.arg(bn, 4)?);
        let k = height * pw;
        let mut w = vec![0f32; c * k];
        let mut b = vec![0f32; c];
        for o in 0..c {
            let s = gamma.f32[o] / (var.f32[o] + eps).sqrt();
            for i in 0..k {
                w[o * k + i] = sw.f32[o * k + i] * s;
            }
            b[o] = (sb.f32[o] - mean.f32[o]) * s + beta.f32[o];
        }
        let stem = PackedB::from_nk(&w, c, k, Some(&b));
        // blocks
        let (blocks, e) = load_blocks(g, &convs[1..convs.len() - 1], c)?;
        // the SE mean divides by max(length, Clip min): check that min is 1
        if let Some(clip) = g.ops("Clip").find(|n| n.inputs.len() > 1 && !n.inputs[1].is_empty()) {
            let t = g.arg(clip, 1)?;
            if t.f32.first().is_some_and(|&v| v != 1.0) {
                return Err(bad("squeeze-excite count is clipped at a value other than 1"));
            }
        }
        let (hw, hb) = conv_wb(g, convs[convs.len() - 1])?;
        let classes = hw.dims[0];
        if hw.dims[1] != c {
            return Err(bad("classifier input is not the block width"));
        }
        let head = PackedB::from_nk(&hw.f32, classes, c, Some(&hb.f32));
        // lengths from widths: Add, Sub, Div on the "widths" input
        let add = g.nodes.iter().find(|n| n.op == "Add" && n.inputs.iter().any(|i| i == "widths")).ok_or_else(|| bad("no widths -> lengths"))?;
        let sub = g.nodes.iter().find(|n| n.op == "Sub" && n.inputs[0] == add.outputs[0]).ok_or_else(|| bad("no widths -> lengths"))?;
        let div = g.nodes.iter().find(|n| n.op == "Div" && n.inputs[0] == sub.outputs[0]).ok_or_else(|| bad("no widths -> lengths"))?;
        let (len_add, len_sub, len_div) = (scalar_i64(g, add, 1)?, scalar_i64(g, sub, 1)?, scalar_i64(g, div, 1)?);
        if len_div <= 0 {
            return Err(bad("lengths divisor"));
        }
        Ok(Recognizer { height, pw, c, e, stem, blocks, head, classes, len_add, len_sub, len_div, all_conf: false })
    }

    pub fn height(&self) -> usize {
        self.height
    }
    pub fn classes(&self) -> usize {
        self.classes
    }
    /// Input columns per output timestep.
    pub fn stride(&self) -> usize {
        self.pw
    }

    /// Valid timesteps of a line ``w`` columns wide.
    pub fn length(&self, w: usize) -> usize {
        let v = w as i64 + self.len_add - self.len_sub;
        v.div_euclid(self.len_div).max(0) as usize
    }

    /// Recognize lines: each ``(pixels, width)`` is a [height, width] row-major
    /// ink map.
    pub fn run(&self, lines: &[(&[f32], usize)]) -> Vec<LineOut> {
        let mut out = vec![LineOut::default(); lines.len()];
        let mut ws = Workspace::default();
        // chunks of whole lines, about CHUNK timesteps each
        const CHUNK: usize = 384;
        let mut i = 0;
        while i < lines.len() {
            let mut j = i;
            let mut t = 0;
            while j < lines.len() && (j == i || t + self.length(lines[j].1) <= CHUNK) {
                t += self.length(lines[j].1);
                j += 1;
            }
            self.run_chunk(&lines[i..j], &mut out[i..j], &mut ws);
            i = j;
        }
        out
    }

    fn run_chunk(&self, lines: &[(&[f32], usize)], out: &mut [LineOut], ws: &mut Workspace) {
        let mut prof = crate::profile::Timer::start();
        let (c, e) = (self.c, self.e);
        let segs: Vec<(usize, usize)> = {
            let mut s = 0;
            lines
                .iter()
                .map(|l| {
                    let t = self.length(l.1);
                    let r = (s, t);
                    s += t;
                    r
                })
                .collect()
        };
        let total: usize = segs.iter().map(|s| s.1).sum();
        if total == 0 {
            return;
        }
        let kp = self.height * self.pw;
        ws.patches.resize(total * kp, 0.0);
        ws.x.resize(total * c, 0.0);
        ws.d.resize(total * c, 0.0);
        ws.ex.resize(total * e, 0.0);
        // stem: patches [T, H * PW] (columns past the line are paper, 0)
        for ((px, w), &(s0, t)) in lines.iter().zip(&segs) {
            for ti in 0..t {
                let row = &mut ws.patches[(s0 + ti) * kp..(s0 + ti + 1) * kp];
                for y in 0..self.height {
                    for xk in 0..self.pw {
                        let col = ti * self.pw + xk;
                        row[y * self.pw + xk] = if col < *w { px[y * w + col] } else { 0.0 };
                    }
                }
            }
        }
        prof.lap("rec.patches");
        gemm(&ws.patches, kp, total, &self.stem, &mut ws.x, c, Act::HardSwish, Out::Store);
        prof.lap("rec.stem");
        run_blocks(&self.blocks, &segs, &mut ws.x, &mut ws.d, &mut ws.ex, c, e, &mut prof);
        // classifier, a few rows at a time, and CTC greedy decoding
        const ROWS: usize = 48;
        let v = self.classes;
        ws.logits.resize(ROWS * v, 0.0);
        for ((&(s0, t), o), _) in segs.iter().zip(out.iter_mut()).zip(lines) {
            o.ids = vec![0; t];
            o.conf = vec![0.0; t];
            let mut prev = u32::MAX;
            for r0 in (0..t).step_by(ROWS) {
                let rows = ROWS.min(t - r0);
                gemm(&ws.x[(s0 + r0) * c..(s0 + r0 + rows) * c], c, rows, &self.head, &mut ws.logits, v, Act::None, Out::Store);
                prof.lap("rec.head");
                for r in 0..rows {
                    let row = &ws.logits[r * v..(r + 1) * v];
                    let (id, m) = argmax_row(row);
                    let id = id as u32;
                    o.ids[r0 + r] = id;
                    if self.all_conf || (id != 0 && id != prev) {
                        o.conf[r0 + r] = 1.0 / sum_exp(row, m);
                    }
                    prev = id;
                }
                prof.lap("rec.decode");
            }
        }
    }
}

/// Residual blocks from their Conv nodes (5 per block: depthwise, expand,
/// squeeze, excite, project): (blocks, expanded width).
pub(crate) fn load_blocks(g: &Graph, convs: &[&Node], c: usize) -> Result<(Vec<Block>, usize)> {
    if convs.len() % 5 != 0 {
        return Err(bad("block convolutions are not in fives"));
    }
    let mut blocks = Vec::with_capacity(convs.len() / 5);
    let mut e = 0;
    for (bi, ns) in convs.chunks_exact(5).enumerate() {
        let (dw, dwb) = conv_wb(g, ns[0])?;
        if dw.dims.len() != 3 || dw.dims[0] != c || dw.dims[1] != 1 || ns[0].int("group") != Some(c as i64) {
            return Err(bad(format!("block {bi}: depthwise conv is {:?}", dw.dims)));
        }
        let kk = dw.dims[2];
        if ns[0].ints("pads") != Some(&[(kk / 2) as i64, (kk / 2) as i64][..]) {
            return Err(bad(format!("block {bi}: depthwise conv is not 'same' padded")));
        }
        // [C, 1, K] -> [K, C]
        let mut dw_w = vec![0f32; kk * c];
        for ch in 0..c {
            for j in 0..kk {
                dw_w[j * c + ch] = dw.f32[ch * kk + j];
            }
        }
        let (ew, eb) = conv_wb(g, ns[1])?;
        let (s1w, s1b) = conv_wb(g, ns[2])?;
        let (s2w, s2b) = conv_wb(g, ns[3])?;
        let (pjw, pjb) = conv_wb(g, ns[4])?;
        e = ew.dims[0];
        let r = s1w.dims[0];
        if ew.dims[1] != c || s1w.dims[1] != e || s2w.dims[..2] != [e, r] || pjw.dims[..2] != [c, e] {
            return Err(bad(format!("block {bi}: unexpected 1x1 conv shapes")));
        }
        blocks.push(Block {
            dw_w,
            dw_b: dwb.f32.clone(),
            expand: PackedB::from_nk(&ew.f32, e, c, Some(&eb.f32)),
            se1_w: s1w.f32.clone(),
            se1_b: s1b.f32.clone(),
            se2_w: s2w.f32.clone(),
            se2_b: s2b.f32.clone(),
            project: PackedB::from_nk(&pjw.f32, c, e, Some(&pjb.f32)),
        });
    }
    Ok((blocks, e))
}

/// Run residual blocks on ``x`` [T, C] holding the lines ``segs`` ((start,
/// length) in rows); ``d`` [T, C] and ``ex`` [T, E] are scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_blocks(blocks: &[Block], segs: &[(usize, usize)], x: &mut [f32], d: &mut [f32], ex: &mut [f32], c: usize, e: usize, prof: &mut crate::profile::Timer) {
    let total: usize = segs.iter().map(|s| s.1).sum();
    let mut mean = vec![0f32; e];
    for b in blocks {
        for &(s0, t) in segs {
            dwconv1d(&x[s0 * c..(s0 + t) * c], t, c, &b.dw_w, &b.dw_b, &mut d[s0 * c..(s0 + t) * c]);
        }
        prof.lap("rec.dwconv");
        gemm(d, c, total, &b.expand, ex, e, Act::HardSwish, Out::Store);
        prof.lap("rec.expand");
        // squeeze-excite per line
        for &(s0, t) in segs {
            let seg = &mut ex[s0 * e..(s0 + t) * e];
            col_sums(seg, e, &mut mean);
            let inv = 1.0 / (t as f32).max(1.0);
            for v in mean.iter_mut() {
                *v *= inv;
            }
            let s = squeeze_excite(&mean, &b.se1_w, &b.se1_b, &b.se2_w, &b.se2_b);
            scale_cols(seg, &s);
        }
        prof.lap("rec.se");
        gemm(ex, e, total, &b.project, x, c, Act::None, Out::Add);
        prof.lap("rec.project");
    }
}

/// s = sigmoid(W2 relu(W1 m + b1) + b2)
fn squeeze_excite(m: &[f32], w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let (r, e) = (b1.len(), m.len());
    let z: Vec<f32> = (0..r).map(|i| (b1[i] + w1[i * e..(i + 1) * e].iter().zip(m).map(|(a, b)| a * b).sum::<f32>()).max(0.0)).collect();
    (0..e).map(|i| sigmoid(b2[i] + w2[i * r..(i + 1) * r].iter().zip(&z).map(|(a, b)| a * b).sum::<f32>())).collect()
}

#[derive(Default)]
struct Workspace {
    patches: Vec<f32>,
    x: Vec<f32>,
    d: Vec<f32>,
    ex: Vec<f32>,
    logits: Vec<f32>,
}
