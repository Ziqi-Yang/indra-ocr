//! MathRec (math_enc.onnx + math_dec_step.onnx): reads a line as tokens with
//! inline LaTeX.
//!
//! Encoder: three 3 x 3 stride-2 convs + GELU, an (H/8) x 1 conv to one row,
//! BatchNorm + GELU, the recognizer's residual blocks, then pre-norm
//! self-attention layers with ALiBi distance biases; memory = output +
//! sinusoidal positions.
//!
//! Decoder (one token per step, per line): token + position embeddings, then
//! pre-norm layers of self-attention over the step's cache, cross-attention
//! over the memory, and a GELU feed-forward; a final norm and the vocabulary.
//!
//! The front convolutions see the zero padding right of a line, so a line runs
//! at its batch's padded width (as ONNX Runtime ran it); everything after the
//! front runs on the line's own length.

use std::path::Path;

use crate::cnn::{ConvKind, Feat, conv, conv_kind};
use crate::gemm::{Act, Out, PackedB, gemm};
use crate::onnx::{Graph, Node, Tensor};
use crate::rec::{Block, load_blocks, run_blocks};
use crate::simd::gelu;
use crate::{EngineError, Result};

fn bad(m: impl Into<String>) -> EngineError {
    EngineError::Model(format!("math: {}", m.into()))
}

struct Linear {
    w: PackedB,
}

impl Linear {
    /// From a MatMul weight [K, N] and bias.
    fn kn(w: &Tensor, b: &[f32]) -> Self {
        Linear { w: PackedB::from_kn(&w.f32, w.dims[0], w.dims[1], Some(b)) }
    }
    /// From rows [r0, r1) of an [N, K] weight (a packed in_proj) and bias.
    fn rows(w: &Tensor, b: &Tensor, r0: usize, r1: usize) -> Self {
        let k = w.dims[1];
        Linear { w: PackedB::from_nk(&w.f32[r0 * k..r1 * k], r1 - r0, k, Some(&b.f32[r0..r1])) }
    }
    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * self.w.n];
        gemm(x, self.w.k, n, &self.w, &mut y, self.w.n, Act::None, Out::Store);
        y
    }
    fn add_into(&self, x: &[f32], n: usize, y: &mut [f32]) {
        gemm(x, self.w.k, n, &self.w, y, self.w.n, Act::None, Out::Add);
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl Norm {
    fn new(g: &Graph, n: &Node) -> Result<Norm> {
        Ok(Norm { g: g.arg(n, 1)?.f32.clone(), b: g.arg(n, 2)?.f32.clone(), eps: n.float("epsilon").unwrap_or(1e-5) })
    }
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let d = self.g.len();
        let mut y = vec![0f32; x.len()];
        for (row, out) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            for i in 0..d {
                out[i] = (row[i] - mean) * inv * self.g[i] + self.b[i];
            }
        }
        y
    }
}

/// Softmax(q K^T * scale + bias) V for one head: q [d], K, V [n, stride] at
/// column ``off``.
#[allow(clippy::too_many_arguments)]
fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, stride: usize, off: usize, scale: f32, bias: impl Fn(usize) -> f32, out: &mut [f32]) {
    let d = q.len();
    let mut s: Vec<f32> = (0..n)
        .map(|j| {
            let kr = &k[j * stride + off..j * stride + off + d];
            q.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale + bias(j)
        })
        .collect();
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for e in s.iter_mut() {
        *e = (*e - m).exp();
        sum += *e;
    }
    out.fill(0.0);
    for (j, &p) in s.iter().enumerate() {
        let vr = &v[j * stride + off..j * stride + off + d];
        for (o, &x) in out.iter_mut().zip(vr) {
            *o += p / sum * x;
        }
    }
}

struct EncLayer {
    ln1: Norm,
    qkv: Linear,
    out: Linear,
    ln2: Norm,
    ff1: Linear,
    ff2: Linear,
}

pub struct MathEncoder {
    front: Vec<ConvKind>,
    collapse: PackedB, // the (H/8) x 1 conv, BatchNorm folded in
    collapse_h: usize,
    blocks: Vec<Block>,
    c: usize,
    e: usize,
    layers: Vec<EncLayer>,
    slopes: Vec<f32>,
    heads: usize,
    len_add: i64,
    len_sub: i64,
    len_div: i64,
    height: usize,
}

struct DecLayer {
    ln1: Norm,
    qkv: Linear,
    out1: Linear,
    ln2: Norm,
    q2: Linear,
    k2: Linear,
    v2: Linear,
    out2: Linear,
    ln3: Norm,
    ff1: Linear,
    ff2: Linear,
}

pub struct MathDecoder {
    tok: Tensor,
    pos: Tensor,
    layers: Vec<DecLayer>,
    ln: Norm,
    head: Linear,
    heads: usize,
    d: usize,
}

/// Weighted MatMuls in graph order: (MatMul node, its bias Add's constant).
fn linears(g: &Graph) -> Result<Vec<(&Tensor, &Tensor)>> {
    g.ops("MatMul")
        .filter(|n| g.tensors.contains_key(n.inputs[1].as_str()))
        .map(|n| {
            let add = g
                .nodes
                .iter()
                .find(|a| a.op == "Add" && a.inputs.iter().any(|i| i == &n.outputs[0]))
                .ok_or_else(|| bad("a MatMul without a bias"))?;
            let b = add.inputs.iter().find_map(|i| g.tensors.get(i.as_str())).ok_or_else(|| bad("a bias that is not a constant"))?;
            Ok((g.arg(n, 1)?, b))
        })
        .collect()
}

impl MathEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Graph::load(path)?;
        Self::from_graph(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn from_graph(g: &Graph) -> Result<Self> {
        let convs: Vec<&Node> = g.ops("Conv").collect();
        // front: 2-D convs until the one with a (H, 1) kernel
        let ci = convs
            .iter()
            .position(|n| g.arg(n, 1).is_ok_and(|w| w.dims.len() == 4 && w.dims[3] == 1 && w.dims[2] > 1))
            .ok_or_else(|| bad("no column conv"))?;
        let front = convs[..ci].iter().map(|n| conv_kind(g, n)).collect::<Result<Vec<_>>>()?;
        let (cw, cb) = (g.arg(convs[ci], 1)?, g.arg(convs[ci], 2)?);
        let (c, cin, kh) = (cw.dims[0], cw.dims[1], cw.dims[2]);
        let bn = g.ops("BatchNormalization").next().ok_or_else(|| bad("no BatchNormalization"))?;
        let eps = bn.float("epsilon").unwrap_or(1e-5);
        let (gm, bt, mn, vr) = (g.arg(bn, 1)?, g.arg(bn, 2)?, g.arg(bn, 3)?, g.arg(bn, 4)?);
        // [C, Cin, KH, 1] -> rows (kh, cin), BatchNorm folded
        let kk = kh * cin;
        let mut w = vec![0f32; c * kk];
        let mut b = vec![0f32; c];
        for o in 0..c {
            let s = gm.f32[o] / (vr.f32[o] + eps).sqrt();
            for i in 0..kh {
                for ch in 0..cin {
                    w[o * kk + i * cin + ch] = cw.f32[(o * cin + ch) * kh + i] * s;
                }
            }
            b[o] = (cb.f32[o] - mn.f32[o]) * s + bt.f32[o];
        }
        let collapse = PackedB::from_nk(&w, c, kk, Some(&b));
        let (blocks, e) = load_blocks(g, &convs[ci + 1..], c)?;
        // attention layers
        let lns: Vec<&Node> = g.ops("LayerNormalization").collect();
        let lin = linears(g)?;
        if !lns.len().is_multiple_of(2) || lin.len() != 2 * lns.len() {
            return Err(bad("unexpected encoder attention layers"));
        }
        let layers = (0..lns.len() / 2)
            .map(|l| {
                let m = &lin[4 * l..4 * l + 4];
                Ok(EncLayer {
                    ln1: Norm::new(g, lns[2 * l])?,
                    qkv: Linear::kn(m[0].0, &m[0].1.f32),
                    out: Linear::kn(m[1].0, &m[1].1.f32),
                    ln2: Norm::new(g, lns[2 * l + 1])?,
                    ff1: Linear::kn(m[2].0, &m[2].1.f32),
                    ff2: Linear::kn(m[3].0, &m[3].1.f32),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // ALiBi slopes: a constant [1, heads, 1, 1] times |i - j|
        let slopes = g
            .ops("Mul")
            .find_map(|n| n.inputs.iter().find_map(|i| g.tensors.get(i.as_str()).filter(|t| t.dims.len() == 4 && t.len() > 1)))
            .ok_or_else(|| bad("no distance slopes"))?
            .f32
            .clone();
        let heads = slopes.len();
        // lengths from widths
        let add =
            g.nodes.iter().find(|n| n.op == "Add" && n.inputs.iter().any(|i| i == "widths")).ok_or_else(|| bad("no widths -> lengths"))?;
        let sub = g.nodes.iter().find(|n| n.op == "Sub" && n.inputs[0] == add.outputs[0]).ok_or_else(|| bad("no widths -> lengths"))?;
        let div = g.nodes.iter().find(|n| n.op == "Div" && n.inputs[0] == sub.outputs[0]).ok_or_else(|| bad("no widths -> lengths"))?;
        let sc = |n: &Node| -> Result<i64> {
            let t = g.arg(n, 1)?;
            t.i64.first().copied().ok_or_else(|| bad("lengths constant"))
        };
        let height = kh << front.len();
        Ok(MathEncoder {
            front,
            collapse,
            collapse_h: kh,
            blocks,
            c,
            e,
            layers,
            slopes,
            heads,
            len_add: sc(add)?,
            len_sub: sc(sub)?,
            len_div: sc(div)?,
            height,
        })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Memory [T, C] for one line ``x`` [H, w], run at ``padded`` columns.
    pub fn encode(&self, x: &[f32], w: usize, padded: usize) -> Vec<f32> {
        let h = self.height;
        let mut f = Feat::new(h, padded, 1);
        for y in 0..h {
            f.data[y * padded..y * padded + w].copy_from_slice(&x[y * w..(y + 1) * w]);
        }
        for k in &self.front {
            f = conv(&f, k, Act::None, None, crate::par::serial());
            gelu(&mut f.data);
        }
        // the column conv: one patch (kh rows x channels) per column
        let (tw, cin, c, e) = (f.w, f.c, self.c, self.e);
        let t_all = tw;
        let len = ((w as i64 + self.len_add - self.len_sub).div_euclid(self.len_div)).clamp(1, t_all as i64) as usize;
        let kk = self.collapse_h * cin;
        let mut patches = vec![0f32; len * kk];
        for t in 0..len {
            for i in 0..self.collapse_h {
                patches[t * kk + i * cin..t * kk + (i + 1) * cin].copy_from_slice(&f.data[(i * tw + t) * cin..(i * tw + t + 1) * cin]);
            }
        }
        let mut xs = vec![0f32; len * c];
        gemm(&patches, kk, len, &self.collapse, &mut xs, c, Act::None, Out::Store);
        gelu(&mut xs);
        let (mut d, mut ex) = (vec![0f32; len * c], vec![0f32; len * e]);
        let mut prof = crate::profile::Timer::start();
        run_blocks(&self.blocks, &[(0, len)], &mut xs, &mut d, &mut ex, c, e, &mut prof);
        // attention layers with distance biases
        let hd = c / self.heads;
        let scale = (hd as f32).powf(-0.5);
        for l in &self.layers {
            let a = l.ln1.apply(&xs);
            let qkv = l.qkv.apply(&a, len);
            let mut att = vec![0f32; len * c];
            for hh in 0..self.heads {
                let slope = self.slopes[hh];
                for i in 0..len {
                    let q = &qkv[i * 3 * c + hh * hd..i * 3 * c + (hh + 1) * hd];
                    let bias = |j: usize| -(slope * (i as f32 - j as f32).abs());
                    attend(q, &qkv[c..], &qkv[2 * c..], len, 3 * c, hh * hd, scale, bias, &mut att[i * c + hh * hd..i * c + (hh + 1) * hd]);
                }
            }
            l.out.add_into(&att, len, &mut xs);
            let b = l.ln2.apply(&xs);
            let mut ff = l.ff1.apply(&b, len);
            gelu(&mut ff);
            l.ff2.add_into(&ff, len, &mut xs);
        }
        // + sinusoidal positions (even channels sin, odd cos)
        for t in 0..len {
            for i in 0..c / 2 {
                let div = 10000f32.powf((2 * i) as f32 / c as f32);
                let a = t as f32 / div;
                xs[t * c + 2 * i] += a.sin();
                xs[t * c + 2 * i + 1] += a.cos();
            }
        }
        xs
    }
}

impl MathDecoder {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Graph::load(path)?;
        Self::from_graph(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn from_graph(g: &Graph) -> Result<Self> {
        let gathers: Vec<&Node> = g.ops("Gather").filter(|n| g.tensors.contains_key(n.inputs[0].as_str())).collect();
        let tok = g.tensor(&gathers.iter().find(|n| n.inputs[1] == "token").ok_or_else(|| bad("no token embedding"))?.inputs[0])?.clone();
        let pos = g.tensor(&gathers.iter().find(|n| n.inputs[1] == "pos").ok_or_else(|| bad("no position embedding"))?.inputs[0])?.clone();
        let d = tok.dims[1];
        let lns: Vec<&Node> = g.ops("LayerNormalization").collect();
        let lin = linears(g)?;
        let n_layers = (lns.len() - 1) / 3;
        if lns.len() != 3 * n_layers + 1 || lin.len() != 5 * n_layers + 1 {
            return Err(bad("unexpected decoder layers"));
        }
        // cross-attention in_proj weights [3d, d] (+ bias [3d]), one per layer
        let in_proj: Vec<(&Tensor, &Tensor)> = g
            .ops("Slice")
            .filter_map(|n| g.tensors.get(n.inputs[0].as_str()).filter(|t| t.dims == [3 * d, d]).map(|w| (n, w)))
            .map(|(n, w)| {
                // the bias slice that follows it
                let pos = g.nodes.iter().position(|m| std::ptr::eq(m, n)).unwrap();
                let b = g.nodes[pos..]
                    .iter()
                    .find_map(|m| if m.op == "Slice" { g.tensors.get(m.inputs[0].as_str()).filter(|t| t.dims == [3 * d]) } else { None });
                b.map(|b| (w, b)).ok_or_else(|| bad("in_proj without a bias"))
            })
            .collect::<Result<Vec<_>>>()?;
        if in_proj.len() != n_layers {
            return Err(bad("cross-attention weights per layer"));
        }
        let heads = g
            .nodes
            .iter()
            .filter(|n| n.op == "Concat" && n.inputs.len() == 4)
            .find_map(|n| g.tensors.get(n.inputs[2].as_str()).and_then(|t| t.i64.first().copied()))
            .ok_or_else(|| bad("no head count"))? as usize;
        let layers = (0..n_layers)
            .map(|l| {
                let m = &lin[5 * l..5 * l + 5];
                let (w, b) = in_proj[l];
                Ok(DecLayer {
                    ln1: Norm::new(g, lns[3 * l])?,
                    qkv: Linear::kn(m[0].0, &m[0].1.f32),
                    out1: Linear::kn(m[1].0, &m[1].1.f32),
                    ln2: Norm::new(g, lns[3 * l + 1])?,
                    q2: Linear::rows(w, b, 0, d),
                    k2: Linear::rows(w, b, d, 2 * d),
                    v2: Linear::rows(w, b, 2 * d, 3 * d),
                    out2: Linear::kn(m[2].0, &m[2].1.f32),
                    ln3: Norm::new(g, lns[3 * l + 2])?,
                    ff1: Linear::kn(m[3].0, &m[3].1.f32),
                    ff2: Linear::kn(m[4].0, &m[4].1.f32),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (hw, hb) = lin[lin.len() - 1];
        Ok(MathDecoder { tok, pos, layers, ln: Norm::new(g, lns[lns.len() - 1])?, head: Linear::kn(hw, &hb.f32), heads, d })
    }

    pub fn vocab(&self) -> usize {
        self.head.w.n
    }

    /// Decoder state of a batch of lines from their memories.
    pub fn begin(&self, memories: Vec<Vec<f32>>) -> DecodeState {
        let d = self.d;
        let lines = memories
            .into_iter()
            .map(|m| {
                let t = m.len() / d;
                let cross = self.layers.iter().map(|l| (l.k2.apply(&m, t), l.v2.apply(&m, t))).collect();
                LineState { t, cross, cache: vec![(vec![], vec![]); self.layers.len()] }
            })
            .collect();
        DecodeState { lines }
    }

    /// One greedy step for every line: logits [B, vocab].
    pub fn step(&self, st: &mut DecodeState, tokens: &[i64], pos: usize) -> Vec<f32> {
        let (d, b) = (self.d, st.lines.len());
        let hd = d / self.heads;
        let scale = (hd as f32).powf(-0.5);
        let mut x = vec![0f32; b * d];
        let pos = pos.min(self.pos.dims[0] - 1);
        for (i, &tk) in tokens.iter().enumerate() {
            let tk = (tk.max(0) as usize).min(self.tok.dims[0] - 1);
            for j in 0..d {
                x[i * d + j] = self.tok.f32[tk * d + j] + self.pos.f32[pos * d + j];
            }
        }
        for (li, l) in self.layers.iter().enumerate() {
            // self-attention over the cache (this step included)
            let a = l.ln1.apply(&x);
            let qkv = l.qkv.apply(&a, b);
            let mut att = vec![0f32; b * d];
            for (i, ls) in st.lines.iter_mut().enumerate() {
                let (kc, vc) = &mut ls.cache[li];
                kc.extend_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
                vc.extend_from_slice(&qkv[i * 3 * d + 2 * d..i * 3 * d + 3 * d]);
                let n = kc.len() / d;
                for h in 0..self.heads {
                    let q = &qkv[i * 3 * d + h * hd..i * 3 * d + (h + 1) * hd];
                    attend(q, kc, vc, n, d, h * hd, scale, |_| 0.0, &mut att[i * d + h * hd..i * d + (h + 1) * hd]);
                }
            }
            l.out1.add_into(&att, b, &mut x);
            // cross-attention over the memory
            let c = l.ln2.apply(&x);
            let q = l.q2.apply(&c, b);
            for (i, ls) in st.lines.iter().enumerate() {
                let (kc, vc) = &ls.cross[li];
                for h in 0..self.heads {
                    attend(
                        &q[i * d + h * hd..i * d + (h + 1) * hd],
                        kc,
                        vc,
                        ls.t,
                        d,
                        h * hd,
                        scale,
                        |_| 0.0,
                        &mut att[i * d + h * hd..i * d + (h + 1) * hd],
                    );
                }
            }
            l.out2.add_into(&att, b, &mut x);
            let e = l.ln3.apply(&x);
            let mut ff = l.ff1.apply(&e, b);
            gelu(&mut ff);
            l.ff2.add_into(&ff, b, &mut x);
        }
        let z = self.ln.apply(&x);
        self.head.apply(&z, b)
    }
}

struct LineState {
    t: usize,
    /// per layer: cross-attention keys and values [T, d]
    cross: Vec<(Vec<f32>, Vec<f32>)>,
    /// per layer: self-attention keys and values so far [P, d]
    cache: Vec<(Vec<f32>, Vec<f32>)>,
}

/// The decoder caches of one batch.
pub struct DecodeState {
    lines: Vec<LineState>,
}
