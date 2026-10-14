//! The reading-order model (order.onnx): a pre-norm transformer encoder over
//! the page's lines, with pairwise heads.
//!
//!   embed: [feats, Fourier(feats[:4])] -> Linear -> GELU -> Linear
//!   layers: x += Attn(LN(x)); x += FFN(LN(x))       (FFN: Linear, GELU, Linear)
//!   z = LN(x)
//!   a    = D - D^T,  D = (z A1)(z A2)^T / sqrt(d)   (i before j)
//!   same = S + S^T,  S = (z S1)(z S2)^T / sqrt(d)   (same block)
//!   tab  = T + T^T per head (same row / column / cell)
//!   cls  = z C                                      (block class)

use std::path::Path;

use crate::gemm::{Act, Out, PackedB, gemm};
use crate::onnx::{Graph, Node};
use crate::simd::gelu;
use crate::{EngineError, Result};

struct Linear {
    w: PackedB,
}

impl Linear {
    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * self.w.n];
        gemm(x, self.w.k, n, &self.w, &mut y, self.w.n, Act::None, Out::Store);
        y
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl Norm {
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

struct Layer {
    ln1: Norm,
    qkv: Linear,
    proj: Linear,
    ln2: Norm,
    ff1: Linear,
    ff2: Linear,
}

/// The order model's outputs for N lines (row-major N x N matrices).
pub struct OrderOut {
    pub a: Vec<f32>,
    pub cls: Vec<f32>,
    pub same: Vec<f32>,
    /// [3, N, N]
    pub tab: Vec<f32>,
    pub n_cls: usize,
}

pub struct OrderNet {
    freqs: Vec<f32>,
    n_four: usize,
    n_feats: usize,
    emb1: Linear,
    emb2: Linear,
    layers: Vec<Layer>,
    heads: usize,
    ln: Norm,
    a1: Linear,
    a2: Linear,
    s1: Linear,
    s2: Linear,
    t1: Linear,
    t2: Linear,
    tab_heads: usize,
    cls: Linear,
    d: usize,
}

fn bad(m: impl Into<String>) -> EngineError {
    EngineError::Model(format!("order: {}", m.into()))
}

impl OrderNet {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Graph::load(path)?;
        Self::from_graph(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn from_graph(g: &Graph) -> Result<Self> {
        // MatMuls with a weight, in graph order, each with the Add of its bias
        let lin = |n: &Node| -> Result<Linear> {
            let w = g.arg(n, 1)?;
            let (k, m) = (w.dims[0], w.dims[1]);
            let add = g
                .nodes
                .iter()
                .find(|a| a.op == "Add" && a.inputs.iter().any(|i| i == &n.outputs[0]))
                .ok_or_else(|| bad("a MatMul without a bias Add"))?;
            let b = add.inputs.iter().find_map(|i| g.tensors.get(i.as_str())).ok_or_else(|| bad("bias is not a constant"))?;
            Ok(Linear { w: PackedB::from_kn(&w.f32, k, m, Some(&b.f32)) })
        };
        let mms: Vec<&Node> = g.ops("MatMul").filter(|n| g.tensors.contains_key(n.inputs[1].as_str())).collect();
        let lns: Vec<&Node> = g.ops("LayerNormalization").collect();
        let norm = |n: &Node| -> Result<Norm> {
            Ok(Norm { g: g.arg(n, 1)?.f32.clone(), b: g.arg(n, 2)?.f32.clone(), eps: n.float("epsilon").unwrap_or(1e-5) })
        };
        if lns.is_empty() || lns.len() % 2 != 1 {
            return Err(bad("unexpected LayerNormalization count"));
        }
        let n_layers = (lns.len() - 1) / 2;
        if mms.len() != 2 + 4 * n_layers + 7 {
            return Err(bad(format!("{} weighted MatMuls for {n_layers} layers: not the order architecture", mms.len())));
        }
        // Fourier features: Slice(feats, 0..n_four) * freqs -> sin, cos
        let slice = g.nodes.iter().find(|n| n.op == "Slice" && n.inputs[0] == "feats").ok_or_else(|| bad("no Fourier slice"))?;
        let n_four = g.arg(slice, 2)?.i64[0] as usize;
        let mul = g
            .nodes
            .iter()
            .find(|n| n.op == "Mul" && n.inputs.iter().any(|i| g.tensors.get(i.as_str()).is_some_and(|t| t.len() > 1)))
            .ok_or_else(|| bad("no Fourier frequencies"))?;
        let freqs = mul.inputs.iter().find_map(|i| g.tensors.get(i.as_str()).filter(|t| t.len() > 1)).unwrap().f32.clone();
        let emb1 = lin(mms[0])?;
        let n_feats = emb1.w.k - n_four * 2 * freqs.len();
        let d = emb1.w.n;
        let mut layers = vec![];
        for l in 0..n_layers {
            let m = &mms[2 + 4 * l..6 + 4 * l];
            layers.push(Layer {
                ln1: norm(lns[2 * l])?,
                qkv: lin(m[0])?,
                proj: lin(m[1])?,
                ln2: norm(lns[2 * l + 1])?,
                ff1: lin(m[2])?,
                ff2: lin(m[3])?,
            });
        }
        // attention heads: Reshape to [B, N, 3, heads, d / heads]
        let heads = g
            .nodes
            .iter()
            .filter(|n| n.op == "Concat" && n.inputs.len() == 5)
            .find_map(|n| g.tensors.get(n.inputs[3].as_str()).and_then(|t| t.i64.first().copied()))
            .ok_or_else(|| bad("no attention head count"))? as usize;
        let h = &mms[2 + 4 * n_layers..];
        let t1 = lin(h[4])?;
        let tab_heads = t1.w.n / d;
        Ok(OrderNet {
            freqs,
            n_four,
            n_feats,
            emb1,
            emb2: lin(mms[1])?,
            layers,
            heads,
            ln: norm(lns[lns.len() - 1])?,
            a1: lin(h[0])?,
            a2: lin(h[1])?,
            s1: lin(h[2])?,
            s2: lin(h[3])?,
            t1,
            t2: lin(h[5])?,
            tab_heads,
            cls: lin(h[6])?,
            d,
        })
    }

    /// The model on ``n`` lines of ``feats`` [N, 16].
    pub fn run(&self, feats: &[f32], n: usize) -> OrderOut {
        let d = self.d;
        let nf = self.n_feats;
        let fdim = self.n_four * 2 * self.freqs.len();
        // embedding input [N, feats ++ fourier]
        let mut x0 = vec![0f32; n * (nf + fdim)];
        for i in 0..n {
            let f = &feats[i * nf..(i + 1) * nf];
            let row = &mut x0[i * (nf + fdim)..(i + 1) * (nf + fdim)];
            row[..nf].copy_from_slice(f);
            let nfr = self.freqs.len();
            for k in 0..self.n_four {
                for (j, &fr) in self.freqs.iter().enumerate() {
                    let v = f[k] * fr;
                    row[nf + k * 2 * nfr + j] = v.sin();
                    row[nf + k * 2 * nfr + nfr + j] = v.cos();
                }
            }
        }
        let mut h1 = self.emb1.apply(&x0, n);
        gelu(&mut h1);
        let mut x = self.emb2.apply(&h1, n);
        for l in &self.layers {
            let a = l.ln1.apply(&x);
            let qkv = l.qkv.apply(&a, n);
            let att = self.attention(&qkv, n);
            gemm(&att, d, n, &l.proj.w, &mut x, d, Act::None, Out::Add);
            let b = l.ln2.apply(&x);
            let mut f = l.ff1.apply(&b, n);
            gelu(&mut f);
            gemm(&f, l.ff1.w.n, n, &l.ff2.w, &mut x, d, Act::None, Out::Add);
        }
        let z = self.ln.apply(&x);
        let scale = 1.0 / (d as f32).sqrt();
        let pair = |p: &Linear, q: &Linear, sign: f32| -> Vec<f32> {
            let (u, v) = (p.apply(&z, n), q.apply(&z, n));
            let m = bilinear(&u, &v, n, d, scale);
            let mut out = vec![0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    out[i * n + j] = m[i * n + j] + sign * m[j * n + i];
                }
            }
            out
        };
        let a = pair(&self.a1, &self.a2, -1.0);
        let same = pair(&self.s1, &self.s2, 1.0);
        // table heads: [N, heads, d] each
        let (tq, tk) = (self.t1.apply(&z, n), self.t2.apply(&z, n));
        let th = self.tab_heads;
        let mut tab = vec![0f32; th * n * n];
        for hd in 0..th {
            let take = |m: &[f32]| -> Vec<f32> { (0..n).flat_map(|i| m[(i * th + hd) * d..(i * th + hd + 1) * d].to_vec()).collect() };
            let s = bilinear(&take(&tq), &take(&tk), n, d, scale);
            for i in 0..n {
                for j in 0..n {
                    tab[hd * n * n + i * n + j] = s[i * n + j] + s[j * n + i];
                }
            }
        }
        let cls = self.cls.apply(&z, n);
        OrderOut { a, cls, same, tab, n_cls: self.cls.w.n }
    }

    /// Multi-head self-attention over packed [N, 3 * d] q, k, v (no mask:
    /// every line is real).
    fn attention(&self, qkv: &[f32], n: usize) -> Vec<f32> {
        let d = self.d;
        let hd = d / self.heads;
        // q and k are both scaled by sqrt(1 / sqrt(hd)), as in the export
        let s = (1.0 / (hd as f32).sqrt()).sqrt();
        let mut out = vec![0f32; n * d];
        for h in 0..self.heads {
            let part = |which: usize| -> Vec<f32> {
                (0..n).flat_map(|i| qkv[i * 3 * d + which * d + h * hd..i * 3 * d + which * d + (h + 1) * hd].to_vec()).collect()
            };
            let (q, k, v) = (part(0), part(1), part(2));
            let q: Vec<f32> = q.iter().map(|x| x * s).collect();
            let k: Vec<f32> = k.iter().map(|x| x * s).collect();
            let mut p = bilinear(&q, &k, n, hd, 1.0);
            for row in p.chunks_exact_mut(n) {
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for e in row.iter_mut() {
                    *e = (*e - m).exp();
                    sum += *e;
                }
                for e in row.iter_mut() {
                    *e /= sum;
                }
            }
            let vb = PackedB::from_kn(&v, n, hd, None);
            let mut o = vec![0f32; n * hd];
            gemm(&p, n, n, &vb, &mut o, hd, Act::None, Out::Store);
            for i in 0..n {
                out[i * d + h * hd..i * d + (h + 1) * hd].copy_from_slice(&o[i * hd..(i + 1) * hd]);
            }
        }
        out
    }
}

/// (U V^T) * scale for U, V [N, d].
fn bilinear(u: &[f32], v: &[f32], n: usize, d: usize, scale: f32) -> Vec<f32> {
    let vb = PackedB::from_nk(v, n, d, None);
    let mut m = vec![0f32; n * n];
    gemm(u, d, n, &vb, &mut m, n, Act::None, Out::Store);
    if scale != 1.0 {
        m.iter_mut().for_each(|x| *x *= scale);
    }
    m
}
