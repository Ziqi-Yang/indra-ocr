//! A small executor for the convolutional models (det.onnx, orient.onnx):
//! the ONNX graph is compiled once into steps on NHWC feature maps (batch 1),
//! with each convolution's activation and residual add fused into it, and
//! the graph's shape arithmetic (Shape / Gather / Concat ... for Resize sizes)
//! evaluated on small integer vectors.
//!
//! Only the operators these models use are supported; anything else is an
//! error at load time, not a wrong answer at run time.

use std::collections::HashMap;
use std::path::Path;

use crate::gemm::{Act, Out, PackedB, gemm, gemm_par};
use crate::onnx::{Graph, Node};
use crate::par::Par;
use crate::simd::multiversion;
use crate::{EngineError, Result};

/// A feature map, NHWC with N = 1.
#[derive(Clone, Debug, Default)]
pub struct Feat {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Feat {
    pub fn new(h: usize, w: usize, c: usize) -> Feat {
        Feat { h, w, c, data: vec![0.0; h * w * c] }
    }
    /// From a [C, H, W] planar array.
    pub fn from_chw(c: usize, h: usize, w: usize, x: &[f32]) -> Feat {
        let mut f = Feat::new(h, w, c);
        for ch in 0..c {
            for i in 0..h * w {
                f.data[i * c + ch] = x[ch * h * w + i];
            }
        }
        f
    }
    /// As a [C, H, W] planar array.
    pub fn to_chw(&self) -> Vec<f32> {
        let (hw, c) = (self.h * self.w, self.c);
        let mut out = vec![0f32; hw * c];
        for i in 0..hw {
            for ch in 0..c {
                out[ch * hw + i] = self.data[i * c + ch];
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
enum Val {
    Feat(Feat),
    Ints(Vec<i64>),
    Vec(Vec<f32>),
    None,
}

#[derive(Clone, Debug)]
pub(crate) enum ConvKind {
    /// 1x1, stride 1: one product per pixel
    Pointwise(PackedB),
    /// one filter per channel: w [KH, KW, C]
    Depthwise { w: Vec<f32>, b: Vec<f32>, k: usize, stride: usize, pad: usize },
    /// any other: patches [KH, KW, Cin] times the packed filters
    General { b: PackedB, cin: usize, k: usize, stride: usize, pad: usize },
    /// a full convolution of a 1-channel map: w [KH, KW, Cout]
    Single { w: Vec<f32>, b: Vec<f32>, k: usize, stride: usize, pad: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unary {
    HardSwish,
    Relu,
    Sigmoid,
    Softplus,
}

#[derive(Clone, Debug)]
enum Step {
    Conv { x: usize, out: usize, kind: ConvKind, act: Act, residual: Option<usize> },
    Unary { x: usize, out: usize, op: Unary },
    Add { a: usize, b: usize, out: usize },
    MulScalar { x: usize, s: f32, out: usize },
    Resize { x: usize, sizes: usize, out: usize },
    DepthToSpace { x: usize, b: usize, out: usize },
    SpaceToDepth { x: usize, b: usize, out: usize },
    SliceC { x: usize, start: i64, end: i64, out: usize },
    ConcatC { xs: Vec<usize>, out: usize },
    Pool { x: usize, max: bool, out: usize },
    MeanHW { x: usize, out: usize },
    Gemm { x: usize, w: Vec<f32>, b: Vec<f32>, n: usize, out: usize },
    Softmax { x: usize, out: usize },
    // integer shape arithmetic
    Shape { x: usize, out: usize },
    Gather { x: usize, idx: Vec<i64>, out: usize },
    ConcatI { xs: Vec<usize>, out: usize },
    SliceI { x: usize, start: i64, end: i64, out: usize },
    Copy { x: usize, out: usize },
}

/// A compiled convolutional model.
pub struct Net {
    steps: Vec<Step>,
    slots: usize,
    input: usize,
    outputs: Vec<usize>,
    consts: Vec<(usize, Val)>,
    /// the step after which each slot is no longer read
    last_use: Vec<usize>,
}

fn bad(n: &Node, msg: &str) -> EngineError {
    EngineError::Model(format!("{} {:?}: {msg}", n.op, n.name))
}

impl Net {
    pub fn load(path: &Path) -> Result<Net> {
        let g = Graph::load(path)?;
        Net::compile(&g).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn compile(g: &Graph) -> Result<Net> {
        let mut slot: HashMap<String, usize> = HashMap::new();
        let mut next = 0usize;
        let mut id = |name: &str, slot: &mut HashMap<String, usize>| -> usize {
            *slot.entry(name.to_string()).or_insert_with(|| {
                next += 1;
                next - 1
            })
        };
        // readers per tensor, to decide fusions
        let mut readers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, n) in g.nodes.iter().enumerate() {
            for x in &n.inputs {
                readers.entry(x.as_str()).or_default().push(i);
            }
        }
        for o in &g.outputs {
            readers.entry(o.as_str()).or_default().push(usize::MAX);
        }
        let only_reader = |name: &str, i: usize| readers.get(name).is_some_and(|r| r.len() == 1 && r[0] == i);
        let ints = |name: &str| -> Option<Vec<i64>> {
            g.tensors.get(name).map(|t| if t.i64.is_empty() { t.f32.iter().map(|&v| v as i64).collect() } else { t.i64.clone() })
        };
        let input_name = g.inputs.first().ok_or_else(|| EngineError::Model("graph has no input".into()))?.clone();
        let input = id(&input_name, &mut slot);
        let mut steps = vec![];
        let mut consts: Vec<(usize, Val)> = vec![];
        let mut skip = vec![false; g.nodes.len()];
        for (i, n) in g.nodes.iter().enumerate() {
            if skip[i] {
                continue;
            }
            let inp = |k: usize, slot: &mut HashMap<String, usize>, id: &mut dyn FnMut(&str, &mut HashMap<String, usize>) -> usize| {
                id(&n.inputs[k], slot)
            };
            match n.op.as_str() {
                "Conv" => {
                    let kind = conv_kind(g, n)?;
                    // fuse a following activation, or a residual Add
                    let mut out_name = n.outputs[0].clone();
                    let mut act = Act::None;
                    let mut residual = None;
                    if let Some(&j) = readers.get(out_name.as_str()).filter(|r| r.len() == 1).and_then(|r| r.first())
                        && j != usize::MAX
                    {
                        let m = &g.nodes[j];
                        if m.op == "HardSwish" || m.op == "Relu" {
                            act = if m.op == "HardSwish" { Act::HardSwish } else { Act::Relu };
                            skip[j] = true;
                            out_name = m.outputs[0].clone();
                        } else if m.op == "Add" && matches!(kind, ConvKind::Pointwise(_)) && only_reader(&out_name, j) {
                            let other = if m.inputs[0] == out_name { &m.inputs[1] } else { &m.inputs[0] };
                            if !g.tensors.contains_key(other.as_str()) {
                                residual = Some(id(other, &mut slot));
                                skip[j] = true;
                                out_name = m.outputs[0].clone();
                            }
                        }
                    }
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Conv { x, out: id(&out_name, &mut slot), kind, act, residual });
                }
                "HardSwish" | "Relu" | "Sigmoid" | "Softplus" => {
                    let op = match n.op.as_str() {
                        "HardSwish" => Unary::HardSwish,
                        "Relu" => Unary::Relu,
                        "Sigmoid" => Unary::Sigmoid,
                        _ => Unary::Softplus,
                    };
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Unary { x, out: id(&n.outputs[0], &mut slot), op });
                }
                "Add" => {
                    if n.inputs.iter().any(|x| g.tensors.contains_key(x.as_str())) {
                        return Err(bad(n, "adding a constant"));
                    }
                    let (a, b) = (inp(0, &mut slot, &mut id), inp(1, &mut slot, &mut id));
                    steps.push(Step::Add { a, b, out: id(&n.outputs[0], &mut slot) });
                }
                "Mul" => {
                    let (x, s) = match (g.tensors.get(n.inputs[0].as_str()), g.tensors.get(n.inputs[1].as_str())) {
                        (None, Some(t)) if t.len() <= 1 => (inp(0, &mut slot, &mut id), t.f32[0]),
                        (Some(t), None) if t.len() <= 1 => (inp(1, &mut slot, &mut id), t.f32[0]),
                        _ => return Err(bad(n, "only products with a scalar constant")),
                    };
                    steps.push(Step::MulScalar { x, s, out: id(&n.outputs[0], &mut slot) });
                }
                "Resize" => {
                    if n.attrs.get("mode").is_some_and(|m| !matches!(m, crate::onnx::Attr::Str(s) if s == "nearest"))
                        || n.attrs
                            .get("coordinate_transformation_mode")
                            .is_some_and(|m| !matches!(m, crate::onnx::Attr::Str(s) if s == "asymmetric"))
                        || n.attrs.get("nearest_mode").is_some_and(|m| !matches!(m, crate::onnx::Attr::Str(s) if s == "floor"))
                        || n.inputs.len() < 4
                    {
                        return Err(bad(n, "only nearest (asymmetric, floor) resizes to given sizes"));
                    }
                    let x = inp(0, &mut slot, &mut id);
                    let sizes = inp(3, &mut slot, &mut id);
                    steps.push(Step::Resize { x, sizes, out: id(&n.outputs[0], &mut slot) });
                }
                "DepthToSpace" => {
                    if !n.attrs.get("mode").is_some_and(|m| matches!(m, crate::onnx::Attr::Str(s) if s == "CRD")) {
                        return Err(bad(n, "only CRD mode"));
                    }
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::DepthToSpace { x, b: n.int("blocksize").unwrap_or(2) as usize, out: id(&n.outputs[0], &mut slot) });
                }
                "Unsqueeze" if !g.tensors.contains_key(n.inputs[0].as_str()) && is_feature_unsqueeze(g, i) => {
                    // Unsqueeze -> Reshape -> Reshape -> Transpose -> Reshape -> Squeeze:
                    // space-to-depth by 2, channel = c * 4 + dy * 2 + dx
                    let chain = space_to_depth_chain(g, i).ok_or_else(|| bad(n, "unsupported reshape chain on a feature map"))?;
                    for &j in &chain {
                        skip[j] = true;
                    }
                    let x = inp(0, &mut slot, &mut id);
                    let last = &g.nodes[*chain.last().unwrap()];
                    steps.push(Step::SpaceToDepth { x, b: 2, out: id(&last.outputs[0], &mut slot) });
                }
                "Slice" => {
                    let starts = ints(&n.inputs[1]).ok_or_else(|| bad(n, "dynamic starts"))?;
                    let ends = ints(&n.inputs[2]).ok_or_else(|| bad(n, "dynamic ends"))?;
                    let axes = n.inputs.get(3).and_then(|a| ints(a)).unwrap_or(vec![0]);
                    if n.inputs.get(4).and_then(|s| ints(s)).is_some_and(|s| s.iter().any(|&v| v != 1)) {
                        return Err(bad(n, "strided slices"));
                    }
                    let x = inp(0, &mut slot, &mut id);
                    let out = id(&n.outputs[0], &mut slot);
                    match axes[..] {
                        [1] => steps.push(Step::SliceC { x, start: starts[0], end: ends[0], out }),
                        [0] => steps.push(Step::SliceI { x, start: starts[0], end: ends[0], out }),
                        _ => return Err(bad(n, "slices on channels or shape vectors only")),
                    }
                }
                "Concat" => {
                    let axis = n.int("axis").unwrap_or(0);
                    let xs: Vec<usize> = (0..n.inputs.len()).map(|k| inp(k, &mut slot, &mut id)).collect();
                    let out = id(&n.outputs[0], &mut slot);
                    steps.push(if axis == 1 { Step::ConcatC { xs, out } } else { Step::ConcatI { xs, out } });
                }
                "AveragePool" | "MaxPool" => {
                    if n.ints("kernel_shape") != Some(&[2, 2][..])
                        || n.ints("strides") != Some(&[2, 2][..])
                        || n.ints("pads").is_some_and(|p| p.iter().any(|&v| v != 0))
                        || n.int("ceil_mode").unwrap_or(0) != 0
                    {
                        return Err(bad(n, "2x2 pools with stride 2 only"));
                    }
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Pool { x, max: n.op == "MaxPool", out: id(&n.outputs[0], &mut slot) });
                }
                "ReduceMean" => {
                    if n.ints("axes") != Some(&[2, 3][..]) || n.int("keepdims") != Some(0) {
                        return Err(bad(n, "a mean over H and W only"));
                    }
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::MeanHW { x, out: id(&n.outputs[0], &mut slot) });
                }
                "Gemm" => {
                    let w = g.arg(n, 1)?;
                    if n.int("transB") != Some(1) || n.float("alpha").unwrap_or(1.0) != 1.0 || n.float("beta").unwrap_or(1.0) != 1.0 {
                        return Err(bad(n, "Gemm with transB only"));
                    }
                    let b = n.inputs.get(2).map(|_| g.arg(n, 2)).transpose()?.map(|t| t.f32.clone()).unwrap_or(vec![0.0; w.dims[0]]);
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Gemm { x, w: w.f32.clone(), b, n: w.dims[0], out: id(&n.outputs[0], &mut slot) });
                }
                "Softmax" => {
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Softmax { x, out: id(&n.outputs[0], &mut slot) });
                }
                "Shape" => {
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Shape { x, out: id(&n.outputs[0], &mut slot) });
                }
                "Gather" => {
                    let idx = ints(&n.inputs[1]).ok_or_else(|| bad(n, "dynamic indices"))?;
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Gather { x, idx, out: id(&n.outputs[0], &mut slot) });
                }
                "Unsqueeze" | "Cast" | "Squeeze" | "Identity" => {
                    // on shape vectors these change nothing we keep
                    let x = inp(0, &mut slot, &mut id);
                    steps.push(Step::Copy { x, out: id(&n.outputs[0], &mut slot) });
                }
                other => return Err(bad(n, &format!("operator {other} is not supported"))),
            }
            // constant inputs the steps read by slot
            for x in &n.inputs {
                if let (Some(&s), Some(t)) = (slot.get(x.as_str()), g.tensors.get(x.as_str()))
                    && !consts.iter().any(|c| c.0 == s)
                {
                    let v = if t.i64.is_empty() { Val::Vec(t.f32.clone()) } else { Val::Ints(t.i64.clone()) };
                    consts.push((s, v));
                }
            }
        }
        let outputs = g
            .outputs
            .iter()
            .map(|o| slot.get(o.as_str()).copied().ok_or_else(|| EngineError::Model(format!("output {o} is never computed"))))
            .collect::<Result<Vec<_>>>()?;
        let slots = next;
        let mut last_use = vec![0usize; slots];
        for (k, s) in steps.iter().enumerate() {
            for x in step_inputs(s) {
                last_use[x] = k;
            }
        }
        for &o in &outputs {
            last_use[o] = usize::MAX;
        }
        Ok(Net { steps, slots, input, outputs, consts, last_use })
    }

    /// Run on one input feature map: the graph outputs (features or vectors).
    pub fn run(&self, x: Feat, par: &Par) -> Result<Vec<NetOut>> {
        let mut v: Vec<Val> = vec![Val::None; self.slots];
        for (s, c) in &self.consts {
            v[*s] = c.clone();
        }
        v[self.input] = Val::Feat(x);
        let mut prof = crate::profile::Timer::start();
        for (k, s) in self.steps.iter().enumerate() {
            let out = self.exec(s, &mut v, k, par)?;
            let o = step_output(s);
            v[o] = out;
            for x in step_inputs(s) {
                if self.last_use[x] == k {
                    v[x] = Val::None;
                }
            }
            prof.lap(step_name(s));
        }
        self.outputs
            .iter()
            .map(|&o| match std::mem::replace(&mut v[o], Val::None) {
                Val::Feat(f) => Ok(NetOut::Feat(f)),
                Val::Vec(x) => Ok(NetOut::Vec(x)),
                _ => Err(EngineError::Model("an output is not a tensor".into())),
            })
            .collect()
    }

    /// A feature input: moved out of its slot when this is its last reader.
    fn take(&self, v: &mut [Val], x: usize, k: usize) -> Result<Feat> {
        let f = if self.last_use[x] == k { std::mem::replace(&mut v[x], Val::None) } else { v[x].clone() };
        match f {
            Val::Feat(f) => Ok(f),
            _ => Err(EngineError::Model("expected a feature map".into())),
        }
    }

    fn feat<'a>(&self, v: &'a [Val], x: usize) -> Result<&'a Feat> {
        match &v[x] {
            Val::Feat(f) => Ok(f),
            _ => Err(EngineError::Model("expected a feature map".into())),
        }
    }

    fn ints<'a>(&self, v: &'a [Val], x: usize) -> Result<&'a [i64]> {
        match &v[x] {
            Val::Ints(i) => Ok(i),
            _ => Err(EngineError::Model("expected a shape vector".into())),
        }
    }

    fn exec(&self, s: &Step, v: &mut [Val], k: usize, par: &Par) -> Result<Val> {
        Ok(match s {
            Step::Conv { x, kind, act, residual, .. } => {
                let res = match residual {
                    Some(r) => Some(self.take(v, *r, k)?),
                    None => None,
                };
                Val::Feat(conv(self.feat(v, *x)?, kind, *act, res, par))
            }
            Step::Unary { x, op, .. } => {
                let mut f = self.take(v, *x, k)?;
                unary(&mut f.data, *op);
                Val::Feat(f)
            }
            Step::Add { a, b, .. } => {
                let mut fa = self.take(v, *a, k)?;
                let fb = self.feat(v, *b)?;
                if (fa.h, fa.w, fa.c) != (fb.h, fb.w, fb.c) {
                    return Err(EngineError::Model("Add of feature maps of different shapes".into()));
                }
                add_into(&mut fa.data, &fb.data);
                Val::Feat(fa)
            }
            Step::MulScalar { x, s, .. } => {
                let mut f = self.take(v, *x, k)?;
                f.data.iter_mut().for_each(|e| *e *= s);
                Val::Feat(f)
            }
            Step::Resize { x, sizes, .. } => {
                let sz = self.ints(v, *sizes)?;
                if sz.len() != 4 {
                    return Err(EngineError::Model("Resize sizes are not [N, C, H, W]".into()));
                }
                Val::Feat(resize_nearest(self.feat(v, *x)?, sz[2] as usize, sz[3] as usize))
            }
            Step::DepthToSpace { x, b, .. } => Val::Feat(depth_to_space(self.feat(v, *x)?, *b)),
            Step::SpaceToDepth { x, b, .. } => Val::Feat(space_to_depth(self.feat(v, *x)?, *b)),
            Step::SliceC { x, start, end, .. } => {
                let f = self.feat(v, *x)?;
                let c = f.c as i64;
                let clamp = |i: i64| if i < 0 { (i + c).clamp(0, c) } else { i.min(c) } as usize;
                let (s0, s1) = (clamp(*start), clamp(*end).max(clamp(*start)));
                let mut o = Feat::new(f.h, f.w, s1 - s0);
                for (dst, src) in o.data.chunks_exact_mut(s1 - s0).zip(f.data.chunks_exact(f.c)) {
                    dst.copy_from_slice(&src[s0..s1]);
                }
                Val::Feat(o)
            }
            Step::ConcatC { xs, .. } => {
                let fs: Vec<&Feat> = xs.iter().map(|&x| self.feat(v, x)).collect::<Result<_>>()?;
                let (h, w) = (fs[0].h, fs[0].w);
                if fs.iter().any(|f| (f.h, f.w) != (h, w)) {
                    return Err(EngineError::Model("Concat of feature maps of different sizes".into()));
                }
                let c: usize = fs.iter().map(|f| f.c).sum();
                let mut o = Feat::new(h, w, c);
                for p in 0..h * w {
                    let mut off = 0;
                    for f in &fs {
                        o.data[p * c + off..p * c + off + f.c].copy_from_slice(&f.data[p * f.c..(p + 1) * f.c]);
                        off += f.c;
                    }
                }
                Val::Feat(o)
            }
            Step::Pool { x, max, .. } => Val::Feat(pool2(self.feat(v, *x)?, *max)),
            Step::MeanHW { x, .. } => {
                let f = self.feat(v, *x)?;
                let mut m = vec![0f32; f.c];
                crate::simd::col_sums(&f.data, f.c, &mut m);
                let inv = 1.0 / (f.h * f.w).max(1) as f32;
                Val::Vec(m.into_iter().map(|s| s * inv).collect())
            }
            Step::Gemm { x, w, b, n, .. } => {
                let xv = match &v[*x] {
                    Val::Vec(x) => x,
                    _ => return Err(EngineError::Model("Gemm input is not a vector".into())),
                };
                let kk = xv.len();
                Val::Vec((0..*n).map(|j| b[j] + w[j * kk..(j + 1) * kk].iter().zip(xv).map(|(a, b)| a * b).sum::<f32>()).collect())
            }
            Step::Softmax { x, .. } => {
                let xv = match &v[*x] {
                    Val::Vec(x) => x.clone(),
                    _ => return Err(EngineError::Model("Softmax input is not a vector".into())),
                };
                let m = xv.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = xv.iter().map(|&a| (a - m).exp()).collect();
                let s: f32 = e.iter().sum();
                Val::Vec(e.into_iter().map(|a| a / s).collect())
            }
            Step::Shape { x, .. } => {
                let f = self.feat(v, *x)?;
                Val::Ints(vec![1, f.c as i64, f.h as i64, f.w as i64])
            }
            Step::Gather { x, idx, .. } => {
                let xs = self.ints(v, *x)?;
                let n = xs.len() as i64;
                Val::Ints(idx.iter().map(|&i| xs[(if i < 0 { i + n } else { i }) as usize]).collect())
            }
            Step::ConcatI { xs, .. } => {
                Val::Ints(xs.iter().map(|&x| self.ints(v, x).map(|s| s.to_vec())).collect::<Result<Vec<_>>>()?.concat())
            }
            Step::SliceI { x, start, end, .. } => {
                let xs = self.ints(v, *x)?;
                let n = xs.len() as i64;
                let clamp = |i: i64| if i < 0 { (i + n).clamp(0, n) } else { i.min(n) } as usize;
                Val::Ints(xs[clamp(*start)..clamp(*end).max(clamp(*start))].to_vec())
            }
            Step::Copy { x, .. } => v[*x].clone(),
        })
    }
}

/// A graph output.
#[derive(Debug)]
pub enum NetOut {
    Feat(Feat),
    Vec(Vec<f32>),
}

fn step_inputs(s: &Step) -> Vec<usize> {
    match s {
        Step::Conv { x, residual, .. } => std::iter::once(*x).chain(*residual).collect(),
        Step::Add { a, b, .. } => vec![*a, *b],
        Step::Resize { x, sizes, .. } => vec![*x, *sizes],
        Step::ConcatC { xs, .. } | Step::ConcatI { xs, .. } => xs.clone(),
        Step::Unary { x, .. }
        | Step::MulScalar { x, .. }
        | Step::DepthToSpace { x, .. }
        | Step::SpaceToDepth { x, .. }
        | Step::SliceC { x, .. }
        | Step::Pool { x, .. }
        | Step::MeanHW { x, .. }
        | Step::Gemm { x, .. }
        | Step::Softmax { x, .. }
        | Step::Shape { x, .. }
        | Step::Gather { x, .. }
        | Step::SliceI { x, .. }
        | Step::Copy { x, .. } => vec![*x],
    }
}

fn step_output(s: &Step) -> usize {
    match s {
        Step::Conv { out, .. }
        | Step::Unary { out, .. }
        | Step::Add { out, .. }
        | Step::MulScalar { out, .. }
        | Step::Resize { out, .. }
        | Step::DepthToSpace { out, .. }
        | Step::SpaceToDepth { out, .. }
        | Step::SliceC { out, .. }
        | Step::ConcatC { out, .. }
        | Step::Pool { out, .. }
        | Step::MeanHW { out, .. }
        | Step::Gemm { out, .. }
        | Step::Softmax { out, .. }
        | Step::Shape { out, .. }
        | Step::Gather { out, .. }
        | Step::ConcatI { out, .. }
        | Step::SliceI { out, .. }
        | Step::Copy { out, .. } => *out,
    }
}

fn step_name(s: &Step) -> &'static str {
    match s {
        Step::Conv { kind: ConvKind::Pointwise(_), .. } => "cnn.conv1x1",
        Step::Conv { kind: ConvKind::Depthwise { .. }, .. } => "cnn.dwconv",
        Step::Conv { kind: ConvKind::General { .. }, .. } => "cnn.conv",
        Step::Conv { kind: ConvKind::Single { .. }, .. } => "cnn.conv1ch",
        Step::Resize { .. } => "cnn.resize",
        Step::Add { .. } => "cnn.add",
        _ => "cnn.other",
    }
}

fn is_feature_unsqueeze(g: &Graph, i: usize) -> bool {
    space_to_depth_chain(g, i).is_some()
}

/// Node indices of Unsqueeze(3) -> Reshape[0,0,-1,2,0] -> Reshape[0,0,0,0,-1,2]
/// -> Transpose(0,1,3,5,2,4) -> Reshape[0,-1,1,1,0,0] -> Squeeze(2,3).
fn space_to_depth_chain(g: &Graph, i: usize) -> Option<Vec<usize>> {
    let want: [(&str, Option<Vec<i64>>); 6] = [
        ("Unsqueeze", Some(vec![3])),
        ("Reshape", Some(vec![0, 0, -1, 2, 0])),
        ("Reshape", Some(vec![0, 0, 0, 0, -1, 2])),
        ("Transpose", None),
        ("Reshape", Some(vec![0, -1, 1, 1, 0, 0])),
        ("Squeeze", Some(vec![2, 3])),
    ];
    let mut chain = vec![];
    let mut cur = i;
    for (k, (op, arg)) in want.iter().enumerate() {
        let n = g.nodes.get(cur)?;
        if n.op != *op {
            return None;
        }
        if let Some(a) = arg {
            let t = g.tensors.get(n.inputs.get(1)?.as_str())?;
            if &t.i64 != a {
                return None;
            }
        } else if n.ints("perm") != Some(&[0, 1, 3, 5, 2, 4][..]) {
            return None;
        }
        chain.push(cur);
        if k + 1 < want.len() {
            let out = &n.outputs[0];
            cur = g.nodes.iter().position(|m| m.inputs.first() == Some(out))?;
        }
    }
    Some(chain)
}

/// The convolution of a Conv node, prepared for its kind.
pub(crate) fn conv_kind(g: &Graph, n: &Node) -> Result<ConvKind> {
    let (w, b) = (g.arg(n, 1)?, n.inputs.get(2).map(|_| g.arg(n, 2)).transpose()?);
    let (co, ci_g, kh, kw) = match w.dims[..] {
        [a, b, c, d] => (a, b, c, d),
        _ => return Err(bad(n, "2-D convolutions only")),
    };
    if kh != kw {
        return Err(bad(n, "square kernels only"));
    }
    let group = n.int("group").unwrap_or(1) as usize;
    let stride = n.ints("strides").map_or(1, |s| s[0] as usize);
    let pad = n.ints("pads").map_or(0, |p| p[0] as usize);
    if n.ints("strides").is_some_and(|s| s.iter().any(|&v| v as usize != stride))
        || n.ints("pads").is_some_and(|p| p.iter().any(|&v| v as usize != pad))
        || n.ints("dilations").is_some_and(|d| d.iter().any(|&v| v != 1))
    {
        return Err(bad(n, "uneven strides, pads or dilations"));
    }
    let bias = b.map(|t| t.f32.clone()).unwrap_or_else(|| vec![0.0; co]);
    Ok(if group == 1 && kh == 1 && stride == 1 && pad == 0 {
        ConvKind::Pointwise(PackedB::from_nk(&w.f32, co, ci_g, Some(&bias)))
    } else if group == co && ci_g == 1 {
        let mut dw = vec![0f32; kh * kw * co];
        for ch in 0..co {
            for j in 0..kh * kw {
                dw[j * co + ch] = w.f32[ch * kh * kw + j];
            }
        }
        ConvKind::Depthwise { w: dw, b: bias, k: kh, stride, pad }
    } else if group == 1 && ci_g == 1 {
        let mut sw = vec![0f32; kh * kw * co];
        for o in 0..co {
            for j in 0..kh * kw {
                sw[j * co + o] = w.f32[o * kh * kw + j];
            }
        }
        ConvKind::Single { w: sw, b: bias, k: kh, stride, pad }
    } else if group == 1 {
        // [Co, Ci, KH, KW] -> rows (kh, kw, ci)
        let kk = kh * kw * ci_g;
        let mut wt = vec![0f32; co * kk];
        for o in 0..co {
            for ci in 0..ci_g {
                for j in 0..kh * kw {
                    wt[o * kk + j * ci_g + ci] = w.f32[(o * ci_g + ci) * kh * kw + j];
                }
            }
        }
        ConvKind::General { b: PackedB::from_nk(&wt, co, kk, Some(&bias)), cin: ci_g, k: kh, stride, pad }
    } else {
        return Err(bad(n, "grouped convolutions other than depthwise"));
    })
}

/// One convolution with its fused activation (and residual: out = residual + act(conv)).
pub(crate) fn conv(x: &Feat, kind: &ConvKind, act: Act, residual: Option<Feat>, par: &Par) -> Feat {
    match kind {
        ConvKind::Pointwise(b) => {
            let (mut o, out) = match residual {
                Some(r) => (r, Out::Add),
                None => (Feat::new(x.h, x.w, b.n), Out::Store),
            };
            gemm_par(par, &x.data, x.c, x.h * x.w, b, &mut o.data, b.n, act, out);
            o
        }
        ConvKind::Depthwise { w, b, k, stride, pad } => {
            let oh = (x.h + 2 * pad - k) / stride + 1;
            let ow = (x.w + 2 * pad - k) / stride + 1;
            let mut o = Feat::new(oh, ow, x.c);
            let rows = oh.div_ceil(par.threads() * 4).max(1);
            par.chunks(&mut o.data, rows * ow * x.c, |i, chunk| {
                dwconv2d(&x.data, x.h, x.w, x.c, w, b, *k, *stride, *pad, chunk, i * rows, chunk.len() / (ow * x.c), ow, false);
            });
            unary_act(&mut o.data, act);
            if let Some(r) = residual {
                add_into(&mut o.data, &r.data);
            }
            o
        }
        ConvKind::Single { w, b, k, stride, pad } => {
            let oh = (x.h + 2 * pad - k) / stride + 1;
            let ow = (x.w + 2 * pad - k) / stride + 1;
            let mut o = Feat::new(oh, ow, b.len());
            let c = b.len();
            let rows = oh.div_ceil(par.threads() * 4).max(1);
            par.chunks(&mut o.data, rows * ow * c, |i, chunk| {
                dwconv2d(&x.data, x.h, x.w, c, w, b, *k, *stride, *pad, chunk, i * rows, chunk.len() / (ow * c), ow, true);
            });
            unary_act(&mut o.data, act);
            if let Some(r) = residual {
                add_into(&mut o.data, &r.data);
            }
            o
        }
        ConvKind::General { b, cin, k, stride, pad } => {
            let oh = (x.h + 2 * pad - k) / stride + 1;
            let ow = (x.w + 2 * pad - k) / stride + 1;
            let kk = k * k * cin;
            let mut o = match &residual {
                Some(r) => r.clone(),
                None => Feat::new(oh, ow, b.n),
            };
            // im2col in row blocks, so the patches stay in cache
            let rows_per = (4096 / ow).max(1);
            let out = if residual.is_some() { Out::Add } else { Out::Store };
            par.chunks(&mut o.data, rows_per * ow * b.n, |i, chunk| {
                let r0 = i * rows_per;
                let rows = chunk.len() / (ow * b.n);
                let mut patches = vec![0f32; rows * ow * kk];
                im2col(&x.data, x.h, x.w, *cin, *k, *stride, *pad, r0, rows, ow, &mut patches);
                gemm(&patches, kk, rows * ow, b, chunk, b.n, act, out);
            });
            o
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col(
    x: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    k: usize,
    stride: usize,
    pad: usize,
    r0: usize,
    rows: usize,
    ow: usize,
    out: &mut [f32],
) {
    let kk = k * k * cin;
    for r in 0..rows {
        let oy = r0 + r;
        for ox in 0..ow {
            let p = &mut out[(r * ow + ox) * kk..(r * ow + ox + 1) * kk];
            for i in 0..k {
                let iy = (oy * stride + i) as isize - pad as isize;
                for j in 0..k {
                    let ix = (ox * stride + j) as isize - pad as isize;
                    let dst = &mut p[(i * k + j) * cin..(i * k + j + 1) * cin];
                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                        dst.fill(0.0);
                    } else {
                        let s = (iy as usize * w + ix as usize) * cin;
                        dst.copy_from_slice(&x[s..s + cin]);
                    }
                }
            }
        }
    }
}

/// Depthwise k x k convolution of an NHWC map (zero padding). With
/// ``bcast`` the input has one channel, shared by every output channel (a
/// full convolution of a 1-channel map).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dw_impl<const C: usize>(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    wt: &[f32],
    b: &[f32],
    k: usize,
    stride: usize,
    pad: usize,
    out: &mut [f32],
    oy0: usize,
    rows: usize,
    ow: usize,
    bcast: bool,
) {
    // C > 0: the channel count is known at compile time (straight vector code)
    let c = if C > 0 { C } else { c };
    let xc = if bcast { 1 } else { c };
    for r in 0..rows {
        let oy = oy0 + r;
        let orow = &mut out[r * ow * c..(r + 1) * ow * c];
        for o in orow.chunks_exact_mut(c) {
            o.copy_from_slice(&b[..c]);
        }
        for i in 0..k {
            let iy = (oy * stride + i) as isize - pad as isize;
            if iy < 0 || iy >= h as isize {
                continue;
            }
            let xrow = &x[iy as usize * w * xc..(iy as usize + 1) * w * xc];
            for j in 0..k {
                let wr = &wt[(i * k + j) * c..(i * k + j + 1) * c];
                // output columns whose input column ox * stride + j - pad is inside
                let lo = pad.saturating_sub(j).div_ceil(stride);
                let hi = ((w + pad).saturating_sub(j)).div_ceil(stride).min(ow);
                for ox in lo..hi {
                    let ix = ox * stride + j - pad;
                    let o = &mut orow[ox * c..(ox + 1) * c];
                    if bcast {
                        let s = xrow[ix];
                        for (o, &wv) in o.iter_mut().zip(wr) {
                            *o += wv * s;
                        }
                    } else {
                        let src = &xrow[ix * c..(ix + 1) * c];
                        for ((o, &s), &wv) in o.iter_mut().zip(src).zip(wr) {
                            *o += wv * s;
                        }
                    }
                }
            }
        }
    }
}

macro_rules! dw_dispatch {
    ($($n:literal)*) => {
        multiversion! {
            #[allow(clippy::too_many_arguments)]
            fn dwconv2d(x: &[f32], h: usize, w: usize, c: usize, wt: &[f32], b: &[f32], k: usize, stride: usize, pad: usize, out: &mut [f32], oy0: usize, rows: usize, ow: usize, bcast: bool) {
                match c {
                    $($n => dw_impl::<$n>(x, h, w, c, wt, b, k, stride, pad, out, oy0, rows, ow, bcast),)*
                    _ => dw_impl::<0>(x, h, w, c, wt, b, k, stride, pad, out, oy0, rows, ow, bcast),
                }
            }
        }
    };
}
dw_dispatch!(8 16 24 32 48 64 96 128 192);

multiversion! {
    fn add_into(a: &mut [f32], b: &[f32]) {
        for (x, &y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }
}

multiversion! {
    fn unary_act(x: &mut [f32], act: Act) {
        match act {
            Act::None => {}
            Act::Relu => x.iter_mut().for_each(|v| *v = v.max(0.0)),
            Act::HardSwish => x.iter_mut().for_each(|v| *v *= (*v * (1.0 / 6.0) + 0.5).clamp(0.0, 1.0)),
        }
    }
}

fn unary(x: &mut [f32], op: Unary) {
    match op {
        Unary::HardSwish => unary_act(x, Act::HardSwish),
        Unary::Relu => unary_act(x, Act::Relu),
        Unary::Sigmoid => x.iter_mut().for_each(|v| *v = crate::simd::sigmoid(*v)),
        // log(1 + e^x), without overflow
        Unary::Softplus => x.iter_mut().for_each(|v| *v = if *v > 0.0 { *v + (-*v).exp().ln_1p() } else { v.exp().ln_1p() }),
    }
}

/// Nearest-neighbour resize, asymmetric coordinates, floor (ONNX Resize).
fn resize_nearest(x: &Feat, oh: usize, ow: usize) -> Feat {
    let mut o = Feat::new(oh, ow, x.c);
    let (sy, sx) = (oh as f32 / x.h as f32, ow as f32 / x.w as f32);
    let map = |i: usize, s: f32, n: usize| ((i as f32 / s).floor() as usize).min(n - 1);
    let cols: Vec<usize> = (0..ow).map(|j| map(j, sx, x.w)).collect();
    for i in 0..oh {
        let iy = map(i, sy, x.h);
        for (j, &ix) in cols.iter().enumerate() {
            o.data[(i * ow + j) * x.c..(i * ow + j + 1) * x.c].copy_from_slice(&x.data[(iy * x.w + ix) * x.c..(iy * x.w + ix + 1) * x.c]);
        }
    }
    o
}

/// DepthToSpace, CRD: out[c][y * b + i][x * b + j] = in[c * b * b + i * b + j][y][x].
fn depth_to_space(x: &Feat, b: usize) -> Feat {
    let c = x.c / (b * b);
    let mut o = Feat::new(x.h * b, x.w * b, c);
    for y in 0..x.h {
        for xx in 0..x.w {
            let src = &x.data[(y * x.w + xx) * x.c..(y * x.w + xx + 1) * x.c];
            for i in 0..b {
                for j in 0..b {
                    let d = ((y * b + i) * o.w + xx * b + j) * c;
                    for ch in 0..c {
                        o.data[d + ch] = src[ch * b * b + i * b + j];
                    }
                }
            }
        }
    }
    o
}

/// The inverse of depth_to_space (channel c * b * b + i * b + j).
fn space_to_depth(x: &Feat, b: usize) -> Feat {
    let (oh, ow, oc) = (x.h / b, x.w / b, x.c * b * b);
    let mut o = Feat::new(oh, ow, oc);
    for y in 0..oh {
        for xx in 0..ow {
            let d = &mut o.data[(y * ow + xx) * oc..(y * ow + xx + 1) * oc];
            for i in 0..b {
                for j in 0..b {
                    let s = ((y * b + i) * x.w + xx * b + j) * x.c;
                    for ch in 0..x.c {
                        d[ch * b * b + i * b + j] = x.data[s + ch];
                    }
                }
            }
        }
    }
    o
}

/// 2 x 2 pool with stride 2: mean or max. The output size follows ONNX
/// Runtime: (n - 2) / 2 + 1 in C integer division, so a size of 1 pools to 1
/// (the window is cut at the edge) rather than to 0.
pub(crate) fn pool2(x: &Feat, max: bool) -> Feat {
    let out = |n: usize| if n == 1 { 1 } else { n / 2 };
    let (oh, ow) = (out(x.h), out(x.w));
    let mut o = Feat::new(oh, ow, x.c);
    let c = x.c;
    for y in 0..oh {
        for xx in 0..ow {
            let dst = &mut o.data[(y * ow + xx) * c..(y * ow + xx + 1) * c];
            let mut n = 0;
            for i in 0..2 {
                for j in 0..2 {
                    let (sy, sx) = (2 * y + i, 2 * xx + j);
                    if sy >= x.h || sx >= x.w {
                        continue;
                    }
                    let src = &x.data[(sy * x.w + sx) * c..(sy * x.w + sx + 1) * c];
                    for ch in 0..c {
                        dst[ch] = if n == 0 {
                            src[ch]
                        } else if max {
                            dst[ch].max(src[ch])
                        } else {
                            dst[ch] + src[ch]
                        };
                    }
                    n += 1;
                }
            }
            if !max {
                let inv = 1.0 / n as f32;
                dst.iter_mut().for_each(|v| *v *= inv);
            }
        }
    }
    o
}
