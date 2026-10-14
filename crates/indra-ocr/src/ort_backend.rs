//! The models on ONNX Runtime (the `ort` crate, loading the shared library at
//! run time): the same sessions, options, batching and padding as the Python
//! package, so its outputs are the reference for the native engine.

use std::path::Path;
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::ep::CPU;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::backend::{Backend, BackendError, DetOut, MathBackend, OrderOut, RecModel, RecOut, Result};
use crate::crop::{WIDTH_MULTIPLE, pad_batch, rec_batches};
use crate::imgproc::ImageF;
use crate::order_post::Mat;

fn err(e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("onnxruntime: {e}"))
}

/// The process-wide ONNX Runtime environment. The library is ORT_DYLIB_PATH, or
/// libonnxruntime found by the dynamic loader.
pub fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        let lib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "onnxruntime.dll".into()
            } else if cfg!(target_os = "macos") {
                "libonnxruntime.dylib".into()
            } else {
                "libonnxruntime.so".into()
            }
        });
        let b = ort::init_from(&lib).map_err(|e| format!("cannot load ONNX Runtime from {lib}: {e} (set ORT_DYLIB_PATH)"))?;
        b.with_name("indra-ocr").build().map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| BackendError(e.clone()))
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    let env = environment()?;
    let b = Session::builder(env).map_err(err)?;
    // No CPU memory arena: with line batches of changing shape it keeps growing
    // and buys no measurable speed (see pipeline.py).
    let b = b.with_execution_providers([CPU::default().with_arena_allocator(false).build()]).map_err(err)?;
    let b = b.with_optimization_level(GraphOptimizationLevel::All).map_err(err)?;
    let b = b.with_intra_threads(threads).map_err(err)?;
    let mut b = b.with_inter_threads(1).map_err(err)?;
    b.commit_from_file(path).map_err(|e| BackendError(format!("{}: {e}", path.display())))
}

fn optional(dir: &Path, name: &str, threads: usize) -> Result<Option<Session>> {
    let p = dir.join(name);
    if p.exists() { session(&p, threads).map(Some) } else { Ok(None) }
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::from_array((shape, data.into_boxed_slice())).map_err(err)
}

fn i64_tensor(shape: Vec<usize>, data: Vec<i64>) -> Result<Tensor<i64>> {
    Tensor::from_array((shape, data.into_boxed_slice())).map_err(err)
}

fn extract_f32(v: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    let (s, d) = v.try_extract_tensor::<f32>().map_err(err)?;
    Ok((s.iter().map(|&x| x as usize).collect(), d.to_vec()))
}

fn extract_i64(v: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<i64>)> {
    let (s, d) = v.try_extract_tensor::<i64>().map_err(err)?;
    Ok((s.iter().map(|&x| x as usize).collect(), d.to_vec()))
}

/// A line-crop model run in width batches: (padded batch, widths) per batch.
fn run_batches<T>(
    lines: &[&ImageF],
    batches: Vec<Vec<usize>>,
    mut run: impl FnMut(&[&ImageF], Vec<f32>, usize, Vec<i64>) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    let mut out: Vec<Option<T>> = (0..lines.len()).map(|_| None).collect();
    for idx in batches {
        let part: Vec<&ImageF> = idx.iter().map(|&j| lines[j]).collect();
        let (arr, wm, widths) = pad_batch(&part, WIDTH_MULTIPLE);
        for (j, r) in idx.iter().zip(run(&part, arr, wm, widths)?) {
            out[*j] = Some(r);
        }
    }
    Ok(out.into_iter().map(|r| r.expect("every line is in a batch")).collect())
}

/// Lines per recognizer batch (pipeline.py: ``batch=64``).
pub const REC_BATCH: usize = 64;

pub struct OrtBackend {
    det: Session,
    rec: Session,
    rec_cjk: Option<Session>,
    script: Option<Session>,
    orient: Option<Session>,
    order: Option<Session>,
    mask: Option<Session>,
    math: Option<OrtMath>,
}

impl OrtBackend {
    /// The sessions of the models in ``dir`` (``math``: also the math models).
    pub fn new(dir: &Path, threads: usize, math: bool) -> Result<Self> {
        let need = |name: &str| {
            let p = dir.join(name);
            if p.exists() { session(&p, threads) } else { Err(BackendError(format!("{} not found", p.display()))) }
        };
        let (script, rec_cjk) = if dir.join("script.onnx").exists() && dir.join("rec_cjk.onnx").exists() {
            (Some(need("script.onnx")?), Some(need("rec_cjk.onnx")?))
        } else {
            (None, None)
        };
        let rec = if dir.join("rec.int8.onnx").exists() { need("rec.int8.onnx")? } else { need("rec.onnx")? };
        Ok(OrtBackend {
            det: need("det.onnx")?,
            rec,
            rec_cjk,
            script,
            orient: optional(dir, "orient.onnx", threads)?,
            order: optional(dir, "order.onnx", threads)?,
            mask: optional(dir, "mask.onnx", threads)?,
            math: if math { Some(OrtMath::new(dir, threads)?) } else { None },
        })
    }
}

impl Backend for OrtBackend {
    fn name(&self) -> &'static str {
        "ort"
    }

    fn det(&mut self, x: &ImageF) -> Result<DetOut> {
        let t = f32_tensor(vec![1, 1, x.h, x.w], x.data.clone())?;
        let out = self.det.run(ort::inputs!["x" => t]).map_err(err)?;
        let (s, d) = extract_f32(&out[0])?;
        Ok(DetOut { h: s[2], w: s[3], data: d })
    }

    fn rec(&mut self, model: RecModel, lines: &[&ImageF]) -> Result<Vec<RecOut>> {
        let sess = match model {
            RecModel::Main => &mut self.rec,
            RecModel::Cjk => self.rec_cjk.as_mut().ok_or_else(|| BackendError("no rec_cjk.onnx".into()))?,
        };
        let widths: Vec<usize> = lines.iter().map(|l| l.w).collect();
        run_batches(lines, rec_batches(&widths, REC_BATCH), |part, arr, wm, widths| {
            let x = f32_tensor(vec![part.len(), 1, part[0].h, wm], arr)?;
            let w = i64_tensor(vec![part.len()], widths)?;
            let out = sess.run(ort::inputs!["x" => x, "widths" => w]).map_err(err)?;
            let (s, ids) = extract_i64(&out["ids"])?;
            let (_, conf) = extract_f32(&out["conf"])?;
            let (_, lengths) = extract_i64(&out["lengths"])?;
            let t = s[1];
            let stride = wm as f64 / t.max(1) as f64;
            Ok((0..part.len())
                .map(|k| {
                    let n = (lengths[k].max(0) as usize).min(t);
                    RecOut { ids: ids[k * t..k * t + n].iter().map(|&v| v as u32).collect(), conf: conf[k * t..k * t + n].to_vec(), stride }
                })
                .collect())
        })
    }

    fn has_cjk(&self) -> bool {
        self.script.is_some()
    }

    fn script(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>> {
        let sess = self.script.as_mut().ok_or_else(|| BackendError("no script.onnx".into()))?;
        let widths: Vec<usize> = lines.iter().map(|l| l.w).collect();
        run_batches(lines, rec_batches(&widths, REC_BATCH), |part, arr, wm, widths| {
            let x = f32_tensor(vec![part.len(), 1, part[0].h, wm], arr)?;
            let w = i64_tensor(vec![part.len()], widths)?;
            let out = sess.run(ort::inputs!["x" => x, "widths" => w]).map_err(err)?;
            Ok(extract_f32(&out[0])?.1)
        })
    }

    fn has_orient(&self) -> bool {
        self.orient.is_some()
    }

    fn orient(&mut self, x: &ImageF) -> Result<[f32; 4]> {
        let sess = self.orient.as_mut().ok_or_else(|| BackendError("no orient.onnx".into()))?;
        let t = f32_tensor(vec![1, 1, x.h, x.w], x.data.clone())?;
        let out = sess.run(ort::inputs!["x" => t]).map_err(err)?;
        let (_, p) = extract_f32(&out[0])?;
        Ok([p[0], p[1], p[2], p[3]])
    }

    fn has_order(&self) -> bool {
        self.order.is_some()
    }

    fn order(&mut self, feats: &[f32], n: usize) -> Result<OrderOut> {
        let sess = self.order.as_mut().ok_or_else(|| BackendError("no order.onnx".into()))?;
        let f = f32_tensor(vec![1, n, feats.len() / n.max(1)], feats.to_vec())?;
        let m: Tensor<bool> = Tensor::from_array((vec![1usize, n], vec![true; n].into_boxed_slice())).map_err(err)?;
        let out = sess.run(ort::inputs!["feats" => f, "mask" => m]).map_err(err)?;
        let mat = |i: usize| -> Result<Mat> {
            let (s, d) = extract_f32(&out[i])?;
            Ok(Mat::new(s[1], s[2], d))
        };
        let a = mat(0)?;
        if out.len() < 3 {
            return Ok(OrderOut { a, cls: None, same: None, tab: None });
        }
        let tab = if out.len() > 3 {
            let (_, d) = extract_f32(&out[3])?;
            let nn = n * n;
            Some([0, 1, 2].map(|k| Mat::new(n, n, d[k * nn..(k + 1) * nn].to_vec())))
        } else {
            None
        };
        Ok(OrderOut { a, cls: Some(mat(1)?), same: Some(mat(2)?), tab })
    }

    fn has_mask(&self) -> bool {
        self.mask.is_some()
    }

    fn mask(&mut self, x: &[f32], b: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let sess = self.mask.as_mut().ok_or_else(|| BackendError("no mask.onnx".into()))?;
        let t = f32_tensor(vec![b, 1, h, w], x.to_vec())?;
        let out = sess.run(ort::inputs!["x" => t]).map_err(err)?;
        Ok(extract_f32(&out[0])?.1)
    }

    fn math(&mut self) -> Option<&mut dyn MathBackend> {
        self.math.as_mut().map(|m| m as &mut dyn MathBackend)
    }
}

/// The math models on ONNX Runtime, with the decoder caches of the current batch.
pub struct OrtMath {
    cls: Session,
    enc: Session,
    init: Session,
    step: Session,
    init_inputs: Vec<String>,
    step_inputs: Vec<String>,
    state: Option<MathCaches>,
}

struct MathCaches {
    b: usize,
    mask: (Vec<usize>, Vec<f32>),
    cross_k: (Vec<usize>, Vec<f32>),
    cross_v: (Vec<usize>, Vec<f32>),
    self_k: (Vec<usize>, Vec<f32>),
    self_v: (Vec<usize>, Vec<f32>),
}

impl OrtMath {
    fn new(dir: &Path, threads: usize) -> Result<Self> {
        let need = |name: &str| session(&dir.join(name), threads);
        let init = need("math_dec_init.onnx")?;
        let step = need("math_dec_step.onnx")?;
        let names = |s: &Session| s.inputs().iter().map(|i| i.name().to_string()).collect::<Vec<_>>();
        Ok(OrtMath {
            cls: need("mathcls.onnx")?,
            enc: need("math_enc.onnx")?,
            init_inputs: names(&init),
            step_inputs: names(&step),
            init,
            step,
            state: None,
        })
    }
}

impl MathBackend for OrtMath {
    fn classify(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>> {
        let widths: Vec<usize> = lines.iter().map(|l| l.w).collect();
        let sess = &mut self.cls;
        run_batches(lines, rec_batches(&widths, REC_BATCH), |part, arr, wm, widths| {
            let x = f32_tensor(vec![part.len(), 1, part[0].h, wm], arr)?;
            let w = i64_tensor(vec![part.len()], widths)?;
            let out = sess.run(ort::inputs!["x" => x, "widths" => w]).map_err(err)?;
            Ok(extract_f32(&out[0])?.1)
        })
    }

    fn begin(&mut self, lines: &[&ImageF]) -> Result<()> {
        let b = lines.len();
        let (arr, wm, widths) = pad_batch(lines, WIDTH_MULTIPLE);
        let x = f32_tensor(vec![b, 1, lines[0].h, wm], arr)?;
        let w = i64_tensor(vec![b], widths)?;
        let out = self.enc.run(ort::inputs!["x" => x, "widths" => w]).map_err(err)?;
        let memory = extract_f32(&out["memory"])?;
        let mask = extract_f32(&out["mask"])?;
        drop(out);
        let mut feed: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![];
        feed.push(("memory".into(), f32_tensor(memory.0.clone(), memory.1.clone())?.into()));
        if self.init_inputs.iter().any(|n| n == "mask") {
            feed.push(("mask".into(), f32_tensor(mask.0.clone(), mask.1.clone())?.into()));
        }
        let out = self.init.run(feed).map_err(err)?;
        let cross_k = extract_f32(&out[0])?;
        let cross_v = extract_f32(&out[1])?;
        // self caches start empty: [layers, B, heads, 0, head_dim]
        let mut sk = cross_k.0.clone();
        sk[3] = 0;
        self.state = Some(MathCaches { b, mask, cross_k, cross_v, self_k: (sk.clone(), vec![]), self_v: (sk, vec![]) });
        Ok(())
    }

    fn step(&mut self, tokens: &[i64], pos: i64) -> Result<Vec<f32>> {
        let st = self.state.as_mut().ok_or_else(|| BackendError("math: step before begin".into()))?;
        let mut feed: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![];
        for name in &self.step_inputs {
            let v: ort::session::SessionInputValue = match name.as_str() {
                "token" => i64_tensor(vec![st.b], tokens.to_vec())?.into(),
                "pos" => i64_tensor(vec![], vec![pos])?.into(),
                "self_k" => f32_tensor(st.self_k.0.clone(), st.self_k.1.clone())?.into(),
                "self_v" => f32_tensor(st.self_v.0.clone(), st.self_v.1.clone())?.into(),
                "cross_k" => f32_tensor(st.cross_k.0.clone(), st.cross_k.1.clone())?.into(),
                "cross_v" => f32_tensor(st.cross_v.0.clone(), st.cross_v.1.clone())?.into(),
                "mask" => f32_tensor(st.mask.0.clone(), st.mask.1.clone())?.into(),
                other => return Err(BackendError(format!("math_dec_step.onnx: unknown input {other}"))),
            };
            feed.push((name.clone().into(), v));
        }
        let out = self.step.run(feed).map_err(err)?;
        let logits = extract_f32(&out[0])?.1;
        st.self_k = extract_f32(&out[1])?;
        st.self_v = extract_f32(&out[2])?;
        Ok(logits)
    }
}
