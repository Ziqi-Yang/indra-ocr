//! The models on the native engine (indra-engine). The one model it does not
//! implement, CenterMask (mask.onnx, absent from current model sets), runs on
//! ONNX Runtime when a model folder has it.

use std::path::Path;

use indra_engine::classifier::LineClassifier;
use indra_engine::cnn::{Feat, Net, NetOut};
use indra_engine::math::{DecodeState, MathDecoder, MathEncoder};
use indra_engine::order::OrderNet;
use indra_engine::par::Par;
use indra_engine::rec::Recognizer;

use crate::backend::{Backend, BackendError, DetOut, MathBackend, OrderOut, RecModel, RecOut, Result};
use crate::imgproc::ImageF;

fn err(e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("native engine: {e}"))
}

pub struct NativeBackend {
    par: std::sync::Arc<Par>,
    det: Net,
    orient: Option<Net>,
    rec: Recognizer,
    rec_cjk: Option<Recognizer>,
    script: Option<LineClassifier>,
    order: Option<OrderNet>,
    math: Option<NativeMath>,
    #[cfg(feature = "ort")]
    ort: Option<crate::ort_backend::OrtBackend>,
}

impl NativeBackend {
    pub fn new(dir: &Path, threads: usize, math: bool) -> Result<Self> {
        let par = std::sync::Arc::new(Par::new(threads));
        let rec_path = if dir.join("rec.int8.onnx").exists() { dir.join("rec.int8.onnx") } else { dir.join("rec.onnx") };
        let rec = Recognizer::load(&rec_path).map_err(err)?;
        let (rec_cjk, script) = if dir.join("script.onnx").exists() && dir.join("rec_cjk.onnx").exists() {
            (Some(Recognizer::load(&dir.join("rec_cjk.onnx")).map_err(err)?), Some(LineClassifier::load(&dir.join("script.onnx")).map_err(err)?))
        } else {
            (None, None)
        };
        let orient = if dir.join("orient.onnx").exists() { Some(Net::load(&dir.join("orient.onnx")).map_err(err)?) } else { None };
        Ok(NativeBackend {
            par: par.clone(),
            det: Net::load(&dir.join("det.onnx")).map_err(err)?,
            orient,
            rec,
            rec_cjk,
            script,
            math: if math { Some(NativeMath::load(dir, par)?) } else { None },
            order: if dir.join("order.onnx").exists() { Some(OrderNet::load(&dir.join("order.onnx")).map_err(err)?) } else { None },
            #[cfg(feature = "ort")]
            ort: if dir.join("mask.onnx").exists() { Some(crate::ort_backend::OrtBackend::new(dir, threads, false)?) } else { None },
        })
    }

    fn fallback(&mut self) -> Result<&mut dyn Backend> {
        #[cfg(feature = "ort")]
        if let Some(o) = self.ort.as_mut() {
            return Ok(o);
        }
        Err(BackendError("this model needs ONNX Runtime (feature \"ort\")".into()))
    }
}

/// A line classifier on line crops, each run at its width batch's padded
/// width (as ONNX Runtime runs it, since the result depends on the padding).
pub fn classify(c: &LineClassifier, lines: &[&ImageF], par: &Par) -> Vec<f32> {
    let widths: Vec<usize> = lines.iter().map(|l| l.w).collect();
    // each line with its batch's padded width
    let mut padded = vec![0usize; lines.len()];
    for batch in crate::crop::rec_batches(&widths, crate::ort_backend_batch()) {
        let wm = batch.iter().map(|&j| widths[j]).max().unwrap_or(0).div_ceil(crate::crop::WIDTH_MULTIPLE) * crate::crop::WIDTH_MULTIPLE;
        for j in batch {
            padded[j] = wm;
        }
    }
    par.map(lines.len(), |j| c.run(&lines[j].data, lines[j].h, lines[j].w, padded[j]))
}

impl Backend for NativeBackend {
    fn name(&self) -> &'static str {
        "native"
    }

    fn det(&mut self, x: &ImageF) -> Result<DetOut> {
        let out = self.det.run(Feat { h: x.h, w: x.w, c: 1, data: x.data.clone() }, &self.par).map_err(err)?;
        match out.into_iter().next() {
            Some(NetOut::Feat(f)) if f.c == 3 => Ok(DetOut { h: f.h, w: f.w, data: f.to_chw() }),
            _ => Err(err("det.onnx did not give a [3, h, w] map")),
        }
    }

    fn rec(&mut self, model: RecModel, lines: &[&ImageF]) -> Result<Vec<RecOut>> {
        let r = match model {
            RecModel::Main => &self.rec,
            RecModel::Cjk => self.rec_cjk.as_ref().ok_or_else(|| BackendError("no rec_cjk.onnx".into()))?,
        };
        let input: Vec<(&[f32], usize)> = lines.iter().map(|l| (l.data.as_slice(), l.w)).collect();
        let stride = r.stride() as f64;
        Ok(r.run(&input, &self.par).into_iter().map(|o| RecOut { ids: o.ids, conf: o.conf, stride }).collect())
    }

    fn has_cjk(&self) -> bool {
        self.rec_cjk.is_some()
    }

    fn script(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>> {
        let c = self.script.as_ref().ok_or_else(|| BackendError("no script.onnx".into()))?;
        Ok(classify(c, lines, &self.par))
    }

    fn has_orient(&self) -> bool {
        self.orient.is_some()
    }

    fn orient(&mut self, x: &ImageF) -> Result<[f32; 4]> {
        let net = self.orient.as_ref().ok_or_else(|| BackendError("no orient.onnx".into()))?;
        let out = net.run(Feat { h: x.h, w: x.w, c: 1, data: x.data.clone() }, &self.par).map_err(err)?;
        match out.into_iter().next() {
            Some(NetOut::Vec(p)) if p.len() == 4 => Ok([p[0], p[1], p[2], p[3]]),
            _ => Err(err("orient.onnx did not give 4 probabilities")),
        }
    }

    fn has_order(&self) -> bool {
        self.order.is_some()
    }

    fn order(&mut self, feats: &[f32], n: usize) -> Result<OrderOut> {
        use crate::order_post::Mat;
        let net = self.order.as_ref().ok_or_else(|| BackendError("no order.onnx".into()))?;
        let o = net.run(feats, n);
        let nn = n * n;
        Ok(OrderOut {
            a: Mat::new(n, n, o.a),
            cls: Some(Mat::new(n, o.n_cls, o.cls)),
            same: Some(Mat::new(n, n, o.same)),
            tab: Some([0, 1, 2].map(|k| Mat::new(n, n, o.tab[k * nn..(k + 1) * nn].to_vec()))),
        })
    }

    fn has_mask(&self) -> bool {
        #[cfg(feature = "ort")]
        return self.ort.as_ref().is_some_and(|o| o.has_mask());
        #[cfg(not(feature = "ort"))]
        false
    }

    fn mask(&mut self, x: &[f32], b: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        self.fallback()?.mask(x, b, h, w)
    }

    fn math(&mut self) -> Option<&mut dyn MathBackend> {
        self.math.as_mut().map(|m| m as &mut dyn MathBackend)
    }
}

/// The math models on the native engine.
pub struct NativeMath {
    par: std::sync::Arc<Par>,
    cls: LineClassifier,
    enc: MathEncoder,
    dec: MathDecoder,
    state: Option<DecodeState>,
}

impl NativeMath {
    pub fn load(dir: &Path, par: std::sync::Arc<Par>) -> Result<Self> {
        Ok(NativeMath {
            par,
            cls: LineClassifier::load(&dir.join("mathcls.onnx")).map_err(err)?,
            enc: MathEncoder::load(&dir.join("math_enc.onnx")).map_err(err)?,
            dec: MathDecoder::load(&dir.join("math_dec_step.onnx")).map_err(err)?,
            state: None,
        })
    }
}

impl MathBackend for NativeMath {
    fn classify(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>> {
        Ok(classify(&self.cls, lines, &self.par))
    }

    fn begin(&mut self, lines: &[&ImageF]) -> Result<()> {
        // the batch's padded width, as ONNX Runtime runs it
        let wm = lines.iter().map(|l| l.w).max().unwrap_or(0).div_ceil(crate::crop::WIDTH_MULTIPLE) * crate::crop::WIDTH_MULTIPLE;
        let mem = self.par.map(lines.len(), |i| self.enc.encode(&lines[i].data, lines[i].w, wm));
        self.state = Some(self.dec.begin(mem));
        Ok(())
    }

    fn step(&mut self, tokens: &[i64], pos: i64) -> Result<Vec<f32>> {
        let st = self.state.as_mut().ok_or_else(|| BackendError("math: step before begin".into()))?;
        Ok(self.dec.step(st, tokens, pos.max(0) as usize))
    }
}
