//! The models on the native engine (indra-engine). Models it does not run
//! natively yet run on ONNX Runtime.

use std::path::Path;

use indra_engine::cnn::{Feat, Net, NetOut};
use indra_engine::rec::Recognizer;

use crate::backend::{Backend, BackendError, DetOut, MathBackend, OrderOut, RecModel, RecOut, Result};
use crate::imgproc::ImageF;

fn err(e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("native engine: {e}"))
}

pub struct NativeBackend {
    det: Net,
    orient: Option<Net>,
    rec: Recognizer,
    rec_cjk: Option<Recognizer>,
    #[cfg(feature = "ort")]
    ort: crate::ort_backend::OrtBackend,
}

impl NativeBackend {
    pub fn new(dir: &Path, threads: usize, math: bool) -> Result<Self> {
        let rec_path = if dir.join("rec.int8.onnx").exists() { dir.join("rec.int8.onnx") } else { dir.join("rec.onnx") };
        let rec = Recognizer::load(&rec_path).map_err(err)?;
        let rec_cjk = if dir.join("script.onnx").exists() && dir.join("rec_cjk.onnx").exists() {
            Some(Recognizer::load(&dir.join("rec_cjk.onnx")).map_err(err)?)
        } else {
            None
        };
        let orient = if dir.join("orient.onnx").exists() { Some(Net::load(&dir.join("orient.onnx")).map_err(err)?) } else { None };
        Ok(NativeBackend {
            det: Net::load(&dir.join("det.onnx")).map_err(err)?,
            orient,
            rec,
            rec_cjk,
            #[cfg(feature = "ort")]
            ort: crate::ort_backend::OrtBackend::new(dir, threads, math)?,
        })
    }

    fn fallback(&mut self) -> Result<&mut dyn Backend> {
        #[cfg(feature = "ort")]
        return Ok(&mut self.ort);
        #[cfg(not(feature = "ort"))]
        Err(BackendError("this model needs ONNX Runtime (feature \"ort\")".into()))
    }
}

impl Backend for NativeBackend {
    fn name(&self) -> &'static str {
        "native"
    }

    fn det(&mut self, x: &ImageF) -> Result<DetOut> {
        let out = self.det.run(Feat { h: x.h, w: x.w, c: 1, data: x.data.clone() }).map_err(err)?;
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
        Ok(r.run(&input).into_iter().map(|o| RecOut { ids: o.ids, conf: o.conf, stride }).collect())
    }

    fn has_cjk(&self) -> bool {
        self.rec_cjk.is_some()
    }

    fn script(&mut self, lines: &[&ImageF]) -> Result<Vec<f32>> {
        self.fallback()?.script(lines)
    }

    fn has_orient(&self) -> bool {
        self.orient.is_some()
    }

    fn orient(&mut self, x: &ImageF) -> Result<[f32; 4]> {
        let net = self.orient.as_ref().ok_or_else(|| BackendError("no orient.onnx".into()))?;
        let out = net.run(Feat { h: x.h, w: x.w, c: 1, data: x.data.clone() }).map_err(err)?;
        match out.into_iter().next() {
            Some(NetOut::Vec(p)) if p.len() == 4 => Ok([p[0], p[1], p[2], p[3]]),
            _ => Err(err("orient.onnx did not give 4 probabilities")),
        }
    }

    fn has_order(&self) -> bool {
        #[cfg(feature = "ort")]
        return self.ort.has_order();
        #[cfg(not(feature = "ort"))]
        false
    }

    fn order(&mut self, feats: &[f32], n: usize) -> Result<OrderOut> {
        self.fallback()?.order(feats, n)
    }

    fn has_mask(&self) -> bool {
        #[cfg(feature = "ort")]
        return self.ort.has_mask();
        #[cfg(not(feature = "ort"))]
        false
    }

    fn mask(&mut self, x: &[f32], b: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        self.fallback()?.mask(x, b, h, w)
    }

    fn math(&mut self) -> Option<&mut dyn MathBackend> {
        #[cfg(feature = "ort")]
        return self.ort.math();
        #[cfg(not(feature = "ort"))]
        None
    }
}
