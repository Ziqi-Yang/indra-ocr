//! Model files (models.py): a local folder, or a download from the Hugging Face
//! repo (cached in the same layout as Python's huggingface_hub).

use std::path::{Path, PathBuf};

use crate::vocab::Vocab;

pub const HF_REPO: &str = "datalab-to/indra-ocr";
/// Model set this crate was built and tested against (a tag in HF_REPO).
pub const HF_REVISION: &str = "v3.1";
pub const FILES: [&str; 4] = ["det.onnx", "rec.onnx", "order.onnx", "meta.json"];
pub const MATH_FILES: [&str; 5] =
    ["mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json"];

/// The models are not available: a download failed, or a local model folder
/// lacks files.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ModelError(pub String);

fn missing(dir: &Path, math: bool) -> Vec<&'static str> {
    let mut need: Vec<&str> = vec![FILES[0], FILES[1], FILES[3]];
    if math {
        need.extend(MATH_FILES);
    }
    need.into_iter().filter(|f| !dir.join(f).exists()).collect()
}

/// Folder with det.onnx, rec.onnx, order.onnx and meta.json (and, with
/// ``math``, the math models).
///
/// Order of lookup: ``path``, the INDRA_OCR_MODEL_DIR environment variable, then
/// the Hugging Face cache, then a download of HF_REPO.
pub fn model_dir(path: Option<&Path>, revision: Option<&str>, math: bool) -> Result<PathBuf, ModelError> {
    let env = std::env::var_os("INDRA_OCR_MODEL_DIR").filter(|v| !v.is_empty());
    if let Some(p) = path.map(Path::to_path_buf).or(env.map(PathBuf::from)) {
        let m = missing(&p, math);
        if !m.is_empty() {
            return Err(ModelError(format!("the model folder {} lacks {}", p.display(), m.join(", "))));
        }
        return Ok(p);
    }
    let rev = revision.unwrap_or(HF_REVISION);
    if let Some(dir) = cached(rev).filter(|d| missing(d, math).is_empty() && d.join("order.onnx").exists()) {
        return Ok(dir);
    }
    download(rev, math)
}

fn hf_cache_root() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default();
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

/// The cached snapshot of ``rev``, if the cache has one.
fn cached(rev: &str) -> Option<PathBuf> {
    let repo = hf_cache_root().join(format!("models--{}", HF_REPO.replace('/', "--")));
    let commit = std::fs::read_to_string(repo.join("refs").join(rev)).ok()?;
    let dir = repo.join("snapshots").join(commit.trim());
    dir.is_dir().then_some(dir)
}

#[cfg(feature = "download")]
fn download(rev: &str, math: bool) -> Result<PathBuf, ModelError> {
    use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
    let fail = |e: &dyn std::fmt::Display| {
        let reason: String = e.to_string().lines().next().unwrap_or("").chars().take(200).collect();
        ModelError(format!(
            "could not download the models from Hugging Face ({HF_REPO}, {rev}): {reason}. Check the network \
             connection, or point at a local model folder with --model-dir or INDRA_OCR_MODEL_DIR."
        ))
    };
    let api = ApiBuilder::from_env().with_cache_dir(hf_cache_root()).with_progress(false).build().map_err(|e| fail(&e))?;
    let repo = api.repo(Repo::with_revision(HF_REPO.into(), RepoType::Model, rev.into()));
    let info = repo.info().map_err(|e| fail(&e))?;
    let mut dir = None;
    for s in info.siblings {
        let f = s.rfilename;
        let is_math = MATH_FILES.contains(&f.as_str());
        let wanted = (f.ends_with(".onnx") || f == "meta.json") && (math || !is_math) || (math && is_math);
        if wanted {
            let p = repo.get(&f).map_err(|e| fail(&e))?;
            dir = p.parent().map(Path::to_path_buf);
        }
    }
    let dir = dir.ok_or_else(|| fail(&"the repository has no model files"))?;
    let m = missing(&dir, math);
    if !m.is_empty() {
        return Err(fail(&format!("the download lacks {}", m.join(", "))));
    }
    Ok(dir)
}

#[cfg(not(feature = "download"))]
fn download(rev: &str, _math: bool) -> Result<PathBuf, ModelError> {
    Err(ModelError(format!(
        "no cached models for {HF_REPO} {rev}, and this build cannot download them: pass --model-dir or set \
         INDRA_OCR_MODEL_DIR"
    )))
}

/// meta.json: the detector's input size, the line height and the vocabularies.
#[derive(Clone, Debug)]
pub struct Meta {
    pub det_long_side: usize,
    pub rec_height: usize,
    pub vocab: Vocab,
    pub vocab_cjk: Option<Vocab>,
}

impl Meta {
    pub fn load(dir: &Path) -> Result<Meta, ModelError> {
        let p = dir.join("meta.json");
        let s = std::fs::read_to_string(&p).map_err(|e| ModelError(format!("{}: {e}", p.display())))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| ModelError(format!("{}: {e}", p.display())))?;
        let int = |k: &str| {
            v[k].as_u64().map(|x| x as usize).ok_or_else(|| ModelError(format!("{}: no {k}", p.display())))
        };
        let vocab = |k: &str| -> Result<Option<Vocab>, ModelError> {
            match v.get(k).and_then(|x| x.as_str()) {
                None => Ok(None),
                Some(s) => Vocab::from_json(s).map(Some).map_err(|e| ModelError(format!("{}: {k}: {e}", p.display()))),
            }
        };
        Ok(Meta {
            det_long_side: int("det_long_side")?,
            rec_height: int("rec_height")?,
            vocab: vocab("vocab")?.ok_or_else(|| ModelError(format!("{}: no vocab", p.display())))?,
            vocab_cjk: vocab("vocab_cjk")?,
        })
    }
}
