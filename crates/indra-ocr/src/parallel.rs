//! Many pages on many cores (parallel.py): one single-threaded pipeline per
//! worker thread, one page per task, results in input order.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use crate::document::{ConvertOptions, Converter};
use crate::io::{InputError, is_pdf};
use crate::pipeline::{Ocr, OcrError, OcrOptions, Page};

/// Inference threads in each worker.
pub const THREADS_PER_WORKER: usize = 1;
/// Fewer pages: one pipeline, all cores on each page.
pub const MIN_POOL_PAGES: usize = 4;
/// Bytes a worker can reach (giant scans); caps the pool.
pub const WORKER_MEMORY: f64 = 1.5e9;

/// Which inference engine runs the models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Engine {
    /// ONNX Runtime (the reference)
    #[default]
    Ort,
}

/// Everything ``load`` needs to build a pipeline.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// model folder (default: INDRA_OCR_MODEL_DIR, the HF cache, or a download)
    pub model_dir: Option<PathBuf>,
    pub math: bool,
    pub ocr: OcrOptions,
    pub convert: ConvertOptions,
    pub engine: Engine,
}

/// A ready Converter with ``threads`` inference threads.
pub fn load(opts: &LoadOptions, threads: usize) -> Result<Converter, OcrError> {
    let dir = crate::models::model_dir(opts.model_dir.as_deref(), None, opts.math)?;
    let ocr = match opts.engine {
        #[cfg(feature = "ort")]
        Engine::Ort => Ocr::with_ort(&dir, threads.max(1), opts.ocr.clone(), opts.math)?,
        #[cfg(not(feature = "ort"))]
        Engine::Ort => return Err(OcrError::Other("this build has no ONNX Runtime engine (feature \"ort\")".into())),
    };
    Ok(Converter::new(ocr, opts.convert.clone()))
}

/// Cores this process may run on (CPU affinity counts).
pub fn usable_cores() -> usize {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            let n = libc::CPU_COUNT(&set);
            if n > 0 {
                return n as usize;
            }
        }
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn max_workers_for_memory() -> usize {
    #[cfg(unix)]
    unsafe {
        let (ps, np) = (libc::sysconf(libc::_SC_PAGESIZE), libc::sysconf(libc::_SC_PHYS_PAGES));
        if ps > 0 && np > 0 {
            return ((ps as f64 * np as f64) / WORKER_MEMORY).max(1.0) as usize;
        }
    }
    1 << 30
}

/// One unit of work: a page of a PDF, or a whole image file.
#[derive(Clone, Debug)]
enum Unit {
    PdfPage { file: usize, page: usize },
    Image { file: usize },
}

/// Per input: its work units, or the InputError that stops it.
fn plan(paths: &[PathBuf], max_pages: usize, password: Option<&str>) -> Vec<Result<Vec<Unit>, InputError>> {
    paths
        .iter()
        .enumerate()
        .map(|(file, p)| {
            if !p.is_file() {
                return Err(InputError(format!("{}: no such file", p.display())));
            }
            if !is_pdf(p) {
                return Ok(vec![Unit::Image { file }]);
            }
            #[cfg(feature = "pdf")]
            {
                let doc = crate::pdfium::Document::open(p, password)?;
                let n = doc.page_count();
                let n = if max_pages > 0 { n.min(max_pages) } else { n };
                Ok((0..n).map(|page| Unit::PdfPage { file, page }).collect())
            }
            #[cfg(not(feature = "pdf"))]
            {
                let _ = (password, max_pages);
                Err(InputError(format!("{}: this build reads no PDFs (feature \"pdf\")", p.display())))
            }
        })
        .collect()
}

pub type FileResult = Result<Vec<(usize, Page)>, InputError>;

/// Convert ``paths`` on ``cores`` cores (0: all usable), calling ``f`` with each
/// input's pages (or its InputError) in input order. With fewer than
/// MIN_POOL_PAGES pages, one pipeline uses all the cores on each page.
pub fn convert_many(
    paths: &[PathBuf],
    cores: usize,
    max_pages: usize,
    password: Option<&str>,
    opts: &LoadOptions,
    mut f: impl FnMut(&Path, FileResult),
) -> Result<(), OcrError> {
    // Resolve (and on a first run download) the models once, not in every worker.
    let mut opts = opts.clone();
    opts.model_dir = Some(crate::models::model_dir(opts.model_dir.as_deref(), None, opts.math)?);
    let cores = if cores == 0 { usable_cores() } else { cores }.max(1);
    let plan = plan(paths, max_pages, password);
    let n_pages: usize = plan.iter().map(|u| u.as_ref().map_or(0, |v| v.len())).sum();
    let workers = (cores / THREADS_PER_WORKER).min(n_pages).min(max_workers_for_memory());
    if workers < 2 || n_pages < MIN_POOL_PAGES {
        let mut conv = load(&opts, cores)?;
        for (p, units) in paths.iter().zip(plan) {
            match units {
                Err(e) => f(p, Err(e)),
                Ok(_) => f(p, conv.convert(p, max_pages, password)),
            }
        }
        return Ok(());
    }
    pooled(paths, plan, workers, max_pages, password, &opts, f)
}

fn pooled(
    paths: &[PathBuf],
    plan: Vec<Result<Vec<Unit>, InputError>>,
    workers: usize,
    max_pages: usize,
    password: Option<&str>,
    opts: &LoadOptions,
    mut f: impl FnMut(&Path, FileResult),
) -> Result<(), OcrError> {
    let units: Vec<(usize, Unit)> =
        plan.iter().filter_map(|u| u.as_ref().ok()).flatten().cloned().enumerate().collect();
    let queue = Arc::new(Mutex::new(units.into_iter()));
    let (tx, rx) = mpsc::channel::<(usize, Result<Vec<(usize, Page)>, InputError>)>();
    // Each worker builds its own pipeline; the first model error stops the run.
    let load_err: Mutex<Option<OcrError>> = Mutex::new(None);
    std::thread::scope(|s| {
        for wk in 0..workers {
            let (queue, tx, load_err) = (queue.clone(), tx.clone(), &load_err);
            s.spawn(move || {
                // this worker's own pdfium (see pdfium::instance)
                #[cfg(feature = "pdf")]
                let _ = crate::pdfium::use_instance(wk);
                #[cfg(not(feature = "pdf"))]
                let _ = wk;
                let mut conv = match load(opts, THREADS_PER_WORKER) {
                    Ok(c) => c,
                    Err(e) => {
                        load_err.lock().unwrap().get_or_insert(e);
                        // stop the others: nothing more to hand out
                        while queue.lock().unwrap().next().is_some() {}
                        return;
                    }
                };
                #[cfg(feature = "pdf")]
                let mut doc: Option<(usize, crate::pdfium::Document)> = None;
                loop {
                    let next = queue.lock().unwrap().next();
                    let Some((k, unit)) = next else { break };
                    let r = match unit {
                        Unit::Image { file } => conv.convert(&paths[file], max_pages, password),
                        #[cfg(feature = "pdf")]
                        Unit::PdfPage { file, page } => {
                            if doc.as_ref().is_none_or(|d| d.0 != file) {
                                doc = None;
                                match crate::pdfium::Document::open(&paths[file], password) {
                                    Ok(d) => doc = Some((file, d)),
                                    Err(e) => {
                                        let _ = tx.send((k, Err(e)));
                                        continue;
                                    }
                                }
                            }
                            Ok(vec![(page, conv.pdf_page(&doc.as_ref().unwrap().1, page))])
                        }
                        #[cfg(not(feature = "pdf"))]
                        Unit::PdfPage { .. } => unreachable!("no PDF units without the pdf feature"),
                    };
                    if tx.send((k, r)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        // Hand out the inputs in order as their units complete.
        let mut done: BTreeMap<usize, Result<Vec<(usize, Page)>, InputError>> = BTreeMap::new();
        let mut next_unit = 0;
        let mut files = paths.iter().zip(plan);
        let mut pending: Option<(&PathBuf, usize, Vec<(usize, Page)>, Option<InputError>)> = None;
        let mut emit = |done: &mut BTreeMap<usize, _>, next_unit: &mut usize, pending: &mut Option<_>| loop {
            if pending.is_none() {
                match files.next() {
                    None => return,
                    Some((p, Err(e))) => {
                        f(p, Err(e));
                        continue;
                    }
                    Some((p, Ok(u))) => *pending = Some((p, u.len(), vec![], None)),
                }
            }
            let (_, left, pages, err) = pending.as_mut().unwrap();
            while *left > 0 {
                let Some(r) = done.remove(next_unit) else { return };
                *next_unit += 1;
                *left -= 1;
                match r {
                    Ok(v) => pages.extend(v),
                    Err(e) => {
                        err.get_or_insert(e);
                    }
                }
            }
            let (p, _, pages, err) = pending.take().unwrap();
            f(p, match err {
                Some(e) => Err(e),
                None => Ok(pages),
            });
        };
        emit(&mut done, &mut next_unit, &mut pending);
        for (k, r) in rx {
            done.insert(k, r);
            emit(&mut done, &mut next_unit, &mut pending);
        }
        emit(&mut done, &mut next_unit, &mut pending);
    });
    match load_err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
