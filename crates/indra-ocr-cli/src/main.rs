//! indra-ocr: PDFs or images in, markdown (or text, or JSON lines) out.
//!
//!   indra-ocr paper.pdf                    # markdown to stdout
//!   indra-ocr *.pdf --out results/         # one .md per input
//!   indra-ocr scan.png --format json       # lines with boxes, confidences, timings
//!
//! Pages run in parallel on all the cores the process may use (--cores to limit).

use std::collections::HashSet;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use indra_ocr::{ConvertOptions, LoadOptions, Mode, OcrOptions, convert_many};

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Md,
    Text,
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Auto,
    Ocr,
    Text,
}

#[derive(Parser)]
#[command(name = "indra-ocr", version, about = "PDFs or images in, markdown (or text, or JSON lines) out.")]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// write one file per input here
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "md")]
    format: Format,
    /// auto: text layer where it passes the checks, OCR elsewhere; ocr: never read
    /// the text layer; text: use the text layer even when the checks fail
    #[arg(long, value_enum, default_value = "auto", help_heading = "text layer or OCR (PDFs)")]
    mode: ModeArg,
    /// order text-layer lines with the order model, not the PDF content order
    #[arg(long, help_heading = "text layer or OCR (PDFs)")]
    model_order: bool,
    /// on text-layer pages, do not OCR text inside images
    #[arg(long, help_heading = "text layer or OCR (PDFs)")]
    no_image_ocr: bool,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// cores to use (default: all this process may run on; with the native
    /// engine, one per physical core)
    #[arg(long, alias = "threads", default_value_t = 0)]
    cores: usize,
    #[arg(long, default_value_t = 200.0)]
    dpi: f64,
    #[arg(long, default_value_t = 0)]
    max_pages: usize,
    /// password of encrypted PDFs
    #[arg(long)]
    password: Option<String>,
    /// write math as LaTeX ($...$); slower on pages with math (off by default)
    #[arg(long)]
    math: bool,
    /// keep page headers and footers (page numbers, running titles) in the markdown
    #[arg(long)]
    keep_furniture: bool,
    /// print per-page timings to stderr
    #[arg(long)]
    timings: bool,
    /// inference engine: native (indra-engine) or ort (ONNX Runtime)
    #[arg(long, default_value = "native")]
    engine: indra_ocr::Engine,
    /// native engine: the recognizer in int8 on CPUs with AVX-512 VNNI (faster;
    /// about 0.1% of characters read differently)
    #[arg(long)]
    int8: bool,
}

/// Python's str(Path(p)): no leading "./".
fn py_path(p: &Path) -> String {
    let parts: Vec<Component> = p.components().collect();
    let parts = if parts.len() > 1 && parts[0] == Component::CurDir { &parts[1..] } else { &parts[..] };
    parts.iter().collect::<PathBuf>().display().to_string()
}

fn main() -> ExitCode {
    let a = Args::parse();
    if let Some(out) = &a.out
        && let Err(e) = std::fs::create_dir_all(out)
    {
        eprintln!("indra-ocr: {}: {e}", out.display());
        return ExitCode::from(1);
    }
    let opts = LoadOptions {
        model_dir: a.model_dir.clone(),
        math: a.math,
        ocr: OcrOptions { furniture: a.keep_furniture, ..Default::default() },
        convert: ConvertOptions {
            dpi: a.dpi,
            mode: match a.mode {
                ModeArg::Auto => Mode::Auto,
                ModeArg::Ocr => Mode::Ocr,
                ModeArg::Text => Mode::Text,
            },
            pdf_order: !a.model_order,
            image_ocr: !a.no_image_ocr,
        },
        engine: a.engine,
        int8: a.int8,
    };
    let ext = match a.format {
        Format::Md => ".md",
        Format::Text => ".txt",
        Format::Json => ".jsonl",
    };
    let mut failed = 0;
    let mut names: HashSet<String> = HashSet::new();
    let stdout = std::io::stdout();
    let res = convert_many(&a.inputs, a.cores, a.max_pages, a.password.as_deref(), &opts, |path, result| {
        let shown = py_path(path);
        let pages = match result {
            Err(e) => {
                eprintln!("indra-ocr: {e}");
                failed += 1;
                return;
            }
            Ok(p) => p,
        };
        let mut chunks = vec![];
        for (i, page) in &pages {
            if page.source == "error" {
                eprintln!("indra-ocr: {shown} page {}: {}", i + 1, page.note);
            }
            chunks.push(match a.format {
                Format::Json => {
                    let mut v = serde_json::Map::new();
                    v.insert("file".into(), shown.clone().into());
                    v.insert("page".into(), (*i).into());
                    if let serde_json::Value::Object(m) = page.to_json() {
                        v.extend(m);
                    }
                    indra_ocr::pyjson::dumps(&v)
                }
                Format::Md => page.markdown.clone().unwrap_or_else(|| page.text()),
                Format::Text => page.text(),
            });
            if a.timings {
                let t = page.timings.get("total_ms").unwrap_or(0.0);
                eprintln!("{shown} page {}: {}, {t:.0} ms {}", i + 1, page.source, page.note);
            }
        }
        let sep = if matches!(a.format, Format::Json) { "\n" } else { "\n\n" };
        let body = chunks.join(sep) + "\n";
        if let Some(out) = &a.out {
            let stem = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().to_string());
            let (mut name, mut k) = (stem.clone(), 1);
            while names.contains(&name) {
                // two inputs with one name: keep both outputs
                k += 1;
                name = format!("{stem}_{k}");
            }
            names.insert(name.clone());
            if let Err(e) = std::fs::write(out.join(name + ext), body) {
                eprintln!("indra-ocr: {e}");
                failed += 1;
            }
        } else {
            let mut o = stdout.lock();
            let _ = o.write_all(body.as_bytes());
            let _ = o.flush();
        }
    });
    if let Err(e) = res {
        eprintln!("indra-ocr: {e}");
        return ExitCode::from(1);
    }
    if std::env::var_os("INDRA_OCR_PROFILE").is_some() {
        let ns = indra_ocr::pdfium::LOCK_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("indra-ocr: pdfium lock wait {:.0} ms (all threads)", ns as f64 / 1e6);
    }
    if failed > 0 {
        eprintln!("indra-ocr: {failed} of {} inputs failed", a.inputs.len());
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
