//! indra-ocr-viewer: a local web viewer. Drop in a PDF or an image, see what
//! Indra OCR reads.
//!
//!   indra-ocr-viewer                      # then open http://localhost:8765
//!
//! Each page shows its lines, layout blocks and reading order over the page,
//! the rendered markdown, the markdown source, the lines with their
//! confidences, the JSON the CLI writes, where the text came from and the time
//! of each stage. The pipeline options are set in the page and apply when the
//! document is read again; pages are read one at a time and appear as they
//! finish. With --tesseract (needs the tesseract binary) the page can also run
//! Tesseract on each page.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use base64::Engine as _;
use clap::Parser;
use indra_ocr::imgproc::{Gray, Interp, resize_u8, rot90_cw};
use indra_ocr::py::round1;
use indra_ocr::{ConvertOptions, Converter, Engine, LoadOptions, Mode, OcrOptions};
use serde_json::{Value, json};

const MAX_SIDE: f64 = 1600.0;
const TIMING_KEYS: [&str; 7] = ["text_ms", "det_ms", "rec_ms", "ocr_ms", "math_ms", "order_ms", "total_ms"];
const PAGE: &str = include_str!("../assets/viewer.html");
const LOGO: &str = include_str!("../assets/logo.b64");

#[derive(Parser)]
#[command(name = "indra-ocr-viewer", version, about = "Local viewer: drop in a PDF or an image, see what Indra OCR reads.")]
struct Args {
    /// folder with the models (default: download from Hugging Face)
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "auto")]
    mode: String,
    /// order text-layer lines with the order model, not the PDF content order
    #[arg(long)]
    model_order: bool,
    /// on text-layer pages, do not OCR text inside images
    #[arg(long)]
    no_image_ocr: bool,
    #[arg(long, default_value_t = 8765)]
    port: u16,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// initial PDF render resolution
    #[arg(long, default_value_t = 200.0)]
    dpi: f64,
    /// initial page limit per document
    #[arg(long, default_value_t = 20)]
    max_pages: usize,
    /// initially keep page headers and footers
    #[arg(long)]
    keep_furniture: bool,
    /// allow comparing with Tesseract
    #[arg(long)]
    tesseract: bool,
    #[arg(long, default_value = "eng")]
    tesseract_lang: String,
    /// initially write math as LaTeX
    #[arg(long)]
    math: bool,
    /// inference engine: native (indra-engine) or ort (ONNX Runtime)
    #[arg(long, default_value = "native")]
    engine: Engine,
}

struct Doc {
    id: String,
    pdf: Option<PathBuf>,
    image: Option<Gray>,
    pages: usize,
}

struct State {
    convs: HashMap<(bool, bool), Converter>,
    docs: VecDeque<Doc>,
}

struct Viewer {
    a: Args,
    defaults: Value,
    tmp: PathBuf,
    state: Mutex<State>,
}

fn new_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
    format!("{:012x}", h.finish() & 0xffff_ffff_ffff)
}

impl Viewer {
    /// The converter for these options (built on first use: the math models
    /// download then).
    fn conv<'a>(&self, st: &'a mut State, math: bool, furniture: bool) -> Result<&'a mut Converter, String> {
        if let std::collections::hash_map::Entry::Vacant(e) = st.convs.entry((math, furniture)) {
            let opts = LoadOptions {
                model_dir: self.a.model_dir.clone(),
                math,
                ocr: OcrOptions { furniture, ..Default::default() },
                convert: ConvertOptions::default(),
                engine: self.a.engine,
                int8: false,
            };
            let mut c = indra_ocr::load(&opts, self.a.threads).map_err(|e| e.to_string())?;
            // warm-up: the first call's allocations are not timed
            let _ = c.ocr.ocr(&Gray::filled(200, 200, 255), true);
            e.insert(c);
        }
        Ok(st.convs.get_mut(&(math, furniture)).unwrap())
    }

    fn upload(&self, data: Vec<u8>) -> Result<Value, String> {
        let id = new_id();
        let doc = if data.starts_with(b"%PDF-") {
            let path = self.tmp.join(format!("{id}.pdf"));
            std::fs::write(&path, &data).map_err(|e| e.to_string())?;
            let n = indra_ocr::pdfium::Document::open(&path, None).map_err(|e| e.to_string())?.page_count();
            Doc { id: id.clone(), pdf: Some(path), image: None, pages: n }
        } else {
            let mut frames = indra_ocr::io::decode_images(&data).ok_or("not a PDF or a readable image")?;
            Doc { id: id.clone(), pdf: None, image: Some(frames.swap_remove(0)), pages: 1 }
        };
        let pages = doc.pages;
        let mut st = self.state.lock().unwrap();
        st.docs.push_back(doc);
        while st.docs.len() > 5 {
            // keep the last few uploads only
            if let Some(old) = st.docs.pop_front()
                && let Some(p) = old.pdf
            {
                let _ = std::fs::remove_file(p);
            }
        }
        Ok(json!({"id": id, "pages": pages}))
    }

    fn page(&self, id: &str, n: usize, opts: &Value) -> Result<Value, String> {
        let mode: Mode = opts.get("mode").and_then(|v| v.as_str()).unwrap_or("auto").parse()?;
        let flag = |k: &str| opts.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let (math, furniture) = (flag("math"), flag("furniture"));
        let dpi = opts.get("dpi").and_then(|v| v.as_f64()).filter(|&d| d > 0.0).unwrap_or(200.0);
        let image_ocr = opts.get("image_ocr").and_then(|v| v.as_bool()).unwrap_or(true);
        let mut st = self.state.lock().unwrap();
        let (pdf, image) = {
            let d = st.docs.iter().find(|d| d.id == id).ok_or("unknown document")?;
            (d.pdf.clone(), d.image.clone())
        };
        let conv = self.conv(&mut st, math, furniture)?;
        conv.opts = ConvertOptions { dpi, mode, pdf_order: !flag("model_order"), image_ocr };
        let (gray, page, render_ms) = match (&pdf, image) {
            (Some(path), _) => {
                let doc = indra_ocr::pdfium::Document::open(path, None).map_err(|e| e.to_string())?;
                let t0 = Instant::now();
                // the converter's own scale, so the boxes match the picture
                let gray = doc.with_page(n, |api, p| p.render(api, conv.page_scale(p.size(api)))).map_err(|e| e.to_string())?;
                let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                (gray, conv.pdf_page(&doc, n), render_ms)
            }
            (None, Some(g)) => {
                let page = conv.ocr.ocr(&g, true).map_err(|e| e.to_string())?;
                (g, page, 0.0)
            }
            _ => return Err("empty document".into()),
        };
        // the boxes are in the upright page: show it
        let gray = if page.rotation != 0 { rot90_cw(&gray, (page.rotation / 90) as u32) } else { gray };
        let s = (MAX_SIDE / gray.w.max(gray.h) as f64).min(1.0);
        let small = if s < 1.0 {
            resize_u8(&gray, ((gray.w as f64 * s).round() as usize).max(1), ((gray.h as f64 * s).round() as usize).max(1), Interp::Area)
        } else {
            gray.clone()
        };
        let mut jpg = vec![];
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 82)
            .encode(&small.data, small.w as u32, small.h as u32, image::ExtendedColorType::L8)
            .map_err(|e| e.to_string())?;
        let scaled = |b: &[f64; 4]| b.iter().map(|v| round1(v * s)).collect::<Vec<_>>();
        let mut timings = serde_json::Map::new();
        timings.insert("render_ms".into(), json!(round1(render_ms)));
        for k in TIMING_KEYS {
            timings.insert(k.into(), json!(0.0));
        }
        for (k, v) in &page.timings.0 {
            timings.insert(k.clone(), json!(round1(*v)));
        }
        let mut rec = json!({
            "page": n + 1,
            "w": small.w,
            "h": small.h,
            "img": format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(&jpg)),
            "lines": page.lines.iter().map(|l| json!({"box": scaled(&l.bbox), "text": l.text, "conf": l.conf})).collect::<Vec<_>>(),
            "blocks": page.blocks.iter().map(|b| json!({"label": b.label, "box": scaled(&b.bbox), "text": b.text})).collect::<Vec<_>>(),
            "markdown": page.markdown.clone().unwrap_or_else(|| page.text()),
            "source": page.source,
            "note": page.note,
            "rotation": page.rotation,
            "skew": page.skew,
            "json": {
                "page": n,
                "source": page.source,
                "rotation": page.rotation,
                "skew": page.skew,
                "note": page.note,
                "lines": page.lines,
                "markdown": page.markdown,
                "blocks": page.blocks,
                "timings": page.timings,
            },
            "timings": Value::Object(timings),
        });
        if flag("tesseract") && self.a.tesseract {
            let t0 = Instant::now();
            rec["tesseract"] = json!(self.tesseract(&gray).unwrap_or_else(|e| format!("[tesseract failed: {e}]")));
            rec["timings"]["tesseract_ms"] = json!(round1(t0.elapsed().as_secs_f64() * 1000.0));
        }
        let total = rec["timings"]["total_ms"].as_f64().unwrap_or(0.0);
        println!(
            "page {}: {}, {} lines, {} blocks, {total:.0} ms ({}{})",
            n + 1,
            page.source,
            page.lines.len(),
            page.blocks.len(),
            opts.get("mode").and_then(|v| v.as_str()).unwrap_or("auto"),
            if math { ", math" } else { "" }
        );
        Ok(rec)
    }

    /// Tesseract's text for a page (the tesseract binary, --psm 3).
    fn tesseract(&self, gray: &Gray) -> Result<String, String> {
        let png = self.tmp.join(format!("{}.png", new_id()));
        image::save_buffer(&png, &gray.data, gray.w as u32, gray.h as u32, image::ExtendedColorType::L8).map_err(|e| e.to_string())?;
        let out = std::process::Command::new("tesseract")
            .arg(&png)
            .arg("stdout")
            .args(["-l", &self.a.tesseract_lang, "--psm", "3"])
            .env("OMP_THREAD_LIMIT", self.a.threads.to_string())
            .output();
        let _ = std::fs::remove_file(&png);
        let out = out.map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

fn respond(req: tiny_http::Request, code: u16, body: Vec<u8>, ctype: &str) {
    let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes()).unwrap();
    let _ = req.respond(tiny_http::Response::from_data(body).with_status_code(code).with_header(h));
}

fn handle(v: &Viewer, mut req: tiny_http::Request) {
    let url = req.url().to_string();
    match (req.method(), url.as_str()) {
        (tiny_http::Method::Get, "/" | "/index.html") => {
            respond(req, 200, PAGE.replace("__LOGO__", LOGO.trim()).into_bytes(), "text/html; charset=utf-8")
        }
        (tiny_http::Method::Get, "/info") => {
            let info = json!({"tesseract": v.a.tesseract, "version": env!("CARGO_PKG_VERSION"), "defaults": v.defaults});
            respond(req, 200, info.to_string().into_bytes(), "application/json")
        }
        (tiny_http::Method::Post, "/upload" | "/page") => {
            let mut body = vec![];
            let res = req.as_reader().read_to_end(&mut body).map_err(|e| e.to_string()).and_then(|_| {
                if url == "/upload" {
                    v.upload(body)
                } else {
                    let q: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
                    let id = q["id"].as_str().ok_or("no id")?.to_string();
                    let n = q["n"].as_u64().or_else(|| q["n"].as_str().and_then(|s| s.parse().ok())).ok_or("no page number")? as usize;
                    v.page(&id, n, q.get("opts").unwrap_or(&json!({})))
                }
            });
            match res {
                Ok(r) => respond(req, 200, r.to_string().into_bytes(), "application/json"),
                Err(e) => {
                    println!("error: {e}");
                    respond(req, 400, json!({"error": e}).to_string().into_bytes(), "application/json")
                }
            }
        }
        _ => respond(req, 404, b"not found".to_vec(), "text/plain"),
    }
}

fn main() {
    let a = Args::parse();
    let mode: Mode = match a.mode.parse() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("indra-ocr-viewer: {e}");
            std::process::exit(2);
        }
    };
    let _ = mode;
    let defaults = json!({
        "mode": a.mode,
        "pdf_order": !a.model_order,
        "image_ocr": !a.no_image_ocr,
        "math": a.math,
        "furniture": a.keep_furniture,
        "dpi": a.dpi,
        "max_pages": a.max_pages,
    });
    let tmp = std::env::temp_dir().join(format!("indra_ocr_viewer_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&tmp) {
        eprintln!("indra-ocr-viewer: {}: {e}", tmp.display());
        std::process::exit(1);
    }
    let port = a.port;
    let threads = a.threads;
    let v = Arc::new(Viewer { a, defaults, tmp, state: Mutex::new(State { convs: HashMap::new(), docs: VecDeque::new() }) });
    // the default pipeline now, so a model problem shows before the first upload
    {
        let mut st = v.state.lock().unwrap();
        if let Err(e) = v.conv(&mut st, v.a.math, v.a.keep_furniture) {
            eprintln!("indra-ocr-viewer: {e}");
            std::process::exit(1);
        }
    }
    let server = match tiny_http::Server::http(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("indra-ocr-viewer: port {port}: {e}");
            std::process::exit(1);
        }
    };
    println!("Indra OCR viewer on http://localhost:{port}  ({threads} threads)");
    for req in server.incoming_requests() {
        let v = v.clone();
        std::thread::spawn(move || handle(&v, req));
    }
}
