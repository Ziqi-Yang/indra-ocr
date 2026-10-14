//! The few pdfium functions the pipeline calls, loaded from the shared library
//! at run time. pdfium is not thread-safe: every call runs under its
//! instance's lock, and parallel workers each load their own instance (a
//! private copy of the library file, see ``instance``).
//!
//! The library is PDFIUM_LIB (a file), or libpdfium in PDFIUM_DYNAMIC_LIB_PATH
//! (a folder), next to the executable, or wherever the dynamic loader finds it.

#![allow(non_snake_case, non_camel_case_types)]

use std::ffi::{CString, c_char, c_double, c_float, c_int, c_uint, c_ulong, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::imgproc::Gray;
use crate::io::InputError;

type DOC = *mut c_void;
type PAGE = *mut c_void;
type TEXTPAGE = *mut c_void;
type BITMAP = *mut c_void;
type PAGEOBJ = *mut c_void;

pub const FPDF_ANNOT: c_int = 0x01;
pub const FPDF_GRAYSCALE: c_int = 0x08;
pub const FPDFBITMAP_GRAY: c_int = 1;
pub const FPDF_PAGEOBJ_TEXT: c_int = 1;
pub const FPDF_PAGEOBJ_FORM: c_int = 5;
pub const FPDF_TEXTRENDERMODE_INVISIBLE: c_int = 3;
pub const FPDF_ERR_PASSWORD: c_ulong = 4;

#[repr(C)]
pub struct LibraryConfig {
    version: c_int,
    user_font_paths: *const *const c_char,
    isolate: *mut c_void,
    v8_embedder_slot: c_uint,
    platform: *mut c_void,
    renderer_type: c_int,
}

macro_rules! pdfium_api {
    ($($name:ident : fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        pub struct Api {
            _lib: libloading::Library,
            $(pub $name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }
        impl Api {
            unsafe fn load(lib: libloading::Library) -> Result<Api, String> {
                Ok(Api {
                    $($name: unsafe {
                        *lib.get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(concat!(stringify!($name), "\0").as_bytes())
                            .map_err(|e| format!("pdfium lacks {}: {e}", stringify!($name)))?
                    },)*
                    _lib: lib,
                })
            }
        }
    };
}

pdfium_api! {
    FPDF_InitLibraryWithConfig: fn(*const LibraryConfig);
    FPDF_GetLastError: fn() -> c_ulong;
    FPDF_LoadDocument: fn(*const c_char, *const c_char) -> DOC;
    FPDF_CloseDocument: fn(DOC);
    FPDF_GetPageCount: fn(DOC) -> c_int;
    FPDF_LoadPage: fn(DOC, c_int) -> PAGE;
    FPDF_ClosePage: fn(PAGE);
    FPDF_GetPageWidthF: fn(PAGE) -> c_float;
    FPDF_GetPageHeightF: fn(PAGE) -> c_float;
    FPDF_PageToDevice: fn(PAGE, c_int, c_int, c_int, c_int, c_int, c_double, c_double, *mut c_int, *mut c_int) -> c_int;
    FPDFBitmap_CreateEx: fn(c_int, c_int, c_int, *mut c_void, c_int) -> BITMAP;
    FPDFBitmap_FillRect: fn(BITMAP, c_int, c_int, c_int, c_int, c_ulong) -> c_int;
    FPDFBitmap_GetBuffer: fn(BITMAP) -> *mut c_void;
    FPDFBitmap_GetStride: fn(BITMAP) -> c_int;
    FPDFBitmap_Destroy: fn(BITMAP);
    FPDF_RenderPageBitmap: fn(BITMAP, PAGE, c_int, c_int, c_int, c_int, c_int, c_int);
    FPDFText_LoadPage: fn(PAGE) -> TEXTPAGE;
    FPDFText_ClosePage: fn(TEXTPAGE);
    FPDFText_CountChars: fn(TEXTPAGE) -> c_int;
    FPDFText_GetUnicode: fn(TEXTPAGE, c_int) -> c_uint;
    FPDFText_GetCharAngle: fn(TEXTPAGE, c_int) -> c_float;
    FPDFText_GetCharBox: fn(TEXTPAGE, c_int, *mut c_double, *mut c_double, *mut c_double, *mut c_double) -> c_int;
    FPDFPage_CountObjects: fn(PAGE) -> c_int;
    FPDFPage_GetObject: fn(PAGE, c_int) -> PAGEOBJ;
    FPDFPageObj_GetType: fn(PAGEOBJ) -> c_int;
    FPDFFormObj_CountObjects: fn(PAGEOBJ) -> c_int;
    FPDFFormObj_GetObject: fn(PAGEOBJ, c_ulong) -> PAGEOBJ;
    FPDFTextObj_GetTextRenderMode: fn(PAGEOBJ) -> c_int;
}

// The function pointers are plain C entry points; the lock serializes their use.
unsafe impl Send for Api {}

fn candidates() -> Vec<PathBuf> {
    let name = if cfg!(target_os = "windows") {
        "pdfium.dll"
    } else if cfg!(target_os = "macos") {
        "libpdfium.dylib"
    } else {
        "libpdfium.so"
    };
    let mut c = vec![];
    if let Some(p) = std::env::var_os("PDFIUM_LIB") {
        c.push(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("PDFIUM_DYNAMIC_LIB_PATH") {
        c.push(PathBuf::from(p).join(name));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        c.push(dir.join(name));
    }
    c.push(PathBuf::from(name));
    c
}

fn load_from(p: &Path) -> Result<Api, String> {
    let lib = unsafe { libloading::Library::new(p) }.map_err(|e| format!("{}: {e}", p.display()))?;
    let api = unsafe { Api::load(lib)? };
    let cfg = LibraryConfig {
        version: 2,
        user_font_paths: std::ptr::null(),
        isolate: std::ptr::null_mut(),
        v8_embedder_slot: 0,
        platform: std::ptr::null_mut(),
        renderer_type: 0,
    };
    unsafe { (api.FPDF_InitLibraryWithConfig)(&cfg) };
    Ok(api)
}

/// The library file that loads, from the candidate locations.
fn find_library() -> Result<(PathBuf, Api), String> {
    let mut errs = vec![];
    for p in candidates() {
        match load_from(&p) {
            Ok(api) => {
                // the file the loader actually opened (for copies)
                let real = if p.is_file() { p.clone() } else { loaded_path(&p).unwrap_or(p.clone()) };
                return Ok((real, api));
            }
            Err(e) => errs.push(e),
        }
    }
    Err(format!("cannot load pdfium (set PDFIUM_LIB or PDFIUM_DYNAMIC_LIB_PATH): {}", errs.join("; ")))
}

/// Where the dynamic loader found a bare library name (Linux: /proc/self/maps).
fn loaded_path(name: &Path) -> Option<PathBuf> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let n = name.file_name()?.to_string_lossy().to_string();
    maps.lines().filter_map(|l| l.split_whitespace().nth(5)).find(|p| p.ends_with(&n)).map(PathBuf::from)
}

/// One loaded copy of pdfium with its own global state and its own lock.
pub struct Instance {
    api: Mutex<Api>,
}

impl Instance {
    /// The library, locked for the calling thread.
    pub fn lock(&self) -> MutexGuard<'_, Api> {
        let t0 = std::time::Instant::now();
        let g = self.api.lock().unwrap_or_else(|e| e.into_inner());
        LOCK_WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        g
    }
}

struct Pool {
    path: PathBuf,
    instances: Mutex<Vec<std::sync::Arc<Instance>>>,
}

static POOL: OnceLock<Result<Pool, String>> = OnceLock::new();

fn pool() -> Result<&'static Pool, InputError> {
    POOL.get_or_init(|| {
        let (path, api) = find_library()?;
        Ok(Pool { path, instances: Mutex::new(vec![std::sync::Arc::new(Instance { api: Mutex::new(api) })]) })
    })
    .as_ref()
    .map_err(|e| InputError(e.clone()))
}

/// Nanoseconds threads spent waiting for a pdfium lock (for profiling).
pub static LOCK_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static THREAD_INSTANCE: std::cell::RefCell<Option<std::sync::Arc<Instance>>> = const { std::cell::RefCell::new(None) };
}

/// Instance ``k`` of the library: 0 is the library itself, the others are
/// private copies of the file (a loaded library's globals are per file, so
/// copies run in parallel). Falls back to instance 0 when a copy fails.
pub fn instance(k: usize) -> Result<std::sync::Arc<Instance>, InputError> {
    let pool = pool()?;
    let mut inst = pool.instances.lock().unwrap();
    if k < inst.len() {
        return Ok(inst[k].clone());
    }
    let copies = std::env::var("INDRA_OCR_PDFIUM_COPIES").map_or(true, |v| v != "0");
    while inst.len() <= k {
        let n = inst.len();
        let made = copies.then(|| copy_library(&pool.path, n)).flatten();
        match made {
            Some(api) => inst.push(std::sync::Arc::new(Instance { api: Mutex::new(api) })),
            None => return Ok(inst[0].clone()),
        }
    }
    Ok(inst[k].clone())
}

/// Load a private copy of the library file (removed again once loaded).
fn copy_library(path: &Path, n: usize) -> Option<Api> {
    let dir = std::env::temp_dir().join(format!("indra-ocr-pdfium-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let ext = path.extension().map_or(String::new(), |e| format!(".{}", e.to_string_lossy()));
    let copy = dir.join(format!("pdfium-{n}{ext}"));
    std::fs::copy(path, &copy).ok()?;
    let api = load_from(&copy).ok();
    // A loaded library stays mapped after its file is gone (not on Windows,
    // where the file is locked; it is left for the OS to clean).
    let _ = std::fs::remove_file(&copy);
    let _ = std::fs::remove_dir(&dir);
    api
}

/// Make this thread open its documents with instance ``k``.
pub fn use_instance(k: usize) -> Result<(), InputError> {
    let inst = instance(k)?;
    THREAD_INSTANCE.with(|t| *t.borrow_mut() = Some(inst));
    Ok(())
}

fn thread_instance() -> Result<std::sync::Arc<Instance>, InputError> {
    if let Some(i) = THREAD_INSTANCE.with(|t| t.borrow().clone()) {
        return Ok(i);
    }
    instance(0)
}

/// pypdfium2's wording of pdfium's error codes.
fn error_text(code: c_ulong) -> &'static str {
    match code {
        0 => "Success",
        1 => "Unknown error",
        2 => "File access error",
        3 => "Data format error",
        4 => "Incorrect password error",
        5 => "Unsupported security scheme error",
        6 => "Page not found or content error",
        _ => "Unknown error",
    }
}

/// An open PDF document. Every method takes the pdfium lock.
pub struct Document {
    doc: DOC,
    inst: std::sync::Arc<Instance>,
}

// The handle is only used under the pdfium lock.
unsafe impl Send for Document {}

impl Document {
    /// Open a PDF; a locked one without the right password is an InputError
    /// saying so.
    pub fn open(path: &Path, password: Option<&str>) -> Result<Document, InputError> {
        let inst = thread_instance()?;
        let api = inst.lock();
        let cpath = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| InputError(format!("{}: bad path", path.display())))?;
        let cpw = password.map(|p| CString::new(p).unwrap_or_default());
        let doc = unsafe { (api.FPDF_LoadDocument)(cpath.as_ptr(), cpw.as_ref().map_or(std::ptr::null(), |p| p.as_ptr())) };
        if doc.is_null() {
            let code = unsafe { (api.FPDF_GetLastError)() };
            return Err(InputError(if code == FPDF_ERR_PASSWORD {
                format!("{}: password-protected PDF (pass password=, or --password)", path.display())
            } else {
                format!("{}: cannot open the PDF (Failed to load document (PDFium: {}).)", path.display(), error_text(code))
            }));
        }
        drop(api);
        Ok(Document { doc, inst })
    }

    pub fn page_count(&self) -> usize {
        let api = self.inst.lock();
        unsafe { (api.FPDF_GetPageCount)(self.doc).max(0) as usize }
    }

    /// Run ``f`` on page ``index`` with the library locked.
    pub fn with_page<T>(&self, index: usize, f: impl FnOnce(&Api, PageRef) -> T) -> Result<T, InputError> {
        let api = self.inst.lock();
        let page = unsafe { (api.FPDF_LoadPage)(self.doc, index as c_int) };
        if page.is_null() {
            let code = unsafe { (api.FPDF_GetLastError)() };
            return Err(InputError(format!("Failed to load page {index} (PDFium: {}).", error_text(code))));
        }
        let out = f(&api, PageRef(page));
        unsafe { (api.FPDF_ClosePage)(page) };
        Ok(out)
    }
}

impl Drop for Document {
    fn drop(&mut self) {
        let api = self.inst.lock();
        unsafe { (api.FPDF_CloseDocument)(self.doc) };
    }
}

/// A loaded page (valid inside ``Document::with_page``).
#[derive(Clone, Copy)]
pub struct PageRef(PAGE);

impl PageRef {
    /// Width and height in points, as rendered (page rotation applied).
    pub fn size(&self, api: &Api) -> (f64, f64) {
        unsafe { ((api.FPDF_GetPageWidthF)(self.0) as f64, (api.FPDF_GetPageHeightF)(self.0) as f64) }
    }

    /// The page in grayscale at ``scale`` pixels per point (pypdfium2's
    /// ``render(scale=.., grayscale=True)``: white fill, annotations drawn).
    pub fn render(&self, api: &Api, scale: f64) -> Gray {
        let (w, h) = self.size(api);
        let (bw, bh) = ((w * scale).ceil().max(1.0) as usize, (h * scale).ceil().max(1.0) as usize);
        unsafe {
            let bmp = (api.FPDFBitmap_CreateEx)(bw as c_int, bh as c_int, FPDFBITMAP_GRAY, std::ptr::null_mut(), 0);
            if bmp.is_null() {
                return Gray::filled(bw, bh, 255);
            }
            (api.FPDFBitmap_FillRect)(bmp, 0, 0, bw as c_int, bh as c_int, 0xFFFF_FFFF);
            (api.FPDF_RenderPageBitmap)(bmp, self.0, 0, 0, bw as c_int, bh as c_int, 0, FPDF_GRAYSCALE | FPDF_ANNOT);
            let stride = (api.FPDFBitmap_GetStride)(bmp) as usize;
            let buf = (api.FPDFBitmap_GetBuffer)(bmp) as *const u8;
            let mut out = Gray::new(bw, bh);
            for y in 0..bh {
                let row = std::slice::from_raw_parts(buf.add(y * stride), bw);
                out.row_mut(y).copy_from_slice(row);
            }
            (api.FPDFBitmap_Destroy)(bmp);
            out
        }
    }

    /// [2, 3] map from PDF page space to the rendered frame (points, top-left
    /// origin), from pdfium's own FPDF_PageToDevice.
    pub fn device_affine(&self, api: &Api) -> [[f64; 3]; 2] {
        let (w, h) = self.size(api);
        let s = 1000.0;
        let dev = |x: f64, y: f64| -> (f64, f64) {
            let (mut dx, mut dy) = (0, 0);
            unsafe {
                (api.FPDF_PageToDevice)(
                    self.0,
                    0,
                    0,
                    (w * s).round_ties_even() as c_int,
                    (h * s).round_ties_even() as c_int,
                    0,
                    x,
                    y,
                    &mut dx,
                    &mut dy,
                )
            };
            (dx as f64 / s, dy as f64 / s)
        };
        let (o, ex, ey) = (dev(0.0, 0.0), dev(1.0, 0.0), dev(0.0, 1.0));
        [[ex.0 - o.0, ey.0 - o.0, o.0], [ex.1 - o.1, ey.1 - o.1, o.1]]
    }

    /// Text objects (into form XObjects, ``max_depth`` levels): (total, invisible).
    pub fn text_objects(&self, api: &Api, max_depth: usize) -> (usize, usize) {
        fn walk(api: &Api, form: Option<PAGEOBJ>, page: PAGE, level: usize, max_depth: usize, acc: &mut (usize, usize)) {
            let n = unsafe {
                match form {
                    Some(f) => (api.FPDFFormObj_CountObjects)(f),
                    None => (api.FPDFPage_CountObjects)(page),
                }
            };
            for i in 0..n.max(0) {
                let obj = unsafe {
                    match form {
                        Some(f) => (api.FPDFFormObj_GetObject)(f, i as c_ulong),
                        None => (api.FPDFPage_GetObject)(page, i),
                    }
                };
                if obj.is_null() {
                    continue;
                }
                let ty = unsafe { (api.FPDFPageObj_GetType)(obj) };
                if ty == FPDF_PAGEOBJ_TEXT {
                    acc.0 += 1;
                    if unsafe { (api.FPDFTextObj_GetTextRenderMode)(obj) } == FPDF_TEXTRENDERMODE_INVISIBLE {
                        acc.1 += 1;
                    }
                }
                if ty == FPDF_PAGEOBJ_FORM && level + 1 < max_depth {
                    walk(api, Some(obj), page, level + 1, max_depth, acc);
                }
            }
        }
        let mut acc = (0, 0);
        walk(api, None, self.0, 0, max_depth, &mut acc);
        acc
    }

    /// The page's characters: (code point, angle, box l r b t) in content order.
    pub fn chars(&self, api: &Api) -> Result<Vec<(u32, f32, [f64; 4])>, InputError> {
        unsafe {
            let tp = (api.FPDFText_LoadPage)(self.0);
            if tp.is_null() {
                return Err(InputError("Failed to load text page.".into()));
            }
            let n = (api.FPDFText_CountChars)(tp).max(0);
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let u = (api.FPDFText_GetUnicode)(tp, i);
                let a = (api.FPDFText_GetCharAngle)(tp, i);
                let (mut l, mut r, mut b, mut t) = (0.0, 0.0, 0.0, 0.0);
                let ok = (api.FPDFText_GetCharBox)(tp, i, &mut l, &mut r, &mut b, &mut t);
                out.push((u, a, if ok != 0 { [l, r, b, t] } else { [f64::NAN; 4] }));
            }
            (api.FPDFText_ClosePage)(tp);
            Ok(out)
        }
    }
}
