//! Test helpers: golden fixture paths and a minimal .npy reader.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn golden(sub: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/golden").join(sub);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: no golden fixtures at {} (run tools/gen_*.py)", p.display());
        None
    }
}

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bytes: Vec<u8>,
}

impl Npy {
    pub fn read(path: &Path) -> Npy {
        let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(&raw[..6], b"\x93NUMPY", "{}: not a .npy file", path.display());
        let (hlen, off) = if raw[6] == 1 {
            (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&raw[off..off + hlen]).unwrap();
        let field = |k: &str| -> &str {
            let i = header.find(&format!("'{k}'")).unwrap() + k.len() + 3;
            header[i..].trim_start()
        };
        let d = field("descr");
        let dtype = d[1..].split('\'').next().unwrap().to_string();
        assert!(field("fortran_order").starts_with("False"), "fortran order not supported");
        let s = field("shape");
        let s = &s[1..s.find(')').unwrap()];
        let shape = s.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
        Npy { shape, dtype, bytes: raw[off + hlen..].to_vec() }
    }
    pub fn u8(&self) -> Vec<u8> {
        assert!(self.dtype == "|u1" || self.dtype == "|b1", "dtype {}", self.dtype);
        self.bytes.clone()
    }
    pub fn f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "<f4");
        self.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn f64(&self) -> Vec<f64> {
        assert_eq!(self.dtype, "<f8");
        self.bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype, "<i4");
        self.bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn i64(&self) -> Vec<i64> {
        assert_eq!(self.dtype, "<i8");
        self.bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    /// Any numeric dtype as f64.
    pub fn as_f64(&self) -> Vec<f64> {
        match self.dtype.as_str() {
            "<f4" => self.f32().into_iter().map(|v| v as f64).collect(),
            "<f8" => self.f64(),
            "<i4" => self.i32().into_iter().map(|v| v as f64).collect(),
            "<i8" => self.i64().into_iter().map(|v| v as f64).collect(),
            "|u1" | "|b1" => self.u8().into_iter().map(|v| v as f64).collect(),
            d => panic!("dtype {d}"),
        }
    }
    pub fn hw(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1])
    }
}
