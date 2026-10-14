//! Input files: PDF detection by the file's bytes, and grayscale image pages
//! decoded as cv2.imread(..., IMREAD_GRAYSCALE) decodes them (every frame of a
//! multi-page TIFF).

use std::io::Cursor;
use std::path::Path;

use crate::imgproc::Gray;

/// A file that cannot be read: not a PDF or an image, corrupt, or locked.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct InputError(pub String);

/// The file's own bytes say PDF (the header may follow a little junk).
pub fn is_pdf(path: &Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 1024];
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let n = f.read(&mut head).unwrap_or(0);
    head[..n].windows(5).any(|w| w == b"%PDF-")
}

/// libpng's rgb_to_gray (as OpenCV asks for it: 0.299, 0.587), on 8-bit RGB.
#[inline]
fn png_gray(r: u8, g: u8, b: u8) -> u8 {
    if r == g && r == b {
        return r;
    }
    ((9797 * r as u32 + 19234 * g as u32 + 3737 * b as u32) >> 15) as u8
}

/// OpenCV's BGR -> gray (imgcodecs icvCvt_BGR2Gray, 14-bit, rounded).
#[inline]
fn cv_gray(r: u8, g: u8, b: u8) -> u8 {
    ((4899 * r as u32 + 9617 * g as u32 + 1868 * b as u32 + 8192) >> 14) as u8
}

/// cv::cvtColor(COLOR_BGR2GRAY) (15-bit, rounded): decoders that decode colour first (WebP).
#[inline]
fn cvt_gray(r: u8, g: u8, b: u8) -> u8 {
    ((9798 * r as u32 + 19235 * g as u32 + 3735 * b as u32 + 16384) >> 15) as u8
}

/// libpng's 16-bit rgb_to_gray, then the 16 -> 8 bit chop.
#[inline]
fn png_gray16(r: u16, g: u16, b: u16) -> u8 {
    (((9797 * r as u32 + 19234 * g as u32 + 3737 * b as u32 + 16384) >> 15) >> 8) as u8
}

/// How a decoder turns colour into gray.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Conv {
    /// libpng's rgb_to_gray (PNG)
    Png,
    /// imgcodecs' 14-bit helper (BMP, TIFF, PNM)
    Codec,
    /// cv::cvtColor (WebP)
    CvtColor,
}

impl Conv {
    fn gray(self, r: u8, g: u8, b: u8) -> u8 {
        match self {
            Conv::Png => png_gray(r, g, b),
            Conv::Codec => cv_gray(r, g, b),
            Conv::CvtColor => cvt_gray(r, g, b),
        }
    }
    fn gray16(self, r: u16, g: u16, b: u16) -> u8 {
        match self {
            Conv::Png => png_gray16(r, g, b),
            _ => self.gray((r >> 8) as u8, (g >> 8) as u8, (b >> 8) as u8),
        }
    }
}

fn to_gray(img: image::DynamicImage, conv: Conv) -> Gray {
    use image::DynamicImage as D;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<u8> = match img {
        D::ImageLuma8(b) => b.into_raw(),
        D::ImageLumaA8(b) => b.pixels().map(|p| p.0[0]).collect(),
        D::ImageLuma16(b) => b.pixels().map(|p| (p.0[0] >> 8) as u8).collect(),
        D::ImageLumaA16(b) => b.pixels().map(|p| (p.0[0] >> 8) as u8).collect(),
        D::ImageRgb8(b) => b.pixels().map(|p| conv.gray(p.0[0], p.0[1], p.0[2])).collect(),
        D::ImageRgba8(b) => b.pixels().map(|p| conv.gray(p.0[0], p.0[1], p.0[2])).collect(),
        D::ImageRgb16(b) => b.pixels().map(|p| conv.gray16(p.0[0], p.0[1], p.0[2])).collect(),
        D::ImageRgba16(b) => b.pixels().map(|p| conv.gray16(p.0[0], p.0[1], p.0[2])).collect(),
        other => other.to_rgb8().pixels().map(|p| conv.gray(p.0[0], p.0[1], p.0[2])).collect(),
    };
    Gray::from_vec(w, h, data)
}

fn decode_jpeg(bytes: &[u8]) -> Option<Gray> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    // libjpeg's grayscale output is the Y channel (no colour conversion).
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma).set_max_width(1 << 16).set_max_height(1 << 16);
    let mut d = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    let px = d.decode().ok()?;
    let (w, h) = d.dimensions()?;
    (px.len() == w * h).then(|| Gray::from_vec(w, h, px))
}

fn decode_tiff(bytes: &[u8]) -> Option<Vec<Gray>> {
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult};
    let mut d = Decoder::new(Cursor::new(bytes)).ok()?;
    let mut frames = vec![];
    loop {
        let (w, h) = d.dimensions().ok()?;
        let (w, h) = (w as usize, h as usize);
        let ct = d.colortype().ok()?;
        let img = d.read_image().ok()?;
        let n = w * h;
        let gray: Vec<u8> = match (ct, img) {
            (ColorType::Gray(8), DecodingResult::U8(v)) => v,
            (ColorType::Gray(16), DecodingResult::U16(v)) => v.iter().map(|&x| (x >> 8) as u8).collect(),
            (ColorType::GrayA(8), DecodingResult::U8(v)) => v.as_chunks::<2>().0.iter().map(|p| p[0]).collect(),
            (ColorType::RGB(8), DecodingResult::U8(v)) => v.as_chunks::<3>().0.iter().map(|p| cv_gray(p[0], p[1], p[2])).collect(),
            (ColorType::RGBA(8), DecodingResult::U8(v)) => v.as_chunks::<4>().0.iter().map(|p| cv_gray(p[0], p[1], p[2])).collect(),
            (ColorType::RGB(16), DecodingResult::U16(v)) => {
                v.as_chunks::<3>().0.iter().map(|p| cv_gray((p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8)).collect()
            }
            (ColorType::Gray(1), DecodingResult::U8(v)) => {
                // bilevel: rows padded to whole bytes, 0 = black unless WhiteIsZero
                let stride = w.div_ceil(8);
                (0..h)
                    .flat_map(|y| (0..w).map(move |x| (y, x)))
                    .map(|(y, x)| if v[y * stride + x / 8] >> (7 - x % 8) & 1 == 1 { 255 } else { 0 })
                    .collect()
            }
            _ => return None,
        };
        if gray.len() != n {
            return None;
        }
        frames.push(Gray::from_vec(w, h, gray));
        if !d.more_images() || d.next_image().is_err() {
            break;
        }
    }
    Some(frames)
}

/// Grayscale pages of an image file; every frame of a multi-page TIFF.
pub fn read_images(path: &Path) -> Result<Vec<Gray>, InputError> {
    let bad = || InputError(format!("{}: not a PDF or a readable image", path.display()));
    let bytes = std::fs::read(path).map_err(|_| bad())?;
    decode_images(&bytes).ok_or_else(bad)
}

/// Grayscale pages of an image in memory (None: not a readable image).
pub fn decode_images(bytes: &[u8]) -> Option<Vec<Gray>> {
    let fmt = image::guess_format(bytes).ok()?;
    let frames = match fmt {
        image::ImageFormat::Jpeg => decode_jpeg(bytes).map(|g| vec![g]),
        image::ImageFormat::Tiff => decode_tiff(bytes),
        image::ImageFormat::Png => image::load_from_memory_with_format(bytes, fmt).ok().map(|i| vec![to_gray(i, Conv::Png)]),
        image::ImageFormat::WebP => image::load_from_memory_with_format(bytes, fmt).ok().map(|i| vec![to_gray(i, Conv::CvtColor)]),
        _ => image::load_from_memory_with_format(bytes, fmt).ok().map(|i| vec![to_gray(i, Conv::Codec)]),
    };
    let frames = frames.or_else(|| image::load_from_memory(bytes).ok().map(|i| vec![to_gray(i, Conv::Codec)]));
    frames.filter(|f| !f.is_empty() && f.iter().all(|g| g.w > 0 && g.h > 0))
}
