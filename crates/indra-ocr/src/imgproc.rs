//! The image operations the pipeline takes from OpenCV, ported from OpenCV's
//! reference (non-IPP) implementations: `resize` (INTER_LINEAR fixed point and
//! INTER_AREA for uint8, INTER_LINEAR for float32), `warpAffine` (INTER_LINEAR,
//! replicate or constant border), `getRotationMatrix2D` and
//! `connectedComponentsWithStats` (4-connectivity).
//!
//! The pipeline's thresholds were tuned on OpenCV's output, so these keep its
//! arithmetic (fixed-point coefficients, rounding, float accumulation order)
//! rather than using a generic resampler.

/// A row-major single-channel image.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    pub w: usize,
    pub h: usize,
    pub data: Vec<T>,
}

pub type Gray = Image<u8>;
pub type ImageF = Image<f32>;

impl<T: Copy + Default> Image<T> {
    pub fn new(w: usize, h: usize) -> Self {
        Image { w, h, data: vec![T::default(); w * h] }
    }
    pub fn filled(w: usize, h: usize, v: T) -> Self {
        Image { w, h, data: vec![v; w * h] }
    }
    pub fn from_vec(w: usize, h: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), w * h, "image data does not match its size");
        Image { w, h, data }
    }
    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.w..(y + 1) * self.w]
    }
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[y * self.w..(y + 1) * self.w]
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> T {
        self.data[y * self.w + x]
    }
    /// The sub-image [x0, x1) x [y0, y1) (clipped to the image).
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        let (x1, y1) = (x1.min(self.w), y1.min(self.h));
        let (x0, y0) = (x0.min(x1), y0.min(y1));
        let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            out.extend_from_slice(&self.row(y)[x0..x1]);
        }
        Image { w: x1 - x0, h: y1 - y0, data: out }
    }
}

/// cvRound / saturate_cast<int>: round half to even.
#[inline]
pub fn cv_round(v: f64) -> i32 {
    let r = v.round_ties_even();
    if r >= i32::MAX as f64 {
        i32::MAX
    } else if r <= i32::MIN as f64 {
        i32::MIN
    } else {
        r as i32
    }
}

#[inline]
fn cv_round_f32(v: f32) -> i32 {
    cv_round(v as f64)
}

#[inline]
fn sat_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn sat_short(v: f32) -> i16 {
    cv_round_f32(v).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Area,
}

const RESIZE_COEF_BITS: i32 = 11;
const RESIZE_COEF_SCALE: i32 = 1 << RESIZE_COEF_BITS;

struct LinearTab {
    /// Left (top) tap; along y it is not clamped (the rows are, when read).
    ofs: Vec<i64>,
    alpha: Vec<[f32; 2]>,
    /// First index whose right tap falls off the source (taps clamp to the edge).
    max: usize,
}

/// Source offsets and weights of OpenCV's linear resize along one axis
/// (``area_mode``: the INTER_AREA variant for upscaling).
fn linear_tab(ssize: usize, dsize: usize, scale: f64, inv_scale: f64, area_mode: bool, clamp: bool) -> LinearTab {
    let mut ofs = Vec::with_capacity(dsize);
    let mut alpha = Vec::with_capacity(dsize);
    let mut max = dsize;
    for d in 0..dsize {
        let (mut s, mut f): (i64, f32);
        if !area_mode {
            f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            s = f.floor() as i64;
            f -= s as f32;
        } else {
            s = (d as f64 * scale).floor() as i64;
            f = ((d as f64 + 1.0) - (s as f64 + 1.0) * inv_scale) as f32;
            f = if f <= 0.0 { 0.0 } else { f - f.floor() };
        }
        if !clamp {
            ofs.push(s);
            alpha.push([1.0 - f, f]);
            continue;
        }
        if s < 0 {
            f = 0.0;
            s = 0;
        }
        if s + 1 >= ssize as i64 {
            max = max.min(d);
            if s >= ssize as i64 - 1 {
                f = 0.0;
                s = ssize as i64 - 1;
            }
        }
        ofs.push(s);
        alpha.push([1.0 - f, f]);
    }
    LinearTab { ofs, alpha, max }
}

fn resize_linear_u8(src: &Gray, dw: usize, dh: usize, area_mode: bool) -> Gray {
    let (sx, sy) = (dw as f64 / src.w as f64, dh as f64 / src.h as f64);
    let xt = linear_tab(src.w, dw, 1.0 / sx, sx, area_mode, true);
    let yt = linear_tab(src.h, dh, 1.0 / sy, sy, area_mode, false);
    let ia: Vec<[i32; 2]> = xt
        .alpha
        .iter()
        .map(|a| [sat_short(a[0] * RESIZE_COEF_SCALE as f32) as i32, sat_short(a[1] * RESIZE_COEF_SCALE as f32) as i32])
        .collect();
    let mut out = Gray::new(dw, dh);
    // Horizontal pass of one source row, cached (rows repeat between output rows).
    let hrow = |y: usize, buf: &mut Vec<i32>| {
        let s = src.row(y);
        buf.clear();
        for dx in 0..dw {
            let o = xt.ofs[dx] as usize;
            if dx < xt.max {
                buf.push(s[o] as i32 * ia[dx][0] + s[o + 1] as i32 * ia[dx][1]);
            } else {
                buf.push(s[o] as i32 * RESIZE_COEF_SCALE);
            }
        }
    };
    let (mut r0, mut r1) = (Vec::with_capacity(dw), Vec::with_capacity(dw));
    let mut cached: [Option<usize>; 2] = [None, None];
    for dy in 0..dh {
        let s0 = yt.ofs[dy];
        let last = src.h as i64 - 1;
        let y0 = s0.clamp(0, last) as usize;
        let y1 = (s0 + 1).clamp(0, last) as usize;
        if cached[0] != Some(y0) {
            if cached[1] == Some(y0) {
                std::mem::swap(&mut r0, &mut r1);
                cached[0] = Some(y0);
                cached[1] = None;
            } else {
                hrow(y0, &mut r0);
                cached[0] = Some(y0);
            }
        }
        if cached[1] != Some(y1) {
            hrow(y1, &mut r1);
            cached[1] = Some(y1);
        }
        let b0 = sat_short(yt.alpha[dy][0] * RESIZE_COEF_SCALE as f32) as i32;
        let b1 = sat_short(yt.alpha[dy][1] * RESIZE_COEF_SCALE as f32) as i32;
        let d = out.row_mut(dy);
        for x in 0..dw {
            let v = (((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2;
            d[x] = v as u8;
        }
    }
    out
}

fn resize_area_fast_u8(src: &Gray, dw: usize, dh: usize, kx: usize, ky: usize) -> Gray {
    let mut out = Gray::new(dw, dh);
    let area = kx * ky;
    let scale = 1.0f32 / area as f32;
    let dwidth1 = src.w / kx;
    let fast2 = kx == 2 && ky == 2;
    for dy in 0..dh {
        let sy0 = dy * ky;
        let d = &mut out.data[dy * dw..(dy + 1) * dw];
        if sy0 >= src.h {
            d.fill(0);
            continue;
        }
        let w = if sy0 + ky <= src.h { dwidth1 } else { 0 };
        let mut dx = 0;
        if fast2 {
            let (s, n) = (src.row(sy0), src.row(sy0 + 1));
            while dx < w.min(dw) {
                let i = dx * 2;
                d[dx] = ((s[i] as i32 + s[i + 1] as i32 + n[i] as i32 + n[i + 1] as i32 + 2) >> 2) as u8;
                dx += 1;
            }
        }
        while dx < w.min(dw) {
            let mut sum = 0i32;
            for yy in 0..ky {
                let s = &src.row(sy0 + yy)[dx * kx..dx * kx + kx];
                sum += s.iter().map(|&v| v as i32).sum::<i32>();
            }
            d[dx] = sat_u8(cv_round_f32(sum as f32 * scale));
            dx += 1;
        }
        while dx < dw {
            let sx0 = dx * kx;
            let (mut sum, mut count) = (0i32, 0i32);
            for yy in 0..ky {
                if sy0 + yy >= src.h {
                    break;
                }
                let s = src.row(sy0 + yy);
                for xx in 0..kx {
                    if sx0 + xx >= src.w {
                        break;
                    }
                    sum += s[sx0 + xx] as i32;
                    count += 1;
                }
            }
            d[dx] = if count == 0 { 0 } else { sat_u8(cv_round_f32(sum as f32 / count as f32)) };
            dx += 1;
        }
    }
    out
}

struct AreaTab {
    di: usize,
    si: usize,
    alpha: f32,
}

fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<AreaTab> {
    let mut tab = Vec::new();
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(AreaTab { di: dx, si: (sx1 - 1) as usize, alpha: ((sx1 as f64 - fsx1) / cell) as f32 });
        }
        for sx in sx1..sx2 {
            tab.push(AreaTab { di: dx, si: sx as usize, alpha: (1.0 / cell) as f32 });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(AreaTab {
                di: dx,
                si: sx2 as usize,
                alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32,
            });
        }
    }
    tab
}

fn resize_area_u8(src: &Gray, dw: usize, dh: usize) -> Gray {
    let scale_x = src.w as f64 / dw as f64;
    let scale_y = src.h as f64 / dh as f64;
    let xtab = area_tab(src.w, dw, scale_x);
    let ytab = area_tab(src.h, dh, scale_y);
    let mut out = Gray::new(dw, dh);
    let mut buf = vec![0f32; dw];
    let mut sum = vec![0f32; dw];
    let mut prev_dy = ytab.first().map_or(0, |t| t.di);
    for t in &ytab {
        let s = src.row(t.si);
        buf.fill(0.0);
        for x in &xtab {
            buf[x.di] += s[x.si] as f32 * x.alpha;
        }
        if t.di != prev_dy {
            let d = out.row_mut(prev_dy);
            for dx in 0..dw {
                d[dx] = sat_u8(cv_round_f32(sum[dx]));
                sum[dx] = t.alpha * buf[dx];
            }
            prev_dy = t.di;
        } else {
            for dx in 0..dw {
                sum[dx] += t.alpha * buf[dx];
            }
        }
    }
    let d = out.row_mut(prev_dy);
    for dx in 0..dw {
        d[dx] = sat_u8(cv_round_f32(sum[dx]));
    }
    out
}

/// cv2.resize of a uint8 image to (dw, dh).
pub fn resize_u8(src: &Gray, dw: usize, dh: usize, interp: Interp) -> Gray {
    assert!(src.w > 0 && src.h > 0 && dw > 0 && dh > 0, "resize of an empty image");
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let inv_x = dw as f64 / src.w as f64;
    let inv_y = dh as f64 / src.h as f64;
    let (scale_x, scale_y) = (1.0 / inv_x, 1.0 / inv_y);
    let (ix, iy) = (cv_round(scale_x), cv_round(scale_y));
    let area_fast = (scale_x - ix as f64).abs() < f64::EPSILON && (scale_y - iy as f64).abs() < f64::EPSILON;
    let mut interp = interp;
    if interp == Interp::Linear && area_fast && ix == 2 && iy == 2 {
        interp = Interp::Area;
    }
    if interp == Interp::Area && scale_x >= 1.0 && scale_y >= 1.0 {
        if area_fast {
            return resize_area_fast_u8(src, dw, dh, ix as usize, iy as usize);
        }
        return resize_area_u8(src, dw, dh);
    }
    resize_linear_u8(src, dw, dh, interp == Interp::Area)
}

/// cv2.resize of a float32 image with INTER_LINEAR.
pub fn resize_linear_f32(src: &ImageF, dw: usize, dh: usize) -> ImageF {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let (sx, sy) = (dw as f64 / src.w as f64, dh as f64 / src.h as f64);
    let xt = linear_tab(src.w, dw, 1.0 / sx, sx, false, true);
    let yt = linear_tab(src.h, dh, 1.0 / sy, sy, false, false);
    let hrow = |y: usize| -> Vec<f32> {
        let s = src.row(y);
        (0..dw)
            .map(|dx| {
                let o = xt.ofs[dx] as usize;
                if dx < xt.max { s[o] * xt.alpha[dx][0] + s[o + 1] * xt.alpha[dx][1] } else { s[o] }
            })
            .collect()
    };
    let mut out = ImageF::new(dw, dh);
    for dy in 0..dh {
        let s0 = yt.ofs[dy];
        let last = src.h as i64 - 1;
        let r0 = hrow(s0.clamp(0, last) as usize);
        let r1 = hrow((s0 + 1).clamp(0, last) as usize);
        let [b0, b1] = yt.alpha[dy];
        let d = out.row_mut(dy);
        for x in 0..dw {
            d[x] = r0[x] * b0 + r1[x] * b1;
        }
    }
    out
}

/// cv2.getRotationMatrix2D: [2, 3] for a counter-clockwise turn of ``degrees``
/// about ``center`` (cast to float32, as OpenCV's Point2f).
pub fn rotation_matrix(center: (f64, f64), degrees: f64, scale: f64) -> [f64; 6] {
    let (cx, cy) = (center.0 as f32 as f64, center.1 as f32 as f64);
    let a = degrees.to_radians();
    let alpha = a.cos() * scale;
    let beta = a.sin() * scale;
    [alpha, beta, (1.0 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1.0 - alpha) * cy]
}

fn invert_affine(m: &[f64; 6]) -> [f64; 6] {
    let mut m = *m;
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22) = (m[4] * d, m[0] * d);
    m[0] = a11;
    m[1] *= -d;
    m[3] *= -d;
    m[4] = a22;
    let b1 = -m[0] * m[2] - m[1] * m[5];
    let b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1;
    m[5] = b2;
    m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    Replicate,
    Constant,
}

/// The 2x2 bilinear sample of OpenCV's float warp kernels (4.11+): coordinates
/// in float32, weights from the fraction, three fused multiply-adds.
#[inline(always)]
fn bilinear_f32(p00: f32, p01: f32, p10: f32, p11: f32, a: f32, b: f32) -> f32 {
    let v0 = a.mul_add(p01 - p00, p00);
    let v1 = a.mul_add(p11 - p10, p10);
    b.mul_add(v1 - v0, v0)
}

/// Pixel types the warp reads.
pub trait WarpPixel: Copy + Default + Send + Sync + 'static {
    const IS_U8: bool;
    fn to_f32(self) -> f32;
}
impl WarpPixel for u8 {
    const IS_U8: bool = true;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl WarpPixel for f32 {
    const IS_U8: bool = false;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// A source pixel as the warp sees it: as is, or (with INK, for uint8) the
/// ink value (255 - v) / 255 computed as the float image would hold it.
#[inline(always)]
fn tap<T: WarpPixel, const INK: bool>(v: T) -> f32 {
    if INK { (255.0 - v.to_f32()) / 255.0 } else { v.to_f32() }
}

/// AVX2 kernel for the pixels of a row whose four taps are inside the source:
/// 8 at a time, the same float operations as the scalar path (so the same
/// result), written as floats from ``x0``. Returns how many it wrote.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn warp_row_avx2<T: WarpPixel, const INK: bool>(src: &Image<T>, mi: &[f32; 6], mx: f32, my: f32, x0: usize, out: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let (sw, sh) = (src.w as i32, src.h as i32);
    let len = src.data.len() as i64;
    let w = src.w as i32;
    let base = src.data.as_ptr();
    let mut x = x0;
    unsafe {
        let lane = _mm256_setr_ps(0., 1., 2., 3., 4., 5., 6., 7.);
        let (m0, m3) = (_mm256_set1_ps(mi[0]), _mm256_set1_ps(mi[3]));
        let (vmx, vmy) = (_mm256_set1_ps(mx), _mm256_set1_ps(my));
        let (lim_x, lim_y) = (_mm256_set1_epi32(sw - 1), _mm256_set1_epi32(sh - 1));
        let vw = _mm256_set1_epi32(w);
        let (c255, inv255) = (_mm256_set1_ps(255.0), _mm256_set1_ps(255.0));
        let ink = |v: __m256| if INK { _mm256_div_ps(_mm256_sub_ps(c255, v), inv255) } else { v };
        while x + 8 <= out.len() {
            let xf = _mm256_add_ps(_mm256_set1_ps(x as f32), lane);
            let sx = _mm256_fmadd_ps(m0, xf, vmx);
            let sy = _mm256_fmadd_ps(m3, xf, vmy);
            let (fx, fy) = (_mm256_floor_ps(sx), _mm256_floor_ps(sy));
            let (a, b) = (_mm256_sub_ps(sx, fx), _mm256_sub_ps(sy, fy));
            let (ix, iy) = (_mm256_cvttps_epi32(fx), _mm256_cvttps_epi32(fy));
            // inside: 0 <= ix < sw - 1 and 0 <= iy < sh - 1 (and no float overflow)
            let ok = _mm256_and_si256(
                _mm256_and_si256(_mm256_cmpgt_epi32(lim_x, ix), _mm256_cmpgt_epi32(lim_y, iy)),
                _mm256_and_si256(_mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1))),
            );
            if _mm256_movemask_epi8(ok) != -1 {
                break;
            }
            let idx = _mm256_add_epi32(_mm256_mullo_epi32(iy, vw), ix);
            let idx2 = _mm256_add_epi32(idx, vw);
            let (p00, p01, p10, p11);
            if T::IS_U8 {
                // one 32-bit load per row of taps (bytes idx, idx + 1): the last
                // lane may read 2 bytes past a tap, so stay 4 bytes inside.
                let mut far = [0i32; 8];
                _mm256_storeu_si256(far.as_mut_ptr() as *mut __m256i, idx2);
                if far.iter().any(|&f| f as i64 + 4 > len) {
                    break;
                }
                let bp = base as *const i32;
                let r0 = _mm256_i32gather_epi32::<1>(bp, idx);
                let r1 = _mm256_i32gather_epi32::<1>(bp, idx2);
                let ff = _mm256_set1_epi32(0xFF);
                p00 = _mm256_cvtepi32_ps(_mm256_and_si256(r0, ff));
                p01 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32::<8>(r0), ff));
                p10 = _mm256_cvtepi32_ps(_mm256_and_si256(r1, ff));
                p11 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32::<8>(r1), ff));
            } else {
                let fp = base as *const f32;
                p00 = _mm256_i32gather_ps::<4>(fp, idx);
                p01 = _mm256_i32gather_ps::<4>(fp, _mm256_add_epi32(idx, _mm256_set1_epi32(1)));
                p10 = _mm256_i32gather_ps::<4>(fp, idx2);
                p11 = _mm256_i32gather_ps::<4>(fp, _mm256_add_epi32(idx2, _mm256_set1_epi32(1)));
            }
            let (p00, p01, p10, p11) = (ink(p00), ink(p01), ink(p10), ink(p11));
            let v0 = _mm256_fmadd_ps(a, _mm256_sub_ps(p01, p00), p00);
            let v1 = _mm256_fmadd_ps(a, _mm256_sub_ps(p11, p10), p10);
            let v = _mm256_fmadd_ps(b, _mm256_sub_ps(v1, v0), v0);
            _mm256_storeu_ps(out.as_mut_ptr().add(x), v);
            x += 8;
        }
    }
    x - x0
}

/// warpAffine (INTER_LINEAR) row by row: ``row(y, values)`` with each output
/// row as floats. Source positions are computed as OpenCV computes them (the
/// inverted matrix in float32, one fused multiply-add per coordinate);
/// ``border`` taps read ``value`` (in the float domain).
#[inline(always)]
fn warp_rows_impl<T: WarpPixel, const SIMD: bool, const INK: bool>(src: &Image<T>, m: &[f64; 6], w: usize, h: usize, border: Border, value: f32, mut row: impl FnMut(usize, &[f32])) {
    let mi = invert_affine(m).map(|v| v as f32);
    let (sw, sh) = (src.w as i64, src.h as i64);
    let d = &src.data;
    let px = |x: i64, y: i64| -> f32 {
        match border {
            Border::Replicate => tap::<T, INK>(src.at(x.clamp(0, sw - 1) as usize, y.clamp(0, sh - 1) as usize)),
            Border::Constant => {
                if x < 0 || y < 0 || x >= sw || y >= sh {
                    value
                } else {
                    tap::<T, INK>(src.at(x as usize, y as usize))
                }
            }
        }
    };
    let mut buf = vec![0f32; w];
    for y in 0..h {
        let mx = mi[1] * y as f32 + mi[2];
        let my = mi[4] * y as f32 + mi[5];
        let mut x = 0;
        while x < w {
            #[cfg(target_arch = "x86_64")]
            if SIMD {
                // SAFETY: SIMD is set only when the CPU has AVX2 and FMA.
                let n = unsafe { warp_row_avx2::<T, INK>(src, &mi, mx, my, x, &mut buf) };
                if n > 0 {
                    x += n;
                    continue;
                }
            }
            let xf = x as f32;
            let (sx, sy) = (mi[0].mul_add(xf, mx), mi[3].mul_add(xf, my));
            let (fx, fy) = (sx.floor(), sy.floor());
            let (a, b) = (sx - fx, sy - fy);
            let (ix, iy) = (fx as i64, fy as i64);
            buf[x] = if ix >= 0 && iy >= 0 && ix < sw - 1 && iy < sh - 1 {
                let i = iy as usize * src.w + ix as usize;
                bilinear_f32(tap::<T, INK>(d[i]), tap::<T, INK>(d[i + 1]), tap::<T, INK>(d[i + src.w]), tap::<T, INK>(d[i + src.w + 1]), a, b)
            } else if border == Border::Constant && (ix >= sw || ix + 1 < 0 || iy >= sh || iy + 1 < 0) {
                value
            } else {
                bilinear_f32(px(ix, iy), px(ix + 1, iy), px(ix, iy + 1), px(ix + 1, iy + 1), a, b)
            };
            x += 1;
        }
        row(y, &buf);
    }
}

/// ``warp_rows_impl`` compiled with hardware FMA (the fused multiply-adds are
/// part of the result, so without FMA they run in software).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn warp_rows_fma<T: WarpPixel, const INK: bool>(src: &Image<T>, m: &[f64; 6], w: usize, h: usize, border: Border, value: f32, row: impl FnMut(usize, &[f32])) {
    warp_rows_impl::<T, true, INK>(src, m, w, h, border, value, row)
}

fn warp_rows<T: WarpPixel, const INK: bool>(src: &Image<T>, m: &[f64; 6], w: usize, h: usize, border: Border, value: f32, row: impl FnMut(usize, &[f32])) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has the features the function is compiled for.
        return unsafe { warp_rows_fma::<T, INK>(src, m, w, h, border, value, row) };
    }
    warp_rows_impl::<T, false, INK>(src, m, w, h, border, value, row)
}

/// cv2.warpAffine of a uint8 image (INTER_LINEAR), output size (w, h).
pub fn warp_affine_u8(src: &Gray, m: &[f64; 6], w: usize, h: usize, border: Border, value: u8) -> Gray {
    let mut out = Gray::new(w, h);
    warp_rows::<u8, false>(src, m, w, h, border, value as f32, |y, r| {
        for (o, &v) in out.row_mut(y).iter_mut().zip(r) {
            // round half to even (the values are far below 2^22), then saturate
            let rounded = (v.clamp(-1.0, 256.0) + 12_582_912.0) - 12_582_912.0;
            *o = sat_u8(rounded as i32);
        }
    });
    out
}

/// cv2.warpAffine of a float32 image (INTER_LINEAR), output size (w, h).
pub fn warp_affine_f32(src: &ImageF, m: &[f64; 6], w: usize, h: usize, border: Border, value: f32) -> ImageF {
    let mut out = ImageF::new(w, h);
    warp_rows::<f32, false>(src, m, w, h, border, value, |y, r| out.row_mut(y).copy_from_slice(r));
    out
}

/// The row sums (NumPy's pairwise sum) of cv2.warpAffine of the ink map
/// (255 - g) / 255 of ``src`` (float32, constant 0 border), read from the
/// uint8 image without building the float one.
pub fn warp_ink_row_sums(src: &Gray, m: &[f64; 6], w: usize, h: usize) -> Vec<f32> {
    let mut sums = vec![0f32; h];
    warp_rows::<u8, true>(src, m, w, h, Border::Constant, 0.0, |y, r| sums[y] = crate::py::pairwise_sum_f32(r));
    sums
}

/// One component of ``connected_components``.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CcStat {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
    pub area: i32,
}

/// cv2.connectedComponentsWithStats(mask, connectivity=4): (labels, stats).
/// Label 0 is the background; components are numbered in raster order of their
/// first pixel, as OpenCV numbers them.
pub fn connected_components(mask: &Image<u8>) -> (Vec<i32>, Vec<CcStat>) {
    let (w, h) = (mask.w, mask.h);
    let mut labels = vec![0i32; w * h];
    // Two-pass union-find; a component's root is its smallest provisional label,
    // created at its first pixel in raster order.
    let mut parent: Vec<i32> = vec![0];
    fn find(p: &mut [i32], mut x: i32) -> i32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask.data[i] == 0 {
                continue;
            }
            let up = if y > 0 { labels[i - w] } else { 0 };
            let left = if x > 0 { labels[i - 1] } else { 0 };
            labels[i] = match (up, left) {
                (0, 0) => {
                    let l = parent.len() as i32;
                    parent.push(l);
                    l
                }
                (u, 0) => u,
                (0, l) => l,
                (u, l) => {
                    let (ru, rl) = (find(&mut parent, u), find(&mut parent, l));
                    let r = ru.min(rl);
                    parent[ru as usize] = r;
                    parent[rl as usize] = r;
                    r
                }
            };
        }
    }
    let mut fin = vec![0i32; parent.len()];
    let mut n = 1;
    for k in 1..parent.len() {
        let r = find(&mut parent, k as i32) as usize;
        if r == k {
            fin[k] = n;
            n += 1;
        } else {
            fin[k] = fin[r];
        }
    }
    let mut stats = vec![CcStat { left: i32::MAX, top: i32::MAX, width: i32::MIN, height: i32::MIN, area: 0 }; n as usize];
    let mut bg = 0;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = fin[labels[i] as usize];
            labels[i] = l;
            if l == 0 {
                bg += 1;
                continue;
            }
            let s = &mut stats[l as usize];
            s.left = s.left.min(x as i32);
            s.top = s.top.min(y as i32);
            s.width = s.width.max(x as i32); // right edge until the end
            s.height = s.height.max(y as i32); // bottom edge until the end
            s.area += 1;
        }
    }
    for s in stats.iter_mut().skip(1) {
        s.width = s.width - s.left + 1;
        s.height = s.height - s.top + 1;
    }
    stats[0] = if bg > 0 {
        // OpenCV's background stats: the bounding box of the background pixels.
        let (mut l, mut t, mut r, mut b) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for y in 0..h {
            for x in 0..w {
                if labels[y * w + x] == 0 {
                    l = l.min(x as i32);
                    t = t.min(y as i32);
                    r = r.max(x as i32);
                    b = b.max(y as i32);
                }
            }
        }
        CcStat { left: l, top: t, width: r - l + 1, height: b - t + 1, area: bg }
    } else {
        CcStat::default()
    };
    (labels, stats)
}

/// np.rot90(img, -k): the image turned clockwise by k quarter turns.
pub fn rot90_cw<T: Copy + Default>(img: &Image<T>, k: u32) -> Image<T> {
    match k % 4 {
        0 => img.clone(),
        2 => {
            let mut d = img.data.clone();
            d.reverse();
            Image::from_vec(img.w, img.h, d)
        }
        1 => {
            // out[y][x] = in[h-1-x][y], out is h wide and w high
            let (w, h) = (img.h, img.w);
            let mut out = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    out.data[y * w + x] = img.at(y, img.h - 1 - x);
                }
            }
            out
        }
        _ => {
            // out[y][x] = in[x][w-1-y]
            let (w, h) = (img.h, img.w);
            let mut out = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    out.data[y * w + x] = img.at(img.w - 1 - y, x);
                }
            }
            out
        }
    }
}
