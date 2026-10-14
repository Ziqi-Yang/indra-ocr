//! Int8 matrix products with AVX-512 VNNI (opt-in reduced precision).
//!
//! Weights are quantized once per output column (symmetric, int8); each row of
//! A is quantized when the product runs (symmetric, scale max|a| / 127) and
//! shifted to uint8 for ``vpdpbusd`` (u8 x s8 -> s32); the shift is taken back
//! out with the weight column sums:
//!
//!   sum_k (a_k + 128) w_k - 128 sum_k w_k = sum_k a_k w_k
//!   C[i, j] = act(acc * sa_i * sw_j + bias_j)
//!
//! Only on CPUs with AVX-512 VNNI; elsewhere ``QPackedB::new`` returns None and
//! callers keep the float product.

use crate::gemm::{Act, Out};

const NR: usize = 32; // two zmm of int32 lanes
const MR: usize = 12;

/// A weight matrix B [K, N] packed as int8 for the VNNI kernel.
#[derive(Clone, Debug)]
// off x86-64 there is no kernel to read the packed data (none is ever built)
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub struct QPackedB {
    pub k: usize,
    pub n: usize,
    k4: usize,
    panels: usize,
    /// per panel: k4 / 4 groups of NR lanes x 4 bytes
    data: Vec<i8>,
    /// per column: weight scale, 128 * column sum, bias (padded to panels * NR)
    scale: Vec<f32>,
    shift: Vec<i32>,
    bias: Vec<f32>,
}

/// Whether the int8 kernels can run here.
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::cpu::isa() == crate::cpu::Isa::Avx512
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

impl QPackedB {
    /// From a weight laid out [N, K] (out x in) and its bias; None without VNNI.
    pub fn from_nk(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Option<Self> {
        if !available() {
            return None;
        }
        let k4 = k.div_ceil(4) * 4;
        let panels = n.div_ceil(NR);
        let mut data = vec![0i8; panels * k4 * NR];
        let (mut scale, mut shift, mut b) = (vec![0f32; panels * NR], vec![0i32; panels * NR], vec![0f32; panels * NR]);
        for j in 0..n {
            let col = &w[j * k..(j + 1) * k];
            let m = col.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if m > 0.0 { m / 127.0 } else { 1.0 };
            scale[j] = s;
            b[j] = bias.map_or(0.0, |b| b[j]);
            let (p, lane) = (j / NR, j % NR);
            let mut sum = 0i32;
            for (kk, &v) in col.iter().enumerate() {
                let q = (v / s).round().clamp(-127.0, 127.0) as i8;
                sum += q as i32;
                data[(p * (k4 / 4) + kk / 4) * NR * 4 + lane * 4 + kk % 4] = q;
            }
            shift[j] = 128 * sum;
        }
        Some(QPackedB { k, n, k4, panels, data, scale, shift, bias: b })
    }
}

/// C = A [M, K] times B (int8), as ``gemm``.
pub fn qgemm(a: &[f32], lda: usize, m: usize, b: &QPackedB, c: &mut [f32], ldc: usize, act: Act, out: Out) {
    if m == 0 || b.n == 0 {
        return;
    }
    // quantize the rows of A: uint8 (a / sa + 128), padded to k4
    let mut qa = vec![128u8; m.div_ceil(MR) * MR * b.k4];
    let mut sa = vec![0f32; m.div_ceil(MR) * MR];
    quantize_rows(a, lda, m, b.k, b.k4, &mut qa, &mut sa);
    tiles(b, &qa, &sa, m, c, ldc, act, out);
}

/// The panels and row tiles of ``qgemm`` (the kernel is x86-64 only).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn tiles(b: &QPackedB, qa: &[u8], sa: &[f32], m: usize, c: &mut [f32], ldc: usize, act: Act, out: Out) {
    let mut tmp = vec![0f32; MR * NR];
    for p in 0..b.panels {
        let cols = NR.min(b.n - p * NR);
        let bp = &b.data[p * (b.k4 / 4) * NR * 4..(p + 1) * (b.k4 / 4) * NR * 4];
        let ep = Epi { scale: &b.scale[p * NR..], shift: &b.shift[p * NR..], bias: &b.bias[p * NR..], act };
        for r0 in (0..m).step_by(MR) {
            let rows = MR.min(m - r0);
            if rows == MR && cols == NR {
                // SAFETY: QPackedB exists only where available() holds; the tile is inside C.
                unsafe { tile_vnni(b.k4, &qa[r0 * b.k4..], bp, &sa[r0..], &ep, c.as_mut_ptr().add(r0 * ldc + p * NR), ldc, out) };
                continue;
            }
            // SAFETY: as above, into a full-size scratch tile
            unsafe { tile_vnni(b.k4, &qa[r0 * b.k4..], bp, &sa[r0..], &ep, tmp.as_mut_ptr(), NR, Out::Store) };
            for r in 0..rows {
                let dst = &mut c[(r0 + r) * ldc + p * NR..(r0 + r) * ldc + p * NR + cols];
                for (d, &v) in dst.iter_mut().zip(&tmp[r * NR..r * NR + cols]) {
                    *d = if out == Out::Add { *d + v } else { v };
                }
            }
        }
    }
}

/// Off x86-64 there is no int8 kernel, and no ``QPackedB`` to call it with
/// (``QPackedB::from_nk`` returns None where ``available()`` is false).
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn tiles(_b: &QPackedB, _qa: &[u8], _sa: &[f32], _m: usize, _c: &mut [f32], _ldc: usize, _act: Act, _out: Out) {
    unreachable!("int8 products need AVX-512 VNNI")
}

/// The per-column epilogue of one panel.
#[cfg(target_arch = "x86_64")]
struct Epi<'a> {
    scale: &'a [f32],
    shift: &'a [i32],
    bias: &'a [f32],
    act: Act,
}

crate::simd::multiversion! {
    /// Each row of A as uint8 (round(a / s) + 128, s = max|a| / 127) and its scale.
    fn quantize_rows(a: &[f32], lda: usize, m: usize, k: usize, k4: usize, qa: &mut [u8], sa: &mut [f32]) {
        for i in 0..m {
            let row = &a[i * lda..i * lda + k];
            let mx = row.iter().fold(0f32, |x, &v| x.max(v.abs()));
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            sa[i] = s;
            let inv = 1.0 / s;
            for (q, &v) in qa[i * k4..i * k4 + k].iter_mut().zip(row) {
                // round half to even via the 1.5 * 2^23 trick (|v * inv| <= 127)
                let r = (v * inv + 12_582_912.0) - 12_582_912.0;
                *q = (r as i32 + 128) as u8;
            }
        }
    }
}

/// MR rows x 32 columns: act((acc - shift) * sa * sw + bias), stored or added.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_vnni(k4: usize, qa: &[u8], bp: &[i8], sa: &[f32], ep: &Epi, c: *mut f32, ldc: usize, out: Out) {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = [[_mm512_setzero_si512(); 2]; MR];
        let a = qa.as_ptr();
        let b = bp.as_ptr();
        for g in 0..k4 / 4 {
            let b0 = _mm512_loadu_si512(b.add(g * NR * 4) as *const _);
            let b1 = _mm512_loadu_si512(b.add(g * NR * 4 + 64) as *const _);
            for r in 0..MR {
                let av = _mm512_set1_epi32((a.add(r * k4 + g * 4) as *const i32).read_unaligned());
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
            }
        }
        let sh = [_mm512_loadu_si512(ep.shift.as_ptr() as *const _), _mm512_loadu_si512(ep.shift.as_ptr().add(16) as *const _)];
        let sw = [_mm512_loadu_ps(ep.scale.as_ptr()), _mm512_loadu_ps(ep.scale.as_ptr().add(16))];
        let bi = [_mm512_loadu_ps(ep.bias.as_ptr()), _mm512_loadu_ps(ep.bias.as_ptr().add(16))];
        let (zero, one, sixth, half) = (_mm512_setzero_ps(), _mm512_set1_ps(1.0), _mm512_set1_ps(1.0 / 6.0), _mm512_set1_ps(0.5));
        for (r, row) in acc.iter().enumerate() {
            let s = _mm512_set1_ps(sa[r]);
            let d = c.add(r * ldc);
            for h in 0..2 {
                let x = _mm512_cvtepi32_ps(_mm512_sub_epi32(row[h], sh[h]));
                let mut v = _mm512_fmadd_ps(x, _mm512_mul_ps(s, sw[h]), bi[h]);
                v = match ep.act {
                    Act::None => v,
                    Act::Relu => _mm512_max_ps(v, zero),
                    Act::HardSwish => _mm512_mul_ps(v, _mm512_min_ps(_mm512_max_ps(_mm512_fmadd_ps(v, sixth, half), zero), one)),
                };
                if out == Out::Add {
                    v = _mm512_add_ps(v, _mm512_loadu_ps(d.add(16 * h)));
                }
                _mm512_storeu_ps(d.add(16 * h), v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_to_float_product() {
        let (m, k, n) = (29, 128, 70);
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
        let bias: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let Some(q) = QPackedB::from_nk(&w, n, k, Some(&bias)) else { return };
        let mut c = vec![0f32; m * n];
        qgemm(&a, k, m, &q, &mut c, n, Act::None, Out::Store);
        let mut worst = 0f32;
        for i in 0..m {
            for j in 0..n {
                let want: f32 = bias[j] + (0..k).map(|kk| a[i * k + kk] * w[j * k + kk]).sum::<f32>();
                worst = worst.max((c[i * n + j] - want).abs());
            }
        }
        // int8 on both sides: about 1% of the typical magnitude (~1.6)
        assert!(worst < 0.05, "max error {worst}");
    }
}
