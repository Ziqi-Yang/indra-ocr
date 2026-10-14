//! Single-precision matrix products against prepacked weights, with the
//! bias, activation and residual add fused into the store:
//!
//!   C[i, j] (=|+=) act(sum_k A[i, k] B[k, j] + bias[j])
//!
//! B (the weights) is packed once into panels of NR columns (K x NR each,
//! zero-padded), so the micro-kernel streams it linearly while MR rows of A are
//! broadcast. Panels are 32, 16 or 8 columns wide (narrow matrices get narrow
//! panels). Kernels: AVX-512 (12 x 32/16/8), AVX2/FMA (6 x 16, 12 x 8), NEON
//! (6 x 16, 8 x 8) and portable Rust (4 x 16/8).

use crate::cpu::{Isa, isa};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    /// ONNX HardSwish: x * clamp(x / 6 + 0.5, 0, 1)
    HardSwish,
}

/// Where the result goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    Store,
    /// C += result (a residual connection)
    Add,
}

/// Panel width for an N-column weight: wide panels for wide matrices, narrow
/// ones when most of a wide panel would be padding.
fn panel_width(isa: Isa, n: usize) -> usize {
    let widths: &[usize] = match isa {
        Isa::Avx512 => &[32, 16, 8],
        _ => &[16, 8],
    };
    let waste = |nr: usize| n.div_ceil(nr) * nr - n;
    // the widest panel that wastes no more than the narrowest would
    let best = waste(*widths.last().unwrap());
    *widths.iter().find(|&&nr| waste(nr) <= best.max(n / 8)).unwrap()
}

/// Rows per tile for a panel width.
fn tile_rows(isa: Isa, nr: usize) -> usize {
    match (isa, nr) {
        (Isa::Avx512, _) => 12,
        (Isa::Avx2, 16) => 6,
        (Isa::Avx2, _) => 12,
        (Isa::Neon, 16) => 6,
        (Isa::Neon, _) => 8,
        _ => 4,
    }
}

/// Rows of A per cache block (a multiple of every MR).
const MC: usize = 96;

/// A weight matrix B [K, N] (and bias [N]) packed for the kernels of one ISA.
#[derive(Clone, Debug)]
pub struct PackedB {
    pub k: usize,
    pub n: usize,
    nr: usize,
    panels: usize,
    data: Vec<f32>,
    bias: Vec<f32>,
    isa: Isa,
}

impl PackedB {
    /// From ``get(k, n)`` for every k < K, n < N.
    fn build(k: usize, n: usize, bias: Option<&[f32]>, get: impl Fn(usize, usize) -> f32) -> Self {
        let isa = isa();
        let nr = panel_width(isa, n);
        let panels = n.div_ceil(nr);
        let mut data = vec![0f32; panels * k * nr];
        for p in 0..panels {
            for kk in 0..k {
                for j in 0..nr {
                    let col = p * nr + j;
                    if col < n {
                        data[(p * k + kk) * nr + j] = get(kk, col);
                    }
                }
            }
        }
        let mut b = vec![0f32; panels * nr];
        if let Some(bias) = bias {
            b[..n].copy_from_slice(&bias[..n]);
        }
        PackedB { k, n, nr, panels, data, bias: b, isa }
    }

    /// From a weight laid out [N, K] (ONNX Conv / Gemm with transB): out x in.
    pub fn from_nk(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), n * k, "weight size");
        Self::build(k, n, bias, |kk, j| w[j * k + kk])
    }

    /// From a weight laid out [K, N] (ONNX MatMul's B).
    pub fn from_kn(w: &[f32], k: usize, n: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), n * k, "weight size");
        Self::build(k, n, bias, |kk, j| w[kk * n + j])
    }
}

/// C = A [M, K] (row stride lda) times B, into C (row stride ldc).
#[allow(clippy::too_many_arguments)]
pub fn gemm(a: &[f32], lda: usize, m: usize, b: &PackedB, c: &mut [f32], ldc: usize, act: Act, out: Out) {
    if m == 0 || b.n == 0 {
        return;
    }
    assert!(lda >= b.k && ldc >= b.n, "strides");
    assert!(a.len() >= (m - 1) * lda + b.k, "A is too small");
    assert!(c.len() >= (m - 1) * ldc + b.n, "C is too small");
    let nr = b.nr;
    let mr = tile_rows(b.isa, nr);
    let k = b.k;
    let mut atmp = vec![0f32; mr * k];
    let mut ctmp = vec![0f32; mr * nr];
    for i0 in (0..m).step_by(MC) {
        let i1 = (i0 + MC).min(m);
        for p in 0..b.panels {
            let bp = &b.data[p * k * nr..(p + 1) * k * nr];
            let bias = &b.bias[p * nr..(p + 1) * nr];
            let cols = nr.min(b.n - p * nr);
            let mut r0 = i0;
            while r0 < i1 {
                let rows = mr.min(i1 - r0);
                if rows == mr && cols == nr {
                    // SAFETY: the tile lies inside A and C (asserted above).
                    unsafe {
                        kernel(b.isa, nr, k, a.as_ptr().add(r0 * lda), lda, bp, bias, c.as_mut_ptr().add(r0 * ldc + p * nr), ldc, act, out)
                    };
                } else {
                    atmp.fill(0.0);
                    for r in 0..rows {
                        atmp[r * k..(r + 1) * k].copy_from_slice(&a[(r0 + r) * lda..(r0 + r) * lda + k]);
                    }
                    unsafe { kernel(b.isa, nr, k, atmp.as_ptr(), k, bp, bias, ctmp.as_mut_ptr(), nr, act, Out::Store) };
                    for r in 0..rows {
                        let dst = &mut c[(r0 + r) * ldc + p * nr..(r0 + r) * ldc + p * nr + cols];
                        let src = &ctmp[r * nr..r * nr + cols];
                        match out {
                            Out::Store => dst.copy_from_slice(src),
                            Out::Add => dst.iter_mut().zip(src).for_each(|(d, s)| *d += s),
                        }
                    }
                }
                r0 += rows;
            }
        }
    }
}

/// ``gemm`` with the rows of A and C split over a thread pool.
#[allow(clippy::too_many_arguments)]
pub fn gemm_par(par: &crate::par::Par, a: &[f32], lda: usize, m: usize, b: &PackedB, c: &mut [f32], ldc: usize, act: Act, out: Out) {
    let t = par.threads();
    if t <= 1 || m < 2 * MC {
        return gemm(a, lda, m, b, c, ldc, act, out);
    }
    let mr = tile_rows(b.isa, b.nr);
    let rows = m.div_ceil(t).div_ceil(mr) * mr;
    par.chunks(&mut c[..m * ldc], rows * ldc, |i, cc| {
        let r0 = i * rows;
        let n = cc.len() / ldc;
        gemm(&a[r0 * lda..], lda, n, b, cc, ldc, act, out);
    });
}

/// One MR x NR tile (MR from ``tile_rows``).
///
/// # Safety
/// A holds MR rows of K values at stride lda, B a K x NR panel, C an MR x NR
/// tile at stride ldc, and the CPU supports ``isa``.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn kernel(
    isa: Isa,
    nr: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: &[f32],
    bias: &[f32],
    c: *mut f32,
    ldc: usize,
    act: Act,
    out: Out,
) {
    let (b, bias) = (b.as_ptr(), bias.as_ptr());
    unsafe {
        match (isa, nr) {
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx512, 32) => x86::tile_z::<12, 2>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx512, 16) => x86::tile_z::<12, 1>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx512, 8) => x86::tile_y::<12, 1>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx2, 16) => x86::tile_y::<6, 2>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx2, 8) => x86::tile_y::<12, 1>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "aarch64")]
            (Isa::Neon, 16) => arm::tile_q::<6, 4>(k, a, lda, b, bias, c, ldc, act, out),
            #[cfg(target_arch = "aarch64")]
            (Isa::Neon, 8) => arm::tile_q::<8, 2>(k, a, lda, b, bias, c, ldc, act, out),
            (_, 16) => tile_scalar::<4, 16>(k, a, lda, b, bias, c, ldc, act, out),
            (_, 8) => tile_scalar::<4, 8>(k, a, lda, b, bias, c, ldc, act, out),
            _ => unreachable!("no kernel for panel width {nr}"),
        }
    }
}

#[inline(always)]
pub(crate) fn act_scalar(v: f32, act: Act) -> f32 {
    match act {
        Act::None => v,
        Act::Relu => v.max(0.0),
        Act::HardSwish => v * (v * (1.0 / 6.0) + 0.5).clamp(0.0, 1.0),
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn tile_scalar<const MR: usize, const NR: usize>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    ldc: usize,
    act: Act,
    out: Out,
) {
    let mut acc = [[0f32; NR]; MR];
    unsafe {
        for kk in 0..k {
            let bk = std::slice::from_raw_parts(b.add(kk * NR), NR);
            for (r, row) in acc.iter_mut().enumerate() {
                let av = *a.add(r * lda + kk);
                for j in 0..NR {
                    row[j] = av.mul_add(bk[j], row[j]);
                }
            }
        }
        for (r, row) in acc.iter().enumerate() {
            for j in 0..NR {
                let v = act_scalar(row[j] + *bias.add(j), act);
                let d = c.add(r * ldc + j);
                *d = if out == Out::Add { *d + v } else { v };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Act, Out};
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn act512(v: __m512, act: Act) -> __m512 {
        unsafe {
            match act {
                Act::None => v,
                Act::Relu => _mm512_max_ps(v, _mm512_setzero_ps()),
                Act::HardSwish => {
                    let t = _mm512_fmadd_ps(v, _mm512_set1_ps(1.0 / 6.0), _mm512_set1_ps(0.5));
                    let t = _mm512_min_ps(_mm512_max_ps(t, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
                    _mm512_mul_ps(v, t)
                }
            }
        }
    }

    /// MR rows x NV zmm vectors (NR = 16 NV).
    #[target_feature(enable = "avx512f,avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile_z<const MR: usize, const NV: usize>(
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        ldc: usize,
        act: Act,
        out: Out,
    ) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); NV]; MR];
            let mut bp = b;
            let mut ap = a;
            for _ in 0..k {
                let mut bv = [_mm512_setzero_ps(); NV];
                for (h, v) in bv.iter_mut().enumerate() {
                    *v = _mm512_loadu_ps(bp.add(16 * h));
                }
                for r in 0..MR {
                    let av = _mm512_set1_ps(*ap.add(r * lda));
                    for h in 0..NV {
                        acc[r][h] = _mm512_fmadd_ps(av, bv[h], acc[r][h]);
                    }
                }
                bp = bp.add(16 * NV);
                ap = ap.add(1);
            }
            for (r, row) in acc.iter().enumerate() {
                let d = c.add(r * ldc);
                for h in 0..NV {
                    let mut v = act512(_mm512_add_ps(row[h], _mm512_loadu_ps(bias.add(16 * h))), act);
                    if out == Out::Add {
                        v = _mm512_add_ps(v, _mm512_loadu_ps(d.add(16 * h)));
                    }
                    _mm512_storeu_ps(d.add(16 * h), v);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn act256(v: __m256, act: Act) -> __m256 {
        unsafe {
            match act {
                Act::None => v,
                Act::Relu => _mm256_max_ps(v, _mm256_setzero_ps()),
                Act::HardSwish => {
                    let t = _mm256_fmadd_ps(v, _mm256_set1_ps(1.0 / 6.0), _mm256_set1_ps(0.5));
                    let t = _mm256_min_ps(_mm256_max_ps(t, _mm256_setzero_ps()), _mm256_set1_ps(1.0));
                    _mm256_mul_ps(v, t)
                }
            }
        }
    }

    /// MR rows x NV ymm vectors (NR = 8 NV).
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile_y<const MR: usize, const NV: usize>(
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        ldc: usize,
        act: Act,
        out: Out,
    ) {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); NV]; MR];
            let mut bp = b;
            let mut ap = a;
            for _ in 0..k {
                let mut bv = [_mm256_setzero_ps(); NV];
                for (h, v) in bv.iter_mut().enumerate() {
                    *v = _mm256_loadu_ps(bp.add(8 * h));
                }
                for r in 0..MR {
                    let av = _mm256_broadcast_ss(&*ap.add(r * lda));
                    for h in 0..NV {
                        acc[r][h] = _mm256_fmadd_ps(av, bv[h], acc[r][h]);
                    }
                }
                bp = bp.add(8 * NV);
                ap = ap.add(1);
            }
            for (r, row) in acc.iter().enumerate() {
                let d = c.add(r * ldc);
                for h in 0..NV {
                    let mut v = act256(_mm256_add_ps(row[h], _mm256_loadu_ps(bias.add(8 * h))), act);
                    if out == Out::Add {
                        v = _mm256_add_ps(v, _mm256_loadu_ps(d.add(8 * h)));
                    }
                    _mm256_storeu_ps(d.add(8 * h), v);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{Act, Out};
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn act128(v: float32x4_t, act: Act) -> float32x4_t {
        unsafe {
            match act {
                Act::None => v,
                Act::Relu => vmaxq_f32(v, vdupq_n_f32(0.0)),
                Act::HardSwish => {
                    let t = vfmaq_f32(vdupq_n_f32(0.5), v, vdupq_n_f32(1.0 / 6.0));
                    let t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
                    vmulq_f32(v, t)
                }
            }
        }
    }

    /// MR rows x NV q vectors (NR = 4 NV).
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile_q<const MR: usize, const NV: usize>(
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        ldc: usize,
        act: Act,
        out: Out,
    ) {
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); NV]; MR];
            let mut bp = b;
            let mut ap = a;
            for _ in 0..k {
                let mut bv = [vdupq_n_f32(0.0); NV];
                for (h, v) in bv.iter_mut().enumerate() {
                    *v = vld1q_f32(bp.add(4 * h));
                }
                for r in 0..MR {
                    let av = vdupq_n_f32(*ap.add(r * lda));
                    for h in 0..NV {
                        acc[r][h] = vfmaq_f32(acc[r][h], av, bv[h]);
                    }
                }
                bp = bp.add(4 * NV);
                ap = ap.add(1);
            }
            for (r, row) in acc.iter().enumerate() {
                let d = c.add(r * ldc);
                for h in 0..NV {
                    let mut v = act128(vaddq_f32(row[h], vld1q_f32(bias.add(4 * h))), act);
                    if out == Out::Add {
                        v = vaddq_f32(v, vld1q_f32(d.add(4 * h)));
                    }
                    vst1q_f32(d.add(4 * h), v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32], act: Act) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    s += a[i * k + kk] as f64 * w[j * k + kk] as f64;
                }
                c[i * n + j] = act_scalar(s as f32 + bias[j], act);
            }
        }
        c
    }

    #[test]
    fn matches_naive_product() {
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for &(m, k, n) in
            &[(1, 1, 1), (13, 128, 256), (97, 256, 128), (5, 33, 1678), (200, 64, 17), (50, 16, 4), (31, 24, 12), (40, 48, 24), (9, 7, 40)]
        {
            let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd()).collect();
            for act in [Act::None, Act::Relu, Act::HardSwish] {
                let pb = PackedB::from_nk(&w, n, k, Some(&bias));
                let mut c = vec![0f32; m * n];
                gemm(&a, k, m, &pb, &mut c, n, act, Out::Store);
                let want = naive(&a, m, k, &w, n, &bias, act);
                let err = c.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                assert!(err < 1e-4, "{m}x{k}x{n} {act:?}: max error {err}");
                // residual add
                let mut c2 = want.clone();
                gemm(&a, k, m, &pb, &mut c2, n, act, Out::Add);
                let err = c2.iter().zip(&want).map(|(x, y)| (x - 2.0 * y).abs()).fold(0f32, f32::max);
                assert!(err < 2e-4, "{m}x{k}x{n} {act:?} add: max error {err}");
            }
        }
    }
}
