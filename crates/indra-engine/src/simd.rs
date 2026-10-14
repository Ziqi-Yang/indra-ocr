//! Element-wise kernels written once in plain Rust and compiled per ISA
//! (``multiversion!``), so the compiler vectorizes them for the CPU at hand.

use crate::cpu::{Isa, isa};

/// Define ``$name`` as plain Rust, plus copies compiled for AVX-512 and AVX2,
/// chosen at run time.
macro_rules! multiversion {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$m])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn imp($($arg: $ty),*) $(-> $ret)? $body
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx2,fma")]
            unsafe fn avx512($($arg: $ty),*) $(-> $ret)? { imp($($arg),*) }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn avx2($($arg: $ty),*) $(-> $ret)? { imp($($arg),*) }
            match $crate::cpu::isa() {
                #[cfg(target_arch = "x86_64")]
                $crate::cpu::Isa::Avx512 => unsafe { avx512($($arg),*) },
                #[cfg(target_arch = "x86_64")]
                $crate::cpu::Isa::Avx2 => unsafe { avx2($($arg),*) },
                _ => imp($($arg),*),
            }
        }
    };
}

pub(crate) use multiversion;

/// e^x for float32, branch-free (vectorizes): 2^n times a degree-6 polynomial
/// of the remainder; relative error about 2e-7.
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    let x = x.clamp(-87.33, 88.72);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    // r = x - n ln 2, in two parts for precision
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_45 + r * (0.008_333_45 + r * 0.001_394_8)))));
    let bits = ((n as i32 + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// ln(x) for float32 x > 0, branch-free (vectorizes): exponent and mantissa
/// in [sqrt(1/2), sqrt(2)), then the Cephes polynomial; error about 1 ulp.
#[inline(always)]
pub fn ln_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 126;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000); // [0.5, 1)
    let small = m < std::f32::consts::FRAC_1_SQRT_2;
    e -= small as i32;
    m = if small { m + m - 1.0 } else { m - 1.0 };
    let z = m * m;
    let mut y = 7.037_683_6e-2;
    for c in
        [-1.151_461e-1, 1.167_699_9e-1, -1.242_014_1e-1, 1.424_932_3e-1, -1.666_805_8e-1, 2.000_071_4e-1, -2.499_999_4e-1, 3.333_333e-1]
    {
        y = y * m + c;
    }
    y = y * m * z;
    let ef = e as f32;
    y += -2.121_944_4e-4 * ef;
    y += -0.5 * z;
    m + y + 0.693_359_4 * ef
}

multiversion! {
    /// log(1 + e^x) in place, without overflow: max(x, 0) + log1p(e^-|x|).
    pub fn softplus(x: &mut [f32]) {
        for v in x.iter_mut() {
            let t = exp_f32(-v.abs());
            // log1p(t) = ln(u) t / (u - 1), u = 1 + t (corrects the rounding of u)
            let u = 1.0 + t;
            let l = if u == 1.0 { t } else { ln_f32(u) * (t / (u - 1.0)) };
            *v = v.max(0.0) + l;
        }
    }
}

multiversion! {
    /// Depthwise 1-D convolution along time of a [T, C] sequence (zero padding
    /// ``k / 2`` on each side): out[t, c] = b[c] + sum_j w[j, c] x[t + j - k/2, c].
    /// ``w`` is laid out [K, C].
    pub fn dwconv1d(x: &[f32], t: usize, c: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
        let k = w.len() / c;
        let half = k / 2;
        for i in 0..t {
            let o = &mut out[i * c..(i + 1) * c];
            o.copy_from_slice(&b[..c]);
            for j in 0..k {
                let src = i + j;
                if src < half || src - half >= t {
                    continue;
                }
                let xr = &x[(src - half) * c..(src - half + 1) * c];
                let wr = &w[j * c..(j + 1) * c];
                for ((o, &xv), &wv) in o.iter_mut().zip(xr).zip(wr) {
                    *o += wv * xv;
                }
            }
        }
    }
}

multiversion! {
    /// Column sums of a [T, C] matrix into ``out`` [C].
    pub fn col_sums(x: &[f32], c: usize, out: &mut [f32]) {
        out[..c].fill(0.0);
        for row in x.chunks_exact(c) {
            for (o, &v) in out.iter_mut().zip(row) {
                *o += v;
            }
        }
    }
}

multiversion! {
    /// Each row of a [T, C] matrix times ``s`` [C], in place.
    pub fn scale_cols(x: &mut [f32], s: &[f32]) {
        let c = s.len();
        for row in x.chunks_exact_mut(c) {
            for (v, &f) in row.iter_mut().zip(s) {
                *v *= f;
            }
        }
    }
}

multiversion! {
    /// The index of the first maximum of ``row`` and the maximum.
    pub fn argmax_row(row: &[f32]) -> (usize, f32) {
        // the maximum first (vectorizes), then its first position
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            m = m.max(v);
        }
        let i = row.iter().position(|&v| v == m).unwrap_or(0);
        (i, m)
    }
}

multiversion! {
    /// sum_j exp(row[j] - m)
    pub fn sum_exp(row: &[f32], m: f32) -> f32 {
        let mut acc = [0f32; 16];
        let mut chunks = row.chunks_exact(16);
        for ch in &mut chunks {
            for (a, &v) in acc.iter_mut().zip(ch) {
                *a += exp_f32(v - m);
            }
        }
        let mut s: f32 = acc.iter().sum();
        for &v in chunks.remainder() {
            s += exp_f32(v - m);
        }
        s
    }
}

/// erf(x) for float32 (a Chebyshev-fitted erfc: |error| < 1.2e-7).
pub fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * x.abs() as f64);
    let y = 1.0
        - t * (-((x as f64) * (x as f64)) - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

/// GELU with the exact erf, in place.
pub fn gelu(x: &mut [f32]) {
    for v in x {
        *v = *v * 0.5 * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

/// The logistic function.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// For tests and reports: which ISA the kernels run.
pub fn active_isa() -> Isa {
    isa()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_accurate() {
        let mut worst = 0f32;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (a, b) = (exp_f32(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 5e-7, "relative error {worst}");
        assert!(exp_f32(-200.0) < 1e-37 && exp_f32(-200.0) >= 0.0);
    }

    #[test]
    fn softplus_accuracy() {
        let mut worst = 0f32;
        let mut x = -30.0f32;
        while x < 30.0 {
            let mut v = [x];
            softplus(&mut v);
            let want = if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() };
            worst = worst.max((v[0] - want).abs() / want.max(1e-30));
            x += 0.0731;
        }
        assert!(worst < 1e-6, "relative error {worst}");
        for &x in &[0.5f32, 1.0, 1.5, 3.0, 1e-3, 1e5] {
            assert!((ln_f32(x) - x.ln()).abs() <= 2e-7 * x.ln().abs().max(1.0), "ln({x})");
        }
    }

    #[test]
    fn erf_accuracy() {
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (-1.5, -0.966_105_15), (3.0, 0.999_977_9)] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn dwconv_matches_definition() {
        let (t, c, k) = (7, 3, 5);
        let x: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.1).collect();
        let w: Vec<f32> = (0..k * c).map(|i| (i as f32 - 7.0) * 0.05).collect();
        let b = [0.5, -0.5, 0.25];
        let mut out = vec![0f32; t * c];
        dwconv1d(&x, t, c, &w, &b, &mut out);
        for i in 0..t {
            for ch in 0..c {
                let mut s = b[ch];
                for j in 0..k {
                    let src = i as isize + j as isize - 2;
                    if (0..t as isize).contains(&src) {
                        s += w[j * c + ch] * x[src as usize * c + ch];
                    }
                }
                assert!((out[i * c + ch] - s).abs() < 1e-5);
            }
        }
    }
}
