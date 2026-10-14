//! Instruction-set selection, once per process: the widest vector unit the CPU
//! has, unless INDRA_FORCE_ISA (avx512 | avx2 | neon | scalar) asks for a
//! narrower one.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512F (with AVX2 and FMA)
    Avx512,
    /// AVX2 with FMA
    Avx2,
    /// AArch64 Advanced SIMD (always has FMA)
    Neon,
    /// portable Rust
    Scalar,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => "avx512",
            Isa::Avx2 => "avx2",
            Isa::Neon => "neon",
            Isa::Scalar => "scalar",
        }
    }
}

fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        let fma = std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2");
        if fma && std::arch::is_x86_feature_detected!("avx512f") {
            return Isa::Avx512;
        }
        if fma {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Isa::Neon;
        }
    }
    Isa::Scalar
}

/// Whether this CPU can run code compiled for ``isa``.
pub fn supported(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        Isa::Avx512 => detect() == Isa::Avx512,
        Isa::Avx2 => matches!(detect(), Isa::Avx512 | Isa::Avx2),
        Isa::Neon => detect() == Isa::Neon,
    }
}

/// The instruction set the kernels use.
pub fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        let best = detect();
        match std::env::var("INDRA_FORCE_ISA").ok().as_deref() {
            Some("avx512") if supported(Isa::Avx512) => Isa::Avx512,
            Some("avx2") if supported(Isa::Avx2) => Isa::Avx2,
            Some("neon") if supported(Isa::Neon) => Isa::Neon,
            Some("scalar") => Isa::Scalar,
            _ => best,
        }
    })
}
