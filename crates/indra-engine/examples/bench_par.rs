//! Thread scaling of the parallel GEMM on a memory-bound and a compute-bound shape.
use std::time::Instant;

use indra_engine::gemm::{Act, Out, PackedB, gemm_par};
use indra_engine::par::Par;

fn main() {
    for &(m, k, n) in &[(51200usize, 24usize, 48usize), (8192, 256, 256)] {
        let a = vec![0.5f32; m * k];
        let w = vec![0.25f32; n * k];
        let pb = PackedB::from_nk(&w, n, k, None);
        let mut c = vec![0f32; m * n];
        for t in [1, 2, 4, 8] {
            let par = Par::new(t);
            gemm_par(&par, &a, k, m, &pb, &mut c, n, Act::HardSwish, Out::Store);
            let reps = 20;
            let t0 = Instant::now();
            for _ in 0..reps {
                gemm_par(&par, &a, k, m, &pb, &mut c, n, Act::HardSwish, Out::Store);
            }
            let s = t0.elapsed().as_secs_f64() / reps as f64;
            let gb = ((m * k + m * n) * 4) as f64 / s / 1e9;
            println!("{m}x{k}x{n} threads {t}: {:.2} ms, {:.0} GFLOP/s, {gb:.1} GB/s", s * 1e3, 2.0 * (m * k * n) as f64 / s / 1e9);
        }
    }
}
