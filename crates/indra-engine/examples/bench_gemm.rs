//! GFLOP/s of the GEMM on the recognizer's shapes.
use std::time::Instant;

use indra_engine::gemm::{Act, Out, PackedB, gemm};

fn main() {
    println!("isa: {}", indra_engine::cpu::isa().name());
    for &(m, k, n) in &[(256, 128, 256), (256, 256, 128), (256, 128, 1678), (2048, 128, 256), (32, 128, 256)] {
        let a = vec![0.5f32; m * k];
        let w = vec![0.25f32; n * k];
        let pb = PackedB::from_nk(&w, n, k, None);
        let mut c = vec![0f32; m * n];
        let reps = (2e9 / (2.0 * (m * k * n) as f64)) as usize + 1;
        gemm(&a, k, m, &pb, &mut c, n, Act::HardSwish, Out::Store);
        let t0 = Instant::now();
        for _ in 0..reps {
            gemm(&a, k, m, &pb, &mut c, n, Act::HardSwish, Out::Store);
        }
        let s = t0.elapsed().as_secs_f64();
        println!("{m}x{k}x{n}: {:.1} GFLOP/s", 2.0 * (m * k * n * reps) as f64 / s / 1e9);
    }
}
