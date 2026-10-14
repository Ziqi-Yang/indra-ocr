//! Native CPU inference for the Indra OCR models.
//!
//! Each model is a fixed architecture: its weights are read from the ONNX file
//! (the graph is checked against the architecture) and run by fused,
//! model-specific code on matrix kernels chosen for the CPU at run time
//! (AVX-512, AVX2/FMA, NEON or portable Rust).

pub mod classifier;
pub mod cnn;
pub mod cpu;
pub mod gemm;
pub mod math;
pub mod onnx;
pub mod order;
pub mod par;
pub mod qgemm;
pub mod profile;
pub mod rec;
pub mod simd;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The model file is missing, unreadable, or not the architecture this
    /// engine implements.
    #[error("{0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
