//! Intra-op parallelism: a model call can split its work (line chunks, rows of
//! a feature map) over a thread pool. With one thread (the default, and what
//! page-parallel workers use) everything runs on the calling thread.

use rayon::prelude::*;

/// A thread pool for one model user, or none.
#[derive(Default)]
pub struct Par {
    pool: Option<rayon::ThreadPool>,
}

impl Par {
    /// ``threads`` <= 1: no pool.
    pub fn new(threads: usize) -> Par {
        let pool = (threads > 1).then(|| rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok()).flatten();
        Par { pool }
    }

    pub fn threads(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    /// ``f(i)`` for i in 0..n, collected in order.
    pub fn map<T: Send>(&self, n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
        match &self.pool {
            Some(p) if n > 1 => p.install(|| (0..n).into_par_iter().map(&f).collect()),
            _ => (0..n).map(f).collect(),
        }
    }

    /// ``f(i, chunk)`` over ``data`` split into chunks of ``len`` elements.
    pub fn chunks<T: Send>(&self, data: &mut [T], len: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
        let len = len.max(1);
        match &self.pool {
            Some(p) if data.len() > len => p.install(|| data.par_chunks_mut(len).enumerate().for_each(|(i, c)| f(i, c))),
            _ => data.chunks_mut(len).enumerate().for_each(|(i, c)| f(i, c)),
        }
    }
}

static SERIAL: std::sync::OnceLock<Par> = std::sync::OnceLock::new();

/// No pool: run on the calling thread.
pub fn serial() -> &'static Par {
    SERIAL.get_or_init(Par::default)
}
