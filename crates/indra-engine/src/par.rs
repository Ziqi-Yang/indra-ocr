//! Intra-op parallelism: a model call can split its work (line chunks, rows of
//! a feature map) over a pool of threads. With one thread (the default, and
//! what page-parallel workers use) everything runs on the calling thread.
//!
//! The pool's idle workers block (no spinning): a model call alternates many
//! short parallel steps with serial ones, and spinning workers would take CPU
//! from the serial steps (badly so on virtual machines).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// One parallel step: ``n`` tasks, claimed by index.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    left: AtomicUsize,
}

// The closure is only called while the submitting thread waits in ``run``.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Claim and run tasks until none are left; true when this call ran the last.
    fn work(&self) -> bool {
        let mut last = false;
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                return last;
            }
            // SAFETY: the submitter keeps the closure alive until left == 0.
            unsafe { (*self.f)(i) };
            last = self.left.fetch_sub(1, Ordering::AcqRel) == 1;
        }
    }
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

struct State {
    job: Option<Arc<Job>>,
    generation: u64,
    quit: bool,
}

/// A fork-join pool for one model user, or none.
pub struct Par {
    shared: Option<Arc<Shared>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
    /// one step at a time per pool
    busy: Mutex<()>,
}

impl Default for Par {
    fn default() -> Self {
        Par { shared: None, workers: vec![], threads: 1, busy: Mutex::new(()) }
    }
}

impl Par {
    /// ``threads`` <= 1: no pool (``threads`` counts the calling thread).
    pub fn new(threads: usize) -> Par {
        if threads <= 1 {
            return Par::default();
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, generation: 0, quit: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let workers = (1..threads)
            .map(|_| {
                let sh = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut st = sh.state.lock().unwrap();
                            while !st.quit && (st.generation == seen || st.job.is_none()) {
                                st = sh.wake.wait(st).unwrap();
                            }
                            if st.quit {
                                return;
                            }
                            seen = st.generation;
                            st.job.clone().unwrap()
                        };
                        if job.work() {
                            let _g = sh.state.lock().unwrap();
                            sh.done.notify_all();
                        }
                    }
                })
            })
            .collect();
        Par { shared: Some(shared), workers, threads, busy: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// ``f(i)`` for i in 0..n, on the pool and the calling thread.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let sh = match &self.shared {
            Some(sh) if n > 1 => sh,
            _ => return (0..n).for_each(f),
        };
        let _one = self.busy.lock().unwrap();
        // SAFETY: the lifetime is erased only while this call waits for every
        // task (left == 0) before returning.
        let fp: *const (dyn Fn(usize) + Sync) =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        let job = Arc::new(Job { f: fp, n, next: AtomicUsize::new(0), left: AtomicUsize::new(n) });
        {
            let mut st = sh.state.lock().unwrap();
            st.job = Some(job.clone());
            st.generation += 1;
            sh.wake.notify_all();
        }
        job.work();
        let mut st = sh.state.lock().unwrap();
        while job.left.load(Ordering::Acquire) != 0 {
            st = sh.done.wait(st).unwrap();
        }
        st.job = None;
    }

    /// ``f(i)`` for i in 0..n, collected in order.
    pub fn map<T: Send>(&self, n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
        let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
        self.run(n, &|i| *slots[i].lock().unwrap() = Some(f(i)));
        slots.into_iter().map(|s| s.into_inner().unwrap().expect("every task ran")).collect()
    }

    /// ``f(i, chunk)`` over ``data`` split into chunks of ``len`` elements.
    pub fn chunks<T: Send>(&self, data: &mut [T], len: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
        let len = len.max(1);
        let n = data.len().div_ceil(len);
        let total = data.len();
        let base = SendPtr(data.as_mut_ptr());
        self.run(n, &|i| {
            let start = i * len;
            let end = (start + len).min(total);
            // SAFETY: chunk i is [start, end), disjoint from every other chunk.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
            f(i, chunk)
        });
    }
}

struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

impl Drop for Par {
    fn drop(&mut self) {
        if let Some(sh) = &self.shared {
            sh.state.lock().unwrap().quit = true;
            sh.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

static SERIAL: std::sync::OnceLock<Par> = std::sync::OnceLock::new();

/// No pool: run on the calling thread.
pub fn serial() -> &'static Par {
    SERIAL.get_or_init(Par::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_and_chunks() {
        let p = Par::new(4);
        for round in 0..50 {
            let v = p.map(37, |i| i * i + round);
            assert_eq!(v, (0..37).map(|i| i * i + round).collect::<Vec<_>>());
            let mut d = vec![0usize; 1000];
            p.chunks(&mut d, 33, |i, c| c.iter_mut().for_each(|x| *x = i));
            assert!(d.iter().enumerate().all(|(k, &x)| x == k / 33));
        }
    }
}
