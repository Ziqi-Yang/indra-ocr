//! Optional stage timers (INDRA_ENGINE_PROFILE=1): per-thread totals, printed
//! with ``report``.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INDRA_ENGINE_PROFILE").is_some())
}

thread_local! {
    static TOTALS: RefCell<BTreeMap<&'static str, (f64, u64)>> = const { RefCell::new(BTreeMap::new()) };
}

/// Laps of one run; does nothing unless profiling is on.
pub struct Timer {
    last: Option<Instant>,
}

impl Timer {
    pub fn start() -> Timer {
        Timer { last: enabled().then(Instant::now) }
    }
    /// Time since the previous lap, added to ``name``.
    pub fn lap(&mut self, name: &'static str) {
        if let Some(t) = self.last {
            let now = Instant::now();
            let ms = (now - t).as_secs_f64() * 1000.0;
            TOTALS.with(|m| {
                let mut m = m.borrow_mut();
                let e = m.entry(name).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
            });
            self.last = Some(now);
        }
    }
}

/// This thread's totals (ms), as "name: ms" lines.
pub fn report() -> String {
    TOTALS.with(|m| m.borrow().iter().map(|(k, (ms, n))| format!("{k}: {ms:.1} ms ({n} laps)\n")).collect())
}
