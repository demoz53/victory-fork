//! Clock reads on the hot paths (`VICTORY_TIMING_LEAN=1`; off by default).
//!
//! A clock read (`Instant::now`, a `QueryPerformanceCounter`) costs some
//! 36 ns on this PC: 72.7 ns for the pair `now` + `elapsed` against 5.0 ns
//! for a sampled timing's average and 2.4 ns for a tick count
//! ([`tests::clock_reads_cost`], diag profile, idle PC). The recording threads read it
//! twice for every list call (48,000 a frame in the open world), once for
//! every host copy or texture looked up (`crate::host::moment`), and twice
//! for every texture view bound; the submitting thread four times for
//! every title copy and twice for every refreshed range (45,000 a frame).
//! The profiler put 5.5% of each recording thread's time and 3.7% of the
//! submitting thread's in `Instant::now`.
//!
//! With the switch:
//! - the statistics' timings on those paths are taken for one call in
//!   [`EVERY`] per thread ([`Timer::sampled`]) and counted [`EVERY`] times:
//!   their counts stay exact, their times become estimates;
//! - per-call timings whose sums only the diagnostics lines print, and that
//!   are cheap to lose, are not taken ([`Timer::stats`]);
//! - [`crate::host::moment`] (tenths of a second, for letting go of idle
//!   copies) reads the system's tick count, which costs no clock read.
//!
//! Timings a statistics switch asks for (`VICTORY_FRAME_PARTS=1`) are taken
//! in full whatever this says.

use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

/// One sampled timing in this many.
pub(crate) const EVERY: u64 = 16;

/// Whether the hot paths read the clock less (`VICTORY_TIMING_LEAN=1`).
pub(crate) fn lean() -> bool {
    #[cfg(test)]
    {
        let forced = TEST_LEAN.with(Cell::get);
        if forced != 0 {
            return forced == 2;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TIMING_LEAN").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                every = EVERY,
                "VICTORY_TIMING_LEAN=1: list-call, title-copy and refresh times sampled (one call in 16, counted 16 times), per-slot heap-pass parts, dynamic-slot search, texture-alias, copy-move and wait-check times not taken, idle moments from the tick count"
            );
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests, for the calling thread: 0 as the switch says, 1 off, 2 on.
    pub(crate) static TEST_LEAN: Cell<u8> = const { Cell::new(0) };
}

thread_local! {
    /// Calls of [`Timer::sampled`] on this thread so far.
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

/// A timing begun, or not taken.
#[derive(Clone, Copy)]
pub(crate) struct Timer {
    began: Option<Instant>,
    /// What one taken timing stands for: 1, or [`EVERY`] when sampled.
    weight: u64,
}

impl Timer {
    /// Always taken (what the code did before).
    #[inline]
    pub(crate) fn always() -> Self {
        Timer { began: Some(Instant::now()), weight: 1 }
    }

    /// Taken unless [`lean`]: a timing only the statistics read.
    #[inline]
    pub(crate) fn stats() -> Self {
        if lean() { Timer { began: None, weight: 0 } } else { Self::always() }
    }

    /// Taken for every call, or with [`lean`] for one call in [`EVERY`] on
    /// this thread, which then counts for [`EVERY`].
    #[inline]
    pub(crate) fn sampled() -> Self {
        if !lean() {
            return Self::always();
        }
        let n = CALLS.with(|c| {
            let n = c.get();
            c.set(n.wrapping_add(1));
            n
        });
        if n.is_multiple_of(EVERY) { Timer { began: Some(Instant::now()), weight: EVERY } } else { Timer { began: None, weight: 0 } }
    }

    /// Whether the clock was read.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn taken(&self) -> bool {
        self.began.is_some()
    }

    /// When it began, if it was taken.
    #[inline]
    pub(crate) fn began(&self) -> Option<Instant> {
        self.began
    }

    /// Nanoseconds since it began, weighted; 0 if not taken.
    #[inline]
    pub(crate) fn ns(&self) -> u64 {
        self.began.map_or(0, |b| (b.elapsed().as_nanos() as u64).saturating_mul(self.weight))
    }

    /// Microseconds since it began, weighted; 0 if not taken. A sampled
    /// timing is weighted before it is cut to microseconds (short calls
    /// would count as 0 otherwise).
    #[inline]
    pub(crate) fn us(&self) -> u64 {
        if self.weight == 1 {
            return self.began.map_or(0, |b| b.elapsed().as_micros() as u64);
        }
        self.ns() / 1000
    }
}

/// Milliseconds of the system's tick count (`GetTickCount64`: read from
/// memory the kernel keeps up to date, no clock read; steps of 10-16 ms)
/// since the first call.
pub(crate) fn tick_ms() -> u64 {
    // SAFETY: a plain query.
    let now = || unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() };
    static EPOCH: OnceLock<u64> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(now);
    now().saturating_sub(epoch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lean: one call in 16 reads the clock and counts 16 times; off: every
    /// call does, once. (The switch is forced for this test's thread
    /// only.)
    #[test]
    fn sampled_timings_read_the_clock_once_in_sixteen() {
        TEST_LEAN.set(2);
        let taken = (0..160).filter(|_| Timer::sampled().taken()).count();
        assert_eq!(taken, 10);
        assert!(!Timer::stats().taken());
        let weighted = (0..32).map(|_| Timer::sampled()).find(Timer::taken).unwrap();
        assert_eq!(weighted.weight, EVERY);
        TEST_LEAN.set(1);
        assert_eq!((0..160).filter(|_| Timer::sampled().taken()).count(), 160);
        assert!(Timer::stats().taken());
        TEST_LEAN.set(0);
    }

    /// The micro-measurement behind the switch: a clock pair against the
    /// sampled timer's untaken path (printed; `--nocapture`).
    #[test]
    #[ignore = "a measurement: cargo test -p victory-gpu --profile diag --lib timing -- --ignored --nocapture"]
    fn clock_reads_cost() {
        const N: u32 = 2_000_000;
        let mut sum = 0u64;
        let t = Instant::now();
        for _ in 0..N {
            sum = sum.wrapping_add(Timer::always().ns());
        }
        let full = t.elapsed().as_nanos() as f64 / f64::from(N);
        TEST_LEAN.set(2);
        let t = Instant::now();
        for _ in 0..N {
            sum = sum.wrapping_add(Timer::sampled().ns());
        }
        let lean = t.elapsed().as_nanos() as f64 / f64::from(N);
        let t = Instant::now();
        for _ in 0..N {
            sum = sum.wrapping_add(tick_ms());
        }
        let tick = t.elapsed().as_nanos() as f64 / f64::from(N);
        TEST_LEAN.set(0);
        println!("timing per call: clock pair {full:.1} ns, sampled 1/16 {lean:.1} ns, tick count {tick:.1} ns ({sum})");
    }
}
