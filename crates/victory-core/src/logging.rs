//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.
//!
//! `VICTORY_LOG_ASYNC=1` (off by default): lines are formatted on the thread
//! that logs them and written to stderr by a thread of their own
//! ([`AsyncLog`]), so no thread that matters to the frame (the present, the
//! submitting thread, the recording threads) stands in a write to the log
//! file or on the stderr lock. After p3n1cillin/yuriko e6a4c9f8e (GPL-3.0;
//! there the present thread spent 36 ms a frame in `NtWriteFile` writing
//! the frame statistics, the recording threads 12 ms on the stderr lock).
//! The queue is bounded: past [`BOUND`] bytes waiting, an INFO (or more
//! verbose) line is counted and dropped, a WARN or ERROR line is written at
//! once, after everything before it. [`flush`] writes what waits (the crash
//! and panic reports, the host GPU's failure and the process's exits call
//! it).

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// The most verbose level this build can log.
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

/// Bytes of lines that may wait for the writer thread.
pub const BOUND: usize = 8 << 20;

/// Where an [`AsyncLog`]'s lines go.
type Sink = Arc<dyn Fn(&[u8]) + Send + Sync>;

#[derive(Default)]
struct Queue {
    lines: VecDeque<Vec<u8>>,
    bytes: usize,
    /// Lines queued or written at once so far, and of them written.
    taken: u64,
    written: u64,
    /// INFO (and more verbose) lines dropped since the last report of them.
    dropped: u64,
}

/// A log whose lines a thread of its own writes ([`AsyncLog::start`]).
pub struct AsyncLog {
    queue: Mutex<Queue>,
    /// Told when lines wait (the writer), and when some were written
    /// ([`AsyncLog::flush`]).
    more: Condvar,
    done: Condvar,
    /// Held while bytes go to the sink (taken while holding `queue`): the
    /// writer's batch and a line written at once never cross.
    writing: Mutex<()>,
    bound: usize,
    sink: Sink,
}

impl AsyncLog {
    /// A log of at most `bound` bytes waiting, written to `sink`.
    pub fn new(bound: usize, sink: Sink) -> Arc<AsyncLog> {
        Arc::new(AsyncLog {
            queue: Mutex::new(Queue::default()),
            more: Condvar::new(),
            done: Condvar::new(),
            writing: Mutex::new(()),
            bound,
            sink,
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts the writer thread.
    pub fn start(self: &Arc<Self>) -> std::io::Result<()> {
        let log = self.clone();
        std::thread::Builder::new().name("victory-log".into()).spawn(move || log.writer()).map(|_| ())
    }

    fn writer(&self) {
        let mut queue = self.lock();
        loop {
            while queue.lines.is_empty() && queue.dropped == 0 {
                queue = self.more.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
            let lines = std::mem::take(&mut queue.lines);
            let dropped = std::mem::take(&mut queue.dropped);
            queue.bytes = 0;
            // The batch goes out with the queue let go: loggers wait only
            // for the swap, not for the disk.
            let writing = self.writing.lock().unwrap_or_else(|e| e.into_inner());
            drop(queue);
            let mut out = Vec::with_capacity(lines.iter().map(Vec::len).sum::<usize>() + 128);
            for line in &lines {
                out.extend_from_slice(line);
            }
            if dropped > 0 {
                out.extend_from_slice(
                    format!("VIctory log: {dropped} INFO lines dropped, the queue to the log was full (VICTORY_LOG_ASYNC=1)\n")
                        .as_bytes(),
                );
            }
            (self.sink)(&out);
            drop(writing);
            queue = self.lock();
            queue.written += lines.len() as u64;
            self.done.notify_all();
        }
    }

    /// A formatted line (`urgent`: WARN or ERROR).
    pub fn push(&self, bytes: Vec<u8>, urgent: bool) {
        if bytes.is_empty() {
            return;
        }
        let mut queue = self.lock();
        if queue.bytes + bytes.len() > self.bound {
            if !urgent {
                queue.dropped += 1;
                return;
            }
            // Written at once, after everything queued before it.
            let lines = std::mem::take(&mut queue.lines);
            queue.bytes = 0;
            queue.taken += 1;
            let _writing = self.writing.lock().unwrap_or_else(|e| e.into_inner());
            for line in &lines {
                (self.sink)(line);
            }
            (self.sink)(&bytes);
            queue.written += lines.len() as u64 + 1;
            self.done.notify_all();
            return;
        }
        queue.bytes += bytes.len();
        queue.taken += 1;
        queue.lines.push_back(bytes);
        drop(queue);
        self.more.notify_one();
    }

    /// Waits until every line given so far is written, for at most
    /// `limit` (a writer that cannot write must not stop a crash report).
    /// Whether they were.
    pub fn flush(&self, limit: std::time::Duration) -> bool {
        let began = std::time::Instant::now();
        // Not `lock`: the thread that crashed may hold it.
        let mut queue = loop {
            match self.queue.try_lock() {
                Ok(queue) => break queue,
                Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) if began.elapsed() < limit => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(std::sync::TryLockError::WouldBlock) => return false,
            }
        };
        let target = queue.taken;
        self.more.notify_one();
        while queue.written < target {
            let left = limit.saturating_sub(began.elapsed());
            if left.is_zero() {
                return false;
            }
            queue = self.done.wait_timeout(queue, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        true
    }
}

/// The log [`init`] made asynchronous, if it did.
static ASYNC: OnceLock<Arc<AsyncLog>> = OnceLock::new();

/// Writes every line still waiting (`VICTORY_LOG_ASYNC=1`; at most two
/// seconds), then stderr's own buffer. Nothing without the switch.
pub fn flush() {
    if let Some(log) = ASYNC.get() {
        log.flush(std::time::Duration::from_secs(2));
    }
    let _ = std::io::stderr().flush();
}

/// A line being formatted, queued when it is done.
pub struct Line {
    log: Arc<AsyncLog>,
    urgent: bool,
    bytes: Vec<u8>,
}

impl Write for Line {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Line {
    fn drop(&mut self) {
        self.log.push(std::mem::take(&mut self.bytes), self.urgent);
    }
}

/// Makes a [`Line`] per event, urgent for WARN and ERROR.
pub struct Lines(pub Arc<AsyncLog>);

impl<'a> MakeWriter<'a> for Lines {
    type Writer = Line;
    fn make_writer(&'a self) -> Line {
        Line { log: self.0.clone(), urgent: false, bytes: Vec::with_capacity(256) }
    }
    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Line {
        Line { log: self.0.clone(), urgent: *meta.level() <= tracing::Level::WARN, bytes: Vec::with_capacity(256) }
    }
}

pub fn init(default_level: &str) {
    let filter = std::env::var("VICTORY_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .ok()
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level));
    let registry = tracing_subscriber::registry().with(CAP).with(filter);
    if std::env::var_os("VICTORY_LOG_ASYNC").is_some_and(|v| v == "1") {
        let log = AsyncLog::new(
            BOUND,
            Arc::new(|bytes: &[u8]| {
                let mut err = std::io::stderr().lock();
                let _ = err.write_all(bytes);
                let _ = err.flush();
            }),
        );
        if log.start().is_ok() {
            let _ = ASYNC.set(log.clone());
            let _ = registry
                .with(tracing_subscriber::fmt::layer().with_target(true).with_writer(Lines(log)))
                .try_init();
            tracing::info!("log lines are written by a thread of their own (VICTORY_LOG_ASYNC=1)");
            return;
        }
    }
    let _ = registry
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collecting(bound: usize) -> (Arc<AsyncLog>, Arc<Mutex<Vec<u8>>>) {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink_out = out.clone();
        let log = AsyncLog::new(bound, Arc::new(move |b: &[u8]| sink_out.lock().unwrap().extend_from_slice(b)));
        (log, out)
    }

    /// Lines of several threads all arrive below the bound, each thread's
    /// in its order; a flush waits until they are written.
    #[test]
    fn every_line_arrives_in_its_threads_order() {
        let (log, out) = collecting(BOUND);
        log.start().unwrap();
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let log = log.clone();
                std::thread::spawn(move || {
                    for i in 0..2000 {
                        log.push(format!("{t} {i}\n").into_bytes(), i % 97 == 0);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(log.flush(std::time::Duration::from_secs(10)), "the flush timed out");
        let text = String::from_utf8(out.lock().unwrap().clone()).unwrap();
        let mut next = [0usize; 4];
        for line in text.lines() {
            let (t, i) = line.split_once(' ').unwrap();
            let (t, i): (usize, usize) = (t.parse().unwrap(), i.parse().unwrap());
            assert_eq!(i, next[t], "thread {t}'s lines out of order");
            next[t] += 1;
        }
        assert_eq!(next, [2000; 4], "lines were lost below the bound");
    }

    /// Past the bound INFO lines are dropped and counted, a warning is
    /// written at once after what came before it; the drop is reported.
    #[test]
    fn past_the_bound_info_is_dropped_and_warnings_are_not() {
        let (log, out) = collecting(16);
        // No writer yet: the queue fills.
        log.push(b"one 12345\n".to_vec(), false);
        log.push(b"two 12345\n".to_vec(), false);
        log.push(b"WARN three\n".to_vec(), true);
        assert_eq!(String::from_utf8(out.lock().unwrap().clone()).unwrap(), "one 12345\nWARN three\n");
        log.start().unwrap();
        log.push(b"four\n".to_vec(), false);
        assert!(log.flush(std::time::Duration::from_secs(10)));
        let text = String::from_utf8(out.lock().unwrap().clone()).unwrap();
        assert!(text.starts_with("one 12345\nWARN three\n"), "{text}");
        assert!(text.contains("four\n") && text.contains("VIctory log: 1 INFO lines dropped"), "{text}");
        assert!(!text.contains("two"), "{text}");
    }

    /// The flush at exit writes what waits even when the writer is idle.
    #[test]
    fn a_flush_writes_what_waits() {
        let (log, out) = collecting(BOUND);
        log.start().unwrap();
        for i in 0..100 {
            log.push(format!("{i}\n").into_bytes(), false);
        }
        assert!(log.flush(std::time::Duration::from_secs(10)));
        assert_eq!(String::from_utf8(out.lock().unwrap().clone()).unwrap().lines().count(), 100);
    }
}
