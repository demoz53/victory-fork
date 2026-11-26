//! Waits on the shared locks of the recording path, by lock
//! (`VICTORY_LOCK_STATS=1`; off by default).
//!
//! With the switch every lock named in [`Lock`] is first tried; a try that
//! finds it held counts one wait, and the time until the lock is had is
//! added to that lock's sum. The frame line then says, per frame and
//! summed over all threads, how often each was waited for and how long:
//! "lock waits per frame: <name> <n> waits <ms> ms, ...". Without the
//! switch a lock is taken as before, after one load of a flag set once.
//!
//! The pipeline cache is not among them: a draw finds its host pipeline in
//! a once-set cell (`crate::pipeline::make::ready`), without a lock.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

/// The locks counted, in the order the frame line names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lock {
    /// `descriptor::host::HostHeap::made`: the record each host slot of a
    /// shader-visible heap was made from, and the heap map (every slot
    /// lookup of a table entry, every descriptor write the title makes).
    Slots,
    /// `HostHeap::kept`: the host resource each slot views (every slot
    /// made, dynamic slots of tables included).
    Kept,
    /// `HostHeap::next`: the ring of dynamic slots tables are made in.
    Ring,
    /// `HostHeap::textures`: the queue of texture slots picked by index.
    Textures,
    /// `descriptor::SHADER_HEAPS`: shader-visible heaps by address (every
    /// descriptor write and copy the title makes).
    Heaps,
    /// `descriptor::VIEWS`: the table of views the slot records name
    /// (every record looked up).
    Views,
    /// `shadow::BUFFERS`: the title's buffers by address, with their host
    /// copies (the resource table every buffer lookup takes).
    Buffers,
    /// `resource::PLACED_TEXTURES`: placed textures by address and shape.
    Placed,
    /// `host::RETIRED`: objects let go once the submissions running now are
    /// done.
    Retired,
    /// `host::Host::submitted`: the submission lock.
    Submit,
}

const NAMES: [&str; COUNT] =
    ["slots", "kept", "ring", "textures", "heaps", "views", "buffers", "placed", "retired", "submit"];
const COUNT: usize = 10;

/// Per lock: waits, nanoseconds waited, times taken.
static WAITS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
static NANOS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
static TAKEN: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];

/// Whether waits are counted (`VICTORY_LOCK_STATS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_LOCK_STATS").as_deref() == Ok("1"))
}

fn waited(which: Lock, since: std::time::Instant) {
    let i = which as usize;
    WAITS[i].fetch_add(1, Relaxed);
    NANOS[i].fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
}

/// `mutex`, locked (a poisoned lock is taken all the same, as everywhere in
/// the crate), its wait counted as `which`'s with the switch.
pub(crate) fn lock<T>(mutex: &Mutex<T>, which: Lock) -> MutexGuard<'_, T> {
    if !on() {
        return mutex.lock().unwrap_or_else(|e| e.into_inner());
    }
    TAKEN[which as usize].fetch_add(1, Relaxed);
    match mutex.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let since = std::time::Instant::now();
            let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
            waited(which, since);
            guard
        }
    }
}

/// `lock`'s shared side, read.
#[allow(dead_code)] // used by VICTORY_HEAP_LOCKS
pub(crate) fn read<T>(lock: &RwLock<T>, which: Lock) -> RwLockReadGuard<'_, T> {
    if !on() {
        return lock.read().unwrap_or_else(|e| e.into_inner());
    }
    TAKEN[which as usize].fetch_add(1, Relaxed);
    match lock.try_read() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let since = std::time::Instant::now();
            let guard = lock.read().unwrap_or_else(|e| e.into_inner());
            waited(which, since);
            guard
        }
    }
}

/// `lock`'s exclusive side, for writing.
#[allow(dead_code)] // used by VICTORY_HEAP_LOCKS
pub(crate) fn write<T>(lock: &RwLock<T>, which: Lock) -> RwLockWriteGuard<'_, T> {
    if !on() {
        return lock.write().unwrap_or_else(|e| e.into_inner());
    }
    TAKEN[which as usize].fetch_add(1, Relaxed);
    match lock.try_write() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let since = std::time::Instant::now();
            let guard = lock.write().unwrap_or_else(|e| e.into_inner());
            waited(which, since);
            guard
        }
    }
}

/// The counts since the last call, per lock: (taken, waits, nanoseconds).
pub(crate) fn take() -> [(u64, u64, u64); COUNT] {
    std::array::from_fn(|i| (TAKEN[i].swap(0, Relaxed), WAITS[i].swap(0, Relaxed), NANOS[i].swap(0, Relaxed)))
}

/// The frame line's part ([`take`] over `frames` frames), or `None` without
/// the switch.
pub(crate) fn line(frames: f64) -> Option<String> {
    if !on() {
        return None;
    }
    let frames = frames.max(1.0);
    let parts: Vec<String> = take()
        .iter()
        .zip(NAMES)
        .map(|(&(taken, waits, nanos), name)| {
            format!(
                "{name} {:.0} waits {:.2} ms (of {:.0} taken)",
                waits as f64 / frames,
                nanos as f64 / 1e6 / frames,
                taken as f64 / frames
            )
        })
        .collect();
    Some(format!("lock waits per frame (all threads; VICTORY_LOCK_STATS=1): {}", parts.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_names_every_lock() {
        assert_eq!(NAMES.len(), Lock::Submit as usize + 1);
        let m = Mutex::new(1);
        assert_eq!(*lock(&m, Lock::Slots), 1);
        let r = RwLock::new(2);
        assert_eq!(*read(&r, Lock::Views), 2);
        *write(&r, Lock::Views) = 3;
        assert_eq!(*read(&r, Lock::Views), 3);
    }
}
