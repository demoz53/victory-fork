//! Waits of the card and of the submitting thread, per frame
//! (`VICTORY_WAIT_STATS=1`; off by default, always compiled).
//!
//! With the switch the 5 s frame line is followed by one line that says,
//! per frame: how many barriers on every shader write (global unordered
//! access barriers) our lists recorded and why (the title's barrier
//! groups, its own barriers, the indirect clamps before and after their
//! dispatch, the batched clamp passes, the title's pipeline syncs, passes
//! of ours), how many unordered access barriers named one resource
//! instead, what the title's barrier groups held, how often the flights
//! were settled and why, and how long the submitting thread stood in each
//! kind of wait (settles by why, texture staging, a synchronous
//! submission, the submission lock). Without the switch each count is one
//! load of a flag set once.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Why a barrier on every shader write was recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cause {
    /// The title's enhanced barrier groups (`list::barrier_groups`).
    Groups,
    /// The title's own (legacy) barriers: on a buffer, or on every write.
    Barriers,
    /// Before an indirect clamp's dispatch (the title's argument passes).
    ClampBefore,
    /// After it (the clamp's own writes).
    ClampAfter,
    /// Before a batched clamp pass (`VICTORY_INDIRECT_BATCH`).
    Batch,
    /// A pipeline sync of the title's (`VICTORY_SYNC_BARRIER=1`).
    Sync,
    /// A root constant view copied on the card (`HostList::card_constants`
    /// from `draw.rs` `root_view`).
    ConstantsRoot,
    /// A table constant view copied on the card (`card_constants` from a
    /// table's views).
    ConstantsTable,
    /// Passes of ours: the launch wait, card compares and fills, the
    /// stand-in light.
    Ours,
}

const CAUSES: usize = 9;
const CAUSE_NAMES: [&str; CAUSES] = [
    "title groups",
    "title barriers",
    "clamps before",
    "clamps after",
    "batched clamps",
    "syncs",
    "root constants copied on the card",
    "table constants copied on the card",
    "ours",
];

/// Per cause: global barriers, and barriers on one resource instead.
static GLOBAL: [AtomicU64; CAUSES] = [const { AtomicU64::new(0) }; CAUSES];
static SCOPED: [AtomicU64; CAUSES] = [const { AtomicU64::new(0) }; CAUSES];
/// Per cause: such waits left out, nothing having stored since the list's
/// last one (`VICTORY_CONSTANT_WAITS=lazy`).
static LEFT_OUT: [AtomicU64; CAUSES] = [const { AtomicU64::new(0) }; CAUSES];

/// The title's barrier groups: calls, groups of each kind (global,
/// texture, buffer, other), barriers in them, barriers whose accesses ask
/// for an order of shader writes, calls that asked none, calls that fell
/// back to the full wait under `VICTORY_BARRIER_GROUPS`.
static GROUPS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Settles by why (the host's `WHY_` places as `waited_at` gives them,
/// 2..=11), their count and microseconds; and the settles for title
/// memory read by what read (a buffer copy, a fill of upload memory, a
/// texture, another copy).
static SETTLES: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static SETTLE_US: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static READS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Other waits of the submitting thread: (count, microseconds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stood {
    /// For a flight still copying from texture staging.
    Staging,
    /// For a synchronous submission to run.
    Synchronous,
    /// For the submission lock (another thread submitting or landing).
    SubmitLock,
}

const STOOD: usize = 3;
static STOOD_N: [AtomicU64; STOOD] = [const { AtomicU64::new(0) }; STOOD];
static STOOD_US: [AtomicU64; STOOD] = [const { AtomicU64::new(0) }; STOOD];

/// Whether the waits are counted (`VICTORY_WAIT_STATS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_WAIT_STATS").as_deref() == Ok("1"))
}

/// `n` barriers on every shader write recorded for `cause`.
pub(crate) fn global(cause: Cause, n: u64) {
    if on() {
        GLOBAL[cause as usize].fetch_add(n, Relaxed);
    }
}

/// `n` unordered access barriers on one resource recorded for `cause`.
pub(crate) fn scoped(cause: Cause, n: u64) {
    if on() {
        SCOPED[cause as usize].fetch_add(n, Relaxed);
    }
}

/// `n` waits for every shader write left out for `cause`.
pub(crate) fn left_out(cause: Cause, n: u64) {
    if on() {
        LEFT_OUT[cause as usize].fetch_add(n, Relaxed);
    }
}

/// One call of the title's barrier groups: groups by kind (global,
/// texture, buffer, other), the barriers in them, those that ask for an
/// order of shader writes.
pub(crate) fn groups(kinds: [u64; 4], barriers: u64, ordering: u64) {
    if !on() {
        return;
    }
    GROUPS[0].fetch_add(1, Relaxed);
    for (i, n) in kinds.into_iter().enumerate() {
        GROUPS[1 + i].fetch_add(n, Relaxed);
    }
    GROUPS[5].fetch_add(barriers, Relaxed);
    GROUPS[6].fetch_add(ordering, Relaxed);
    if ordering == 0 {
        GROUPS[7].fetch_add(1, Relaxed);
    }
}

/// A call of the title's barrier groups that `VICTORY_BARRIER_GROUPS`
/// could not give barriers on its textures: it got the full wait.
pub(crate) fn group_fell_back() {
    if on() {
        GROUPS[8].fetch_add(1, Relaxed);
    }
}

/// A settle for `at` (the host's `waited_at` place, 2..=11) that stood
/// `us` microseconds (0 for one that found nothing to wait for).
pub(crate) fn settled(at: usize, us: u64) {
    if on() && at < SETTLES.len() {
        SETTLES[at].fetch_add(1, Relaxed);
        SETTLE_US[at].fetch_add(us, Relaxed);
    }
}

/// The time a settle counted by [`settled`] went on standing (a settle
/// that lands several rounds of flights).
pub(crate) fn settle_stood(at: usize, us: u64) {
    if on() && at < SETTLE_US.len() {
        SETTLE_US[at].fetch_add(us, Relaxed);
    }
}

/// A settle for title memory read, by what read it (the host's
/// `LATE_STATS` places 15-18).
pub(crate) fn read_by(slot: usize) {
    if on() && (15..=18).contains(&slot) {
        READS[slot - 15].fetch_add(1, Relaxed);
    }
}

/// The submitting thread stood `us` microseconds for `what`.
pub(crate) fn stood(what: Stood, us: u64) {
    if on() {
        STOOD_N[what as usize].fetch_add(1, Relaxed);
        STOOD_US[what as usize].fetch_add(us, Relaxed);
    }
}

/// The settles' places (`waited_at`) named in the line, from 2 on.
const SETTLE_NAMES: [&str; 10] = [
    "a synchronous submission or present",
    "the card had run them",
    "a memory value",
    "title memory changed under shader writes",
    "title memory read",
    "a step of ours",
    "lists in the air",
    "bottom-of-pipe writes",
    "a blocking fence wait",
    "the next submission",
];

/// The frame line's part over `frames` frames (the counts since the last
/// call, taken), or `None` without the switch.
pub(crate) fn line(frames: f64) -> Option<String> {
    if !on() {
        return None;
    }
    let f = frames.max(1.0);
    let per = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / f;
    let global: Vec<f64> = GLOBAL.iter().map(per).collect();
    let scoped: Vec<f64> = SCOPED.iter().map(per).collect();
    let left_out: Vec<f64> = LEFT_OUT.iter().map(per).collect();
    let causes = |v: &[f64]| -> String {
        v.iter().zip(CAUSE_NAMES).map(|(n, name)| format!("{name} {n:.0}")).collect::<Vec<_>>().join(", ")
    };
    let g: Vec<f64> = GROUPS.iter().map(per).collect();
    let settles: Vec<f64> = SETTLES.iter().map(per).collect();
    let settle_ms: Vec<f64> = SETTLE_US.iter().map(|a| per(a) / 1000.0).collect();
    let reads: Vec<f64> = READS.iter().map(per).collect();
    let by_why: Vec<String> = SETTLE_NAMES
        .iter()
        .enumerate()
        .filter(|&(i, _)| settles[i + 2] > 0.0 || settle_ms[i + 2] > 0.0)
        .map(|(i, name)| format!("{name} {:.1} ({:.1} ms)", settles[i + 2], settle_ms[i + 2]))
        .collect();
    let stood_n: Vec<f64> = STOOD_N.iter().map(per).collect();
    let stood_ms: Vec<f64> = STOOD_US.iter().map(|a| per(a) / 1000.0).collect();
    Some(format!(
        "waits per frame (VICTORY_WAIT_STATS=1): barriers on every shader write {:.0} ({}); on one resource instead {:.0} ({}); left out, nothing having stored since \
         the list's last one (VICTORY_CONSTANT_WAITS=lazy) {:.0} ({}); \
         the title's barrier groups: {:.0} calls, {:.0} global / {:.0} texture / {:.0} buffer / {:.0} other groups, {:.0} barriers, \
         {:.0} asking an order of shader writes, {:.0} calls asking none, {:.0} calls that fell back to the full wait under VICTORY_BARRIER_GROUPS (a global or buffer barrier, or a texture with no \
         host copy made for unordered access);settles {:.1}, {:.1} ms standing (by why: {}; title memory \
         read by a buffer copy {:.1}, a fill of upload memory {:.1}, a texture {:.1}, another copy {:.1}); the submitting thread also \
         stood: texture staging {:.1} ({:.1} ms), synchronous submissions {:.1} ({:.1} ms); waits for the submission lock (any thread) {:.1} ({:.1} ms)",
        global.iter().sum::<f64>(),
        causes(&global),
        scoped.iter().sum::<f64>(),
        causes(&scoped),
        left_out.iter().sum::<f64>(),
        causes(&left_out),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        g[5],
        g[6],
        g[7],
        g[8],
        settles.iter().sum::<f64>(),
        settle_ms.iter().sum::<f64>(),
        if by_why.is_empty() { "none".to_owned() } else { by_why.join(", ") },
        reads[0],
        reads[1],
        reads[2],
        reads[3],
        stood_n[Stood::Staging as usize],
        stood_ms[Stood::Staging as usize],
        stood_n[Stood::Synchronous as usize],
        stood_ms[Stood::Synchronous as usize],
        stood_n[Stood::SubmitLock as usize],
        stood_ms[Stood::SubmitLock as usize],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_counts() {
        assert_eq!(CAUSE_NAMES.len(), Cause::Ours as usize + 1);
        assert_eq!(STOOD, Stood::SubmitLock as usize + 1);
        assert_eq!(SETTLE_NAMES.len() + 2, SETTLES.len());
    }
}
