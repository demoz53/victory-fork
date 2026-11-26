//! The submit audit (`VICTORY_SUBMIT_AUDIT=1`; off by default; acts with
//! `VICTORY_ASYNC_SUBMIT=1`, where a submission is prepared by a thread of
//! ours some time after the title's submit call).
//!
//! What a submission's preparing reads of title memory on the CPU (fills of
//! upload memory, the sources of the title's buffer copies, the ranges its
//! host copies are refreshed over, its textures' memory) is read when our
//! thread prepares it, not when the title called. `crate::queue::behind`
//! assumes the title does not rewrite that memory in between. This counts
//! where it does: every such range is fingerprinted on the title's thread
//! at its submit call ([`called`]) and again where the submission is
//! prepared, just before its steps run ([`prepared`]); a range whose
//! fingerprint changed is counted by what it is for ([`KINDS`]).
//!
//! Not every change is the title's. Between the call and the preparing our
//! own threads write title memory too, and a submission prepared on the
//! title's thread would have seen those writes as well (it waits for the
//! flights that write what it reads): what landing flights' shaders wrote,
//! read-backs, bottom-of-pipe values, and the title's buffer copies of
//! earlier submissions (done on the CPU when those are prepared). Those
//! writes are noted ([`wrote`]) by 256-byte block and 64 KiB page, and a
//! changed range one of them touched since the call is counted apart
//! ("written by us meanwhile"). A step of ours that may write anywhere
//! ([`wrote_anywhere`]) makes every change after it "not told apart".
//!
//! The fingerprint is of all the bytes up to 4 KiB; of longer ranges it is
//! of the first and last 256 bytes and fourteen 64-byte samples between
//! (their places differ from call to call, the same at a call and its
//! preparing): a change in a long range is found only when it is wide.
//! Constants and indirect arguments are short and are compared whole.
//!
//! It also counts the submit calls whose lists were prepared after those of
//! a later call: the lines are run queue by queue, so two queues' parts
//! that are both free to run need not be sent in the order the title
//! submitted them (prepared at the call, they are). Those that stood behind
//! a wait of the title's (their own, or one their queue's line held) are
//! counted apart: they are sent late with or without `behind`.
//!
//! Cost (measured by the line itself: "ms at the calls", "ms at the
//! preparings"): two fingerprints a range, each range once a submission.
//! `VICTORY_SUBMIT_AUDIT=<n>` (n > 1) audits one submit call in n.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use super::{HostList, Step, fill_kind};

/// What a range of title memory is read for.
pub(crate) const CONSTANTS: u8 = 0;
pub(crate) const VERTICES: u8 = 1;
pub(crate) const TEXTURES: u8 = 2;
pub(crate) const INDIRECT: u8 = 3;
pub(crate) const VIEWS: u8 = 4;
const COPIES: u8 = 5;
const OTHER: u8 = 6;
const KINDS: usize = 7;
const NAMES: [&str; KINDS] = [
    "constants (root and table constant views)",
    "vertex and index data",
    "texture sources (textures' memory and texture copies' sources)",
    "indirect arguments",
    "other buffers given to shaders (views, root descriptors, the heap's and the memory pass's copies)",
    "sources of the title's buffer copies",
    "other fills",
];

/// 0: off; n: one submit call in n is audited.
fn every() -> u64 {
    #[cfg(test)]
    if TEST_EVERY.load(Relaxed) > 0 {
        return TEST_EVERY.load(Relaxed);
    }
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let n = std::env::var("VICTORY_SUBMIT_AUDIT").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
        if n > 0 {
            tracing::info!(
                every = n,
                "submit audit: title memory a queued submission's preparing reads is fingerprinted at the title's submit \
                 call and again at its preparing (VICTORY_SUBMIT_AUDIT; acts with VICTORY_ASYNC_SUBMIT=1)"
            );
        }
        n
    })
}

/// Tests: [`every`], whatever the switch says (0: as the switch says). For
/// every thread: our writes are noted on the threads that make them.
#[cfg(test)]
pub(crate) static TEST_EVERY: AtomicU64 = AtomicU64::new(0);

/// Whether the audit is on.
pub(crate) fn on() -> bool {
    every() > 0
}

/// A range as it was at the title's submit call.
pub(super) struct Print {
    kind: u8,
    from: usize,
    len: usize,
    hash: u64,
}

/// A list's submit call.
#[derive(Clone, Copy)]
pub(super) struct Call {
    /// The call's number: the seed of the samples' places.
    number: u64,
    /// [`SEQ`] and [`ANYWHERE`] at the call.
    seq: u64,
    anywhere: u64,
    at: std::time::Instant,
    /// Whether the lists went behind a wait of the title's.
    waited: bool,
}

/// The highest number of a submit call whose lists were prepared so far.
static PREPARED_CALL: AtomicU64 = AtomicU64::new(0);
/// Since the last frame line: submit calls whose lists were prepared after
/// those of a later call, free to run all along; and those that stood
/// behind a wait.
static LATER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Bytes compared whole at most.
const FULL: usize = 4096;
/// Bytes at each end of a longer range, and the samples between.
const END: usize = 256;
const SAMPLES: usize = 14;
const SAMPLE: usize = 64;
/// The fingerprint of a range a page of which cannot be read.
const UNREADABLE: u64 = 0x756e_7265_6164_0001;

fn mix(h: u64, w: u64) -> u64 {
    (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn hash(mut h: u64, bytes: &[u8]) -> u64 {
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        h = mix(h, u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]));
    }
    let rest = words.remainder();
    if !rest.is_empty() {
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        h = mix(h, u64::from_le_bytes(last));
    }
    h
}

/// Title memory at `src` into `dst`; false if a page of it is not there.
fn read(src: usize, dst: &mut [u8]) -> bool {
    if crate::guarded::on() {
        // SAFETY: `dst` is ours and writable; a source page that is not
        // there stops the copy (the guarded copy's handler is installed).
        return unsafe { crate::guarded::copy(src as *const u8, dst.as_mut_ptr(), dst.len()) };
    }
    super::read_into(src as *const u8, dst)
}

/// The fingerprint of `[from, from + len)` as it is now (`seed`: where the
/// samples of a long range lie), and the bytes it read.
fn print(from: usize, len: usize, seed: u64) -> (u64, usize) {
    let mut bytes = [0u8; FULL];
    if len <= FULL {
        if !read(from, &mut bytes[..len]) {
            return (UNREADABLE, 0);
        }
        return (hash(len as u64, &bytes[..len]), len);
    }
    let mut h = len as u64;
    let mut take = |at: usize, n: usize| -> bool {
        if !read(at, &mut bytes[..n]) {
            return false;
        }
        h = hash(h, &bytes[..n]);
        true
    };
    if !take(from, END) || !take(from + len - END, END) {
        return (UNREADABLE, 0);
    }
    let stride = (len - 2 * END) / SAMPLES;
    let phase = (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 33) as usize % (stride - SAMPLE + 1);
    for i in 0..SAMPLES {
        if !take(from + END + i * stride + phase, SAMPLE) {
            return (UNREADABLE, 0);
        }
    }
    (h, 2 * END + SAMPLES * SAMPLE)
}

/// Our own writes of title memory so far ([`wrote`]).
static SEQ: AtomicU64 = AtomicU64::new(0);
/// Steps of ours that may have written title memory anywhere so far.
static ANYWHERE: AtomicU64 = AtomicU64::new(0);
/// Submit calls so far.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// A write of ours of this many bytes or fewer is noted by 256-byte block;
/// a longer one by 64 KiB page alone.
const SMALL: usize = 16 << 10;
const BLOCK: usize = 256;
const PAGE: usize = 64 << 10;

/// Where our threads wrote title memory, and at which [`SEQ`].
#[derive(Default)]
struct Wrote {
    /// By 256-byte block: the last small write.
    blocks: HashMap<usize, u64>,
    /// By 64 KiB page: the last write of any size, and the last long one.
    pages: HashMap<usize, (u64, u64)>,
    /// Notes of a [`SEQ`] below this were dropped ([`line`]).
    pruned: u64,
    /// [`SEQ`] at the frame line before the last.
    line_before: u64,
}

static WROTE: Mutex<Option<Wrote>> = Mutex::new(None);

/// A thread of ours wrote (or is about to write) `len` bytes of title
/// memory at `at`: what landing flights' shaders wrote, a read-back, a
/// bottom-of-pipe value, a buffer copy of the title's done at preparing.
pub(crate) fn wrote(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let seq = SEQ.fetch_add(1, Relaxed) + 1;
    let mut wrote = WROTE.lock().unwrap_or_else(|e| e.into_inner());
    let wrote = wrote.get_or_insert_with(Wrote::default);
    let end = at.saturating_add(len);
    let long = len > SMALL;
    for page in at / PAGE..=(end - 1) / PAGE {
        let p = wrote.pages.entry(page).or_insert((0, 0));
        p.0 = seq;
        if long {
            p.1 = seq;
        }
    }
    if !long {
        for block in at / BLOCK..=(end - 1) / BLOCK {
            wrote.blocks.insert(block, seq);
        }
    }
}

/// A step of ours ran that may write title memory anywhere.
pub(crate) fn wrote_anywhere() {
    if on() {
        ANYWHERE.fetch_add(1, Relaxed);
    }
}

/// Whether a write of ours touched `[from, from + len)` after `seq`; `None`
/// if that cannot be told any more (the notes of then are gone).
fn ours_since(from: usize, len: usize, seq: u64) -> Option<bool> {
    let wrote = WROTE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(wrote) = wrote.as_ref() else {
        return Some(false);
    };
    if seq < wrote.pruned {
        return None;
    }
    let end = from + len;
    let pages = from / PAGE..=(end - 1) / PAGE;
    // A long write of ours on one of its pages, or (a long range) any.
    let long = len > PAGE;
    if pages.clone().any(|page| wrote.pages.get(&page).is_some_and(|&(any, big)| big > seq || (long && any > seq))) {
        return Some(true);
    }
    if long {
        return Some(false);
    }
    Some((from / BLOCK..=(end - 1) / BLOCK).any(|block| wrote.blocks.get(&block).is_some_and(|&at| at > seq)))
}

/// Since the last frame line, per kind: ranges compared; ranges the title
/// changed between its call and the preparing; ranges changed that a write
/// of ours touched meanwhile; ranges changed after a step of ours that may
/// write anywhere (or too long ago to tell).
static COUNTS: [[AtomicU64; KINDS]; 4] = [const { [const { AtomicU64::new(0) }; KINDS] }; 4];
const COMPARED: usize = 0;
const CHANGED: usize = 1;
const BY_US: usize = 2;
const UNTOLD: usize = 3;
/// Since the last frame line: microseconds at the calls, at the
/// preparings; bytes fingerprinted; ranges with a page not there (at
/// either time); lists compared; the microseconds from their calls to
/// their preparings in all, and the longest; ranges not compared again
/// because the same range of the same call was (counted once).
static TOTALS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Changed ranges kept for the frame line: kind, address, bytes,
/// microseconds from the call to the preparing, frame.
static EXAMPLES: Mutex<Vec<(u8, usize, usize, u64, u64)>> = Mutex::new(Vec::new());
/// Examples kept per kind between two frame lines.
const EXAMPLES_A_KIND: usize = 3;

/// What `step` reads of title memory on the CPU when its list is prepared:
/// what for, where, how many bytes.
fn reads(
    step: &Step,
    fill_kinds: &crate::fasthash::Map<usize, u8>,
    refresh_kinds: &crate::fasthash::Map<(usize, usize), u8>,
) -> Option<(u8, usize, usize)> {
    match step {
        &Step::Copy(_, src, len) => Some((COPIES, src, len)),
        &Step::Fill(dst, src, len) => {
            let kind = match fill_kinds.get(&dst).copied().unwrap_or(fill_kind::OTHER) {
                fill_kind::ROOT_CBV | fill_kind::TABLE_CBV | fill_kind::RAW_CBV => CONSTANTS,
                fill_kind::TEXTURE_SOURCE => TEXTURES,
                _ => OTHER,
            };
            Some((kind, src, len))
        }
        // (Rectangle vertices, expanded on the CPU.)
        &Step::HostRead(src, len, _) => Some((VERTICES, src, len)),
        Step::Refresh(shadow, address, len) => {
            // (None: title memory in place, which nothing copies.)
            let (from, to) = shadow.refreshed(*address, *len)?;
            let copy = std::sync::Arc::as_ptr(shadow) as usize;
            Some((refresh_kinds.get(&(copy, *address)).copied().unwrap_or(VIEWS), from, to - from))
        }
        Step::Texture { staging, .. } => {
            let (at, len) = staging.memory();
            Some((TEXTURES, at, len))
        }
        Step::Run(_) | Step::Host(_) | Step::GpuWrite(..) => None,
    }
}

/// The title submitted `lists` and they went behind their queue's line
/// (`crate::queue::behind`): what their preparing will read of title
/// memory is fingerprinted now, on the title's thread. `waited`: whether
/// they stand behind a wait of the title's there.
pub(crate) fn called<'a>(lists: impl Iterator<Item = &'a HostList>, waited: bool) {
    let every = every();
    if every == 0 {
        return;
    }
    let number = CALLS.fetch_add(1, Relaxed) + 1;
    if !number.is_multiple_of(every) {
        return;
    }
    let started = std::time::Instant::now();
    let call = Call { number, seq: SEQ.load(Relaxed), anywhere: ANYWHERE.load(Relaxed), at: started, waited };
    // Each range once a call, however many lists and steps read it.
    let mut seen: HashMap<(usize, usize), u64> = HashMap::new();
    let mut hashed = 0;
    for list in lists {
        let mut deferred = list.deferred();
        let super::Deferred { copies, fill_kinds, refresh_kinds, audit, audit_call, .. } = &mut *deferred;
        audit.clear();
        for step in copies.iter() {
            let Some((kind, from, len)) = reads(step, fill_kinds, refresh_kinds) else {
                continue;
            };
            if len == 0 {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(place) = seen.entry((from, len)) {
                let (hash, bytes) = print(from, len, number);
                hashed += bytes;
                place.insert(hash);
                audit.push(Print { kind, from, len, hash });
            }
        }
        *audit_call = Some(call);
    }
    TOTALS[0].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    TOTALS[2].fetch_add(hashed as u64, Relaxed);
}

/// `runs` are about to be prepared (their steps run next, on this thread):
/// what their lists' submit calls fingerprinted is fingerprinted again and
/// compared.
pub(super) fn prepared(runs: &[&HostList]) {
    if !on() {
        return;
    }
    let started = std::time::Instant::now();
    let frame = crate::frame::current();
    let anywhere = ANYWHERE.load(Relaxed);
    let mut hashed = 0;
    // The call last looked at for the order.
    let mut ordered = 0;
    for list in runs {
        let (mut prints, call) = {
            let mut deferred = list.deferred();
            (std::mem::take(&mut deferred.audit), deferred.audit_call.take())
        };
        // (None: not submitted through a line, left out by `every`, or
        // named twice by its submission.)
        let Some(call) = call else {
            continue;
        };
        if call.number != ordered {
            ordered = call.number;
            if call.number < PREPARED_CALL.fetch_max(call.number, Relaxed) {
                LATER[usize::from(call.waited)].fetch_add(1, Relaxed);
            }
        }
        let late = started.saturating_duration_since(call.at).as_micros() as u64;
        TOTALS[4].fetch_add(1, Relaxed);
        TOTALS[5].fetch_add(late, Relaxed);
        TOTALS[6].fetch_max(late, Relaxed);
        for p in &prints {
            let kind = usize::from(p.kind);
            let (now, bytes) = print(p.from, p.len, call.number);
            hashed += bytes;
            COUNTS[COMPARED][kind].fetch_add(1, Relaxed);
            if now == UNREADABLE || p.hash == UNREADABLE {
                TOTALS[3].fetch_add(1, Relaxed);
            }
            if now == p.hash {
                continue;
            }
            let how = match ours_since(p.from, p.len, call.seq) {
                Some(true) => BY_US,
                Some(false) if anywhere == call.anywhere => CHANGED,
                _ => UNTOLD,
            };
            COUNTS[how][kind].fetch_add(1, Relaxed);
            if how == CHANGED {
                let mut examples = EXAMPLES.lock().unwrap_or_else(|e| e.into_inner());
                if examples.iter().filter(|e| e.0 == p.kind).count() < EXAMPLES_A_KIND {
                    examples.push((p.kind, p.from, p.len, late, frame));
                }
            }
        }
        // (The room is kept for the list's next recording.)
        prints.clear();
        list.deferred().audit = prints;
    }
    TOTALS[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    TOTALS[2].fetch_add(hashed as u64, Relaxed);
}

/// With the frame line.
pub(super) fn line(frames: f64) {
    if !on() {
        return;
    }
    let counts: Vec<[u64; KINDS]> = COUNTS.iter().map(|c| c.each_ref().map(|n| n.swap(0, Relaxed))).collect();
    let t: Vec<u64> = TOTALS.iter().map(|t| t.swap(0, Relaxed)).collect();
    let later = LATER.each_ref().map(|l| l.swap(0, Relaxed) as f64 / frames);
    let examples = std::mem::take(&mut *EXAMPLES.lock().unwrap_or_else(|e| e.into_inner()));
    // Notes of our writes older than the line before this one are of no
    // use: no list waits in a line that long (a present runs the lines).
    {
        let mut wrote = WROTE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(wrote) = wrote.as_mut() {
            let below = wrote.line_before;
            wrote.blocks.retain(|_, seq| *seq > below);
            wrote.pages.retain(|_, (any, _)| *any > below);
            wrote.pruned = below;
            wrote.line_before = SEQ.load(Relaxed);
        }
    }
    let compared: u64 = counts[COMPARED].iter().sum();
    if compared == 0 {
        return;
    }
    // The kinds, the most changed first.
    let by_kind = |how: usize| -> String {
        let mut kinds: Vec<(u64, usize)> = (0..KINDS).map(|k| (counts[how][k], k)).filter(|k| k.0 > 0).collect();
        kinds.sort_unstable_by(|a, b| b.cmp(a));
        if kinds.is_empty() {
            return "none".to_owned();
        }
        kinds
            .iter()
            .map(|&(n, k)| format!("{} {:.2} of {:.0}", NAMES[k], n as f64 / frames, counts[COMPARED][k] as f64 / frames))
            .collect::<Vec<_>>()
            .join("; ")
    };
    let lists = t[4].max(1) as f64;
    tracing::info!(
        "submit audit per frame (VICTORY_SUBMIT_AUDIT; title memory read at preparing, fingerprinted at the title's submit \
         call and again at the preparing; ranges over 4 KiB by samples): {:.0} ranges of {:.1} lists compared, {:.2} MB read, \
         {:.2} ms at the calls and {:.2} ms at the preparings; a list was prepared {:.2} ms after its call on average, the \
         longest {:.1} ms; {:.1} ranges had a page not there. CHANGED BY THE TITLE between its call and the preparing (of \
         those compared, by kind, the most first): {}. Changed, and written by us meanwhile (a landing, a read-back, a \
         bottom-of-pipe value, a buffer copy of the title's done at an earlier preparing; not a late capture): {}. Changed \
         after a step of ours that may write anywhere, not told apart: {}. Order: {:.1} submit calls' lists were prepared \
         after those of a LATER call although free to run all along (the lines are run queue by queue), {:.1} more that \
         stood behind a wait of the title's",
        compared as f64 / frames,
        t[4] as f64 / frames,
        t[2] as f64 / (1 << 20) as f64 / frames,
        t[0] as f64 / 1000.0 / frames,
        t[1] as f64 / 1000.0 / frames,
        t[5] as f64 / 1000.0 / lists,
        t[6] as f64 / 1000.0,
        t[3] as f64 / frames,
        by_kind(CHANGED),
        by_kind(BY_US),
        by_kind(UNTOLD),
        later[0],
        later[1],
    );
    for (kind, from, len, late, frame) in examples {
        tracing::info!(
            "submit audit: {} at {from:#x}+{len:#x} changed between the title's submit call and its preparing {:.2} ms later \
             (frame {frame}), by no write of ours",
            NAMES[usize::from(kind)],
            late as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short range is compared whole; a long one by its ends and its
    /// samples, whose places follow the seed and stay inside it.
    #[test]
    fn fingerprints_follow_the_bytes() {
        let mut memory = vec![7u8; 64 << 10];
        let at = memory.as_ptr() as usize;
        let (short, read) = print(at + 100, 300, 1);
        assert_eq!(read, 300);
        assert_eq!(print(at + 100, 300, 2).0, short, "a short range does not follow the seed");
        memory[399] ^= 1;
        assert_ne!(print(at + 100, 300, 1).0, short, "the last byte of a short range");
        memory[399] ^= 1;
        assert_eq!(print(at + 100, 300, 1).0, short);
        memory[400] ^= 1;
        assert_eq!(print(at + 100, 300, 1).0, short, "the byte after it");
        memory[400] ^= 1;

        let len = 48 << 10;
        let (long, read) = print(at, len, 1);
        assert_eq!(read, 2 * END + SAMPLES * SAMPLE);
        memory[len - 1] ^= 1;
        assert_ne!(print(at, len, 1).0, long, "the last byte of a long range");
        memory[len - 1] ^= 1;
        memory[0] ^= 1;
        assert_ne!(print(at, len, 1).0, long, "its first byte");
        memory[0] ^= 1;
        assert_eq!(print(at, len, 1).0, long);
        // Everything between the ends rewritten: found whatever the seed.
        for seed in 1..40 {
            let before = print(at, len, seed).0;
            memory[END..len - END].fill(9);
            assert_ne!(print(at, len, seed).0, before, "seed {seed}");
            memory[END..len - END].fill(7);
        }
        // The samples stay inside the shortest long range, and inside any.
        let guard = memory.len();
        for seed in 0..200 {
            assert_ne!(print(at + guard - (FULL + 1), FULL + 1, seed).0, UNREADABLE);
            assert_ne!(print(at + guard - 9999, 9999, seed).0, UNREADABLE);
        }
    }

    /// A range a page of which is not there has the fingerprint of one that
    /// cannot be read.
    #[cfg(windows)]
    #[test]
    fn a_range_not_there_is_unreadable() {
        assert_eq!(print(0x10, 64, 1), (UNREADABLE, 0));
        assert_eq!(print(0x10, 3 * FULL, 1), (UNREADABLE, 0));
    }

    /// Our own writes are told from the title's: by block for short
    /// writes, by page for long ones, and only those after the call.
    #[test]
    fn our_writes_since_a_call_are_told() {
        TEST_EVERY.store(1, Relaxed);
        let base = 0x7a00_0000_0000usize;
        let before = SEQ.load(Relaxed);
        assert_eq!(ours_since(base, 256, before), Some(false));
        wrote(base + 1024, 100);
        assert_eq!(ours_since(base + 1024, 256, before), Some(true), "the block written");
        assert_eq!(ours_since(base + 1100, 8, before), Some(true), "bytes of it");
        assert_eq!(ours_since(base, 512, before), Some(false), "another block of the page");
        assert_eq!(ours_since(base, 4 * PAGE, before), Some(true), "a long range over the page");
        let after = SEQ.load(Relaxed);
        assert_eq!(ours_since(base + 1024, 256, after), Some(false), "written before the call");
        wrote(base + PAGE, 2 * PAGE);
        assert_eq!(ours_since(base + PAGE + 77, 16, after), Some(true), "under a long write");
        assert_eq!(ours_since(base + 3 * PAGE, 16, after), Some(false), "past its end");
    }

    /// What a list's steps read of title memory is compared with what it
    /// was at the title's submit call: a range the title rewrote is counted
    /// by its kind, one a write of ours touched meanwhile is counted apart,
    /// one left alone is not counted; and each call is compared once.
    #[cfg(windows)]
    #[test]
    fn a_range_rewritten_after_the_call_is_counted_by_kind_and_writer() {
        let Some(host) = super::super::host() else {
            return; // no host GPU here
        };
        TEST_EVERY.store(1, Relaxed);
        let memory: &'static mut [u8] = Box::leak(vec![3u8; 8192].into_boxed_slice());
        let at = memory.as_ptr() as usize;
        let to: &'static mut [u8] = Box::leak(vec![0u8; 4096].into_boxed_slice());
        let to = to.as_mut_ptr();
        let list = host.new_list().unwrap();
        list.defer_fill_as(to, at as *const u8, 256, fill_kind::ROOT_CBV);
        list.defer_fill_as(to.wrapping_add(256), (at + 1024) as *const u8, 256, fill_kind::TABLE_CBV);
        list.defer_fill_as(to.wrapping_add(512), (at + 2048) as *const u8, 256, fill_kind::TEXTURE_SOURCE);
        list.defer_copy(to.wrapping_add(1024), (at + 4096) as *const u8, 64);
        list.close();
        let count = |how: usize, kind: u8| COUNTS[how][usize::from(kind)].load(Relaxed);
        let before = [count(CHANGED, CONSTANTS), count(BY_US, CONSTANTS), count(COMPARED, CONSTANTS), count(COMPARED, TEXTURES)];
        let quiet = [count(CHANGED, TEXTURES), count(CHANGED, COPIES), count(BY_US, TEXTURES), count(BY_US, COPIES)];
        called(std::iter::once(&list), false);
        // The title rewrites a byte of the first view's constants; a landing
        // of ours puts four bytes into the second's.
        memory[5] ^= 0xff;
        wrote(at + 1024 + 8, 4);
        memory[1024 + 8] ^= 0xff;
        prepared(&[&list]);
        assert_eq!(count(CHANGED, CONSTANTS) - before[0], 1, "the title's rewrite");
        assert_eq!(count(BY_US, CONSTANTS) - before[1], 1, "our own write");
        assert_eq!(count(COMPARED, CONSTANTS) - before[2], 2);
        assert_eq!(count(COMPARED, TEXTURES) - before[3], 1);
        assert_eq!(
            [count(CHANGED, TEXTURES), count(CHANGED, COPIES), count(BY_US, TEXTURES), count(BY_US, COPIES)],
            quiet,
            "what nobody wrote"
        );
        assert!(EXAMPLES.lock().unwrap().iter().any(|e| e.1 == at && e.2 == 256), "the rewritten range is kept as an example");
        // The call was compared: preparing the list again compares nothing.
        memory[2048] ^= 0xff;
        prepared(&[&list]);
        assert_eq!(count(COMPARED, TEXTURES) - before[3], 1);
        // Two calls prepared in the other order: the first one's lists come
        // after a later call's, and are counted so (free to run all along,
        // or behind a wait of the title's).
        for waited in [false, true] {
            let (first, second) = (host.new_list().unwrap(), host.new_list().unwrap());
            first.close();
            second.close();
            let later = [LATER[0].load(Relaxed), LATER[1].load(Relaxed)];
            called(std::iter::once(&first), waited);
            called(std::iter::once(&second), false);
            prepared(&[&second]);
            prepared(&[&first]);
            let now = [LATER[0].load(Relaxed) - later[0], LATER[1].load(Relaxed) - later[1]];
            assert_eq!(now, if waited { [0, 1] } else { [1, 0] });
        }
    }
}
