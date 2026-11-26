//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Changes of guarded textures the hash found and the guard did not see.
    TexHot,
    /// Frames presented since the last line (what the line divides by).
    Frames,
    /// Calls of list methods that record into a host list, and the
    /// nanoseconds they took, summed over the title's recording threads.
    RecordCalls,
    RecordNs,
    /// Waiting for host pipelines being made (`pipeline::make`).
    PipelineWaitUs,
    /// Dynamic descriptor slots: tables asked for, refused (no free run),
    /// and the nanoseconds of the searches.
    DynamicAsked,
    DynamicRefused,
    DynamicNs,
    /// Dynamic slots taken by the tables that got them.
    DynamicTaken,
    /// Descriptor tables draws and dispatches read, those of them read
    /// from the title's own slots (no dynamic slots), and those made
    /// afresh that the list had made already for an earlier draw.
    TablesRead,
    TablesInPlace,
    TablesReused,
    /// Table constant buffers read in place and copied for the draw
    /// (`VICTORY_TABLE_CONSTANTS_IN_PLACE`).
    TableConstantsInPlace,
    TableConstantsCopied,
    /// Marker texts for device-removed data (`HostList::note`).
    NoteNs,
    /// Ranges given to shaders (steps), their snapshots, and the bytes
    /// compared after the submission.
    GpuWrites,
    SnapshotUs,
    BackCompared,
    /// Reading around unmapped pages after a failed bulk read.
    SparseUs,
    /// Guarding view memory (part of the write-watch askings' time).
    GuardUs,
    /// Bytes copied whole because their memory is neither write-watched
    /// nor a view (`Shadow::update_free`).
    WholeBytes,
    /// Submissions with ranges shaders write (they need the GPU's results
    /// before they return).
    BackSubmissions,
    /// The title's frame waits (`frame::wait`) and the time slept in them,
    /// over all its threads.
    PaceWaits,
    PaceUs,
    /// After a present: until the title's first list call and until its
    /// first submission; and from its last submission to its next present.
    UntilRecordUs,
    UntilSubmitUs,
    TailUs,
    /// Resetting host lists (`HostList::reset`).
    ResetNs,
    /// What shaders wrote: finding the 64 KiB pieces that differ, how many
    /// do, and putting them into title memory.
    BackScanUs,
    BackDiffer,
    BackSyncUs,
    /// Texture uploads recorded, and left out as unchanged.
    TexUploads,
    TexSkipped,
    /// Held submissions a present ran although their waits were not met.
    Forced,
    /// Ranges named through unordered access views by shaders that cannot
    /// store to buffers, not given to shaders as written, and their bytes.
    GpuWritesLeftOut,
    GpuWritesLeftOutBytes,
    /// Waits for memory values looked at (`list::ValueWait::met`) and the
    /// nanoseconds the looks took, on every thread.
    WaitChecks,
    WaitCheckNs,
    /// Host lists made (`Host::new_list`: for the lists a queue holds and
    /// for the parts of lists cut at waits) and the nanoseconds that took.
    HostLists,
    HostListNs,
    /// Parts of held submissions that ran while the part behind them could
    /// have run with them (its waits were met already).
    PartsTogether,
    /// Microseconds of the copies counted in `WholeBytes`.
    WholeUs,
    /// Submissions by how long the card was waited for (under 2 ms, 2-8,
    /// 8-32, longer), and the microseconds waited in each class.
    CardUnder2,
    Card2To8,
    Card8To32,
    CardOver32,
    CardUnder2Us,
    Card2To8Us,
    Card8To32Us,
    CardOver32Us,
    /// Submissions that hand nothing back to the title (no range shaders
    /// write, no value written when done), and the microseconds the card
    /// was waited for them.
    NothingBack,
    NothingBackUs,
    /// Waits for every earlier shader write recorded into host lists: for
    /// the title's barrier groups, for its barriers on buffers, and the
    /// indirect commands bounded (each a dispatch of ours between two such
    /// waits).
    WaitsForGroups,
    WaitsForBuffers,
    IndirectBounded,
    /// Of `ResetNs`: letting go of what the lists held (their steps, upload
    /// buffers and objects), where the reset was called.
    ResetDropNs,
    /// Other views of guarded pages made read-only with them, and guarded
    /// pages shown at too many addresses for that (`watch::guard_aliases`).
    GuardAliasProtects,
    GuardAliasMany,
    /// With `watch::pieces`: bytes of stale pages a refresh left out as
    /// not written (their other kernel pages), and the stale pages copied
    /// by their written kernel pages and copied whole.
    PieceBytesLeftOut,
    PiecePages,
    PiecePagesWhole,
    /// Of the guards' checks (`watch::guard`): questions to Windows whether
    /// a range is view memory and their microseconds, calls that make
    /// blocks read-only and their microseconds, and microseconds waiting
    /// for the guards' lock.
    GuardQueries,
    GuardQueryUs,
    GuardProtects,
    GuardProtectUs,
    GuardLockUs,
    /// Questions to Windows whether blocks noted unmapped still are
    /// (`watch::uncommitted_end`), and their microseconds.
    GuardAbsentQueries,
    GuardAbsentUs,
}
const COUNTERS: usize = 96;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// Per range copied whole because its memory is not watched (start,
/// length): copies and microseconds.
static WHOLE: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src`, memory that is not watched, were copied whole in
/// `us`.
pub(crate) fn whole(src: usize, len: usize, us: u64) {
    add(C::WholeUs, us);
    let mut map = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += us;
}

/// The longest wait for the card since the last line: microseconds, and
/// what ran (lists and pipelines).
static LONGEST: Mutex<(u64, String)> = Mutex::new((0, String::new()));

/// The waits for the card since the last line by what ran (lists and
/// pipelines): how many, and their microseconds.
static WAITS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

/// With `VICTORY_GPU_TIMING`: the commands' time on the card since the last
/// line by what they ran (a pipeline, or the kind of command): how many,
/// their microseconds, and the longest one's.
static COMMANDS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64, u64)>>> = Mutex::new(None);

/// A command took `ms` on the card (timing run). `op` is its note: the
/// kind and size first, then the pipeline in brackets, then what it binds.
pub(crate) fn command_took(op: &str, ms: f64) {
    let us = (ms * 1e3) as u64;
    let mut by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let e = by.get_or_insert_with(HashMap::new).entry(command_key(op)).or_insert((0, 0, 0));
    e.0 += 1;
    e.1 += us;
    e.2 = e.2.max(us);
}

/// What a command's note is summed under: its pipeline (`[compute ...]`,
/// `[mesh ..., pixel ...]`) with the command's first word, or without a
/// pipeline its first two words.
fn command_key(op: &str) -> String {
    let head = op.split(" | ").next().unwrap_or(op);
    let word = head.split(' ').next().unwrap_or("");
    match (head.find('['), head.find(']')) {
        (Some(a), Some(b)) if a < b => format!("{word} {}", &head[a..=b]),
        _ => head.split(' ').take(2).collect::<Vec<_>>().join(" "),
    }
}

/// The card was waited for `us` for a submission; `what` names it if it is
/// the longest since the last line.
pub(crate) fn card_waited(us: u64, what: impl FnOnce() -> String) {
    let (n, t) = match us {
        0..2_000 => (C::CardUnder2, C::CardUnder2Us),
        2_000..8_000 => (C::Card2To8, C::Card2To8Us),
        8_000..32_000 => (C::Card8To32, C::Card8To32Us),
        _ => (C::CardOver32, C::CardOver32Us),
    };
    add(n, 1);
    add(t, us);
    // Every submission is named (some forty a frame): the same lists run
    // every frame, and the eight longest waits of a line were one list
    // eight times.
    let text = what();
    {
        let mut by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let e = by.get_or_insert_with(HashMap::new).entry(text.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
    let mut longest = LONGEST.lock().unwrap_or_else(|e| e.into_inner());
    if us > longest.0 {
        *longest = (us, text);
    }
}

/// Whether an experiment on barriers is on (`VICTORY_BARRIER_TEST`): bit 0
/// for `groups` (the title's barrier groups record no wait for earlier
/// shader writes), bit 1 for `indirect` (the pass that bounds an indirect
/// command runs without its two such waits; its state changes stay), both
/// bits for `both`. For timing only: without the waits a shader may read
/// what another has not finished writing.
pub(crate) fn barrier_test() -> u8 {
    static TEST: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *TEST.get_or_init(|| match std::env::var("VICTORY_BARRIER_TEST").as_deref() {
        Ok("groups") => 1,
        Ok("indirect") => 2,
        Ok("both") => 3,
        _ => 0,
    })
}

/// Per range compared after a submission (start, length): times, and the
/// bytes of it written back to the title.
static COMPARED: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src` were compared after a submission and `written` of
/// them put back into title memory.
pub(crate) fn compared(src: usize, len: usize, written: usize) {
    let mut map = COMPARED.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += written as u64;
}

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

/// Our own allocations alive now, by size of block (under 4 KiB, to 64 KiB,
/// to 1 MiB, larger): bytes, then blocks. "Our own" is the Rust code of
/// every crate in the process: in diagnostics builds [`Counting`] stands in
/// front of the system's heap. The memory map's "process heap segments"
/// hold the blocks under about half a megabyte of ours, of the graphics
/// runtime and of the driver alike (3.5 GB in run 60's gameplay, growing by
/// some 2 MB a frame, and nothing said whose): this says which part is
/// ours. Bigger blocks are allocations of their own in the map, and heaps
/// other than the process heap (AMD's compiler keeps one) count apart, as
/// "segments of other heaps".
///
/// Striped: each thread adds to one of [`STRIPES`] copies of the eight
/// counters, each copy on cache lines of its own, and a reading sums them
/// (wrapping: a block freed on another thread than made it takes from
/// another stripe). With one shared copy, the 12 pipeline workers spent
/// 60-75% of their time on these additions while a cutscene's shaders were
/// translated (VICTORY_PROFILE, play-5, 2026-11-25): every allocation and
/// release of every thread went through the same cache line.
///
/// The first [`ALONE`] threads each have a stripe of their own, which only
/// that thread writes: a plain load and store per counter, no locked
/// addition (the game runs some 130 threads; 32 shared stripes had four
/// threads to a stripe). Later threads share the last [`SHARED`] stripes
/// with atomic additions, as all did before.
static OWN: [Stripe; STRIPES] = [const { Stripe([const { AtomicU64::new(0) }; 8]) }; STRIPES];

/// One thread's (or thread group's) copy of the counters, alone on its
/// cache lines.
#[repr(align(128))]
struct Stripe([AtomicU64; 8]);
const ALONE: usize = 256;
const SHARED: usize = 16;
const STRIPES: usize = ALONE + SHARED;

/// This thread's stripe, and whether it is the thread's alone: handed out
/// in turn on a thread's first count. The thread-local has a constant
/// initialiser and no destructor, so reading it inside the allocator
/// neither allocates nor runs after the thread's end.
fn stripe() -> (&'static [AtomicU64; 8], bool) {
    use std::cell::Cell;
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local! { static MINE: Cell<usize> = const { Cell::new(usize::MAX) }; }
    let i = MINE.with(|m| {
        if m.get() == usize::MAX {
            let n = NEXT.fetch_add(1, Relaxed);
            m.set(if n < ALONE { n } else { ALONE + n % SHARED });
        }
        m.get()
    });
    (&OWN[i].0, i < ALONE)
}

/// Whether allocations are counted (`VICTORY_ALLOC_COUNT=0`: not; counted
/// by default, as before): 0 not known yet, 1 counted, 2 not. Read from the
/// environment on the first allocation with `GetEnvironmentVariableW` into
/// a buffer on the stack (that allocates nothing on our heap, so it can be
/// asked from inside the allocator), so that it holds from the process's
/// first block on: a block counted when made is counted when released.
///
/// Without the counts the memory line's "our own allocations" reads 0 and
/// "our own blocks of 32 MB and more" none; every allocation and release
/// then costs one load of this flag beside the system's heap.
static COUNTING: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[inline]
fn counting() -> bool {
    match COUNTING.load(Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = !alloc_count_off();
            COUNTING.store(if on { 1 } else { 2 }, Relaxed);
            on
        }
    }
}

/// `VICTORY_ALLOC_COUNT` is `0`, asked without our heap.
fn alloc_count_off() -> bool {
    let mut name = [0u16; 20];
    for (wide, &byte) in name.iter_mut().zip(b"VICTORY_ALLOC_COUNT") {
        *wide = u16::from(byte);
    }
    let mut value = [0u16; 4];
    // SAFETY: a terminated name and a buffer of the length given.
    let n = unsafe {
        windows_sys::Win32::System::Environment::GetEnvironmentVariableW(
            name.as_ptr(),
            value.as_mut_ptr(),
            value.len() as u32,
        )
    };
    n == 1 && value[0] == u16::from(b'0')
}

/// The counter `i` of [`OWN`] over all stripes.
fn own_total(i: usize) -> u64 {
    OWN.iter().fold(0u64, |sum, s| sum.wrapping_add(s.0[i].load(Relaxed)))
}

/// The class of [`OWN`] a block of `size` bytes counts in.
fn block_class(size: usize) -> usize {
    match size {
        0..0x1000 => 0,
        0x1000..0x1_0000 => 1,
        0x1_0000..0x10_0000 => 2,
        _ => 3,
    }
}

fn own(size: usize, made: bool) {
    let class = block_class(size);
    let (own, alone) = stripe();
    let (bytes, blocks) = if made { (size as u64, 1) } else { ((size as u64).wrapping_neg(), u64::MAX) };
    if alone {
        // Only this thread writes its stripe; a reader meanwhile reads a
        // whole word either way.
        own[class].store(own[class].load(Relaxed).wrapping_add(bytes), Relaxed);
        own[4 + class].store(own[4 + class].load(Relaxed).wrapping_add(blocks), Relaxed);
    } else {
        own[class].fetch_add(bytes, Relaxed);
        own[4 + class].fetch_add(blocks, Relaxed);
    }
}

/// Our own blocks of [`BIG`] bytes and more alive now: where each is (0:
/// none) and its bytes. The memory map lists the process's big allocations
/// by size without saying whose they are (run 65: one of 544 MB, one of
/// 401 MB, two of 174 MB, two of 115 MB); the sizes here say which of them
/// are ours.
static BIG_AT: [std::sync::atomic::AtomicUsize; 64] = [const { std::sync::atomic::AtomicUsize::new(0) }; 64];
static BIG_BYTES: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
const BIG: usize = 32 << 20;

/// A block of [`BIG`] bytes or more was made at `at`, or released there.
fn big(at: *mut u8, size: usize, made: bool) {
    if size < BIG {
        return;
    }
    for (place, bytes) in BIG_AT.iter().zip(&BIG_BYTES) {
        if made {
            if place.compare_exchange(0, at as usize, Relaxed, Relaxed).is_ok() {
                bytes.store(size as u64, Relaxed);
                return;
            }
        } else if place.load(Relaxed) == at as usize {
            bytes.store(0, Relaxed);
            place.store(0, Relaxed);
            return;
        }
    }
}

/// The sizes of our own big blocks alive, largest first, in MB.
fn big_blocks() -> String {
    let mut sizes: Vec<u64> = BIG_BYTES.iter().map(|b| b.load(Relaxed)).filter(|&b| b != 0).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let text: Vec<String> = sizes.iter().map(|b| format!("{:.2}", *b as f64 / (1 << 20) as f64)).collect();
    if text.is_empty() { "none".into() } else { text.join(", ") }
}

/// The system's allocator with every block counted in [`OWN`] (two
/// additions per allocation and per release: diagnostics builds only;
/// `VICTORY_ALLOC_COUNT=0` counts nothing, [`COUNTING`]).
struct Counting;

// SAFETY: every call goes to the system's allocator unchanged; only
// counters are kept beside it.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { std::alloc::System.alloc(layout) };
        if !at.is_null() && counting() {
            own(layout.size(), true);
            big(at, layout.size(), true);
        }
        at
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !at.is_null() && counting() {
            own(layout.size(), true);
            big(at, layout.size(), true);
        }
        at
    }

    unsafe fn dealloc(&self, at: *mut u8, layout: std::alloc::Layout) {
        if counting() {
            own(layout.size(), false);
            big(at, layout.size(), false);
        }
        // SAFETY: the caller's contract, passed on.
        unsafe { std::alloc::System.dealloc(at, layout) };
    }

    unsafe fn realloc(&self, at: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let moved = unsafe { std::alloc::System.realloc(at, layout, size) };
        if !moved.is_null() && counting() {
            own(layout.size(), false);
            own(size, true);
            big(at, layout.size(), false);
            big(moved, size, true);
        }
        moved
    }
}

#[global_allocator]
static HEAP: Counting = Counting;

#[cfg(test)]
mod command_tests {
    /// Commands are summed by pipeline and first word; those without a
    /// pipeline by their first two words.
    #[test]
    fn commands_are_named_by_their_pipeline() {
        use super::command_key;
        assert_eq!(
            command_key("dispatch 182x1x1 [compute 24f989b98dcb03f0] | 24f989b98dcb03f0: u4=0xa6b080 [heap 341380: x]"),
            "dispatch [compute 24f989b98dcb03f0]"
        );
        assert_eq!(
            command_key("indirect kind 10 max 1 at 0xd9358fce0 [mesh 0750cb2d975c7563, pixel 085d374c5a795549] | t"),
            "indirect [mesh 0750cb2d975c7563, pixel 085d374c5a795549]"
        );
        assert_eq!(command_key("texture upload 0x271b61c28b0 (2048x2048x1 f99 mips 10) sub 0"), "texture upload");
        assert_eq!(command_key("alias copy of 0x26f66aca780 into 0x26fbdc93890"), "alias copy");
    }
}

#[cfg(test)]
mod heap_tests {
    use super::segment_heap;
    use windows_sys::Win32::System::Memory::{
        GetProcessHeap, HeapAlloc, HeapCreate, HeapDestroy, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };

    /// A segment names its heap: the process heap's first segment is the
    /// heap itself, and a private heap filled past its first segment names
    /// that heap (AMD's compiler grew a heap of its own past 30 GB, which
    /// the memory map counted as the process heap's).
    #[test]
    fn segments_name_their_heap() {
        // SAFETY: a heap of our own, destroyed at the end; the query only
        // fills `m`.
        unsafe {
            let process = GetProcessHeap() as usize;
            assert_eq!(segment_heap(process), Some(process));
            let heap = HeapCreate(0, 0, 0);
            assert!(!heap.is_null());
            let mut last = std::ptr::null_mut();
            for _ in 0..20_000 {
                last = HeapAlloc(heap, 0, 4000);
                assert!(!last.is_null());
            }
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(
                VirtualQuery(last, &mut m, size_of::<MEMORY_BASIC_INFORMATION>()),
                0
            );
            let base = m.AllocationBase as usize;
            assert_ne!(base, heap as usize, "the heap added no segment");
            assert_eq!(segment_heap(base), Some(heap as usize));
            assert_eq!(segment_heap(last as usize), None);
            HeapDestroy(heap);
        }
    }
}

#[cfg(test)]
mod own_tests {
    use super::{block_class, own_total};
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn blocks_count_in_their_size_class() {
        assert_eq!(block_class(0), 0);
        assert_eq!(block_class(4095), 0);
        assert_eq!(block_class(4096), 1);
        assert_eq!(block_class(65535), 1);
        assert_eq!(block_class(65536), 2);
        assert_eq!(block_class((1 << 20) - 1), 2);
        assert_eq!(block_class(1 << 20), 3);
    }

    /// A block of 32 MiB or more is listed by its size while it lives.
    #[test]
    fn big_blocks_are_listed_while_they_live() {
        let size = (48usize << 20) + 4096;
        let listed = |at: usize| {
            super::BIG_AT
                .iter()
                .zip(&super::BIG_BYTES)
                .any(|(place, bytes)| place.load(Relaxed) == at && bytes.load(Relaxed) == size as u64)
        };
        let block = vec![0u8; size];
        let at = block.as_ptr() as usize;
        assert!(listed(at));
        assert!(super::big_blocks().contains("48.00"));
        drop(block);
        assert!(!listed(at));
    }

    /// The micro-measurement behind the lone stripes: the two additions of
    /// a count on a stripe of the thread's own against the locked additions
    /// of a shared one, and a small block made and released (printed;
    /// `--nocapture`).
    #[test]
    #[ignore = "a measurement: cargo test -p victory-gpu --profile diag --features diagnostics --lib own_tests -- --ignored --nocapture"]
    fn counting_costs() {
        const N: u32 = 4_000_000;
        let (own, _) = super::stripe();
        let t = std::time::Instant::now();
        for i in 0..N {
            let add = u64::from(i & 1);
            own[0].store(own[0].load(Relaxed).wrapping_add(add), Relaxed);
            own[4].store(own[4].load(Relaxed).wrapping_add(add), Relaxed);
        }
        let alone = t.elapsed().as_nanos() as f64 / f64::from(N);
        let shared = &super::OWN[super::ALONE].0;
        let t = std::time::Instant::now();
        for i in 0..N {
            let add = u64::from(i & 1);
            shared[0].fetch_add(add, Relaxed);
            shared[4].fetch_add(add, Relaxed);
        }
        let locked = t.elapsed().as_nanos() as f64 / f64::from(N);
        let t = std::time::Instant::now();
        for i in 0..N {
            std::hint::black_box(Box::new(i));
        }
        let block = t.elapsed().as_nanos() as f64 / f64::from(N);
        println!("a count: lone stripe {alone:.2} ns, shared stripe {locked:.2} ns; a 4-byte block made and released {block:.1} ns");
    }

    /// A big block shows in the counters while it lives (other tests
    /// allocate small blocks meanwhile: only the big class is looked at,
    /// and only for at least this block).
    #[test]
    fn a_block_is_counted_while_it_lives() {
        let block = vec![1u8; 8 << 20];
        assert!(own_total(3) >= 8 << 20);
        assert!(own_total(7) >= 1);
        drop(block);
        assert!(own_total(7) < u64::MAX / 2, "no count below zero");
    }
}

/// The process's memory in bytes: in physical memory, and committed
/// (private bytes).
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.WorkingSetSize as u64, c.PagefileUsage as u64)
}

/// Available physical memory and the process's available commit, in bytes.
fn system_memory() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let sampled = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    if sampled && m.ullAvailPageFile < 2 << 30 {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                available_commit_mib = m.ullAvailPageFile >> 20,
                "Windows commit headroom is low"
            );
        });
    }
    (m.ullAvailPhys, m.ullAvailPageFile)
}

/// Whether the address space is walked for the memory map line
/// ([`log_memory_map`]). On by default (this module is in diagnostics
/// builds only); `VICTORY_MEMORY_MAP=0` leaves the walk out.
fn memory_map() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_MAP").is_none_or(|v| v != "0"))
}

/// Asks which of the pages in `asked` (address, answer) are in physical
/// memory and adds the bytes each stands for (kind, bytes) to its kind.
fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, u64)>, present: &mut [u64; 11]) {
    use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: our own process; `asked` holds that many entries of two
    // words each, as the call fills them in.
    let ok = unsafe {
        K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), (asked.len() * 16) as u32)
    } != 0;
    if ok {
        for (a, &(kind, bytes)) in asked.iter().zip(stands.iter()) {
            // Bit 0 of the answer: the page is in the process's memory.
            if a[1] & 1 != 0 {
                present[kind] += bytes;
            }
        }
    }
    asked.clear();
    stands.clear();
}

/// The heap whose segment starts at `base`, from the segment's header (an
/// NT heap's `HEAP_SEGMENT`, 64-bit layout: signature 0xffeeffee at +0x10,
/// its heap at +0x28), or `None` if no heap segment starts there.
fn segment_heap(base: usize) -> Option<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut header = [0u8; 0x30];
    let mut done = 0usize;
    // SAFETY: our own process, read through the kernel into `header`: an
    // unreadable page fails the call instead of faulting.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            base as *const _,
            header.as_mut_ptr().cast(),
            header.len(),
            &mut done,
        )
    } != 0;
    if !ok || done != header.len() || header[0x10..0x14] != 0xffee_ffee_u32.to_le_bytes() {
        return None;
    }
    Some(usize::from_le_bytes(header[0x28..0x30].try_into().ok()?))
}

/// Logs what the process has committed by kind, from a walk of its address
/// space, and how much of each kind is in physical memory (one page in
/// sixteen is asked about). The kinds tell the title's own memory from
/// ours: the title's heaps and page views lie at console addresses (below
/// 1 TiB), everything the host allocates lies far above. Run 30, seen
/// from outside at 18.0 GB private: 4.6 GB the title's heaps, 2.6 GB in
/// 162 segments of the process heap, 2.7 GB in 707 allocations of 4 MiB
/// (windows of title memory), 0.46 GB the pipeline library's file; mapped
/// and so not private: 6.2 GB the title's page pool, 2.0 GB the driver's
/// shader cache file.
/// shortcut: the walk takes about 0.6 s on a thread of its own, and a
/// query of a region of gigabytes holds up other threads' memory calls
/// for its length (89 ms for the 6.2 GB pool), so it runs with every
/// sixth memory line only.
fn log_memory_map() {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE,
        VirtualQuery,
    };
    const KINDS: [&str; 11] = [
        "the title's heaps (private, console addresses)",
        "views of the title's pages (mapped, console addresses)",
        "the title's page pool (mapped, 1 GB and more)",
        "process heap segments (private, 15.8 MB each)",
        "windows of title memory and other 4 MiB allocations",
        "write-combined (buffers shared with the card)",
        "other private allocations of 1 MB and more",
        "other private allocations under 1 MB",
        "files and other mappings at host addresses",
        "programs and libraries",
        "segments of other heaps (private, 15.8 MB each)",
    ];
    /// Console addresses lie below this.
    const HOST: usize = 1 << 40;
    /// One page asked about stands for this much.
    const STEP: usize = 64 << 10;
    let started = std::time::Instant::now();
    // Allocation, start, bytes, committed, type, protection.
    type Region = (usize, usize, usize, bool, u32, u32);
    let mut regions: Vec<Region> = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        if m.State != MEM_FREE {
            regions.push((
                m.AllocationBase as usize,
                m.BaseAddress as usize,
                m.RegionSize,
                m.State == MEM_COMMIT,
                m.Type,
                m.Protect,
            ));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    // Each allocation's whole size, reserved parts included.
    let mut whole: HashMap<usize, usize> = HashMap::new();
    for r in &regions {
        *whole.entry(r.0).or_default() += r.2;
    }
    // The heap each 15.8 MB allocation's segment header names, once per
    // allocation: the process heap's segments are told from other heaps'
    // (AMD's compiler kept a heap of its own that grew past 30 GB).
    // SAFETY: a plain call.
    let process_heap = unsafe { windows_sys::Win32::System::Memory::GetProcessHeap() } as usize;
    let mut owner: HashMap<usize, usize> = HashMap::new();
    for r in regions
        .iter()
        .filter(|r| r.0 == r.1 && r.3 && whole.get(&r.0) == Some(&0xfd_0000))
    {
        if let Some(heap) = segment_heap(r.0) {
            owner.insert(r.0, heap);
        }
    }
    let kind = |r: &Region| -> usize {
        let size = whole.get(&r.0).copied().unwrap_or(0);
        let console = r.0 < HOST;
        if r.4 == MEM_IMAGE {
            9
        } else if r.4 == MEM_MAPPED {
            match (console, size >= 1 << 30) {
                (false, _) => 8,
                (true, true) => 2,
                (true, false) => 1,
            }
        } else if console {
            0
        } else if size == 0xfd_0000 {
            if owner.get(&r.0) == Some(&process_heap) {
                3
            } else {
                10
            }
        } else if size == 4 << 20 {
            4
        } else if r.5 & PAGE_WRITECOMBINE != 0 {
            5
        } else if size >= 1 << 20 {
            6
        } else {
            7
        }
    };
    let mut committed = [0u64; 11];
    let mut present = [0u64; 11];
    let mut asked: Vec<[usize; 2]> = Vec::with_capacity(4096);
    let mut stands: Vec<(usize, u64)> = Vec::with_capacity(4096);
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        committed[k] += r.2 as u64;
        let mut offset = 0;
        while offset < r.2 {
            asked.push([r.1 + offset, 0]);
            stands.push((k, STEP.min(r.2 - offset) as u64));
            if asked.len() == 4096 {
                in_memory(&mut asked, &mut stands, &mut present);
            }
            offset += STEP;
        }
    }
    if !asked.is_empty() {
        in_memory(&mut asked, &mut stands, &mut present);
    }
    let text: Vec<String> = (0..KINDS.len())
        .filter(|&k| committed[k] > 0)
        .map(|k| format!("{} {} MB ({} MB in memory)", KINDS[k], committed[k] >> 20, present[k] >> 20))
        .collect();
    tracing::info!(
        "memory map ({} regions, {} ms): {}",
        regions.len(),
        started.elapsed().as_millis(),
        text.join("; ")
    );
    // The heaps other than the process heap, biggest first: their segments
    // and the bytes committed in them (heap 0: no header could be read).
    let mut heaps: HashMap<usize, (u64, u64)> = HashMap::new();
    for heap in owner.values().filter(|&&heap| heap != process_heap) {
        heaps.entry(*heap).or_default().0 += 1;
    }
    for r in regions.iter().filter(|r| r.3 && kind(r) == 10) {
        heaps
            .entry(owner.get(&r.0).copied().unwrap_or(0))
            .or_default()
            .1 += r.2 as u64;
    }
    if !heaps.is_empty() {
        let mut heaps: Vec<(usize, (u64, u64))> = heaps.into_iter().collect();
        heaps.sort_by_key(|h| std::cmp::Reverse(h.1.1));
        let biggest: Vec<String> = heaps
            .iter()
            .take(5)
            .map(|(heap, (segments, bytes))| {
                format!("heap {heap:#x}: {} MB in {segments} segments", bytes >> 20)
            })
            .collect();
        tracing::info!(
            "memory map, heaps other than the process heap ({}, the biggest first): {}",
            heaps.len(),
            biggest.join("; ")
        );
    }
    // The private allocations the kinds above leave as "other" (and the
    // write-combined ones), grouped by their size: the sizes name their
    // owners (the pipeline library's file to the byte, a snapshot buffer
    // of a title heap's size, the driver's blocks). Committed bytes by
    // allocation first, then allocations by (kind, whole size).
    let mut by_allocation: HashMap<usize, (usize, u64)> = HashMap::new();
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        if k == 5 || k == 6 {
            by_allocation.entry(r.0).or_insert((k, 0)).1 += r.2 as u64;
        }
    }
    let mut groups: HashMap<(usize, usize), (u64, u64)> = HashMap::new();
    for (base, &(k, bytes)) in &by_allocation {
        let group = groups.entry((k, whole.get(base).copied().unwrap_or(0))).or_default();
        group.0 += 1;
        group.1 += bytes;
    }
    let mut groups: Vec<((usize, usize), (u64, u64))> = groups.into_iter().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.1.1));
    let biggest: Vec<String> = groups
        .iter()
        .take(20)
        .map(|&((k, size), (n, bytes))| {
            format!(
                "{n} of {:.2} MB{}: {} MB",
                size as f64 / (1 << 20) as f64,
                if k == 5 { " write-combined" } else { "" },
                bytes >> 20
            )
        })
        .collect();
    tracing::info!(
        "memory map, other private allocations by size, the {} biggest groups of {} (committed): {}",
        biggest.len(),
        groups.len(),
        biggest.join("; ")
    );
}

/// Logs where the memory is: the process and the PC, then what this crate
/// holds by kind (alive now), then the host buffers made since the start.
fn log_memory() {
    use std::sync::atomic::Ordering::Relaxed;
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let (working, private) = process_memory();
    let (free, left) = system_memory();
    let copies = crate::shadow::memory();
    let staged = crate::texture::memory();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let made: Vec<u64> = crate::host::MADE.iter().map(|m| m.load(Relaxed)).collect();
    let windows = crate::shadow::windows();
    let stagings = crate::texture::let_go();
    let census = crate::resource::census();
    let kinds: Vec<u64> = crate::resource::KINDS.iter().map(|k| k.load(Relaxed)).collect();
    let card = crate::host::host()
        .and_then(|h| h.video_memory())
        .unwrap_or_default();
    // Everything the system says of the two memory groups, beside the
    // budget and use the lines below print.
    if let Some(h) = crate::host::host() {
        h.log_video_memory("with the memory line");
    }
    // What else holds memory: the map, on a thread of its own, with every
    // sixth line; and what the shader table and the pipeline library keep.
    static LINES: AtomicU64 = AtomicU64::new(0);
    static WALKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if memory_map() && LINES.fetch_add(1, Relaxed) % 6 == 5 && !WALKING.swap(true, Relaxed) {
        let walk = std::thread::Builder::new().name("victory-memory-map".into()).spawn(|| {
            log_memory_map();
            WALKING.store(false, Relaxed);
        });
        if walk.is_err() {
            WALKING.store(false, Relaxed);
        }
    }
    let kept = crate::pipeline::kept();
    tracing::info!(
        "memory kept beside the lines below: {} shader containers {:.0} MB, {} translations \
         {:.0} MB; the pipeline library's file {:.0} MB (its last save wrote {:.0} MB)",
        kept[0],
        mb(kept[1]),
        kept[2],
        mb(kept[3]),
        mb(kept[4]),
        mb(kept[5]),
    );
    tracing::info!(
        "memory of textures: {} title textures have no host copy yet ({:.0} MB of title memory); \
         host copies the host renders to or writes: {} alive {:.0} MB; host copies the title \
         fills: {} alive {:.0} MB ({} of them placed), {} let go {:.0} MB and {} made again \
         {:.0} MB since the start; {} host copies made in all; passes that made none for lack of \
         room: {}; the card: {:.0} MB in use of a budget of {:.0} MB, {:.0} MB of the PC's memory \
         shared with it in use",
        census[2],
        mb(census[3]),
        census[0],
        mb(census[1]),
        census[4],
        mb(census[5]),
        census[6],
        kinds[2],
        mb(kinds[3]),
        kinds[4],
        mb(kinds[5]),
        kinds[6],
        crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        mb(card[1]),
        mb(card[0]),
        mb(card[3]),
    );
    tracing::info!(
        "memory let go since the start: {} windows outside buffers {:.0} MB, {} staging buffers \
         {:.0} MB; of the windows alive, {:.0} MB lie between the lowest and highest address \
         asked for in each",
        windows[0],
        mb(windows[1]),
        stagings[0],
        mb(stagings[1]),
        mb(windows[2]),
    );
    tracing::info!(
        "memory: {:.0} MB private ({:.0} MB of it in memory; {:.0} MB physical memory available, \
         {:.0} MB left to commit for this process); host copies of title memory: {} alive {:.0} MB, of them {} windows \
         outside buffers {:.0} MB; snapshot buffers kept: {} reserving {:.0} MB; staging of \
         textures: {} alive {:.0} MB, {} kept by their texture, {} buffers taken again and {} made \
         (VICTORY_STAGING_SPARE_MB); host textures: {} alive {:.0} MB, \
         {} placed in title memory; made so far: {} upload buffers {:.0} MB, {} shared buffers \
         {:.0} MB, {} card buffers {:.0} MB",
        mb(private),
        mb(working),
        mb(free),
        mb(left),
        copies[0],
        mb(copies[1]),
        copies[2],
        mb(copies[3]),
        copies[4],
        mb(copies[5]),
        staged[0],
        mb(staged[1]),
        staged[2],
        crate::host::STAGING_TAKEN[0].load(Relaxed),
        crate::host::STAGING_TAKEN[1].load(Relaxed),
        textures[0],
        mb(textures[1]),
        crate::resource::placed_textures(),
        made[0],
        mb(made[1]),
        made[2],
        mb(made[3]),
        made[4],
        mb(made[5]),
    );
    tracing::info!(
        "memory, placed textures: {} of them were asked for again with other filler words in their description          (one resource and host copy each since the filler left the key; VICTORY_PLACED_FILLER=1 keeps it)",
        crate::resource::placed_filler_twins()
    );
    // What the line above leaves open: what is alive of the buffers "made
    // so far", what the windows hold and what holds them, and what our own
    // code has allocated.
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let window = crate::shadow::window_use();
    let views = crate::descriptor::views_kept();
    let own: Vec<u64> = (0..8).map(own_total).collect();
    tracing::info!(
        "memory, what holds it: {} host lists alive; their scratch memory: {} chunks {:.0} MB, of          them {} chunks {:.0} MB spare ({} taken from the spare ones so far); their other upload          buffers: {} alive {:.0} MB; windows outside buffers (grid {} KiB): {:.0} MB of their          pages ever copied, {} shown by descriptor slots {:.0} MB, {} held by lists or the pass          over the heap {:.0} MB, {} asked for without a length {:.0} MB; buffer views of the heap          whose slot showed another copy than the one brought up to date: {}; distinct views          kept: {} ({} free) {:.0} MB; our own allocations: {:.0} MB in {} blocks (under 4 KiB          {:.0} MB in {}, to 64 KiB {:.0} MB in {}, to 1 MiB {:.0} MB in {}, larger {:.0} MB in {})",
        alive[0],
        alive[1],
        mb(alive[2]),
        alive[3],
        mb(alive[4]),
        alive[5],
        alive[6],
        mb(alive[7]),
        window[7] >> 10,
        mb(window[0]),
        window[1],
        mb(window[2]),
        window[3],
        mb(window[4]),
        window[5],
        mb(window[6]),
        crate::draw::HEAP_VIEWS_ELSEWHERE.load(Relaxed),
        views[0],
        views[1],
        mb(views[2]),
        mb(own[..4].iter().sum()),
        own[4..].iter().sum::<u64>(),
        mb(own[0]),
        own[4],
        mb(own[1]),
        own[5],
        mb(own[2]),
        own[6],
        mb(own[3]),
        own[7],
    );
    // Round 2: what the view table still holds and what goes around it,
    // what host lists cost, the card's share that is the layer's own or
    // the title's no more, and which big blocks are ours.
    let lists: Vec<u64> = crate::host::LISTS.iter().map(|l| l.load(Relaxed)).collect();
    tracing::info!(
        "memory, more: views named whole in their records since the start: {} constant buffer \
         views, {} views of placed memory; of the view table's entries {} are constant buffer \
         views and {} views of placed memory; shapes of placed views kept: {}; host lists: {} \
         made by the host, {} preludes made for them, private bytes grew {:.0} MB while \
         lists were made; twins of depth textures: {:.0} MB; host copies of \
         placed textures whose title memory is not mapped any more (estimate): {:.0} MB, of \
         them still shown by a descriptor slot {:.0} MB; the pipeline library's mapped file \
         let go from memory so far: {:.0} MB; our own blocks of 32 MB and more, in MB: {}",
        views[6],
        views[7],
        views[3],
        views[4],
        views[5],
        lists[0],
        lists[1],
        lists[2] as i64 as f64 / (1 << 20) as f64,
        mb(census[7]),
        mb(census[8]),
        mb(census[9]),
        mb(kept[6]),
        big_blocks(),
    );
    // Round 3: whether windows hold the same memory more than once (they
    // can, once they are cut finer than 4 MiB), and how many host lists
    // the title made and never used.
    tracing::info!(
        "memory, windows: the {:.0} MB of windows lie over {:.0} MB of title memory (the rest is \
         memory that is in more than one window, and is copied once for each); host lists never \
         reset since they were made: {}",
        mb(copies[3]),
        mb(crate::shadow::window_cover()),
        lists[3],
    );
}

/// Microseconds since the first call.
fn now_us() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// When the last frame was presented and when the last submission returned
/// ([`now_us`]), and whether a list call and a submission followed that
/// present yet.
static PRESENTED_AT: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_AT: AtomicU64 = AtomicU64::new(0);
static RECORDED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static SUBMITTED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// What a thread that submits did between its submissions and in them
/// (microseconds; list calls in nanoseconds).
#[derive(Default, Clone, Copy)]
struct Submitter {
    submissions: u64,
    /// From a submission's return to the next one's call: wall time, of it
    /// the thread running (user, kernel), and inside our list calls.
    between: u64,
    between_user: u64,
    between_kernel: u64,
    between_ours: u64,
    /// Waiting for the submission lock; running inside submissions.
    lock: u64,
    inside_user: u64,
    inside_kernel: u64,
}

/// By thread: the submitting threads, and list-call time (nanoseconds,
/// calls) of every recording thread.
static SUBMITTERS: Mutex<Vec<(u32, Submitter)>> = Mutex::new(Vec::new());
static RECORDERS: Mutex<Vec<(u32, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// This thread's last submission: when it returned ([`now_us`]) and the
    /// thread's processor time then (user, kernel).
    static LEFT: std::cell::Cell<Option<(u64, u64, u64)>> = const { std::cell::Cell::new(None) };
    /// The thread's processor time when its running submission was called.
    static ENTERED: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    /// List-call nanoseconds of this thread since its last submission.
    static OURS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// List-call time not yet added to `RECORDERS` (nanoseconds, calls).
    static UNSENT: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

/// This thread's processor time so far (user, kernel), in microseconds.
fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread and four valid out pointers.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's processor time so far (user, kernel), in microseconds.
fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

fn submitter(all: &mut Vec<(u32, Submitter)>, id: u32) -> &mut Submitter {
    let at = match all.iter().position(|s| s.0 == id) {
        Some(at) => at,
        None => {
            all.push((id, Submitter::default()));
            all.len() - 1
        }
    };
    &mut all[at].1
}

/// Whether list-call time is also counted by the method that was called
/// (`VICTORY_FRAME_PARTS=1`; off by default): a line beside "list calls by
/// thread" names the places in `list.rs` whose calls took longest.
///
/// The recording line says how long all list calls took together (run
/// all15b-on1, the boat scene: 190-360 ms a frame on five threads, 7-9
/// microseconds a call; the avatarmanz build in its own log 100-160 ms,
/// 3.4 a call) and by thread, but not which methods the time is in: the
/// draws, the indirect executions with their clamps, the tables with their
/// constant fills, the copies. Counting costs a look in a short list a call.
fn frame_parts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FRAME_PARTS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, with [`frame_parts`]: list-call time by
/// the line of `list.rs` the call was made at (line, nanoseconds, calls).
static PLACES: Mutex<Vec<(u32, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// This thread's part of [`PLACES`] not yet added to it.
    static PLACES_UNSENT: std::cell::RefCell<Vec<(u32, u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Adds `ns` in `calls` calls made at `line` to `places`.
fn place_add(places: &mut Vec<(u32, u64, u64)>, line: u32, ns: u64, calls: u64) {
    match places.iter_mut().find(|p| p.0 == line) {
        Some(p) => (p.1, p.2) = (p.1 + ns, p.2 + calls),
        None => places.push((line, ns, calls)),
    }
}

/// The `most` places of `places` whose calls took longest, longest first,
/// as text per frame, and the milliseconds per frame of all of them.
fn places_text(mut places: Vec<(u32, u64, u64)>, frames: f64, most: usize) -> (f64, Vec<String>) {
    places.sort_by_key(|p| std::cmp::Reverse(p.1));
    let total = places.iter().map(|p| p.1).sum::<u64>() as f64 / 1e6 / frames;
    let text = places
        .iter()
        .take(most)
        .map(|p| format!("list.rs:{} {:.1} ms in {:.0} calls", p.0, p.1 as f64 / 1e6 / frames, p.2 as f64 / frames))
        .collect();
    (total, text)
}

/// The timing of a list call about to be made: every call's, or with
/// `VICTORY_TIMING_LEAN=1` one call's in 16 on each thread (counted 16
/// times: the list-call times on the frame line, by thread and between a
/// thread's submissions are then estimates; their call counts stay exact).
/// Every call's with `VICTORY_FRAME_PARTS=1`, and the first after a present
/// (for "until the first list call").
pub(crate) fn record_timer() -> crate::timing::Timer {
    if !crate::timing::lean() || frame_parts() || !RECORDED_SINCE.load(Relaxed) {
        return crate::timing::Timer::always();
    }
    crate::timing::Timer::sampled()
}

/// A list call on this thread, made at line `place` of `list.rs`, timed by
/// `started` ([`record_timer`]), has returned.
pub(crate) fn recorded(started: crate::timing::Timer, place: u32) {
    let ns = started.ns();
    add(C::RecordCalls, 1);
    add(C::RecordNs, ns);
    if frame_parts() {
        PLACES_UNSENT.with(|p| place_add(&mut p.borrow_mut(), place, ns, 1));
    }
    if !RECORDED_SINCE.load(Relaxed) && !RECORDED_SINCE.swap(true, Relaxed) {
        let began = now_us().saturating_sub(ns / 1000);
        add(C::UntilRecordUs, began.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
    OURS.with(|o| o.set(o.get() + ns));
    let (unsent, calls) = UNSENT.with(|u| u.get());
    let (unsent, calls) = (unsent + ns, calls + 1);
    // By thread, a couple of milliseconds at a time.
    if unsent < 2_000_000 {
        UNSENT.with(|u| u.set((unsent, calls)));
        return;
    }
    UNSENT.with(|u| u.set((0, 0)));
    if frame_parts() {
        let mine = PLACES_UNSENT.with(|p| std::mem::take(&mut *p.borrow_mut()));
        let mut all = PLACES.lock().unwrap_or_else(|e| e.into_inner());
        for (line, ns, calls) in mine {
            place_add(&mut all, line, ns, calls);
        }
    }
    let id = thread_id();
    let mut all = RECORDERS.lock().unwrap_or_else(|e| e.into_inner());
    match all.iter_mut().find(|r| r.0 == id) {
        Some(r) => (r.1, r.2) = (r.1 + unsent, r.2 + calls),
        None => all.push((id, unsent, calls)),
    }
}

/// A submission is called on this thread (before it waits for the
/// submission lock).
pub(crate) fn submission_begins() {
    let now = now_us();
    if !SUBMITTED_SINCE.load(Relaxed) && !SUBMITTED_SINCE.swap(true, Relaxed) {
        add(C::UntilSubmitUs, now.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
    let (user, kernel) = thread_times();
    ENTERED.with(|e| e.set((user, kernel)));
    let ours = OURS.with(|o| o.replace(0));
    if let Some((left, u, k)) = LEFT.with(|l| l.get()) {
        let mut all = SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner());
        let s = submitter(&mut all, thread_id());
        s.between += now.saturating_sub(left);
        s.between_user += user.saturating_sub(u);
        s.between_kernel += kernel.saturating_sub(k);
        s.between_ours += ours;
    }
}

/// The submission returns; it waited `lock` microseconds for the
/// submission lock.
pub(crate) fn submission_ends(lock: u64) {
    let now = now_us();
    SUBMITTED_AT.store(now, Relaxed);
    let (user, kernel) = thread_times();
    let (u, k) = ENTERED.with(|e| e.get());
    LEFT.with(|l| l.set(Some((now, user, kernel))));
    OURS.with(|o| o.set(0));
    let mut all = SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner());
    let s = submitter(&mut all, thread_id());
    s.submissions += 1;
    s.lock += lock;
    s.inside_user += user.saturating_sub(u);
    s.inside_kernel += kernel.saturating_sub(k);
}

/// The title calls its present (before our own copy to the window runs).
pub(crate) fn present_begins() {
    add(C::TailUs, now_us().saturating_sub(SUBMITTED_AT.load(Relaxed)));
}

/// A frame was presented.
pub(crate) fn presented() {
    add(C::Frames, 1);
    PRESENTED_AT.store(now_us(), Relaxed);
    RECORDED_SINCE.store(false, Relaxed);
    SUBMITTED_SINCE.store(false, Relaxed);
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame, and the ranges that took most bytes. The
/// counters run from line to line, so they are divided by the frames
/// presented since the last line, not by `frames` (the frames of the last
/// second or more: a 4 s frame after an unlogged 4 s frame showed two
/// frames' counts as one's).
pub(crate) fn log(frames: f64) {
    let counted = COUNT[C::Frames as usize].swap(0, Relaxed);
    let frames = if counted > 0 { counted as f64 } else { frames };
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults ({:.0} other views of guarded pages made read-only with \
         them, {:.0} pages shown at too many places for that, VICTORY_GUARD_ALIASES); textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed, {:.1} changes the guard did not see;          stale pages by their written 4 KiB pages: {:.0}, whole {:.0}, {:.1} MB left out (VICTORY_SUBPAGE_COPIES)",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        c[C::GuardAliasProtects as usize],
        c[C::GuardAliasMany as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        c[C::TexHot as usize],
        c[C::PiecePages as usize],
        c[C::PiecePagesWhole as usize],
        mb(c[C::PieceBytesLeftOut as usize]),
    );
    tracing::info!(
        "guards per frame: {:.1} ms in all; {:.0} questions whether a range is view memory {:.1} ms, {:.0} calls making \
         blocks read-only {:.1} ms, {:.1} ms waiting for the guards' lock, {:.0} questions whether blocks noted          unmapped still are {:.1} ms (VICTORY_GUARD_BITS)",
        ms(c[C::GuardUs as usize]),
        c[C::GuardQueries as usize],
        ms(c[C::GuardQueryUs as usize]),
        c[C::GuardProtects as usize],
        ms(c[C::GuardProtectUs as usize]),
        ms(c[C::GuardLockUs as usize]),
        c[C::GuardAbsentQueries as usize],
        ms(c[C::GuardAbsentUs as usize]),
    );
    tracing::info!(
        "recording per frame ({frames:.0} frames): {:.0} list calls {:.1} ms (all recording \
         threads); {:.1} ms waiting for pipelines; dynamic slots: {:.0} tables asked, {:.0} \
         refused, {:.1} ms searching, {:.0} slots taken; tables read {:.0}, {:.0} of them in the \
         title's slots (no dynamic slots), {:.0} made afresh again in a list; table constants {:.0} \
         read in place, {:.0} copied; notes {:.1} ms; shader-written ranges: {:.0} steps in \
         {:.1} submissions, snapshots {:.1} ms, {:.1} MB compared back; sparse reads {:.1} ms; \
         guards {:.1} ms; {:.1} MB copied whole (memory not watched); {:.1} frame waits \
         sleeping {:.1} ms (all threads); ranges of shaders that cannot store to buffers: \
         {:.0} steps {:.1} MB left out",
        c[C::RecordCalls as usize],
        c[C::RecordNs as usize] / 1e6,
        ms(c[C::PipelineWaitUs as usize]),
        c[C::DynamicAsked as usize],
        c[C::DynamicRefused as usize],
        c[C::DynamicNs as usize] / 1e6,
        c[C::DynamicTaken as usize],
        c[C::TablesRead as usize],
        c[C::TablesInPlace as usize],
        c[C::TablesReused as usize],
        c[C::TableConstantsInPlace as usize],
        c[C::TableConstantsCopied as usize],
        c[C::NoteNs as usize] / 1e6,
        c[C::GpuWrites as usize],
        c[C::BackSubmissions as usize],
        ms(c[C::SnapshotUs as usize]),
        mb(c[C::BackCompared as usize]),
        ms(c[C::SparseUs as usize]),
        ms(c[C::GuardUs as usize]),
        mb(c[C::WholeBytes as usize]),
        c[C::PaceWaits as usize],
        ms(c[C::PaceUs as usize]),
        c[C::GpuWritesLeftOut as usize],
        mb(c[C::GpuWritesLeftOutBytes as usize]),
    );
    // Why dynamic descriptor slots were refused (`HostHeap::dynamic`).
    {
        use std::sync::atomic::Ordering::Relaxed;
        let why: Vec<f64> = crate::descriptor::host::DYNAMIC_WHY
            .iter()
            .map(|n| n.swap(0, Relaxed) as f64 / frames)
            .collect();
        let low = crate::descriptor::host::DYNAMIC_LOW[0].swap(u32::MAX, Relaxed);
        let low = if low == u32::MAX { "no table asked".to_owned() } else { format!("{low}") };
        tracing::info!(
            "dynamic slots per frame, tables refused by cause: {:.0} asked more slots than were free (the ring was full), {:.0} had room \
             but a search for a table no longer had failed since the last release (not searched), {:.0} were searched for and no run of \
             free slots was that long (the free slots lay apart); {:.0} tables got a run lying across the place the search starts from \
             (missed by the search as it was before, VICTORY_DYNAMIC_ACROSS=0); the fewest free slots seen since the last line: {low} of {}",
            why[0],
            why[1],
            why[2],
            why[3],
            crate::descriptor::host::DYNAMIC_LOW[1].load(Relaxed),
        );
    }
    // Where the wall clock of a frame goes: the phases of a frame, the
    // processor time of the whole process, and what each submitting and
    // recording thread did.
    static CPU: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let (user, kernel) = process_times();
    let (user, kernel) = {
        let mut last = CPU.lock().unwrap_or_else(|e| e.into_inner());
        let spent = (user.saturating_sub(last.0), kernel.saturating_sub(last.1));
        *last = (user, kernel);
        (spent.0 as f64 / 1e3 / frames, spent.1 as f64 / 1e3 / frames)
    };
    tracing::info!(
        "frame account per frame: after a present {:.1} ms until the first list call and {:.1} ms \
         until the first submission; {:.1} ms from the last submission to the present call; \
         lists reset in {:.1} ms ({:.1} ms of it letting go of what they held, VICTORY_RESET_REAPER); the process ran {user:.0} ms user + {kernel:.0} ms kernel on \
         all threads; what shaders wrote: {:.1} ms finding {:.0} pieces that differ, {:.1} ms \
         putting them back; texture uploads: {:.0} recorded, {:.0} left out; {:.1} held \
         submissions run at the present with their waits not met; waits for memory values: \
         {:.0} looks {:.1} ms; {:.0} host lists made {:.1} ms; {:.0} parts of held submissions \
         ran alone that the next part could have run with",
        ms(c[C::UntilRecordUs as usize]),
        ms(c[C::UntilSubmitUs as usize]),
        ms(c[C::TailUs as usize]),
        c[C::ResetNs as usize] / 1e6,
        c[C::ResetDropNs as usize] / 1e6,
        ms(c[C::BackScanUs as usize]),
        c[C::BackDiffer as usize],
        ms(c[C::BackSyncUs as usize]),
        c[C::TexUploads as usize],
        c[C::TexSkipped as usize],
        c[C::Forced as usize],
        c[C::WaitChecks as usize],
        c[C::WaitCheckNs as usize] / 1e6,
        c[C::HostLists as usize],
        c[C::HostListNs as usize] / 1e6,
        c[C::PartsTogether as usize],
    );
    // What is copied whole, largest first: memory nobody tells us about.
    let whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut whole: Vec<_> = whole.into_iter().flatten().collect();
    whole.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!(
        "  copied whole (memory not watched): {:.1} ms per frame in {} ranges; the largest follow",
        ms(c[C::WholeUs as usize]),
        whole.len()
    );
    for ((src, len), (n, us)) in whole.into_iter().take(6) {
        tracing::info!(
            "  copied whole from {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
    // The waiting for the card: many short waits or a few long ones, how
    // much of it is for submissions the title gets nothing back from, and
    // the longest one.
    let longest = std::mem::take(&mut *LONGEST.lock().unwrap_or_else(|e| e.into_inner()));
    tracing::info!(
        "  waiting for the card per frame: {:.1} submissions under 2 ms ({:.1} ms), {:.1} of 2-8 ms \
         ({:.1} ms), {:.1} of 8-32 ms ({:.1} ms), {:.1} longer ({:.1} ms); {:.1} submissions hand \
         nothing back to the title ({:.1} ms of the waiting); the longest since the last line: \
         {:.1} ms, {}",
        c[C::CardUnder2 as usize],
        ms(c[C::CardUnder2Us as usize]),
        c[C::Card2To8 as usize],
        ms(c[C::Card2To8Us as usize]),
        c[C::Card8To32 as usize],
        ms(c[C::Card8To32Us as usize]),
        c[C::CardOver32 as usize],
        ms(c[C::CardOver32Us as usize]),
        c[C::NothingBack as usize],
        ms(c[C::NothingBackUs as usize]),
        longest.0 as f64 / 1e3,
        longest.1
    );
    let by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
    for (what, (n, us)) in by.into_iter().take(12) {
        tracing::info!(
            "  the card's time by what ran: {:.1} ms a frame in {:.1} submissions a frame: {}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            what.chars().take(700).collect::<String>()
        );
    }
    let by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us, _))| std::cmp::Reverse(us));
    // All of them together, and by their first word: the forty lines below
    // are the heaviest only (100 of 226 ms waited in run 102), and what the
    // commands do not account for is the layer's own lists (the card
    // copies' passes, preludes) and the waits themselves.
    if !by.is_empty() {
        let mut kinds: Vec<(String, u64, u64)> = Vec::new();
        for (what, (n, us, _)) in &by {
            let word = what.split(' ').next().unwrap_or("").to_owned();
            match kinds.iter_mut().find(|k| k.0 == word) {
                Some(k) => (k.1, k.2) = (k.1 + n, k.2 + us),
                None => kinds.push((word, *n, *us)),
            }
        }
        kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
        let all: u64 = kinds.iter().map(|k| k.2).sum();
        let text: Vec<String> =
            kinds.iter().map(|(word, n, us)| format!("{word} {:.1} ms in {:.0}", *us as f64 / 1e3 / frames, *n as f64 / frames)).collect();
        tracing::info!(
            "  the card's time by command (timing run), all {} kinds together: {:.1} ms a frame; by first word: {}",
            by.len(),
            all as f64 / 1e3 / frames,
            text.join(", ")
        );
    }
    for (what, (n, us, longest)) in by.into_iter().take(40) {
        tracing::info!(
            "  the card's time by command (timing run): {:.2} ms a frame in {:.1} commands a frame, the longest {:.2} ms: {what}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            longest as f64 / 1e3
        );
    }
    tracing::info!(
        "  waits for every earlier shader write, per frame: {:.0} for the title's barrier groups, {:.0} \
         for its barriers on buffers; {:.0} indirect commands bounded (a dispatch of ours between two \
         such waits each); barrier experiment {}",
        c[C::WaitsForGroups as usize],
        c[C::WaitsForBuffers as usize],
        c[C::IndirectBounded as usize],
        barrier_test()
    );
    // What is compared after submissions, largest first, and how much of
    // it shaders had written.
    let compared = COMPARED.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut compared: Vec<_> = compared.into_iter().flatten().collect();
    compared.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!("  compared after submissions: {} ranges; the largest follow", compared.len());
    for ((src, len), (n, written)) in compared.into_iter().take(8) {
        tracing::info!(
            "  compared {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.3} MB/frame of it written back",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            written as f64 / 1e6 / frames
        );
    }
    let mut submitters = std::mem::take(&mut *SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner()));
    submitters.sort_by_key(|s| std::cmp::Reverse(s.1.submissions));
    for (id, s) in submitters.into_iter().take(4) {
        let per = |v: u64| v as f64 / 1e3 / frames;
        tracing::info!(
            "  thread {id} submitted {:.1}x/frame: between its submissions {:.1} ms (running \
             {:.1} ms user + {:.1} ms kernel; {:.1} ms in list calls), in them running {:.1} ms \
             user + {:.1} ms kernel, {:.1} ms waiting for the submission lock",
            s.submissions as f64 / frames,
            per(s.between),
            per(s.between_user),
            per(s.between_kernel),
            s.between_ours as f64 / 1e6 / frames,
            per(s.inside_user),
            per(s.inside_kernel),
            per(s.lock),
        );
    }
    let mut recorders = std::mem::take(&mut *RECORDERS.lock().unwrap_or_else(|e| e.into_inner()));
    recorders.sort_by_key(|r| std::cmp::Reverse(r.1));
    let busiest: Vec<String> = recorders
        .iter()
        .take(6)
        .map(|r| format!("thread {} {:.0} ms in {:.0} calls", r.0, r.1 as f64 / 1e6 / frames, r.2 as f64 / frames))
        .collect();
    tracing::info!("  list calls by thread: {} threads; {}", recorders.len(), busiest.join(", "));
    if frame_parts() {
        let places = std::mem::take(&mut *PLACES.lock().unwrap_or_else(|e| e.into_inner()));
        let n = places.len();
        let (total, most) = places_text(places, frames, 12);
        tracing::info!(
            "  list calls by the place in list.rs that makes them, per frame, all threads (VICTORY_FRAME_PARTS=1):              {n} places, {total:.1} ms; {}",
            most.join(", ")
        );
    }
    log_memory();
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

#[cfg(test)]
mod frame_parts_tests {
    use super::{place_add, places_text};

    /// Calls made at one line are counted together, and the places that
    /// took longest are named first, per frame.
    #[test]
    fn list_call_time_is_counted_by_place_and_the_longest_named_first() {
        let mut places = Vec::new();
        place_add(&mut places, 300, 2_000_000, 2);
        place_add(&mut places, 120, 9_000_000, 3);
        place_add(&mut places, 300, 4_000_000, 4);
        assert_eq!(places, [(300, 6_000_000, 6), (120, 9_000_000, 3)]);
        let (total, text) = places_text(places, 3.0, 1);
        assert!((total - 5.0).abs() < 1e-9);
        assert_eq!(text, ["list.rs:120 3.0 ms in 1 calls"]);
    }
}
