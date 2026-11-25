//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the section and offset it wrote itself
//! ([`title_memory_written`], xmem's `write_physical` and DirectStorage's
//! page destinations), and a write by the title through any view of a
//! guarded page faults like one at the guarded address (the guard makes
//! every view of the page read-only, [`open`] gives them all back). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
    /// With [`pieces`]: per page, which of its kernel pages the last
    /// askings found written ([`Pieces`]).
    pieces: BTreeMap<usize, Pieces>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    pieces: BTreeMap::new(),
});

/// The kernel pages of a page, one bit each.
const PIECES: usize = PAGE / KERNEL_PAGE;
const _: () = assert!(PIECES == 16);
/// Every kernel page of a page: what a write of unknown extent marks, and
/// what a copy takes when the history cannot say.
pub(crate) const ALL_PIECES: u16 = u16::MAX;

/// A page's last writes by kernel page: the askings that found it written
/// (the latest last; 0 for none) and the kernel pages each found. A copy
/// that last took the page at asking `a` needs the pieces of the askings
/// after `a`, which are all here if `a` is at or after `lost`.
#[derive(Clone, Copy, Default)]
struct Pieces {
    last: [(u64, u16); PIECE_HISTORY],
    /// The latest asking whose pieces are no longer here.
    lost: u64,
}

/// How many askings a page's [`Pieces`] remember. A copy refreshed less
/// often than its page is written takes the whole page, as without them.
const PIECE_HISTORY: usize = 4;

/// Whether a host copy's refresh takes only the kernel pages (4 KiB) the
/// title wrote of each written page (64 KiB), not the whole page
/// (`VICTORY_SUBPAGE_COPIES=1`; off by default).
///
/// The journal keeps 64 KiB pages: the checks of the world's big buffers
/// walk them, and finer pages made those checks cost more than the copies
/// they saved. But the kernel's answers name 4 KiB pages, and a 64 KiB page
/// one 4 KiB write touched was copied whole: 120-130 MB of page copies a
/// frame, 20-23 ms ("moves"), and what is copied goes up to the copies in
/// the card's memory after it (70-80 MB a frame). With the switch the
/// journal also keeps, per page, the kernel pages its last few askings
/// found written ([`Pieces`]), and a refresh copies those of the askings
/// since it last took the page; the whole page when that history is
/// incomplete (written more often than refreshed, written by a guard's
/// fault or by our own writes of unknown extent, never copied).
pub(crate) fn pieces() -> bool {
    #[cfg(test)]
    if FORCE_PIECES.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SUBPAGE_COPIES").is_some_and(|v| v == "1"))
}

/// Tests: [`pieces`] on, whatever the switch says. Pages noted before it
/// was set have no pieces and are copied whole.
#[cfg(test)]
pub(crate) static FORCE_PIECES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Notes that asking `asking` found kernel pages `mask` of page `page`
/// written ([`pieces`]). An asking older than the latest noted counts as
/// the latest: a copy that needs it needs that one too.
fn note_pieces(j: &mut Journal, page: usize, asking: u64, mask: u16) {
    let p = j.pieces.entry(page).or_default();
    let newest = &mut p.last[PIECE_HISTORY - 1];
    if newest.0 != 0 && asking <= newest.0 {
        newest.1 |= mask;
        return;
    }
    let oldest = p.last[0].0;
    p.lost = p.lost.max(oldest);
    p.last.rotate_left(1);
    p.last[PIECE_HISTORY - 1] = (asking, mask);
}

/// Notes a write of page `page` at asking `asking` in the journal, with
/// the kernel pages it wrote ([`pieces`]).
fn note_written(j: &mut Journal, page: usize, asking: u64, mask: u16) {
    j.written.insert(page, asking);
    if pieces() {
        note_pieces(j, page, asking, mask);
    }
}

/// The kernel pages of `[at, at + len)` in page `page`, one bit each.
fn pieces_of(page: usize, at: usize, len: usize) -> u16 {
    let (start, end) = ((page * PAGE).max(at), ((page + 1) * PAGE).min(at.saturating_add(len)));
    if start >= end {
        return 0;
    }
    let (first, last) = ((start - page * PAGE) / KERNEL_PAGE, (end - page * PAGE).div_ceil(KERNEL_PAGE));
    (first..last).fold(0u16, |mask, piece| mask | 1 << piece)
}

/// With [`pieces`]: for each `(page, last written, last taken)` of a copy's
/// stale pages, the kernel pages written since it last took the page
/// ([`ALL_PIECES`] where the journal cannot say: never taken, the history
/// does not reach back so far, or a write it does not hold the pieces of).
pub(crate) fn pieces_since(pages: &[(usize, u64, u64)]) -> Vec<u16> {
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    pages.iter().map(|&(page, written, taken)| pieces_after(j.pieces.get(&page), written, taken)).collect()
}

/// [`pieces_since`] of one page whose pieces are `pieces`.
fn pieces_after(pieces: Option<&Pieces>, written: u64, taken: u64) -> u16 {
    match pieces {
        Some(p) if taken != 0 && taken >= p.lost && written <= p.last[PIECE_HISTORY - 1].0 => {
            p.last.iter().filter(|e| e.0 > taken).fold(0, |mask, e| mask | e.1)
        }
        _ => ALL_PIECES,
    }
}

/// Groups the kernel's answer (4 KiB page addresses, in order) by page:
/// `(page, kernel pages)`.
fn by_page(found: &[usize]) -> Vec<(usize, u16)> {
    let mut pages: Vec<(usize, u16)> = Vec::new();
    for &at in found {
        let (page, piece) = (at / PAGE, (at % PAGE) / KERNEL_PAGE);
        match pages.last_mut() {
            Some(last) if last.0 == page => last.1 |= 1 << piece,
            _ => pages.push((page, 1 << piece)),
        }
    }
    pages
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    #[cfg(feature = "diagnostics")]
    let _who = asking_for(1);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Who asks the kernel ([`asking_for`]): 0 other, 1 a submission's
/// refreshes together ([`ask_ahead`], [`ask_once`]), 2 one copy's refresh,
/// 3 a copy's check under shader writes, 4 a texture's check.
#[cfg(feature = "diagnostics")]
pub(crate) const ASKERS: usize = 5;

#[cfg(feature = "diagnostics")]
thread_local! {
    static WHO: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Since the last frame line, per [`ASKERS`]: kernel askings and their
/// microseconds.
#[cfg(feature = "diagnostics")]
static BY_WHO: [[std::sync::atomic::AtomicU64; 2]; ASKERS] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; ASKERS];

/// The askings this thread makes until the result is dropped are counted
/// as `who`'s ([`ASKERS`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn asking_for(who: usize) -> impl Drop {
    struct Was(usize);
    impl Drop for Was {
        fn drop(&mut self) {
            WHO.with(|w| w.set(self.0));
        }
    }
    Was(WHO.with(|w| w.replace(who)))
}

/// [`BY_WHO`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn asks_by_who() -> [[u64; 2]; ASKERS] {
    BY_WHO.each_ref().map(|w| w.each_ref().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)))
}

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let who = WHO.with(|w| w.get()).min(ASKERS - 1);
    BY_WHO[who][0].fetch_add(1, Relaxed);
    BY_WHO[who][1].fetch_add(us, Relaxed);
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    ask_ahead(ranges);
    #[cfg(feature = "diagnostics")]
    let _who = asking_for(1);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// Tests: the journal forgets the pages of `[start, end)`, memory a test
/// frees (the next allocation there is another test's, and the journal is
/// one for the process).
#[cfg(test)]
pub(crate) fn forget_pages(start: usize, end: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    j.written.retain(|&page, _| !(first..last).contains(&page));
    j.pieces.retain(|&page, _| !(first..last).contains(&page));
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        note_written(&mut j, page, now, pieces_of(page, at, len));
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for (page, mask) in by_page(&j.found[..count]) {
        note_written(j, page, now, mask);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    // A guard's block is a page: all of it.
    let pages: Vec<(usize, u16)> =
        by_page(&j.found[..count]).into_iter().chain(guarded.into_iter().map(|page| (page, ALL_PIECES))).collect();
    for (page, mask) in pages {
        note_written(j, page, now, mask);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks a write through the kernel was announced for ([`writing`]),
    /// and the other views of their pages. While one is there no guard
    /// makes them read-only again: the kernel's write fails at a read-only
    /// page (a file read straight into title memory then fails with
    /// ERROR_INVALID_USER_BUFFER).
    kernel: Vec<KernelOpen>,
}

/// Blocks `[first, end)` a write through the kernel may be writing
/// ([`Guards::kernel`]): until the write is done ([`KernelWrite`], `until`
/// [`IN_FLIGHT`]), or until `until` (`crate::host::now_ms`) for a write that
/// goes on after the call that started it ([`KernelWrite::pending`]).
struct KernelOpen {
    first: usize,
    end: usize,
    until: u64,
    /// The [`KernelWrite`] that holds it; 0 for none (it ends with time).
    write: u64,
}

/// [`KernelOpen::until`] of a write under way: it ends when its
/// [`KernelWrite`] is dropped.
const IN_FLIGHT: u64 = u64::MAX;

/// How long blocks stay writable after a write through the kernel that goes
/// on after the call that started it ([`KernelWrite::pending`]: an
/// overlapped `ReadFile` the kernel finishes later; with
/// [`kernel_open_long`], after every announced write). While they do, every
/// check of a guard over them finds them written and its copy takes them
/// again.
///
/// Two runs with other views of guarded pages guarded too (fxceless #115,
/// `guard_aliases`) crashed in the opening cutscene a second after one
/// DirectStorage plain read failed with 0x800706f8: the read's destination
/// was opened before the read, and a guard check over another view of the
/// same pages made it read-only again while the read was under way.
const KERNEL_OPEN_MS: u64 = 2000;

/// Whether every announced write through the kernel keeps its blocks
/// writable for [`KERNEL_OPEN_MS`] (`VICTORY_KERNEL_OPEN_LONG=1`, as in
/// 498c2d5), not only while it is under way.
///
/// Our own writes into title memory announce themselves too: the title's
/// buffer copies done at submission (`crate::host::copy_mapped`, some
/// 14,000 a frame) and every write back of what shaders wrote. Kept open
/// for 2 s, their blocks were taken as written at every guard check for a
/// dozen frames: all6-cold and all6-base copied 350 MB a frame against 130
/// (page copies 48 ms against 23, 256 MB a frame sent up to copies in the
/// card's memory against 76). Those writes, the runtime's file reads and
/// DirectStorage's are all done when the call that makes them returns.
fn kernel_open_long() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KERNEL_OPEN_LONG").is_some_and(|v| v == "1"))
}

/// A write through the kernel announced with [`writing`]: the blocks it
/// writes stay writable, out of every guard's reach, until this is dropped
/// (the write is done; the next guard check over them takes them as written
/// and guards them again) or [`KernelWrite::pending`] says the write goes
/// on.
#[must_use = "the blocks may be guarded again as soon as it is dropped"]
pub struct KernelWrite {
    write: u64,
    /// The title's write ([`title_writing`]): `[at, at + len)`.
    title: Option<(usize, usize)>,
}

impl KernelWrite {
    /// The write goes on after the call that started it returned (an
    /// overlapped read the kernel finishes later): its blocks stay writable
    /// for [`KERNEL_OPEN_MS`] from now.
    pub fn pending(mut self) {
        if self.write != 0 {
            let until = crate::host::now_ms().saturating_add(KERNEL_OPEN_MS);
            let mut g = guards();
            for open in g.kernel.iter_mut().filter(|open| open.write == self.write) {
                open.until = until;
                open.write = 0;
            }
            self.write = 0;
        }
        // Dropped: the journal hears of it now ([`title_writing`]).
    }
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if self.write != 0 {
            let id = self.write;
            guards().kernel.retain(|open| open.write != id);
        }
        if let Some((at, len)) = self.title {
            journal_written(at, len);
        }
    }
}

/// With [`pieces`]: the title's write through the kernel into `[at, at +
/// len)` is done; the journal takes its pages as written now, with the
/// kernel pages it wrote. The kernel's own answer should say as much; a
/// write by a device that passes the page tables by would not, and a
/// whole-page copy took it with the page's other writes.
fn journal_written(at: usize, len: usize) {
    if !pieces() || len == 0 {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.asked += 1;
    let now = j.asked;
    // Answers from the journal stand at or above what it holds
    // ([`wrote`]).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        note_written(&mut j, page, now, pieces_of(page, at, len));
    }
}

/// [`Guards::kernel`] without the entries whose time is up.
fn kernel_expire(g: &mut Guards) {
    if !g.kernel.is_empty() {
        let now = crate::host::now_ms();
        g.kernel.retain(|open| now < open.until);
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    kernel: Vec::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`guards`] for a guard's check, the wait for the lock counted.
fn guards_timed() -> std::sync::MutexGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    {
        let waiting = std::time::Instant::now();
        let g = guards();
        crate::copystat::add(crate::copystat::C::GuardLockUs, waiting.elapsed().as_micros() as u64);
        g
    }
    #[cfg(not(feature = "diagnostics"))]
    guards()
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards_timed();
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains(&first)
    };
    if !known {
        #[cfg(feature = "diagnostics")]
        let asking = std::time::Instant::now();
        let view = is_view(start, end);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::GuardQueries, 1);
            add(C::GuardQueryUs, asking.elapsed().as_micros() as u64);
        }
        if !view {
            return None;
        }
    }
    install();
    let mut g = guards_timed();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to, true);
    let most = guard_aliases();
    if most == 0 {
        return;
    }
    let (aliases, many) = crate::inplace::aliases_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK), most);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::GuardAliasMany, many as u64);
    let _ = many;
    // Read-only already (its own guard's, or another fan-out's): no call.
    // Made read-only here, a block is not clean: its own guard, if any, has
    // not taken its bytes, and its next check copies them (as for a block
    // written through the runtime: read-only, not clean).
    let mut blocks: Vec<usize> = aliases
        .into_iter()
        .map(|alias| alias / BLOCK)
        .filter(|block| !(from..to).contains(block) && !g.protected.contains(block))
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::GuardAliasProtects, blocks.len() as u64);
    // A run of blocks in one call (each texture's other view is as
    // contiguous as the texture): 3,400-4,400 calls a frame, one a block,
    // were 15 ms of "guards" in run a2-base.
    for (first, end) in block_runs(&blocks) {
        protect_here(g, first, end, false);
    }
}

/// `blocks` (in order, distinct) as runs of consecutive blocks: first, end.
fn block_runs(blocks: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &block in blocks {
        match runs.last_mut() {
            Some(run) if run.1 == block => run.1 += 1,
            _ => runs.push((block, block + 1)),
        }
    }
    runs
}

/// How many addresses a guarded page may be shown at for its other views
/// to be guarded with it ([`protect`], [`open`], [`views_mapped`];
/// `VICTORY_GUARD_ALIASES`, 8; 0: none, as before fxceless #115).
///
/// fxceless #115 guards every view of the guarded pages. Its run had
/// preparing at about 1.7 s a frame: every protect, open and new view went
/// to every view of the page, a filler page among them that the runtime
/// maps at thousands of addresses; and each of those views was made clean
/// when protected, though nothing had taken its bytes (the stale pages
/// that remained). Here a page shown at more addresses than this is left
/// to the guards of its own views and to the texture hash, as before; the
/// pages behind the walls' stale copies were shown at six.
fn guard_aliases() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_GUARD_ALIASES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8)
    })
}

/// [`protect`] for one range, without the other views of its pages;
/// `clean`: the blocks' bytes were just taken (else they are only made
/// read-only, for a write through them to fault).
fn protect_here(g: &mut Guards, from: usize, to: usize, clean: bool) {
    // Not blocks a write through the kernel may be writing now.
    kernel_expire(g);
    if !g.kernel.is_empty() {
        let mut busy: Vec<(usize, usize)> = g
            .kernel
            .iter()
            .filter(|k| k.first < to && k.end > from)
            .map(|k| (k.first.max(from), k.end.min(to)))
            .collect();
        if !busy.is_empty() {
            busy.sort_unstable();
            let mut at = from;
            for (a, e) in busy {
                if a > at {
                    protect_runs(g, at, a, clean);
                }
                at = at.max(e);
            }
            if at < to {
                protect_runs(g, at, to, clean);
            }
            return;
        }
    }
    protect_runs(g, from, to, clean);
}

/// [`protect_here`] of blocks no write through the kernel is announced for.
fn protect_runs(g: &mut Guards, from: usize, to: usize, clean: bool) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        let protecting = std::time::Instant::now();
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::GuardProtects, 1);
            add(C::GuardProtectUs, protecting.elapsed().as_micros() as u64);
        }
        done
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        if clean {
            g.clean.extend(from..to);
        }
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let mut any = open_here(g, start, end);
    let most = guard_aliases();
    if start < end && most > 0 {
        let first = start / BLOCK * BLOCK;
        let (aliases, _) = crate::inplace::aliases_between(first, end.div_ceil(BLOCK).saturating_mul(BLOCK), most);
        let mut blocks: Vec<usize> = aliases.into_iter().map(|alias| alias / BLOCK).collect();
        blocks.sort_unstable();
        blocks.dedup();
        for (first, end) in block_runs(&blocks) {
            any |= open_here(g, first * BLOCK, end * BLOCK);
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only.
        unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READWRITE, &mut old) != 0 }
    };
    // A run in one call; block by block where it spans views.
    for (first, end) in block_runs(&blocks) {
        if end - first > 1 && set(first, end) {
            continue;
        }
        for block in first..end {
            set(block, block + 1);
        }
    }
    !blocks.is_empty()
}

/// Before the title's kernel read into its own memory `[at, at + len)`
/// (the runtime calls it as `kernel_will_write`): [`writing`], and the
/// journal hears of the write when it is done ([`journal_written`]).
pub fn title_writing(at: usize, len: usize) -> KernelWrite {
    let mut write = writing(at, len);
    write.title = Some((at, len));
    write
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks,
/// which no guard makes read-only again until the returned [`KernelWrite`]
/// is dropped, after the write.
pub fn writing(at: usize, len: usize) -> KernelWrite {
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    let long = kernel_open_long();
    let write = if long || at >= end { 0 } else { WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) };
    {
        let mut g = guards();
        open(&mut g, at, end);
        // Left writable, here and at the other views of the pages, while
        // the write is under way ([`KERNEL_OPEN_MS`] with the switch).
        kernel_expire(&mut g);
        if at < end {
            let until = if long { crate::host::now_ms().saturating_add(KERNEL_OPEN_MS) } else { IN_FLIGHT };
            g.kernel.push(KernelOpen { first: at / BLOCK, end: end.div_ceil(BLOCK), until, write });
            let most = guard_aliases();
            if most > 0 {
                let (aliases, _) =
                    crate::inplace::aliases_between(at / BLOCK * BLOCK, end.div_ceil(BLOCK).saturating_mul(BLOCK), most);
                let mut blocks: Vec<usize> = aliases.into_iter().map(|alias| alias / BLOCK).collect();
                blocks.sort_unstable();
                blocks.dedup();
                for (first, last) in block_runs(&blocks) {
                    g.kernel.push(KernelOpen { first, end: last, until, write });
                }
            }
        }
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
    KernelWrite { write, title: None }
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    let most = guard_aliases();
    if most == 0 {
        return;
    }
    // The new view is one of them: a page shown at more than `most`
    // addresses with it is left to its views' own guards.
    let ranges = crate::inplace::views_of_section_most(section, offset, len, most);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        open_here(&mut g, at, at.saturating_add(n));
    }
}

/// The runtime wrote the physical bytes `[offset, offset + len)` of
/// `section` through a view of its own (`victory-hle`: `xmem`'s
/// `write_physical`, DirectStorage's page destinations): no fault tells us,
/// so the title views that show those bytes are noted written here, as a
/// write of the title's through another view would be. Call it after the
/// write has landed: a check that runs before it may copy the old bytes,
/// and the one after sees these blocks unwritten no longer.
pub fn title_memory_written(section: usize, offset: u64, len: usize) {
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    {
        // Every view of the bytes is opened here, so no other view of them
        // has to be looked for.
        let mut g = guards();
        for &(at, n) in &ranges {
            open_here(&mut g, at, at.saturating_add(n));
        }
    }
    for (at, n) in ranges {
        // A placed texture there changes with no descriptor written.
        crate::descriptor::host::remapped(at, n);
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// Only with `VICTORY_MAPPING_RANGES=0` ([`by_range`]); otherwise reads and
/// changes hold the bytes they touch ([`SPANS`]).
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether a read in place and a change of mapping keep apart only when
/// their bytes meet ([`SPANS`]), not whatever memory each touches
/// ([`MAPPING`]); `VICTORY_MAPPING_RANGES=0` holds all of title memory for
/// each, as before. On by default.
///
/// With one lock over all of title memory each texture refresh waited for
/// every change of mapping anywhere, and each change for the refresh under
/// way. While the world streams in the title changes mappings 650-1,100
/// times a frame, many a page at a time (`XMemMapPhysicalPagesScatter`
/// maps each page on its own). With the one lock the refreshes waited 203
/// ms of an 801 ms frame for title memory to hold still; holding only the
/// bytes touched, 0 ms of a 486 ms frame (medians, a pair of runs).
pub(crate) fn by_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAPPING_RANGES").is_none_or(|v| v != "0"))
}

/// Reads of title memory in place under way, and changes of its mapping,
/// by the bytes they touch ([`by_range`]).
struct Spans {
    /// To tell them apart.
    next: u64,
    /// Reads under way: number, start and end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes: number, start, end, and whether under way (else waiting for
    /// the reads over its bytes to end; no read begins over them meanwhile,
    /// so that one read after another cannot keep a change waiting).
    changes: Vec<(u64, usize, usize, bool)>,
    /// Threads waiting on [`SPANS_LEFT`].
    waiting: usize,
}

static SPANS: Mutex<Spans> = Mutex::new(Spans {
    next: 0,
    reads: Vec::new(),
    changes: Vec::new(),
    waiting: 0,
});

/// Told when a read or a change ends while a thread waits.
static SPANS_LEFT: std::sync::Condvar = std::sync::Condvar::new();

fn spans() -> std::sync::MutexGuard<'static, Spans> {
    SPANS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until a read or a change ends.
fn spans_wait(mut s: std::sync::MutexGuard<'static, Spans>) -> std::sync::MutexGuard<'static, Spans> {
    s.waiting += 1;
    let mut s = SPANS_LEFT.wait(s).unwrap_or_else(|e| e.into_inner());
    s.waiting -= 1;
    s
}

/// Whether `[a, b)` and `[c, d)` share a byte.
fn meet(a: usize, b: usize, c: usize, d: usize) -> bool {
    a < d && c < b
}

impl Spans {
    fn read(&mut self, at: usize, end: usize) -> Span {
        self.next += 1;
        self.reads.push((self.next, at, end));
        Span(self.next, false)
    }
}

/// A read's or a change's place in [`SPANS`] (its number, and whether a
/// change), given up when dropped.
struct Span(u64, bool);

impl Drop for Span {
    fn drop(&mut self) {
        let mut s = spans();
        if self.1 {
            s.changes.retain(|c| c.0 != self.0);
        } else if let Some(i) = s.reads.iter().position(|r| r.0 == self.0) {
            s.reads.swap_remove(i);
        }
        if s.waiting > 0 {
            SPANS_LEFT.notify_all();
        }
    }
}

/// What a read or a change of mapping holds: its bytes ([`by_range`]) or
/// the lock over all of title memory.
enum Held<G> {
    All(#[allow(dead_code)] G),
    Bytes(#[allow(dead_code)] Span),
}

/// Since the last frame-rate line: changes of title memory's mapping, the
/// microseconds they waited for reads in place, and those they were under
/// way.
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    _held: Held<std::sync::RwLockWriteGuard<'static, ()>>,
    since: std::time::Instant,
}

impl Drop for Remapping {
    fn drop(&mut self) {
        REMAPS[2].fetch_add(self.since.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory `[at, at + len)`: until what it returns is dropped, no read of
/// those bytes in place is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the unmapping
/// and the mapping again, over all the bytes whose mapping it changes.
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    use std::sync::atomic::Ordering::Relaxed;
    let started = std::time::Instant::now();
    let held = if by_range() {
        let end = at.saturating_add(len);
        let mut s = spans();
        s.next += 1;
        let id = s.next;
        s.changes.push((id, at, end, false));
        while s.reads.iter().any(|r| meet(r.1, r.2, at, end))
            || s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end))
        {
            s = spans_wait(s);
        }
        if let Some(c) = s.changes.iter_mut().find(|c| c.0 == id) {
            c.3 = true;
        }
        Held::Bytes(Span(id, true))
    } else {
        Held::All(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    REMAPS[0].fetch_add(1, Relaxed);
    REMAPS[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    Remapping {
        _held: held,
        since: std::time::Instant::now(),
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Held<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping of those bytes, so
/// memory found mapped stays mapped. Not to be taken twice by one thread,
/// nor held across anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    if !by_range() {
        return Reading(Held::All(MAPPING.read().unwrap_or_else(|e| e.into_inner())));
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        s = spans_wait(s);
    }
    Reading(Held::Bytes(s.read(at, end)))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping of those bytes (or waits to), for a caller
/// that has another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    if !by_range() {
        return match MAPPING.try_read() {
            Ok(guard) => Some(Reading(Held::All(guard))),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(Held::All(e.into_inner()))),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return None;
    }
    Some(Reading(Held::Bytes(s.read(at, end))))
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    // Host copies of placed textures there have nothing left to come from:
    // let go of those nothing else holds now, rather than seconds later.
    crate::resource::evict_placed_in(at as u64, len as u64);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range.clone()).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    // A write through the kernel announced there wrote what the memory
    // showed before (or failed): the new memory may be guarded at once.
    g.kernel.retain(|k| k.end <= range.start || k.first >= range.end);
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        let open = writing(view + 100, 13);
        assert!(read(&mut file));
        drop(open);
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        if super::guard_aliases() < 2 {
            return; // VICTORY_GUARD_ALIASES: other views are not guarded
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        if super::guard_aliases() < 2 {
            return; // VICTORY_GUARD_ALIASES: other views are not guarded
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Blocks a write through the kernel was announced for stay writable
    /// while it is under way: a guard check over them takes them as written
    /// and leaves them so, beside blocks it guards again. Once the write is
    /// done, the next check takes them once more and guards them again.
    #[test]
    fn blocks_announced_to_the_kernel_stay_writable_while_it_writes() {
        const SIZE: usize = 3 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        let pages = |block: usize| (view + block * BLOCK) / PAGE;
        let middle = (view + BLOCK) / BLOCK;
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1), pages(2)]));
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        // A file read into the middle block is announced; then a guard
        // check of the view runs while the read is under way.
        let read = writing(view + BLOCK, BLOCK);
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(1)]));
        // Still writable: the kernel's write would succeed (a write of
        // ours here would fault if it were not).
        assert!(!guards().protected.contains(&middle), "made read-only under the read");
        assert!(guards().protected.contains(&(view / BLOCK)));
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(1)]), "taken as written while it is open");
        // The read is done: taken once more, then guarded again (unless the
        // switch keeps every announced write open for a while).
        drop(read);
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(1)]), "what the read wrote is taken");
        if !kernel_open_long() {
            assert!(guards().protected.contains(&middle), "guarded again after the read");
            assert_eq!(guard(view, view + SIZE), Some(Vec::new()), "not taken again once guarded");
        }
        drop_views(section, &[view, other], SIZE);
    }

    /// A write through the kernel that goes on after its call (an
    /// overlapped read) keeps its blocks writable for a while after the
    /// announcement is let go.
    #[test]
    fn a_pending_kernel_write_stays_open_after_its_call() {
        const SIZE: usize = 2 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        let pages = |block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]));
        writing(view, BLOCK).pending();
        for _ in 0..2 {
            assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]), "taken as written while it may be writing");
            assert!(!guards().protected.contains(&(view / BLOCK)), "made read-only under the read");
        }
        // Remapping the memory ends it.
        drop_views(section, &[view, other], SIZE);
        assert!(!guards().kernel.iter().any(|k| k.first < (view + SIZE) / BLOCK && k.end > view / BLOCK));
    }

    /// Blocks in order make runs of consecutive ones.
    #[test]
    fn blocks_make_runs() {
        assert_eq!(super::block_runs(&[]), Vec::new());
        assert_eq!(super::block_runs(&[3]), vec![(3, 4)]);
        assert_eq!(super::block_runs(&[3, 4, 5, 9, 10, 12]), vec![(3, 6), (9, 11), (12, 13)]);
    }

    /// A view made read-only because another view of its pages was guarded
    /// is not taken for clean: its own first check reports its pages
    /// written (its bytes were never taken), then they are clean.
    #[test]
    fn a_view_guarded_through_another_is_not_clean() {
        if super::guard_aliases() < 2 {
            return; // VICTORY_GUARD_ALIASES: other views are not guarded
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(guarded, 0), pages(guarded, 1)]));
        // The other view is read-only now, but nothing took its bytes.
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Pages shown at more addresses than `guard_aliases` allows (a filler
    /// the runtime maps everywhere) keep their other views as they were:
    /// a write through one of them is not seen at the guarded view.
    #[test]
    fn pages_shown_at_many_places_leave_their_other_views_alone() {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let most = super::guard_aliases();
        if most == 0 || most > 32 {
            return;
        }
        const SIZE: usize = BLOCK;
        // SAFETY: a fresh pagefile section mapped `most + 2` times.
        let (section, views) = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views: Vec<usize> = (0..most + 2)
                .map(|_| {
                    let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
                    assert_ne!(at, 0);
                    crate::inplace::title_memory_view(at, SIZE, section as usize, 0);
                    at
                })
                .collect();
            (section as usize, views)
        };
        let guarded = views[0];
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![guarded / PAGE]));
        // Not made read-only: this write does not fault, and the guarded
        // view does not learn of it (the texture hash is the net there).
        // SAFETY: a writable view of the section.
        unsafe { ((views[most + 1] + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &views, SIZE);
    }

    /// The runtime's own write into physical pages (`xmem`'s
    /// `write_physical`, DirectStorage's page destinations) writes through
    /// a view of its own: told here, the guarded view's next check reports
    /// the pages written.
    #[test]
    fn a_runtime_write_into_physical_pages_is_reported_at_the_views() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, _unused) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // A page of the section, at its offset in it, as `xmem` tells it.
        title_memory_written(section, BLOCK as u64, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        title_memory_written(section, (BLOCK / 2) as u64, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, _unused], SIZE);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{PAGE, by_range, title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses only these tests hold (nothing is read there).
    const HERE: usize = 0x7e00_0000_0000;

    /// A change of mapping waits for a read in place of its bytes that is
    /// under way, and such a read waits for the change.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(HERE + PAGE / 2, PAGE);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(HERE + PAGE, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(HERE, 2 * PAGE);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(HERE + PAGE + 16, 16);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes whose bytes do not meet do not wait for each other,
    /// and a change waiting for a read keeps new reads of its bytes out (but
    /// not of others).
    #[test]
    fn reads_and_remappings_of_other_bytes_go_on() {
        if !by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let at = HERE + 16 * PAGE;
        let reading = title_memory_reading(at, PAGE);
        // Another page changes while the read is under way, and another
        // read goes on while that change is.
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let changing = title_memory_remapping(at + PAGE, PAGE);
            let (read, was_read) = mpsc::channel();
            std::thread::spawn(move || {
                let _reading = title_memory_reading(at + 2 * PAGE, PAGE);
                read.send(()).unwrap();
            });
            let other = was_read.recv_timeout(Duration::from_secs(10)).is_ok();
            drop(changing);
            done.send(other).unwrap();
        });
        assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok(true), "kept apart though their bytes do not meet");
        // A change of the read's page waits, and keeps a new read of its
        // bytes out meanwhile; a read elsewhere is not kept out.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping(at, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        while !super::spans().changes.iter().any(|c| c.1 == at) {
            std::thread::yield_now();
        }
        assert!(super::title_memory_reading_now(at + 8, 8).is_none(), "a read began over a change that waits");
        assert!(super::title_memory_reading_now(at + 3 * PAGE, 8).is_some());
        std::thread::sleep(Duration::from_millis(50));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod wrote_tests {
    use super::{PAGE, ask_once, ask_once_end, asked, writes, wrote};

    #[test]
    fn wrote_keeps_ahead_asked_at_or_above_written_timestamps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        // SAFETY: private write-watched 64 KiB page owned by this test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);

        ask_once(&[(p, p + PAGE)]);
        wrote(p, PAGE);

        let (now, written) = writes(p, p + PAGE, false).expect("write-watched page");
        let (asked_now, asked_written) = asked(p, p + PAGE).expect("in ask_once range");
        ask_once_end();

        // SAFETY: freeing our private reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };

        assert_eq!(now, asked_now);
        let page = p / PAGE;
        let &(_, w) = written
            .iter()
            .find(|&&(pg, _)| pg == page)
            .expect("page recorded by wrote()");
        assert_eq!(asked_written, written);
        assert!(
            now >= w,
            "writes() returned now={now} < written timestamp w={w} after wrote()"
        );
    }
}

#[cfg(test)]
mod piece_tests {
    use super::{ALL_PIECES, KERNEL_PAGE, PAGE, PIECE_HISTORY, Pieces, by_page, pieces_after, pieces_of};

    /// The kernel pages a byte range covers, cut to the page.
    #[test]
    fn a_range_covers_the_kernel_pages_it_touches() {
        assert_eq!(pieces_of(3, 3 * PAGE, 1), 1);
        assert_eq!(pieces_of(3, 3 * PAGE + KERNEL_PAGE - 1, 2), 0b11);
        assert_eq!(pieces_of(3, 3 * PAGE - 100, 200), 1, "begins in the page before");
        assert_eq!(pieces_of(3, 2 * PAGE, 3 * PAGE), ALL_PIECES);
        assert_eq!(pieces_of(3, 4 * PAGE, 10), 0, "outside");
        assert_eq!(pieces_of(3, 4 * PAGE - 1, usize::MAX), 1 << 15, "to the end of the address space");
    }

    /// The kernel's answer, in order, folds into one mask per page.
    #[test]
    fn the_kernels_pages_fold_by_page() {
        let found = [2 * PAGE, 2 * PAGE + 5 * KERNEL_PAGE, 3 * PAGE + 15 * KERNEL_PAGE, 9 * PAGE + KERNEL_PAGE];
        assert_eq!(by_page(&found), vec![(2, 1 | 1 << 5), (3, 1 << 15), (9, 1 << 1)]);
        assert!(by_page(&[]).is_empty());
    }

    /// A copy needs the pieces of the askings after the one it took the
    /// page at; the whole page when the history does not reach back so far,
    /// when it never took the page, or when the page was written without
    /// pieces.
    #[test]
    fn the_history_says_what_a_copy_needs_or_gives_up() {
        let mut j = super::Journal {
            asked: 0,
            written: Default::default(),
            found: Vec::new(),
            ahead: Vec::new(),
            ahead_asked: 0,
            kept: false,
            pieces: Default::default(),
        };
        super::note_pieces(&mut j, 7, 10, 1);
        super::note_pieces(&mut j, 7, 12, 1 << 3);
        super::note_pieces(&mut j, 7, 12, 1 << 4);
        // Out of order: counted with the latest.
        super::note_pieces(&mut j, 7, 11, 1 << 9);
        let p = j.pieces.get(&7).copied();
        assert_eq!(pieces_after(p.as_ref(), 12, 9), 1 | 1 << 3 | 1 << 4 | 1 << 9);
        assert_eq!(pieces_after(p.as_ref(), 12, 10), 1 << 3 | 1 << 4 | 1 << 9);
        assert_eq!(pieces_after(p.as_ref(), 12, 12), 0, "taken after its last write");
        assert_eq!(pieces_after(p.as_ref(), 12, 0), ALL_PIECES, "never taken");
        assert_eq!(pieces_after(p.as_ref(), 13, 10), ALL_PIECES, "a write the history does not hold");
        assert_eq!(pieces_after(None, 12, 10), ALL_PIECES, "no history");
        // More askings than the history holds: the oldest goes.
        for asking in 13..13 + PIECE_HISTORY as u64 - 1 {
            super::note_pieces(&mut j, 7, asking, 1 << 6);
        }
        let p: Pieces = j.pieces[&7];
        assert_eq!(p.lost, 10);
        assert_eq!(pieces_after(Some(&p), 15, 9), ALL_PIECES, "from before what was lost");
        assert_eq!(pieces_after(Some(&p), 15, 10), 1 << 3 | 1 << 4 | 1 << 9 | 1 << 6);
    }
}
