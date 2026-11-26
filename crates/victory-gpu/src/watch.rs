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
//!
//! With `VICTORY_LIGHT_REFRESH` a block that is read-only by our hand,
//! clean, and whose earlier writes are all in the journal has a bit
//! ([`PUBLISHED`]): a range whose blocks all have theirs is answered from
//! the journal alone, with no lock of the guards and no kernel call
//! ([`light`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call. The flag: asked of
    /// the guards (view memory, with [`views_once`]), not of the kernel;
    /// such a range answers only an asker that would ask the guards.
    ahead: Vec<(usize, usize, bool)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
    /// With [`pieces`]: per page, which of its kernel pages the last
    /// askings found written ([`Pieces`]).
    pieces: BTreeMap<usize, Pieces>,
    /// With [`unlocked`]: the askings whose kernel call or guards' check is
    /// under way with the journal's lock let go: `(first, last, asking)`.
    /// An answer of a range one of them overlaps, under a number above
    /// its, waits for it ([`settle`]).
    flying: Vec<(usize, usize, u64)>,
    /// Blocks [`publish`] passed over because an asking in flight covered
    /// them: looked at again when an asking lands ([`land`]).
    unpublished: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    pieces: BTreeMap::new(),
    flying: Vec::new(),
    unpublished: Vec::new(),
});

/// Pages of a piece of [`Stamps`]: 4 GiB of title addresses.
const STAMP_PAGES: usize = 1 << 16;
/// Pieces of [`Stamps`]: 128 TiB, what a process has.
const STAMP_PIECES: usize = 1 << 15;

/// Per 64 KiB page, the asking that last found it written (0: never), as
/// `Journal::written` holds it, in a flat table read with no lock
/// ([`stamps_flat`]); in pieces made when first written and never let go.
/// Written only under the journal's lock, by [`note_written`], always
/// raising (a stamp never falls).
struct Stamps([std::sync::atomic::AtomicPtr<StampsPiece>; STAMP_PIECES]);

type StampsPiece = [std::sync::atomic::AtomicU64; STAMP_PAGES];

impl Stamps {
    const fn new() -> Stamps {
        Stamps([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; STAMP_PIECES])
    }

    fn piece(&self, page: usize) -> Option<&StampsPiece> {
        let piece = self.0.get(page / STAMP_PAGES)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `raise`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// The page's stamp (0 for none).
    #[cfg(test)]
    fn get(&self, page: usize) -> u64 {
        self.piece(page).map_or(0, |p| p[page % STAMP_PAGES].load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Raises the page's stamp to `stamp` (never lowers it).
    fn raise(&self, page: usize, stamp: u64) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let Some(place) = self.0.get(page / STAMP_PAGES) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let made: *mut StampsPiece = Box::into_raw(
                (0..STAMP_PAGES)
                    .map(|_| std::sync::atomic::AtomicU64::new(0))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("a piece's length")),
            );
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        if let Some(piece) = self.piece(page) {
            piece[page % STAMP_PAGES].fetch_max(stamp, SeqCst);
        }
    }

    /// Tests: the stamps of `[first, last)` forgotten ([`forget_pages`]).
    #[cfg(test)]
    fn forget(&self, first: usize, last: usize) {
        for page in first..last {
            if let Some(piece) = self.piece(page) {
                piece[page % STAMP_PAGES].store(0, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    /// The pages of `[first, last)` that have a stamp, with it, in order:
    /// what `Journal::written.range` gives.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        let mut out = Vec::new();
        let mut page = first;
        while page < last {
            let Some(piece) = self.piece(page) else {
                page = (page / STAMP_PAGES + 1) * STAMP_PAGES;
                continue;
            };
            let until = ((page / STAMP_PAGES + 1) * STAMP_PAGES).min(last);
            for at in page..until {
                let stamp = piece[at % STAMP_PAGES].load(std::sync::atomic::Ordering::SeqCst);
                if stamp != 0 {
                    out.push((at, stamp));
                }
            }
            page = until;
        }
        out
    }
}

static STAMPS: Stamps = Stamps::new();

/// Whether the journal's "last written" stamps are kept in a flat table
/// as well, read with no lock (`VICTORY_STAMPS_FLAT=1`; off by default):
/// an asking the light refresh's bits answer ([`writes`]) and a refresh
/// they show has nothing new ([`nothing_new`]) then take no lock at all
/// (they took the journal's twice a refresh), and every answer reads the
/// table. The journal's lock was taken some 45,000 times a frame on the
/// Series S scene (all22 with today's set).
///
/// Why an answer read with no lock is right: a block's bit is set only
/// after the stamps of what the asking that cleaned it found are in the
/// table ([`publish`] after [`note_written`], both sequentially
/// consistent), and cleared before the block can be written; a reader
/// looks at the bits, then reads the stamps. Every stamp is noted under a
/// number taken after the look that found the write (the kernel's call,
/// the guards' check), so an answer numbered in between sees no write it
/// missed noted below its number.
fn stamps_flat() -> bool {
    #[cfg(test)]
    if STAMPS_FLAT_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_STAMPS_FLAT").is_some_and(|v| v == "1") || stamps_check()
    })
}

/// Tests: [`stamps_flat`] on, whatever the switch says.
#[cfg(test)]
pub(crate) static STAMPS_FLAT_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `VICTORY_STAMPS_CHECK=1` (sets [`stamps_flat`]): every answer is made
/// from the journal's map as well and compared; an answer read with no
/// lock is compared with what the journal holds a moment later (a stamp
/// there may have risen since, never fallen). Differences counted on the
/// journal-lock line and warned (the first 16, then at powers of two).
fn stamps_check() -> bool {
    #[cfg(test)]
    if STAMPS_FLAT_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STAMPS_CHECK").is_some_and(|v| v == "1"))
}

/// [`stamps_check`]: answers compared, differences found; answers made
/// with no lock.
static STAMP_CHECKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The pages of `[first, last)` as last written: from the flat table with
/// [`stamps_flat`] (compared with the map with [`stamps_check`]), else from
/// the map.
fn answer(j: &Journal, first: usize, last: usize) -> Vec<(usize, u64)> {
    if !stamps_flat() {
        return map_answer(j, first, last);
    }
    let flat = STAMPS.range(first, last);
    if stamps_check() {
        let map = map_answer(j, first, last);
        stamp_compared(first, map != flat);
    }
    flat
}

/// [`answer`] from the journal's map.
fn map_answer(j: &Journal, first: usize, last: usize) -> Vec<(usize, u64)> {
    j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
}

fn stamp_compared(first: usize, differs: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    STAMP_CHECKS[0].fetch_add(1, Relaxed);
    if differs {
        let n = STAMP_CHECKS[1].fetch_add(1, Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            tracing::warn!(
                n,
                address = format_args!("{:#x}", first * PAGE),
                "the flat stamps differ from the journal's (VICTORY_STAMPS_CHECK)"
            );
        }
    }
}

/// [`stamps_check`] of an answer read with no lock: each of its stamps is
/// at most what the journal holds now, and the journal holds none the
/// table did not show but above it.
fn stamps_checked_later(first: usize, last: usize, flat: &[(usize, u64)]) {
    if !stamps_check() {
        return;
    }
    let j = journal();
    let map = map_answer(&j, first, last);
    drop(j);
    let fell = flat.iter().any(|&(page, stamp)| map.iter().find(|m| m.0 == page).is_none_or(|m| m.1 < stamp));
    stamp_compared(first, fell);
}

/// Numbers the askings; 0 is "never". An atomic, so that an answer by the
/// light refresh's bits takes its number with no lock ([`stamps_flat`]);
/// every other number is still taken under the journal's lock.
static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Journal {
    /// The latest number given.
    fn asked(&self) -> u64 {
        ASKED.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// A new number, above every one given so far.
    fn bump(&mut self) -> u64 {
        ASKED.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
    }
}

/// Told when an asking in [`Journal::flying`] has noted what it found.
static LANDED_ASKING: std::sync::Condvar = std::sync::Condvar::new();

/// Whether an asking makes its kernel call and asks the guards with the
/// journal's lock let go (`VICTORY_JOURNAL_UNLOCKED=1`; off by default).
///
/// The journal's one lock was held over each asking's `GetWriteWatch`
/// (with reset) and, for view memory, over the guards' check with its
/// `VirtualProtect` and `VirtualQuery` calls. A sampling profile of the
/// Series S house scene (vf-bis-prof-s2): the submitting thread waited
/// for that lock in `writes <- Shadow::update_free` for 22-40 % of its
/// waiting and spun in it for 4-6 % of its running, while the eight
/// recording ("wide") threads and the relay, asking about their own
/// copies, held it across their kernel calls (`ask_kernel` alone was
/// 14-17 % of the submitting thread's own running).
///
/// With the switch an asking takes its number under the lock and notes
/// itself in flight ([`Journal::flying`]), lets the lock go for the kernel
/// call and the guards, and takes it again to note what it found. What
/// keeps the answers right: a kernel asking resets the pages it reports,
/// so another asking of the same pages made meanwhile does not see them;
/// every answer of a range (by an asking, from the standing asking ahead,
/// [`asked`]) therefore waits until every asking in flight over it when
/// it reads the journal has noted what it found ([`settle`]); askings that
/// start later reset after it and note stamps above its number, which its
/// copy takes at its next refresh.
/// The journal never takes a page's stamp back to a lower number.
pub(crate) fn unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOURNAL_UNLOCKED").is_some_and(|v| v == "1"))
}

/// The journal's lock, the wait for it counted (diagnostics builds).
fn journal() -> std::sync::MutexGuard<'static, Journal> {
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        if let Ok(j) = JOURNAL.try_lock() {
            JOURNAL_STATS[0].fetch_add(1, Relaxed);
            return j;
        }
        let waiting = std::time::Instant::now();
        let j = waited(&JOURNAL, 1);
        JOURNAL_STATS[0].fetch_add(1, Relaxed);
        JOURNAL_STATS[1].fetch_add(1, Relaxed);
        JOURNAL_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Relaxed);
        j
    }
    #[cfg(not(feature = "diagnostics"))]
    waited(&JOURNAL, 1)
}

/// Since the last frame line, of the journal's lock as the askings take it
/// ([`journal`], [`settle`]): takes, takes that found it held, their
/// microseconds waiting; waits for an overlapping asking in flight, their
/// microseconds.
#[cfg(feature = "diagnostics")]
static JOURNAL_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// The frame line's account of the journal's lock, per frame, taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn journal_line(frames: f64) -> String {
    let f = frames.max(1.0);
    let [takes, held, wait_us, settles, settle_us] =
        [0, 1, 2, 3, 4].map(|i| JOURNAL_STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / f);
    format!(
        "the write watch's journal lock (VICTORY_JOURNAL_UNLOCKED {}): per frame {takes:.0} takes by askings, {held:.0} of \
         them found it held, {:.1} ms waiting for it (all threads); {settles:.0} waits for an overlapping asking in flight \
         {:.1} ms; flat stamps (VICTORY_STAMPS_FLAT {}): {:.0} answers with no lock, check {:.0} compared, {} differences \
         (all since the start)",
        if unlocked() { "on" } else { "off" },
        wait_us / 1e3,
        settle_us / 1e3,
        if stamps_flat() { "on" } else { "off" },
        STAMP_CHECKS[2].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / f,
        STAMP_CHECKS[0].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / f,
        STAMP_CHECKS[1].load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// The journal, once no asking that was in flight over pages `[first,
/// last)` when this was called is left ([`unlocked`]).
///
/// Every one of them, whatever its number: an asking numbered above the
/// answer that reached the kernel first has reset pages whose writes the
/// answer's own asking then did not see (found by a review of a2b96a048,
/// which waited only for those numbered below). Askings that start after
/// this call reset after the answer's own asking, and are not waited for
/// (no answer waits for ever behind new askings).
fn settle(
    mut j: std::sync::MutexGuard<'static, Journal>,
    first: usize,
    last: usize,
) -> std::sync::MutexGuard<'static, Journal> {
    let before = j.asked();
    let overlapping = |j: &Journal| j.flying.iter().any(|&(a, b, n)| n <= before && a < last && first < b);
    if j.flying.is_empty() || !overlapping(&j) {
        return j;
    }
    #[cfg(feature = "diagnostics")]
    let waiting = std::time::Instant::now();
    while overlapping(&j) {
        j = LANDED_ASKING.wait(j).unwrap_or_else(|e| e.into_inner());
    }
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        JOURNAL_STATS[3].fetch_add(1, Relaxed);
        JOURNAL_STATS[4].fetch_add(waiting.elapsed().as_micros() as u64, Relaxed);
    }
    j
}

/// The asking numbered `asking` has landed ([`unlocked`]): out of flight,
/// the askings waiting for it told, and the blocks [`publish`] passed over
/// because of it looked at again (the light refresh's bits of those still
/// clean; the journal holds what this asking found, noted or about to be
/// under the same hold, and [`publish`] passes over any that another
/// asking in flight still covers).
fn land(j: &mut Journal, asking: u64) {
    j.flying.retain(|&(_, _, n)| n != asking);
    LANDED_ASKING.notify_all();
    if !j.unpublished.is_empty() && (light_kept() || cfg!(test)) {
        let mut again = std::mem::take(&mut j.unpublished);
        again.sort_unstable();
        again.dedup();
        publish(j, &again);
    }
}

/// Tests: called once by the next unlocked asking from the page given,
/// right after it is in flight, before its kernel call or guards' check.
#[cfg(test)]
static AFTER_REGISTER: Mutex<Option<(usize, fn())>> = Mutex::new(None);

thread_local! {
    /// [`unlocked`]: room for a kernel answer asked with the journal's
    /// lock let go.
    static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

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
///
/// A stamp only rises: a page some asking numbered above `asking` noted
/// already keeps that asking's number (its pieces take these with it).
fn note_written(j: &mut Journal, page: usize, asking: u64, mask: u16) {
    let stamp = j.written.entry(page).or_insert(asking);
    *stamp = (*stamp).max(asking);
    // (Kept in tests whatever the switch says: tests turn it on for a while.)
    if stamps_flat() || cfg!(test) {
        STAMPS.raise(page, *stamp);
    }
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

/// Since the last frame line, the write watch's kernel calls by kind
/// ([`kernel_line`]): `GetWriteWatch` answered, failed (memory that is not
/// write-watched: view memory, or a range across reservations), not made
/// (view memory the guards know, [`skip_views`]); hole checks
/// (`VirtualQuery`, [`uncommitted_end`]) made and answered from a cache,
/// placeholder questions.
static KERNEL_CALLS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn kernel_count(kind: usize) {
    KERNEL_CALLS[kind].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The frame line's account of the write watch's kernel calls, per frame,
/// taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn kernel_line(frames: f64) -> String {
    let f = frames.max(1.0);
    let [answered, failed, skipped, holes, cached, placeholders] =
        [0, 1, 2, 3, 4, 5].map(|i| KERNEL_CALLS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / f);
    format!(
        "the write watch's kernel calls per frame: GetWriteWatch {answered:.0} answered, {failed:.0} failed (not \
         write-watched), {skipped:.0} not made (view memory the guards know, VICTORY_ASK_SKIP_VIEWS {}); hole checks \
         (VirtualQuery) {holes:.0} made, {cached:.0} answered from the cache (VICTORY_HOLES_KNOWN {}), {placeholders:.0}          placeholder questions",
        if skip_views() { "on" } else { "off" },
        if holes_known() { "on" } else { "off" },
    )
}

/// Whether the write watch leaves out of its kernel askings the ranges
/// whose first block the guards hold (read-only by our hand, or noted
/// unmapped: view memory, which has no write watch), at a submission's
/// one asking ([`ask_ahead`], [`ask_lone`], [`ask_views`]) and at every
/// asking ([`writes`], as `VICTORY_GUARD_FIRST`) (`VICTORY_ASK_SKIP_VIEWS=1`;
/// off by default).
///
/// The submission's one asking asks the kernel about the span of every
/// copy it refreshes, view memory too: such an asking can only fail, and
/// a run of them asked together (8 MiB apart at most) fails and is asked
/// again by halves, down to single ranges, each call walking the page
/// tables of its range. A profile of the Series S house scene
/// (vf-bis-all22-s-wft-prof) has `ask_kernel` at 19 % of the submitting
/// thread's running, while the askings it counts, the answered ones, are
/// 9,855 a frame. What the guards answer is what they answered after the
/// failed call.
pub(crate) fn skip_views() -> bool {
    #[cfg(test)]
    if SKIP_VIEWS_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SKIP_VIEWS").is_some_and(|v| v == "1"))
}

/// Tests: [`skip_views`] on, whatever the switch says.
#[cfg(test)]
pub(crate) static SKIP_VIEWS_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// With [`skip_views`]: for each of `firsts` (page numbers), whether the
/// guards hold it; one look under their lock.
fn views_held(firsts: impl Iterator<Item = usize>) -> Vec<bool> {
    let g = guards();
    firsts.map(|first| g.protected.contains(&first) || g.absent.contains(&first)).collect()
}

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
    // View memory the guards know: no kernel asking, which would fail.
    if skip_views() && !pieces.is_empty() {
        let held = views_held(pieces.iter().map(|p| p.0));
        let before = pieces.len();
        let mut i = 0;
        pieces.retain(|_| {
            i += 1;
            !held[i - 1]
        });
        for _ in pieces.len()..before {
            kernel_count(2);
        }
    }
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
    j.ahead_asked = j.bump();
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
    ask_lone(ranges);
}

/// [`ask_once`] after [`ask_ahead`]: the ranges that lie alone, asked under
/// the standing asking.
fn ask_lone(ranges: &[(usize, usize)]) {
    // View memory the guards know is left to the guards ([`skip_views`]).
    let held = if skip_views() { views_held(ranges.iter().map(|r| r.0 / PAGE)) } else { Vec::new() };
    #[cfg(feature = "diagnostics")]
    let _who = asking_for(1);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.ahead_asked = j.bump();
    }
    // Other askings were made since `ask_ahead` numbered the standing
    // asking (the journal's lock was let go between): what is noted from
    // here on goes under a number above theirs, else a page one of them
    // took, written again since, would be noted below what its copy took
    // and never copied again (review finding, test/watch5). The answers
    // of the standing asking take the new number, as after [`wrote`].
    if j.ahead_asked < j.asked() {
        j.ahead_asked = j.bump();
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for (i, &(start, end)) in ranges.iter().enumerate() {
        if start >= end {
            continue;
        }
        if held.get(i).copied().unwrap_or(false) {
            kernel_count(2);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !ahead_covers(j, first, last, true) && ask_kernel(j, first, last) {
            j.ahead.push((first, last, false));
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

/// Whether pages `[first, last)` lie inside one range of the standing
/// ahead asking (with `views` false, one asked of the kernel).
fn ahead_covers(j: &Journal, first: usize, last: usize, views: bool) -> bool {
    j.ahead.iter().any(|&(a, b, view)| a <= first && last <= b && (views || !view))
}

/// Whether the submission's one asking ([`once`]) takes in the view memory
/// its lists refresh as well (`VICTORY_ASK_ONCE_VIEWS=1`; off by default).
///
/// View memory (xmem's physical-page views) has no write watch: each of
/// its refreshes failed to ask the kernel and asked the guards, once per
/// flush, some 70 flushes a submission (here some 17,000-20,000 askings a
/// frame by one copy's refresh, 70-100 ms of threads' time under the
/// journal's lock; making each cheaper, VICTORY_GUARD_FIRST, did not move
/// the frame). With the switch each such range is guarded once when the
/// submission starts ([`ask_once_views`]) and the journal answers its
/// refreshes, as it does for write-watched memory: what the title's other
/// threads write while the submission is prepared is seen by the next
/// submission's asking (the write faults and opens the block, as it sets
/// the kernel's bit in watched memory); our own writes are told to the
/// journal or end the asking, as for watched memory ([`once`]); a remap
/// of view memory takes its ranges out of the asking ([`views_remapped`]).
///
/// From fxceless/victory-fork PR #995 (tunczzz), `ask_once_views`.
pub(crate) fn views_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE_VIEWS").is_some_and(|v| v == "1"))
}

/// With [`views_once`], right after [`ask_once`]: the ranges the
/// submission refreshes (each as one refresh asks, cut to its copy) that
/// the asking did not take in are asked now as [`writes`] would ask them:
/// the kernel, else the guards of view memory; what either says goes into
/// the journal under the standing asking, and the refreshes inside them
/// answer from it. What neither takes is left to `writes`.
pub(crate) fn ask_once_views(ranges: &[(usize, usize)]) {
    if once() && views_once() && !ranges.is_empty() {
        ask_views(ranges);
    }
}

/// [`ask_once_views`], whatever the switches say.
fn ask_views(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    // In order of start, the longest first: a range inside one asked
    // before it is not asked again.
    pieces.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    pieces.dedup();
    #[cfg(feature = "diagnostics")]
    let _who = asking_for(1);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // `ask_once` let its asking go if it held nothing: this one stands
    // under a number of its own.
    if !j.kept {
        j.ahead_asked = j.bump();
        j.ahead.clear();
    }
    // Standing already, under a number other askings may have passed
    // since: noted under a new one (as in [`ask_lone`]).
    if j.ahead_asked < j.asked() {
        j.ahead_asked = j.bump();
    }
    let mut reach = 0;
    // Ranges the guards know to be views (their first block is guarded or
    // absent): asked below, joined where they overlap or touch.
    let mut known = Vec::new();
    for (first, last) in pieces {
        if last <= reach {
            continue;
        }
        if ahead_covers(j, first, last, true) {
            continue;
        }
        // With `VICTORY_GUARD_FIRST=1`, as in [`writes`]: a range whose
        // first block the guards hold cannot be write-watched, no kernel
        // asking first.
        let held = (guard_first() || skip_views()) && guards_know(first);
        if held {
            kernel_count(2);
        }
        if !held && ask_kernel(j, first, last) {
            j.ahead.push((first, last, false));
            reach = reach.max(last);
            continue;
        }
        if held || guards_know(first) {
            known.push((first, last));
            continue;
        }
        // The guards decide whether it is a view, as for `writes`.
        if guard_ahead(j, first, last) {
            reach = reach.max(last);
        }
    }
    // A submission refreshes many overlapping ranges of the same views:
    // their union is walked once, each block once. The guards say of the
    // union what they would say of its ranges one after another, all of
    // whose first blocks they know.
    let mut i = 0;
    while i < known.len() {
        let (first, mut last) = known[i];
        let mut k = i + 1;
        while k < known.len() && known[k].0 <= last {
            last = last.max(known[k].1);
            k += 1;
        }
        if !ahead_covers(j, first, last, true) {
            guard_ahead(j, first, last);
        }
        i = k;
    }
    j.kept = !j.ahead.is_empty();
}

/// [`ask_views`]: the guards asked about blocks `[first, last)`, what they
/// report noted under the standing asking; whether the range is a view
/// (and now asked ahead).
fn guard_ahead(j: &mut Journal, first: usize, last: usize) -> bool {
    #[cfg(feature = "diagnostics")]
    let guarding = std::time::Instant::now();
    let written = guard(first * PAGE, last * PAGE);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        let us = guarding.elapsed().as_micros() as u64;
        add(C::GuardUs, us);
        add(C::Watches, 1);
        add(C::WatchUs, us);
        slow(us);
    }
    let Some(written) = written else {
        return false;
    };
    // With [`stamps_flat`] an answer by the bits takes its number with no
    // lock, perhaps after the standing asking's: what the guards found
    // here goes under a number taken after their look, which the standing
    // asking takes too.
    if stamps_flat() && !written.is_empty() {
        j.ahead_asked = j.bump();
    }
    let now = j.ahead_asked;
    // A guard's block is a page: all of it.
    for &page in &written {
        note_written(j, page, now, ALL_PIECES);
    }
    // The light refresh: as in [`writes`].
    if light_kept() {
        publish(j, &written);
    }
    j.ahead.push((first, last, true));
    true
}

/// With [`views_once`]: view memory `[at, at + len)` is mapped anew while a
/// submission's asking stands. The ranges of view memory the asking holds
/// over it are taken out of it, and the refreshes there ask the guards
/// themselves (which know of the remap), as without the switch.
fn views_remapped(at: usize, len: usize) {
    if !views_once() || len == 0 {
        return;
    }
    forget_views_ahead(at, len);
}

/// [`views_remapped`], whatever the switch says.
fn forget_views_ahead(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.ahead.retain(|&(a, b, view)| !(view && a < last && first < b));
        j.kept = !j.ahead.is_empty();
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
    STAMPS.forget(first, last);
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
    let now = j.bump();
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
        j.ahead.push((first, last, false));
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
    kernel_count(usize::from(failed));
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
    let j = journal();
    if !ahead_covers(&j, first, last, true) {
        return None;
    }
    let now = j.ahead_asked;
    let j = settle(j, first, last);
    Some((now, answer(&j, first, last)))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    // The light refresh: its bits kept, and asked or not, as the switch
    // says now.
    let light = if light_kept() { 1 + u8::from(light()) } else { 0 };
    if guarded && light_mode() == 1 {
        light_said();
    }
    // Its self-check: this asking goes to the guards even if the bits
    // vouch for the range, and what they find is compared.
    let check = light == 2 && guarded && (FORCE_CHECK.with(|f| f.replace(false)) || light_check_due());
    writes_as(start, end, guarded, light, check, unlocked())
}

/// [`writes`] with the light refresh as `light` says ([`light`]): 0, as
/// without the switch, no bit is set or asked; 1, the blocks the guards
/// leave clean get their bits and none is asked (the phase of `alt`
/// without); 2, a range whose blocks all have their bit is answered from
/// the journal. `check` ([`light_check`]): a range the bits vouch for is
/// asked of the guards anyway, and a page they find written whose bit
/// stood is counted a mismatch. `unlocked` ([`unlocked`]): the kernel
/// call and the guards' check are made with the journal's lock let go.
fn writes_as(
    start: usize,
    end: usize,
    guarded: bool,
    light: u8,
    check: bool,
    unlocked: bool,
) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // With [`stamps_flat`]: a range the bits vouch for is answered with no
    // lock (a new number, the table's stamps).
    if light == 2 && guarded && !check && stamps_flat() && first < last
        && PUBLISHED.all(first * PAGE / BLOCK, (last * PAGE).div_ceil(BLOCK))
    {
        light_count(0);
        STAMP_CHECKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = ASKED.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let flat = STAMPS.range(first, last);
        stamps_checked_later(first, last, &flat);
        return Some((now, flat));
    }
    let mut held = journal();
    // Asked ahead with the submission's other ranges: the journal has it.
    // View memory asked ahead ([`views_once`]) answers only an asker that
    // would ask the guards.
    if ahead_covers(&held, first, last, guarded) {
        let now = held.ahead_asked;
        let held = settle(held, first, last);
        return Some((
            now,
            answer(&held, first, last),
        ));
    }
    // (The light refresh's self-check asks with the lock held throughout:
    // its comparison needs no other asking between the look at the bits
    // and the guards' answer.)
    if unlocked && !check {
        return writes_unlocked(held, first, last, guarded, light);
    }
    let j = &mut *held;
    // The light refresh: every block of the range is read-only by our hand
    // and clean since an asking noted it ([`PUBLISHED`]). The guards would
    // find nothing written and this asking would note nothing: a number of
    // its own and the journal as it is are the answer, with the guards'
    // lock not taken and no kernel call.
    let vouched = guarded && light == 2 && published(j, first * PAGE, last * PAGE);
    if vouched && !check {
        light_count(0);
        let now = j.bump();
        return Some((
            now,
            answer(j, first, last),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // With `VICTORY_GUARD_FIRST=1`: view memory the guards hold already is
    // not asked of the kernel, whose answer there is a failure.
    let failed = guarded && (guard_first() || skip_views()) && guards_know(first) && {
        kernel_count(2);
        true
    } || {
        // SAFETY: the answer buffer holds `count` entries.
        let answer = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        kernel_count(usize::from(answer != 0));
        answer != 0
    };
    // With [`guard_outside`]: the guards' blocks to make read-only, once
    // the journal is let go.
    let mut arming = Arming(Vec::new());
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard_later(first * PAGE, last * PAGE, guard_outside());
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        let (written, later) = written?;
        arming = later;
        written
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
    let now = j.bump();
    // A guard's block is a page: all of it.
    let pages: Vec<(usize, u16)> =
        by_page(&j.found[..count]).into_iter().chain(guarded.iter().map(|&page| (page, ALL_PIECES))).collect();
    for (page, mask) in pages {
        note_written(j, page, now, mask);
    }
    // The light refresh: the guards were asked, and what they found is
    // noted. The blocks they made clean are known by their bits from now
    // on (the journal's lock is still held: no other asking came between).
    if failed && light != 0 {
        light_count(1);
        if vouched {
            light_compare(&guarded);
        }
        publish(j, &guarded);
    }
    // An asking in flight over the range ([`unlocked`]: this asking kept
    // the lock for the light refresh's self-check): it may have reset
    // pages this one's kernel call then did not see, so the answer waits
    // for it ([`settle`]), the blocks made read-only first (an asking in
    // flight may be waiting for them in the guards).
    let overlapping = j.flying.iter().any(|&(a, b, _)| a < last && first < b);
    if !overlapping && arming.0.is_empty() {
        let answer = (
            now,
            answer(j, first, last),
        );
        drop(held);
        return Some(answer);
    }
    // Blocks made read-only outside the locks ([`guard_outside`]): until
    // they are, a write to them lands with no fault, and an answer from the
    // standing asking given meanwhile, under a number [`wrote`] raised
    // above this asking's, may have taken a block before such a write and
    // be told nothing more. Once they stand, they are noted again under a
    // number above every one given so far (review finding, test/watch5),
    // and this asking answers under it (its copy reads them after now).
    let armed: Vec<usize> =
        arming.0.iter().flat_map(|&(from, to)| from * BLOCK / PAGE..to * BLOCK / PAGE).collect();
    drop(held);
    #[cfg(test)]
    if !armed.is_empty()
        && let Some(hook) = BEFORE_FINISH.with(|h| h.take())
    {
        hook();
    }
    arming.finish(false);
    let mut held = journal();
    if overlapping {
        held = settle(held, first, last);
    }
    let mut now = now;
    if !armed.is_empty() {
        let j = &mut *held;
        now = j.bump();
        for &page in &armed {
            note_written(j, page, now, ALL_PIECES);
        }
        if failed && light != 0 {
            publish(j, &armed);
        }
    }
    Some((now, answer(&held, first, last)))
}

/// [`writes_as`] with [`unlocked`], from where the journal could not answer
/// at once: the asking takes its number and notes itself in flight, makes
/// the kernel call (else asks the guards, for view memory) with the lock
/// let go, then notes what it found and answers once the askings in
/// flight before it that overlap its range have noted theirs.
fn writes_unlocked(
    mut held: std::sync::MutexGuard<'static, Journal>,
    first: usize,
    last: usize,
    guarded: bool,
    light: u8,
) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    if guarded && light == 2 && published(&held, first * PAGE, last * PAGE) {
        light_count(0);
        let now = held.bump();
        return Some((now, answer(&held, first, last)));
    }
    let now = held.bump();
    held.flying.push((first, last, now));
    drop(held);
    #[cfg(test)]
    {
        let mut hook = AFTER_REGISTER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((page, run)) = *hook
            && page == first
        {
            *hook = None;
            drop(hook);
            run();
        }
    }
    // The lock let go: the kernel, else the guards.
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let (found, viewed) = FOUND.with(|room| {
        let mut room = room.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        let size = count.max(room.len());
        room.resize(size, 0);
        let mut granularity = 0;
        // With `VICTORY_GUARD_FIRST=1`: view memory the guards hold already
        // is not asked of the kernel, whose answer there is a failure.
        let failed = guarded && (guard_first() || skip_views()) && guards_know(first) && {
            kernel_count(2);
            true
        } || {
            // SAFETY: the answer buffer holds `count` entries.
            let answer = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    (first * PAGE) as *const _,
                    (last - first) * PAGE,
                    room.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            };
            kernel_count(usize::from(answer != 0));
            answer != 0
        };
        if !failed {
            return (Some(by_page(&room[..count])), None);
        }
        if !guarded {
            return (None, None);
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        // No lock of ours is held here: blocks made read-only outside the
        // guards' lock ([`guard_outside`]) are made so at once, before the
        // journal hears of them (their bits are set after the noting).
        let written = guard_later(first * PAGE, last * PAGE, guard_outside()).map(|(written, arming)| {
            arming.finish(false);
            written
        });
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        (None, written)
    });
    #[cfg(feature = "diagnostics")]
    if found.is_some() || viewed.is_some() {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let mut held = journal();
    land(&mut held, now);
    // What it found is noted, and answered, under a number taken now, above
    // every one given while it was in flight: an answer given meanwhile
    // without waiting for it (by the light refresh's bits) may hold a page
    // that the title wrote after that answer, and that this asking found;
    // noted under the number it was given in flight, below that answer's,
    // the page would never be taken again by that copy (found in test/watch6).
    let now = held.bump();
    if found.is_none() && viewed.is_none() {
        return None;
    }
    let j = &mut *held;
    // A guard's block is a page: all of it. Not a stamp taken back: an
    // asking numbered above this one may have noted the page already.
    let pages: Vec<(usize, u16)> = found
        .unwrap_or_default()
        .into_iter()
        .chain(viewed.iter().flatten().map(|&page| (page, ALL_PIECES)))
        .collect();
    for (page, mask) in pages {
        if j.written.get(&page).is_none_or(|&w| w < now) {
            note_written(j, page, now, mask);
        } else if pieces() {
            note_pieces(j, page, now, mask);
        }
    }
    // The light refresh, as in [`writes_as`]: under the journal's lock,
    // after the noting.
    if let Some(viewed) = &viewed
        && light != 0
    {
        light_count(1);
        publish(j, viewed);
    }
    let held = settle(held, first, last);
    Some((now, answer(&held, first, last)))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// `VICTORY_LIGHT_REFRESH`: 0 off, 1 on, 2 in turn ([`light`]).
fn light_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LIGHT_REFRESH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the bits of the light refresh are kept ([`PUBLISHED`]): with
/// the switch on, and in both phases of `alt`, so that the phase that asks
/// them finds them true whatever was written in the phase before.
fn light_kept() -> bool {
    light_mode() != 0
}

/// Whether an asking about view memory is answered from the bits of its
/// blocks where they can answer (`VICTORY_LIGHT_REFRESH=1`; off by
/// default).
///
/// Nearly every asking about view memory hears that nothing was written,
/// and each takes the guards' lock (which the title's threads wait for in
/// their write faults) inside the journal's, and reads the range's clean
/// blocks off the set. With the switch a range whose blocks all have their
/// bit ([`PUBLISHED`]) is answered from the journal alone ([`writes`]), and
/// a host copy's refresh that the journal shows has nothing new to take
/// copies nothing and allocates nothing ([`nothing_new`]).
///
/// What changes in the answers: nothing. A range is known by its bits
/// exactly while the guards would find it all clean; only who is asked
/// changes. Holes (blocks noted unmapped) get no bit here: a range with one
/// is asked of the guards, which ask Windows, as without the switch.
///
/// `VICTORY_LIGHT_REFRESH=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
/// The bits are kept in both phases ([`light_kept`]) and asked in one.
/// `VICTORY_LIGHT_REFRESH_CHECK=<n>`: a self-check ([`light_check`]).
///
/// After fxceless/victory-fork 94ed363cc (fxceless, hand-ported there
/// after avatarmanz/VIctory-7900XTX 2aa8134b and 74045e46), by hand onto
/// this tree's guards.
pub(crate) fn light() -> bool {
    match light_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [bits, locked, skipped] = light_asks();
                let [samples, mismatches, inconclusive] = light_checks();
                tracing::info!(
                    "refreshes of host copies go the light way from now (VICTORY_LIGHT_REFRESH=alt): {}; since the change before: {bits} askings of view memory answered by the bits of their blocks, {skipped} refreshes that found nothing new by them, {locked} askings that took the guards' lock; self-check (VICTORY_LIGHT_REFRESH_CHECK): {samples} vouched ranges asked of the guards anyway, {mismatches} mismatches, {inconclusive} inconclusive",
                    if on { "ON" } else { "off, as usual (the bits are kept)" }
                );
            }
            on
        }
    }
}

/// With `VICTORY_LIGHT_REFRESH=1`: the counts every 65,536 askings that
/// allow for view memory.
fn light_said() {
    static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
        let [bits, locked, skipped] = light_asks();
        let [samples, mismatches, inconclusive] = light_checks();
        tracing::info!(
            "refreshes of host copies go the light way (VICTORY_LIGHT_REFRESH=1): of the last 65536 askings that allowed for view memory {bits} were answered by the bits of their blocks and {locked} took the guards' lock (the others were of memory the kernel watches), and {skipped} refreshes found nothing new by them with nothing asked; self-check (VICTORY_LIGHT_REFRESH_CHECK): {samples} vouched ranges asked of the guards anyway, {mismatches} mismatches, {inconclusive} inconclusive"
        );
    }
}

/// With the light refresh kept: askings of view memory answered by the bits
/// of their blocks, askings that went to the guards, and refreshes that
/// found nothing new with nothing asked ([`nothing_new`]); since they were
/// last said ([`light_asks`]) and since the last frame line
/// ([`light_line`]).
static LIGHT_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static LIGHT_FRAME: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn light_count(i: usize) {
    LIGHT_ASKS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LIGHT_FRAME[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`LIGHT_ASKS`], taken.
fn light_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The frame line's account of the light refresh, per frame, taken; `None`
/// without the switch.
#[cfg(feature = "diagnostics")]
pub(crate) fn light_line(frames: f64) -> Option<String> {
    if !light_kept() {
        return None;
    }
    let f = frames.max(1.0);
    let [bits, locked, skipped] =
        [0, 1, 2].map(|i| LIGHT_FRAME[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / f);
    let [samples, mismatches, inconclusive] =
        [0, 1, 2].map(|i| LIGHT_CHECK_FRAME[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    Some(format!(
        "light refresh (VICTORY_LIGHT_REFRESH {}, now {}): per frame {bits:.0} askings of view memory answered by the bits of \
         their blocks, {skipped:.0} refreshes that found nothing new by them, {locked:.0} askings that took the guards' lock; \
         self-check since the last line: {samples} vouched ranges asked of the guards anyway, {mismatches} mismatches, \
         {inconclusive} inconclusive",
        if light_mode() == 1 { "1" } else { "alt" },
        if light() { "on" } else { "off" },
    ))
}

/// `VICTORY_LIGHT_REFRESH_CHECK=<n>` (`1`: 256; `all`: every one; off by
/// default): a self-check of the light refresh. Every n-th asking the bits
/// would answer goes to the guards anyway, under the same hold of the
/// journal's lock; a page they find written whose block's bit still stood
/// is a mismatch (the bits said clean, the guards did not, and no write
/// cleared the bit), one whose bit is gone was written meanwhile
/// (inconclusive). The guards' answer is the one used. Counted on the
/// light refresh's lines; a mismatch is said (the first 16, then at powers
/// of two). After CuteBooty/victory-8gb ea131ae01 (its
/// VICTORY_LIGHT_REFRESH_CHECK, by idea; done here at the asking).
fn light_check() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| match std::env::var("VICTORY_LIGHT_REFRESH_CHECK").unwrap_or_default().trim() {
        "1" => 256,
        "all" => 1,
        value => value.parse().unwrap_or(0),
    })
}

/// Whether this vouched asking is one [`light_check`] takes.
fn light_check_due() -> bool {
    static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let every = light_check();
    every != 0 && TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(every)
}

thread_local! {
    /// [`nothing_new`] passed a range it would have answered to [`writes`]
    /// for the self-check ([`light_check`]): the next asking on this
    /// thread is checked.
    static FORCE_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Self-check counts: vouched ranges asked of the guards anyway,
/// mismatches, inconclusive; since last said and since the frame line.
static LIGHT_CHECKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static LIGHT_CHECK_FRAME: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn light_checks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_CHECKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// [`light_check`]: the guards found `written` (pages) of a range the bits
/// vouched for, before [`publish`] looks at them.
fn light_compare(written: &[usize]) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut outcome = 0;
    for &page in written {
        let block = page * PAGE / BLOCK;
        if PUBLISHED.all(block, block + 1) {
            outcome = 1;
            static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let said = SAID.fetch_add(1, Relaxed) + 1;
            if said <= 16 || said.is_power_of_two() {
                tracing::warn!(
                    said,
                    address = format_args!("{:#x}", block * BLOCK),
                    "light refresh mismatch: a block known clean by its bit was found written by the guards (VICTORY_LIGHT_REFRESH_CHECK)"
                );
            }
            break;
        }
        outcome = 2;
    }
    for counts in [&LIGHT_CHECKS, &LIGHT_CHECK_FRAME] {
        counts[0].fetch_add(1, Relaxed);
        if outcome != 0 {
            counts[outcome].fetch_add(1, Relaxed);
        }
    }
}

/// The words of a piece of [`Bits`]: the bits of 4 GiB.
const BITS_WORDS: usize = 1 << 10;
/// The pieces of [`Bits`]: 128 TiB, what a process has.
const BITS_PIECES: usize = 1 << 15;

/// A bit per 64 KiB block of the address space, read and cleared with no
/// lock; in pieces made when first needed and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; BITS_PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; BITS_WORDS];

impl Bits {
    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; BITS_PIECES])
    }

    /// The piece that holds word `word`, if it was ever made.
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / BITS_WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// Sets `block`'s bit (nothing past the address space).
    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / BITS_WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let made: *mut BitsPiece =
                Box::into_raw(Box::new([const { std::sync::atomic::AtomicU64::new(0) }; BITS_WORDS]));
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody: another thread's
                // piece is in the place.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_or(1u64 << (block % 64), SeqCst);
        }
    }

    /// Clears `block`'s bit: ahead of what makes the block writable, and
    /// seen by whoever looks after it. Allocates nothing (the write fault's
    /// handler calls it).
    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_and(!(1u64 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Clears the bits of `[first, last)` (pieces never made are passed
    /// over whole).
    fn clear_range(&self, first: usize, last: usize) {
        let last = last.min(BITS_PIECES * BITS_WORDS * 64);
        let mut block = first;
        while block < last {
            let word = block / 64;
            let Some(piece) = self.piece(word) else {
                block = (word / BITS_WORDS + 1) * BITS_WORDS * 64;
                continue;
            };
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            piece[word % BITS_WORDS].fetch_and(!mask, std::sync::atomic::Ordering::SeqCst);
            block = until;
        }
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks an asking can answer for from the journal alone. A block has
/// its bit only while both of these hold: it is read-only by our hand,
/// noted clean (`Guards::clean`), not faulted and not noted unmapped; and
/// the journal holds every write to it that an asking found.
///
/// Why that is so. A bit is set in one place, [`publish`]: by the asking
/// that found the block written and had the guards make it clean, after it
/// noted that in the journal and with the journal's lock still held, under
/// the guards' lock, and only if the block is clean at that moment (every
/// asking of the guards is made under the journal's lock and notes what
/// they found there). It is cleared under the guards' lock wherever a block
/// stops being clean: before its write access is given back (`open_here`,
/// which the write fault's handler, [`writing`], [`title_memory_written`]
/// and [`views_mapped`] all go through), when a faulted view's other views
/// are left read-only but told as written (`open_fault`, `rearmed`), and
/// when the runtime tells that what the block shows is new
/// ([`title_memory_remapped`]). Clearing does nothing where no bit was ever
/// set, so it is not behind the switch.
///
/// Who asks the bits holds the journal's lock from before the look until
/// it has read the journal ([`published`]), and nobody sets a bit without
/// that lock: a block opened and guarded again between the look and the
/// reading cannot have its bit back in time to be taken for unwritten. A
/// write that comes while the bits are looked at is no different from one
/// that comes just after an asking: its block's bit is gone, and the next
/// asking goes to the guards and finds the block open.
static PUBLISHED: Bits = Bits::new();

/// An asking of view memory found the pages `written` (page numbers)
/// written, had the guards make their blocks read-only and clean
/// ([`guard`]) and has noted them in the journal, whose lock it holds
/// still (`_noted`): the blocks of them that are clean now are known by
/// their bits from here on ([`PUBLISHED`]).
///
/// The blocks an asking leaves clean that have no bit yet are the ones it
/// reports written: every other clean block was left clean by an asking
/// before it, which did this (the bits are kept from the start).
///
/// With [`unlocked`], not a block that an asking still in flight covers:
/// its guards may have found the block written again and made it clean
/// after this asking's guards did, and the journal does not hold that yet.
///
/// Such a block is kept (`Journal::unpublished`) and looked at again when
/// an asking lands ([`land`]): passed over for good, it would be asked of
/// the guards at every refresh until written again, and the light
/// refresh's share would fall the longer a run lasts.
fn publish(noted: &mut Journal, written: &[usize]) {
    if written.is_empty() {
        return;
    }
    let g = guards();
    let mut last = usize::MAX;
    for &page in written {
        let block = page * PAGE / BLOCK;
        if block == last {
            continue;
        }
        last = block;
        if noted.flying.iter().any(|&(a, b, _)| a <= page && page < b) {
            // (A bounded list: past it, such blocks wait for their next
            // write, as before.)
            if noted.unpublished.len() < 1 << 16 {
                noted.unpublished.push(page);
            }
            continue;
        }
        // (Not one that a write has opened since the guards were asked,
        // nor one that could not be made read-only, nor a hole.)
        if g.clean.contains(&block)
            && g.protected.contains(&block)
            && !g.absent.contains(&block)
            && !g.faulted.contains_key(&block)
            && !g.arming.contains(&block)
        {
            PUBLISHED.set(block);
        }
    }
}

/// Whether every block of `[start, end)` has its bit ([`PUBLISHED`]): what
/// [`writes`] would answer of the range is in the journal. `_held`: the
/// journal's lock, which the caller holds from before this look until it
/// has read the journal (no bit is set meanwhile: [`publish`]).
fn published(_held: &Journal, start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether nothing of `[start, end)` was written that a copy does not
/// hold, known with nothing asked of the guards or the kernel: `copied` is
/// the copy's note, for each page from page `base` on, of the asking after
/// which it last took the page (0: never), as `crate::shadow` keeps it.
/// True only if [`writes`] would answer from the journal (the range was
/// asked ahead, or its blocks are known by their bits) and the answer would
/// have the copy take no page. `false` if something would have to be
/// asked, and always without the light refresh ([`light`]).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if !light() {
        return false;
    }
    if stamps_flat() && nothing_flat(start, end, base, copied) {
        return true;
    }
    nothing_noted(start, end, base, copied)
}

/// [`nothing_new`] with [`stamps_flat`]: true when the bits vouch for the
/// range and the table shows nothing the copy does not hold, read with no
/// lock; false when the journal must be asked.
fn nothing_flat(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if start < end
            && let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base))
            && !taken.contains(&0)
            && PUBLISHED.all(first * PAGE / BLOCK, (last * PAGE).div_ceil(BLOCK))
        {
            let flat = STAMPS.range(first, last);
            let same = flat.iter().all(|&(page, write)| write <= taken[page - first]);
            if same {
                if light_check_due() {
                    FORCE_CHECK.with(|f| f.set(true));
                    return false;
                }
                light_count(2);
                STAMP_CHECKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                stamps_checked_later(first, last, &flat);
                return true;
            }
        }
    }
    false
}

/// [`nothing_new`], whatever the switch says.
fn nothing_noted(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    // A page never taken is taken at the next refresh (so is, at every
    // refresh, a page the copy's refreshes only ever hold in part).
    if taken.contains(&0) {
        return false;
    }
    let mut j = journal();
    // The bits first, then the journal, both under its lock ([`published`]).
    let by_bits = !ahead_covers(&j, first, last, true);
    if by_bits && !published(&j, first * PAGE, last * PAGE) {
        return false;
    }
    if !by_bits {
        j = settle(j, first, last);
    }
    let same = answer(&j, first, last).iter().all(|&(page, write)| write <= taken[page - first]);
    if same && by_bits {
        // The self-check takes this one: [`writes`] asks the guards.
        if light_check_due() {
            FORCE_CHECK.with(|f| f.set(true));
            return false;
        }
        light_count(2);
    }
    same
}

/// `VICTORY_GUARD_FIRST=1` (off by default): [`writes`] of a range whose
/// first block the guards hold (read-only by our hand, or noted absent: a
/// view of physical pages, which the runtime tells us when it unmaps) goes
/// to the guards without asking the kernel's write watch first.
///
/// The write watch cannot watch views, so there that asking always fails,
/// and only then were the guards asked. In the open world (all19-s-best1)
/// one copy's refreshes made some 20,300 askings a frame, 104.5 ms summed,
/// of which only about 3,300 were answered by the kernel (24.6 ms for all
/// 5,073 kernel answers): most of the rest were failed kernel calls ahead
/// of the guards' answer, each under the journal's one lock, which every
/// thread that refreshes a copy or checks a texture waits for. The guards'
/// own test is the one [`guard`] already trusts to skip asking Windows
/// whether the range is a view.
fn guard_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_FIRST").is_some_and(|v| v == "1"))
}

/// Whether the guards hold page `first` (64 KiB): guarded, or noted absent.
fn guards_know(first: usize) -> bool {
    let g = guards_timed();
    g.protected.contains(&first) || g.absent.contains(&first)
}

/// Whether the guards keep their blocks as bits, 4,096 blocks (256 MiB of
/// address space) to a chunk of 64 words, and a check walks a range of them
/// a word at a time (`VICTORY_GUARD_BITS=0`: a tree of block numbers, block
/// by block, as before).
///
/// The guards' checks were 16-19 ms a frame of preparing (all8, all9), and
/// of that only 4 ms the calls that make blocks read-only again and 0.1 ms
/// the questions to Windows: the rest was their own bookkeeping over some
/// 9.4 GB of view memory a frame, ~150,000 blocks: the clean blocks of each
/// range read off a tree into a vector, then a lookup or two in the trees
/// for each block that was not clean.
fn guard_bits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_BITS").is_none_or(|v| v != "0"))
}

/// Blocks per chunk of a [`BlockSet`]'s bits.
const CHUNK_BLOCKS: usize = 4096;

/// A set of block numbers: a tree of them, or bits by chunk of
/// [`CHUNK_BLOCKS`] ([`guard_bits`]).
struct BlockSet {
    tree: BTreeSet<usize>,
    bits: BTreeMap<usize, Box<[u64; CHUNK_BLOCKS / 64]>>,
}

impl BlockSet {
    const fn new() -> Self {
        BlockSet { tree: BTreeSet::new(), bits: BTreeMap::new() }
    }

    fn contains(&self, block: &usize) -> bool {
        if !guard_bits() {
            return self.tree.contains(block);
        }
        let (chunk, at) = (block / CHUNK_BLOCKS, block % CHUNK_BLOCKS);
        self.bits.get(&chunk).is_some_and(|words| words[at / 64] >> (at % 64) & 1 != 0)
    }

    fn insert(&mut self, block: usize) {
        if !guard_bits() {
            self.tree.insert(block);
            return;
        }
        let (chunk, at) = (block / CHUNK_BLOCKS, block % CHUNK_BLOCKS);
        self.bits.entry(chunk).or_insert_with(|| Box::new([0; CHUNK_BLOCKS / 64]))[at / 64] |= 1 << (at % 64);
    }

    fn remove(&mut self, block: &usize) {
        if !guard_bits() {
            self.tree.remove(block);
            return;
        }
        let (chunk, at) = (block / CHUNK_BLOCKS, block % CHUNK_BLOCKS);
        if let Some(words) = self.bits.get_mut(&chunk) {
            words[at / 64] &= !(1 << (at % 64));
        }
    }

    /// Calls `each(words, word, mask)` for the words of the chunks that
    /// cover blocks `[from, to)`, with the mask of the range's bits in it;
    /// `make`: make the chunks that are not there (else they are passed
    /// over, as all-zero).
    fn words(&mut self, from: usize, to: usize, make: bool, mut each: impl FnMut(&mut u64, u64)) {
        let mut at = from;
        while at < to {
            let chunk = at / CHUNK_BLOCKS;
            let end = to.min((chunk + 1) * CHUNK_BLOCKS);
            let words = if make {
                Some(self.bits.entry(chunk).or_insert_with(|| Box::new([0; CHUNK_BLOCKS / 64])))
            } else {
                self.bits.get_mut(&chunk)
            };
            if let Some(words) = words {
                let mut b = at;
                while b < end {
                    let (w, bit) = ((b % CHUNK_BLOCKS) / 64, b % 64);
                    let n = (64 - bit).min(end - b);
                    let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
                    each(&mut words[w], mask);
                    b += n;
                }
            }
            at = end;
        }
    }

    /// Adds blocks `[range.start, range.end)`.
    fn extend(&mut self, range: std::ops::Range<usize>) {
        if !guard_bits() {
            self.tree.extend(range);
            return;
        }
        self.words(range.start, range.end, true, |word, mask| *word |= mask);
    }

    /// Takes blocks `[range.start, range.end)` out.
    fn remove_range(&mut self, range: std::ops::Range<usize>) {
        if !guard_bits() {
            let blocks: Vec<usize> = self.tree.range(range).copied().collect();
            for block in blocks {
                self.tree.remove(&block);
            }
            return;
        }
        self.words(range.start, range.end, false, |word, mask| *word &= !mask);
    }

    /// The blocks of `[range.start, range.end)` in the set, in order.
    fn range_vec(&self, range: std::ops::Range<usize>) -> Vec<usize> {
        if !guard_bits() {
            return self.tree.range(range).copied().collect();
        }
        self.scan(range, true)
    }

    /// The blocks of `[range.start, range.end)` in the set (`set`), or not
    /// in it, in order.
    fn scan(&self, range: std::ops::Range<usize>, set: bool) -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = range.start;
        while at < range.end {
            let chunk = at / CHUNK_BLOCKS;
            let end = range.end.min((chunk + 1) * CHUNK_BLOCKS);
            let words = self.bits.get(&chunk);
            let mut b = at;
            while b < end {
                let (w, bit) = ((b % CHUNK_BLOCKS) / 64, b % 64);
                let n = (64 - bit).min(end - b);
                let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
                let word = words.map_or(0, |words| words[w]);
                let mut found = (if set { word } else { !word }) & mask;
                while found != 0 {
                    let i = found.trailing_zeros() as usize;
                    out.push(b - bit + i);
                    found &= found - 1;
                }
                b += n;
            }
            at = end;
        }
        out
    }

    /// The blocks of `[range.start, range.end)` not in the set, in order.
    fn missing(&self, range: std::ops::Range<usize>) -> Vec<usize> {
        if !guard_bits() {
            return range.filter(|block| !self.tree.contains(block)).collect();
        }
        self.scan(range, false)
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BlockSet,
    /// Blocks a write through the kernel was announced for ([`writing`]),
    /// and the other views of their pages. While one is there no guard
    /// makes them read-only again: the kernel's write fails at a read-only
    /// page (a file read straight into title memory then fails with
    /// ERROR_INVALID_USER_BUFFER).
    kernel: Vec<KernelOpen>,
    /// Faulted views and their tracked aliases, until each is taken with
    /// every view protected again. Open memory stays dirty at every ask.
    faulted: BTreeMap<usize, std::time::Instant>,
    /// With [`guard_outside`]: blocks noted read-only and clean whose
    /// protection an asking is changing outside the lock ([`Arming`]).
    arming: BTreeSet<usize>,
    /// Of those, the ones an opening met meanwhile: made writable again
    /// when the asking is done.
    disturbed: BTreeSet<usize>,
    /// With [`holes_known`]: blocks of `absent` found uncommitted in a
    /// placeholder reservation, where pages come only by the runtime's
    /// mapping, which it tells ([`title_memory_remapped`]): not asked of
    /// Windows again until then.
    holes: BTreeSet<usize>,
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
    let now = j.bump();
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
    protected: BlockSet::new(),
    clean: BlockSet::new(),
    absent: BlockSet::new(),
    kernel: Vec::new(),
    faulted: BTreeMap::new(),
    arming: BTreeSet::new(),
    disturbed: BTreeSet::new(),
    holes: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    waited(&GUARDS, 0)
}

/// Since the start: the times the guards' lock (0) and the journal's (1)
/// were found taken, and the microseconds waited for them; a lock found
/// free costs nothing more than before. (fxceless PR #1315, KainanP.)
pub(crate) static WAITS: [[std::sync::atomic::AtomicU64; 2]; 2] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 2];

thread_local! {
    /// [`WAITS`]' microseconds of this thread alone (guards', journal's).
    static WAITED: [std::cell::Cell<u64>; 2] = const { [std::cell::Cell::new(0), std::cell::Cell::new(0)] };
}

/// The microseconds this thread has waited for the guards' lock and for
/// the journal's since it began ([`WAITS`]).
#[cfg(test)]
pub(crate) fn waited_here() -> [u64; 2] {
    WAITED.with(|w| [w[0].get(), w[1].get()])
}

/// Takes `lock`, timing the wait if it is taken ([`WAITS`]`[which]`). The
/// fault handler comes here too: no allocation, a clock read only when
/// it waits.
fn waited<T>(lock: &'static Mutex<T>, which: usize) -> std::sync::MutexGuard<'static, T> {
    match lock.try_lock() {
        Ok(held) => held,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let started = std::time::Instant::now();
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            let us = started.elapsed().as_micros() as u64;
            WAITS[which][0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            WAITS[which][1].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            WAITED.with(|w| w[which].set(w[which].get() + us));
            held
        }
    }
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
    guard_later(start, end, false).map(|(written, arming)| {
        arming.finish(false);
        written
    })
}

/// [`guard`], with `later` ([`guard_outside`]): the blocks to make
/// read-only again are noted so under the guards' lock and made so by
/// [`Arming::finish`], which the caller calls once it has let go of the
/// journal, and before it reads them.
fn guard_later(start: usize, end: usize, later: bool) -> Option<(Vec<usize>, Arming)> {
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
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Blocks of the range another asking is making read-only outside the
    // lock ([`Arming`]): noted clean, and perhaps writable still. Read now
    // and taken for unwritten, a write that lands before their guard
    // stands would be in nobody's copy; so the asking waits for that one.
    while first < last && g.arming.range(first..last).next().is_some() {
        g = ARMED.wait(g).unwrap_or_else(|e| e.into_inner());
    }
    let _held = GuardHold::new();
    let mut runs = Vec::new();
    let written = guard_after_faults(&mut g, first, last, guard_rest(), later.then_some(&mut runs));
    drop(g);
    Some((written, Arming(runs)))
}

fn guard_after_faults(
    g: &mut Guards,
    first: usize,
    last: usize,
    rest: std::time::Duration,
    mut later: Option<&mut Vec<(usize, usize)>>,
) -> Vec<usize> {
    // Without `VICTORY_GUARD_REST` and `VICTORY_OPEN_ONE_VIEW` nothing is
    // ever in `faulted`: the walk as it was, with no look at the map (which
    // panics over a range that ends before it starts).
    if g.faulted.is_empty() || first >= last || g.faulted.range(first..last).next().is_none() {
        return guard_range(g, first, last, later);
    }
    // Each faulted block of the range, and whether its rest is over.
    let faults: Vec<(usize, bool)> =
        g.faulted.range(first..last).map(|(&block, &when)| (block, when.elapsed() >= rest)).collect();
    let most = guard_aliases();
    let mut written = Vec::new();
    let mut next = first;
    let mut i = 0;
    while i < faults.len() {
        let (block, rested) = faults[i];
        if next < block {
            written.extend(guard_range(g, next, block, later.as_deref_mut()));
        }
        // With the faulted blocks that follow it, as long as they answer
        // alike: a run, made read-only in one call as the walk's runs of
        // written blocks are (the title writes a buffer block after block;
        // a call for each was hundreds of them a frame).
        let mut end = block + 1;
        i += 1;
        while i < faults.len() && faults[i] == (end, rested) {
            end += 1;
            i += 1;
        }
        // Told as written at every asking while they are here.
        written.extend(block * BLOCK / PAGE..end * BLOCK / PAGE);
        if rested {
            protect(g, block, end);
            rearmed(g, block, end, most);
        }
        next = end;
    }
    if next < last {
        written.extend(guard_range(g, next, last, later));
    }
    written
}

/// After [`protect`] of the faulted blocks `[from, to)`: those read-only
/// now at every tracked view of their pages are faulted no longer (their
/// bytes are taken by the asker); the others stay, not clean, and are told
/// as written again.
fn rearmed(g: &mut Guards, from: usize, to: usize, most: usize) {
    let all = |g: &Guards, from: usize, to: usize| {
        g.protected.range_vec(from..to).len() == to - from && {
            let (aliases, _) = crate::inplace::aliases_between(from * BLOCK, to * BLOCK, most);
            aliases.iter().all(|at| g.protected.contains(&(at / BLOCK)))
        }
    };
    // The whole run in one look at the views, as a rule.
    if all(g, from, to) {
        for block in from..to {
            g.faulted.remove(&block);
        }
        return;
    }
    for block in from..to {
        if to - from > 1 && all(g, block, block + 1) {
            g.faulted.remove(&block);
        } else {
            PUBLISHED.clear(block);
            g.clean.remove(&block);
        }
    }
}

fn guard_range(g: &mut Guards, first: usize, last: usize, mut later: Option<&mut Vec<(usize, usize)>>) -> Vec<usize> {
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    if guard_bits() {
        return guard_bits_here(g, first, last, later);
    }
    let clean: Vec<usize> = g.clean.range_vec(first..last);
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return written;
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
                uncommitted_until = hole_end(g, block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
                g.holes.remove(&block);
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
            protect_walk(g, from, block, later.as_deref_mut());
        }
    }
    written
}

/// [`guard`]'s walk of blocks `[first, last)` with [`guard_bits`]: only the
/// blocks that are not clean are looked at (found a word at a time), with
/// the same outcome as the walk of every block: the pages of each block
/// written (or newly mapped) since it was last taken, the runs of such
/// blocks guarded again, a block written through the runtime while still
/// read-only taken as clean again, an uncommitted block left alone.
fn guard_bits_here(
    g: &mut Guards,
    first: usize,
    last: usize,
    mut later: Option<&mut Vec<(usize, usize)>>,
) -> Vec<usize> {
    let mut written = Vec::new();
    let open = g.clean.missing(first..last);
    let mut run: Option<(usize, usize)> = None;
    let mut uncommitted_until = first;
    for block in open {
        // A block between the run and this one was clean: the run ends.
        if let Some((from, end)) = run
            && end != block
        {
            protect_walk(g, from, end, later.as_deref_mut());
            run = None;
        }
        if g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = hole_end(g, block);
            }
            if block < uncommitted_until {
                // Still not there: as a clean block.
                if let Some((from, end)) = run.take() {
                    protect_walk(g, from, end, later.as_deref_mut());
                }
                continue;
            }
            g.absent.remove(&block);
            g.holes.remove(&block);
        }
        written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
        if g.protected.contains(&block) {
            // Written through the runtime, still read-only.
            g.clean.insert(block);
            if let Some((from, end)) = run.take() {
                protect_walk(g, from, end, later.as_deref_mut());
            }
        } else {
            run = Some(run.map_or((block, block + 1), |(from, _)| (from, block + 1)));
        }
    }
    if let Some((from, end)) = run {
        protect_walk(g, from, end, later);
    }
    written
}

/// [`protect`] of a run of the walk's written blocks `[from, to)`; with
/// `later` ([`guard_outside`]) noted read-only and clean, and in
/// `Guards::arming`, for [`Arming::finish`] to make so outside the locks,
/// where no other view of the pages is known (those are guarded together,
/// under the lock) and no write through the kernel is announced there.
fn protect_walk(g: &mut Guards, from: usize, to: usize, later: Option<&mut Vec<(usize, usize)>>) {
    let Some(later) = later else {
        protect(g, from, to);
        return;
    };
    kernel_expire(g);
    let busy = g.kernel.iter().any(|k| k.first < to && k.end > from);
    let most = guard_aliases();
    let aliased = most != 0 && !crate::inplace::aliases_between(from * BLOCK, to * BLOCK, most).0.is_empty();
    if busy || aliased {
        protect(g, from, to);
        return;
    }
    g.protected.extend(from..to);
    g.clean.extend(from..to);
    g.arming.extend(from..to);
    later.push((from, to));
}

#[cfg(test)]
thread_local! {
    /// Tests: called once by the next locked asking on this thread between
    /// letting the journal go and making its armed blocks read-only.
    static BEFORE_FINISH: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
}

/// Signalled, under the guards' lock, when an asking's blocks are made
/// read-only ([`Arming::finish`]).
static ARMED: std::sync::Condvar = std::sync::Condvar::new();

/// Runs of blocks `[from, to)` an asking noted read-only and clean, and
/// in `Guards::arming`, whose protection it has yet to change
/// ([`guard_outside`]).
///
/// Why the answers stay what they were. Until `finish` has made a block
/// read-only, a write to it lands with no fault: the asking's own copy,
/// which reads after `finish`, holds it; no other asking of the guards
/// reads the block meanwhile (they wait, [`guard_later`]), and no bit of
/// the light refresh is set for it before `finish` ([`publish`]).
/// Whatever opens the block meanwhile ([`open_here`],
/// [`title_memory_remapped`]) notes it in `Guards::disturbed` and makes it
/// writable as before; `finish` makes it writable again after its own
/// change, and a write that faults on it in between is let through by the
/// handler like any other opening. With `VICTORY_GUARD_VIEWS_ONLY` (on)
/// `finish` changes view memory only, as [`protect_runs`] does.
/// After fxceless/victory-fork PR #1315 (KainanP).
#[must_use]
struct Arming(Vec<(usize, usize)>);

impl Arming {
    /// Makes the blocks read-only and takes them out of `arming`. With
    /// `bits` (the light refresh's bits kept), the blocks that are clean
    /// afterwards get their bit ([`publish`], under the journal's lock).
    /// Called with neither lock held.
    fn finish(mut self, bits: bool) {
        Self::arm(std::mem::take(&mut self.0), bits);
    }

    /// [`Arming::finish`] of `runs`.
    fn arm(runs: Vec<(usize, usize)>, bits: bool) {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
        if runs.is_empty() {
            return;
        }
        let set = |from: usize, to: usize| {
            let mut old = 0;
            let started = std::time::Instant::now();
            // SAFETY: title view memory; only its protection changes.
            let set = unsafe {
                VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) != 0
            };
            let us = started.elapsed().as_micros() as u64;
            GUARD_WORK[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            GUARD_WORK[3].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::GuardProtects, 1);
                add(C::GuardProtectUs, us);
            }
            set
        };
        let mut failed = Vec::new();
        let run = |from: usize, to: usize, failed: &mut Vec<usize>| {
            if set(from, to) {
                return;
            }
            // Across views: a block at a time, as `protect_runs`.
            for block in from..to {
                if !set(block, block + 1) {
                    failed.push(block);
                }
            }
        };
        for &(from, to) in &runs {
            if !views_only() {
                run(from, to, &mut failed);
                continue;
            }
            // View memory only, as `protect_runs`.
            let mut at = from;
            while at < to {
                let (end, view) = view_run(at, to);
                if view {
                    run(at, end, &mut failed);
                } else {
                    failed.extend(at..end);
                }
                at = end;
            }
        }
        let mut held = guards();
        let g = &mut *held;
        for &block in &failed {
            // Not read-only: written at every asking, as before. One the
            // runtime has not committed is noted absent.
            g.protected.remove(&block);
            g.clean.remove(&block);
            if !g.disturbed.contains(&block) && block < hole_end(g, block) {
                g.absent.insert(block);
            }
        }
        for &(from, to) in &runs {
            for block in from..to {
                g.arming.remove(&block);
                if g.disturbed.remove(&block) {
                    let mut old = 0;
                    // SAFETY: a block of ours that was opened while it was
                    // being made read-only: writable, as the opening left it.
                    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
                }
            }
        }
        drop(held);
        ARMED.notify_all();
        if !bits {
            return;
        }
        // The bits, as [`publish`] sets them: the journal's lock first (an
        // asking waiting for these blocks may hold it, so not before they
        // are out of `arming`), then the guards'.
        let pages: Vec<usize> = runs.iter().flat_map(|&(from, to)| from * BLOCK / PAGE..to * BLOCK / PAGE).collect();
        let mut j = journal();
        publish(&mut j, &pages);
    }
}

impl Drop for Arming {
    /// Not finished (a panic on the way): made read-only all the same, or
    /// every later asking of the blocks would wait for good.
    fn drop(&mut self) {
        Self::arm(std::mem::take(&mut self.0), false);
    }
}

/// Tests: [`guard_outside`] as the environment says (0), off (1), on (2).
#[cfg(test)]
pub(crate) static GUARD_OUTSIDE_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether an asking of view memory makes the blocks it found written
/// read-only again outside the guards' lock and the journal's
/// (`VICTORY_GUARD_OUTSIDE=1`; off by default: under both, as before).
/// `alt`: twenty seconds with, twenty without, in turn, each change said
/// in the log with the guards' counts since the start.
///
/// The protection changes are most of the time an asking holds the
/// guards' lock, and the title's threads wait for that lock at every write
/// fault (`watch::install::on_write` in our profiles); while they are made
/// the journal is held as well (without `VICTORY_JOURNAL_UNLOCKED`).
/// Windows makes them side by side. What changes in the answers: nothing
/// ([`Arming`]). Blocks with other views known, or a write through the
/// kernel announced, are made read-only under the lock as before.
/// From fxceless/victory-fork PR #1315 (KainanP, 33c2b7f32).
fn guard_outside() -> bool {
    #[cfg(test)]
    match GUARD_OUTSIDE_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_GUARD_OUTSIDE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                use std::sync::atomic::Ordering::Relaxed;
                let [askings, held, calls, us] = GUARD_WORK.each_ref().map(|c| c.load(Relaxed));
                tracing::info!(
                    "blocks are made read-only again outside the guards' lock from now (VICTORY_GUARD_OUTSIDE=alt): {}; since the start: {askings} askings held the guards' lock {} ms, {calls} protection changes took {} ms, the guards' lock was waited for {} times ({} ms), the journal's {} times ({} ms)",
                    if on { "ON" } else { "off, under the lock as usual" },
                    held / 1000,
                    us / 1000,
                    WAITS[0][0].load(Relaxed),
                    WAITS[0][1].load(Relaxed) / 1000,
                    WAITS[1][0].load(Relaxed),
                    WAITS[1][1].load(Relaxed) / 1000,
                );
            }
            on
        }
    }
}

/// Since the start: askings of the guards that took their lock to walk a
/// range ([`guard_later`]), the microseconds they held it, the calls that
/// made blocks read-only again ([`protect_runs`], [`Arming`]) and their
/// microseconds.
pub(crate) static GUARD_WORK: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Counts a hold of the guards' lock in [`GUARD_WORK`] when dropped.
struct GuardHold(std::time::Instant);

impl GuardHold {
    fn new() -> GuardHold {
        GuardHold(std::time::Instant::now())
    }
}

impl Drop for GuardHold {
    fn drop(&mut self) {
        GUARD_WORK[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GUARD_WORK[1].fetch_add(self.0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
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
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
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
        // the answer as it was before. Free memory is no one's yet
        // (`VICTORY_GUARD_VIEWS_ONLY`, on): taken for a view, its blocks
        // were noted absent and guarded once something was allocated
        // there, the process heap too. Its own code then faulted while
        // holding the heap's lock, the fault waited for the guards' lock,
        // and that lock's holder for the heap: every thread waits, no
        // frame, the title's watchdog ends the process after 30 s.
        // After fxceless/victory-fork PR #1147 (tippythetaidum, 772da7e19).
        if info.State == MEM_COMMIT || (views_only() && info.State == MEM_FREE) || !holes {
            return false;
        }
        // A hole (a placeholder; or free, with VICTORY_GUARD_VIEWS_ONLY=0):
        // on to what follows it.
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
/// `VICTORY_GUARD_ALIASES`, 64 from test/all13; 0: none, as before fxceless #115).
///
/// 64 (2026-11-25, all12w): with every run at 0 the prologue room lost its walls
/// and ceiling in 16 and 9 of 16 shots, 21,372 64 KiB blocks of title-memory
/// copies frozen stale; at 64 views a page, 2 and 0 of 16, none stale, the
/// same frame time (fxceless #866: the same with their view guard).
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
        std::env::var("VICTORY_GUARD_ALIASES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64)
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
        let us = protecting.elapsed().as_micros() as u64;
        GUARD_WORK[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GUARD_WORK[3].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::GuardProtects, 1);
            add(C::GuardProtectUs, us);
        }
        done
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        if clean {
            g.clean.extend(from..to);
        }
    };
    if !views_only() {
        protect_run(g, from, to, &set, &note);
        return;
    }
    // Only view memory is made read-only. A range can run past the views
    // into other committed memory: a 4 MiB window that begins in mapped
    // memory (so [`is_view`] says yes) had every committed private block
    // in it made read-only block by block, the process heap's among them.
    // The heap's own code then faulted while holding its lock, the faults
    // waited for the guards' lock, and the world froze (fxceless #942:
    // every thread waiting, the card idle; here the 30-s watchdog).
    // Committed private memory and images are never guarded.
    // After fxceless/victory-fork PR #1147 (tippythetaidum, e8a93d2bc).
    let mut at = from;
    while at < to {
        let (end, view) = view_run(at, to);
        if view {
            protect_run(g, at, end, &set, &note);
        } else {
            for block in at..end {
                not_guarded(g, block);
            }
        }
        at = end;
    }
}

/// From `block` on (to `to` at most): the end of the run of blocks whose
/// memory is of one kind, and whether that kind may be guarded (anything
/// but committed private memory and images: views, and placeholders,
/// which [`not_guarded`] notes absent when a protection fails).
fn view_run(block: usize, to: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    let asking = std::time::Instant::now();
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::GuardQueries, 1);
        add(C::GuardQueryUs, asking.elapsed().as_micros() as u64);
    }
    if n == 0 {
        return (block + 1, false);
    }
    let end = (info.BaseAddress as usize).saturating_add(info.RegionSize).div_ceil(BLOCK).clamp(block + 1, to);
    let other = info.State == MEM_COMMIT && (info.Type == MEM_PRIVATE || info.Type == MEM_IMAGE);
    (end, !other)
}

/// [`protect_runs`] for a run of view blocks: whole, else block by block.
fn protect_run(
    g: &mut Guards,
    from: usize,
    to: usize,
    set: &dyn Fn(usize, usize) -> bool,
    note: &dyn Fn(&mut Guards, usize, usize),
) {
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else {
            not_guarded(g, block);
        }
    }
}

/// A block of a range to guard that was not made read-only: noted absent
/// where nothing is committed (a placeholder the runtime maps later).
fn not_guarded(g: &mut Guards, block: usize) {
    if block < hole_end(g, block) {
        g.absent.insert(block);
    }
}

/// Guards make view memory read-only and nothing else
/// (`VICTORY_GUARD_VIEWS_ONLY`, on by default; `=0` as before): free
/// memory is no view, and committed private memory and images inside a
/// range to guard stay writable. After fxceless/victory-fork PR #1147
/// (tippythetaidum): both ways made the process heap read-only, which
/// deadlocks its lock against the guards' lock.
fn views_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_VIEWS_ONLY").is_none_or(|v| v != "0"))
}

/// [`uncommitted_end`] through the guards' cache of holes ([`holes_known`]):
/// a block noted a hole in a placeholder reservation is still one (at
/// least itself), with no question to Windows.
fn hole_end(g: &mut Guards, block: usize) -> usize {
    if !holes_known() {
        return uncommitted_end(block);
    }
    if g.holes.contains(&block) {
        kernel_count(4);
        return block + 1;
    }
    let end = uncommitted_end(block);
    if end > block && placeholder(block) {
        g.holes.insert(block);
    }
    end
}

/// Whether the guards keep the blocks they found uncommitted inside a
/// placeholder reservation as holes until the runtime tells of a mapping
/// there, instead of asking Windows again at every asking
/// (`VICTORY_HOLES_KNOWN=1`; off by default).
///
/// Every asking of a range with unmapped blocks asks `VirtualQuery`
/// whether they still are: some 7,000 questions a frame on the Series S
/// house scene (all22-s-wft-prof: `uncommitted_end` 5 % of the submitting
/// thread's running). In a placeholder reservation pages come to be only
/// by mapping a view or an allocation in a placeholder's place, which the
/// runtime does and tells of (`title_memory_remapped`, xmem); an ordinary
/// reservation can be committed by anybody with no word to us, so its
/// holes are asked as before. The placeholder question is asked once a
/// hole (`NtQueryVirtualMemory`, as fxceless 94ed363cc asks it).
fn holes_known() -> bool {
    #[cfg(test)]
    if HOLES_KNOWN_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HOLES_KNOWN").is_some_and(|v| v == "1"))
}

/// Tests: [`holes_known`] on, whatever the switch says.
#[cfg(test)]
pub(crate) static HOLES_KNOWN_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's question about a process's memory, `(process, address,
    /// class of the answer, answer out, its room, bytes answered out)`: an
    /// NTSTATUS, negative when it cannot answer. It changes nothing.
    fn NtQueryVirtualMemory(
        process: *mut std::ffi::c_void,
        at: *const std::ffi::c_void,
        class: u32,
        out: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Whether `block` lies in a placeholder reservation: memory in which
/// pages come to be only by mapping a view, or making an allocation, in
/// its place. `VirtualQuery` does not tell it from an ordinary reservation;
/// this question does. (After fxceless/victory-fork 94ed363cc.)
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk); only the kinds are read.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    // (`MemoryRegionInformationEx`; class 3 answers with the kind of memory
    // in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    kernel_count(5);
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    let asking = std::time::Instant::now();
    kernel_count(3);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::GuardAbsentQueries, 1);
        add(C::GuardAbsentUs, asking.elapsed().as_micros() as u64);
    }
    // Placeholders only: free memory is not noted absent (see `is_view`;
    // `VICTORY_GUARD_VIEWS_ONLY=0`: free memory too, as before).
    if n != 0 && (info.State == MEM_RESERVE || (info.State == MEM_FREE && !views_only())) {
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
    let mut blocks: Vec<usize> = g.protected.range_vec(start / BLOCK..end.div_ceil(BLOCK));
    // Being made read-only by an asking outside the lock ([`Arming`]): opened
    // as any guarded block (it may be read-only already, or become so, and
    // be out of `protected` after an earlier opening), and made writable
    // again when the asking is done.
    if !g.arming.is_empty() && start / BLOCK < end.div_ceil(BLOCK) {
        let arming: Vec<usize> = g.arming.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
        g.disturbed.extend(arming.iter().copied());
        blocks.extend(arming);
        blocks.sort_unstable();
        blocks.dedup();
    }
    for &block in &blocks {
        // Before the block can be written ([`PUBLISHED`]).
        PUBLISHED.clear(block);
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

/// Whether small fills and copies of title memory go on without waiting
/// for flights where the title wrote the bytes since they were taken, or
/// no shader write landed there lately (`VICTORY_FILL_TITLE=1`; off by
/// default; `crate::host`'s `fill_is_titles`). With it, every landing of
/// shader writes into title memory notes its 4 KiB pages' frame
/// ([`landing`]).
pub(crate) fn fill_title() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_TITLE").is_some_and(|v| v == "1"))
}

/// The frame shader writes last landed in, by 4 KiB page: pieces of 256
/// MiB made when first written, never freed.
struct Landed([std::sync::atomic::AtomicPtr<LandedPiece>; Landed::PIECES]);

/// Of 256 MiB: the frame (its low 31 bits and one), 0 for never.
type LandedPiece = [std::sync::atomic::AtomicU32; Landed::PAGES];

impl Landed {
    const PAGES: usize = 1 << 16;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 19;

    fn stamp(frame: u64) -> u32 {
        (frame as u32 & 0x7fff_ffff) + 1
    }

    fn note(&self, page: usize, frame: u64) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return;
        };
        let mut piece = place.load(Acquire);
        if piece.is_null() {
            let made: Box<LandedPiece> = Box::new([const { std::sync::atomic::AtomicU32::new(0) }; Landed::PAGES]);
            let made = Box::into_raw(made);
            piece = match place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire) {
                Ok(_) => made,
                Err(there) => {
                    // SAFETY: made just now, and another thread's is there.
                    drop(unsafe { Box::from_raw(made) });
                    there
                }
            };
        }
        // SAFETY: a piece, which is never freed.
        let at = unsafe { &(&*piece)[page % Landed::PAGES] };
        let stamp = Landed::stamp(frame);
        if at.load(Relaxed) != stamp {
            at.store(stamp, Relaxed);
        }
    }

    /// Whether `page` was noted in a frame from `since` on, up to `now`.
    fn since(&self, page: usize, since: u64, now: u64) -> bool {
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return false;
        };
        // SAFETY: null, or a piece, which is never freed.
        let Some(piece) = (unsafe { place.load(std::sync::atomic::Ordering::Acquire).as_ref() }) else {
            return false;
        };
        let stamp = piece[page % Landed::PAGES].load(std::sync::atomic::Ordering::Relaxed);
        stamp != 0 && u64::from(Landed::stamp(now).wrapping_sub(stamp) & 0x7fff_ffff) <= now.saturating_sub(since)
    }
}

static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);

/// What shaders wrote is about to go into title memory at `[at, at +
/// len)` (with [`fill_title`]): its pages are noted as landed in this
/// frame, and the whole 64 KiB pages of the memory's other views too.
pub(crate) fn landing(at: usize, len: usize) {
    if !fill_title() || len == 0 {
        return;
    }
    let frame = crate::frame::current();
    let (first, last) = (at / KERNEL_PAGE, (at + len - 1) / KERNEL_PAGE);
    (first..=last).for_each(|page| LANDED.note(page, frame));
    let most = guard_aliases();
    if most > 0 {
        let (aliases, _) = crate::inplace::aliases_between(at, at + len, most);
        for alias in aliases {
            let page = alias / KERNEL_PAGE;
            (page..page + PAGE / KERNEL_PAGE).for_each(|page| LANDED.note(page, frame));
        }
    }
}

/// Whether shaders' writes went into a page of `[at, at + len)` in the
/// last `frames` frames ([`landing`]).
pub(crate) fn landed_lately(at: usize, len: usize, frames: u64) -> bool {
    let now = crate::frame::current();
    (at / KERNEL_PAGE..=(at + len.max(1) - 1) / KERNEL_PAGE)
        .any(|page| LANDED.since(page, now.saturating_sub(frames), now))
}

/// Whether every block of `[at, at + len)` is view memory written since a
/// host copy last took it: guarded then, and the guard is off now
/// (written by the title, or by a landing of shader writes). Not for
/// memory that is no view, which is never guarded (no guard can say).
pub(crate) fn written_since_taken(at: usize, len: usize) -> bool {
    if len == 0 || !is_view(at, at + len) {
        return false;
    }
    let g = guards();
    (at / BLOCK..=(at + len - 1) / BLOCK)
        .all(|b| !g.clean.contains(&b) && !g.absent.contains(&b) && !g.protected.contains(&b))
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
        // Blocks an asking is making read-only outside the lock
        // ([`Arming`]): the kernel's write would fail at a page made
        // read-only after this opening; it waits for that asking.
        while at < end && g.arming.range(at / BLOCK..end.div_ceil(BLOCK)).next().is_some() {
            g = ARMED.wait(g).unwrap_or_else(|e| e.into_inner());
        }
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
    // And at the other views of its pages: what the card wrote into a
    // texture's memory through a buffer and we put back reached the textures
    // placed at another view of those pages only when their turn to be
    // looked at again came (`VICTORY_WRITE_ALIAS_TEXTURES`).
    for (from, to) in alias_pages_written(at, end) {
        crate::descriptor::host::remapped(from, to - from);
    }
    KernelWrite { write, title: None }
}

/// `VICTORY_WRITE_ALIAS_TEXTURES` (ON by default; `0` as before): a write of
/// ours into title memory ([`writing`]: the shader results we put back, the
/// title's copies, the title's kernel reads) also makes the textures placed
/// at the other views of the written pages be looked at again, not only
/// those placed in the written range. Belobug/VIctory-Belobug-AMD PR #139
/// (Zetsukae, a208d61db and faac6de43, GPL-3.0): the card writes skin and
/// tattoo textures' memory through a buffer view; their uploads stayed stale
/// until their next look (noise on skin, a new pattern each launch).
fn write_alias_textures() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_WRITE_ALIAS_TEXTURES").map_or(true, |v| v.trim() != "0"))
}

/// Since the start: 64 KiB pages at other views of memory we wrote, made
/// to be looked at again for the textures placed there
/// ([`write_alias_textures`]; the frame lines' "textures picked by index"
/// line).
pub(crate) static ALIAS_PAGES_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The 64 KiB pages at other views of the pages of `[at, end)` (no more than
/// [`guard_aliases`] views a page), as ranges; none with
/// `VICTORY_WRITE_ALIAS_TEXTURES=0` or `VICTORY_GUARD_ALIASES=0`.
fn alias_pages_written(at: usize, end: usize) -> Vec<(usize, usize)> {
    let most = guard_aliases();
    if at >= end || most == 0 || !write_alias_textures() {
        return Vec::new();
    }
    const K64: usize = 1 << 16;
    let (mut pages, _) = crate::inplace::aliases_between(at, end, most);
    pages.sort_unstable();
    pages.dedup();
    ALIAS_PAGES_WRITTEN.fetch_add(pages.len() as u64, std::sync::atomic::Ordering::Relaxed);
    pages.into_iter().map(|page| (page, page + K64)).collect()
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

/// A read the runtime did into title memory `[at, at + len)` at a plain
/// address (DirectStorage, not physical pages), of which [`writing`] was
/// told before it began, has landed: the placed textures under it that a
/// texture pass uploaded meanwhile are due again
/// (`crate::descriptor::host::page_writes`; `VICTORY_PAGE_WRITES=0`: not).
pub fn title_memory_read_done(at: usize, len: usize) {
    if len != 0 && crate::descriptor::host::page_writes() {
        crate::descriptor::host::page_written(at, len);
    }
}

/// Since the start ([`title_memory_read_done`]): ranges kept for a texture
/// pass, ranges dropped, slots of placed textures under them, slots of
/// those uploaded before the read landed and so uploaded again.
pub fn page_write_counts() -> [u64; 4] {
    std::array::from_fn(|i| crate::descriptor::host::PAGE_WRITES[i].load(std::sync::atomic::Ordering::Relaxed))
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

/// Whether [`title_memory_remapped`] gives write access back to the blocks
/// of its range that a guard made read-only, not only forgets them
/// (`VICTORY_REMAP_REOPEN=0`: forgets them, as before). On by default.
///
/// The runtime tells of a new mapping after it has made it (`xmem`'s `map`:
/// the view is mapped and committed, then told), and a guard check runs on
/// another thread under no lock the runtime holds (`writes` for a host copy
/// over the range). A check that fell between the two made the new view
/// read-only and noted it guarded; the notice then only forgot the blocks.
/// They stayed read-only with no guard that knew them: the title's next
/// write there was an access violation our handler passed on (`open` found
/// no guarded block, `writable` said no), and a kernel read into them
/// failed, until some later check guarded them again. The same for a view
/// told before it is unmapped and written in between.
///
/// What it costs: one protection call for each run of blocks the range
/// still holds guarded (none for a range no check has guarded since its
/// last notice).
fn reopen_remapped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REMAP_REOPEN").is_none_or(|v| v != "0"))
}

/// Whether textures uploaded from memory the runtime maps anew are uploaded
/// again (`VICTORY_REMAP_REUPLOAD=0`: not).
fn remap_reupload() -> bool {
    static REUPLOAD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REUPLOAD.get_or_init(|| std::env::var_os("VICTORY_REMAP_REUPLOAD").is_none_or(|v| v != "0"))
}

/// The title asked the runtime to map at `[at, at + len)` the very pages
/// that are shown there already, and the runtime left the view as it is
/// (`VICTORY_REMAP_SAME=1`, `victory-hle`'s `xmem`): the bytes and the
/// mapping are what they were, but all that [`title_memory_remapped`] said
/// of the BYTES is said here too, so that nothing over them stays "known
/// unchanged" across the call, as nothing did when the view was made anew:
///
/// - guarded blocks there are opened (read-write again, here and at the
///   other views of their pages) and taken as written at the next guard
///   check, whole ([`writing`], a write through the kernel that is done at
///   once). After a real mapping the fresh view was read-write and the
///   guards over it forgotten: the same state, reached without leaving a
///   block read-only that the guards no longer know (a write to such a
///   block would be a fault nobody takes). The journal is not written to:
///   it hears of view memory from the guards' answers, as after a mapping;
/// - placed textures there are looked at again (in [`writing`]);
/// - textures uploaded from there are uploaded again ([`remap_reupload`]).
///
/// Left out is only what told of the MAPPING, which did not change: the
/// views the runtime told us of stay, what is placed in place stays, host
/// copies of placed textures keep their memory, and nothing was unmapped.
pub fn title_memory_same(at: usize, len: usize) {
    drop(writing(at, len));
    if remap_reupload() {
        crate::texture::remapped(at, len);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped). Blocks of it that a guard made read-only are made writable
/// again ([`reopen_remapped`]): the check that guarded them may have come
/// after the mapping this tells of.
pub fn title_memory_remapped(at: usize, len: usize) {
    // Texels on their way back to a texture there have nowhere to land.
    #[cfg(windows)]
    crate::texback::remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // Textures already uploaded from there must be uploaded again, or the
    // next list that samples them reads their old contents
    // (`VICTORY_REMAP_REUPLOAD=0`: not, as before fxceless #606).
    if remap_reupload() {
        crate::texture::remapped(at, len);
    }
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    // Host copies of placed textures there have nothing left to come from:
    // let go of those nothing else holds now, rather than seconds later.
    crate::resource::evict_placed_in(at as u64, len as u64);
    // Before the guards' lock: the journal's is taken first.
    views_remapped(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    // What it shows is new ([`PUBLISHED`]).
    PUBLISHED.clear_range(range.start, range.end);
    // Being made read-only outside the lock ([`Arming`]): writable again
    // when that is done.
    if !g.arming.is_empty() && range.start < range.end {
        let arming: Vec<usize> = g.arming.range(range.clone()).copied().collect();
        g.disturbed.extend(arming);
    }
    if !g.faulted.is_empty() {
        let gone: Vec<_> = g.faulted.range(range.clone()).map(|(&block, _)| block).collect();
        for block in gone {
            g.faulted.remove(&block);
        }
    }
    g.absent.remove_range(range.clone());
    if !g.holes.is_empty() && range.start < range.end {
        let told: Vec<usize> = g.holes.range(range.clone()).copied().collect();
        for block in told {
            g.holes.remove(&block);
        }
    }
    if reopen_remapped() {
        // Forgotten and writable again (a block whose memory is gone has
        // no protection to change: the call fails and nothing is lost).
        open_here(&mut g, at, at.saturating_add(len));
    }
    g.protected.remove_range(range.clone());
    g.clean.remove_range(range.clone());
    // A write through the kernel announced there wrote what the memory
    // showed before (or failed): the new memory may be guarded at once.
    g.kernel.retain(|k| k.end <= range.start || k.first >= range.end);
}

/// Zero disables the rest. One selects the upstream 3 ms interval, and
/// larger values are microseconds.
fn guard_rest() -> std::time::Duration {
    static REST: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *REST.get_or_init(|| {
        let us = std::env::var("VICTORY_GUARD_REST").ok()
            .and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
        std::time::Duration::from_micros(if us == 1 { 3000 } else { us })
    })
}

fn open_one_view() -> bool {
    static ONE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONE.get_or_init(|| std::env::var_os("VICTORY_OPEN_ONE_VIEW").is_some_and(|v| v == "1"))
}

/// Open the faulting view, invalidating every tracked view before the
/// writer resumes. Kernel writes still take the normal all-view path.
fn open_fault(g: &mut Guards, at: usize, one: bool, rest: std::time::Duration) -> bool {
    if !one && rest.is_zero() {
        return open(g, at, at + 1);
    }
    let block = at / BLOCK;
    // (Or being made read-only outside the lock, [`Arming`].)
    if !g.protected.contains(&block) && !g.arming.contains(&block) {
        return false;
    }
    let (aliases, _) = crate::inplace::aliases_between(block * BLOCK, (block + 1) * BLOCK, guard_aliases());
    let now = std::time::Instant::now();
    g.faulted.insert(block, now);
    for alias in aliases {
        let other = alias / BLOCK;
        g.faulted.insert(other, now);
        PUBLISHED.clear(other);
        g.clean.remove(&other);
        if !one {
            open_here(g, alias, alias + 1);
        }
    }
    open_here(g, at, at + 1)
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
        if open_fault(&mut g, at, open_one_view(), guard_rest()) {
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
    ONCE.call_once(|| {
        // The handler's switches are read here, ahead of it (`guard_rest`
        // by every `guard` as well): `open_one_view`, read first inside a
        // fault, was the environment looked up, and memory asked for, in
        // the middle of whatever the title's thread was doing when it wrote
        // the block.
        let _ = (open_one_view(), guard_rest());
        // SAFETY: the handler only touches our guard state and protections.
        unsafe { AddVectoredExceptionHandler(1, Some(on_write)) };
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
pub(crate) mod tests {
    /// A write into memory one page of which another view also shows: that
    /// view's page is made to be looked at again (#139), the written view's
    /// own pages are not listed (they go as the range itself).
    #[test]
    fn a_write_reaches_the_textures_at_other_views_of_its_pages() {
        const K64: usize = 1 << 16;
        let (a, b) = (0x7_8000_0000usize, 0x7_9000_0000usize);
        // Two views of section 991: a shows pages 0..4, b shows pages 2..3.
        crate::inplace::title_memory_view(a, 4 * K64, 991, 0);
        crate::inplace::title_memory_view(b, K64, 991, 2 * K64 as u64);
        let before = ALIAS_PAGES_WRITTEN.load(std::sync::atomic::Ordering::Relaxed);
        // A write into a's page 2: b's page.
        assert_eq!(super::alias_pages_written(a + 2 * K64 + 100, a + 2 * K64 + 200), vec![(b, b + K64)]);
        // Into b: a's page 2.
        assert_eq!(super::alias_pages_written(b + 8, b + 16), vec![(a + 2 * K64, a + 3 * K64)]);
        // Into a's page 0: no other view.
        assert!(super::alias_pages_written(a, a + K64).is_empty());
        assert!(ALIAS_PAGES_WRITTEN.load(std::sync::atomic::Ordering::Relaxed) >= before + 2);
        crate::inplace::views_gone(a, 4 * K64);
        crate::inplace::views_gone(b, K64);
    }

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
        // The journal is one for the process, and this address may have been
        // another test's until it freed it.
        forget_pages(start, end);
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
        forget_pages(start, end);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Tests: the runtime's notice of a new mapping at `[at, at + len)`
    /// (`title_memory_remapped`), which every test gives for the memory it
    /// maps, as `xmem` does: the guards are one set for the process, and
    /// the address may have been another test's view, whose blocks would
    /// still count as guarded and clean here.
    fn mapped(at: usize, len: usize) {
        title_memory_remapped(at, len);
    }

    /// Free memory is not tracked: tracked, it would be guarded once
    /// something is allocated there (the process heap: its writes then
    /// fault into the handler while a thread holds the guards' lock).
    /// From fxceless/victory-fork PR #1147 (tippythetaidum, 772da7e19).
    #[test]
    fn free_memory_is_not_tracked() {
        use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
        if !views_only() {
            return; // VICTORY_GUARD_VIEWS_ONLY=0: free memory is taken for a view, as before
        }
        // SAFETY: a reservation of ours, released at once: free memory.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
        let view = is_view(r, r + 2 * BLOCK);
        let absent = uncommitted_end(r / BLOCK) > r / BLOCK;
        // Other tests allocate meanwhile: only memory still free counts.
        // SAFETY: plain data filled in by the query, of the right size.
        let still_free = unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(r as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_FREE
                && info.BaseAddress as usize + info.RegionSize >= r + 2 * BLOCK
        };
        assert!(!(view && still_free), "free memory taken for a view");
        assert!(!(absent && still_free), "free memory noted absent");
    }

    /// A range that begins in a view and runs into committed private
    /// memory guards the view only: the private block (the process heap's,
    /// say) stays writable. From fxceless/victory-fork PR #1147
    /// (tippythetaidum, e8a93d2bc).
    #[test]
    fn committed_private_memory_beside_a_view_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MapViewOfFile3, PAGE_NOACCESS,
            PAGE_READONLY, UnmapViewOfFile2, VirtualAlloc2, VirtualQuery,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        if !views_only() {
            return; // VICTORY_GUARD_VIEWS_ONLY=0: the private block is guarded too, as before
        }
        // SAFETY: a fresh placeholder of two blocks, split; a pagefile
        // section's block mapped over the first, private memory committed
        // over the second.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, p);
            let private = VirtualAlloc2(
                GetCurrentProcess(),
                (p + BLOCK) as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_eq!(private, p + BLOCK);
            (p, section, view)
        };
        title_memory_remapped(p, 2 * BLOCK);
        assert!(is_view(p, p + 2 * BLOCK), "it begins in a view");
        assert!(guard(p, p + 2 * BLOCK).is_some());
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query, of the right size.
            unsafe {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(at as *const _, &mut info, size_of_val(&info)), 0);
                info.Protect
            }
        };
        assert_eq!(protection(p), PAGE_READONLY, "the view is guarded");
        assert_eq!(protection(p + BLOCK), PAGE_READWRITE, "the private block is not");
        // SAFETY: committed private memory of ours, writable.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our view, section and private block.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// `VICTORY_ASK_ONCE_VIEWS`: one asking of a submission's view ranges
    /// guards their union once; every range inside answers from the journal
    /// under that asking with what the guards found written, and only an
    /// asker that would ask the guards; a remap there takes the ranges out
    /// of the asking. After fxceless PR #995 (tunczzz).
    #[test]
    fn view_ranges_asked_once_answer_from_the_journal() {
        // (The journal's way, `writes_as` with no light refresh: with its
        // bits and the flat stamps an asking may be answered by the bits.)
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        const SIZE: usize = 8 * BLOCK;
        // SAFETY: a pagefile section of eight blocks, mapped read-write.
        let (section, view) = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize)
        };
        assert_ne!(view, 0);
        mapped(view, SIZE);
        forget_pages(view, view + SIZE);
        let page = view / PAGE;
        let b = |k: usize| view + k * BLOCK;
        let ranges = [(b(0), b(4)), (b(1), b(2)), (b(2), b(3) + 100), (b(3), b(8)), (b(4), b(5)), (b(6), b(7))];
        // The journal's standing asking is one for the process: a test
        // beside this one may end it. Such a round is made again.
        let mut rounds = 0;
        let undisturbed = loop {
            rounds += 1;
            if rounds > 1000 {
                break false;
            }
            // Guarded once, so the guards know it.
            ask_once_end();
            ask_views(&[(view, view + SIZE)]);
            ask_once_end();
            // SAFETY: inside the view; the writes fault and open the blocks.
            unsafe {
                (b(1) as *mut u8).write_volatile(1);
                ((b(5) + 9) as *mut u8).write_volatile(1);
            }
            ask_views(&ranges);
            let (now, written) = writes_as(view, view + SIZE, true, 0, false, unlocked()).expect("the union was asked");
            // Written since the first asking took the range in (with the
            // flat stamps each block the guards ask about alone is noted
            // under a number of its own, all above that one).
            let first_taken = written.iter().map(|w| w.1).min().unwrap_or(0);
            let fresh: Vec<usize> = written.iter().filter(|&&(_, w)| w > first_taken).map(|&(p, _)| p).collect();
            let journal = ranges.iter().all(|&(a, e)| writes_as(a, e, true, 0, false, unlocked()).map(|w| w.0) == Some(now))
                && asked(b(2), b(3)).map(|w| w.0) == Some(now);
            // Not write-watched: an asker that would not ask the guards
            // hears so, asking ahead or not.
            let unwatched = writes_as(b(2), b(3), false, 0, false, unlocked()).is_none();
            // A remap there: the refreshes ask the guards again.
            forget_views_ahead(b(4), BLOCK);
            let after = asked(b(0), b(1));
            let again = writes_as(b(0), b(1), true, 0, false, unlocked()).map(|w| w.0);
            ask_once_end();
            if !journal || after.is_some() || again.is_none_or(|w| w <= now) {
                std::thread::yield_now();
                continue;
            }
            assert_eq!(fresh, [page + 1, page + 5], "what the guards found written since");
            assert!(unwatched, "view memory answers only an asker that would ask the guards");
            break true;
        };
        // SAFETY: our view and section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(view, SIZE);
        forget_pages(view, view + SIZE);
        assert!(undisturbed, "the ranges asked once answer from the journal");
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
        mapped(view, 2 * BLOCK);
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
        // Told before the view goes: once it is unmapped the address is
        // anyone's (another test's view, guarded at once), and a notice
        // given after that would take the other's guard away under it.
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
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
        mapped(start, SIZE);
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
        mapped(p, 3 * BLOCK);
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
        // Told before the memory goes (the address is anyone's after).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
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
        mapped(p, 3 * BLOCK);
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
        // Told before the memory goes (the address is anyone's after).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
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
            mapped(one, bytes);
            mapped(two, bytes);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The tests of the light refresh take turns: they read its counts,
    /// which no other test moves (the switch is not set in tests, so
    /// [`writes`] itself keeps no bits and asks none).
    pub(crate) fn light_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Askings answered by the bits, askings that went to the guards,
    /// refreshes that found nothing new by them.
    fn light_counts() -> [u64; 3] {
        [0, 1, 2].map(|i| LIGHT_ASKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Self-check counts: samples, mismatches, inconclusive.
    fn check_counts() -> [u64; 3] {
        [0, 1, 2].map(|i| LIGHT_CHECKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `[start, end)` is known by its bits.
    fn known(start: usize, end: usize) -> bool {
        published(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), start, end)
    }

    /// The pages of an answer of [`writes`] written after the asking `since`.
    fn since(since: u64, written: Vec<(usize, u64)>) -> Vec<usize> {
        written.into_iter().filter(|w| w.1 > since).map(|w| w.0).collect()
    }

    /// The bits: set and cleared one by one and by range, across words and
    /// pieces; a range is all set or not; none past the address space.
    #[test]
    fn bits_are_set_cleared_and_read_by_range() {
        static B: Bits = Bits::new();
        // Across a word and a piece (4 GiB of blocks) boundary.
        let base = (1 << 30) - 70;
        assert!(!B.all(base, base + 1), "never made");
        for block in base..base + 140 {
            B.set(block);
        }
        assert!(B.all(base, base + 140));
        assert!(!B.all(base, base + 141));
        assert!(!B.all(base, base), "an empty range is not known");
        B.clear(base + 69);
        assert!(!B.all(base, base + 140) && B.all(base, base + 69) && B.all(base + 70, base + 140));
        B.clear_range(base + 60, base + 75);
        assert!(B.all(base, base + 60) && !B.all(base + 60, base + 61) && !B.all(base + 74, base + 75));
        assert!(B.all(base + 75, base + 140));
        B.clear_range(base, usize::MAX);
        assert!((base..base + 140).all(|block| !B.all(block, block + 1)));
        B.set(usize::MAX / 64);
        assert!(!B.all(usize::MAX / 64, usize::MAX / 64 + 1), "past the address space");
    }

    /// The asking that makes blocks clean gives them their bits; the next
    /// one is answered by them, and the guards are not asked. A write takes
    /// the written block's bit, through the guarded view or through another
    /// view of the same pages, and the next asking reports it; so does a
    /// remap. After fxceless 94ed363cc's tests.
    #[test]
    fn a_range_is_known_by_its_bits_until_a_write() {
        let _turn = light_turn();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
        let page = |block: usize| (one + block * BLOCK) / PAGE;
        assert!(!known(one, one + SIZE), "nothing asked yet");
        // The first asking takes every block in, read-only and clean.
        let [bits, locked, _] = light_counts();
        let (first, _) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
        assert_eq!(light_counts()[1], locked + 1, "the guards were asked");
        assert!(known(one, one + SIZE));
        // The next is answered by the bits.
        let (second, written) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
        assert_eq!(light_counts()[0], bits + 1, "answered by the bits");
        assert!(second > first);
        assert!(since(first, written).is_empty(), "nothing written");
        // With the bits kept but not asked (`alt`'s other phase) the
        // guards are asked, and say the same.
        let (third, written) = writes_as(one, one + SIZE, true, 1, false, unlocked()).expect("a view");
        assert_eq!(light_counts()[1], locked + 2);
        assert!(since(second, written).is_empty());
        // A write through the guarded view takes its block's bit.
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + SIZE) && known(one, one + BLOCK) && known(one + 2 * BLOCK, one + SIZE));
        let (fourth, written) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
        assert_eq!(since(third, written), [page(1)]);
        assert!(known(one, one + SIZE), "guarded again, known again");
        // A write through the other view of the same pages too.
        if guard_aliases() > 0 {
            // SAFETY: inside the other view; it faults like one at `one`.
            unsafe { ((two + 2 * BLOCK + 8) as *mut u8).write_volatile(1) };
            assert!(!known(one + 2 * BLOCK, one + SIZE), "through another view");
            let (_, written) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
            assert_eq!(since(fourth, written), [page(2)]);
            assert!(known(one, one + SIZE));
        }
        // What a remap shows is new.
        title_memory_remapped(one, BLOCK);
        assert!(!known(one, one + BLOCK) && known(one + BLOCK, one + SIZE));
        drop_views(section, &[one, two], SIZE);
        assert!(!known(one, one + SIZE));
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
    }

    /// A refresh that the journal and the bits show has nothing new to take
    /// asks nothing; one with a page never taken, or a page written since,
    /// does.
    #[test]
    fn a_copy_that_holds_everything_finds_nothing_new() {
        if light_check() != 0 {
            return; // VICTORY_LIGHT_REFRESH_CHECK: some answers go to the guards for the check
        }
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        let base = one / PAGE;
        let (taken, _) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
        assert!(!nothing_noted(one, one + SIZE, base, &[0, taken]), "a page never taken");
        let skipped = light_counts()[2];
        assert!(nothing_noted(one, one + SIZE, base, &[taken, taken]));
        assert_eq!(light_counts()[2], skipped + 1);
        assert!(!nothing_noted(one, one + SIZE, base, &[taken, taken - 1]), "written after it was taken");
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_noted(one, one + SIZE, base, &[taken, taken]), "written since: the bits are gone");
        drop_views(section, &[one, two], SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
    }

    /// `VICTORY_LIGHT_REFRESH_CHECK`: a vouched range asked of the guards
    /// anyway counts a sample; a block whose bit stood but that the guards
    /// found written counts a mismatch, and the guards' answer stands.
    #[test]
    fn the_self_check_counts_a_bit_that_should_not_stand() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        let (first, _) = writes_as(one, one + SIZE, true, 2, false, unlocked()).expect("a view");
        let [samples, mismatches, _] = check_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 2, true, unlocked()).expect("a view");
        assert!(since(first, written).is_empty());
        assert_eq!(check_counts()[..2], [samples + 1, mismatches], "a sample, no mismatch");
        // A write, and the bit set again by hand (what must not happen).
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        PUBLISHED.set(one / BLOCK + 1);
        assert!(known(one, one + SIZE));
        let (_, written) = writes_as(one, one + SIZE, true, 2, true, unlocked()).expect("a view");
        assert_eq!(since(second, written), [one / PAGE + 1], "the guards' answer");
        assert_eq!(check_counts()[..2], [samples + 2, mismatches + 1], "a mismatch");
        drop_views(section, &[one, two], SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
    }

    /// `VICTORY_JOURNAL_UNLOCKED`: askers on threads of their own, each
    /// keeping a copy of overlapping parts of the same pages as a host copy
    /// does (a page taken again when the journal says it was written after
    /// the copy last took it), while two threads write those pages. Once
    /// the writes stop, one more refresh each leaves every copy equal to
    /// memory: no write was lost to a kernel asking made beside another.
    /// The same with the journal's one lock held throughout (the reference)
    /// and with it let go; for write-watched memory (the kernel) and for
    /// view memory (the guards and their faults).
    fn copies_under_contention_stay_true(memory: usize, pages: usize, guarded: bool, unlocked: bool) {
        copies_under_contention_stay_true_as(memory, pages, guarded, unlocked, 0);
    }

    /// [`copies_under_contention_stay_true`] with the light refresh as
    /// `light` says (as [`writes_as`] takes it).
    fn copies_under_contention_stay_true_as(memory: usize, pages: usize, guarded: bool, unlocked: bool, light: u8) {
        use std::sync::atomic::{AtomicBool, Ordering};
        const ASKERS: usize = 6;
        const WRITERS: usize = 2;
        const ROUNDS: usize = 300;
        let stop = AtomicBool::new(false);
        let done = std::sync::Barrier::new(ASKERS + 1);
        let still = std::sync::Barrier::new(ASKERS + WRITERS + 1);
        let word = |page: usize, w: usize| (memory + page * PAGE + w * 8) as *mut u64;
        std::thread::scope(|scope| {
            for w in 0..WRITERS {
                let (stop, still) = (&stop, &still);
                scope.spawn(move || {
                    let mut n = 1u64;
                    while !stop.load(Ordering::Relaxed) {
                        let page = (n.wrapping_mul(2_654_435_761).wrapping_add(w as u64 * 7) as usize) % pages;
                        // SAFETY: inside the test's memory.
                        unsafe { word(page, w).write_volatile(n) };
                        n += 1;
                        if n.is_multiple_of(64) {
                            std::thread::yield_now();
                        }
                    }
                    still.wait();
                });
            }
            for a in 0..ASKERS {
                let (done, still) = (&done, &still);
                scope.spawn(move || {
                    // Overlapping parts: the whole, halves, a middle.
                    let (from, to) = match a % 3 {
                        0 => (0, pages),
                        1 => (0, pages / 2 + 1),
                        _ => (pages / 4, pages),
                    };
                    let mut copy = vec![[0u64; WRITERS]; pages];
                    let mut taken = vec![0u64; pages];
                    let refresh = |copy: &mut Vec<[u64; WRITERS]>, taken: &mut Vec<u64>| {
                        let (now, written) =
                            writes_as(word(from, 0) as usize, word(to, 0) as usize, guarded, light, false, unlocked)
                                .expect("tracked memory");
                        for page in from..to {
                            let stamp = written.iter().find(|w| w.0 == memory / PAGE + page).map_or(0, |w| w.1);
                            if taken[page] == 0 || stamp > taken[page] {
                                for (w, value) in copy[page].iter_mut().enumerate() {
                                    // SAFETY: inside the test's memory.
                                    *value = unsafe { word(page, w).read_volatile() };
                                }
                                taken[page] = now;
                            }
                        }
                    };
                    for _ in 0..ROUNDS {
                        refresh(&mut copy, &mut taken);
                    }
                    done.wait();
                    still.wait();
                    refresh(&mut copy, &mut taken);
                    for (page, values) in copy.iter().enumerate().take(to).skip(from) {
                        for (w, &value) in values.iter().enumerate() {
                            // SAFETY: inside the test's memory.
                            let now = unsafe { word(page, w).read_volatile() };
                            assert_eq!(value, now, "asker {a}, page {page}, word {w} (unlocked {unlocked})");
                        }
                    }
                });
            }
            done.wait();
            stop.store(true, Ordering::Relaxed);
            still.wait();
        });
    }

    #[test]
    fn write_watched_copies_under_contention_stay_true() {
        const PAGES: usize = 24;
        for unlocked in [false, true] {
            // SAFETY: a fresh write-watched allocation, freed at the end.
            let p = unsafe {
                VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
            } as usize;
            assert_ne!(p, 0);
            forget_pages(p, p + PAGES * PAGE);
            copies_under_contention_stay_true(p, PAGES, false, unlocked);
            forget_pages(p, p + PAGES * PAGE);
            // SAFETY: our allocation.
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn view_copies_under_contention_stay_true() {
        const PAGES: usize = 24;
        for unlocked in [false, true] {
            let (section, one, two) = two_views(PAGES * BLOCK);
            forget_pages(one, one + PAGES * BLOCK);
            copies_under_contention_stay_true(one, PAGES, true, unlocked);
            drop_views(section, &[one, two], PAGES * BLOCK);
            forget_pages(one, one + PAGES * BLOCK);
            forget_pages(two, two + PAGES * BLOCK);
        }
    }

    /// `VICTORY_JOURNAL_UNLOCKED` with `VICTORY_LIGHT_REFRESH_CHECK`: an
    /// asking that keeps the journal's lock (the self-check's) answers only
    /// once an asking in flight over its range has noted what it found: that
    /// one may have reset the pages first.
    #[test]
    fn a_locked_asking_waits_for_an_asking_in_flight() {
        use std::sync::atomic::{AtomicBool, Ordering};
        const PAGES: usize = 4;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        forget_pages(p, p + PAGES * PAGE);
        let (first, last) = (p / PAGE, p / PAGE + PAGES);
        let other = {
            let mut j = journal();
            let n = j.bump();
            j.flying.push((first, last, n));
            n
        };
        let answered = AtomicBool::new(false);
        let answer = std::thread::scope(|scope| {
            let checked = scope.spawn(|| {
                // The self-check's way: `check` keeps the lock.
                let answer = writes_as(p, p + PAGES * PAGE, false, 0, true, true);
                answered.store(true, Ordering::SeqCst);
                answer
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            let early = answered.load(Ordering::SeqCst);
            {
                let mut j = journal();
                note_written(&mut j, first + 2, other, ALL_PIECES);
                j.flying.retain(|&(_, _, n)| n != other);
                LANDED_ASKING.notify_all();
            }
            let answer = checked.join().expect("the checked asking");
            assert!(!early, "answered before the asking in flight noted what it found");
            answer
        });
        let (_, written) = answer.expect("write-watched");
        assert_eq!(written, [(first + 2, other)]);
        forget_pages(p, p + PAGES * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// `VICTORY_JOURNAL_UNLOCKED` with the light refresh: a block an asking
    /// in flight covered when the guards made it clean gets its bit once
    /// that asking lands, not only at the block's next write.
    #[test]
    fn a_block_passed_over_for_an_asking_in_flight_gets_its_bit_later() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        let other = {
            let mut j = journal();
            let n = j.bump();
            j.flying.push((one / PAGE, one / PAGE + 2, n));
            n
        };
        std::thread::scope(|scope| {
            // The guards take the range in and its answer waits for the
            // asking in flight.
            let asking = scope.spawn(|| writes_as(one, one + SIZE, true, 2, false, false));
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!known(one, one + SIZE), "passed over: an asking in flight covers it");
            {
                let mut j = journal();
                land(&mut j, other);
            }
            assert!(asking.join().expect("the asking").is_some());
        });
        assert!(known(one, one + SIZE), "its bits once that asking landed");
        drop_views(section, &[one, two], SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
    }

    /// `VICTORY_ASK_ONCE`: a lone range asked under the standing asking
    /// after another asking took a page of it, and the title wrote the page
    /// again: the page is noted above what that copy took.
    #[test]
    fn a_lone_range_is_noted_above_askings_made_since_the_standing_one() {
        const PAGES: usize = 2;
        let alloc = || {
            // SAFETY: a fresh write-watched allocation, freed at the end.
            let p = unsafe {
                VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
            } as usize;
            assert_ne!(p, 0);
            forget_pages(p, p + PAGES * PAGE);
            p
        };
        let (a, lone) = (alloc(), alloc());
        // The standing asking (two ranges of one allocation, asked ahead).
        ask_once_end();
        ask_ahead(&[(a, a + PAGE), (a + PAGE, a + 2 * PAGE)]);
        // Another asking takes the lone range's page; the title writes it
        // again.
        // SAFETY: committed pages of ours.
        unsafe { (lone as *mut u8).write_volatile(1) };
        let (taken, _) = writes(lone, lone + PAGE, false).expect("write-watched");
        // SAFETY: as above.
        unsafe { (lone as *mut u8).write_volatile(2) };
        ask_lone(&[(lone, lone + PAGE)]);
        let stamp = journal().written.get(&(lone / PAGE)).copied().unwrap_or(0);
        ask_once_end();
        assert!(stamp > taken, "noted at {stamp}, the copy took the page at {taken}");
        for p in [a, lone] {
            forget_pages(p, p + PAGES * PAGE);
            // SAFETY: our allocation.
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// `VICTORY_ASK_ONCE_VIEWS`: view memory asked under a standing asking
    /// after another asking took a block of it, and the title wrote the
    /// block again: the block is noted above what that copy took.
    #[test]
    fn a_view_asked_once_is_noted_above_askings_made_since() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        // Guarded once, and a standing asking of the other view's memory.
        ask_once_end();
        ask_views(&[(one, one + SIZE)]);
        ask_once_end();
        let (s2, v2, w2) = two_views(SIZE);
        ask_views(&[(v2, v2 + SIZE)]);
        // Another asking takes the block; the title writes it again.
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { (one as *mut u8).write_volatile(1) };
        let (taken, _) = writes_as(one, one + BLOCK, true, 0, false, false).expect("a view");
        // SAFETY: as above.
        unsafe { (one as *mut u8).write_volatile(2) };
        ask_views(&[(one, one + BLOCK)]);
        let stamp = journal().written.get(&(one / PAGE)).copied().unwrap_or(0);
        ask_once_end();
        assert!(stamp > taken, "noted at {stamp}, the copy took the block at {taken}");
        drop_views(s2, &[v2, w2], SIZE);
        drop_views(section, &[one, two], SIZE);
        for p in [one, two, v2, w2] {
            forget_pages(p, p + SIZE);
        }
    }

    /// `VICTORY_GUARD_OUTSIDE` without `VICTORY_JOURNAL_UNLOCKED`: a block
    /// the journal heard of before it was read-only again, a standing
    /// answer under a number raised above that ([`wrote`]) taking it in
    /// that moment, and a write landing unfaulted: once the block stands it
    /// is noted above that answer's number, so that copy takes it again.
    #[test]
    fn an_armed_block_is_noted_again_once_it_stands() {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        static AT: AtomicUsize = AtomicUsize::new(0);
        static RAISED: AtomicU64 = AtomicU64::new(0);
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        // SAFETY: a pagefile section of two blocks, mapped read-write, told
        // of no other view.
        let (section, view) = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize)
        };
        assert_ne!(view, 0);
        mapped(view, SIZE);
        forget_pages(view, view + SIZE);
        GUARD_OUTSIDE_TEST.store(2, Ordering::Relaxed);
        assert!(writes_as(view, view + SIZE, true, 0, false, false).is_some());
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { (view as *mut u64).write_volatile(1) };
        AT.store(view, Ordering::SeqCst);
        fn meanwhile() {
            let mut j = journal();
            j.ahead_asked = j.bump();
            RAISED.store(j.asked(), Ordering::SeqCst);
            drop(j);
            // The block is not read-only yet: no fault.
            // SAFETY: inside the test's view.
            unsafe { (AT.load(Ordering::SeqCst) as *mut u64).write_volatile(2) };
        }
        BEFORE_FINISH.with(|h| h.set(Some(meanwhile)));
        assert!(writes_as(view, view + SIZE, true, 0, false, false).is_some());
        let hooked = BEFORE_FINISH.with(|h| h.take()).is_none();
        GUARD_OUTSIDE_TEST.store(0, Ordering::Relaxed);
        let stamp = journal().written.get(&(view / PAGE)).copied().unwrap_or(0);
        title_memory_remapped(view, SIZE);
        // SAFETY: our view and section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        forget_pages(view, view + SIZE);
        assert!(hooked, "the block was armed outside the locks");
        let raised = RAISED.load(Ordering::SeqCst);
        assert!(stamp > raised, "noted at {stamp}, a standing answer took it at {raised}");
    }

    /// `VICTORY_ASK_SKIP_VIEWS`: the submission's one asking makes no kernel
    /// call for view memory the guards know (it could only fail), and every
    /// copy of views kept beside writers under contention stays true with
    /// it, the journal's lock held or let go.
    #[test]
    fn known_views_are_not_asked_of_the_kernel() {
        use std::sync::atomic::Ordering;
        let _turn = light_turn();
        const PAGES: usize = 8;
        let (section, one, two) = two_views(PAGES * BLOCK);
        forget_pages(one, one + PAGES * BLOCK);
        SKIP_VIEWS_TEST.store(true, Ordering::Relaxed);
        // Guarded once: the guards know it.
        assert!(writes_as(one, one + PAGES * BLOCK, true, 0, false, false).is_some());
        let ranges: Vec<(usize, usize)> = (0..PAGES).map(|k| (one + k * BLOCK, one + (k + 1) * BLOCK)).collect();
        let skipped = KERNEL_CALLS[2].load(Ordering::Relaxed);
        ask_once_end();
        ask_ahead(&ranges);
        ask_lone(&ranges);
        ask_once_end();
        let now = KERNEL_CALLS[2].load(Ordering::Relaxed);
        for unlocked in [false, true] {
            copies_under_contention_stay_true(one, PAGES, true, unlocked);
        }
        SKIP_VIEWS_TEST.store(false, Ordering::Relaxed);
        drop_views(section, &[one, two], PAGES * BLOCK);
        forget_pages(one, one + PAGES * BLOCK);
        forget_pages(two, two + PAGES * BLOCK);
        assert!(now >= skipped + 2 * PAGES as u64, "every range left out, ahead and alone");
    }

    /// `VICTORY_HOLES_KNOWN`: a hole in a placeholder reservation is asked
    /// of Windows once, then known until the runtime tells of a mapping
    /// there, and then found mapped; a hole in an ordinary reservation is
    /// asked every time.
    #[test]
    fn holes_in_a_placeholder_are_asked_once_until_told() {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER,
            MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _turn = light_turn();
        HOLES_KNOWN_TEST.store(true, Ordering::Relaxed);
        // A placeholder of two blocks, split: a view over the first, the
        // second a hole.
        // SAFETY: a fresh placeholder of ours.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, 2 * BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            let view =
                MapViewOfFile3(section, GetCurrentProcess(), p as *const _, 0, BLOCK, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert_eq!(view.Value as usize, p);
            (p, section, view)
        };
        title_memory_remapped(p, 2 * BLOCK);
        forget_pages(p, p + 2 * BLOCK);
        let hole = p + BLOCK;
        assert!(placeholder(hole / BLOCK), "a placeholder");
        assert!(guard(p, p + 2 * BLOCK).is_some());
        assert!(guards().absent.contains(&(hole / BLOCK)), "noted unmapped");
        assert!(guards().holes.contains(&(hole / BLOCK)), "known a hole");
        let cached = KERNEL_CALLS[4].load(Ordering::Relaxed);
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()), "nothing written, the hole still one");
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        assert!(KERNEL_CALLS[4].load(Ordering::Relaxed) >= cached + 2, "answered from the cache, no question to Windows");
        // The runtime maps a view there and tells: the block is written.
        // SAFETY: the hole is our placeholder; a view of the section's second block over it.
        let second = unsafe {
            MapViewOfFile3(section, GetCurrentProcess(), hole as *const _, BLOCK as u64, BLOCK, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0)
        };
        assert_eq!(second.Value as usize, hole);
        title_memory_remapped(hole, BLOCK);
        let written = guard(p, p + 2 * BLOCK).expect("a view");
        assert!(written.contains(&(hole / PAGE)), "mapped since: taken");
        HOLES_KNOWN_TEST.store(false, Ordering::Relaxed);
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our views, placeholder and section.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
            VirtualFree(hole as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        forget_pages(p, p + 2 * BLOCK);
        // An ordinary reservation is no placeholder.
        // SAFETY: a reservation of ours, freed at once.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert!(!placeholder(plain / BLOCK));
        // SAFETY: ours.
        unsafe { VirtualFree(plain as *mut _, 0, MEM_RELEASE) };
    }

    /// `VICTORY_JOURNAL_UNLOCKED` with the light refresh: an asking in flight
    /// finds a write made after an answer the bits gave meanwhile (with no
    /// wait for it); it notes the page above that answer's number, so that
    /// copy takes it again.
    #[test]
    fn a_write_after_an_answer_by_the_bits_is_noted_above_it() {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        static AT: AtomicUsize = AtomicUsize::new(0);
        static ANSWERED: AtomicU64 = AtomicU64::new(0);
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        // Guarded, clean, known by the bits.
        assert!(writes_as(one, one + SIZE, true, 2, false, false).is_some());
        assert!(known(one, one + SIZE));
        AT.store(one, Ordering::SeqCst);
        fn meanwhile() {
            let at = AT.load(Ordering::SeqCst);
            let (n, _) = writes_as(at, at + 2 * BLOCK, true, 2, false, false).expect("a view");
            ANSWERED.store(n, Ordering::SeqCst);
            // The title writes the block after that answer.
            // SAFETY: inside the test's view; the write faults and opens it.
            unsafe { ((at + 8) as *mut u8).write_volatile(1) };
        }
        *AFTER_REGISTER.lock().unwrap_or_else(|e| e.into_inner()) = Some((one / PAGE, meanwhile));
        assert!(writes_as(one, one + SIZE, true, 0, false, true).is_some());
        let stamp = journal().written.get(&(one / PAGE)).copied().unwrap_or(0);
        drop_views(section, &[one, two], SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
        let answered = ANSWERED.load(Ordering::SeqCst);
        assert!(answered != 0, "the hook ran");
        assert!(stamp > answered, "noted at {stamp}, an answer by the bits gave {answered}");
    }

    /// `VICTORY_STAMPS_FLAT` (with its check): copies of write-watched and of
    /// view memory kept beside writers stay true, the journal's lock held or
    /// let go, the light refresh's bits asked (answers with no lock) or not;
    /// and the table answered as the journal's map every time it was
    /// compared.
    #[test]
    fn flat_stamps_answer_as_the_journal_under_contention() {
        use std::sync::atomic::Ordering;
        let _turn = light_turn();
        const PAGES: usize = 16;
        STAMPS_FLAT_TEST.store(true, Ordering::Relaxed);
        let differences = STAMP_CHECKS[1].load(Ordering::Relaxed);
        let lock_free = STAMP_CHECKS[2].load(Ordering::Relaxed);
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        forget_pages(p, p + PAGES * PAGE);
        for unlocked in [false, true] {
            copies_under_contention_stay_true(p, PAGES, false, unlocked);
        }
        forget_pages(p, p + PAGES * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        let (section, one, two) = two_views(PAGES * BLOCK);
        forget_pages(one, one + PAGES * BLOCK);
        for (unlocked, light) in [(false, 0), (true, 0), (false, 2), (true, 2)] {
            copies_under_contention_stay_true_as(one, PAGES, true, unlocked, light);
        }
        drop_views(section, &[one, two], PAGES * BLOCK);
        forget_pages(one, one + PAGES * BLOCK);
        forget_pages(two, two + PAGES * BLOCK);
        STAMPS_FLAT_TEST.store(false, Ordering::Relaxed);
        assert_eq!(STAMP_CHECKS[1].load(Ordering::Relaxed), differences, "the table and the map differed");
        assert!(STAMP_CHECKS[2].load(Ordering::Relaxed) > lock_free, "some answers came with no lock");
    }

    /// `VICTORY_STAMPS_FLAT`: the table holds what the journal's map holds,
    /// stamp for stamp, and never lowers one; a refresh the bits and the
    /// table show has nothing new is answered with no lock, one with a
    /// newer page is not.
    #[test]
    fn the_flat_table_is_the_map() {
        let _turn = light_turn();
        let base = (1 << 26) + 77;
        // Far from any memory of ours: only the table and the map.
        {
            let mut j = journal();
            note_written(&mut j, base, 10, ALL_PIECES);
            note_written(&mut j, base + 3, 12, ALL_PIECES);
            note_written(&mut j, base + 3, 11, ALL_PIECES);
            note_written(&mut j, base + STAMP_PAGES, 9, ALL_PIECES);
            let map: Vec<(usize, u64)> =
                j.written.range(base..base + STAMP_PAGES + 1).map(|(&p, &w)| (p, w)).collect();
            assert_eq!(map, STAMPS.range(base, base + STAMP_PAGES + 1));
            assert_eq!(STAMPS.get(base + 3), 12, "never lowered");
        }
        forget_pages(base * PAGE, (base + STAMP_PAGES + 1) * PAGE);
        assert!(STAMPS.range(base, base + STAMP_PAGES + 1).is_empty());
        // A view known by its bits: nothing new for a copy that holds it.
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        forget_pages(one, one + SIZE);
        let (taken, _) = writes_as(one, one + SIZE, true, 2, false, false).expect("a view");
        assert!(known(one, one + SIZE));
        let first = one / PAGE;
        assert!(nothing_flat(one, one + SIZE, first, &[taken, taken]));
        assert!(!nothing_flat(one, one + SIZE, first, &[taken, taken - 1]), "a page written after the copy took it");
        assert!(!nothing_flat(one, one + SIZE, first, &[0, taken]), "a page never taken");
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_flat(one, one + SIZE, first, &[taken, taken]), "written since: the bits are gone");
        drop_views(section, &[one, two], SIZE);
        forget_pages(one, one + SIZE);
        forget_pages(two, two + SIZE);
    }

    /// `VICTORY_JOURNAL_UNLOCKED`, the review finding on a2b96a048: two
    /// askings of one range, the lower-numbered one (`low`) started first,
    /// the higher (`high`) reaching the kernel first. `high` resets the
    /// page the title wrote, so `low`'s kernel call does not see the write;
    /// `low`, done and out of flight, must not answer before `high` notes
    /// it, or its copy misses the write until its next refresh. a2b96a048
    /// waited only for askings numbered below the answer (it answered at
    /// once here); every asking in flight over the range is waited for now.
    #[test]
    fn an_answer_waits_for_a_later_asking_that_reset_first() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        const PAGES: usize = 4;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        forget_pages(p, p + PAGES * PAGE);
        let (first, last) = (p / PAGE, p / PAGE + PAGES);
        assert!(writes_as(p, p + PAGES * PAGE, false, 0, false, true).is_some());
        // The title writes page 1.
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        // Both in flight, numbered as `writes_unlocked` numbers them.
        let (low, high) = {
            let mut j = journal();
            let low = j.bump();
            j.flying.push((first, last, low));
            let high = j.bump();
            j.flying.push((first, last, high));
            (low, high)
        };
        // `high`'s kernel call first: it takes the write.
        let mut found = vec![0usize; PAGES * PAGE / KERNEL_PAGE];
        let (mut count, mut granularity) = (found.len(), 0);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(1, p as *const _, PAGES * PAGE, found.as_mut_ptr().cast(), &mut count, &mut granularity)
        } != 0;
        assert!(!failed && count > 0, "the write was there to reset");
        // `low`'s: nothing there now. It notes nothing, is out of flight,
        // and answers as `writes_unlocked` does.
        count = found.len();
        // SAFETY: as above.
        unsafe {
            GetWriteWatch(1, p as *const _, PAGES * PAGE, found.as_mut_ptr().cast(), &mut count, &mut granularity)
        };
        assert_eq!(count, 0);
        journal().flying.retain(|&(_, _, n)| n != low);
        let answered = AtomicBool::new(false);
        let written = std::thread::scope(|scope| {
            let answer = scope.spawn(|| {
                let held = settle(journal(), first, last);
                let written: Vec<(usize, u64)> = answer(&held, first, last);
                answered.store(true, Ordering::SeqCst);
                written
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            let early = answered.load(Ordering::SeqCst);
            {
                let mut j = journal();
                note_written(&mut j, first + 1, high, ALL_PIECES);
                j.flying.retain(|&(_, _, n)| n != high);
                LANDED_ASKING.notify_all();
            }
            let written = answer.join().expect("the answer");
            assert!(!early, "answered before the asking that reset the page noted it");
            written
        });
        assert_eq!(written, [(first + 1, high)], "the write the other asking found");
        forget_pages(p, p + PAGES * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// `VICTORY_JOURNAL_UNLOCKED`: an answer waits for an asking in flight
    /// numbered below it over the same pages, and holds what that asking
    /// notes; an asking in flight elsewhere does not hold it up.
    #[test]
    fn an_answer_waits_for_an_earlier_asking_in_flight() {
        use std::sync::atomic::{AtomicBool, Ordering};
        const PAGES: usize = 4;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        forget_pages(p, p + PAGES * PAGE);
        let (first, last) = (p / PAGE, p / PAGE + PAGES);
        // Elsewhere: no wait.
        let away = {
            let mut j = journal();
            let n = j.bump();
            j.flying.push((last + 100, last + 101, n));
            n
        };
        assert!(writes_as(p, p + PAGES * PAGE, false, 0, false, true).is_some());
        // Over these pages: an asking in flight, numbered below the next.
        let early = {
            let mut j = journal();
            let n = j.bump();
            j.flying.push((first + 1, first + 2, n));
            n
        };
        let answered = AtomicBool::new(false);
        let answer = std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let answer = writes_as(p, p + PAGES * PAGE, false, 0, false, true);
                answered.store(true, Ordering::SeqCst);
                answer
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!answered.load(Ordering::SeqCst), "answered before the asking in flight landed");
            {
                let mut j = journal();
                note_written(&mut j, first + 1, early, ALL_PIECES);
                j.flying.retain(|&(_, _, n)| n != early && n != away);
                LANDED_ASKING.notify_all();
            }
            waiter.join().expect("the waiter")
        });
        let (now, written) = answer.expect("write-watched");
        assert!(now > early);
        assert_eq!(written, [(first + 1, early)], "what the asking in flight noted");
        forget_pages(p, p + PAGES * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
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

    #[test]
    fn resting_faults_keep_other_views_dirty_until_all_views_are_rearmed() {
        use std::time::Duration;
        if super::guard_aliases() < 2 {
            return;
        }
        let (section, reader, writer) = two_views(BLOCK);
        guard(reader, reader + BLOCK).unwrap();
        let block = reader / BLOCK;
        {
            let mut g = super::guards();
            assert!(super::open_fault(&mut g, writer, true, Duration::from_secs(60)));
            assert!(g.protected.contains(&block));
            assert!(!g.protected.contains(&(writer / BLOCK)));
            for _ in 0..2 {
                assert!(!super::guard_after_faults(&mut g, block, block + 1, Duration::from_secs(60), None).is_empty());
                assert!(!g.protected.contains(&(writer / BLOCK)));
            }
            assert!(!super::guard_after_faults(&mut g, block, block + 1, Duration::ZERO, None).is_empty());
            assert!(g.protected.contains(&(writer / BLOCK)));
            assert!(super::guard_after_faults(&mut g, block, block + 1, Duration::ZERO, None).is_empty());
        }
        drop_views(section, &[reader, writer], BLOCK);
    }

    /// Faulted blocks next to each other rest and are made read-only again
    /// as one run, at both views; a block of the range that did not fault
    /// stays as it was.
    #[test]
    fn rested_faults_next_to_each_other_are_rearmed_together() {
        use std::time::Duration;
        if super::guard_aliases() < 2 {
            return;
        }
        let (section, reader, writer) = two_views(3 * BLOCK);
        guard(reader, reader + 3 * BLOCK).unwrap();
        let (r, w) = (reader / BLOCK, writer / BLOCK);
        let pages = |from: usize, to: usize| (from * BLOCK / PAGE..to * BLOCK / PAGE).collect::<Vec<_>>();
        {
            let mut g = super::guards();
            // The title writes the first two blocks through its own view.
            assert!(super::open_fault(&mut g, writer, true, Duration::from_secs(60)));
            assert!(super::open_fault(&mut g, writer + BLOCK, true, Duration::from_secs(60)));
            // Resting: told as written, the written view left open.
            assert_eq!(super::guard_after_faults(&mut g, r, r + 3, Duration::from_secs(60), None), pages(r, r + 2));
            assert!(!g.protected.contains(&w) && !g.protected.contains(&(w + 1)));
            // Rested: told once more, and read-only again at both views.
            assert_eq!(super::guard_after_faults(&mut g, r, r + 3, Duration::ZERO, None), pages(r, r + 2));
            assert!((0..3).all(|i| g.protected.contains(&(r + i)) && g.protected.contains(&(w + i))));
            assert!(g.faulted.range(r..r + 3).next().is_none());
            assert!(super::guard_after_faults(&mut g, r, r + 3, Duration::ZERO, None).is_empty());
            // The written view's own guard has not taken them yet.
            assert_eq!(g.faulted.range(w..w + 3).count(), 2);
            let taken = super::guard_after_faults(&mut g, w, w + 3, Duration::ZERO, None);
            assert_eq!(taken[..2 * BLOCK / PAGE], pages(w, w + 2));
            assert!(g.faulted.range(w..w + 3).next().is_none());
            assert!(super::guard_after_faults(&mut g, w, w + 3, Duration::ZERO, None).is_empty());
        }
        drop_views(section, &[reader, writer], 3 * BLOCK);
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

    /// The runtime left a view as it was for a mapping of the pages it
    /// shows already (`VICTORY_REMAP_SAME=1`): its guarded blocks are
    /// read-write again and the next check takes all their pages as
    /// written, as after a view made anew; no block stays read-only behind
    /// the guards' back.
    #[test]
    fn a_mapping_of_what_is_shown_already_opens_the_guards_and_reports_the_pages() {
        const SIZE: usize = 2 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        let all: Vec<usize> = (view / PAGE..(view + SIZE) / PAGE).collect();
        assert_eq!(guard(view, view + SIZE), Some(all.clone()));
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()), "nothing written since");
        assert!(!writable(view), "guarded: read-only");
        title_memory_same(view, SIZE);
        assert!(writable(view) && writable(view + BLOCK), "opened, not left read-only");
        // SAFETY: the view, read-write again; no fault is needed for this.
        unsafe { ((view + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(view, view + SIZE), Some(all), "taken as written, as after a mapping");
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        drop_views(section, &[view, other], SIZE);
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

    /// A block set answers as a set of numbers does, across the words and
    /// chunks of its bits: inserts, removals, ranges added and taken out,
    /// and the members and non-members of ranges.
    #[test]
    fn a_block_set_is_a_set_of_numbers() {
        let mut set = BlockSet::new();
        let mut model = BTreeSet::new();
        // A small generator of the test's own (no randomness crate here).
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let base = 7 * CHUNK_BLOCKS - 300;
        for _ in 0..3000 {
            let a = base + next(3 * CHUNK_BLOCKS);
            let b = a + next(200);
            match next(5) {
                0 => {
                    set.insert(a);
                    model.insert(a);
                }
                1 => {
                    set.remove(&a);
                    model.remove(&a);
                }
                2 => {
                    set.extend(a..b);
                    model.extend(a..b);
                }
                3 => {
                    set.remove_range(a..b);
                    let gone: Vec<usize> = model.range(a..b).copied().collect();
                    for x in gone {
                        model.remove(&x);
                    }
                }
                _ => {}
            }
            let (c, d) = (base + next(3 * CHUNK_BLOCKS), 0);
            let d = c + d + next(300);
            assert_eq!(set.range_vec(c..d), model.range(c..d).copied().collect::<Vec<_>>());
            assert_eq!(set.missing(c..d), (c..d).filter(|x| !model.contains(x)).collect::<Vec<_>>());
            assert_eq!(set.contains(&c), model.contains(&c));
        }
    }

    /// Benchmark (ignored): a guard check of 256 MiB of view memory all
    /// clean, and with every 32nd block written, guarded again each time.
    /// `cargo test ... guard_check_bench -- --ignored --nocapture`, with and
    /// without `VICTORY_GUARD_BITS=0`.
    #[test]
    #[ignore]
    fn guard_check_bench() {
        const SIZE: usize = 4096 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        assert!(guard(view, view + SIZE).is_some());
        let rounds = 200;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        }
        let clean = started.elapsed() / rounds;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            for block in (0..4096).step_by(32) {
                // SAFETY: inside our view; the guard takes the fault.
                unsafe { ((view + block * BLOCK) as *mut u8).write_volatile(1) };
            }
            assert_eq!(guard(view, view + SIZE).map(|w| w.len()), Some(128 * BLOCK / PAGE));
        }
        let written = started.elapsed() / rounds;
        println!("guard check of 4,096 blocks: all clean {clean:?}, 128 written {written:?} (bits {})", guard_bits());
        drop_views(section, &[view, other], SIZE);
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
                    mapped(at, SIZE);
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

    /// The runtime's notice of a new mapping comes after the mapping itself
    /// (`xmem`'s `map`), and a guard check of another thread can fall
    /// between the two: it makes the new view read-only and notes it
    /// guarded. The notice gives the write access back. Before, it only
    /// forgot the blocks: they stayed read-only with no guard to open them,
    /// and the title's next write there was an access violation nothing
    /// took.
    #[test]
    fn a_notice_after_a_guard_check_leaves_the_view_writable() {
        const SIZE: usize = 2 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        let pages = |block: usize| (view + block * BLOCK) / PAGE;
        // The check that fell between the mapping and its notice.
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]));
        assert!(!writable(view) && !writable(view + BLOCK), "guarded: read-only");
        // The notice, and what `xmem` tells after it.
        title_memory_remapped(view, SIZE);
        if reopen_remapped() {
            assert!(writable(view) && writable(view + BLOCK), "left read-only with no guard to open it");
            // SAFETY: the view, read-write again; no fault to take.
            unsafe { ((view + BLOCK + 8) as *mut u8).write_volatile(1) };
        }
        assert!(!guards().protected.contains(&(view / BLOCK)), "forgotten");
        crate::inplace::title_memory_view(view, SIZE, section, 0);
        // Not guarded any more: the next check takes all of it again.
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        drop_views(section, &[view, other], SIZE);
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
        // The journal is one for the process, and this address may have been
        // another test's until it freed it.
        super::forget_pages(p, p + PAGE);

        // So is the journal's standing asking: a submission of a test
        // running beside this one ends it (`ask_once_end`) or notes writes
        // of its own under it, and the answers below are then not this
        // test's alone. Such a round is made again.
        let mut undisturbed = None;
        for _ in 0..1000 {
            ask_once(&[(p, p + PAGE)]);
            wrote(p, PAGE);
            let (now, written) = writes(p, p + PAGE, false).expect("write-watched page");
            let from_journal = asked(p, p + PAGE);
            ask_once_end();
            if let Some((asked_now, asked_written)) = from_journal
                && asked_now == now
            {
                undisturbed = Some((now, written, asked_written));
                break;
            }
            std::thread::yield_now();
        }

        super::forget_pages(p, p + PAGE);
        // SAFETY: freeing our private reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };

        let (now, written, asked_written) =
            undisturbed.expect("writes() and asked() agree on the asking that stands after wrote()");
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
            written: Default::default(),
            found: Vec::new(),
            ahead: Vec::new(),
            ahead_asked: 0,
            kept: false,
            pieces: Default::default(),
            flying: Vec::new(),
            unpublished: Vec::new(),
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

#[cfg(test)]
mod landed_tests {
    use super::Landed;

    /// The frames pages were last landed in (`VICTORY_FILL_TITLE`; after
    /// avatarmanz/VIctory-7900XTX 9fea7434c's test).
    #[test]
    fn landed_pages_are_known_by_their_frame() {
        static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);
        let landed = &LANDED;
        for page in [0, 5, Landed::PAGES - 1, Landed::PAGES, 7 * Landed::PAGES + 3] {
            assert!(!landed.since(page, 0, 100));
            landed.note(page, 100);
            assert!(landed.since(page, 100, 100) && landed.since(page, 0, 100) && landed.since(page, 90, 110));
            assert!(!landed.since(page, 101, 110));
            landed.note(page, 0x1_0000_0005);
            assert!(landed.since(page, 0x1_0000_0000, 0x1_0000_0007) && !landed.since(page, 0x1_0000_0006, 0x1_0000_0007));
            assert!(!landed.since(page + 1, 0, 0x1_0000_0007));
        }
        landed.note(usize::MAX / 4096, 1);
        assert!(!landed.since(usize::MAX / 4096, 0, 1));
    }
}
