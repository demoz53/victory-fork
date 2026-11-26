//! `VICTORY_METERS=1` (off by default; `2` the detailed form): exact meters
//! of the card's time, the submitting thread's time and the recording
//! threads' time, `METERS` lines on the 5 s frame line and a summary of the
//! run at the title's exit ([`final_report`]).
//!
//! - The card, per host submission (`Host::submit`): a stamp list of ours
//!   before its lists and one after them (pre-recorded pairs in a ring of
//!   [`PAIRS`]), so the submission's interval; busy time is the union of
//!   those intervals and of our lists sent on their own (below).
//! - The card, inside the lists: every list of ours and every title host
//!   list writes a time stamp where what it records changes kind
//!   ([`Kind`]: the title's draws, dispatches, copies, clears and resolves;
//!   our texture uploads and copies, buffer copies, copy-backs, hash and
//!   compare passes, indirect clamps, other work of ours in a title list),
//!   and the title's draws start a new segment at each change of render
//!   targets (a pass), its dispatches at each kind change (`=2`: at each
//!   change of pipeline, and the title's barriers as segments of their own).
//!   Stamps go into blocks of [`BLOCK`] queries of one heap, each block
//!   resolved by the list that wrote it; a block is used again once the
//!   fence after the list's submission is reached. A segment runs from its
//!   stamp to the next of its list. What a submission's interval holds
//!   beyond its stamped segments (state, barriers before a list's first
//!   stamp, lists without stamps) is "everything else", or "our own passes"
//!   for a submission that holds no title list.
//! - The submitting thread: each title call of a queue's
//!   `ExecuteCommandLists` and of `PresentX` is timed on its thread: wall
//!   time, the thread's cycles and its kernel and user times
//!   (`GetThreadTimes`, whose ratio splits the cycles). Time between two such
//!   calls is the title's own work (and our other calls).
//! - The recording threads: each list call through `list::with_state` is
//!   timed with the time-stamp counter, summed per thread.
//!
//! What the stamps tell apart: a time stamp is written when the card has
//! finished everything recorded before it, so a segment is the time from
//! the end of the work before it to the end of its own work: work that
//! overlaps on the card (a dispatch that runs beside the draws before it)
//! is charged to the segment that finishes last, and a wait or a cache
//! flush the card makes at a barrier goes to the segment the barrier is in
//! (with `=2` the title's barriers are segments of their own). On one queue
//! the submissions do not overlap. Stamps cost the card a little: each is
//! a point where it finishes everything before it.
//!
//! Without the switch each hook is one load of a flag set once.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// Pairs of stamps around submissions in the ring.
pub(crate) const PAIRS: usize = 256;
/// Stamps a block of the in-list heap holds, and the blocks.
pub(crate) const BLOCK: u32 = 64;
const BLOCKS: u32 = 4096;

/// `VICTORY_METERS`: 0 off, 1 on, 2 detailed.
pub(crate) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = match std::env::var("VICTORY_METERS").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        if level > 0 {
            tracing::info!("METERS on (VICTORY_METERS={level}): card time per host submission and per kind of work inside the lists, the submitting thread's time, the recording threads' time");
        }
        level
    })
}

/// Whether the meters run.
pub(crate) fn on() -> bool {
    level() > 0
}

fn tsc() -> u64 {
    // SAFETY: reads the time-stamp counter.
    unsafe { core::arch::x86_64::_rdtsc() }
}

// ---- kinds of work ----

/// What a segment of card time holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub(crate) enum Kind {
    TitleDraws,
    TitleDispatches,
    TitleCopies,
    ClearsResolves,
    TitleBarriers,
    Uploads,
    BufferCopies,
    CopyBacks,
    Hashes,
    Clamps,
    OursInList,
    OurPasses,
    Other,
}

pub(crate) const KINDS: usize = 13;
const KIND_NAMES: [&str; KINDS] = [
    "title draws",
    "title dispatches",
    "title copies",
    "clears and resolves",
    "title barriers",
    "our texture uploads and copies",
    "our buffer copies",
    "our copy-backs",
    "our hash and compare passes",
    "our indirect clamps",
    "our other work in title lists",
    "our own passes",
    "everything else",
];

impl Kind {
    fn ours(self) -> bool {
        matches!(
            self,
            Kind::Uploads | Kind::BufferCopies | Kind::CopyBacks | Kind::Hashes | Kind::Clamps | Kind::OursInList | Kind::OurPasses
        )
    }
    fn title(self) -> bool {
        matches!(self, Kind::TitleDraws | Kind::TitleDispatches | Kind::TitleCopies | Kind::ClearsResolves | Kind::TitleBarriers)
    }
    fn pass(self) -> bool {
        matches!(self, Kind::TitleDraws | Kind::TitleDispatches)
    }
    fn name(self) -> &'static str {
        KIND_NAMES[self as usize]
    }
}

// ---- the card ----

/// The ring of submission stamp pairs, the in-list heap, and what was read.
pub(crate) struct Card {
    inner: Mutex<CardInner>,
    inline: Inline,
    /// Ticks a second of the queue's timestamps.
    frequency: u64,
}

/// The heap the lists' own stamps go into, its read-back buffer (mapped)
/// and its free blocks.
struct Inline {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    mapped: *const u64,
    free: Mutex<Vec<u32>>,
}

// SAFETY: free-threaded D3D12 objects; `mapped` points into `back`, which
// lives as long as it, and is only read where the card is done writing.
unsafe impl Send for Inline {}
// SAFETY: as above.
unsafe impl Sync for Inline {}

struct CardInner {
    begins: Vec<ID3D12CommandList>,
    ends: Vec<ID3D12CommandList>,
    _allocator: ID3D12CommandAllocator,
    _heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    /// The read-back buffer, mapped for the whole run.
    mapped: *const u64,
    next: usize,
    used: Vec<bool>,
    pending: VecDeque<Pending>,
    /// Read since the last line.
    done: Vec<Done>,
    /// Submissions not metered: the ring was full.
    skipped: u64,
    /// Lists whose stamps ran out of blocks.
    short: u64,
}

// SAFETY: `mapped` points into `back`, which lives as long as it; the rest
// are free-threaded D3D12 objects, used under the mutex.
unsafe impl Send for CardInner {}

struct Pending {
    /// The submission's pair; `None` for lists of ours sent on their own.
    slot: Option<usize>,
    fence: ID3D12Fence,
    value: u64,
    name: Option<String>,
    records: Vec<ListStamps>,
}

/// One submission (or lists sent on their own) as the card ran it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Done {
    /// The submission's interval (its pair of stamps).
    pub(crate) interval: Option<(u64, u64)>,
    pub(crate) name: Option<String>,
    pub(crate) records: Vec<Record>,
}

/// One list's stamped segments, in card ticks.
#[derive(Clone, Debug, Default)]
pub(crate) struct Record {
    pub(crate) first: u64,
    pub(crate) last: u64,
    /// (kind, ticks, the pass it is, the kind of the segment before it).
    pub(crate) segs: Vec<(Kind, u64, Option<Pass>, Option<Kind>)>,
}

/// A slot taken for one submission: run `begin` before its lists and `end`
/// after them, then tell [`Card::sent`] the fence value signalled after.
pub(crate) struct Stamp {
    slot: usize,
    pub(crate) begin: ID3D12CommandList,
    pub(crate) end: ID3D12CommandList,
}

static CARD: OnceLock<Option<Card>> = OnceLock::new();

/// The card's meter if it was made.
pub(crate) fn made_card() -> Option<&'static Card> {
    CARD.get().and_then(Option::as_ref)
}

/// The card's meter, made at its first use on `device` for `queue`.
pub(crate) fn card(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Option<&'static Card> {
    if !on() {
        return None;
    }
    CARD.get_or_init(|| match Card::new(device, queue) {
        Ok(card) => Some(card),
        Err(e) => {
            tracing::warn!("METERS: no card meter ({e})");
            None
        }
    })
    .as_ref()
}

fn readback(device: &ID3D12Device, bytes: u64) -> windows::core::Result<(ID3D12Resource, *const u64)> {
    // SAFETY: plain object creation with a valid description; the mapping
    // lives as long as the buffer.
    unsafe {
        let mut back: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut back,
        )?;
        let back = back.ok_or_else(windows::core::Error::empty)?;
        let mut mapped = std::ptr::null_mut();
        back.Map(0, None, Some(&mut mapped))?;
        Ok((back, mapped.cast()))
    }
}

fn query_heap(device: &ID3D12Device, count: u32) -> windows::core::Result<ID3D12QueryHeap> {
    let mut heap: Option<ID3D12QueryHeap> = None;
    // SAFETY: plain object creation with a valid description.
    unsafe {
        device.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: count, NodeMask: 0 },
            &mut heap,
        )?
    };
    heap.ok_or_else(windows::core::Error::empty)
}

impl Card {
    pub(crate) fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> windows::core::Result<Card> {
        // SAFETY: plain D3D12 object creation with valid descriptions; the
        // lists are recorded once with objects that live as long as they.
        unsafe {
            let frequency = queue.GetTimestampFrequency()?;
            let heap = query_heap(device, (2 * PAIRS) as u32)?;
            let (back, mapped) = readback(device, (16 * PAIRS) as u64)?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let (mut begins, mut ends) = (Vec::with_capacity(PAIRS), Vec::with_capacity(PAIRS));
            for slot in 0..PAIRS as u32 {
                let begin: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                begin.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * slot);
                begin.Close()?;
                let end: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                end.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * slot + 1);
                end.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * slot, 2, &back, 16 * u64::from(slot));
                end.Close()?;
                begins.push(begin.cast()?);
                ends.push(end.cast()?);
            }
            let inline_heap = query_heap(device, BLOCK * BLOCKS)?;
            let (inline_back, inline_mapped) = readback(device, 8 * u64::from(BLOCK * BLOCKS))?;
            Ok(Card {
                inner: Mutex::new(CardInner {
                    begins,
                    ends,
                    _allocator: allocator,
                    _heap: heap,
                    back,
                    mapped,
                    next: 0,
                    used: vec![false; PAIRS],
                    pending: VecDeque::new(),
                    done: Vec::new(),
                    skipped: 0,
                    short: 0,
                }),
                inline: Inline {
                    heap: inline_heap,
                    back: inline_back,
                    mapped: inline_mapped,
                    free: Mutex::new((0..BLOCKS).rev().collect()),
                },
                frequency: frequency.max(1),
            })
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CardInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A slot for the next submission, or `None` (the ring is full).
    pub(crate) fn stamp(&self) -> Option<Stamp> {
        let mut inner = self.lock();
        self.harvest(&mut inner);
        let slot = inner.next;
        if inner.used[slot] {
            inner.skipped += 1;
            return None;
        }
        inner.used[slot] = true;
        inner.next = (slot + 1) % PAIRS;
        Some(Stamp { slot, begin: inner.begins[slot].clone(), end: inner.ends[slot].clone() })
    }

    /// The submission with `stamp` (if it got one) and the lists' stamps
    /// `records` were sent; `fence` reaches `value` when they have run.
    pub(crate) fn sent(&self, stamp: Option<Stamp>, fence: &ID3D12Fence, value: u64, name: Option<String>, records: Vec<ListStamps>) {
        if stamp.is_none() && records.is_empty() {
            return;
        }
        self.lock().pending.push_back(Pending { slot: stamp.map(|s| s.slot), fence: fence.clone(), value, name, records });
    }

    /// Stamps read since the last call (submissions the card has run), the
    /// submissions left out with a full ring, and the lists whose stamps
    /// ran out of blocks.
    pub(crate) fn take(&self) -> (Vec<Done>, u64, u64) {
        let mut inner = self.lock();
        self.harvest(&mut inner);
        (std::mem::take(&mut inner.done), std::mem::take(&mut inner.skipped), std::mem::take(&mut inner.short))
    }

    pub(crate) fn frequency(&self) -> u64 {
        self.frequency
    }

    fn take_block(&self) -> Option<u32> {
        self.inline.free.lock().unwrap_or_else(|e| e.into_inner()).pop()
    }

    fn give_blocks(&self, blocks: &[u32]) {
        self.inline.free.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(blocks);
    }

    fn inline_at(&self, query: u32) -> u64 {
        // SAFETY: inside the mapped read-back buffer; the list that wrote
        // the query has run (its fence was reached).
        unsafe { self.inline.mapped.add(query as usize).read_volatile() }
    }

    /// Reads the stamps of the submissions the card has run (in order: one
    /// queue), and gives their blocks back.
    fn harvest(&self, inner: &mut CardInner) {
        while let Some(p) = inner.pending.front() {
            // SAFETY: a plain fence query.
            if unsafe { p.fence.GetCompletedValue() } < p.value {
                break;
            }
            let p = inner.pending.pop_front().expect("a front");
            let interval = p.slot.and_then(|slot| {
                inner.used[slot] = false;
                // SAFETY: the two stamps of the slot, resolved by its end
                // list, which has run (the fence after it was reached).
                let (begin, end) =
                    unsafe { (inner.mapped.add(2 * slot).read_volatile(), inner.mapped.add(2 * slot + 1).read_volatile()) };
                (end >= begin && begin > 0).then_some((begin, end))
            });
            let mut records = Vec::with_capacity(p.records.len());
            for stamps in p.records {
                if stamps.full {
                    inner.short += 1;
                }
                records.extend(stamps.read(self));
                self.give_blocks(&stamps.blocks);
            }
            // What came before each list's first segment: the last of the
            // list that ran before it.
            records.sort_by_key(|r| r.first);
            let mut before: Option<Kind> = None;
            for r in &mut records {
                if let Some(first) = r.segs.first_mut()
                    && first.3.is_none()
                {
                    first.3 = before;
                }
                before = r.segs.last().map(|s| s.0).or(before);
            }
            inner.done.push(Done { interval, name: p.name, records });
        }
        let _ = &inner.back;
    }
}

// ---- stamps inside a list ----

/// A pass of the title: its segment's draws or dispatches.
#[derive(Clone, Debug, Default)]
pub(crate) struct Pass {
    /// The pipeline bound at its start.
    pub(crate) name: String,
    /// Its render targets (draws).
    pub(crate) target: String,
    pub(crate) draws: u64,
    pub(crate) dispatches: u64,
    /// Vertices x instances, or thread groups.
    pub(crate) work: u64,
}

#[derive(Clone, Copy)]
struct Seg {
    query: u32,
    kind: Kind,
    pass: u32,
}

const NO_PASS: u32 = u32::MAX;

/// The stamps one list writes as it is recorded.
#[derive(Default)]
pub(crate) struct ListStamps {
    blocks: Vec<u32>,
    used: u32,
    segs: Vec<Seg>,
    passes: Vec<Pass>,
    current: Option<(Kind, u64)>,
    end: Option<u32>,
    full: bool,
    /// The pipeline the list binds now (noted at each change).
    pub(crate) pipeline: String,
}

impl ListStamps {
    fn next(&mut self, card: &Card) -> Option<u32> {
        if self.blocks.is_empty() || self.used == BLOCK {
            let Some(block) = card.take_block() else {
                self.full = true;
                return None;
            };
            self.blocks.push(block);
            self.used = 0;
        }
        let query = self.blocks.last().copied().unwrap_or(0) * BLOCK + self.used;
        self.used += 1;
        Some(query)
    }

    /// The command about to be recorded into `list` is of `kind`; with
    /// another kind or `key` than the last, a segment begins (a stamp). A
    /// pass's counts grow by `add` (draws, dispatches, work).
    pub(crate) fn mark(
        &mut self,
        card: &Card,
        list: &ID3D12GraphicsCommandList,
        kind: Kind,
        key: u64,
        target: impl FnOnce() -> String,
        add: (u64, u64, u64),
    ) {
        if self.full || self.end.is_some() {
            return;
        }
        if self.current != Some((kind, key)) {
            let Some(query) = self.next(card) else {
                return;
            };
            // SAFETY: an open list and our query heap.
            unsafe { list.EndQuery(&card.inline.heap, D3D12_QUERY_TYPE_TIMESTAMP, query) };
            let pass = if kind.pass() {
                self.passes.push(Pass { name: self.pipeline.clone(), target: target(), ..Default::default() });
                (self.passes.len() - 1) as u32
            } else {
                NO_PASS
            };
            self.segs.push(Seg { query, kind, pass });
            self.current = Some((kind, key));
        }
        if let Some(seg) = self.segs.last()
            && seg.pass != NO_PASS
        {
            let p = &mut self.passes[seg.pass as usize];
            p.draws += add.0;
            p.dispatches += add.1;
            p.work += add.2;
        }
    }

    /// Before the list closes: the last stamp, and each block resolved into
    /// the read-back buffer.
    pub(crate) fn close(&mut self, card: &Card, list: &ID3D12GraphicsCommandList) {
        if self.segs.is_empty() || self.end.is_some() {
            return;
        }
        self.end = self.next(card);
        if let Some(end) = self.end {
            // SAFETY: an open list and our query heap.
            unsafe { list.EndQuery(&card.inline.heap, D3D12_QUERY_TYPE_TIMESTAMP, end) };
        }
        let last = self.blocks.len().saturating_sub(1);
        for (i, &block) in self.blocks.iter().enumerate() {
            let count = if i == last { self.used } else { BLOCK };
            // SAFETY: an open list; the queries were written by this list
            // before this point; the buffer has room for every block.
            unsafe {
                list.ResolveQueryData(
                    &card.inline.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    block * BLOCK,
                    count,
                    &card.inline.back,
                    8 * u64::from(block * BLOCK),
                )
            };
        }
    }

    /// The stamps of a closed list, for its submission ([`Card::sent`]);
    /// the list starts again with none.
    pub(crate) fn take(&mut self) -> Option<ListStamps> {
        if self.segs.is_empty() {
            return None;
        }
        let pipeline = std::mem::take(&mut self.pipeline);
        let taken = std::mem::take(self);
        self.pipeline = pipeline;
        Some(taken)
    }

    /// A list that is reset or dropped without having been sent: its blocks
    /// are free again (the card never ran its stamps).
    pub(crate) fn forget(&mut self, card: Option<&Card>) {
        if let Some(card) = card {
            card.give_blocks(&self.blocks);
        }
        *self = ListStamps::default();
    }

    /// The segments, as the card ran them.
    fn read(&self, card: &Card) -> Option<Record> {
        let times: Vec<u64> = self.segs.iter().map(|s| card.inline_at(s.query)).collect();
        let end = self.end.map(|q| card.inline_at(q));
        let first = *times.first()?;
        let mut record = Record { first, last: end.unwrap_or(*times.last()?), segs: Vec::with_capacity(self.segs.len()) };
        let mut before = None;
        for (i, seg) in self.segs.iter().enumerate() {
            let Some(next) = times.get(i + 1).copied().or(end) else {
                break;
            };
            let pass = (seg.pass != NO_PASS).then(|| self.passes[seg.pass as usize].clone());
            record.segs.push((seg.kind, next.saturating_sub(times[i]), pass, before));
            before = Some(seg.kind);
        }
        Some(record)
    }
}

impl ListStamps {
    /// Whether the list wrote a stamp yet.
    pub(crate) fn has_any(&self) -> bool {
        !self.segs.is_empty()
    }
}

/// Stamps of lists of ours recorded for the submission being prepared
/// (under the submission lock), taken by it ([`take_queued`]).
fn queued() -> &'static Mutex<Vec<ListStamps>> {
    static QUEUED: OnceLock<Mutex<Vec<ListStamps>>> = OnceLock::new();
    QUEUED.get_or_init(Default::default)
}

/// A list of ours with `stamps`, closed, goes with the next submission.
pub(crate) fn queue(stamps: Option<ListStamps>) {
    if let Some(mut stamps) = stamps
        && let Some(taken) = stamps.take()
    {
        queued().lock().unwrap_or_else(|e| e.into_inner()).push(taken);
    }
}

/// The stamps [`queue`]d since the last submission.
pub(crate) fn take_queued() -> Vec<ListStamps> {
    std::mem::take(&mut *queued().lock().unwrap_or_else(|e| e.into_inner()))
}

/// [`ours_begin`]'s stamps closed into `list` (before it closes).
pub(crate) fn ours_end(stamps: &mut Option<ListStamps>, list: &ID3D12GraphicsCommandList) {
    if let (Some(stamps), Some(card)) = (stamps.as_mut(), made_card()) {
        stamps.close(card, list);
    }
}

/// [`ListStamps::mark`] for work of ours in a list of ours.
pub(crate) fn ours_mark(stamps: &mut Option<ListStamps>, list: &ID3D12GraphicsCommandList, kind: Kind) {
    if let (Some(stamps), Some(card)) = (stamps.as_mut(), made_card()) {
        stamps.mark(card, list, kind, 0, String::new, (0, 0, 0));
    }
}

/// A list of ours recorded in one go: a stamp at its start as `kind`, more
/// with [`ListStamps::mark`], and [`ListStamps::close`] before it closes.
pub(crate) fn ours_begin(list: &ID3D12GraphicsCommandList, kind: Kind) -> Option<ListStamps> {
    let card = made_card()?;
    let mut stamps = ListStamps::default();
    stamps.mark(card, list, kind, 0, String::new, (0, 0, 0));
    Some(stamps)
}

/// The window's card time, from what was read.
#[derive(Default, Debug, Clone)]
pub(crate) struct CardWindow {
    pub(crate) submissions: u64,
    pub(crate) busy: u64,
    pub(crate) span: u64,
    pub(crate) longest: u64,
    pub(crate) kinds: [u64; KINDS],
    pub(crate) passes: HashMap<String, PassSum>,
    pub(crate) subs: HashMap<String, SubSum>,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct PassSum {
    pub(crate) ticks: u64,
    pub(crate) n: u64,
    pub(crate) draws: u64,
    pub(crate) dispatches: u64,
    pub(crate) work: u64,
    pub(crate) before: [u64; KINDS],
    /// List starts (nothing before it in the submission).
    pub(crate) first: u64,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct SubSum {
    pub(crate) ticks: u64,
    pub(crate) n: u64,
    pub(crate) longest: u64,
    pub(crate) kinds: [u64; KINDS],
    pub(crate) passes: HashMap<String, u64>,
}

impl CardWindow {
    fn merge(&mut self, w: &CardWindow) {
        self.submissions += w.submissions;
        self.busy += w.busy;
        self.span += w.span;
        self.longest = self.longest.max(w.longest);
        for (a, b) in self.kinds.iter_mut().zip(w.kinds) {
            *a += b;
        }
        for (k, p) in &w.passes {
            let e = self.passes.entry(k.clone()).or_default();
            e.ticks += p.ticks;
            e.n += p.n;
            e.draws += p.draws;
            e.dispatches += p.dispatches;
            e.work += p.work;
            e.first += p.first;
            for (a, b) in e.before.iter_mut().zip(p.before) {
                *a += b;
            }
        }
        for (k, s) in &w.subs {
            let e = self.subs.entry(k.clone()).or_default();
            e.ticks += s.ticks;
            e.n += s.n;
            e.longest = e.longest.max(s.longest);
            for (a, b) in e.kinds.iter_mut().zip(s.kinds) {
                *a += b;
            }
            for (name, t) in &s.passes {
                *e.passes.entry(name.clone()).or_default() += t;
            }
        }
    }
}

fn pass_key(p: &Pass) -> String {
    if p.target.is_empty() { p.name.clone() } else { format!("{} into {}", p.name, p.target) }
}

pub(crate) fn card_window(done: &mut [Done]) -> CardWindow {
    let mut w = CardWindow::default();
    let mut intervals: Vec<(u64, u64)> = Vec::new();
    for d in done.iter() {
        let sub_name = d.name.clone().unwrap_or_else(|| "(no pipeline)".into());
        let mut inside = [0u64; KINDS];
        let mut passes: HashMap<String, u64> = HashMap::new();
        let mut title = false;
        for r in &d.records {
            intervals.push((r.first, r.last.max(r.first)));
            let within = d.interval.is_some_and(|(b, e)| r.first >= b && r.last <= e);
            let mut stamped = 0;
            for (kind, ticks, pass, before) in &r.segs {
                w.kinds[*kind as usize] += ticks;
                stamped += ticks;
                title |= kind.title();
                if within {
                    inside[*kind as usize] += ticks;
                }
                if let Some(p) = pass {
                    let key = pass_key(p);
                    let e = w.passes.entry(key.clone()).or_default();
                    e.ticks += ticks;
                    e.n += 1;
                    e.draws += p.draws;
                    e.dispatches += p.dispatches;
                    e.work += p.work;
                    match before {
                        Some(b) => e.before[*b as usize] += 1,
                        None => e.first += 1,
                    }
                    if within {
                        *passes.entry(key).or_default() += ticks;
                    }
                }
            }
            if !within {
                // A list of ours sent on its own: what its stamps do not cover.
                w.kinds[Kind::Other as usize] += (r.last.saturating_sub(r.first)).saturating_sub(stamped);
            }
        }
        if let Some((b, e)) = d.interval {
            intervals.push((b, e));
            w.submissions += 1;
            w.longest = w.longest.max(e - b);
            let stamped: u64 = inside.iter().sum();
            let rest = (e - b).saturating_sub(stamped);
            let rest_kind = if title { Kind::Other } else { Kind::OurPasses };
            w.kinds[rest_kind as usize] += rest;
            inside[rest_kind as usize] += rest;
            let s = w.subs.entry(sub_name).or_default();
            s.ticks += e - b;
            s.n += 1;
            s.longest = s.longest.max(e - b);
            for (a, b) in s.kinds.iter_mut().zip(inside) {
                *a += b;
            }
            for (name, t) in passes {
                *s.passes.entry(name).or_default() += t;
            }
        }
    }
    if intervals.is_empty() {
        return w;
    }
    intervals.sort_unstable();
    let (mut from, mut to) = intervals[0];
    let first = from;
    let mut last = to;
    for &(b, e) in &intervals[1..] {
        last = last.max(e);
        if b > to {
            w.busy += to - from;
            (from, to) = (b, e);
        } else {
            to = to.max(e);
        }
    }
    w.busy += to - from;
    w.span = last - first;
    w
}

// ---- the submitting thread ----

const EXECUTE: usize = 0;
const PRESENT: usize = 1;
const CALL_NAMES: [&str; 2] = ["submissions", "presents"];

#[derive(Default, Clone, Copy, Debug)]
struct Calls {
    n: u64,
    wall_ns: u64,
    cycles: u64,
    kernel: u64,
    user: u64,
}

#[derive(Default, Clone, Debug)]
struct Submitter {
    between_ns: u64,
    calls: [Calls; 2],
}

#[derive(Default)]
struct Threads {
    /// Per thread: the window's, the run's, the last return.
    window: HashMap<u32, Submitter>,
    total: HashMap<u32, Submitter>,
    last: HashMap<u32, Instant>,
}

fn threads() -> &'static Mutex<Threads> {
    static THREADS: OnceLock<Mutex<Threads>> = OnceLock::new();
    THREADS.get_or_init(Default::default)
}

/// One title call being timed (dropped at its return).
pub(crate) struct Inside {
    what: usize,
    began: Instant,
    cycles: u64,
    kernel: u64,
    user: u64,
}

fn own_times() -> (u64, u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    let mut cycles = 0;
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut c, mut e, mut k, mut u) = (zero, zero, zero, zero);
    // SAFETY: the calling thread's own counters, written into locals.
    unsafe {
        QueryThreadCycleTime(GetCurrentThread(), &mut cycles);
        GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u);
    }
    let t = |f: FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    (cycles, t(k), t(u))
}

fn me() -> u32 {
    // SAFETY: plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

fn enter(what: usize) -> Option<Inside> {
    if !on() {
        return None;
    }
    let began = Instant::now();
    {
        let mut t = threads().lock().unwrap_or_else(|e| e.into_inner());
        let id = me();
        if let Some(last) = t.last.get(&id).copied() {
            let gap = began.saturating_duration_since(last).as_nanos() as u64;
            t.window.entry(id).or_default().between_ns += gap;
        }
    }
    let (cycles, kernel, user) = own_times();
    Some(Inside { what, began, cycles, kernel, user })
}

/// Times a title call of `ExecuteCommandLists` on the calling thread.
pub(crate) fn submission() -> Option<Inside> {
    enter(EXECUTE)
}

/// Times a title call of `PresentX`.
pub(crate) fn present() -> Option<Inside> {
    enter(PRESENT)
}

impl Drop for Inside {
    fn drop(&mut self) {
        let (cycles, kernel, user) = own_times();
        let now = Instant::now();
        let mut t = threads().lock().unwrap_or_else(|e| e.into_inner());
        let id = me();
        let c = &mut t.window.entry(id).or_default().calls[self.what];
        c.n += 1;
        c.wall_ns += now.saturating_duration_since(self.began).as_nanos() as u64;
        c.cycles += cycles.saturating_sub(self.cycles);
        c.kernel += kernel.saturating_sub(self.kernel);
        c.user += user.saturating_sub(self.user);
        t.last.insert(id, now);
    }
}

// ---- the recording threads ----

struct Recorder {
    id: u32,
    ticks: AtomicU64,
    calls: AtomicU64,
}

fn recorders() -> &'static Mutex<Vec<Arc<Recorder>>> {
    static ALL: OnceLock<Mutex<Vec<Arc<Recorder>>>> = OnceLock::new();
    ALL.get_or_init(Default::default)
}

thread_local! {
    static MINE: Arc<Recorder> = {
        let r = Arc::new(Recorder { id: me(), ticks: AtomicU64::new(0), calls: AtomicU64::new(0) });
        recorders().lock().unwrap_or_else(|e| e.into_inner()).push(r.clone());
        r
    };
}

/// A list call begins: its time-stamp counter, or 0 without the meters.
#[inline]
pub(crate) fn record_begin() -> u64 {
    if on() { tsc() } else { 0 }
}

/// The list call begun at `began` returns.
#[inline]
pub(crate) fn record_end(began: u64) {
    if began != 0 {
        let took = tsc().saturating_sub(began);
        MINE.with(|r| {
            r.ticks.fetch_add(took, Relaxed);
            r.calls.fetch_add(1, Relaxed);
        });
    }
}

// ---- the lines ----

#[derive(Default)]
struct Report {
    started: Option<(Instant, u64)>,
    last: Option<(Instant, u64)>,
    /// Lines given so far (the tables come every sixth).
    lines: u64,
    /// Per recorder: ticks and calls at the last line.
    recorded: HashMap<u32, (u64, u64)>,
    /// Since the last tables.
    tables: CardWindow,
    tables_frames: f64,
    total_card: CardWindow,
    total_frames: f64,
    total_secs: f64,
    total_skipped: u64,
    total_record_ticks: u64,
    total_recorders: HashMap<u32, u64>,
}

fn report() -> &'static Mutex<Report> {
    static REPORT: OnceLock<Mutex<Report>> = OnceLock::new();
    REPORT.get_or_init(Default::default)
}

fn card_lines(card: &CardWindow, frequency: u64, frames: f64, what: &str) -> Vec<String> {
    let ms = |ticks: u64| ticks as f64 * 1000.0 / frequency as f64;
    let per = frames.max(1e-9);
    let mut out = vec![format!(
        "METERS card ({what}): {} host submissions ({:.1} a frame); busy {:.1} ms a frame, idle {:.1} ms a frame in the span they cover ({:.0}% busy); the longest submission {:.2} ms",
        card.submissions,
        card.submissions as f64 / per,
        ms(card.busy) / per,
        ms(card.span.saturating_sub(card.busy)) / per,
        card.busy as f64 * 100.0 / card.span.max(1) as f64,
        ms(card.longest),
    )];
    let sum: u64 = card.kinds.iter().sum();
    let ours: u64 = (0..KINDS).filter(|&k| KIND_ORDER[k].ours()).map(|k| card.kinds[KIND_ORDER[k] as usize]).sum();
    let parts: Vec<String> = KIND_ORDER
        .iter()
        .filter(|k| card.kinds[**k as usize] > 0)
        .map(|k| format!("{} {:.2}", k.name(), ms(card.kinds[*k as usize]) / per))
        .collect();
    out.push(format!(
        "METERS card split ({what}), ms a frame: {}; = {:.1} in all against busy {:.1} (more: work that overlaps; less: idle between our lists sent apart and the submission after them); ours {:.1} ({:.0}% of busy)",
        parts.join(", "),
        ms(sum) / per,
        ms(card.busy) / per,
        ms(ours) / per,
        ours as f64 * 100.0 / card.busy.max(1) as f64,
    ));
    out
}

const KIND_ORDER: [Kind; KINDS] = [
    Kind::TitleDraws,
    Kind::TitleDispatches,
    Kind::TitleCopies,
    Kind::ClearsResolves,
    Kind::TitleBarriers,
    Kind::Uploads,
    Kind::BufferCopies,
    Kind::CopyBacks,
    Kind::Hashes,
    Kind::Clamps,
    Kind::OursInList,
    Kind::OurPasses,
    Kind::Other,
];

/// The top ten title passes and the top ten submissions.
fn table_lines(card: &CardWindow, frequency: u64, frames: f64, what: &str) -> Vec<String> {
    let ms = |ticks: u64| ticks as f64 * 1000.0 / frequency as f64;
    let per = frames.max(1e-9);
    let mut out = Vec::new();
    let mut passes: Vec<(&String, &PassSum)> = card.passes.iter().collect();
    passes.sort_by_key(|(_, p)| std::cmp::Reverse(p.ticks));
    for (i, (name, p)) in passes.iter().take(10).enumerate() {
        let n = p.n.max(1) as f64;
        let count = if p.draws > 0 && p.dispatches == 0 {
            format!("{:.1} draws ({:.0} vertices x instances)", p.draws as f64 / n, p.work as f64 / n)
        } else if p.dispatches > 0 && p.draws == 0 {
            format!("{:.1} dispatches ({:.0} thread groups)", p.dispatches as f64 / n, p.work as f64 / n)
        } else {
            format!("{:.1} draws and {:.1} dispatches", p.draws as f64 / n, p.dispatches as f64 / n)
        };
        let mut before: Vec<(usize, u64)> = p.before.iter().copied().enumerate().filter(|b| b.1 > 0).collect();
        before.sort_by_key(|b| std::cmp::Reverse(b.1));
        let mut said: Vec<String> = before
            .iter()
            .take(2)
            .map(|&(k, c)| format!("{} {:.0}%", KIND_NAMES[k], c as f64 * 100.0 / n))
            .collect();
        if p.first > 0 {
            said.push(format!("nothing (first in its submission) {:.0}%", p.first as f64 * 100.0 / n));
        }
        out.push(format!(
            "METERS pass {}/10 ({what}): {:.2} ms a frame, {:.1} times a frame ({:.3} ms each); {count} each; {name}; right before it: {}",
            i + 1,
            ms(p.ticks) / per,
            p.n as f64 / per,
            ms(p.ticks) / n,
            said.join(", ")
        ));
    }
    let mut subs: Vec<(&String, &SubSum)> = card.subs.iter().collect();
    subs.sort_by_key(|(_, s)| std::cmp::Reverse(s.ticks));
    for (i, (name, s)) in subs.iter().take(10).enumerate() {
        let mut kinds: Vec<usize> = (0..KINDS).filter(|&k| s.kinds[k] > 0).collect();
        kinds.sort_by_key(|&k| std::cmp::Reverse(s.kinds[k]));
        let holds: Vec<String> = kinds.iter().take(4).map(|&k| format!("{} {:.2}", KIND_NAMES[k], ms(s.kinds[k]) / per)).collect();
        let mut passes: Vec<(&String, &u64)> = s.passes.iter().collect();
        passes.sort_by_key(|(_, t)| std::cmp::Reverse(**t));
        let top: Vec<String> = passes.iter().take(3).map(|(p, t)| format!("{:.2} {p}", ms(**t) / per)).collect();
        out.push(format!(
            "METERS submission {}/10 ({what}) first binding {name}: {:.2} ms a frame, {:.1} a frame (the longest {:.2} ms); holds, ms a frame: {}; its top passes: {}",
            i + 1,
            ms(s.ticks) / per,
            s.n as f64 / per,
            ms(s.longest),
            holds.join(", "),
            if top.is_empty() { "-".into() } else { top.join("; ") }
        ));
    }
    out
}

fn submitter_line(window: &HashMap<u32, Submitter>, hz: f64, frames: f64, what: &str) -> Option<String> {
    let per = frames.max(1e-9);
    let inside = |s: &Submitter| s.calls.iter().map(|c| c.wall_ns).sum::<u64>();
    let (&id, s) = window.iter().max_by_key(|(_, s)| inside(s))?;
    let ms = |ns: u64| ns as f64 / 1e6 / per;
    let mut parts = Vec::new();
    for (k, c) in s.calls.iter().enumerate() {
        if c.n == 0 {
            continue;
        }
        let wall = c.wall_ns as f64;
        let on_cpu = (c.cycles as f64 / hz * 1e9).min(wall);
        let kernel_share = c.kernel as f64 / (c.kernel + c.user).max(1) as f64;
        let kernel = on_cpu * kernel_share;
        parts.push(format!(
            "inside {} {:.1} ms ({:.1} calls) = our code {:.1} + kernel {:.1} + waiting {:.1}",
            CALL_NAMES[k],
            wall / 1e6 / per,
            c.n as f64 / per,
            (on_cpu - kernel) / 1e6 / per,
            kernel / 1e6 / per,
            (wall - on_cpu) / 1e6 / per,
        ));
    }
    let others = window.len().saturating_sub(1);
    Some(format!(
        "METERS submitting thread {id} ({what}), ms a frame: {:.1} in all = between calls (the title's own work and our other calls) {:.1} + {}{}",
        ms(s.between_ns + inside(s)),
        ms(s.between_ns),
        parts.join(" + "),
        if others > 0 { format!("; {others} other threads submitted or presented") } else { String::new() }
    ))
}

fn recorder_line(deltas: &[(u32, u64, u64)], hz: f64, frames: f64, secs: f64, what: &str) -> Option<String> {
    let total: u64 = deltas.iter().map(|d| d.1).sum();
    if total == 0 {
        return None;
    }
    let per = frames.max(1e-9);
    let ms = |ticks: u64| ticks as f64 / hz * 1000.0 / per;
    let mut d: Vec<&(u32, u64, u64)> = deltas.iter().filter(|d| d.1 > 0).collect();
    d.sort_by_key(|d| std::cmp::Reverse(d.1));
    let each: Vec<String> = d
        .iter()
        .take(8)
        .map(|(id, t, n)| format!("{id} {:.1} ms ({:.0} calls)", ms(*t), *n as f64 / per))
        .collect();
    Some(format!(
        "METERS recording ({what}), ms a frame inside list calls: {:.1} over {} threads, {:.2} recording at once on average; {}",
        ms(total),
        d.len(),
        total as f64 / hz / secs.max(1e-9),
        each.join(", ")
    ))
}

/// The lines for the 5 s frame line (`frames` frames since the last), or
/// `None` without the meters.
pub(crate) fn lines(frames: f64, card: Option<&Card>) -> Option<Vec<String>> {
    if !on() {
        return None;
    }
    let now = (Instant::now(), tsc());
    let mut r = report().lock().unwrap_or_else(|e| e.into_inner());
    let Some(last) = r.last.replace(now) else {
        r.started = Some(now);
        // The first line only starts the clock: what came before is let go.
        let _ = card.map(Card::take);
        threads().lock().unwrap_or_else(|e| e.into_inner()).window.clear();
        for rec in recorders().lock().unwrap_or_else(|e| e.into_inner()).iter() {
            r.recorded.insert(rec.id, (rec.ticks.load(Relaxed), rec.calls.load(Relaxed)));
        }
        return Some(vec!["METERS started: the first numbers come with the next frame line".into()]);
    };
    r.lines += 1;
    let secs = now.0.saturating_duration_since(last.0).as_secs_f64();
    let hz = (now.1.saturating_sub(last.1)) as f64 / secs.max(1e-9);
    let mut out = Vec::new();
    if let Some(card) = card {
        let (mut done, skipped, short) = card.take();
        let w = card_window(&mut done);
        out.extend(card_lines(&w, card.frequency(), frames, "per frame"));
        if skipped > 0 || short > 0 {
            out.push(format!(
                "METERS card: {skipped} submissions not metered (all {PAIRS} stamp pairs in the air), {short} lists with their stamps cut short (no free block)"
            ));
        }
        r.tables.merge(&w);
        r.tables_frames += frames;
        if r.lines.is_multiple_of(6) {
            let tables = std::mem::take(&mut r.tables);
            let tf = std::mem::take(&mut r.tables_frames);
            out.extend(table_lines(&tables, card.frequency(), tf, "the last 30 s"));
        }
        r.total_card.merge(&w);
        r.total_skipped += skipped;
    }
    {
        let mut t = threads().lock().unwrap_or_else(|e| e.into_inner());
        let window = std::mem::take(&mut t.window);
        out.extend(submitter_line(&window, hz, frames, "per frame"));
        for (id, s) in window {
            let total = t.total.entry(id).or_default();
            total.between_ns += s.between_ns;
            for (a, b) in total.calls.iter_mut().zip(s.calls) {
                a.n += b.n;
                a.wall_ns += b.wall_ns;
                a.cycles += b.cycles;
                a.kernel += b.kernel;
                a.user += b.user;
            }
        }
    }
    let mut deltas = Vec::new();
    for rec in recorders().lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let (ticks, calls) = (rec.ticks.load(Relaxed), rec.calls.load(Relaxed));
        let (t0, c0) = r.recorded.insert(rec.id, (ticks, calls)).unwrap_or((0, 0));
        deltas.push((rec.id, ticks.saturating_sub(t0), calls.saturating_sub(c0)));
    }
    out.extend(recorder_line(&deltas, hz, frames, secs, "per frame"));
    for &(id, t, _) in &deltas {
        *r.total_recorders.entry(id).or_default() += t;
        r.total_record_ticks += t;
    }
    r.total_frames += frames;
    r.total_secs += secs;
    Some(out)
}

/// The run's summary (from the first frame line to the last), once.
pub(crate) fn final_report(card: Option<&Card>) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !on() || DONE.swap(true, Relaxed) {
        return;
    }
    let r = report().lock().unwrap_or_else(|e| e.into_inner());
    let (Some(started), Some(last)) = (r.started, r.last) else {
        return;
    };
    let secs = last.0.saturating_duration_since(started.0).as_secs_f64();
    let hz = last.1.saturating_sub(started.1) as f64 / secs.max(1e-9);
    let frames = r.total_frames;
    let what = format!("the run: {:.0} s, {frames:.0} frames", r.total_secs);
    let mut out = Vec::new();
    if let Some(card) = card {
        out.extend(card_lines(&r.total_card, card.frequency(), frames, &what));
        out.extend(table_lines(&r.total_card, card.frequency(), frames, "the run"));
    }
    let total = threads().lock().unwrap_or_else(|e| e.into_inner()).total.clone();
    out.extend(submitter_line(&total, hz, frames, &what));
    let deltas: Vec<(u32, u64, u64)> = r.total_recorders.iter().map(|(&id, &t)| (id, t, 0)).collect();
    out.extend(recorder_line(&deltas, hz, frames, r.total_secs, &what));
    if r.total_skipped > 0 {
        out.push(format!("METERS card: {} submissions not metered over the run", r.total_skipped));
    }
    drop(r);
    for line in out {
        tracing::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(first: u64, segs: &[(Kind, u64)]) -> Record {
        let mut at = first;
        let mut out = Record { first, last: first, segs: Vec::new() };
        let mut before = None;
        for &(k, t) in segs {
            let pass = k.pass().then(|| Pass { name: format!("{k:?}"), draws: 1, ..Default::default() });
            out.segs.push((k, t, pass, before));
            before = Some(k);
            at += t;
        }
        out.last = at;
        out
    }

    #[test]
    fn the_split_adds_up_to_the_busy_time() {
        // A submission 0..100: a title list 10..90 (draws 50, our clamps 10,
        // dispatches 20); the 20 left are everything else. Our sends and
        // hashes sent apart, 200..230. A submission of ours only, 300..340.
        let mut done = vec![
            Done {
                interval: Some((0, 100)),
                name: Some("a".into()),
                records: vec![rec(10, &[(Kind::TitleDraws, 50), (Kind::Clamps, 10), (Kind::TitleDispatches, 20)])],
            },
            Done { interval: None, name: None, records: vec![rec(200, &[(Kind::BufferCopies, 20), (Kind::Hashes, 10)])] },
            Done { interval: Some((300, 340)), name: Some("b".into()), records: vec![] },
        ];
        let w = card_window(&mut done);
        assert_eq!(w.busy, 100 + 30 + 40);
        assert_eq!(w.kinds.iter().sum::<u64>(), w.busy);
        assert_eq!(w.kinds[Kind::Other as usize], 20);
        assert_eq!(w.kinds[Kind::OurPasses as usize], 40);
        assert_eq!(w.kinds[Kind::Clamps as usize], 10);
        assert_eq!(w.subs["a"].kinds[Kind::TitleDraws as usize], 50);
        assert_eq!(w.subs["a"].passes.len(), 2);
        let draws = &w.passes["TitleDraws"];
        assert_eq!((draws.ticks, draws.n, draws.first), (50, 1, 1));
        assert_eq!(w.passes["TitleDispatches"].before[Kind::Clamps as usize], 1);
        let lines = card_lines(&w, 1000, 1.0, "test");
        assert!(lines[1].contains("= 170.0 in all against busy 170.0"), "{lines:?}");
        assert!(table_lines(&w, 1000, 1.0, "test").len() == 4);
    }

    #[test]
    fn the_calling_threads_time_adds_up() {
        let mut window = HashMap::new();
        window.insert(
            7,
            Submitter {
                between_ns: 30_000_000,
                calls: [
                    Calls { n: 10, wall_ns: 60_000_000, cycles: 40_000_000, kernel: 1, user: 3 },
                    Calls::default(),
                ],
            },
        );
        // 1 GHz: 40 ms on a processor (10 kernel, 30 ours), 20 waiting.
        let line = submitter_line(&window, 1e9, 1.0, "test").unwrap();
        assert!(line.contains("90.0 in all = between calls (the title's own work and our other calls) 30.0 + inside submissions 60.0 ms (10.0 calls) = our code 30.0 + kernel 10.0 + waiting 20.0"), "{line}");
    }

    /// The card's stamps on this PC's card: a heavy dispatch measures longer
    /// than a trivial one, between the submission pairs and inside the
    /// lists; the stamps come in order, and the split adds up.
    #[test]
    fn card_stamps_time_a_heavy_and_a_trivial_dispatch() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::s;
        let Some(host) = crate::host::host() else {
            eprintln!("no host GPU");
            return;
        };
        let device = &host.device;
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer C : register(b0) { uint steps; };
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    for (uint i = 0; i < steps; i++) { x = x * 1664525u + 1013904223u; }
    o.Store((id.x & 1023) * 4, x);
}
"#;
        // SAFETY: plain D3D12 object creation and recording with valid
        // descriptions; every object outlives the work waited for below.
        unsafe {
            let mut code: Option<ID3DBlob> = None;
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("meters"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)
                .unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pipeline: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 4096,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                .unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let meter = Card::new(device, &queue).unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let mut value = 0;
            // Each a submission of its own between its pair: in one list,
            // a dispatch counted as the title's and one as a clamp of ours,
            // heavy and trivial (then the other way round).
            for (first, second) in [((4096u32, 20_000u32), (1u32, 1u32)), ((1, 1), (4096, 20_000))] {
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pipeline).unwrap();
                let mut stamps = ListStamps { pipeline: "test pass".into(), ..Default::default() };
                list.SetComputeRootSignature(&root);
                list.SetComputeRootUnorderedAccessView(1, buffer.GetGPUVirtualAddress());
                for (kind, (groups, steps)) in [(Kind::TitleDispatches, first), (Kind::Clamps, second)] {
                    stamps.mark(&meter, &list, kind, 0, String::new, (0, 1, u64::from(groups)));
                    list.SetComputeRoot32BitConstant(0, steps, 0);
                    list.Dispatch(groups, 1, 1);
                }
                stamps.close(&meter, &list);
                list.Close().unwrap();
                let stamp = meter.stamp().unwrap();
                queue.ExecuteCommandLists(&[Some(stamp.begin.clone()), Some(list.cast().unwrap()), Some(stamp.end.clone())]);
                value += 1;
                queue.Signal(&fence, value).unwrap();
                meter.sent(Some(stamp), &fence, value, Some("test".into()), stamps.take().into_iter().collect());
                fence.SetEventOnCompletion(value, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            }
            let (mut done, skipped, short) = meter.take();
            assert_eq!((done.len(), skipped, short), (2, 0, 0), "{done:?}");
            let ms = |t: u64| t as f64 * 1000.0 / meter.frequency() as f64;
            let (a, b) = (done[0].interval.unwrap(), done[1].interval.unwrap());
            assert!(a.1 <= b.0, "in order: {done:?}");
            let segs: Vec<Vec<f64>> = done.iter().map(|d| d.records[0].segs.iter().map(|s| ms(s.1)).collect()).collect();
            eprintln!("submissions {:.3} and {:.3} ms; segments {segs:?}", ms(a.1 - a.0), ms(b.1 - b.0));
            assert!(segs[0][0] > 10.0 * segs[0][1] && segs[1][1] > 10.0 * segs[1][0], "{segs:?}");
            let w = card_window(&mut done);
            assert_eq!(w.kinds.iter().sum::<u64>(), w.busy);
            assert!(w.kinds[Kind::Clamps as usize] > 0 && w.kinds[Kind::TitleDispatches as usize] > 0);
            assert_eq!(w.passes.len(), 1);
        }
    }
}
