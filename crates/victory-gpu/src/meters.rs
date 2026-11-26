//! `VICTORY_METERS=1` (off by default): exact meters of the card's time, the
//! submitting thread's time and the recording threads' time, one `METERS`
//! line group on the 5 s frame line and a summary of the run at the title's
//! exit ([`final_report`]).
//!
//! - The card: every host submission (`Host::submit`) runs between two
//!   timestamp queries on the direct queue: a list of ours before its lists
//!   and one after them (after the compare list too), which also resolves
//!   both into a read-back buffer. The lists are recorded once, a pair per
//!   slot of a ring of [`PAIRS`]; a slot is used again only once the fence
//!   value signalled after its submission is reached (a full ring leaves a
//!   submission unmetered, counted). The stamps of a window give the card's
//!   busy time (the union of the submissions' intervals), its idle time in
//!   the span they cover, the longest submission, and the submissions'
//!   card time by the first pipeline their first list binds. Other work on
//!   the queue (the relay's fetches, passes of ours run on their own) is
//!   not inside any interval: it shows as idle.
//! - The submitting thread: each title call of a queue's
//!   `ExecuteCommandLists` and of `PresentX` is timed from entry to return
//!   on the calling thread: wall time, the thread's cycles (on a processor)
//!   and its kernel and user times (`GetThreadTimes`, whose ratio splits the
//!   cycles: its own resolution is the clock tick). Time between two such
//!   calls is the title's own work (and our other calls). Per thread, so
//!   between + inside adds up to the thread's time.
//! - The recording threads: each list call that goes through
//!   `list::with_state` (the recording calls) is timed with the time-stamp
//!   counter, summed per thread; the threads recording at once is their
//!   summed time over the window's.
//!
//! Without the switch each hook is one load of a flag set once.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// Pairs of stamps in the ring (submissions in the air at most, metered).
pub(crate) const PAIRS: usize = 256;

/// Whether the meters run (`VICTORY_METERS=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_METERS").is_some_and(|v| v == "1");
        if on {
            tracing::info!("METERS on (VICTORY_METERS=1): card time per host submission, the submitting thread's time, the recording threads' time");
        }
        on
    })
}

fn tsc() -> u64 {
    // SAFETY: reads the time-stamp counter.
    unsafe { core::arch::x86_64::_rdtsc() }
}

// ---- the card ----

/// The ring of timestamp pairs and what was read from it.
pub(crate) struct Card {
    inner: Mutex<CardInner>,
    /// Ticks a second of the queue's timestamps.
    frequency: u64,
}

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
    done: Vec<Sample>,
    /// Submissions not metered: the ring was full.
    skipped: u64,
}

// SAFETY: `mapped` points into `back`, which lives as long as it; the rest
// are free-threaded D3D12 objects, used under the mutex.
unsafe impl Send for CardInner {}

struct Pending {
    slot: usize,
    fence: ID3D12Fence,
    value: u64,
    name: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct Sample {
    pub(crate) begin: u64,
    pub(crate) end: u64,
    pub(crate) name: Option<String>,
}

/// A slot taken for one submission: run `begin` before its lists and `end`
/// after them, then tell [`Card::sent`] the fence value signalled after.
pub(crate) struct Stamp {
    slot: usize,
    pub(crate) begin: ID3D12CommandList,
    pub(crate) end: ID3D12CommandList,
}

/// The card's meter, made at its first use on `device` for `queue`.
static CARD: OnceLock<Option<Card>> = OnceLock::new();

/// The card's meter if it was made.
pub(crate) fn made_card() -> Option<&'static Card> {
    CARD.get().and_then(Option::as_ref)
}

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

impl Card {
    pub(crate) fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> windows::core::Result<Card> {
        // SAFETY: plain D3D12 object creation with valid descriptions; the
        // lists are recorded once with objects that live as long as they.
        unsafe {
            let frequency = queue.GetTimestampFrequency()?;
            let mut heap: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: (2 * PAIRS) as u32,
                    NodeMask: 0,
                },
                &mut heap,
            )?;
            let heap = heap.ok_or_else(windows::core::Error::empty)?;
            let mut back: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: (16 * PAIRS) as u64,
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
            Ok(Card {
                inner: Mutex::new(CardInner {
                    begins,
                    ends,
                    _allocator: allocator,
                    _heap: heap,
                    back,
                    mapped: mapped.cast(),
                    next: 0,
                    used: vec![false; PAIRS],
                    pending: VecDeque::new(),
                    done: Vec::new(),
                    skipped: 0,
                }),
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
        inner.harvest();
        let slot = inner.next;
        if inner.used[slot] {
            inner.skipped += 1;
            return None;
        }
        inner.used[slot] = true;
        inner.next = (slot + 1) % PAIRS;
        Some(Stamp { slot, begin: inner.begins[slot].clone(), end: inner.ends[slot].clone() })
    }

    /// The submission with `stamp` was sent; `fence` reaches `value` when it
    /// (its end list) has run.
    pub(crate) fn sent(&self, stamp: Stamp, fence: &ID3D12Fence, value: u64, name: Option<String>) {
        self.lock().pending.push_back(Pending { slot: stamp.slot, fence: fence.clone(), value, name });
    }

    /// Stamps read since the last call (submissions the card has run), and
    /// those left out with a full ring.
    pub(crate) fn take(&self) -> (Vec<Sample>, u64) {
        let mut inner = self.lock();
        inner.harvest();
        (std::mem::take(&mut inner.done), std::mem::take(&mut inner.skipped))
    }

    pub(crate) fn frequency(&self) -> u64 {
        self.frequency
    }
}

impl CardInner {
    /// Reads the stamps of the submissions the card has run (in order: one
    /// queue).
    fn harvest(&mut self) {
        while let Some(p) = self.pending.front() {
            // SAFETY: a plain fence query.
            if unsafe { p.fence.GetCompletedValue() } < p.value {
                break;
            }
            let p = self.pending.pop_front().expect("a front");
            // SAFETY: the two stamps of the slot, resolved by its end list,
            // which has run (the fence after it was reached).
            let (begin, end) = unsafe {
                (self.mapped.add(2 * p.slot).read_volatile(), self.mapped.add(2 * p.slot + 1).read_volatile())
            };
            self.used[p.slot] = false;
            if end >= begin && begin > 0 {
                self.done.push(Sample { begin, end, name: p.name });
            }
        }
        let _ = &self.back;
    }
}

/// What a window of card stamps says: busy (the union of intervals), the
/// span from the first begin to the last end, the longest, and by name.
#[derive(Default, Debug, Clone)]
pub(crate) struct CardWindow {
    pub(crate) submissions: u64,
    pub(crate) busy: u64,
    pub(crate) span: u64,
    pub(crate) longest: u64,
    pub(crate) by_name: HashMap<String, (u64, u64)>,
}

pub(crate) fn card_window(samples: &mut [Sample]) -> CardWindow {
    let mut w = CardWindow { submissions: samples.len() as u64, ..Default::default() };
    if samples.is_empty() {
        return w;
    }
    samples.sort_by_key(|s| s.begin);
    let (mut from, mut to) = (samples[0].begin, samples[0].end);
    let first = from;
    let mut last = to;
    for s in samples.iter() {
        w.longest = w.longest.max(s.end - s.begin);
        last = last.max(s.end);
        let e = w.by_name.entry(s.name.clone().unwrap_or_else(|| "(no pipeline)".into())).or_default();
        e.0 += s.end - s.begin;
        e.1 += 1;
        if s.begin > to {
            w.busy += to - from;
            (from, to) = (s.begin, s.end);
        } else {
            to = to.max(s.end);
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
    /// Per recorder: ticks and calls at the last line.
    recorded: HashMap<u32, (u64, u64)>,
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

fn card_line(card: &CardWindow, frequency: u64, frames: f64, what: &str) -> Vec<String> {
    let ms = |ticks: u64| ticks as f64 * 1000.0 / frequency as f64;
    let per = frames.max(1e-9);
    let mut out = vec![format!(
        "METERS card ({what}): {} host submissions ({:.1} a frame); busy {:.1} ms a frame, idle {:.1} ms a frame in the span they cover ({:.0}% busy); the longest {:.2} ms",
        card.submissions,
        card.submissions as f64 / per,
        ms(card.busy) / per,
        ms(card.span.saturating_sub(card.busy)) / per,
        card.busy as f64 * 100.0 / card.span.max(1) as f64,
        ms(card.longest),
    )];
    let mut names: Vec<(&String, &(u64, u64))> = card.by_name.iter().collect();
    names.sort_by_key(|(_, (t, _))| std::cmp::Reverse(*t));
    let top: Vec<String> = names
        .iter()
        .take(5)
        .map(|(name, (t, n))| format!("{:.2} ms a frame in {:.1} submissions first binding {name}", ms(*t) / per, *n as f64 / per))
        .collect();
    if !top.is_empty() {
        out.push(format!("METERS card ({what}) by the first pipeline of a submission: {}", top.join("; ")));
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
    let secs = now.0.saturating_duration_since(last.0).as_secs_f64();
    let hz = (now.1.saturating_sub(last.1)) as f64 / secs.max(1e-9);
    let mut out = Vec::new();
    if let Some(card) = card {
        let (mut samples, skipped) = card.take();
        let w = card_window(&mut samples);
        out.extend(card_line(&w, card.frequency(), frames, "per frame"));
        if skipped > 0 {
            out.push(format!("METERS card: {skipped} submissions not metered (all {PAIRS} stamp pairs in the air)"));
        }
        let t = &mut r.total_card;
        t.submissions += w.submissions;
        t.busy += w.busy;
        t.span += w.span;
        t.longest = t.longest.max(w.longest);
        for (name, (ticks, n)) in w.by_name {
            let e = t.by_name.entry(name).or_default();
            e.0 += ticks;
            e.1 += n;
        }
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
        out.extend(card_line(&r.total_card, card.frequency(), frames, &what));
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

    fn s(begin: u64, end: u64, name: &str) -> Sample {
        Sample { begin, end, name: Some(name.into()) }
    }

    #[test]
    fn busy_time_is_the_union_of_intervals() {
        let mut samples = vec![s(30, 40, "b"), s(0, 10, "a"), s(5, 20, "a"), s(100, 101, "c")];
        let w = card_window(&mut samples);
        assert_eq!(w.busy, 20 + 10 + 1);
        assert_eq!(w.span, 101);
        assert_eq!(w.longest, 15);
        assert_eq!(w.by_name["a"], (25, 2));
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

    /// The card's stamps on this PC's card: a heavy dispatch measures
    /// longer than a trivial one, and the stamps come in order.
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
            // Heavy first, then trivial, each a submission of its own
            // between its stamps; then the pair again in the other order.
            for (groups, steps, name) in [(4096u32, 20_000u32, "heavy"), (1, 1, "trivial"), (1, 1, "trivial"), (4096, 20_000, "heavy")] {
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pipeline).unwrap();
                list.SetComputeRootSignature(&root);
                list.SetComputeRoot32BitConstant(0, steps, 0);
                list.SetComputeRootUnorderedAccessView(1, buffer.GetGPUVirtualAddress());
                list.Dispatch(groups, 1, 1);
                list.Close().unwrap();
                let stamp = meter.stamp().unwrap();
                queue.ExecuteCommandLists(&[Some(stamp.begin.clone()), Some(list.cast().unwrap()), Some(stamp.end.clone())]);
                value += 1;
                queue.Signal(&fence, value).unwrap();
                meter.sent(stamp, &fence, value, Some(name.into()));
                fence.SetEventOnCompletion(value, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            }
            let (samples, skipped) = meter.take();
            assert_eq!((samples.len(), skipped), (4, 0), "{samples:?}");
            let ms = |s: &Sample| (s.end - s.begin) as f64 * 1000.0 / meter.frequency() as f64;
            for pair in samples.windows(2) {
                assert!(pair[0].end <= pair[1].begin, "in order: {samples:?}");
            }
            let took: Vec<f64> = samples.iter().map(ms).collect();
            eprintln!("card ms: {took:?} (heavy, trivial, trivial, heavy)");
            assert!(took[0] > 10.0 * took[1] && took[3] > 10.0 * took[2], "{took:?}");
            let mut samples = samples;
            let w = card_window(&mut samples);
            assert_eq!(w.by_name["heavy"].1, 2);
            assert!(w.busy <= w.span);
        }
    }
}
