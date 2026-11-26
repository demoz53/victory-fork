//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

/// A frame has run 500 draws or more: the title's world has begun (menus
/// and movies draw a few each). In every build (it was kept by the
/// diagnostics' frame captures, and a build without them did not build).
#[cfg(windows)]
static BEGAN: AtomicBool = AtomicBool::new(false);

/// Whether the title's world has begun (a frame of 500 draws or more), for
/// skipping only the movies before it (`VICTORY_SKIP_BINK=intro`).
#[cfg(windows)]
pub fn world_began() -> bool {
    BEGAN.load(Ordering::Relaxed)
}

/// At a present: `drawn` draws run on the host so far.
#[cfg(windows)]
fn note_world(drawn: u64) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let before = LAST.swap(drawn, Ordering::Relaxed);
    if drawn.saturating_sub(before) >= 500 {
        BEGAN.store(true, Ordering::Relaxed);
    }
}

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence to reach the value.
    Wait(crate::fence::FenceRef, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(crate::fence::FenceRef, u64),
    /// With [`behind`]: the part that was at the head of the line is being
    /// prepared and sent by the thread that runs the line, with the line
    /// unlocked, so the title can add to its end meanwhile; whatever is
    /// added stays behind this (the line is not empty), signals too.
    Running,
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present runs what queues still hold (on by default: a wait
/// is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for
/// finding out whether the title's waits reach across its presents; a
/// wait nothing ever meets then stops the title for good.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    // With [`behind`] nothing runs on the title's thread: every list goes
    // behind the line, which our thread runs.
    let first = if !behind() && lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 && crate::host::late() {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they are settled.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    } else if first > 0 {
        let now: Vec<_> = lists[..first]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
        // They have run: their descriptor tables' slots are free again, as
        // on the path without cuts ([`execute`]). This path replaced that
        // one without the call, so the lists that run at once kept their
        // slots until the title's reset and the ring ran out as before
        // (run 45: 3,143 of 15,942 tables refused in one frame).
        crate::list::tables_done(&lists[..first]);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        // The submit audit (`VICTORY_SUBMIT_AUDIT`): with [`behind`] the
        // parts are prepared later, by another thread; what that will read
        // of title memory is fingerprinted now, on the title's thread.
        if behind() && crate::host::audit::on() {
            // (Behind a wait of the title's: its own, or one the line holds.)
            let waited = parts.iter().any(|p| !p.0.is_empty())
                || lock(&queue(this).held).iter().any(|h| matches!(h, Held::Wait(..) | Held::Value(_)));
            crate::host::audit::called(
                parts.iter().flat_map(|(_, taken, _)| taken.iter().map(crate::list::Detached::host)),
                waited,
            );
        }
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        lock(&queue(this).held).extend(pieces);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Serializes [`release_held`] and queue teardown in [`release`] so a queue
/// present in `WAITING` cannot be freed while [`release_held`] iterates it.
/// Lock order: `RELEASING` -> `WAITING`.
#[cfg(windows)]
static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    if !all && behind() {
        // The thread of ours runs what is free to run ([`behind`]).
        wake();
        return;
    }
    release_held_now(all);
}

/// Whether a submission is only put in its queue's line by the thread that
/// submits it, and prepared and sent by a thread of ours, `victory-submit`
/// (`VICTORY_ASYNC_SUBMIT=1`; off by default; needs submissions sent ahead,
/// `VICTORY_RELAY` or `VICTORY_LATE_WAIT`, and lists cut at their waits,
/// the default). Our port of fxceless f85101b60 (first as 99250881f on
/// all12), brought onto the relay's tree.
///
/// Why (prof-s1 and waits-base2, Series S, about 125 ms a frame): the title
/// thread that submits is the frame's critical path, 42 % of it running in
/// our preparing and 58 % standing, mostly in settles (5.8 a frame, 35 ms,
/// all of them our preparing reading title memory a flight's shaders may
/// write); the title does nothing with the result of that work, which only
/// has to be done before the lists run, in order.
///
/// With the switch a submission's lists are taken from the title's lists
/// (as they already are when the queue holds anything) and put behind the
/// queue's line; [`worker`] runs the lines in order, settles and all, while
/// the title goes on. A signal of the queue goes behind them, as behind
/// anything held. What must see the lines run goes through them on its own
/// thread first ([`before_sync`]): a present, a fence the title blocks on
/// without an event, and every submission of ours that is waited for. So a
/// present returns, as without the switch, only once everything before it
/// has been prepared and sent, and the frame before is complete on the
/// card side when the title begins the next one; with
/// `crate::host::flip_landed` the title's flip wait also waits until the
/// frame before has landed (fx PR #995's `VICTORY_FLIP_LANDED`, tunczzz:
/// its asynchronous submission, which also put the present behind the
/// line, lost the first room's walls because the title read an older
/// frame's results after its flip wait).
///
/// What it assumes, as fxceless's does: title memory a submission reads is
/// not rewritten by the title between its submit call and our thread's
/// preparing of it (on the console the card reads it later still).
#[cfg(windows)]
pub(crate) fn behind() -> bool {
    #[cfg(test)]
    if BEHIND_HERE.get() || BEHIND_ALL.load(Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1");
        let on = asked && crate::host::late() && cuts();
        if on {
            match std::thread::Builder::new().name("victory-submit".into()).spawn(worker) {
                Ok(_) => tracing::info!(
                    "submissions are prepared and sent by a thread of ours, not by the thread that submits (VICTORY_ASYNC_SUBMIT=1)"
                ),
                Err(e) => {
                    tracing::warn!(error = %e, "no thread to prepare submissions on: they are prepared as before");
                    return false;
                }
            }
        } else if asked {
            tracing::warn!(
                "VICTORY_ASYNC_SUBMIT=1 needs submissions sent ahead (VICTORY_RELAY or VICTORY_LATE_WAIT) and lists cut at their waits: off"
            );
        }
        on
    })
}

#[cfg(all(windows, test))]
thread_local! {
    /// Tests: [`behind`] is on for this thread (no worker runs the lines).
    static BEHIND_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: [`group`] on this thread, whatever the switch says.
    static GROUP_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests that send for real (they run alone): [`behind`] is on for every
/// thread, so that the host's helper thread leaves the lines to the test.
#[cfg(all(windows, test))]
static BEHIND_ALL: AtomicBool = AtomicBool::new(false);

/// Whether the thread that runs the lines takes everything that is free to
/// run from them and prepares it as ONE host submission
/// (`VICTORY_GROUP_COMMIT=1`; off by default; needs [`behind`],
/// `VICTORY_ASYNC_SUBMIT=1`).
///
/// Why (the open world, Series S profile, 100-130 ms a frame): the title
/// sends 43-52 submissions a frame, and each is prepared, sent and partly
/// waited for on its own: 20,900 refreshes of host copies a frame for 4,100
/// distinct copies, a landing per submission, and the card with nothing
/// queued for half the frame. With [`behind`] alone our thread still takes
/// one part at a time. While it prepares one, the title queues more: with
/// the switch it takes all of them at once ([`release_grouped`]) and hands
/// the host one submission of all their lists, in the order it would have
/// sent them: each host copy is refreshed once for the batch, it is one
/// `ExecuteCommandLists` and one flight, and its copy-backs land once.
///
/// What stays as it is, and what cuts a batch:
/// - each queue's parts in the order they were given, behind their waits:
///   a queue's wait for a fence not reached, or for a memory value not met,
///   stops that queue's line (a value an earlier part of the batch writes at
///   the bottom of the pipe meets it, as one of a flight in the air does);
/// - a fence signal of a queue is posted when the batch that holds the
///   parts before it has been sent, behind that flight: set when it has
///   landed, never earlier;
/// - bottom-of-pipe writes and read-backs are done when the batch has
///   landed, in the lists' order;
/// - a part whose preparing reads title memory that an earlier part of the
///   batch writes when it has run does not join it
///   (`crate::host::group::Gathered::blocks`): it starts the next batch,
///   which waits for that flight's landing as a single submission does;
/// - so does a part whose preparing reads what a flight already in the air
///   writes (`crate::host::group::meets_flights`): its submission has to
///   wait for that landing, and the parts before it go out now, while the
///   card works, instead of standing through that wait in one batch with it;
/// - a present, a fence the title blocks on, a synchronous submission of
///   ours and a page-mapping change run the lines first on their own thread
///   ([`before_sync`]): whatever was queued before them is sent before they
///   go on, and nothing queued after them can be in the same batch.
#[cfg(windows)]
pub(crate) fn group() -> bool {
    #[cfg(test)]
    if let Some(on) = GROUP_HERE.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_GROUP_COMMIT").is_some_and(|v| v == "1");
        let on = asked && behind();
        if on {
            tracing::info!(
                most = crate::host::group::most(),
                "the parts waiting in the queues' lines are prepared and sent as one host submission (VICTORY_GROUP_COMMIT=1)"
            );
        } else if asked {
            tracing::warn!("VICTORY_GROUP_COMMIT=1 needs VICTORY_ASYNC_SUBMIT=1 (and what that needs): off");
        }
        on
    })
}

/// Whether there is something for [`worker`] to look at.
#[cfg(windows)]
static WORK: (std::sync::Mutex<bool>, std::sync::Condvar) = (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(windows)]
fn wake() {
    *lock(&WORK.0) = true;
    WORK.1.notify_one();
}

/// The thread of [`behind`]: runs what queues hold as far as it is free to
/// run, each time something was added or a fence moved.
#[cfg(windows)]
fn worker() {
    crate::cores::keep_on_fast_cores("victory-submit");
    loop {
        {
            let mut work = lock(&WORK.0);
            while !*work {
                work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
        }
        release_held_now(false);
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread is inside [`release_held_now`].
    static RELEASING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// With [`behind`], before whatever must see what the queues were given
/// before it (a submission of ours that is waited for, a present, a fence
/// the title blocks on): what the queues hold and is free to run runs
/// first, on this thread (behind the part our thread may be sending), so
/// nothing passes it.
#[cfg(windows)]
pub(crate) fn before_sync(why: usize) {
    if behind() && !RELEASING_HERE.get() {
        let stood = std::time::Instant::now();
        // (What ends the batches sent from here: [`release_grouped`].)
        let before = SYNC_WHY.replace(why);
        release_held_now(false);
        SYNC_WHY.set(before);
        BEHIND_STATS[why].fetch_add(stood.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread runs the lines for one of [`before_sync`]'s callers
    /// (its place in [`BEHIND_STATS`]); `usize::MAX`: it does not.
    static SYNC_WHY: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// [`before_sync`]'s callers, the places of [`BEHIND_STATS`] their time is
/// added to: a present, a fence the title blocks on, a submission of ours,
/// a change of page mappings (done at once, behind every earlier
/// submission: [`copy_page_mappings`]).
#[cfg(windows)]
pub(crate) const BEFORE_PRESENT: usize = 0;
#[cfg(windows)]
pub(crate) const BEFORE_FENCE: usize = 1;
#[cfg(windows)]
pub(crate) const BEFORE_OURS: usize = 2;
#[cfg(windows)]
pub(crate) const BEFORE_MAPPING: usize = 3;
/// [`BEHIND_STATS`]: parts sent from the lines with the line open, and
/// microseconds spent preparing and sending them.
#[cfg(windows)]
const SENT_PARTS: usize = 4;
#[cfg(windows)]
const SENT_US: usize = 5;

/// Since the last frame line, with [`behind`]: microseconds title threads
/// stood running the lines (and waiting for our thread to finish the part
/// it sends) at each of [`before_sync`]'s places; then [`SENT_PARTS`] and
/// [`SENT_US`].
#[cfg(windows)]
static BEHIND_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The frame line's account of [`behind`] (`frames`: since the last).
#[cfg(windows)]
pub(crate) fn behind_line(frames: f64) {
    if !behind() {
        return;
    }
    let s: Vec<f64> = BEHIND_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions prepared off the title's submit calls per frame (VICTORY_ASYNC_SUBMIT=1): {:.1} parts sent from the \
         lines in {:.1} ms; the title's threads stood running the lines first {:.1} ms at presents, {:.1} ms at blocking \
         fence waits, {:.1} ms at submissions of ours, {:.1} ms at page mapping changes",
        s[SENT_PARTS],
        s[SENT_US] / 1000.0,
        s[BEFORE_PRESENT] / 1000.0,
        s[BEFORE_FENCE] / 1000.0,
        s[BEFORE_OURS] / 1000.0,
        s[BEFORE_MAPPING] / 1000.0,
    );
    if !group() {
        return;
    }
    let largest = GROUP_STATS[GROUP_LARGEST].swap(0, Ordering::Relaxed);
    let g: Vec<f64> = GROUP_STATS.iter().map(|g| g.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let batches = g[GROUP_BATCHES].max(f64::MIN_POSITIVE);
    tracing::info!(
        "group commit per frame (VICTORY_GROUP_COMMIT=1): {:.1} batches of {:.1} parts and {:.1} lists each on average (the \
         most parts in one since the last line: {largest}), each one host submission; {:.0} refreshes of host copies saved (a \
         copy refreshed by a part of a batch after the first that refreshes it); batches sent per frame by the title's \
         thread at a present {:.1}, at a blocking fence wait {:.1}, at a synchronous submission of ours {:.1}, at a \
         page-mapping change {:.1} (each with whatever was queued before it), by our own thread {:.1}; lines that gave a \
         batch no more, per frame: the next part reads what a part of the batch writes when it has run {:.1}, or what \
         a flight in the air writes {:.1} (it starts the next batch, which waits for the landing), a wait for a memory \
         value {:.1}, the queue's wait for a fence {:.1}, {} lists in the batch {:.1}",
        g[GROUP_BATCHES],
        g[GROUP_PARTS] / batches,
        g[GROUP_LISTS] / batches,
        g[GROUP_SAVED],
        g[BEFORE_PRESENT],
        g[BEFORE_FENCE],
        g[BEFORE_OURS],
        g[BEFORE_MAPPING],
        g[CUT_DRY],
        g[CUT_READ],
        g[CUT_AIR],
        g[CUT_VALUE],
        g[CUT_FENCE],
        crate::host::group::most(),
        g[CUT_MOST],
    );
}

/// The part at the head of `0` is being sent with the line open
/// ([`Held::Running`]): dropped, the marker comes off the head. Dropped by a
/// panic (in the preparing, on our thread or a title thread running the
/// lines): the marker would have stood at the head for good, every later
/// submission, signal and present (even the present's forced pass) behind
/// it, a silent freeze until the title's watchdog; the process is ended
/// with the panic's message instead.
#[cfg(windows)]
struct Marked<'a>(&'a std::sync::Mutex<Vec<Held>>);

#[cfg(windows)]
impl Drop for Marked<'_> {
    fn drop(&mut self) {
        {
            let mut held = lock(self.0);
            if matches!(held.first(), Some(Held::Running)) {
                held.remove(0);
            }
        }
        if std::thread::panicking() {
            tracing::error!(
                "preparing a submission from a queue's line panicked (VICTORY_ASYNC_SUBMIT; the panic's message is above): \
                 the process ends rather than stand still"
            );
            #[cfg(not(test))]
            std::process::abort();
        }
    }
}

/// [`release_held`] on this thread, now (one thread at a time runs the
/// lines: one that finds another doing it waits until it is done).
#[cfg(windows)]
fn release_held_now(all: bool) {
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    let was_here = RELEASING_HERE.replace(true);
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            RELEASING_HERE.set(self.0);
        }
    }
    let _leave = Leave(was_here);
    // Group commit: everything that is free to run, as one submission.
    if !all
        && behind()
        && group()
        && crate::host::late()
        && let Some(host) = crate::host::host()
    {
        release_grouped(&mut || lock(&WAITING).clone(), &mut |lists, order| host.execute_late(lists, order));
        return;
    }
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            loop {
                match held.first() {
                    None | Some(Held::Running) => break,
                    Some(Held::Wait(fence, value))
                        if !all && crate::fence::completed(fence.as_this()) < *value =>
                    {
                        break;
                    }
                    Some(Held::Value(wait)) if !all && !value_met(wait) => break,
                    Some(_) => {}
                }
                match held.remove(0) {
                    Held::Wait(..) | Held::Running => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if all && !value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(lists, order) => {
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Diagnostics: the part behind this one stands at
                        // waits that are met already, so the two could have
                        // been one host submission (one wait for the card,
                        // each range shaders write compared once).
                        #[cfg(feature = "diagnostics")]
                        {
                            let waits = held.iter().take_while(|h| matches!(h, Held::Value(_))).count();
                            if waits > 0
                                && matches!(held.get(waits), Some(Held::Run(..)))
                                && held[..waits].iter().all(|h| matches!(h, Held::Value(w) if w.met()))
                            {
                                crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                            }
                        }
                        if crate::host::late()
                            && let Some(host) = crate::host::host()
                        {
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            if behind() && !all {
                                // The line is open while this part is
                                // prepared and sent (its settles too): the
                                // title adds behind the marker meanwhile.
                                // Only the thread holding `RELEASING` takes
                                // from a line's head.
                                held.insert(0, Held::Running);
                                drop(held);
                                let sending = std::time::Instant::now();
                                {
                                    // Takes the marker off however this
                                    // ends: a panic in the preparing would
                                    // else leave it at the head for good.
                                    let _marked = Marked(line);
                                    host.execute_late(lists, order);
                                }
                                BEHIND_STATS[SENT_PARTS].fetch_add(1, Ordering::Relaxed);
                                BEHIND_STATS[SENT_US].fetch_add(sending.elapsed().as_micros() as u64, Ordering::Relaxed);
                                held = lock(line);
                            } else {
                                host.execute_late(lists, order);
                            }
                        } else {
                            let ran = {
                                let runs: Vec<&crate::host::HostList> =
                                    order.iter().map(|&i| lists[i].host()).collect();
                                match crate::host::host() {
                                    Some(host) => {
                                        host.execute(&runs);
                                        true
                                    }
                                    None => false,
                                }
                            };
                            // They have run: their host lists are used again.
                            if ran {
                                crate::list::recycle(lists, &order);
                            }
                        }
                        if all {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Forced, 1);
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if let Err(fence) = crate::host::late_signal(fence, value) {
                            crate::fence::set(fence.as_this(), value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            break;
        }
    }
}

/// What [`release_grouped`] has gathered for one host submission.
#[cfg(windows)]
#[derive(Default)]
struct Group {
    /// The parts' host lists and the order they run in.
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    /// The fence signals behind them, in the order they were taken.
    after: Vec<(crate::fence::FenceRef, u64)>,
    /// What the parts write once they have run.
    gathered: crate::host::group::Gathered,
    /// Why each line that still holds something gave no more (queue, one of
    /// the `CUT_` places).
    stops: Vec<(usize, usize)>,
}

#[cfg(windows)]
impl Group {
    fn add(&mut self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        {
            let runs: Vec<&crate::host::HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.gathered.add(&runs);
        }
        let first = self.lists.len();
        self.order.extend(order.iter().map(|&i| first + i));
        self.lists.extend(lists);
    }

    /// Whether a line may pass `wait` with these parts gathered: with none,
    /// as always ([`value_met`], which may settle the flights to know).
    /// Else: memory holds the value; or a bottom-of-pipe write of a part
    /// gathered, or of a flight in the air, stores one that meets it. Where
    /// neither says, the line stands until the parts have been sent and is
    /// then asked the usual way.
    fn passes(&self, wait: &crate::list::ValueWait) -> bool {
        if self.gathered.is_empty() {
            return value_met(wait);
        }
        wait.met()
            || match self.gathered.meets(wait) {
                Some(met) => met,
                None => crate::host::group::late_met_peek(wait),
            }
    }
}

/// The queues whose lines have a [`Held::Running`] marker at their head
/// for the batch being gathered and sent ([`release_grouped`], which holds
/// `RELEASING` for as long as this lives): dropped, the markers come off,
/// as [`Marked`]'s does, and for the same reason the process ends if a
/// panic drops it.
#[cfg(windows)]
struct MarkedLines(Vec<usize>);

#[cfg(windows)]
impl Drop for MarkedLines {
    fn drop(&mut self) {
        for &q in &self.0 {
            // SAFETY: a live queue: one released waits for `RELEASING`.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            if matches!(held.first(), Some(Held::Running)) {
                held.remove(0);
            }
        }
        if std::thread::panicking() {
            tracing::error!(
                "preparing a batch from the queues' lines panicked (VICTORY_GROUP_COMMIT; the panic's message is above): \
                 the process ends rather than stand still"
            );
            #[cfg(not(test))]
            std::process::abort();
        }
    }
}

/// [`GROUP_STATS`]: what ended a batch. The first four are
/// [`before_sync`]'s places: the batch was sent by a title thread at a
/// present, a blocking fence wait, a submission of ours, a page-mapping
/// change, which sends whatever is queued before it and lets nothing queued
/// after it join. Then, per line that still held something when the batch
/// went: its next part reads what a part of the batch writes when it has
/// run, it stands at a wait for a memory value, it stands at its queue's
/// wait for a fence, the batch was full. Then: the batch was sent by our
/// own thread (it took what was queued when it looked). And last, per
/// line: its next part reads what a flight in the air writes ([`CUT_AIR`]).
#[cfg(windows)]
const CUT_READ: usize = 4;
#[cfg(windows)]
const CUT_VALUE: usize = 5;
#[cfg(windows)]
const CUT_FENCE: usize = 6;
#[cfg(windows)]
const CUT_MOST: usize = 7;
#[cfg(windows)]
const CUT_DRY: usize = 8;
/// [`GROUP_STATS`] after the cuts: batches sent, the parts and the host
/// lists in them, refreshes saved (a host copy refreshed by a part of a
/// batch after the first part that refreshes it: sent one by one, each
/// part's own refresh), and the most parts in one batch.
#[cfg(windows)]
const GROUP_BATCHES: usize = 9;
#[cfg(windows)]
const GROUP_PARTS: usize = 10;
#[cfg(windows)]
const GROUP_LISTS: usize = 11;
#[cfg(windows)]
const GROUP_SAVED: usize = 12;
#[cfg(windows)]
const GROUP_LARGEST: usize = 13;
#[cfg(windows)]
const CUT_AIR: usize = 14;

/// Since the last frame line, with [`group`]: the cuts by cause (the `CUT_`
/// places), then the `GROUP_` places.
#[cfg(windows)]
static GROUP_STATS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

/// [`release_held_now`] with [`group`]: the lines of `waiting()`'s queues
/// are run as far as they are free to run, and the parts met on the way
/// are not sent one by one: they are gathered, each queue's in its order
/// and the queues in the order they would have been run, and given to
/// `send` as one submission (their host lists and the order those run in).
/// Then the fence signals met behind them are posted, behind that flight,
/// and the lines are looked at again, until nothing moves.
///
/// While a batch is gathered and sent, every line something was taken from
/// has the [`Held::Running`] marker at its head, as the line of a part
/// being sent has without the switch: what the title adds stays behind it.
/// The caller holds `RELEASING`.
#[cfg(windows)]
fn release_grouped(
    waiting: &mut dyn FnMut() -> Vec<usize>,
    send: &mut dyn FnMut(Vec<crate::list::Detached>, Vec<usize>),
) {
    let most = crate::host::group::most();
    loop {
        let queues = waiting();
        let mut group = Group::default();
        let mut marked = MarkedLines(Vec::new());
        // Pass after pass over the lines, until one adds nothing: what a
        // part taken from one queue writes at the bottom of the pipe may be
        // what another queue's wait stands at, and a signal posted at once
        // (nothing gathered before it) may be what another queue waits for.
        loop {
            let mut added = false;
            for &q in &queues {
                // SAFETY: a live queue: one released is taken off the list,
                // and waits for `RELEASING` first.
                let line = unsafe { &(*(q as *const Queue)).held };
                let mut held = lock(line);
                // Our own marker, from an earlier pass: put back below.
                let was_marked = marked.0.contains(&q);
                if was_marked && matches!(held.first(), Some(Held::Running)) {
                    held.remove(0);
                }
                let mut mark = was_marked;
                group.stops.retain(|s| s.0 != q);
                loop {
                    let stop = match held.first() {
                        None | Some(Held::Running) => break,
                        Some(Held::Wait(fence, value)) => {
                            (crate::fence::completed(fence.as_this()) < *value).then_some(CUT_FENCE)
                        }
                        Some(Held::Value(wait)) => (!group.passes(wait)).then_some(CUT_VALUE),
                        Some(Held::Run(lists, order)) if !group.gathered.is_empty() => {
                            let runs: Vec<&crate::host::HostList> = order.iter().map(|&i| lists[i].host()).collect();
                            if group.gathered.blocks(&runs) {
                                Some(CUT_READ)
                            } else if crate::host::group::meets_flights(&runs) {
                                Some(CUT_AIR)
                            } else if group.gathered.lists + runs.len() > most {
                                Some(CUT_MOST)
                            } else {
                                None
                            }
                        }
                        Some(Held::Run(..) | Held::Signal(..)) => None,
                    };
                    if let Some(cause) = stop {
                        group.stops.push((q, cause));
                        break;
                    }
                    match held.remove(0) {
                        Held::Wait(..) | Held::Value(_) | Held::Running => {}
                        Held::Run(lists, order) => {
                            #[cfg(feature = "diagnostics")]
                            crate::capture::op(q, || format!("RUN held submission of {} lists (in a batch)", order.len()));
                            group.add(lists, order);
                            mark = true;
                        }
                        // Behind the parts gathered so far, whichever
                        // queue's: sent one by one they would be in the air
                        // by now, and the signal behind them. With nothing
                        // gathered: now, as without batches.
                        Held::Signal(fence, value) if group.gathered.is_empty() => {
                            if let Err(fence) = crate::host::late_signal(fence, value) {
                                crate::fence::set(fence.as_this(), value);
                            }
                        }
                        Held::Signal(fence, value) => {
                            group.after.push((fence, value));
                            mark = true;
                        }
                    }
                    added = true;
                }
                if mark {
                    held.insert(0, Held::Running);
                    if !was_marked {
                        marked.0.push(q);
                    }
                } else if held.is_empty() {
                    lock(&WAITING).retain(|&w| w != q);
                }
            }
            if !added {
                break;
            }
        }
        let Group { lists, order, after, mut gathered, stops } = group;
        if lists.is_empty() {
            // Nothing to send (and so no signal kept, no line marked):
            // the passes above ran the lines as far as they go.
            break;
        }
        let sending = std::time::Instant::now();
        // (VICTORY_FILL_AHEAD: the batch's fills ahead of its own parts,
        // for the host to note under the batch's number.)
        gathered.hand_ahead();
        send(lists, order);
        drop(crate::host::group::take_ahead());
        // The signals behind the parts: posted with the markers still on,
        // so that a signal the title gives one of those queues now stays
        // behind them.
        for (fence, value) in after {
            if let Err(fence) = crate::host::late_signal(fence, value) {
                crate::fence::set(fence.as_this(), value);
            }
        }
        drop(marked);
        BEHIND_STATS[SENT_PARTS].fetch_add(gathered.parts as u64, Ordering::Relaxed);
        BEHIND_STATS[SENT_US].fetch_add(sending.elapsed().as_micros() as u64, Ordering::Relaxed);
        GROUP_STATS[GROUP_BATCHES].fetch_add(1, Ordering::Relaxed);
        GROUP_STATS[GROUP_PARTS].fetch_add(gathered.parts as u64, Ordering::Relaxed);
        GROUP_STATS[GROUP_LISTS].fetch_add(gathered.lists as u64, Ordering::Relaxed);
        GROUP_STATS[GROUP_SAVED].fetch_add(gathered.saved, Ordering::Relaxed);
        GROUP_STATS[GROUP_LARGEST].fetch_max(gathered.parts as u64, Ordering::Relaxed);
        // Who sent it: a title thread at one of `before_sync`'s places,
        // or our own.
        let why = SYNC_WHY.get();
        GROUP_STATS[if why <= BEFORE_MAPPING { why } else { CUT_DRY }].fetch_add(1, Ordering::Relaxed);
        for (_, cause) in stops {
            GROUP_STATS[cause].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || (crate::host::late() && crate::host::late_met(wait))
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed_with::<Queue>(this, || {
        #[cfg(windows)]
        {
            let _one = lock(&RELEASING);
            lock(&WAITING).retain(|&q| q != this as usize);
        }
    })
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        crate::cores::keep_submitter_on_fast_cores();
        // VICTORY_METERS=1: this call's time on the calling thread.
        let _inside = crate::meters::submission();
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            Held::Run(taken, order)
        });
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// A present plane's description (`D3D12XBOX_PRESENT_PLANE_PARAMETERS`),
/// [`PLANE_STRIDE`] bytes, in the field order of the public GDK binding:
/// frame token, the plane's resources, their source rectangles, the
/// destination placement, colour space, scale filter, extended
/// descriptions and flags. The title presents an array of them; one plane
/// reads only the first three fields, as before. The 72-byte stride is
/// ZSlayerHQ's runtime #10 (Cleverdee, found against the second plane's
/// token); the layout as vx-layer and Kamiloj9 (1b4e37cc) have it.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
    /// `D3D12_RECT`s (`[left, top, right, bottom]`), one per resource, or
    /// null.
    src_rects: *const [i32; 4],
    /// Where the plane goes in the output (`[left, top, right, bottom]`,
    /// fractions of it), or null.
    dest: *const [f32; 4],
    color_space: u32,
    scale_filter: u32,
    extended_count: u32,
    /// The extended descriptions themselves: not read.
    _extended: *const u8,
    flags: u32,
}

/// Bytes from one plane's description to the next.
const PLANE_STRIDE: usize = 72;
const _: () = assert!(std::mem::size_of::<Plane>() == PLANE_STRIDE);
const _: () = assert!(std::mem::offset_of!(Plane, count) == 8);
const _: () = assert!(std::mem::offset_of!(Plane, resources) == 16);
const _: () = assert!(std::mem::offset_of!(Plane, src_rects) == 24);
const _: () = assert!(std::mem::offset_of!(Plane, dest) == 32);
const _: () = assert!(std::mem::offset_of!(Plane, color_space) == 40);
const _: () = assert!(std::mem::offset_of!(Plane, scale_filter) == 44);
const _: () = assert!(std::mem::offset_of!(Plane, _extended) == 56);
const _: () = assert!(std::mem::offset_of!(Plane, flags) == 64);

/// Planes composed at most (the first four).
#[cfg(windows)]
const MAX_PLANES: usize = 4;

/// A present's plane as the title describes it: its first resource and
/// what the composition reads or logs of it.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct Described {
    resource: This,
    /// Whether its token is the first plane's (a check of the stride).
    same_token: bool,
    src: Option<[i32; 4]>,
    dest: Option<[f32; 4]>,
    color_space: u32,
    scale_filter: u32,
    extended: u32,
    flags: u32,
}

/// The first [`MAX_PLANES`] planes of a present that name a resource, each
/// resource once (a repeat is not drawn over itself).
///
/// # Safety
/// `parameters` is null or points at `planes` plane descriptions, each
/// listing `count` resources at `resources` and, when `src_rects` or
/// `dest` is set, as many rectangles there.
#[cfg(windows)]
unsafe fn described(planes: u32, parameters: *const Plane) -> Vec<Described> {
    let mut found: Vec<Described> = Vec::new();
    if parameters.is_null() || planes == 0 {
        return found;
    }
    // SAFETY: the caller's: at least one description.
    let token = unsafe { (*parameters).token };
    for i in 0..(planes as usize).min(MAX_PLANES) {
        // SAFETY: the caller's: description `i` of `planes`.
        let plane = unsafe { &*parameters.add(i) };
        if plane.count == 0 || plane.resources.is_null() {
            continue;
        }
        // SAFETY: the caller's: the plane lists `count` resources.
        let resource = unsafe { plane.resources.read_unaligned() };
        if resource.is_null() || found.iter().any(|d| d.resource == resource) {
            continue;
        }
        // SAFETY: the caller's: a rectangle per resource when set.
        let src = (!plane.src_rects.is_null()).then(|| unsafe { plane.src_rects.read_unaligned() });
        // SAFETY: as above, for the placement.
        let dest = (!plane.dest.is_null()).then(|| unsafe { plane.dest.read_unaligned() });
        found.push(Described {
            resource,
            same_token: plane.token == token,
            src,
            dest,
            color_space: plane.color_space,
            scale_filter: plane.scale_filter,
            extended: plane.extended_count,
            flags: plane.flags,
        });
    }
    found
}

/// `VICTORY_COMPOSE_PLANES` (`var`): `None` when presents show their first
/// plane alone (`0`, as before); else the blending forced on every plane
/// after the first (`premul`, `straight`; unset or anything else: by each
/// plane's flags). On by default: a present of one plane takes the plain
/// copy whatever this says, so only presents of several planes (Series S)
/// change.
#[cfg(windows)]
fn compose_switch(var: Option<&str>) -> Option<Option<crate::compose::Alpha>> {
    use crate::compose::Alpha;
    match var {
        Some("0") => None,
        Some("premul") => Some(Some(Alpha::Premultiplied)),
        Some("straight") => Some(Some(Alpha::Straight)),
        _ => Some(None),
    }
}

/// [`compose_switch`] as this run has it.
#[cfg(windows)]
fn compose_planes() -> Option<Option<crate::compose::Alpha>> {
    static SWITCH: std::sync::OnceLock<Option<Option<crate::compose::Alpha>>> =
        std::sync::OnceLock::new();
    *SWITCH.get_or_init(|| {
        let switch = compose_switch(std::env::var("VICTORY_COMPOSE_PLANES").ok().as_deref());
        tracing::info!(?switch, "present planes: composed unless None (VICTORY_COMPOSE_PLANES=0)");
        switch
    })
}

/// Shows a present of several planes composed ([`crate::compose`]); false
/// if it is not one (fewer than two planes with a host texture, or a plane
/// the composition cannot read), for the plain copy of the first plane.
///
/// # Safety
/// As [`described`].
#[cfg(windows)]
unsafe fn present_composed(
    host: &crate::host::Host,
    planes: u32,
    parameters: *const Plane,
    alpha: Option<crate::compose::Alpha>,
    frame: u64,
) -> bool {
    use crate::compose::{Alpha, ShownPlane, composable};
    // SAFETY: the caller's.
    let found = unsafe { described(planes, parameters) };
    if found.len() < 2 {
        return false;
    }
    let textures: Vec<_> = found
        .iter()
        .filter_map(|d| Some((d, crate::resource::host_resource(d.resource)?)))
        .collect();
    log_planes(planes, &textures);
    if textures.len() < 2 {
        return false;
    }
    if !textures.iter().all(|(_, t)| composable(t)) {
        static TOLD: AtomicBool = AtomicBool::new(false);
        if !TOLD.swap(true, Ordering::Relaxed) {
            tracing::warn!("present planes the composition cannot read: the first is shown alone");
        }
        return false;
    }
    // Each plane goes to the pixel-shader-readable state for the
    // composition and back ([`crate::host::Host::present_planes`]).
    const READABLE: u32 = 0x80;
    let shown: Vec<ShownPlane<'_>> = textures
        .iter()
        .map(|(d, texture)| ShownPlane {
            texture,
            state: crate::resource::host_transition(d.resource, READABLE)
                .map_or(READABLE, |(_, before)| before),
            src: d.src,
            dest: d.dest,
            alpha: Alpha::of(d.flags, alpha),
        })
        .collect();
    let shown_now = host.present_planes(&shown, frame);
    for ((d, _), plane) in textures.iter().zip(&shown) {
        crate::resource::host_transition(d.resource, plane.state);
    }
    #[cfg(feature = "diagnostics")]
    blinks(frame, textures[0].0.resource, shown[0].state, shown_now);
    #[cfg(not(feature = "diagnostics"))]
    let _ = shown_now;
    true
}

/// Logs a present's planes whenever what they are changes: how many the
/// title gave, then for each one shown its size, format, source rectangle,
/// placement, flags, colour space, scale filter, whether it has extended
/// descriptions, and whether its token is the first plane's.
#[cfg(windows)]
fn log_planes(
    planes: u32,
    textures: &[(
        &Described,
        &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    )],
) {
    static LAST: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
    let mut text = format!("{planes} planes, stride {PLANE_STRIDE}:");
    for (d, texture) in textures {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        text += &format!(
            " [{}x{} format {} src {:?} dest {:?} flags {:#x} colour space {} filter {} extended {} same token {}]",
            desc.Width,
            desc.Height,
            desc.Format.0,
            d.src,
            d.dest,
            d.flags,
            d.color_space,
            d.scale_filter,
            d.extended,
            d.same_token
        );
    }
    let mut last = lock(&LAST);
    if *last != text {
        tracing::info!(planes = %text, "present planes changed");
        *last = text;
    }
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(crate::fence::FenceRef::new(fence), value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late()
        && crate::host::late_signal(crate::fence::FenceRef::new(fence), value).is_ok()
    {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || now < value;
            if behind && waits() {
                held.push(Held::Wait(crate::fence::FenceRef::new(fence), value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// The title's presents so far (each call of its present counts once,
/// inside the call, before its picture is shown): the frame count of `VICTORY_KEYS_FILE`'s frame holds and of the
/// window title.
pub fn presents() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the planes in our window. One plane (or
/// `VICTORY_COMPOSE_PLANES=0`): its first resource, copied as it is.
/// Several (Series S: the game image under the interface): composed, each
/// plane's source rectangle stretched over its placement
/// ([`present_composed`]).
///
/// shortcut: one plane is copied with no scaling or placement.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // VICTORY_METERS=1: this call's time on the calling thread.
    #[cfg(windows)]
    let _inside = crate::meters::present();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    // (With `behind`: what the queues hold runs here first, on this
    // thread, so the present goes behind it as without the switch.)
    #[cfg(windows)]
    before_sync(BEFORE_PRESENT);
    #[cfg(windows)]
    if crate::host::late() && !crate::host::late_free_present() {
        crate::host::settle_for_present();
        release_held_now(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here
        // (but flights older than `VICTORY_LATE_FRAMES` frames).
        crate::host::late_frames_bound();
        release_held_now(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    release_held(forces());
    // What the title gave the card up to this present, for the next
    // frame's flip wait (`crate::host::flip_landed`).
    #[cfg(windows)]
    crate::host::present_sent();
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(windows)]
    note_world(crate::draw::drawn());
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if crate::compose::trace_on() {
        // SAFETY: the title passes `planes` plane descriptions.
        let found = unsafe { described(planes, plane_parameters) };
        let mut text = format!("title presents {planes} planes, {} with a resource:", found.len());
        for d in &found {
            let size = crate::resource::host_resource(d.resource).map(|t| {
                // SAFETY: a live resource.
                let desc = unsafe { t.GetDesc() };
                format!("{}x{} format {}", desc.Width, desc.Height, desc.Format.0)
            });
            text += &format!(" [{size:?} src {:?} dest {:?} flags {:#x}]", d.src, d.dest, d.flags);
        }
        crate::compose::trace(1, "title present", text);
    }
    #[cfg(windows)]
    if planes > 1
        && let Some(alpha) = compose_planes()
        && let Some(host) = crate::host::host()
        // SAFETY: the title passes `planes` plane descriptions.
        && unsafe { present_composed(host, planes, plane_parameters, alpha, frame) }
    {
        return S_OK;
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        // A plane not written since it was last shown, while another holds
        // a newer picture: traced, and with `VICTORY_PRESENT_NEWEST=1` the
        // newer one is shown ([`crate::shown`]). It is read in the state
        // the lists submitted so far leave it in, and stays in it.
        let (state, shown) = if let Some(newer) = crate::shown::present(frame, resource) {
            let state = newer.state.load(Ordering::Relaxed);
            (state, host.present(&newer.texture, state, frame))
        } else {
            let state = crate::resource::host_transition(resource, COPY_SOURCE)
                .map_or(COPY_SOURCE, |(_, before)| before);
            let shown = host.present(texture, state, frame);
            crate::resource::host_transition(resource, state);
            (state, shown)
        };
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (state, shown);
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    // With `VICTORY_ASYNC_SUBMIT` the submissions the title made before
    // this are first prepared (they read the pages as mapped before it).
    #[cfg(windows)]
    before_sync(BEFORE_MAPPING);
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Two planes at the console's 72-byte stride: both read, with their
    /// rectangles, placements and flags; a repeated resource and an empty
    /// plane are left out.
    #[cfg(windows)]
    #[test]
    fn present_planes_are_read_at_the_console_stride() {
        let (game, ui): (This, This) = (std::ptr::without_provenance_mut(0x10), std::ptr::without_provenance_mut(0x20));
        let half = [0, 0, 960, 540];
        let whole = [0.0f32, 0.0, 1.0, 1.0];
        let plane = |resources: *const This, count, src: *const [i32; 4], dest: *const [f32; 4], flags| Plane {
            token: 7,
            count,
            resources,
            src_rects: src,
            dest,
            color_space: 0,
            scale_filter: 0,
            extended_count: 0,
            _extended: std::ptr::null(),
            flags,
        };
        let planes = [
            plane(&game, 1, &half, &whole, 0),
            plane(&ui, 1, std::ptr::null(), std::ptr::null(), 0x2),
            plane(&game, 1, std::ptr::null(), std::ptr::null(), 0),
            plane(std::ptr::null(), 0, std::ptr::null(), std::ptr::null(), 0),
        ];
        // SAFETY: four descriptions whose pointers name live locals.
        let found = unsafe { described(4, planes.as_ptr()) };
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].resource, found[0].src, found[0].dest), (game, Some(half), Some(whole)));
        assert_eq!((found[1].resource, found[1].src, found[1].dest), (ui, None, None));
        assert!(found.iter().all(|d| d.same_token));
        assert_eq!(found[1].flags, 0x2);
        // SAFETY: a null array is allowed.
        assert!(unsafe { described(2, std::ptr::null()) }.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn planes_are_composed_unless_switched_off() {
        use crate::compose::Alpha;
        assert_eq!(compose_switch(None), Some(None));
        assert_eq!(compose_switch(Some("1")), Some(None));
        assert_eq!(compose_switch(Some("0")), None);
        assert_eq!(compose_switch(Some("premul")), Some(Some(Alpha::Premultiplied)));
        assert_eq!(compose_switch(Some("straight")), Some(Some(Alpha::Straight)));
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }

    #[cfg(windows)]
    #[test]
    fn release_waits_for_release_held_before_freeing_waiting_queue() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let mut fence = null;
        assert_eq!(
            crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence),
            S_OK
        );
        // Queue waits on an unsignaled fence, registering `q` in `WAITING`.
        assert_eq!(wait(q, fence, 1), S_OK);
        let q_addr = q as usize;
        assert!(lock(&WAITING).contains(&q_addr));

        // Hold `RELEASING` (as `release_held` does while iterating `WAITING`).
        // Releasing the last reference to `q` must wait before freeing `q` and
        // must remove `q` from `WAITING` before dropping its allocation.
        let guard = lock(&RELEASING);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let releaser = std::thread::spawn(move || {
            let left = slot::<extern "system" fn(This) -> u32>(q_addr as This, 0x10)(q_addr as This);
            let _ = done_tx.send(left);
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(40))
                .is_err(),
            "queue::release freed queue while release_held lock was held"
        );
        drop(guard);
        let left = done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("queue::release finished");
        assert_eq!(left, 0);
        assert!(!lock(&WAITING).contains(&q_addr));
        releaser.join().unwrap();
        assert_eq!(slot::<extern "system" fn(This) -> u32>(fence, 0x10)(fence), 0);
    }

    /// A queue and two fences for the ordering tests.
    #[cfg(windows)]
    fn queue_and_fences() -> (This, This, This) {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        let (mut f1, mut f2) = (null, null);
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut f1), S_OK);
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut f2), S_OK);
        (q, f1, f2)
    }

    #[cfg(windows)]
    fn release_all(objects: &[This]) {
        for &o in objects {
            while slot::<extern "system" fn(This) -> u32>(o, 0x10)(o) > 0 {}
        }
    }

    /// `VICTORY_ASYNC_SUBMIT`: while the part at a line's head is being
    /// sent (the `Running` marker), nothing behind it runs, a signal
    /// included, whoever runs the lines; once it is sent, the rest does.
    #[cfg(windows)]
    #[test]
    fn what_follows_a_part_being_sent_waits_for_it() {
        let (q, f1, f2) = queue_and_fences();
        {
            let mut held = lock(&queue(q).held);
            held.push(Held::Running);
            held.push(Held::Signal(crate::fence::FenceRef::new(f1), 3));
        }
        lock(&WAITING).push(q as usize);
        release_held_now(false);
        assert_eq!(crate::fence::completed(f1), 0, "a signal passed the part being sent");
        {
            let mut held = lock(&queue(q).held);
            assert!(matches!(held.first(), Some(Held::Running)));
            held.remove(0);
        }
        release_held_now(false);
        assert_eq!(crate::fence::completed(f1), 3);
        assert!(!lock(&WAITING).contains(&(q as usize)));
        release_all(&[q, f1, f2]);
    }

    /// `VICTORY_ASYNC_SUBMIT`: a panic while a part is sent with the line
    /// open does not leave the marker at the line's head (everything behind
    /// it would stand for good); what follows runs again.
    #[cfg(windows)]
    #[test]
    fn a_panic_while_sending_takes_the_marker_off() {
        let (q, f1, f2) = queue_and_fences();
        {
            let mut held = lock(&queue(q).held);
            held.push(Held::Running);
            held.push(Held::Signal(crate::fence::FenceRef::new(f1), 5));
        }
        lock(&WAITING).push(q as usize);
        let line = &queue(q).held;
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _marked = Marked(line);
            panic!("preparing failed");
        }));
        assert!(caught.is_err());
        assert!(!matches!(lock(line).first(), Some(Held::Running)), "the marker stayed at the head");
        release_held_now(false);
        assert_eq!(crate::fence::completed(f1), 5);
        release_all(&[q, f1, f2]);
    }

    /// `VICTORY_ASYNC_SUBMIT`: a fence the title blocks on (no event) whose
    /// signal is still in a queue's line, behind a wait our thread has not
    /// looked at again, is reached: the blocking wait runs the lines on its
    /// own thread first. Without that it found the fence short and failed.
    #[cfg(windows)]
    #[test]
    fn a_blocking_fence_wait_sees_signals_still_in_the_lines() {
        use std::ffi::c_void;
        BEHIND_HERE.set(true);
        let (q, f1, f2) = queue_and_fences();
        // The queue waits for f1 = 1, then signals f2 = 7 behind it.
        assert_eq!(wait(q, f1, 1), S_OK);
        assert_eq!(signal(q, f2, 7), S_OK);
        // The title meets f1 on the processor; with the switch that only
        // wakes our thread (none runs in this test).
        assert_eq!(crate::fence::signal(f1, 1), S_OK);
        let set_event = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f2, 0x48);
        assert_eq!(set_event(f2, 7, std::ptr::null_mut()), S_OK, "the blocking wait did not see the signal in the line");
        assert_eq!(crate::fence::completed(f2), 7);
        BEHIND_HERE.set(false);
        release_all(&[q, f1, f2]);
    }

    #[cfg(windows)]
    #[test]
    fn held_wait_and_signal_retain_fences_until_released() {
        use std::ffi::c_void;
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        let mut f1 = null;
        let mut f2 = null;
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut f1), S_OK);
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut f2), S_OK);

        // SAFETY: manual-reset Win32 event for fence completion.
        let ev = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!ev.is_null());
        let set_event = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f2, 0x48);
        assert_eq!(set_event(f2, 7, ev), S_OK);

        // Queue waits on f1 == 1, then holds a signal on f2 == 7.
        assert_eq!(wait(q, f1, 1), S_OK);
        assert_eq!(signal(q, f2, 7), S_OK);

        // Title drops its reference on f2 immediately; Held::Signal keeps it alive.
        let rel2 = slot::<extern "system" fn(This) -> u32>(f2, 0x10);
        assert_eq!(rel2(f2), 1, "Held::Signal must retain a COM ref on f2");

        // Confirm Held::Wait retains a COM ref on f1 as well.
        let add1 = slot::<extern "system" fn(This) -> u32>(f1, 0x08);
        let rel1 = slot::<extern "system" fn(This) -> u32>(f1, 0x10);
        assert_eq!(add1(f1), 3, "test + Held::Wait + add1");
        assert_eq!(rel1(f1), 2);

        // Satisfying f1 drains Held::Wait(f1, 1) and Held::Signal(f2, 7), setting
        // the event and releasing both internal FenceRefs without UAF or leaks.
        assert_eq!(crate::fence::signal(f1, 1), S_OK);
        // SAFETY: valid Win32 event handle.
        assert_eq!(unsafe { WaitForSingleObject(ev, 0) }, WAIT_OBJECT_0);
        // SAFETY: closing the test event handle.
        unsafe { CloseHandle(ev) };
        assert_eq!(rel1(f1), 0, "f1 internal ref must drop after release_held");
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The group-commit tests read [`GROUP_STATS`]: one at a time.
    #[cfg(windows)]
    static GROUP_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A new host list recorded by `record` and closed, as a part in a
    /// queue's line; and its name in these tests (its list's interface
    /// pointer, which stays when the host list moves).
    #[cfg(windows)]
    fn part(host: &crate::host::Host, record: impl FnOnce(&crate::host::HostList)) -> (Held, usize) {
        use windows::core::Interface;
        let list = host.new_list().unwrap();
        record(&list);
        list.close();
        let name = list.list.as_raw() as usize;
        (Held::Run(vec![crate::list::Detached::ours(list)], vec![0]), name)
    }

    /// Runs the lines of `queues` with group commit, each batch given to
    /// these tests instead of the host: the names of its lists, in order.
    /// `sent` is told when a batch is "sent" (which one, counted from 1),
    /// before the signals behind it are posted. The queues are not in
    /// `WAITING`: other tests' threads do not run their lines.
    #[cfg(windows)]
    fn run_grouped(queues: &[This], mut sent: impl FnMut(usize)) -> Vec<Vec<usize>> {
        use windows::core::Interface;
        let waiting: Vec<usize> = queues.iter().map(|&q| q as usize).collect();
        let mut batches: Vec<Vec<usize>> = Vec::new();
        release_grouped(&mut || waiting.clone(), &mut |lists, order| {
            batches.push(order.iter().map(|&i| lists[i].host().list.as_raw() as usize).collect());
            sent(batches.len());
        });
        batches
    }

    /// 64 KiB of "title memory" holding a title buffer, and its host copy.
    #[cfg(windows)]
    fn title_buffer(resource: usize) -> (usize, std::sync::Arc<crate::shadow::Shadow>) {
        let memory: &'static mut [u8] = Box::leak(vec![0x11u8; 1 << 16].into_boxed_slice());
        let start = memory.as_ptr() as u64;
        crate::shadow::register(resource, start, 1 << 16);
        (start as usize, crate::shadow::find(start, 1 << 16).expect("a host copy"))
    }

    #[cfg(windows)]
    fn group_stat(place: usize) -> u64 {
        GROUP_STATS[place].load(Ordering::Relaxed)
    }

    /// `VICTORY_GROUP_COMMIT`: the parts waiting in a line go out as one
    /// submission, in order; a part whose preparing reads title memory that
    /// a part of the batch writes when it has run (here: a fill of bytes an
    /// earlier part's shaders write) does not join it and starts the next
    /// batch, with what follows it. A copy two parts of a batch refresh is
    /// counted as a refresh saved.
    #[cfg(windows)]
    #[test]
    fn a_part_reading_what_the_batch_writes_starts_the_next_batch() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _alone = lock(&GROUP_TESTS);
        let (base, copy) = title_buffer(0x6c00_0001);
        let (q, f1, f2) = queue_and_fences();
        // (The fills' destination: these lists are never run.)
        let to = Box::leak(vec![0u8; 4096].into_boxed_slice()).as_mut_ptr();
        let other = crate::host::fill_kind::OTHER;
        // A: its shaders write [64, 128) of the buffer. B: a fill from
        // elsewhere in it, and a refresh of the copy A refreshed. C: a fill
        // of bytes A's shaders write. D: nothing of the title's.
        let (a, a_name) = part(host, |l| {
            copy.refresh(l, base as u64, 256);
            copy.gpu_writes(l, base as u64 + 64, 64);
        });
        let (b, b_name) = part(host, |l| {
            l.defer_fill_as(to, (base + 1024) as *const u8, 64, other);
            copy.refresh(l, base as u64 + 4096, 256);
        });
        let (c, c_name) = part(host, |l| l.defer_fill_as(to, (base + 96) as *const u8, 16, other));
        let (d, d_name) = part(host, |_| {});
        lock(&queue(q).held).extend([a, b, c, d]);
        let before = [CUT_READ, GROUP_BATCHES, GROUP_PARTS, GROUP_LISTS, GROUP_SAVED].map(group_stat);
        let batches = run_grouped(&[q], |_| {});
        assert_eq!(batches, [vec![a_name, b_name], vec![c_name, d_name]]);
        let after = [CUT_READ, GROUP_BATCHES, GROUP_PARTS, GROUP_LISTS, GROUP_SAVED].map(group_stat);
        assert_eq!([0, 1, 2, 3, 4].map(|i| after[i] - before[i]), [1, 2, 4, 4, 1], "cuts for a read, batches, parts, lists, refreshes saved");
        assert!(lock(&queue(q).held).is_empty());
        crate::shadow::unregister(0x6c00_0001, base as u64);
        release_all(&[q, f1, f2]);
    }

    /// `VICTORY_GROUP_COMMIT`: a queue that waits for a fence is not in the
    /// batch whose signal it waits for: the signal is posted only when the
    /// parts before it have been sent, and the waiting queue's part goes in
    /// the next batch. While a batch is being sent its queue's line has the
    /// marker at its head: a signal the title gives that queue meanwhile
    /// stays behind the batch.
    #[cfg(windows)]
    #[test]
    fn a_queue_waiting_for_a_fence_is_not_in_the_batch_that_signals_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _alone = lock(&GROUP_TESTS);
        let (q1, f1, f2) = queue_and_fences();
        let (q2, f3, f4) = queue_and_fences();
        let (a, a_name) = part(host, |_| {});
        let (b, b_name) = part(host, |_| {});
        let (c, c_name) = part(host, |_| {});
        // Queue 1: A, signal f1 = 1, B. Queue 2: wait for f1 = 1, then C.
        lock(&queue(q1).held).extend([a, Held::Signal(crate::fence::FenceRef::new(f1), 1), b]);
        lock(&queue(q2).held).extend([Held::Wait(crate::fence::FenceRef::new(f1), 1), c]);
        let cuts = group_stat(CUT_FENCE);
        let mut seen = Vec::new();
        let batches = run_grouped(&[q1, q2], |n| {
            // The fence when the batch goes out: not set before the lists
            // before its signal were sent.
            seen.push(crate::fence::completed(f1));
            if n == 1 {
                assert!(matches!(lock(&queue(q1).held).first(), Some(Held::Running)), "no marker on the line being sent");
                assert_eq!(signal(q1, f2, 5), S_OK);
                assert_eq!(crate::fence::completed(f2), 0, "a signal passed the batch being sent");
            }
        });
        assert_eq!(batches, [vec![a_name, b_name], vec![c_name]]);
        assert_eq!(seen, [0, 1]);
        assert_eq!(crate::fence::completed(f1), 1);
        assert_eq!(crate::fence::completed(f2), 5, "the title's signal ran behind the batch");
        assert_eq!(group_stat(CUT_FENCE) - cuts, 1);
        assert!(lock(&queue(q1).held).is_empty() && lock(&queue(q2).held).is_empty());
        release_all(&[q1, q2, f1, f2, f3, f4]);
    }

    /// `VICTORY_GROUP_COMMIT`: a wait for a memory value that a part of the
    /// batch writes at the bottom of the pipe is met (the waiting part goes
    /// to the card behind it, in the same submission); one nothing meets
    /// stops its line, which runs when the value is there.
    #[cfg(windows)]
    #[test]
    fn a_wait_for_a_value_is_met_by_a_part_of_the_batch_or_stops_its_line() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _alone = lock(&GROUP_TESTS);
        let token: &'static mut [u32; 2] = Box::leak(Box::new([0, 0]));
        let at = token.as_mut_ptr() as usize;
        let (q1, f1, f2) = queue_and_fences();
        let (q2, f3, f4) = queue_and_fences();
        let (q3, f5, f6) = queue_and_fences();
        const EQUAL: u32 = 3;
        // Queue 1: A, which writes 7 to the first word when it has run.
        // Queue 2: waits for that 7, then B. Queue 3: waits for a 9 in the
        // second word, which nothing queued writes, then C.
        let (a, a_name) = part(host, |l| l.defer_write(at as *mut u8, Written::Value(7), 4));
        let (b, b_name) = part(host, |_| {});
        let (c, c_name) = part(host, |_| {});
        lock(&queue(q1).held).push(a);
        lock(&queue(q2).held).extend([Held::Value(crate::list::ValueWait::of(at as u64, EQUAL, 7)), b]);
        lock(&queue(q3).held).extend([Held::Value(crate::list::ValueWait::of(at as u64 + 4, EQUAL, 9)), c]);
        let cuts = group_stat(CUT_VALUE);
        assert_eq!(run_grouped(&[q1, q2, q3], |_| {}), [vec![a_name, b_name]]);
        assert_eq!(group_stat(CUT_VALUE) - cuts, 1);
        assert_eq!(lock(&queue(q3).held).len(), 2, "the queue passed a wait nothing met");
        // SAFETY: the leaked words.
        unsafe { (at as *mut u32).add(1).write_volatile(9) };
        assert_eq!(run_grouped(&[q1, q2, q3], |_| {}), [vec![c_name]]);
        release_all(&[q1, q2, q3, f1, f2, f3, f4, f5, f6]);
    }

    /// `VICTORY_GROUP_COMMIT`: a batch holds at most
    /// `crate::host::group::most` host lists; the part that would pass it
    /// starts the next batch.
    #[cfg(windows)]
    #[test]
    fn a_batch_is_cut_at_its_most_lists() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _alone = lock(&GROUP_TESTS);
        let most = crate::host::group::most();
        let (q, f1, f2) = queue_and_fences();
        // Three parts of two fifths of the most each.
        let lists = (most * 2 / 5).max(1);
        let big = || {
            let taken: Vec<crate::list::Detached> = (0..lists)
                .map(|_| {
                    let list = host.new_list().unwrap();
                    list.close();
                    crate::list::Detached::ours(list)
                })
                .collect();
            Held::Run(taken, (0..lists).collect())
        };
        lock(&queue(q).held).extend([big(), big(), big()]);
        let cuts = group_stat(CUT_MOST);
        let batches = run_grouped(&[q], |_| {});
        if 3 * lists > most {
            assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [2 * lists, lists]);
            assert_eq!(group_stat(CUT_MOST) - cuts, 1);
        } else {
            assert_eq!(batches.len(), 1);
        }
        release_all(&[q, f1, f2]);
    }

    /// What a scripted run of fake lists leaves behind: the two title
    /// buffers' bytes, what the lists copied to a read-back buffer on the
    /// card (what their fills and host copies held when they ran), the
    /// bottom-of-pipe word, the fence, and how many host submissions it
    /// took.
    #[cfg(windows)]
    #[derive(Debug)]
    struct Left {
        title: Vec<u8>,
        card: Vec<u8>,
        token: u32,
        fence: u64,
        sent: u64,
    }

    /// Queues the script's six parts on two queues and runs the lines until
    /// they are empty, one part a submission or with group commit:
    ///
    /// graphics queue: G1 writes 64 bytes of buffer A on the card and
    /// declares them (they land in title memory); G2 fills from buffer B;
    /// G3 fills from the bytes G1 writes; G4 reads buffer B's host copy and
    /// writes 7 to a word at the bottom of the pipe; then a signal;
    /// compute queue: a wait for that 7, C1 fills from buffer B; a wait for
    /// the signal's fence, C2 reads buffer A's host copy.
    #[cfg(windows)]
    fn script(grouped: bool) -> Left {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const LEN: usize = 1 << 16;
        let host = crate::host::host().expect("a D3D12 device");
        assert!(crate::host::relay(), "run with VICTORY_RELAY=2");
        // SAFETY: a private allocation, kept until the process ends.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        let (a, b, token) = (base, base + LEN, base + 2 * LEN);
        // SAFETY: the allocation: two buffers of LEN bytes and a word.
        unsafe {
            for i in 0..LEN {
                (a as *mut u8).add(i).write((i * 3 + 1) as u8);
                (b as *mut u8).add(i).write((i * 5 + 2) as u8);
            }
        }
        crate::shadow::register(a, a as u64, LEN as u64);
        crate::shadow::register(b, b as u64, LEN as u64);
        let copy_a = crate::shadow::find(a as u64, LEN as u64).expect("buffer A's copy");
        let copy_b = crate::shadow::find(b as u64, LEN as u64).expect("buffer B's copy");
        let back = host.read_back_buffer(512).expect("a read-back buffer");
        let (upload, data) = host.upload_buffer(256).expect("an upload buffer");
        let pattern: Vec<u8> = (0..64u32).map(|i| (i * 7 + 3) as u8).collect();
        // SAFETY: the upload buffer holds 256 mapped bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 64) };
        let other = crate::host::fill_kind::OTHER;
        // A fill of 64 bytes of title memory at `src`, copied to the
        // read-back buffer at `at` when the list runs.
        let fill = |l: &crate::host::HostList, src: usize, at: u64| {
            let (buffer, offset, to) = l.scratch_view(64).expect("scratch memory");
            l.defer_fill_as(to, src as *const u8, 64, other);
            // SAFETY: an open list; live buffers, ranges inside them.
            unsafe { l.list.CopyBufferRegion(&back, at, &buffer, offset, 64) };
        };
        let (g, f1, f2) = queue_and_fences();
        let (c, f3, f4) = queue_and_fences();
        let (g1, _) = part(host, |l| {
            copy_a.refresh(l, a as u64, 256);
            // SAFETY: as above. (The card writes the copy, as a shader would.)
            unsafe { l.list.CopyBufferRegion(&copy_a.buffer, copy_a.offset(a as u64 + 64), &upload, 0, 64) };
            copy_a.gpu_writes(l, a as u64 + 64, 64);
        });
        let (g2, _) = part(host, |l| fill(l, b, 0));
        let (g3, _) = part(host, |l| fill(l, a + 64, 64));
        let (g4, _) = part(host, |l| {
            copy_b.refresh(l, b as u64, 256);
            // SAFETY: as above.
            unsafe { l.list.CopyBufferRegion(&back, 128, &copy_b.buffer, copy_b.offset(b as u64 + 16), 64) };
            l.defer_write(token as *mut u8, Written::Value(7), 4);
        });
        let (c1, _) = part(host, |l| fill(l, b + 256, 192));
        let (c2, _) = part(host, |l| {
            copy_a.refresh(l, a as u64, 256);
            // SAFETY: as above.
            unsafe { l.list.CopyBufferRegion(&back, 256, &copy_a.buffer, copy_a.offset(a as u64 + 64), 64) };
        });
        const EQUAL: u32 = 3;
        lock(&queue(g).held).extend([g1, g2, g3, g4, Held::Signal(crate::fence::FenceRef::new(f1), 1)]);
        lock(&queue(c).held).extend([
            Held::Value(crate::list::ValueWait::of(token as u64, EQUAL, 7)),
            c1,
            Held::Wait(crate::fence::FenceRef::new(f1), 1),
            c2,
        ]);
        let sent_before = crate::host::submissions().0;
        BEHIND_ALL.store(true, Ordering::Relaxed);
        GROUP_HERE.set(Some(grouped));
        lock(&WAITING).extend([g as usize, c as usize]);
        for _ in 0..16 {
            release_held_now(false);
            // Everything in the air lands (title memory, the word, the fence).
            crate::host::settle_for_present();
            if lock(&queue(g).held).is_empty() && lock(&queue(c).held).is_empty() {
                break;
            }
        }
        assert!(lock(&queue(g).held).is_empty() && lock(&queue(c).held).is_empty(), "the lines did not run dry");
        crate::host::settle_for_present();
        GROUP_HERE.set(None);
        BEHIND_ALL.store(false, Ordering::Relaxed);
        let sent = crate::host::submissions().0 - sent_before;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: every list ran (the settles waited for them); 320 bytes
        // of the read-back buffer; the allocation above.
        let left = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            let card = std::slice::from_raw_parts(mapped.cast::<u8>(), 320).to_vec();
            back.Unmap(0, None);
            Left {
                title: std::slice::from_raw_parts(base as *const u8, 2 * LEN).to_vec(),
                card,
                token: (token as *const u32).read_volatile(),
                fence: crate::fence::completed(f1),
                sent,
            }
        };
        // What the script must leave, however it was sent.
        assert_eq!(left.title[64..128], pattern[..], "what the card wrote to buffer A is in title memory");
        assert_eq!(left.card[64..128], pattern[..], "G3's fill read what G1's lists wrote (it waited for their landing)");
        assert_eq!(left.card[256..320], pattern[..], "C2 read buffer A's copy after G1");
        assert_eq!(left.card[..64], left.title[LEN..LEN + 64], "G2's fill");
        assert_eq!(left.card[128..192], left.title[LEN + 16..LEN + 80], "G4's read of buffer B's copy");
        assert_eq!(left.card[192..256], left.title[LEN + 256..LEN + 320], "C1's fill");
        assert_eq!((left.token, left.fence), (7, 1));
        crate::shadow::unregister(a, a as u64);
        crate::shadow::unregister(b, b as u64);
        release_all(&[g, c, f1, f2, f3, f4]);
        left
    }

    /// `VICTORY_GROUP_COMMIT`: a script of fake lists on two queues (shader
    /// writes that land in title memory, fills that read them, host copies,
    /// a bottom-of-pipe write another queue waits for, a fence another
    /// queue waits for) leaves the same bytes in title memory and gives the
    /// card the same bytes sent in batches as sent one part a submission;
    /// in batches it takes three host submissions for the six parts (cut
    /// where a fill reads what an earlier part writes when it has run, and
    /// where a queue waits for the fence signalled behind the batch).
    /// (`VICTORY_RELAY=2 cargo test -p victory-gpu --lib group_commit --
    /// --ignored --test-threads=1`)
    #[cfg(windows)]
    #[test]
    #[ignore = "needs a D3D12 adapter and VICTORY_RELAY=2; run alone"]
    fn group_commit_leaves_the_bytes_of_parts_sent_singly() {
        let _alone = lock(&GROUP_TESTS);
        let before = [GROUP_BATCHES, GROUP_PARTS, CUT_READ, CUT_FENCE].map(group_stat);
        let singly = script(false);
        assert_eq!([GROUP_BATCHES, GROUP_PARTS, CUT_READ, CUT_FENCE].map(group_stat), before, "group commit ran without its switch");
        let batched = script(true);
        let after = [GROUP_BATCHES, GROUP_PARTS, CUT_READ, CUT_FENCE].map(group_stat);
        assert_eq!(singly.title, batched.title, "title memory");
        assert_eq!(singly.card, batched.card, "what the card was given");
        assert_eq!((singly.token, singly.fence), (batched.token, batched.fence));
        assert_eq!([after[0] - before[0], after[1] - before[1]], [3, 6], "batches and parts");
        assert_eq!([after[2] - before[2], after[3] - before[3]], [1, 1], "cuts for an in-batch read and for a queue's fence wait");
        assert_eq!((singly.sent, batched.sent), (6, 3), "host submissions");
    }

    /// `VICTORY_GROUP_COMMIT`: a part whose preparing reads what a flight in
    /// the air writes does not make the parts before it wait with it: they
    /// go out as a batch while that flight is still on the card, and it
    /// starts the next batch, which waits for the landing.
    #[cfg(windows)]
    #[test]
    #[ignore = "needs a D3D12 adapter and VICTORY_RELAY=2; run alone"]
    fn group_commit_sends_what_can_go_before_a_part_that_waits_for_a_flight() {
        let _alone = lock(&GROUP_TESTS);
        let host = crate::host::host().expect("a D3D12 device");
        assert!(crate::host::relay(), "run with VICTORY_RELAY=2");
        let (base, copy) = title_buffer(0x6c00_0003);
        let (other_base, _other) = title_buffer(0x6c00_0004);
        let (q, f1, f2) = queue_and_fences();
        let other = crate::host::fill_kind::OTHER;
        let fill = |l: &crate::host::HostList, src: usize| {
            let (_, to) = l.scratch(64).expect("scratch memory");
            l.defer_fill_as(to, src as *const u8, 64, other);
        };
        let queued = |parts: Vec<Held>| {
            lock(&queue(q).held).extend(parts);
            let mut waiting = lock(&WAITING);
            if !waiting.contains(&(q as usize)) {
                waiting.push(q as usize);
            }
        };
        BEHIND_ALL.store(true, Ordering::Relaxed);
        GROUP_HERE.set(Some(true));
        // The card stands still: X, whose shaders are declared to write 64
        // bytes of the buffer, stays in the air.
        let gate = host.test_gate();
        queued(vec![
            part(host, |l| {
                copy.refresh(l, base as u64, 256);
                copy.gpu_writes(l, base as u64 + 64, 64);
            })
            .0,
        ]);
        release_held_now(false);
        let x = crate::host::submissions().0;
        // P1 fills from another buffer; P2 fills from the bytes X writes.
        queued(vec![part(host, |l| fill(l, other_base)).0, part(host, |l| fill(l, base + 64)).0]);
        let before = [GROUP_BATCHES, CUT_AIR].map(group_stat);
        // Another thread lets the card go on after a while, and says how
        // many submissions had been started by then.
        let opener = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(400));
            let started = crate::host::submissions().0;
            // SAFETY: our fence.
            unsafe { gate.Signal(1) }.unwrap();
            started
        });
        // Returns once P2's batch is sent: it waits for X's landing.
        release_held_now(false);
        let started_before_the_card_went_on = opener.join().unwrap();
        assert_eq!(started_before_the_card_went_on, x + 1, "P1 waited with P2 (or P2 did not wait for X)");
        assert_eq!(crate::host::submissions().0, x + 2);
        assert_eq!([GROUP_BATCHES, CUT_AIR].map(group_stat), [before[0] + 2, before[1] + 1]);
        crate::host::settle_for_present();
        GROUP_HERE.set(None);
        BEHIND_ALL.store(false, Ordering::Relaxed);
        crate::shadow::unregister(0x6c00_0003, base as u64);
        crate::shadow::unregister(0x6c00_0004, other_base as u64);
        release_all(&[q, f1, f2]);
    }

    /// `VICTORY_GROUP_COMMIT`: what cuts a batch from the title's side. A
    /// present, a fence the title blocks on and a synchronous submission of
    /// ours each send everything queued before them first, on their own
    /// thread, and what is queued after them is another batch.
    #[cfg(windows)]
    #[test]
    #[ignore = "needs a D3D12 adapter and VICTORY_RELAY=2; run alone"]
    fn group_commit_is_cut_at_a_present_a_blocking_fence_wait_and_a_synchronous_submission() {
        use std::ffi::c_void;
        let _alone = lock(&GROUP_TESTS);
        let host = crate::host::host().expect("a D3D12 device");
        assert!(crate::host::relay(), "run with VICTORY_RELAY=2");
        BEHIND_ALL.store(true, Ordering::Relaxed);
        GROUP_HERE.set(Some(true));
        let (q, f1, f2) = queue_and_fences();
        let queued = |parts: usize, signal: Option<(This, u64)>| {
            let mut held = lock(&queue(q).held);
            for _ in 0..parts {
                held.push(part(host, |_| {}).0);
            }
            if let Some((fence, value)) = signal {
                held.push(Held::Signal(crate::fence::FenceRef::new(fence), value));
            }
            drop(held);
            let mut waiting = lock(&WAITING);
            if !waiting.contains(&(q as usize)) {
                waiting.push(q as usize);
            }
        };
        let stats = || [GROUP_BATCHES, GROUP_PARTS, BEFORE_PRESENT, BEFORE_FENCE, BEFORE_OURS, CUT_DRY].map(group_stat);
        let since = |before: [u64; 6]| {
            let now = stats();
            [0, 1, 2, 3, 4, 5].map(|i| now[i] - before[i])
        };

        // A present: the two parts queued before it are one batch, sent
        // before it returns; the part queued after it is the next.
        let before = stats();
        queued(2, None);
        assert_eq!(present(std::ptr::null(), 0, std::ptr::null(), std::ptr::null()), S_OK);
        assert!(lock(&queue(q).held).is_empty(), "the present returned with parts still queued");
        assert_eq!(since(before), [1, 2, 1, 0, 0, 0], "one batch of two parts, cut by the present");
        queued(1, None);
        release_held_now(false);
        assert_eq!(since(before), [2, 3, 1, 0, 0, 1], "the part after the present is a batch of its own");

        // A fence the title blocks on (no event): the parts and the signal
        // queued before the wait go out first, and the wait returns once
        // they have landed and the fence is set.
        let before = stats();
        queued(2, Some((f1, 3)));
        let set_event = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f1, 0x48);
        assert_eq!(set_event(f1, 3, std::ptr::null_mut()), S_OK);
        assert_eq!(crate::fence::completed(f1), 3);
        assert_eq!(since(before), [1, 2, 0, 1, 0, 0], "one batch of two parts, cut by the blocking wait");

        // A submission of ours that is waited for: behind what was queued.
        let before = stats();
        queued(2, None);
        let ours = host.new_list().unwrap();
        ours.close();
        host.execute(&[&ours]);
        assert!(lock(&queue(q).held).is_empty(), "our submission passed parts still queued");
        assert_eq!(since(before), [1, 2, 0, 0, 1, 0], "one batch of two parts, cut by our submission");

        crate::host::settle_for_present();
        GROUP_HERE.set(None);
        BEHIND_ALL.store(false, Ordering::Relaxed);
        release_all(&[q, f1, f2]);
    }
}
