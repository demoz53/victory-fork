//! The pass that uploads textures picked by index, on a thread of its own
//! (`VICTORY_HEAP_WORKER=1`; off by default; needs flights sent ahead,
//! [`crate::host::late`], which `VICTORY_RELAY` or `VICTORY_LATE_WAIT` give).
//!
//! Hand-ported from fxceless/victory-fork d4a528ba0 (fork pull request #1007,
//! its worker commit 233a65f4; measured there in f6a09d3ba) onto this tree's
//! pass, queue and slot bookkeeping.
//!
//! [`crate::draw::State::heap_textures`] runs on whichever thread of the
//! title records the frame's first shader that picks textures by index: it
//! asks what of each due texture is mapped, makes the host texture, makes a
//! staging buffer and records the copy, and the threads waiting for that
//! list wait with it. With the switch the recording thread only wakes the
//! worker ([`kick`]), and the pass runs here.
//!
//! What keeps the picture as it was:
//! - the uploads are recorded into host lists of ours ([`Batch`]), which the
//!   next submission of the title's runs ahead of its own lists
//!   ([`take_ready`]): texture states are settled by submission order, as
//!   between any two lists;
//! - a slot's host view is made when its upload's list is handed to a
//!   submission ([`Publish`]), not when the upload is recorded: until then
//!   shaders see what the slot showed, never a texture without its
//!   contents. The hand-over and the making of the slots happen under
//!   [`order`], which the submission holds until its lists are in the
//!   card's queue: no title list can reach the queue between a slot made
//!   and the upload it shows. A slot the title wrote again meanwhile is
//!   left to that write's own turn in the queue of textures picked by index
//!   (its record's key changed);
//! - what a title texture the pass works on is held alive
//!   ([`crate::resource::hold`]) until its list is in the queue, and so is
//!   the title's heap;
//! - the uploads before the title's decoration bake, which must be there
//!   for that draw, still run on the recording thread; the worker makes way
//!   for them ([`urgent`]), handing over what it recorded so far first.
// shortcut: what a submission shows for a texture is up to a frame later
// than when the recording thread did it all; with a backlog it is earlier,
// since the pass has the whole frame instead of a slice of it.
// shortcut: the detiling of what the worker uploads still runs at
// submission, on the submitting thread, as before (fxceless's
// `Staging::refresh_ahead`, detiling on the worker, is not ported).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

/// `VICTORY_HEAP_WORKER=1` was asked for.
fn wanted() -> bool {
    static WANTED: OnceLock<bool> = OnceLock::new();
    *WANTED.get_or_init(|| std::env::var_os("VICTORY_HEAP_WORKER").is_some_and(|v| v == "1"))
}

/// Whether the pass runs on its own thread. Asked for without flights sent
/// ahead, it is refused (said once): a submission that is waited for could
/// not run the worker's lists ahead of its own.
#[cfg(windows)]
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !wanted() {
            return false;
        }
        if !crate::host::late() {
            tracing::warn!(
                "VICTORY_HEAP_WORKER=1 refused: it needs flights sent ahead (VICTORY_RELAY or VICTORY_LATE_WAIT); \
                 textures picked by index are uploaded by the recording thread, as without it"
            );
            return false;
        }
        true
    })
}

/// For the frame line, since the last one: nanoseconds the worker was at
/// work, its passes, lists given to submissions, slots shown, slots the
/// title had written again before they could be shown, nanoseconds the
/// recording threads spent on the worker (waking it, and waiting for it
/// before a bake draw), and the passes a recording thread took over.
pub(crate) static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// One upload list holds at most this much texture memory, or this many
/// milliseconds of work, so that its textures show soon.
const LIST_BYTES: u64 = 48 << 20;
const LIST_MS: u128 = 40;

/// Things handed from one thread to another, each taken exactly once, in
/// the order put.
pub(crate) struct Shelf<T>(Mutex<Vec<T>>);

impl<T> Shelf<T> {
    pub(crate) const fn new() -> Self {
        Shelf(Mutex::new(Vec::new()))
    }

    pub(crate) fn put(&self, item: T) {
        lock(&self.0).push(item);
    }

    /// Everything put so far and not taken yet.
    pub(crate) fn take(&self) -> Vec<T> {
        std::mem::take(&mut *lock(&self.0))
    }
}

/// The latest job for a worker, and what wakes it: a job posted while the
/// worker is busy replaces one not started yet (passes coalesce), and is
/// run once the worker is free (none is lost).
pub(crate) struct Mailbox<T> {
    next: Mutex<Option<T>>,
    wake: Condvar,
}

impl<T> Mailbox<T> {
    pub(crate) const fn new() -> Self {
        Mailbox { next: Mutex::new(None), wake: Condvar::new() }
    }

    pub(crate) fn post(&self, job: T) {
        *lock(&self.next) = Some(job);
        self.wake.notify_one();
    }

    /// The next job, waited for.
    pub(crate) fn wait(&self) -> T {
        let mut next = lock(&self.next);
        loop {
            match next.take() {
                Some(job) => return job,
                None => next = self.wake.wait(next).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    /// The next job if one is posted.
    #[cfg(test)]
    fn try_take(&self) -> Option<T> {
        lock(&self.next).take()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(windows)]
pub(crate) use worker::*;

#[cfg(windows)]
mod worker {
    use super::*;
    use crate::descriptor::Heap;
    use crate::host::HostList;
    use crate::object::This;

    /// A view to give a host slot once its texture's upload is in the
    /// queue: the slot, its record's handle, and the record's key as it was
    /// when the upload was recorded.
    pub(crate) struct Publish {
        index: u32,
        handle: u64,
        key: u64,
    }

    /// Holds a reference to a title heap.
    struct HeldHeap(This);
    // SAFETY: a reference counted title object, of which only the count and
    // the parts behind locks are touched.
    unsafe impl Send for HeldHeap {}

    impl HeldHeap {
        fn new(heap: &Heap) -> Self {
            let this = std::ptr::from_ref(heap).cast();
            crate::object::add_ref(this);
            HeldHeap(this)
        }

        fn heap(&self) -> &Heap {
            crate::descriptor::heap(self.0)
        }
    }

    impl Drop for HeldHeap {
        fn drop(&mut self) {
            crate::object::release_boxed::<Heap>(self.0);
        }
    }

    /// A closed upload list, what is published when it is handed to a
    /// submission, and what stays alive until then.
    struct Ready {
        list: HostList,
        publish: Vec<Publish>,
        held: Vec<crate::resource::Held>,
        heap: HeldHeap,
    }
    // SAFETY: D3D12 objects are free-threaded; the list is closed and
    // touched by nobody until it is taken out; the held objects are
    // reference counted and only released.
    unsafe impl Send for Ready {}

    static READY: Shelf<Ready> = Shelf::new();

    /// What a pass on the worker's thread records into: one upload list
    /// after the other, made when first needed.
    pub(crate) struct Batch {
        heap: This,
        list: Option<HostList>,
        publish: Vec<Publish>,
        held: Vec<crate::resource::Held>,
        bytes: u64,
        began: std::time::Instant,
    }

    impl Batch {
        fn new(heap: &Heap) -> Self {
            Batch {
                heap: std::ptr::from_ref(heap).cast(),
                list: None,
                publish: Vec::new(),
                held: Vec::new(),
                bytes: 0,
                began: std::time::Instant::now(),
            }
        }

        /// The open upload list.
        pub(crate) fn list(&mut self) -> Option<&HostList> {
            if self.list.is_none() {
                self.list = crate::list::fresh_list();
                self.began = std::time::Instant::now();
            }
            self.list.as_ref()
        }

        /// Slot `index` (its record at `handle`, then with `key`) is made
        /// when the open list is handed to a submission. False without a
        /// list.
        pub(crate) fn publish(&mut self, index: u32, handle: u64, key: u64) -> bool {
            if self.list().is_none() {
                return false;
            }
            self.publish.push(Publish { index, handle, key });
            true
        }

        /// `held` stays alive until the open list is in the queue.
        pub(crate) fn hold(&mut self, held: crate::resource::Held) {
            self.held.push(held);
        }

        /// `bytes` more of texture memory in the open list, which is closed
        /// and handed over when it holds enough.
        pub(crate) fn uploaded(&mut self, bytes: u64) {
            self.bytes += bytes;
            if self.bytes >= LIST_BYTES || self.began.elapsed().as_millis() >= LIST_MS {
                self.flush();
            }
        }

        /// Closes the open list and hands it to the next submission.
        fn flush(&mut self) {
            self.bytes = 0;
            let publish = std::mem::take(&mut self.publish);
            let held = std::mem::take(&mut self.held);
            let Some(list) = self.list.take() else {
                return;
            };
            list.close();
            let heap = HeldHeap::new(crate::descriptor::heap(self.heap));
            READY.put(Ready { list, publish, held, heap });
        }
    }

    impl Drop for Batch {
        fn drop(&mut self) {
            self.flush();
        }
    }

    /// Held by a submission from [`take_ready`] until its lists are in the
    /// card's queue (`None` without the worker).
    pub(crate) fn order() -> Option<MutexGuard<'static, ()>> {
        static ORDER: Mutex<()> = Mutex::new(());
        JOB.get().is_some().then(|| lock(&ORDER))
    }

    /// What stays alive until the lists [`take_ready`] gave are in the
    /// queue: dropped after the submission.
    pub(crate) struct Sent(#[allow(dead_code)] Vec<(Vec<crate::resource::Held>, HeldHeap)>);

    /// The upload lists ready now, for the submission about to be made,
    /// which runs them ahead of its own lists; their host slots are made
    /// now. Called by the thread that submits, holding [`order`].
    pub(crate) fn take_ready() -> (Vec<HostList>, Sent) {
        // (Asked without a lock by every submission of a run without it.)
        if JOB.get().is_none() {
            return (Vec::new(), Sent(Vec::new()));
        }
        let ready = READY.take();
        STATS[2].fetch_add(ready.len() as u64, Relaxed);
        let mut lists = Vec::with_capacity(ready.len());
        let mut kept = Vec::with_capacity(ready.len());
        for Ready { list, publish, held, heap } in ready {
            if let Some(host) = heap.heap().host() {
                for p in &publish {
                    let shown = host.publish_ahead(p.index, p.handle, p.key);
                    STATS[if shown { 3 } else { 4 }].fetch_add(1, Relaxed);
                }
            }
            lists.push(list);
            kept.push((held, heap));
        }
        (lists, Sent(kept))
    }

    static JOB: OnceLock<Mailbox<HeldHeap>> = OnceLock::new();
    /// Held by whoever runs the pass.
    static PASS: Mutex<()> = Mutex::new(());
    /// A recording thread waits for [`PASS`].
    static URGENT: AtomicBool = AtomicBool::new(false);

    /// A pass over `heap` is due. False if there is no worker: the caller
    /// runs the pass itself.
    pub(crate) fn kick(heap: &Heap) -> bool {
        if !on() {
            return false;
        }
        let began = std::time::Instant::now();
        static STARTED: OnceLock<bool> = OnceLock::new();
        let started = *STARTED.get_or_init(|| {
            let _ = JOB.set(Mailbox::new());
            match std::thread::Builder::new().name("victory-heap".into()).spawn(worker) {
                Ok(_) => {
                    tracing::info!("textures picked by index are uploaded by a thread of their own (VICTORY_HEAP_WORKER=1)");
                    true
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no thread for textures picked by index: the recording thread uploads them");
                    false
                }
            }
        });
        let Some(job) = JOB.get().filter(|_| started) else {
            return false;
        };
        job.post(HeldHeap::new(heap));
        STATS[5].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
        true
    }

    /// Takes the pass from the worker for the caller, which cannot wait for
    /// a later submission: what the worker has recorded so far is handed
    /// over first, so the caller's list runs behind it. `None` without the
    /// worker.
    pub(crate) fn urgent() -> Option<MutexGuard<'static, ()>> {
        JOB.get()?;
        STATS[6].fetch_add(1, Relaxed);
        if let Ok(pass) = PASS.try_lock() {
            return Some(pass);
        }
        let began = std::time::Instant::now();
        URGENT.store(true, Relaxed);
        let pass = lock(&PASS);
        URGENT.store(false, Relaxed);
        STATS[5].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
        Some(pass)
    }

    /// Whether the worker is to stop where it is ([`urgent`]).
    pub(crate) fn make_way() -> bool {
        URGENT.load(Relaxed)
    }

    fn worker() {
        let Some(job) = JOB.get() else {
            return;
        };
        loop {
            let heap = job.wait();
            let started = std::time::Instant::now();
            {
                let _pass = lock(&PASS);
                let heap = heap.heap();
                if let Some(host) = heap.host() {
                    // Dropped, and so handed over, before the pass is let go.
                    let mut batch = Batch::new(heap);
                    crate::draw::State::heap_textures(crate::draw::PassInto::Ahead(&mut batch), heap, host);
                }
            }
            drop(heap);
            STATS[0].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
            STATS[1].fetch_add(1, Relaxed);
            // Whoever wants the pass now has it first.
            while make_way() {
                std::thread::yield_now();
            }
        }
    }

    /// The frame line's part (empty without the worker): per frame over
    /// `frames`, since the last line.
    pub(crate) fn frame_line(frames: f64) -> String {
        if JOB.get().is_none() {
            return String::new();
        }
        let [busy, passes, lists, shown, again, waited, urgent] =
            std::array::from_fn::<f64, 7, _>(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
        format!(
            " (texture pass on its own thread, VICTORY_HEAP_WORKER: {:.1} ms on the worker in {passes:.1} passes, \
             recording threads waited {:.2} ms for it and took {urgent:.1} passes over; {lists:.1} lists handed over, \
             {shown:.0} slots shown, {again:.1} written again before)",
            busy / 1e6,
            waited / 1e6
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Upload lists put by one thread while submissions take them on
    /// others: each is taken once, none is lost, and one thread's are taken
    /// in the order it put them.
    #[test]
    fn every_list_handed_over_is_taken_once() {
        let shelf: Arc<Shelf<(u32, u32)>> = Arc::new(Shelf::new());
        let done = Arc::new(AtomicBool::new(false));
        const PUT: u32 = 20_000;
        let putters: Vec<_> = (0..2)
            .map(|who| {
                let shelf = shelf.clone();
                std::thread::spawn(move || {
                    for n in 0..PUT {
                        shelf.put((who, n));
                    }
                })
            })
            .collect();
        let takers: Vec<_> = (0..3)
            .map(|_| {
                let (shelf, done) = (shelf.clone(), done.clone());
                std::thread::spawn(move || {
                    let mut got = Vec::new();
                    loop {
                        let finished = done.load(Relaxed);
                        let taken = shelf.take();
                        // In order within one take.
                        for who in 0..2 {
                            let mine: Vec<u32> = taken.iter().filter(|t| t.0 == who).map(|t| t.1).collect();
                            assert!(mine.windows(2).all(|w| w[0] < w[1]));
                        }
                        got.extend(taken);
                        if finished {
                            return got;
                        }
                        std::thread::yield_now();
                    }
                })
            })
            .collect();
        for p in putters {
            p.join().unwrap();
        }
        done.store(true, Relaxed);
        let mut all: Vec<(u32, u32)> = takers.into_iter().flat_map(|t| t.join().unwrap()).collect();
        all.extend(shelf.take());
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(n, all.len(), "a list was taken twice");
        assert_eq!(all.len(), 2 * PUT as usize, "a list was lost");
    }

    /// A pass asked for while the worker runs one is run after it (what
    /// the title wrote meanwhile is looked at), and asks that pile up while
    /// it is busy make one pass, not many.
    #[test]
    fn a_pass_asked_for_meanwhile_runs_after() {
        let mailbox: Arc<Mailbox<u32>> = Arc::new(Mailbox::new());
        let (began, finish) = (Arc::new(std::sync::Barrier::new(2)), Arc::new(std::sync::Barrier::new(2)));
        let worker = {
            let (mailbox, began, finish) = (mailbox.clone(), began.clone(), finish.clone());
            std::thread::spawn(move || {
                let first = mailbox.wait();
                began.wait();
                // At work: the recording thread asks three more times.
                finish.wait();
                let second = mailbox.wait();
                (first, second)
            })
        };
        mailbox.post(1);
        began.wait();
        for n in 2..=4 {
            mailbox.post(n);
        }
        finish.wait();
        let (first, second) = worker.join().unwrap();
        assert_eq!((first, second), (1, 4));
        assert!(mailbox.try_take().is_none(), "one pass for the asks that piled up");
    }
}
