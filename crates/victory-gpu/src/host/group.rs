//! Group commit (`VICTORY_GROUP_COMMIT=1`, with `VICTORY_ASYNC_SUBMIT=1`;
//! `crate::queue::group`): what the queues ask of the host while they gather
//! the parts waiting in their lines into one host submission.
//!
//! A batch is prepared and sent as one submission of all its lists
//! (`Host::execute_late`): its steps run in the lists' order, each host
//! copy is refreshed once for all of them (`Batch`), it is one
//! `ExecuteCommandLists`, one flight, and so one landing. Everything
//! `Host::submit` does about the flights already in the air it does for
//! the batch as a whole. What it cannot know is what the batch's own parts
//! do to one another: sent one by one, a part whose preparing reads title
//! memory an earlier part writes when it has run (its shaders, a
//! read-back, a bottom-of-pipe value) waits until that part has landed
//! ([`super::late_blocked`]). [`Gathered`] keeps what the parts taken so
//! far will write, and such a part does not join them: the batch is cut
//! before it, and it starts the next one, which `Host::submit` holds back
//! until the flight has landed, as today.

use std::sync::Arc;

use super::{HostList, Late, Step, blocked_by, fills_ahead, join_copy_ranges, join_ranges, line, touches};

/// What the parts gathered for one host submission write once they have
/// run, as [`super::Line`] keeps it of the flights in the air.
#[derive(Default)]
pub(crate) struct Gathered {
    /// Title memory their shaders and read-backs may write, joined.
    written: Vec<(usize, usize)>,
    /// The same by owner (host copy; 0 for read-backs).
    by_copy: Vec<(usize, usize, usize)>,
    /// Their bottom-of-pipe writes' ranges, joined; and the writes, in order.
    immed: Vec<(usize, usize)>,
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The host copies they refresh.
    copies: crate::fasthash::Set<usize>,
    /// With `VICTORY_FILL_AHEAD`: their fills of upload memory that read
    /// ahead of what an earlier one of them writes (source, bytes).
    ahead: Vec<(usize, usize)>,
    /// Parts and host lists gathered; refreshes of a copy by a part after
    /// the first that refreshes it (sent one by one, each its own).
    pub(crate) parts: usize,
    pub(crate) lists: usize,
    pub(crate) saved: u64,
}

impl Gathered {
    pub(crate) fn is_empty(&self) -> bool {
        self.parts == 0
    }

    /// Whether `runs`, a part, must not join the parts gathered: preparing
    /// it reads on the CPU (a buffer copy of the title's, a fill of upload
    /// memory, a texture, a refresh of another host copy over the same
    /// bytes) or overwrites (a buffer copy's destination) title memory they
    /// write once they have run, or runs a step of ours that may read or
    /// write title memory anywhere. Exactly what makes a submission wait
    /// for flights in the air ([`super::late_blocked`]), asked of the parts
    /// that would fly in the same submission.
    pub(crate) fn blocks(&self, runs: &[&HostList]) -> bool {
        blocked_by(&self.written, &self.by_copy, &self.immed, runs, false).is_some()
    }

    /// `runs`, a part, joins: what it writes when it has run is kept for
    /// the parts after it.
    pub(crate) fn add(&mut self, runs: &[&HostList]) {
        // (Before its own writes join: only the parts before it count.)
        self.ahead.extend(fills_ahead(&self.written, &self.by_copy, &self.immed, runs));
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
        let mut part: crate::fasthash::Set<usize> = Default::default();
        for list in runs {
            let deferred = list.deferred();
            for step in &deferred.copies {
                match step {
                    Step::GpuWrite(shadow, address, len) => {
                        ranges.push((Arc::as_ptr(shadow) as usize, *address, *address + *len));
                    }
                    Step::Refresh(shadow, ..) => {
                        part.insert(Arc::as_ptr(shadow) as usize);
                    }
                    _ => {}
                }
            }
            // (As `Host::submit` takes a flight's read-backs.)
            for &(_, at, [row, rows, pitch]) in &deferred.backs {
                if row > 0 && rows > 0 {
                    let span = rows.saturating_sub(1).saturating_mul(pitch).saturating_add(row);
                    ranges.push((0, at, at.saturating_add(span)));
                }
            }
            for &(at, what, bytes) in &deferred.writes {
                self.writes.push((at, what, bytes));
            }
        }
        for copy in part {
            if !self.copies.insert(copy) {
                self.saved += 1;
            }
        }
        join_ranges(&mut self.written, ranges.iter().map(|&(_, a, b)| (a, b)));
        join_copy_ranges(&mut self.by_copy, ranges);
        let immed: Vec<(usize, usize)> = self.writes.iter().map(|&(at, _, n)| (at, at.saturating_add(n))).collect();
        self.immed.clear();
        join_ranges(&mut self.immed, immed);
        self.parts += 1;
        self.lists += runs.len();
    }

    /// `wait`, which memory does not meet now, against what the parts
    /// gathered write: `Some(true)` if a bottom-of-pipe write of theirs
    /// stores a value that meets it (the waiting part then goes to the card
    /// behind that part, in the same submission: the order the wait asks
    /// for, as [`super::late_met`] has it for a flight in the air);
    /// `Some(false)` if they write there in a way that does not say (their
    /// shaders, a timestamp, another width, a value that does not meet it):
    /// the wait stands until they have been sent; `None` if they write
    /// nothing there.
    pub(crate) fn meets(&self, wait: &crate::list::ValueWait) -> Option<bool> {
        let (address, bytes) = wait.place();
        let mut touched = touches(&self.written, address, address + bytes);
        for &(at, what, n) in &self.writes {
            if at >= address + bytes || address >= at + n {
                continue;
            }
            touched = true;
            if let crate::queue::Written::Value(v) = what
                && at == address
                && n >= bytes
                && wait.holds_for(v)
            {
                return Some(true);
            }
        }
        touched.then_some(false)
    }
}

thread_local! {
    /// The fills of the batch this thread is about to send that read ahead
    /// of an earlier part of it ([`Gathered::hand_ahead`]): `Host::submit`
    /// notes them under the batch's own number ([`take_ahead`]).
    static AHEAD: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Gathered {
    /// Tests: the fills kept so far.
    #[cfg(test)]
    pub(crate) fn ahead(&self) -> &[(usize, usize)] {
        &self.ahead
    }

    /// Before the batch is sent, on the thread that sends it.
    pub(crate) fn hand_ahead(&mut self) {
        let ahead = std::mem::take(&mut self.ahead);
        AHEAD.with(|a| *a.borrow_mut() = ahead);
    }
}

/// The fills handed over by [`Gathered::hand_ahead`] on this thread, once.
pub(crate) fn take_ahead() -> Vec<(usize, usize)> {
    AHEAD.with(|a| std::mem::take(&mut *a.borrow_mut()))
}

/// [`super::late_met`] without its settle: whether a bottom-of-pipe write
/// of a flight in the air (or being landed) stores a value that meets
/// `wait`. A queue that has gathered parts asks this way: where flights
/// would have to be settled to know, its line stands for now and is looked
/// at again, the usual way, when the parts have been sent.
pub(crate) fn late_met_peek(wait: &crate::list::ValueWait) -> bool {
    if !super::late() {
        return false;
    }
    let (address, bytes) = wait.place();
    let line = line();
    let writes = line.in_hand.iter().map(|&(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match *item {
        Late::Write(at, what, n) => Some((at, what, n)),
        _ => None,
    }));
    for (at, what, n) in writes {
        if let crate::queue::Written::Value(v) = what
            && at == address
            && n >= bytes
            && wait.holds_for(v)
        {
            return true;
        }
    }
    false
}

/// The most host lists gathered into one batch: half of what may be in the
/// air at once ([`super::late_most`]), so that a batch never fills the air
/// alone (`VICTORY_GROUP_COMMIT_LISTS=<n>` sets it; a part of more lists
/// than that still goes, alone).
pub(crate) fn most() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_GROUP_COMMIT_LISTS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(super::late_most() / 2)
            .max(1)
    })
}
