//! How long textures picked by index wait for room on the card
//! (`VICTORY_ROOM_WAITS=1`; off unless said; a counter, nothing else).
//!
//! The pass over the heap's views makes no new host texture while the card
//! is at its line (`crate::resource::room`): the slot stays due, and the
//! pass ends after 256 such refusals. On the Series X profile a 16 GB card
//! sits at that line from about fifteen seconds into the world for the
//! rest of the run (the "passes that made none for lack of room" count
//! rises by ten a second), so a texture the title asks for after that is
//! made only in a pass that happens to find room, the newest and the
//! oldest writes first. Whether the textures of what is on screen are among
//! the ones kept off the card for a whole run (the player's skin after the
//! cut into the house: black for the run in 5 of 17 Series X runs, with
//! and without `VICTORY_GROUP_SPLIT`) no line said.
//!
//! This keeps, for every slot refused for lack of room, when it was first
//! refused and when last, and says every five seconds how many wait, how
//! many of them the pass has not come back to, and the longest wait; and
//! once for each slot that has waited [`SLOW_MS`] what texture it is.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A wait worth a line of its own, in milliseconds.
pub(crate) const SLOW_MS: u64 = 5_000;
/// Lines of single slots, at most.
const SLOW_LINES: u64 = 96;
/// Slots kept, at most (a pass refuses 256).
const MOST: usize = 2_048;
/// A waiting slot the passes have not refused again for this long was not
/// reached by them: the pass ends at its 256th refusal.
pub(crate) const NOT_REACHED_MS: u64 = 3_000;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ROOM_WAITS").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "waits of textures picked by index for room on the card are counted (VICTORY_ROOM_WAITS=1): a line every five seconds while any waits"
            );
        }
        on
    })
}

/// A texture as the lines name it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct What {
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct Wait {
    /// The queue's write the slot had when it was first refused.
    write: u64,
    first_ms: u64,
    first_frame: u64,
    last_ms: u64,
    what: What,
    said: bool,
}

/// What a five-second line says.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Census {
    pub(crate) waiting: u64,
    pub(crate) waiting_mb: u64,
    pub(crate) not_reached: u64,
    pub(crate) longest_ms: u64,
    pub(crate) longest_slot: u32,
    pub(crate) longest: What,
    pub(crate) made_after_wait: u64,
    pub(crate) longest_made_ms: u64,
    pub(crate) left_unmade: u64,
    pub(crate) refusals: u64,
}

#[derive(Default)]
pub(crate) struct Waits {
    slots: HashMap<u32, Wait>,
    made_after_wait: u64,
    longest_made_ms: u64,
    left_unmade: u64,
    refusals: u64,
    slow_lines: u64,
}

impl Waits {
    /// Slot `index` (the queue's `write`) was refused for lack of room at
    /// `now_ms` in `frame`. Returns its texture and wait if that wait has
    /// just reached [`SLOW_MS`] and a line of its own is still to be had.
    pub(crate) fn refused(&mut self, index: u32, write: u64, now_ms: u64, frame: u64, what: What) -> Option<(What, u64, u64)> {
        self.refusals += 1;
        if let Some(w) = self.slots.get_mut(&index) {
            if w.write != write {
                // Written again: another texture's wait begins.
                self.left_unmade += 1;
                *w = Wait { write, first_ms: now_ms, first_frame: frame, last_ms: now_ms, what, said: false };
                return None;
            }
            w.last_ms = now_ms;
            let waited = now_ms.saturating_sub(w.first_ms);
            if !w.said && waited >= SLOW_MS && self.slow_lines < SLOW_LINES {
                w.said = true;
                self.slow_lines += 1;
                return Some((w.what, waited, frame.saturating_sub(w.first_frame)));
            }
            return None;
        }
        if self.slots.len() < MOST {
            self.slots.insert(index, Wait { write, first_ms: now_ms, first_frame: frame, last_ms: now_ms, what, said: false });
        }
        None
    }

    /// The pass goes on to make slot `index`'s host texture at `now_ms`.
    pub(crate) fn served(&mut self, index: u32, now_ms: u64) {
        if let Some(w) = self.slots.remove(&index) {
            self.made_after_wait += 1;
            self.longest_made_ms = self.longest_made_ms.max(now_ms.saturating_sub(w.first_ms));
        }
    }

    /// The slots that wait, for [`Self::keep`].
    pub(crate) fn waiting(&self) -> Vec<u32> {
        self.slots.keys().copied().collect()
    }

    /// Of the waiting slots only those in `due` (slot and write, as the
    /// queue holds them due now) still wait: the others were written again,
    /// streamed out or let go.
    pub(crate) fn keep(&mut self, due: &[(u32, u64)]) {
        let before = self.slots.len();
        let due: HashMap<u32, u64> = due.iter().copied().collect();
        self.slots.retain(|index, w| due.get(index) == Some(&w.write));
        self.left_unmade += (before - self.slots.len()) as u64;
    }

    pub(crate) fn census(&self, now_ms: u64) -> Census {
        let mut c = Census {
            waiting: self.slots.len() as u64,
            made_after_wait: self.made_after_wait,
            longest_made_ms: self.longest_made_ms,
            left_unmade: self.left_unmade,
            refusals: self.refusals,
            ..Default::default()
        };
        let mut bytes = 0u64;
        for (&index, w) in &self.slots {
            bytes += w.what.bytes;
            if now_ms.saturating_sub(w.last_ms) >= NOT_REACHED_MS {
                c.not_reached += 1;
            }
            let waited = now_ms.saturating_sub(w.first_ms);
            if waited >= c.longest_ms {
                (c.longest_ms, c.longest_slot, c.longest) = (waited, index, w.what);
            }
        }
        c.waiting_mb = bytes >> 20;
        c
    }
}

pub(crate) fn waits() -> std::sync::MutexGuard<'static, Waits> {
    static WAITS: OnceLock<Mutex<Waits>> = OnceLock::new();
    WAITS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether a five-second line is due at `now_ms` (and takes the turn).
pub(crate) fn line_due(now_ms: u64) -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SAID: AtomicU64 = AtomicU64::new(0);
    let said = SAID.load(Ordering::Relaxed);
    now_ms.saturating_sub(said) >= 5_000 && SAID.compare_exchange(said, now_ms, Ordering::Relaxed, Ordering::Relaxed).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKIN: What = What { width: 2048, height: 2048, format: 99, bytes: 5 << 20 };

    /// A slot refused pass after pass waits from its first refusal; its own
    /// line comes once, when the wait reaches five seconds.
    #[test]
    fn a_wait_runs_from_the_first_refusal_and_is_said_once() {
        let mut w = Waits::default();
        assert_eq!(w.refused(7, 1, 1_000, 10, SKIN), None);
        assert_eq!(w.refused(7, 1, 3_000, 20, SKIN), None);
        assert_eq!(w.refused(7, 1, 6_000, 40, SKIN), Some((SKIN, 5_000, 30)));
        assert_eq!(w.refused(7, 1, 9_000, 60, SKIN), None, "said once");
        let c = w.census(9_000);
        assert_eq!((c.waiting, c.waiting_mb, c.longest_ms, c.longest_slot, c.refusals), (1, 5, 8_000, 7, 4));
        assert_eq!(c.longest, SKIN);
    }

    /// A slot the passes stop coming back to still waits, and counts as not
    /// reached; made at last, its wait is the longest made.
    #[test]
    fn a_slot_the_pass_no_longer_reaches_still_waits() {
        let mut w = Waits::default();
        w.refused(7, 1, 1_000, 10, SKIN);
        w.refused(8, 2, 1_000, 10, SKIN);
        w.refused(8, 2, 9_000, 90, SKIN);
        let c = w.census(9_500);
        assert_eq!((c.waiting, c.not_reached), (2, 1));
        w.served(7, 61_000);
        let c = w.census(61_000);
        assert_eq!((c.waiting, c.made_after_wait, c.longest_made_ms), (1, 1, 60_000));
        // A slot never refused that is made is no wait.
        w.served(99, 61_000);
        assert_eq!(w.census(61_000).made_after_wait, 1);
    }

    /// A slot written again, or no longer due, is out: its wait was for a
    /// texture nobody asks for any more.
    #[test]
    fn a_slot_written_again_or_no_longer_due_leaves() {
        let mut w = Waits::default();
        w.refused(7, 1, 1_000, 10, SKIN);
        w.refused(8, 2, 1_000, 10, SKIN);
        w.refused(9, 3, 1_000, 10, SKIN);
        // Slot 7 written again (write 5): its wait begins anew.
        assert_eq!(w.refused(7, 5, 8_000, 80, SKIN), None);
        assert_eq!(w.census(8_000).left_unmade, 1);
        // Slot 8 is due no more; slot 9 is due under another write.
        let mut waiting = w.waiting();
        waiting.sort_unstable();
        assert_eq!(waiting, [7, 8, 9]);
        w.keep(&[(7, 5), (9, 4)]);
        let c = w.census(8_000);
        assert_eq!((c.waiting, c.left_unmade, c.longest_ms, c.longest_slot), (1, 3, 0, 7));
    }
}
