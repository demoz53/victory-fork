//! Which textures a draw needs now (`VICTORY_MATERIAL_TEXTURE_DEMAND=1`; off
//! unless said).
//!
//! Some of the title's pixel shaders take the slots of the textures they
//! sample from a material record: a few dwords of a buffer whose view sits
//! at a fixed place of the draw's first descriptor table, at a byte offset a
//! root constant gives. No table names those textures, so only the queue of
//! textures picked by index brings them to the card (`draw.rs`
//! `heap_textures`), in its own order: a slot written in the frame of a cut
//! drew before the queue's first look at it, and the face it belongs to was
//! black on that shot.
//!
//! Here the record is read when such a draw is bound, and the slots it names
//! that the queue holds due go ahead in the next texture pass, with three
//! other slots between each two of them. The queue still owns every check,
//! upload and retry; a dword that only looks like a slot is one due texture
//! served sooner, no more.
//!
//! Ported from DexNom/VIctory main b8ff22e4c
//! `crates/victory-gpu/src/material_demand.rs` (DexNom, GPL-3.0): the shader
//! list with its record fields, the index decoding, the promotion order and
//! the limits (8 records a frame, 4 frames); the two deferred character
//! shaders are from its PR #49 (Artking175, b8949af9e). Ours differs in where
//! it reads: the record's bytes in title memory when the draw is bound (as
//! the bake's sources are found), not the host copy after the list has run,
//! so the answer is there in the same frame; and a wanted slot the pass's
//! 4,096 do not hold is put in, where theirs only reorders. Its re-enrolment
//! of null slots the queue no longer holds is not ported.

use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

/// Records that may name due slots in one frame.
pub(crate) const RECORDS_A_FRAME: usize = 8;
/// Frames a wanted slot stays wanted without a draw asking again.
pub(crate) const MAX_AGE: u64 = 4;
/// Where the material buffer's view is in the draw's first table, in dwords.
pub(crate) const TABLE_DWORDS: u64 = 688;
/// Records looked at in one frame, each once.
const READS_A_FRAME: usize = 512;
/// Slots kept as wanted.
const MOST_WANTED: usize = 256;
/// The longest record read, in dwords ([`fields`]).
pub(crate) const RECORD_WORDS: usize = 12;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MATERIAL_TEXTURE_DEMAND").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "textures the material record of a draw names go ahead in the queue of textures picked by index \
                 (VICTORY_MATERIAL_TEXTURE_DEMAND=1)"
            );
        }
        on
    })
}

/// The byte offsets, in its material record, of the texture slots the pixel
/// shader with this hash picks.
pub(crate) fn fields(hash: &str) -> Option<&'static [u32]> {
    match hash {
        // The deferred character shaders read heap indices from the same
        // material buffer; the packed variant also samples fields 28 and 32
        // (DexNom/VIctory PR #49).
        "a402e650d0acbc92" => Some(&[0, 4, 8, 12, 16, 20, 24]),
        "b15ba4896d650511" => Some(&[0, 4, 8, 12, 16, 20, 24, 28, 32]),
        "d9fcb8a4425fc59b" | "43fc2af1f70920f1" => Some(&[0, 4, 8]),
        "271a87c54ac648a5" => Some(&[0, 4, 12, 16]),
        "d2e683031b764939" => Some(&[0, 4, 8, 16]),
        "43e24028d4a1faef" | "6590393ed93ff071" | "ee78da6862364ddd" | "c10e7d95d8266d2a"
        | "8a5351822b1a4929" | "09eb0a73df63b87d" | "899ef7a2932ae6c1" | "16053957ff2dff5c"
        | "7c18c77e40e164e4" | "47ec43235084505f" | "247a288b2ba23986" | "6d609ecfb830511f" => {
            Some(&[0, 4, 12, 16, 20])
        }
        "573d1857e903fab5" | "8e9b3db8a3eeba51" | "b8df138593588098" | "b5640b1075483a54"
        | "89afedb5b7720e76" | "085d374c5a795549" | "efc089fe381c43bf" | "0fbc5c91bdab02d2"
        | "072f26dae8824688" | "794198ca2f7c9b67" | "cedd7b38d3be97a1" | "8b901a7fb38c169b" => {
            Some(&[0, 4, 12, 16, 20, 24])
        }
        "3ec62ecabb36f1d2" | "3711fdad456538a1" | "de41d84ca748cc29" | "659fd97066bea4fc"
        | "3f93c9225b740fd9" | "26319e68357cef57" => {
            Some(&[0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
        }
        _ => None,
    }
}

/// The bytes of the record at byte offset `record` of a buffer of `len`
/// bytes that `fields` lie in (offset, length), if they are all in it.
pub(crate) fn byte_range(record: u32, fields: &[u32], len: u64) -> Option<(u64, usize)> {
    let bytes = fields.last()?.checked_add(4)?;
    let end = record.checked_add(bytes)?;
    (record.is_multiple_of(4) && u64::from(end) <= len && bytes as usize <= RECORD_WORDS * 4)
        .then_some((u64::from(record), bytes as usize))
}

/// The slots (below `count`, each once) the `fields` of a record's `words`
/// name.
pub(crate) fn indices(words: &[u32], fields: &[u32], count: u32) -> Vec<u32> {
    let mut slots = Vec::new();
    for field in fields {
        if let Some(&word) = words.get((field / 4) as usize) {
            // The guest shifts the packed index left five bits in a 32-bit register.
            let slot = word.wrapping_shl(5) >> 5;
            if slot < count && !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
}

/// What the draws of the last frames asked for.
#[derive(Default)]
pub(crate) struct Wanted {
    /// The heap the slots are of (its address; a title has one view heap).
    heap: usize,
    /// The frame `read` and `named` are of.
    frame: u64,
    /// Where the records looked at in `frame` are, sorted.
    read: Vec<u64>,
    /// Records of `frame` that named a due slot.
    named: usize,
    /// The wanted slots, each with the frame a draw last asked for it, in
    /// the order they were first asked for.
    slots: Vec<(u32, u64)>,
}

impl Wanted {
    fn turn(&mut self, heap: usize, frame: u64) {
        if self.heap != heap {
            *self = Self { heap, frame, ..Default::default() };
        }
        if self.frame != frame {
            self.frame = frame;
            self.read.clear();
            self.named = 0;
        }
    }

    /// Whether the record at `address` is to be read for a draw of `frame`:
    /// each once a frame, and none once [`RECORDS_A_FRAME`] of them named a
    /// due slot.
    pub(crate) fn asks(&mut self, heap: usize, frame: u64, address: u64) -> bool {
        self.turn(heap, frame);
        if self.named >= RECORDS_A_FRAME || self.read.len() >= READS_A_FRAME {
            return false;
        }
        match self.read.binary_search(&address) {
            Ok(_) => false,
            Err(at) => {
                self.read.insert(at, address);
                true
            }
        }
    }

    /// A record read in `frame` names the slots `due`, which the queue
    /// holds due. Returns how many of them were not wanted already.
    pub(crate) fn note(&mut self, heap: usize, frame: u64, due: &[u32]) -> usize {
        self.turn(heap, frame);
        if due.is_empty() {
            return 0;
        }
        self.named += 1;
        let mut new = 0;
        for &slot in due {
            if let Some(s) = self.slots.iter_mut().find(|s| s.0 == slot) {
                s.1 = frame;
            } else if self.slots.len() < MOST_WANTED {
                self.slots.push((slot, frame));
                new += 1;
            }
        }
        new
    }

    /// The slots of `heap` asked for not more than [`MAX_AGE`] frames
    /// before `frame`, the oldest asked first; older ones are forgotten.
    pub(crate) fn wanted(&mut self, heap: usize, frame: u64) -> Vec<u32> {
        if self.heap != heap {
            return Vec::new();
        }
        self.slots.retain(|s| frame.abs_diff(s.1) <= MAX_AGE);
        self.slots.iter().map(|s| s.0).collect()
    }
}

/// What draws asked for, for the next texture pass.
pub(crate) fn mailbox() -> std::sync::MutexGuard<'static, Wanted> {
    static WANTED: OnceLock<Mutex<Wanted>> = OnceLock::new();
    WANTED.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the start: records read, records that named a due slot, slots that
/// went ahead in a pass, and passes in which one did (a line every five
/// seconds says them, `host.rs`).
pub(crate) static SEEN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Puts the `wanted` slots (each with its write, as the queue hands them
/// out) ahead in `due`, the list a texture pass serves in order: at least
/// three other slots stay between two of them, and who goes first turns
/// with `frame`. One that `due` holds sooner already stays where it is; one
/// it does not hold is put in. Returns how many moved or came in.
pub(crate) fn promote(due: &mut Vec<(u32, u64)>, wanted: &[(u32, u64)], frame: u64) -> usize {
    let mut moved = 0;
    let mut at = (frame % 4) as usize;
    for item in wanted {
        let here = at.min(due.len());
        match due.iter().position(|v| v.0 == item.0) {
            Some(found) if found < here => continue,
            Some(found) => {
                due[here..=found].rotate_right(1);
                due[here] = *item;
                moved += usize::from(found > here);
            }
            None => {
                due.insert(here, *item);
                moved += 1;
            }
        }
        at = here + 4;
    }
    if moved > 0 {
        SEEN[2].fetch_add(moved as u64, Ordering::Relaxed);
        SEEN[3].fetch_add(1, Ordering::Relaxed);
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (DexNom/VIctory b8ff22e4c.)
    #[test]
    fn sampled_fields_use_guest_shift_and_only_verified_words() {
        let fields = fields("43e24028d4a1faef").unwrap();
        let w = [0xf8000000, 17, 2, 0xf8000011, 99, 5];
        assert_eq!(indices(&w, fields, 20), [0, 17, 5]);
        assert_eq!(byte_range(12, fields, 36), Some((12, 24)));
        assert_eq!(byte_range(12, fields, 35), None);
        assert_eq!(byte_range(u32::MAX - 3, fields, u64::MAX), None);
        assert_eq!(byte_range(3, fields, 36), None);
    }

    /// (DexNom/VIctory PR #49.)
    #[test]
    fn character_material_sampling_covers_both_record_layouts() {
        let words = [11, 12, 13, 14, 15, 16, 17, 0xf8000012, 19];
        let regular = fields("a402e650d0acbc92").expect("deferred character material");
        let packed = fields("b15ba4896d650511").expect("packed deferred character material");

        assert_eq!(indices(&words, regular, 20), [11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(indices(&words, packed, 20), [11, 12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(byte_range(16, regular, 44), Some((16, 28)));
        assert_eq!(byte_range(16, packed, 44), None);
        assert_eq!(byte_range(16, packed, 52), Some((16, 36)));
    }

    /// Every listed record fits the buffer a draw reads it into.
    #[test]
    fn the_longest_record_fits_the_read() {
        let longest = fields("3ec62ecabb36f1d2").unwrap();
        assert_eq!(byte_range(0, longest, 4096), Some((0, RECORD_WORDS * 4)));
    }

    /// Wanted slots go ahead with three others between them; one the list
    /// holds sooner stays, one it does not hold comes in, and the write is
    /// the wanted one's.
    #[test]
    fn wanted_slots_go_ahead_with_three_others_between() {
        let slots = |due: &[(u32, u64)]| due.iter().map(|v| v.0).collect::<Vec<_>>();
        let list = || (0..10u32).map(|n| (n, u64::from(n) + 1)).collect::<Vec<_>>();
        // (DexNom's case: only what is due moves; the caller hands no other.)
        let mut due = list();
        assert_eq!(promote(&mut due, &[(8, 9), (9, 10)], 0), 2);
        assert_eq!(slots(&due), [8, 0, 1, 2, 9, 3, 4, 5, 6, 7]);
        // Who goes first turns with the frame.
        let mut due = list();
        assert_eq!(promote(&mut due, &[(8, 9), (9, 10)], 1), 2);
        assert_eq!(slots(&due), [0, 8, 1, 2, 3, 9, 4, 5, 6, 7]);
        // Slot 40 is not among the pass's: it comes in. Slot 0 is first
        // already: it stays and takes no turn. The write is the wanted one's.
        let mut due = list();
        assert_eq!(promote(&mut due, &[(40, 41), (0, 1), (9, 77)], 2), 2);
        assert_eq!(slots(&due), [0, 1, 40, 2, 3, 4, 9, 5, 6, 7, 8]);
        assert_eq!(due[6], (9, 77));
        // A short list: the wanted ones go to its end, in their order.
        let mut due = vec![(1, 1)];
        assert_eq!(promote(&mut due, &[(5, 5), (6, 6)], 3), 2);
        assert_eq!(slots(&due), [1, 5, 6]);
        // Nothing wanted: nothing moves.
        let mut due = list();
        assert_eq!(promote(&mut due, &[], 0), 0);
        assert_eq!(due, list());
    }

    /// A record is read once a frame, at most eight of a frame name due
    /// slots, and a slot nothing asks for again is forgotten after four
    /// frames.
    #[test]
    fn records_are_read_once_a_frame_and_wanted_slots_age() {
        let mut w = Wanted::default();
        assert!(w.asks(1, 10, 0x1000));
        assert!(!w.asks(1, 10, 0x1000), "the same record twice in one frame");
        assert_eq!(w.note(1, 10, &[5, 6]), 2);
        assert_eq!(w.note(1, 10, &[]), 0);
        // Records that name nothing due do not use up the frame's eight.
        for i in 0..100u64 {
            assert!(w.asks(1, 10, 0x2000 + i * 64));
        }
        for i in 1..RECORDS_A_FRAME as u64 {
            assert!(w.asks(1, 10, 0x9000 + i * 64));
            w.note(1, 10, &[100 + i as u32]);
        }
        assert!(!w.asks(1, 10, 0xf000), "eight records of this frame named due slots");
        assert_eq!(w.wanted(1, 10), [5, 6, 101, 102, 103, 104, 105, 106, 107]);
        // The next frame: read again; slot 6 is asked for again, 5 is not.
        assert!(w.asks(1, 11, 0x1000));
        assert_eq!(w.note(1, 11, &[6]), 0);
        assert_eq!(w.wanted(1, 14).len(), 9);
        assert_eq!(w.wanted(1, 15), [6]);
        assert_eq!(w.wanted(1, 16), [] as [u32; 0]);
        // Another heap's slots are not this heap's.
        w.note(1, 20, &[9]);
        assert_eq!(w.wanted(2, 20), [] as [u32; 0]);
        assert!(w.asks(2, 20, 0x1000));
        assert_eq!(w.wanted(1, 20), [] as [u32; 0]);
    }
}
