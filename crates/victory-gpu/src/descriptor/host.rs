//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed. A view heap the
//! host caps below the title's slots keeps that order only for an identity
//! window of them; past it each title slot that holds a view is made in a
//! pool slot, which translated shaders look up in a heap map ([`MapLayout`]).

use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::slot_map::SlotMap;
use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's at their own index (in a view heap with
    /// a heap map, the identity window); view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots from `dynamic_first`: [`DYNAMIC`] in view heaps, none
    /// in sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there, 2^16 and 2^17 both ran out in this title,
    /// and more for a title heap it does not hold whole:
    /// [`capped_dynamic_slots`].)
    dynamic: u32,
    /// The first dynamic slot: `count`, or in a view heap with a heap map
    /// the one after the pool's last ([`MapLayout`]: the one after the
    /// map's slot unless the heap has more than [`DYNAMIC`] of them).
    dynamic_first: u32,
    /// The title's own slot count (`count` may be fewer).
    title_count: u32,
    sampler: bool,
    /// The record (view index and generation) each host slot of the
    /// title's slots was last made from, and the heap map's bookkeeping.
    made: Mutex<Made>,
    /// The heap map of a view heap capped below the title's slots
    /// ([`MapLayout`]); `None` for the others.
    map: Option<HeapMap>,
    /// The title heap's first descriptor, as the first slot mirrored past
    /// the window gives it (0 until then): for [`HostHeap::reclaim`].
    title_start: AtomicU64,
    /// When [`HostHeap::reclaim`] last ran (`crate::host::now_ms`;
    /// `u64::MAX`: never).
    reclaimed: AtomicU64,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `SLOT_STRIDES_COUNT` dwords, its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    buffers: Mutex<std::collections::BTreeSet<u32>>,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Mutex<std::collections::HashSet<u32>>,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// Title slots that go on showing a texture past their view
    /// ([`stale_keep`]). Locked after `made`.
    kept_past: Mutex<KeptPast>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: Mutex<Spans>,
}

/// Diagnostics, since the last line (`crate::copystat::log`): why
/// [`HostHeap::dynamic`] refused a table. 0: fewer slots were free than it
/// asked (the ring is full: nothing to search); 1: enough were free but a
/// search for a table no longer had failed since the last release, so none
/// was made (`too_long`); 2: a search found no run of free slots that long
/// (the free slots lie apart). 3: tables that got a run lying across the
/// place the search starts from, which the search missed before it went on
/// past that place.
#[cfg(feature = "diagnostics")]
pub(crate) static DYNAMIC_WHY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Diagnostics: the fewest free dynamic slots [`HostHeap::dynamic`] saw
/// since the last line (`u32::MAX`: it was not called), and the dynamic
/// slots of the heap it was last asked of.
#[cfg(feature = "diagnostics")]
pub(crate) static DYNAMIC_LOW: [std::sync::atomic::AtomicU32; 2] =
    [std::sync::atomic::AtomicU32::new(u32::MAX), std::sync::atomic::AtomicU32::new(0)];

/// `VICTORY_DYNAMIC_ACROSS` (on by default; `0`: off): the search for free
/// dynamic slots also finds a run lying across the place it starts from
/// ([`free_run`]).
fn across() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DYNAMIC_ACROSS").is_none_or(|v| v != "0"))
}

/// A run of `n` free slots among `taken` (`n` at least 1), searched from
/// `cursor` on and then from the start: its first slot, and whether it
/// lies across `cursor`. A run cannot wrap past the end.
///
/// `across`: the second part goes on past `cursor` by `n - 1` slots, so a
/// free run that lies across it is found (the first part sees only its
/// end, the second, stopping at `cursor`, only its start). Without that
/// such a table was refused with room for it, and `too_long` then refused
/// every table as long without a search until the next release
/// (fxceless/victory-fork 4d0c63455, by hand from upstream c1c32819e9;
/// GPL-3.0).
fn free_run(taken: &[bool], cursor: usize, n: usize, across: bool) -> Option<(usize, bool)> {
    let end = taken.len();
    let second = if across { (cursor + n - 1).min(end) } else { cursor };
    let mut run = 0;
    let mut last = usize::MAX;
    for at in (cursor..end).chain(0..second) {
        if at != last.wrapping_add(1) {
            run = 0;
        }
        last = at;
        run = if taken[at] { 0 } else { run + 1 };
        if run == n {
            let first = at + 1 - n;
            return Some((first, first < cursor && cursor < first + n));
        }
    }
    None
}

/// Host-slot state behind one lock: every write of a host slot for a title
/// slot takes it, so a pool slot is taken, made and published (or given
/// back) at once.
struct Made {
    /// The record (view index and generation) each host slot of the title's
    /// (`count`, and in a heap with a heap map its pool's) was last made
    /// from; 0 for none.
    keys: Vec<u64>,
    /// Which pool slot each title slot past the identity window holds (a
    /// view heap with a heap map).
    map: Option<SlotMap>,
}

impl Made {
    /// The host slot title slot `title` is made in, if it has one (none is
    /// taken here): its own below the window, its pool slot past it.
    fn host(&self, title: u32) -> Option<u32> {
        match &self.map {
            Some(map) => map.host(title),
            None => ((title as usize) < self.keys.len()).then_some(title),
        }
    }
}

/// The heap map of a view heap with a [`MapLayout`]: an upload buffer of one
/// entry per title slot from the identity window on ([`slot_map::entry`]:
/// the bitwise not of its host slot, 0 for none), persistently mapped, and
/// read by translated shaders through a raw view at the layout's map slot
/// (`victory_shader::emit::set_view_heap_map`).
///
/// [`slot_map::entry`]: super::slot_map::entry
struct HeapMap {
    buffer: ID3D12Resource,
    /// Where the entries are mapped.
    at: usize,
}

/// The title's view slots in a view heap whose host heap is capped below
/// them, with a heap map: title slots `0..window` at their own index, as
/// tables made in place need; `pool` slots for the title slots past the
/// window that hold views (looked up in the map by translated shaders);
/// the map's raw view at `slot`; then the fork's [`DYNAMIC`] dynamic slots
/// and its null guard slot last, where they were without a map. A heap with
/// more dynamic slots (`dynamic`, [`capped_dynamic_slots`]) takes them from
/// the pool's end: they then start below the map's slot, which stays where
/// the translations read it and is never handed out.
///
/// On NVIDIA a shader-visible view heap holds at most 1,000,000 views
/// ([`PORTABLE_VIEWS`]); the title's has 1,245,184 slots, some
/// 650,000-700,000 of them holding views, about 335,000 of those past the
/// 737,855 title slots a capped heap kept in place (their bindless reads
/// gave the null slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MapLayout {
    window: u32,
    pool: u32,
    slot: u32,
    /// The dynamic slots, up to the guard: [`DYNAMIC`], or more, and then
    /// the map's slot is one of them.
    dynamic: u32,
}

impl MapLayout {
    /// The layout for `title_count` title slots, an identity window of up
    /// to `window` slots, the map at `slot` and `dynamic` dynamic slots (no
    /// fewer than [`DYNAMIC`], and none of the window's).
    fn new(title_count: u32, window: u32, slot: u32, dynamic: u32) -> Self {
        let window = window.min(slot).min(title_count);
        let guard = slot + 1 + DYNAMIC;
        let dynamic = dynamic.clamp(DYNAMIC, guard - window);
        // The pool ends where the dynamic slots start, at the map's slot
        // at the latest.
        let pool_end = (guard - dynamic).min(slot);
        MapLayout {
            window,
            pool: (pool_end - window).min(title_count - window),
            slot,
            dynamic,
        }
    }

    /// The first dynamic slot.
    fn dynamic_first(self) -> u32 {
        self.guard() - self.dynamic
    }

    /// The map's slot among the dynamic slots (counted from the first), if
    /// it is one of them: it is never handed out.
    fn slot_in_dynamic(self) -> Option<u32> {
        self.slot.checked_sub(self.dynamic_first())
    }

    /// The null guard slot, the heap's last: the limit translated shaders
    /// clamp view indices to (fixed by the map's slot, however many dynamic
    /// slots there are).
    fn guard(self) -> u32 {
        self.slot + 1 + DYNAMIC
    }

    /// The heap's slots.
    fn size(self) -> u32 {
        self.guard() + 1
    }
}

/// Bytes per title view descriptor.
const TITLE_SLOT_BYTES: u64 = 32;

/// The identity window of a heap map without `VICTORY_HEAP_WINDOW`: the
/// title's tables and its first two 64Ki runs of views lie below it, its
/// first empty run (110,940..327,680) spans it.
const WINDOW: u32 = 1 << 17;

/// The heap map's slot in a capped view heap: right before the dynamic
/// slots, so that those and the guard slot stay where they were
/// (`PORTABLE_VIEWS - DYNAMIC - 1` on, the guard at `PORTABLE_VIEWS - 1`).
/// No view index reaches it: one below the window is its own, one past it
/// is a pool slot or the guard, and tables made in place lie below the
/// window.
const MAP_SLOT: u32 = PORTABLE_VIEWS - DYNAMIC - 2;

/// The identity window: [`WINDOW`], or `VICTORY_HEAP_WINDOW` (a smaller one
/// maps more slots, a larger one fewer; at most [`MAP_SLOT`]).
fn window() -> u32 {
    std::env::var("VICTORY_HEAP_WINDOW")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(WINDOW)
        .min(MAP_SLOT)
}

/// Whether a capped view heap gets a heap map; `VICTORY_HEAP_MAP=0` keeps
/// the fork's cap (views past it read the null slot).
fn heap_map_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_MAP").is_none_or(|v| v != "0"))
}

/// The heap map translated shaders read: the identity window and the
/// map's slot ([`probe_view_limit`] sets it before the first translation).
fn heap_map() -> Option<(u32, u32)> {
    #[cfg(test)]
    if let Some(map) = TEST_MAP.get() {
        return Some(map);
    }
    victory_shader::emit::view_heap_map()
}

/// VIcky (diagnostics): the heap map's pool (title slots mapped, pool
/// slots free and waiting, refusals), each time the title slots it holds
/// pass another 64Ki.
#[cfg(feature = "diagnostics")]
fn note_pool(made: &Made) {
    use std::sync::atomic::Ordering::Relaxed;
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let Some(stats) = made.map.as_ref().map(SlotMap::stats) else {
        return;
    };
    let step = stats.mapped >> 16;
    if step > LOGGED.load(Relaxed) {
        LOGGED.store(step, Relaxed);
        tracing::info!(?stats, "heap map pool");
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: the heap map (window, slot) of the view heaps this thread
    /// makes, as [`probe_view_limit`] sets one on a capped host, without
    /// reaching other tests' heaps.
    pub(crate) static TEST_MAP: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    /// Under [`Rule::Start`]: a texture's own first bytes are [`HEAD`] or its size, the
    /// smaller ([`stale_head_size`]).
    head_size: bool,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        let key = crate::resource::placed_desc_key(desc);
        if let Some(&known) = self.sizes.get(&key) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(&key);
        self.sizes.insert(key, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => {
                let head = |len: u64| if self.head_size { HEAD.min(len) } else { HEAD };
                let mine = head(end - start);
                self.by_start
                    .range(start.saturating_sub(HEAD - 1)..start.saturating_add(mine))
                    .filter(|&(&at, &(old_end, _))| at >= start || start < at + head(old_end - at))
                    .map(|(at, _)| *at)
                    .collect()
            }
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
/// Whether a placed texture's own first bytes under [`Rule::Start`] are the smaller of
/// [`HEAD`] and its size (`VICTORY_STALE_HEAD_SIZE=0`: [`HEAD`] whatever its size, as
/// before). The title packs small tint palettes (256 bytes to 4 KiB) a few hundred bytes
/// apart: each view written made its neighbours stale, and the faces that read them drew
/// black. (fxceless/victory-fork #923, HLVM04's second comment.)
fn stale_head_size() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STALE_HEAD_SIZE").is_none_or(|v| v != "0"))
}

fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    // Pages are mapped one after the other.
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        // No pass runs (no shader that picks textures by index is bound):
        // the slots there are looked at when they are due, as before.
        STALE[2].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// Whether the card had room for more host copies of textures at the last
/// texture pass (`draw.rs` `heap_textures` stores it): slots go on showing
/// a texture past their view only then ([`stale_keep`]).
pub(crate) static CARD_ROOM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Since the last line in the log: slots that went on showing their texture
/// where they would have been made null ([`stale_keep`]), kept slots whose
/// time was up or that had to make room (made null), and kept slots made
/// again with their new texture.
pub(crate) static STALE_KEPT: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether a slot that shows a texture goes on showing it where it was made
/// null at once, and for how many milliseconds at most
/// (`VICTORY_STALE_KEEP_MS`, off unless set and not 0; the cap is
/// `VICTORY_STALE_KEEP_MAX_MS`, 120,000 unless said, and never less than
/// the first):
/// - a slot the title writes with a view of another texture: until the
///   texture pass has made the new one (it read as nothing from the write
///   until the upload, seconds at times: a torso or a wall black, then
///   back);
/// - a slot whose texture's memory a look finds unmapped: until it is
///   mapped again and uploaded (the title unmaps and maps a texture's pages
///   while it moves them).
///
/// Only while the card has room ([`CARD_ROOM`]): what a kept slot shows is
/// a host copy that could be let go, and with no room all kept slots are
/// made null at the next pass.
///
/// Ported from zm0kiemedia/Neptune_VI main b900c6467 (GPL-3.0-or-later;
/// Traffic and imgone's Neptune): its `stale_kept` / `stale_expire` and its
/// keeping of a slot written with another texture's view
/// (`VICTORY_KEEP_OTHER_VIEWS` there). As there, the first variable is the
/// switch and the second the cap. Not ported: keeping the slots a newer
/// view of other memory makes stale; theirs finds those again with a look
/// over the whole heap that ours does not have.
fn stale_keep() -> Option<u64> {
    static MS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let max = stale_keep_from(
            std::env::var("VICTORY_STALE_KEEP_MS").ok().as_deref(),
            std::env::var("VICTORY_STALE_KEEP_MAX_MS").ok().as_deref(),
        );
        if let Some(max) = max {
            tracing::info!(
                at_most_ms = max,
                "slots go on showing their texture until the new one is made, while the card has room (VICTORY_STALE_KEEP_MS)"
            );
        }
        max
    })
}

/// [`stale_keep`] from the two variables' values.
fn stale_keep_from(keep: Option<&str>, max: Option<&str>) -> Option<u64> {
    let keep = keep?.trim().parse::<u64>().ok().filter(|&ms| ms != 0)?;
    let max = max.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(120_000);
    Some(max.max(keep))
}

/// The title slots that go on showing a texture past their view
/// ([`stale_keep`]): since when (milliseconds), the view each shows (its
/// key), and whether the title has written the slot with a view of another
/// texture since (else the slot shows its own view's texture, whose memory
/// was found unmapped). No host heap needed, so it is tested without one.
#[derive(Default)]
struct KeptPast {
    slots: std::collections::BTreeMap<u32, (u64, u64, bool)>,
}

impl KeptPast {
    /// Slot `index`, which shows the view `key`, would be made null at
    /// `now` (`rewritten`: because the title wrote it with a view of
    /// another texture): whether it goes on showing it instead, which it
    /// does for `max` milliseconds from the first time this was asked of
    /// that view. Returns that, and whether this is that first time.
    fn keep(&mut self, index: u32, key: u64, now: u64, max: u64, rewritten: bool) -> (bool, bool) {
        match self.slots.get_mut(&index) {
            Some(kept) if kept.1 == key => {
                kept.2 |= rewritten;
                let keep = now.saturating_sub(kept.0) < max;
                if !keep {
                    self.slots.remove(&index);
                }
                (keep, false)
            }
            _ => {
                self.slots.insert(index, (now, key, rewritten));
                (max > 0, true)
            }
        }
    }

    /// Slot `index` was made, or made null: no longer kept. True if it was.
    fn forget(&mut self, index: u32) -> bool {
        self.slots.remove(&index).is_some()
    }

    /// Takes out the slots kept for `max` milliseconds or more at `now`
    /// (`all`: every one), each with the view it was kept showing and
    /// whether the title wrote it with another texture's view.
    fn expired(&mut self, now: u64, max: u64, all: bool) -> Vec<(u32, u64, bool)> {
        let due: Vec<(u32, u64, bool)> = self
            .slots
            .iter()
            .filter(|&(_, &(since, ..))| all || now.saturating_sub(since) >= max)
            .map(|(&index, &(_, key, rewritten))| (index, key, rewritten))
            .collect();
        for (index, ..) in &due {
            self.slots.remove(index);
        }
        due
    }
}

/// Slots the title wrote with a texture view that is due at once, since a
/// pass over textures picked by index last took the due ones (`draw.rs`
/// `heap_textures`): what a pass within the frame catches up with
/// (`VICTORY_TEXTURE_CATCH_UP_MS`). (zm0kiemedia/Neptune_VI b900c6467.)
pub(crate) static FRESH_WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Since the last line in the log: first uploads of textures picked by
/// index, by the frames between the slot's write and the upload
/// ([`age_bucket`]: 0, 1, 2, 3-5, 6-15, 16-63, more). Until the upload a
/// shader that picks the slot reads nothing. Row 0: slots whose memory was
/// there at every look (they waited for the queue and for the host copy
/// alone); row 1: slots a look found unmapped (they waited for the title's
/// own streaming too, which no queue order shortens).
/// Ported from zm0kiemedia/Neptune_VI main b900c6467 (`QUEUE_AGE`; Traffic
/// and imgone's Neptune); the split in two rows is ours.
pub(crate) static QUEUE_AGE: [[std::sync::atomic::AtomicU64; 7]; 2] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 7] }; 2];

/// The bucket of [`QUEUE_AGE`] for an upload `age` frames after the write.
pub(crate) fn age_bucket(age: u64) -> usize {
    match age {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=5 => 3,
        6..=15 => 4,
        16..=63 => 5,
        _ => 6,
    }
}

/// Of a row of [`QUEUE_AGE`]: all its uploads, and those in the same frame
/// as the write or the one after it.
pub(crate) fn age_shares(row: &[u64; 7]) -> (u64, u64) {
    (row.iter().sum(), row[0] + row[1])
}

/// Whether the frame line tells of [`QUEUE_AGE`] (on unless
/// `VICTORY_QUEUE_AGE=0`; the counting is two atomics an upload).
pub(crate) fn queue_age_line() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_AGE").is_none_or(|v| v != "0"))
}

thread_local! {
    /// This thread is making a slot whose texture was uploaded ahead
    /// ([`HostHeap::publish_ahead`]): the slot's place in the queue of
    /// textures picked by index was settled when that upload was recorded.
    static PUBLISHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
    /// A look found none of its memory mapped ([`TextureQueue::away`]).
    away: bool,
    /// Since the slot was written or its memory changed, for a texture
    /// whose first and last page are both away
    /// ([`TextureQueue::between`]): 0, the memory between them was not
    /// looked at; 1, it was, and none was there; 2, some was there.
    between: u8,
}

/// Whether due texture slots are served in turns, a slot looked at before
/// and one never looked at ([`TextureQueue::due`]; on unless
/// `VICTORY_FAIR_DUE=0`). Ported from fxceless/victory-fork PR #1246
/// (Exopus-0x65f7ad0).
fn fair_due() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAIR_DUE").is_none_or(|v| v != "0"))
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
}

impl TextureQueue {
    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0), away: false, between: 0 });
    }

    fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - first the ones looked at before (their memory was not mapped, or
    ///   only part of it): the look is a query, and memory that has
    ///   arrived since is what the title streamed in last;
    /// - then the others, the newest write first and the oldest next, in
    ///   turns: slot order served the low slots of a backlog of
    ///   25,000-40,000 while the textures of the shot on screen waited
    ///   (run 52: 98% of the drawn pixels had no colour texture in the
    ///   first 3D frames).
    ///
    /// With `VICTORY_FAIR_DUE` (on unless `=0`) the two kinds are served
    /// in turns, a slot looked at before first, instead of all those
    /// looked at before first: a backlog of thousands of slots streaming in
    /// a part at a time filled every pass while slots never looked at
    /// waited hundreds of frames (fxceless/victory-fork PR #1246).
    ///
    /// Also counts all slots, as the frame line needs the whole backlog:
    /// queued, due, waiting for memory, partly uploaded, and the age in
    /// frames of the oldest due write.
    fn due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        self.due_in_order(frame, max, fair_due())
    }

    /// [`Self::due`] with the order chosen: `fair` serves the two kinds in
    /// turns, otherwise all those looked at before come first.
    fn due_in_order(&self, frame: u64, max: usize, fair: bool) -> (Vec<(u32, u64)>, [u64; 5]) {
        let mut counts = [self.slots.len() as u64, 0, 0, 0, 0];
        let mut again = Vec::new();
        let mut fresh = Vec::new();
        for (&index, w) in &self.slots {
            if w.mapped > 0 {
                counts[3] += 1;
            } else if w.looks > 0 {
                counts[2] += 1;
            }
            if w.due <= frame {
                counts[1] += 1;
                counts[4] = counts[4].max(frame.saturating_sub(w.written));
                if max > 0 {
                    let class = if w.looks > 0 || w.mapped > 0 {
                        &mut again
                    } else {
                        &mut fresh
                    };
                    class.push((w.write, index));
                }
            }
        }
        let (retries, fresh_take) = if fair {
            let (a, f) = (again.len(), fresh.len());
            (a.min(max - f.min(max / 2)), f.min(max - a.min(max.div_ceil(2))))
        } else {
            let retries = again.len().min(max);
            (retries, fresh.len().min(max - retries))
        };
        let first_retry = again.len() - retries;
        let by_write = |&(write, _): &(u64, u32)| write;
        if first_retry > 0 {
            again.select_nth_unstable_by_key(first_retry, by_write);
        }
        let again = &mut again[first_retry..];
        again.sort_unstable_by_key(by_write);
        let fresh_count = fresh_take;
        let mut order = Vec::with_capacity(retries + fresh_count);
        order.extend(again.iter().rev().map(|&(write, index)| (index, write)));
        if fresh_count == 0 {
            return (order, counts);
        }
        let again_order = std::mem::take(&mut order);

        // The write number gives the order. Only sort the ends we will
        // serve, leaving the middle of a large backlog unordered.
        let oldest_count = fresh_count / 2;
        let newest_count = fresh_count - oldest_count;
        let newest_start = fresh.len() - newest_count;
        if fresh_count == fresh.len() {
            fresh.sort_unstable_by_key(by_write);
        } else {
            fresh.select_nth_unstable_by_key(newest_start, by_write);
            if oldest_count > 0 {
                fresh[..newest_start].select_nth_unstable_by_key(oldest_count, by_write);
                fresh[..oldest_count].sort_unstable_by_key(by_write);
            }
            fresh[newest_start..].sort_unstable_by_key(by_write);
        }
        let mut oldest = fresh[..oldest_count].iter();
        let mut fresh_order = Vec::with_capacity(fresh_count);
        for &(write, index) in fresh[newest_start..].iter().rev() {
            fresh_order.push((index, write));
            if let Some(&(write, index)) = oldest.next() {
                fresh_order.push((index, write));
            }
        }
        if !fair {
            order = again_order;
            order.extend(fresh_order);
            return (order, counts);
        }
        let (mut a, mut f) = (again_order.into_iter(), fresh_order.into_iter());
        loop {
            let (x, y) = (a.next(), f.next());
            if x.is_none() && y.is_none() {
                break;
            }
            order.extend(x);
            order.extend(y);
        }
        (order, counts)
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.slots.remove(&index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
        }
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    #[allow(clippy::too_many_arguments)]
    fn recheck(&mut self, index: u32, write: u64, due: u64, at: (u64, u64), print: u64, looks: u8, uploaded: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
        }
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look.
    fn checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.slots.get(&index).map(|w| w.checked).filter(|c| c.2 != 0)
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0), away: false, between: 0 });
    }

    /// A look at slot `index` found none of its memory mapped.
    fn away(&mut self, index: u32, write: u64) {
        if let Some(w) = self.same(index, write) {
            w.away = true;
        }
    }

    /// Whether the look at slot `index`, whose texture has neither its
    /// first nor its last page there, goes on to the memory between them
    /// (`crate::texture::mapped_bytes_in_turn`). That look asks about every
    /// 64 KiB block of the texture, so a slot has it once after it was
    /// written or the runtime told of its memory, in its turn; then again
    /// at every eighth look in a row that found nothing (for what the
    /// runtime does not tell of); and at every look while its texture is
    /// one that has memory only between its ends.
    fn between(&self, index: u32, write: u64) -> crate::texture::Between {
        use crate::texture::Between;
        match self.slots.get(&index).filter(|w| w.write == write) {
            Some(w) if w.between == 2 => Between::Always,
            Some(w) if w.between == 0 || (w.looks > 0 && w.looks.is_multiple_of(8)) => Between::InBudget,
            _ => Between::Not,
        }
    }

    /// The look between the ends of slot `index`'s texture was made, and
    /// `found` some of its memory there or none.
    fn looked_between(&mut self, index: u32, write: u64, found: bool) {
        if let Some(w) = self.same(index, write) {
            w.between = if found { 2 } else { 1 };
        }
    }

    /// For [`QUEUE_AGE`], asked before the pass says what it did with slot
    /// `index`: if this upload at `frame` is the first since the slot was
    /// written (nothing of it was uploaded before, and it is not a look
    /// after an upload), the frames since the write, and whether a look
    /// found its memory unmapped meanwhile.
    fn first_upload(&self, index: u32, write: u64, frame: u64) -> Option<(u64, bool)> {
        let w = self.slots.get(&index).filter(|w| w.write == write)?;
        (w.mapped == 0 && w.checked.2 == 0).then(|| (frame.saturating_sub(w.written), w.away))
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    let kind = if stride == 0 && !raw && victory_shader::emit::heap_half_bits() {
        integer_kind(format.0 as u32)
    } else {
        0
    };
    Some(if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes as u32) | kind
    })
}

/// The stride table's format-kind bits of a typed buffer view in DXGI format
/// `format`: [`SLOT_INTEGER`] for a `UINT` format, with [`SLOT_SIGNED`] for a
/// `SINT` one; 0 otherwise. Translated programs pass 16-bit typed data of
/// such views as bits, not as halves (`victory_shader::emit::heap_half_bits`;
/// from upstream Marvisak/VIctory 4243a96a7).
///
/// [`SLOT_INTEGER`]: victory_shader::emit::SLOT_INTEGER
/// [`SLOT_SIGNED`]: victory_shader::emit::SLOT_SIGNED
fn integer_kind(format: u32) -> u32 {
    use victory_shader::emit::{SLOT_INTEGER, SLOT_SIGNED};
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => SLOT_INTEGER,
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => SLOT_INTEGER | SLOT_SIGNED,
        _ => 0,
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Asks the host, once, whether it caps view heaps at [`PORTABLE_VIEWS`],
/// and if so tells the translator now (`set_view_heap_limit`) instead of
/// when the title's heap is made: a shader keeps the limit it was
/// translated with, and the title makes 25 pipelines (their
/// shaders translated at once) before it makes its view heap. Those would
/// read a view index past the heap unclamped, the read that hangs NVIDIA.
/// Called before a shader is translated.
/// shortcut: the limit is the capped heap's last slot whatever size the
/// title's heap turns out to have; a heap that fits is not capped and
/// keeps no null slot there (an index past it is as undefined as before).
///
/// A capped host also gets a heap map (unless `VICTORY_HEAP_MAP=0`): the
/// translator looks bindless views past the identity window up in it
/// (`set_view_heap_map`), and from then on every view heap of more title
/// slots than the window is laid out for it ([`MapLayout`], whatever its
/// size: the map's slot is fixed in the translations). Also called before
/// such a heap is made, so that a heap made before the first translation
/// agrees with the shaders.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        // SAFETY: plain heap creation; the heap is released at once.
        let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: PORTABLE_VIEWS + 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        };
        if made.is_err() {
            tracing::info!("the host caps view heaps at {PORTABLE_VIEWS}: view indices are clamped from the first shader on");
            victory_shader::emit::set_view_heap_limit(PORTABLE_VIEWS - 1);
            if heap_map_on() {
                let window = window();
                victory_shader::emit::set_view_heap_map(Some((window, MAP_SLOT)));
                tracing::info!(
                    window,
                    map_slot = MAP_SLOT,
                    "heap map: bindless views past the identity window are looked up (VICTORY_HEAP_MAP=0: off)"
                );
            }
        }
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

// The heap map leaves the fork's dynamic slots and guard slot in place.
const _: () = assert!(MAP_SLOT + 1 + DYNAMIC == PORTABLE_VIEWS - 1);

/// The fewest slots a capped host heap keeps for the title's own (with a
/// heap map: past its identity window), however many dynamic slots
/// `VICTORY_DYNAMIC_SLOTS` asks for ([`capped_dynamic_slots`]).
const LEAST_TITLE: u32 = 1 << 16;

/// The slots a capped host heap with a heap map keeps for the title's
/// (identity window and pool) when `VICTORY_DYNAMIC_SLOTS` is not set: the
/// number fxceless/victory-fork PR #1258 keeps ([`capped_dynamic_slots`]).
const CAPPED_TITLE: u32 = 393_215;

/// `VICTORY_DYNAMIC_SLOTS`, if it is set to a number: the dynamic slots of
/// a title heap that a capped host heap does not hold whole.
fn dynamic_slots_set() -> Option<u32> {
    static SET: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *SET.get_or_init(|| std::env::var("VICTORY_DYNAMIC_SLOTS").ok().and_then(|v| v.trim().parse().ok()))
}

/// The dynamic slots of a host heap capped at [`PORTABLE_VIEWS`] (NVIDIA;
/// its null guard slot is `guard`) for a title heap that does not fit in it
/// with [`DYNAMIC`] more: they come out of the title's. `set` is
/// `VICTORY_DYNAMIC_SLOTS`: at least [`DYNAMIC`], which is what every such
/// heap had before (`VICTORY_DYNAMIC_SLOTS=262144` is the layout as it
/// was), and leaving the title [`LEAST_TITLE`] slots past the heap map's
/// identity window (`window`; `None`: a heap without a map).
///
/// (Ported from fxceless/victory-fork PR #1258 = DexNom/VIctory PR #68, by
/// notquiteog, adapted to the heap map.) 262,144 dynamic slots ran dry in
/// the open world: a frame asks for about 200,000 and holds them until the
/// card has run it, the tables that found none were refused and their
/// draws dropped (our play session 6: 20,958 tables refused, up to 3,595 a
/// frame; the PR: 13,000 to 60,000 per 5 s on an RTX 5060 Ti, none with
/// 606,784 and 350,000 free at the fewest).
///
/// Not set, with a heap map: the title keeps [`CAPPED_TITLE`] slots, the
/// PR's number, here the identity window and 262,143 pool slots; the other
/// 606,784 are dynamic. Nothing is lost by position: a table reaching past
/// the window is made afresh in dynamic slots wherever it lies (`draw.rs`,
/// `tables`), and a view a shader picks by index past the window is in
/// whichever pool slot the map names. Only a pool used up loses views (they
/// read the guard, and "heap map pool used up" is logged): of the 606,782
/// pool slots, fewer than 65,536 were held in every run logged so far (the
/// diagnostics line "heap map pool", written at each 65,536 held, is in
/// none of 161 logs, the play session among them).
///
/// Not set, without a heap map (`VICTORY_HEAP_MAP=0`): [`DYNAMIC`] as
/// before. There a title slot is its own host slot, and the PR's finding
/// that the title's heap from slot 393,216 on is only its ring of per-draw
/// tables (none a view only an index reaches) was not measured on our
/// tree: `VICTORY_DYNAMIC_SLOTS=606784` is the PR's split.
fn capped_dynamic_slots(guard: u32, window: Option<u32>, set: Option<u32>) -> u32 {
    let most = guard
        .saturating_sub(window.unwrap_or(0).saturating_add(LEAST_TITLE))
        .max(DYNAMIC);
    let unset = match window {
        Some(_) => guard.saturating_sub(CAPPED_TITLE),
        None => DYNAMIC,
    };
    set.unwrap_or(unset).clamp(DYNAMIC, most)
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        // Whether the translator looks bindless views up in a heap map, as
        // it does from the first translation on (asked now if none ran).
        #[cfg(not(test))]
        if !sampler && heap_map_on() {
            probe_view_limit();
        }
        let layout = (!sampler)
            .then(heap_map)
            .flatten()
            .filter(|&(window, _)| count > window)
            .map(|(window, slot)| {
                // A title heap the window and the pool do not hold whole
                // gives up pool slots for dynamic ones.
                let dynamic = if count > slot {
                    capped_dynamic_slots(slot + 1 + DYNAMIC, Some(window), dynamic_slots_set())
                } else {
                    DYNAMIC
                };
                MapLayout::new(count, window, slot, dynamic)
            });
        Self::laid_out(sampler, count, layout)
    }

    /// A host heap for a title heap of `count` slots, with a heap map laid
    /// out as `layout` says if there is one.
    fn laid_out(sampler: bool, count: u32, layout: Option<MapLayout>) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let mut extra = if sampler { 0 } else { DYNAMIC };
        let title_count = count;
        let made = match layout {
            Some(l) => {
                extra = l.dynamic;
                tracing::warn!(
                    count,
                    window = l.window,
                    pool = l.pool,
                    map_slot = l.slot,
                    dynamic = l.dynamic,
                    "host view heap with a heap map: title slots past the window that hold views get pool slots (VICTORY_DYNAMIC_SLOTS=262144: the dynamic slots as they were)"
                );
                make(l.size()).map(|heap| (heap, l.window, l.size()))
            }
            None => make(count + extra).map(|heap| (heap, count, count + extra)),
        };
        let (heap, count, size) = match made {
            Ok(made) => made,
            Err(e) if layout.is_some() => {
                tracing::warn!(error = %e, count, "no host view heap with a heap map");
                return None;
            }
            Err(e) if !sampler && count + extra > PORTABLE_VIEWS => {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. The heap's last slot stays null, and shaders read it for
                // any view index from there on (`set_view_heap_limit`): a read past the heap's
                // end hung NVIDIA's GPU (the title's heap is a little over 1 000 000 views).
                // Without a heap map the dynamic slots stay [`DYNAMIC`]
                // unless `VICTORY_DYNAMIC_SLOTS` says otherwise.
                extra = capped_dynamic_slots(PORTABLE_VIEWS - 1, None, dynamic_slots_set());
                tracing::warn!(error = %e, count, dynamic = extra, "host view heap capped at {PORTABLE_VIEWS}; views past it read a null slot");
                victory_shader::emit::set_view_heap_limit(PORTABLE_VIEWS - 1);
                (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - extra - 1, PORTABLE_VIEWS)
            }
            Err(e) => {
                tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                return None;
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(if layout.is_some() { size } else { count + extra }) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // The heap map: an entry per title slot past the window, zeroed (no
        // slot: the guard).
        let map = layout.and_then(|l| {
            let bytes = u64::from((title_count - l.window).max(1)) * 4;
            // Written, never read, on the CPU: on the card where it can be.
            let Some((buffer, at)) = host.table_buffer(bytes, "heap map").or_else(|| host.upload_buffer(bytes)) else {
                tracing::warn!(bytes, "no heap map: title slots past the window read the null slot");
                return None;
            };
            // SAFETY: the buffer's whole mapping.
            unsafe { std::ptr::write_bytes(at, 0, bytes as usize) };
            crate::host::name(&buffer, || "heap map".to_owned());
            Some(HeapMap { buffer, at: at as usize })
        });
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                // Written, never read, on the CPU: on the card where it can be.
                let (buffer, data) = host
                    .table_buffer(bytes, "element strides by view heap slot")
                    .or_else(|| host.upload_buffer(bytes))?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        // Title-keyed state covers the title's slots past the window too.
        let (pool, dynamic_first, titled) = match layout {
            Some(l) => (l.pool, l.dynamic_first(), title_count),
            None => (0, count, count),
        };
        // Which dynamic slots are taken: none, but for the map's slot where
        // it lies among them (a heap with more than [`DYNAMIC`]), for good.
        let mut taken = vec![false; extra as usize];
        let mut free = extra;
        if let Some(at) = layout.and_then(MapLayout::slot_in_dynamic) {
            taken[at as usize] = true;
            free -= 1;
        }
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            dynamic_first,
            title_count,
            sampler,
            made: Mutex::new(Made {
                keys: vec![0; (count + pool) as usize],
                map: map
                    .is_some()
                    .then(|| SlotMap::new(count, title_count, count, pool)),
            }),
            map,
            title_start: AtomicU64::new(0),
            reclaimed: AtomicU64::new(u64::MAX),
            kept: Mutex::new(vec![None; (dynamic_first + extra) as usize]),
            next: Mutex::new((0, taken)),
            free: std::sync::atomic::AtomicU32::new(free),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            buffers: Mutex::default(),
            changed: Mutex::default(),
            textures: Mutex::default(),
            kept_past: Mutex::default(),
            tabled: (0..if sampler { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            placed: Mutex::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        if let Some(l) = layout {
            // The map's raw view (`raw_at` holds the buffer), or one that
            // reads zeros: every lookup then gives the guard slot.
            match &h.map {
                Some(m) => h.raw_at(l.slot, &m.buffer, 0, u64::from(title_count - l.window) * 4),
                None => h.null_raw_at(l.slot),
            }
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let free = self.free.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            DYNAMIC_LOW[0].fetch_min(free, std::sync::atomic::Ordering::Relaxed);
            DYNAMIC_LOW[1].store(self.dynamic, std::sync::atomic::Ordering::Relaxed);
        }
        let room = free >= n;
        let enough = room && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start (and on past `next`
        // for a run across it).
        let found = if enough { free_run(taken, *next as usize, n, across()) } else { None };
        #[cfg(feature = "diagnostics")]
        if matches!(found, Some((_, true))) {
            DYNAMIC_WHY[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let Some((first, _)) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            DYNAMIC_WHY[if !room {
                0
            } else if !enough {
                1
            } else {
                2
            }]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken[first..first + n].fill(true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.dynamic_first + first as u32)
    }

    /// Tests: how many dynamic slots are free.
    #[cfg(test)]
    pub(crate) fn dynamic_free(&self) -> u32 {
        self.free.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The title slots this heap mirrors at their own index (tables read in
    /// place reach only these: one that reaches past them is made afresh);
    /// fewer than the title's when the host caps its size, and with a heap
    /// map the identity window.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The host slot title slot `title` is made in (`made` locked): its own
    /// index below `count`; with a heap map past it, its pool slot, taken
    /// now if it has none (then `true`: [`HostHeap::publish`] writes its
    /// map entry once the slot is made). `None` past the title's slots, or
    /// with the pool used up even after [`HostHeap::reclaim`] (the title
    /// slot then reads the guard slot).
    fn host_slot(&self, made: &mut Made, title: u32) -> Option<(u32, bool)> {
        if title < self.count {
            return Some((title, false));
        }
        if title >= self.title_count {
            return None;
        }
        {
            let map = made.map.as_mut()?;
            if let Some(host) = map.host(title) {
                return Some((host, false));
            }
            if let Some(host) = map.assign(title, crate::host::submissions().1) {
                return Some((host, true));
            }
        }
        if self.reclaim(made) > 0
            && let Some(host) = made.map.as_mut()?.assign(title, crate::host::submissions().1)
        {
            return Some((host, true));
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(title, "heap map pool used up: title slots past the window read the null slot");
        });
        None
    }

    /// Writes title slot `title`'s heap map entry (`made` locked), after
    /// its pool slot was made.
    fn publish(&self, made: &Made, title: u32) {
        let (Some(map), Some(m)) = (&made.map, &self.map) else {
            return;
        };
        let Some(i) = title.checked_sub(map.window()).filter(|&i| i < map.len()) else {
            return;
        };
        // The descriptor first, then the entry naming it.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // SAFETY: entry `i` of the mapped heap map.
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(map.entry(title)) };
    }

    /// Gives back title slot `title`'s pool slot (`made` locked), if it has
    /// one: its map entry is zeroed (the guard slot), and the slot is
    /// handed out again once the submissions that may have read the old
    /// entry are done. Its view stays, held, until it is made again.
    fn unmap(&self, made: &mut Made, title: u32) {
        let (Some(map), Some(m)) = (made.map.as_mut(), &self.map) else {
            return;
        };
        let Some(host) = map.host(title).filter(|_| title >= map.window()) else {
            return;
        };
        let i = title - map.window();
        // SAFETY: entry `i` of the mapped heap map (`title` is mapped).
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(0) };
        // The entry is out before the submissions running are counted.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        map.free(title, crate::host::submissions().0);
        made.keys[host as usize] = 0;
    }

    /// Title slot `title` holds no view (`made` locked): its host slot reads
    /// nothing and leaves the sets of the slots shaders pick by index (as
    /// the fork did), and a pool slot is given back. `for_draw`: the slot
    /// leaves the sets even if its host slot was never made.
    fn empty_slot(&self, made: &mut Made, title: u32, for_draw: bool) {
        match made.host(title) {
            Some(host) if made.keys[host as usize] != 0 => {
                made.keys[host as usize] = 0;
                self.make_null(host);
                let _ = self.note_buffer(title, None, false);
            }
            // Left for later and stale since: out of the sets and the stride table.
            _ if for_draw => {
                let _ = self.note_buffer(title, None, false);
            }
            _ => {}
        }
        self.unmap(made, title);
    }

    /// With the heap map's pool used up (`made` locked): gives back the pool
    /// slots of title slots that hold no live view any more (a view of a
    /// released resource, or no view), which nothing else finds: the fork
    /// learns of a slot's view only from the title's descriptor writes and
    /// copies, and a title that releases a resource need not write its
    /// slots again. At most once every two seconds; the slots given back
    /// are handed out once the submissions running now are done. Returns
    /// how many.
    fn reclaim(&self, made: &mut Made) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        const EVERY_MS: u64 = 2000;
        let now = crate::host::now_ms();
        let last = self.reclaimed.load(Relaxed);
        if last != u64::MAX && now < last.saturating_add(EVERY_MS) {
            return 0;
        }
        self.reclaimed.store(now, Relaxed);
        let start = self.title_start.load(Relaxed);
        let Some(map) = made.map.as_ref().filter(|_| start != 0) else {
            return 0;
        };
        let started = std::time::Instant::now();
        let dead: Vec<u32> = map
            .mapped()
            .map(|(title, _)| title)
            .filter(|&title| {
                let handle = start + u64::from(title) * TITLE_SLOT_BYTES;
                // SAFETY: `title < title_count`, a slot of the live title heap.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                tag != VIEW_TAG || view_at(i, generation).is_none()
            })
            .collect();
        for &title in &dead {
            self.empty_slot(made, title, true);
        }
        // The first passes at warn level, the rest at debug.
        static PASSES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let (given_back, ms) = (dead.len(), started.elapsed().as_millis());
        if PASSES.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        } else {
            tracing::debug!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        }
        dead.len()
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in service order. Also returns counts for
    /// the whole queue ([`TextureQueue::due`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.due(frame, max)
    }

    /// Of the title slots `among`, in their order, those the queue holds
    /// due for an upload at `frame`, with the write each was handed out as
    /// (as [`HostHeap::texture_slots_due`]): a bake's own sources, which go
    /// first in the pass before it.
    pub(crate) fn texture_slots_due_of(&self, frame: u64, among: &[u32]) -> Vec<(u32, u64)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<(u32, u64)> = Vec::new();
        for &index in among {
            if out.iter().any(|&(i, _)| i == index) {
                continue;
            }
            if let Some(w) = textures.slots.get(&index).filter(|w| w.due <= frame) {
                out.push((index, w.write));
            }
        }
        out
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// The texture of title slot `index` (as handed out with `write`) is on
    /// the card, whole or in part: counted in [`QUEUE_AGE`] if that is the
    /// first time since the slot was written. Called before
    /// [`HostHeap::texture_slot_done`] or [`HostHeap::texture_slot_recheck`]
    /// say what becomes of the slot.
    pub(crate) fn texture_slot_uploaded(&self, index: u32, write: u64, frame: u64) {
        {
            let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((age, away)) = textures.first_upload(index, write, frame) {
                QUEUE_AGE[usize::from(away)][age_bucket(age)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // Up to date: no longer one that shows a texture past its view.
        if stale_keep().is_some() && self.kept_past.lock().unwrap_or_else(|e| e.into_inner()).forget(index) {
            STALE_KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// See [`TextureQueue::between`].
    pub(crate) fn texture_slot_between(&self, index: u32, write: u64) -> crate::texture::Between {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).between(index, write)
    }

    /// See [`TextureQueue::looked_between`].
    pub(crate) fn texture_slot_looked_between(&self, index: u32, write: u64, found: bool) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).looked_between(index, write, found);
    }

    /// A look at title slot `index` found none of its memory mapped: see
    /// [`TextureQueue::away`].
    pub(crate) fn texture_slot_away(&self, index: u32, write: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).away(index, write);
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.recheck(index, write, due, at, print, looks, uploaded);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.later(index, write, frame, most);
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full. `made` first, then `kept`,
        // as `mirror` takes them.
        // (With a heap map, a title slot is made in its host slot: its own
        // below the window, its pool slot past it.)
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = made.host(index) else {
            return false;
        };
        made.keys.get(host as usize).is_some_and(|&made| made != 0 && made == key)
            && self
                .kept
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(host as usize)
                .is_some_and(Option::is_some)
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    /// (With a heap map, a title slot past the window keeps its pool slot:
    /// it still holds the view, made again once its memory is mapped.)
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if made
            .host(index)
            .and_then(|host| made.keys.get(host as usize))
            .is_none_or(|&key| key == 0)
        {
            return false;
        }
        // With `VICTORY_STALE_KEEP_MS`: it goes on showing its host copy
        // for a while, until its memory is back and uploaded.
        if self.kept_on(&made, index, false) {
            return false;
        }
        self.unmake(&mut made, index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Title slot `index` shows a texture and would be made null (`made`
    /// locked; `rewritten`: the title wrote it with a view of another
    /// texture): whether it goes on showing it instead ([`stale_keep`]).
    fn kept_on(&self, made: &Made, index: u32, rewritten: bool) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(max) = stale_keep() else {
            return false;
        };
        let Some(host) = made.host(index) else {
            return false;
        };
        let key = made.keys.get(host as usize).copied().unwrap_or(0);
        let mut kept_past = self.kept_past.lock().unwrap_or_else(|e| e.into_inner());
        let shows = key != 0
            && CARD_ROOM.load(Relaxed)
            && view_at(key as u32, (key >> 32) as u32).as_ref().is_some_and(is_texture_view)
            && self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(host as usize).is_some_and(Option::is_some);
        if !shows {
            kept_past.forget(index);
            return false;
        }
        let (keep, first) = kept_past.keep(index, key, crate::host::now_ms(), max, rewritten);
        if keep && first {
            STALE_KEPT[0].fetch_add(1, Relaxed);
        }
        keep
    }

    /// Makes the kept slots null whose time is up, or all of them when the
    /// card has no `room` ([`stale_keep`]): once a texture pass.
    pub(crate) fn stale_expire(&self, room: bool) {
        let Some(max) = stale_keep() else {
            return;
        };
        if self.kept_past.lock().unwrap_or_else(|e| e.into_inner()).slots.is_empty() {
            return;
        }
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let due = self.kept_past.lock().unwrap_or_else(|e| e.into_inner()).expired(crate::host::now_ms(), max, !room);
        for (index, key, rewritten) in due {
            // Made again since, or made null: not this entry's.
            let shown = made.host(index).and_then(|host| made.keys.get(host as usize).copied());
            if shown != Some(key) {
                continue;
            }
            // A slot that shows its own view's texture, whose memory is
            // back: nothing stale about it (a look that found the memory
            // as it was uploads nothing and says nothing).
            let back = !rewritten
                && matches!(
                    view_at(key as u32, (key >> 32) as u32),
                    Some(View::Placed(false, resource, desc)) if resource[0] != 1 && {
                        let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                        let len = crate::resource::placed_bytes(&resource).unwrap_or(1);
                        crate::texture::any_mapped(address as usize, len as usize)
                    }
                );
            if !back {
                self.unmake(&mut made, index);
                STALE_KEPT[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Makes title slot `index`'s host slot null again, as it is before its
    /// view is made: a null view of the kind it held.
    fn unmake(&self, made: &mut Made, index: u32) {
        let Some(host) = made.host(index) else {
            return;
        };
        let Some(key) = made.keys.get_mut(host as usize).map(std::mem::take) else {
            return;
        };
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(host)),
            None => null_view(self.cpu(host)),
        }
        self.keep(host, None);
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            if tabled_out() {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                if textures.slots.remove(&index).is_some() {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        tabled_out() && stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        if PUBLISHING.get() {
            return;
        }
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                if !seen {
                    FRESH_WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                0
            };
            textures.written(index, frame, due, seen);
        } else {
            textures.remove(index);
        }
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        buffers.iter().copied().collect()
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        let mut changed = self.changed.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *changed).into_iter().collect()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]): out of the queue of textures already, their host
    /// slots are the caller's to make null (it holds `made`).
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<u32> {
        if self.sampler {
            return Vec::new();
        }
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, crate::frame::current());
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, crate::frame::current()) =>
            {
                Some((resource, u64::from(desc[10]) | u64::from(desc[11]) << 32))
            }
            _ => None,
        };
        let stale = {
            use std::sync::atomic::Ordering::Relaxed;
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let span = placed_view
                .and_then(|(resource, start)| Some((start, start.saturating_add(placed.bytes(resource)?))));
            match span {
                Some((start, end)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    placed.head_size = stale_head_size();
                    placed.note(index, start, end, !seen)
                }
                None => {
                    placed.forget(index);
                    Vec::new()
                }
            }
        };
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &stale {
                textures.remove(slot);
            }
        }
        let stride = view.and_then(view_stride);
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword, either value is one the slot had).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride.unwrap_or(0)) };
        }
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let was = if stride.is_some() {
            buffers.insert(index);
            true
        } else {
            buffers.remove(&index)
        };
        if was {
            self.changed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(index);
        }
        stale
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.dynamic_first) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|taken| **taken).count() as u32;
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record at `handle` if that is still
    /// the record `key` names, whose texture's upload is in the card's queue
    /// now (`crate::heap_worker`, `VICTORY_HEAP_WORKER=1`). The slot's place
    /// in the queue of textures picked by index was settled when the upload
    /// was recorded and stays as it is. False if the title wrote the slot
    /// again since (that write is queued and looked at in its own turn), or
    /// the slot cannot be made.
    pub(crate) fn publish_ahead(&self, index: u32, handle: u64, key: u64) -> bool {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || (u64::from(generation) << 32 | u64::from(i)) != key {
            return false;
        }
        struct Publishing;
        impl Drop for Publishing {
            fn drop(&mut self) {
                PUBLISHING.set(false);
            }
        }
        PUBLISHING.set(true);
        let _publishing = Publishing;
        self.mirror(index, handle, true).is_some()
    }

    /// The key of the record title slot `index`'s host slot was made from
    /// (0: not made, or made null).
    #[cfg(test)]
    fn made_key(&self, index: u32) -> u64 {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made.host(index).map_or(0, |host| made.keys[host as usize])
    }

    /// Makes title slot `index`'s host slot (its own, or with a heap map
    /// past the identity window its pool slot) from the record the title
    /// has at `handle` if it changed since, and returns the view recorded
    /// there. Buffer views are made only for draws (`for_draw`), which also
    /// copy their data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        let mirrored = if self.map.is_some() { self.title_count } else { self.count };
        if index >= mirrored {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        if self.map.is_some() && index >= self.count {
            // Where the title's descriptors start, for `reclaim`.
            let start = handle.wrapping_sub(u64::from(index) * TITLE_SLOT_BYTES);
            let _ = self.title_start.compare_exchange(
                0,
                start,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            self.empty_slot(&mut made, index, for_draw);
            return None;
        };
        let Some((host, taken)) = self.host_slot(&mut made, index) else {
            // Past the window with the pool used up: the slot reads the
            // guard. What the title wrote is noted all the same (the sets
            // of slots shaders pick by index retry it).
            if !for_draw {
                for stale in self.note_buffer(index, Some(&view), false) {
                    if stale != index {
                        self.unmake(&mut made, stale);
                    }
                }
            }
            return Some(view);
        };
        if taken || made.keys[host as usize] != key {
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            for stale in self.note_buffer(index, Some(&view), for_draw) {
                if stale != index {
                    self.unmake(&mut made, stale);
                }
            }
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            let old = made.keys[host as usize];
            let same = view_at(old as u32, (old >> 32) as u32).is_some_and(|old| same_placed_texture(&old, &view));
            if later && keep_views() && same {
                KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(view);
            }
            // And a view of another texture, with `VICTORY_STALE_KEEP_MS`:
            // the title streams a material's texture in as a new texture at
            // another address and writes the slot with it. The slot goes
            // on showing the host copy it holds until the texture pass has
            // made the new one (zm0kiemedia/Neptune_VI b900c6467).
            if later && !same && !taken && is_texture_view(&view) && self.kept_on(&made, index, true) {
                return Some(view);
            }
            if !later
                && stale_keep().is_some()
                && self.kept_past.lock().unwrap_or_else(|e| e.into_inner()).forget(index)
            {
                STALE_KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            made.keys[host as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(host))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(host));
            }
            self.keep(host, kept.flatten());
            if taken {
                self.publish(&made, index);
                #[cfg(feature = "diagnostics")]
                note_pool(&made);
            }
        }
        Some(view)
    }

    /// Whether title slot `index`'s host slot shows `buffer`; `None` if it
    /// shows no resource (diagnostics: a buffer view's slot must show the
    /// host copy that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let host = self.made.lock().unwrap_or_else(|e| e.into_inner()).host(index)?;
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let shown = kept.get(host as usize)?.as_ref()?;
        Some(windows::core::Interface::as_raw(shown) == windows::core::Interface::as_raw(buffer))
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).keys.get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture view `desc` on a host copy that leaves out the title's
/// `drop` largest levels (`crate::resource::drop_levels`): its levels
/// shift down by as many ([`crate::resource::shift_levels`]). 2D textures,
/// arrays and cubes only, the dimensions such copies have.
fn shift_view_levels(drop: u32, desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) {
    if drop == 0 {
        return;
    }
    let shift = crate::resource::shift_levels;
    // SAFETY: each arm reads and writes the union member its dimension names.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let t = &mut desc.Anonymous.Texture2D;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let t = &mut desc.Anonymous.Texture2DArray;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let t = &mut desc.Anonymous.TextureCube;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let t = &mut desc.Anonymous.TextureCubeArray;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            _ => {}
        }
    }
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

pub(crate) fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((shadow, address, len));
    }
    crate::shadow::find(address, len).map(|shadow| (shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// The format the host makes a typed buffer view in ([`int_views_as_float`]): a 32-bit
/// integer format as the float format of the same width, every other as the title's.
fn bit_exact_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    if !int_views_as_float() {
        return format;
    }
    match format {
        DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT => {
            DXGI_FORMAT_R32G32B32A32_FLOAT
        }
        DXGI_FORMAT_R32G32B32_UINT | DXGI_FORMAT_R32G32B32_SINT => DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_SINT => DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT => DXGI_FORMAT_R32_FLOAT,
        other => other,
    }
}

/// Whether typed buffer views of 32-bit integer formats are made in the float format of the
/// same width (`VICTORY_INT_VIEWS_AS_FLOAT=0`: the title's format, as before). Translated
/// shaders declare every typed buffer load and store a 32-bit float; through an integer view
/// a store of an integer's bits as a float may be written as the integer the float rounds to,
/// where the console moves the bits; the float format of the same width moves them as they
/// are. (fxceless/victory-fork PR #684 by GamingRipperDoc: plaster and clothing seams.) On
/// this machine's card a float load through an integer view already gave the bits.
fn int_views_as_float() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INT_VIEWS_AS_FLOAT").is_none_or(|v| v != "0"))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    desc.Format = bit_exact_format(desc.Format);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    desc.Format = bit_exact_format(desc.Format);
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format, except an RGBA8 shader view that
/// asks for sRGB or linear on a host that allows the cast
/// ([`keeps_gamma_view`]).
/// shortcut: other such views (BGRA8 sRGB of UNORM, say) still read
/// without their conversion.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !keeps_gamma_view(own, view, unordered)
    {
        *format = d.Format;
    }
    true
}

/// Whether a shader view in format `view` of a typed texture in format
/// `own` keeps its own format: an RGBA8 view that picks sRGB or linear,
/// on a host that allows views of typed textures in other formats of their
/// family. Replacing the format there dropped the conversion, so an sRGB
/// view of a UNORM texture read its texels as linear (#578).
/// Unordered access views have no sRGB on the host and stay as they were.
fn keeps_gamma_view(own: u32, view: u32, unordered: bool) -> bool {
    const RGBA8_UNORM: u32 = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
    const RGBA8_SRGB: u32 = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.0 as u32;
    let gamma_pair = matches!((own, view), (RGBA8_UNORM, RGBA8_SRGB) | (RGBA8_SRGB, RGBA8_UNORM));
    !unordered && gamma_pair && host().is_some_and(|gpu| gpu.typed_format_casting())
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The bytes of the host's constant buffer view for a title's of `size`:
/// whole 256s, and at most the 64 KiB a shader can index. The host removes
/// the device for a larger view, and the title's heap holds such sizes in
/// slots it does not use.
fn constant_bytes(size: u32) -> u32 {
    const LARGEST: u32 = 64 << 10;
    size.min(LARGEST).next_multiple_of(256)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_bytes(size),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if let Some(d) = desc.as_mut() {
                    shift_view_levels(crate::resource::dropped(resource), d);
                }
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            resource[0] != 1
                && crate::resource::placed_desc_key(resource)
                    == crate::resource::placed_desc_key(other)
                && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// A shader or unordered access view of a texture the title placed in
/// memory itself, as a view of the resource made for that memory (on first
/// use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    const TARGET_DEPTH_UNORDERED: u32 = 0x7;
    let &View::Placed(unordered, mut resource, ref desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let words = desc[..10].try_into().ok();
    if unordered {
        let texture = crate::resource::uav_texture_at(address, &resource, desc[0])?;
        Some(View::Unordered(texture as usize, 0, words))
    } else {
        // Shader views placed in memory take their texels either from title
        // memory or from `TARGETS_AT` (`resource::alias_of`): clear target
        // flags the title left on the resource description so this view
        // never replaces that target in `TARGETS_AT`.
        resource[12] &= !TARGET_DEPTH_UNORDERED;
        let texture = crate::resource::texture_at(address, &resource)?;
        Some(View::Shader(texture as usize, words))
    }
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Typed buffer views of integer formats carry their kind in the stride
    /// table; float, normalized, raw and structured ones none (upstream
    /// 4243a96a7's test, on our table).
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{SLOT_INTEGER, SLOT_SIGNED};
        if !victory_shader::emit::heap_half_bits() {
            return;
        }
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_stride(&uav(12, 0, 0)), Some(8 | SLOT_INTEGER), "R16G16B16A16_UINT");
        assert_eq!(view_stride(&uav(57, 0, 0)), Some(2 | SLOT_INTEGER), "R16_UINT");
        assert_eq!(view_stride(&uav(59, 0, 0)), Some(2 | SLOT_INTEGER | SLOT_SIGNED), "R16_SINT");
        assert_eq!(view_stride(&uav(64, 0, 0)), Some(1 | SLOT_INTEGER | SLOT_SIGNED), "R8_SINT");
        assert_eq!(view_stride(&uav(10, 0, 0)), Some(8), "R16G16B16A16_FLOAT");
        assert_eq!(view_stride(&uav(58, 0, 0)), Some(2), "R16_SNORM");
        assert_eq!(view_stride(&uav(39, 0, 1)), Some(1), "raw");
        assert_eq!(view_stride(&uav(0, 16, 0)), Some(16), "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_stride(&srv), Some(1 | SLOT_INTEGER), "R8_UINT");
        for v in [uav(12, 0, 0), uav(59, 0, 0), srv] {
            assert!(view_stride(&v).unwrap() & victory_shader::emit::SLOT_STRIDE_BITS <= 16);
        }
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // different stack-filler words [1] and [13] still name the same texture
        let mut with_filler = texture;
        with_filler[1] = 0xDEAD_BEEF;
        with_filler[13] = 0xCAFE_BABE;
        assert!(same_placed_texture(&low, &view(with_filler, 0, 0x1000_0000)));
        // Spans::bytes deduplicates descriptions differing only in filler words
        let mut spans = Spans::default();
        assert_eq!(spans.bytes(&texture), spans.bytes(&with_filler));
        assert_eq!(spans.sizes.len(), 1);
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    /// A constant buffer view larger than the host takes is cut to what it
    /// takes: the device, which such a view removes, stays.
    #[test]
    fn an_oversized_constant_buffer_view_leaves_the_device_alive() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // SAFETY: a plain device query.
        let alive = || unsafe { host.device.GetDeviceRemovedReason() }.is_ok();
        // Made first: an allocation that fails on a removed device must not
        // end the test early.
        let (buffer, _) = host.upload_buffer(1 << 20).expect("an upload buffer");
        let memory = vec![0u8; 2 << 20];
        let at = (memory.as_ptr() as u64).next_multiple_of(64 << 10);
        for size in [969_472u32, 896_256] {
            assert!(make(&View::Constant(at, size), heap.cpu(0)).is_some());
            assert!(alive(), "a view of {size} bytes");
        }
        // SAFETY: a live upload buffer.
        heap.constant_at(1, unsafe { buffer.GetGPUVirtualAddress() }, 969_472);
        assert!(alive(), "a view of the list's own memory");
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// Slots looked at before and slots never looked at are served in
    /// turns, one looked at before first (fxceless/victory-fork PR #1246).
    #[test]
    fn the_texture_queue_serves_looked_at_and_fresh_slots_in_turns() {
        let mut q = TextureQueue::default();
        for i in 0..10u32 {
            q.written(100 + i, 1, 0, false);
        }
        for i in 0..10u32 {
            let w = q.due_in_order(u64::MAX, 100, true).0.iter().find(|d| d.0 == 100 + i).unwrap().1;
            q.later(100 + i, w, 1, 1);
        }
        for i in 1..4u32 {
            q.written(i, 2, 0, false);
        }
        let order: Vec<u32> = q.due_in_order(10, 6, true).0.into_iter().map(|d| d.0).collect();
        assert_eq!(order.len(), 6);
        assert!(order[0] >= 100, "a slot looked at before comes first: {order:?}");
        assert_eq!(order.iter().filter(|&&i| i < 100).count(), 3, "every fresh slot is reached: {order:?}");
        // Short of one kind, the other fills the pass.
        assert_eq!(q.due_in_order(10, 12, true).0.len(), 12);
        // Off: all looked at before come first.
        let old: Vec<u32> = q.due_in_order(10, 6, false).0.into_iter().map(|d| d.0).collect();
        assert!(old.iter().all(|&i| i >= 100), "{old:?}");
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    /// `VICTORY_STALE_KEEP_MS`: off unless set and not 0; the cap is the
    /// second variable's, 120 s unless said, and never less than the first.
    #[test]
    fn stale_keep_is_opt_in_and_capped() {
        assert_eq!(stale_keep_from(None, None), None);
        assert_eq!(stale_keep_from(None, Some("5000")), None);
        assert_eq!(stale_keep_from(Some("0"), None), None);
        assert_eq!(stale_keep_from(Some("x"), None), None);
        assert_eq!(stale_keep_from(Some("10000"), None), Some(120_000));
        assert_eq!(stale_keep_from(Some(" 10000 "), Some("30000")), Some(30_000));
        assert_eq!(stale_keep_from(Some("10000"), Some("500")), Some(10_000));
    }

    /// A slot goes on showing a view from the first time it would have
    /// been made null until the cap, or until it is made again; a slot kept
    /// showing another view starts anew.
    #[test]
    fn a_slot_is_kept_past_its_view_for_a_time() {
        let mut kept = KeptPast::default();
        // First asked at 1,000 ms: kept, and counted once.
        assert_eq!(kept.keep(4, 0x1_0007, 1_000, 5_000, true), (true, true));
        assert_eq!(kept.keep(4, 0x1_0007, 3_000, 5_000, false), (true, false));
        assert_eq!(kept.keep(4, 0x1_0007, 5_999, 5_000, false), (true, false));
        // The cap: null, and no longer kept.
        assert_eq!(kept.keep(4, 0x1_0007, 6_000, 5_000, false), (false, false));
        assert!(!kept.forget(4));
        // Made again with its new texture before the cap.
        assert_eq!(kept.keep(5, 0x1_0008, 1_000, 5_000, true), (true, true));
        assert!(kept.forget(5));
        assert!(!kept.forget(5));
        // Kept showing one view, then asked for another: from that time.
        assert_eq!(kept.keep(6, 0x1_0001, 1_000, 5_000, false), (true, true));
        assert_eq!(kept.keep(6, 0x2_0001, 5_500, 5_000, false), (true, true));
        // Its memory was away; then the title wrote it with another
        // texture's view: the same time goes on, and it counts as rewritten.
        assert_eq!(kept.keep(6, 0x2_0001, 9_000, 5_000, true), (true, false));
        // The pass takes out those whose time is up; all, without room.
        assert_eq!(kept.keep(7, 0x1_0002, 9_000, 5_000, false), (true, true));
        assert_eq!(kept.expired(10_400, 5_000, false), []);
        assert_eq!(kept.expired(10_500, 5_000, false), [(6, 0x2_0001, true)]);
        assert_eq!(kept.expired(10_500, 5_000, true), [(7, 0x1_0002, false)]);
        assert!(kept.slots.is_empty());
    }

    /// A slot whose texture has both ends away is looked at between them
    /// once after a write or a change of its memory, then at every eighth
    /// look that found nothing, and always while memory is there only
    /// between its ends.
    #[test]
    fn the_look_between_a_textures_ends_is_made_in_turns() {
        use crate::texture::Between;
        let mut q = TextureQueue::default();
        q.written(3, 10, 0, false);
        let w = q.slots[&3].write;
        assert_eq!(q.between(3, w), Between::InBudget);
        assert_eq!(q.between(3, w + 1), Between::Not, "another write's");
        assert_eq!(q.between(4, 1), Between::Not, "not queued");
        // Put off for the frame's share: still owed at the next look.
        q.later(3, w, 10, 64);
        assert_eq!(q.between(3, w), Between::InBudget);
        // Made, and nothing there: not again until the eighth empty look.
        q.looked_between(3, w, false);
        for look in 2..=16u8 {
            q.later(3, w, 10, 64);
            assert_eq!(q.slots[&3].looks, look);
            let turn = if look % 8 == 0 { Between::InBudget } else { Between::Not };
            assert_eq!(q.between(3, w), turn, "look {look}");
        }
        // The runtime tells of memory mapped under it: owed again.
        q.changed(3, 40, 41);
        let w = q.slots[&3].write;
        assert_eq!(q.between(3, w), Between::InBudget);
        // Some is there between the ends: asked at every look from now on,
        // until a look finds none (streamed out again).
        q.looked_between(3, w, true);
        q.partly(3, w, 42, 65536);
        assert_eq!(q.between(3, w), Between::Always);
        q.looked_between(3, w, false);
        assert_eq!(q.between(3, w), Between::Not);
    }

    /// The age histogram's buckets: 0, 1, 2, 3-5, 6-15, 16-63, more.
    #[test]
    fn upload_ages_fall_into_the_histograms_buckets() {
        let buckets: Vec<usize> = [0, 1, 2, 3, 5, 6, 15, 16, 63, 64, u64::MAX].map(age_bucket).to_vec();
        assert_eq!(buckets, [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6]);
        assert_eq!(QUEUE_AGE.len(), 2);
        assert!(QUEUE_AGE.iter().all(|row| row.len() == 7));
        // All of a row, and the part within one frame of the write.
        assert_eq!(age_shares(&[5, 3, 2, 1, 0, 0, 9]), (20, 8));
        assert_eq!(age_shares(&[0; 7]), (0, 0));
    }

    /// Only the first upload since a slot's write is counted, with the
    /// frames since that write, and apart if a look found its memory away.
    #[test]
    fn the_first_upload_after_a_write_has_an_age() {
        let mut q = TextureQueue::default();
        q.written(7, 10, 0, false);
        q.written(8, 10, 0, false);
        q.written(9, 10, 0, false);
        let write = |q: &TextureQueue, index: u32| q.slots[&index].write;
        let (w7, w8, w9) = (write(&q, 7), write(&q, 8), write(&q, 9));
        // The same frame, and three frames later; not for another write.
        assert_eq!(q.first_upload(7, w7, 10), Some((0, false)));
        assert_eq!(q.first_upload(7, w7, 13), Some((3, false)));
        assert_eq!(q.first_upload(7, w7 + 100, 13), None);
        assert_eq!(q.first_upload(6, 1, 13), None);
        // Uploaded in part: the uploads of the rest are not first ones.
        q.partly(7, w7, 14, 4096);
        assert_eq!(q.first_upload(7, w7, 14), None);
        // A look found slot 8's memory unmapped; it arrives 20 frames on.
        q.away(8, w8);
        q.later(8, w8, 10, 4);
        assert_eq!(q.first_upload(8, w8, 30), Some((20, true)));
        // A wait for its host copy alone is the queue's own time.
        q.later(9, w9, 10, 1);
        assert_eq!(q.first_upload(9, w9, 11), Some((1, false)));
        // Uploaded whole and looked at again: an upload then is not a first.
        q.recheck(9, w9, 12, (0x1000, 64), 5, 0, 11);
        assert_eq!(q.first_upload(9, w9, 12), None);
        // Written again (or its memory changed): counted from that write.
        q.written(7, 20, 0, false);
        assert_eq!(q.first_upload(7, write(&q, 7), 22), Some((2, false)));
        q.changed(8, 40, 41);
        assert_eq!(q.first_upload(8, write(&q, 8), 41), Some((1, false)));
    }

    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).0.into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).0.len(), 2);
        assert_eq!(q.due(9, 0).1, [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| {
            q.due(u64::MAX, 100).0.into_iter().find(|d| d.0 == index).unwrap().1
        };
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).0.iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100).0[0].0, 17);
        }
        assert_eq!(q.due(9, 0).1[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).0.iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100).0[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100).0[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).0.iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).0.iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).0.iter().any(|d| d.0 == 98));
    }

    #[test]
    fn a_limited_texture_pass_keeps_the_full_queue_priority_and_counts() {
        let mut q = TextureQueue::default();
        for (frame, slot) in [3, 12, 5, 8, 11, 2, 10, 6, 7, 4, 1, 9].into_iter().enumerate() {
            q.written(slot, frame as u64, 0, false);
        }
        q.later(5, 3, 11, 4);
        q.partly(11, 5, 12, 4096);
        q.later(6, 8, 11, 4);
        q.partly(10, 7, 13, 4096);
        q.written(4, 11, 14, false);

        // Three retries first, newest write first. The remaining seven
        // alternate between newest and oldest, with no slot repeated.
        let priority = [
            (6, 8),
            (11, 5),
            (5, 3),
            (9, 12),
            (3, 1),
            (1, 11),
            (12, 2),
            (7, 9),
            (8, 4),
            (2, 6),
        ];
        for max in 0..=priority.len() + 1 {
            let (ready, counts) = q.due_in_order(12, max, false);
            assert_eq!(ready, priority[..max.min(priority.len())], "max={max}");
            assert_eq!(counts, [12, 10, 2, 2, 12], "max={max}");
        }
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// An RGBA8 texture read through an sRGB or linear view gets that
    /// view's conversion, whichever of the two the texture itself is (#578).
    /// Without the cast on the host, the view keeps the texture's format, as
    /// before. Samples one texel of 0x80 on the GPU.
    #[test]
    fn rgba8_views_keep_their_srgb_conversion() {
        const UNORM: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
        const SRGB: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        // 0x80 read as linear, and as sRGB converted to linear.
        const LINEAR_VALUE: f32 = 128.0 / 255.0;
        const SRGB_VALUE: f32 = 0.215_860_5;
        let Some(gpu) = host() else {
            return; // no host GPU
        };
        let casting = gpu.typed_format_casting();
        for (texture_format, view_format) in [(UNORM, UNORM), (UNORM, SRGB), (SRGB, UNORM), (SRGB, SRGB)] {
            let read_as = if casting { view_format } else { texture_format };
            let expected = if read_as == SRGB { SRGB_VALUE } else { LINEAR_VALUE };
            let red = sample_one_texel(&gpu.device, texture_format, view_format).expect("the sample ran");
            assert!(
                (red - expected).abs() < 0.002,
                "texture {} viewed as {}: red {red}, expected {expected}",
                texture_format.0,
                view_format.0
            );
        }
    }

    /// Fills a 1x1 texture in `texture_format` with 0x80 in every channel,
    /// makes a shader view of it in `view_format` the way the title's views
    /// are made ([`fits`]), and returns the red channel a compute shader
    /// reads through that view.
    fn sample_one_texel(
        device: &ID3D12Device,
        texture_format: DXGI_FORMAT,
        view_format: DXGI_FORMAT,
    ) -> windows::core::Result<f32> {
        const SHADER: &str = "#define RS \"DescriptorTable(SRV(t0)), UAV(u0)\"\n\
            Texture2D<float4> texel : register(t0);\n\
            RWByteAddressBuffer result : register(u0);\n\
            [RootSignature(RS)]\n\
            [numthreads(1, 1, 1)]\n\
            void main() { result.Store(0, asuint(texel.Load(int3(0, 0, 0)).r)); }\n";
        // Rows of a texture copy are 256 bytes apart.
        const ROW_PITCH: u64 = 256;
        let mut rig = crate::inplace::Rig::new(device)?;
        let default_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: texture_format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let result_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        let mut result: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of resources of our own.
        unsafe {
            device.CreateCommittedResource(
                &default_heap,
                D3D12_HEAP_FLAG_NONE,
                &texture_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut texture,
            )?;
            device.CreateCommittedResource(
                &default_heap,
                D3D12_HEAP_FLAG_NONE,
                &result_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut result,
            )?;
        }
        let texture = texture.ok_or_else(windows::core::Error::empty)?;
        let result = result.ok_or_else(windows::core::Error::empty)?;

        // The texel's bytes, uploaded through a buffer.
        let upload = rig.committed(D3D12_HEAP_TYPE_UPLOAD, ROW_PITCH, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            upload.Map(0, None, Some(&mut mapped))?;
            mapped.cast::<u8>().write_bytes(0x80, 4);
            upload.Unmap(0, None);
        }

        // The view, made as the title's are: `fits` picks its format.
        let mut view_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: view_format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        assert!(fits(&texture, Some(&mut view_desc.Format), false), "the host allows the view");
        // SAFETY: plain creation; the view names a live texture.
        let view_heap: ID3D12DescriptorHeap = unsafe {
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            device.CreateShaderResourceView(&texture, Some(&view_desc), heap.GetCPUDescriptorHandleForHeapStart());
            heap
        };

        let code = crate::host::compile(SHADER, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
        // SAFETY: the compiled bytes carry the root signature too.
        let (root, pipeline): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
            let pipeline = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            })?;
            (root, pipeline)
        };

        // Upload the texel, then let the shader read it.
        // SAFETY: an open list of ours; every resource named is alive.
        rig.run(|list| unsafe {
            let destination = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: texture_format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: ROW_PITCH as u32,
                        },
                    },
                },
            };
            list.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
            list.ResourceBarrier(&[crate::host::transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            list.SetDescriptorHeaps(&[Some(view_heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pipeline);
            list.SetComputeRootDescriptorTable(0, view_heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRootUnorderedAccessView(1, result.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
        })?;
        let bytes = rig.fetch(&result, 0, 4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    /// Typed buffer views of 32-bit integer formats are made in the float format of the same
    /// width (the shaders' float loads and stores then move the bits); every other format is
    /// the title's. (fxceless/victory-fork PR #684.)
    #[test]
    fn typed_buffer_views_of_32_bit_integers_move_bits() {
        for (title, host) in [
            (DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_FLOAT),
            (DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32_FLOAT),
            (DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT),
        ] {
            assert_eq!(bit_exact_format(title), host);
            assert_eq!(
                crate::resource::element(title.0 as u32),
                crate::resource::element(host.0 as u32)
            );
        }
        for kept in [
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_TYPELESS,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_R8G8B8A8_UNORM,
        ] {
            assert_eq!(bit_exact_format(kept), kept);
        }
    }

    /// Small textures packed a few hundred bytes apart (the title's tint palettes, 256 bytes
    /// to 4 KiB) are neighbours, not replacements, when a texture's own first bytes are its
    /// size: the view written next makes none of them stale. A view at the same start, or
    /// one starting within a larger texture's first 4 KiB, still does; with 4 KiB for every
    /// texture (`VICTORY_STALE_HEAD_SIZE=0`) each palette made the one before it stale.
    /// (fxceless/victory-fork #923.)
    #[test]
    fn small_palettes_a_few_hundred_bytes_apart_are_neighbours() {
        let at = 0x1_2340_0000u64;
        for head_size in [true, false] {
            let mut spans = Spans { head_size, ..Spans::default() };
            assert!(spans.note(1, at, at + 0x100, true).is_empty());
            let second = spans.note(2, at + 0x300, at + 0x400, true);
            let third = spans.note(3, at + 0x600, at + 0x1600, true);
            if head_size {
                assert!(second.is_empty() && third.is_empty(), "palettes are neighbours");
                // inside the 4 KiB palette at 0x600: a replacement
                assert_eq!(spans.note(4, at + 0x700, at + 0x800, true), [3]);
                // the same start with another length: a replacement
                assert_eq!(spans.note(5, at, at + 0x200, true), [1]);
            } else {
                assert_eq!((second, third), (vec![1], vec![2]), "each made the one before stale");
            }
        }
    }

    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// `VICTORY_HEAP_WORKER=1`'s hand-off (`crate::heap_worker`): a slot
    /// whose texture the worker uploaded is made when it is published, and
    /// does not go back into the queue of textures picked by index for that,
    /// nor when published twice; one the title wrote again after the worker
    /// looked at it is refused, goes on showing what it showed, and is due
    /// again for the new write.
    #[test]
    fn a_slot_uploaded_ahead_is_made_once_and_a_rewrite_is_looked_at_again() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let due = || -> Vec<u32> { heap.texture_slots_due(u64::MAX, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at addresses no test and no title uses.
        let view = |at: u64| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let title = [0u32; 8];
        let handle = title.as_ptr() as u64;
        crate::descriptor::record_for_test(handle, view(0x7fec_0000_0000));
        let (first, key) = crate::descriptor::view_and_key(handle).unwrap();
        assert!(heap.note_buffer(1, Some(&first), false).is_empty());
        let (index, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(index, 1);
        // The worker recorded its upload: out of the queue, not made yet.
        heap.texture_slot_done(1, write, None);
        assert_eq!(heap.made_key(1), 0);
        assert_eq!(due(), [] as [u32; 0]);
        // Its list is handed to a submission: made, and not queued again.
        assert!(heap.publish_ahead(1, handle, key));
        assert_eq!(heap.made_key(1), key);
        assert_eq!(due(), [] as [u32; 0]);
        assert!(heap.publish_ahead(1, handle, key));
        assert_eq!(due(), [] as [u32; 0]);
        // Written again after the worker looked at it, before its list was
        // handed over: refused, the slot shows what it did, the write is due.
        crate::descriptor::record_for_test(handle, view(0x7feb_0000_0000));
        let (second, again) = crate::descriptor::view_and_key(handle).unwrap();
        assert_ne!(again, key);
        let _ = heap.note_buffer(1, Some(&second), false);
        assert!(!heap.publish_ahead(1, handle, key));
        assert_eq!(heap.made_key(1), key);
        assert_eq!(due(), [1]);
        // Its own turn: looked at, uploaded, published.
        let (_, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        heap.texture_slot_done(1, write, None);
        assert!(heap.publish_ahead(1, handle, again));
        assert_eq!(heap.made_key(1), again);
        assert_eq!(due(), [] as [u32; 0]);
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [1]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    /// (fxceless/victory-fork 4d0c63455.)
    #[test]
    fn a_free_run_across_the_place_the_search_starts_from_is_found() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Slots 0..8 and 8..16 of the ring: the search now starts at 16.
        assert_eq!((heap.dynamic(8), heap.dynamic(8)), (Some(4), Some(12)));
        // Everything after them, so that only what is given back is free.
        let rest = heap.dynamic - 16;
        assert_eq!(heap.dynamic(rest), Some(4 + 16));
        // The search starts at 0 again. Slots 0..16 back, then 0..12 taken:
        // it starts at 12, with 12..16 free.
        heap.release(4, 16);
        assert_eq!(heap.dynamic(12), Some(4));
        // 10..12 back: the free run 10..16 lies across 12. Six slots fit
        // only there.
        heap.release(4 + 10, 2);
        assert_eq!(heap.dynamic(6), Some(4 + 10));
        // And nothing is found where nothing is.
        assert_eq!(heap.dynamic(1), None);
    }

    /// The search itself: a run across the place it starts from is found
    /// only by the search that goes on past that place; the search as it
    /// was (`VICTORY_DYNAMIC_ACROSS=0`) finds what it found before, and
    /// neither finds a run that wraps past the end.
    #[test]
    fn the_search_goes_on_past_its_starting_place() {
        // Free: 10..16 of 32. The search starts at 12.
        let mut taken = [true; 32];
        taken[10..16].fill(false);
        assert_eq!(free_run(&taken, 12, 6, true), Some((10, true)));
        assert_eq!(free_run(&taken, 12, 6, false), None);
        // A run that does not need the slots before 12: found either way,
        // in the first part.
        assert_eq!(free_run(&taken, 12, 4, true), Some((12, false)));
        assert_eq!(free_run(&taken, 12, 4, false), Some((12, false)));
        // Longer than what is free: no run.
        assert_eq!(free_run(&taken, 12, 7, true), None);
        // Free at both ends: no run wraps past the end.
        let mut ends = [true; 32];
        ends[..3].fill(false);
        ends[29..].fill(false);
        assert_eq!(free_run(&ends, 30, 4, true), None);
        assert_eq!(free_run(&ends, 30, 3, true), Some((0, false)));
        assert_eq!(free_run(&ends, 1, 3, true), Some((29, false)));
        // From the start and from the end of the ring.
        let free = [false; 8];
        assert_eq!(free_run(&free, 0, 8, true), Some((0, false)));
        assert_eq!(free_run(&free, 0, 8, false), Some((0, false)));
        assert_eq!(free_run(&free, 5, 8, true), Some((0, true)));
        assert_eq!(free_run(&free, 5, 8, false), None);
        assert_eq!(free_run(&free, 8, 1, true), Some((0, false)));
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// The title's view heap on NVIDIA's 1,000,000 views with a
    /// heap map: title slots below 131,072 at their own index, 606,782 pool
    /// slots, the map at 737,854, and the fork's dynamic slots (from
    /// 737,855) and guard slot (999,999, the translator's limit) where they
    /// were. A window past the room is cut to it; the pool holds no more
    /// slots than the title has past the window.
    #[test]
    fn the_titles_view_heap_is_laid_out_in_a_million_slots() {
        let l = MapLayout::new(1_245_184, WINDOW, MAP_SLOT, DYNAMIC);
        assert_eq!((l.window, l.pool, l.slot), (131_072, 606_782, 737_854));
        assert_eq!(l.slot_in_dynamic(), None);
        assert_eq!(l.dynamic_first(), PORTABLE_VIEWS - DYNAMIC - 1, "the fork's first dynamic slot");
        assert_eq!(l.guard(), PORTABLE_VIEWS - 1, "the fork's guard slot");
        assert_eq!(l.size(), PORTABLE_VIEWS);
        let wide = MapLayout::new(1_245_184, 900_000, MAP_SLOT, DYNAMIC);
        assert_eq!((wide.window, wide.pool), (MAP_SLOT, 0));
        let small = MapLayout::new(20, 16, 24, DYNAMIC);
        assert_eq!((small.window, small.pool, small.dynamic_first()), (16, 4, 25));
        assert_eq!(small.size(), 25 + DYNAMIC + 1);
    }

    /// A title heap too big for a capped host heap (NVIDIA: the guard slot
    /// at 999,999) keeps 393,215 slots (the heap map's window and 262,143
    /// pool slots) unless `VICTORY_DYNAMIC_SLOTS` says how many are
    /// dynamic: no fewer than [`DYNAMIC`] (the layout as it was), and
    /// leaving the pool [`LEAST_TITLE`]. Without a heap map it keeps
    /// [`DYNAMIC`] unless told. The map's slot and the guard stay where the
    /// translations read them; the map's slot, then among the dynamic
    /// slots, is not one to hand out.
    #[test]
    fn a_capped_heap_gives_pool_slots_to_dynamic_ones() {
        let guard = PORTABLE_VIEWS - 1;
        let mapped = |set| capped_dynamic_slots(guard, Some(WINDOW), set);
        assert_eq!(mapped(None), 606_784);
        assert_eq!(mapped(Some(DYNAMIC)), DYNAMIC);
        assert_eq!(mapped(Some(500_000)), 500_000);
        assert_eq!(mapped(Some(10)), DYNAMIC);
        assert_eq!(mapped(Some(u32::MAX)), guard - WINDOW - LEAST_TITLE);
        assert_eq!(capped_dynamic_slots(guard, None, None), DYNAMIC);
        assert_eq!(capped_dynamic_slots(guard, None, Some(606_784)), 606_784);
        assert_eq!(capped_dynamic_slots(guard, None, Some(u32::MAX)), guard - LEAST_TITLE);
        // A small heap's layout (tests) has no room for more.
        assert_eq!(capped_dynamic_slots(25 + DYNAMIC, Some(16), None), DYNAMIC);

        let l = MapLayout::new(1_245_184, WINDOW, MAP_SLOT, mapped(None));
        assert_eq!((l.window, l.pool, l.slot, l.dynamic), (131_072, 262_143, 737_854, 606_784));
        assert_eq!(l.window + l.pool, CAPPED_TITLE);
        assert_eq!(l.dynamic_first(), 393_215);
        assert_eq!(l.slot_in_dynamic(), Some(737_854 - 393_215));
        assert_eq!((l.guard(), l.size()), (PORTABLE_VIEWS - 1, PORTABLE_VIEWS));
        // As it was, with `VICTORY_DYNAMIC_SLOTS=262144`.
        let old = MapLayout::new(1_245_184, WINDOW, MAP_SLOT, mapped(Some(DYNAMIC)));
        assert_eq!((old.pool, old.dynamic_first(), old.slot_in_dynamic()), (606_782, 737_855, None));
        // Never into the window.
        let all = MapLayout::new(1_245_184, WINDOW, MAP_SLOT, u32::MAX);
        assert_eq!((all.pool, all.dynamic_first()), (0, WINDOW));
    }

    /// A heap with more dynamic slots than [`DYNAMIC`] hands out the ones
    /// below the map's slot and the ones above it, never the map's slot,
    /// and no run across it.
    #[test]
    fn dynamic_slots_around_the_map_slot_are_handed_out() {
        if host().is_none() {
            return;
        }
        // Window 16, map at 40, 12 dynamic slots more than the fork's:
        // pool 16..29, dynamic 29..41+DYNAMIC, the map's slot the 12th.
        let l = MapLayout::new(64, 16, 40, DYNAMIC + 12);
        assert_eq!((l.pool, l.dynamic_first(), l.slot_in_dynamic()), (13, 29, Some(11)));
        let Some(heap) = HostHeap::laid_out(false, 64, Some(l)) else {
            return;
        };
        assert_eq!((heap.count(), heap.dynamic_first, heap.dynamic), (16, 29, DYNAMIC + 12));
        assert!(heap.kept.lock().unwrap()[40].is_some(), "the map's raw view holds it");
        // 11 below the map's slot, then past it: no run lies across it.
        assert_eq!(heap.dynamic(8), Some(29));
        assert_eq!(heap.dynamic(8), Some(41));
        assert_eq!(heap.dynamic(3), Some(49));
        assert_eq!(heap.dynamic(DYNAMIC - 11), Some(52));
        // What is left is the three below the map's slot.
        assert_eq!(heap.dynamic(4), None);
        assert_eq!(heap.dynamic(3), Some(37));
        assert_eq!(heap.dynamic(1), None);
        // Three free below the map's slot and eight above it: no run of
        // nine, though eleven are free.
        heap.release(37, 3);
        heap.release(41, 8);
        assert_eq!(heap.dynamic(9), None);
        assert_eq!(heap.dynamic(8), Some(41));
        assert_eq!(heap.dynamic(3), Some(37));
        assert_eq!(heap.dynamic(1), None);
    }

    /// A view heap made while the translator has a heap map (as a capped
    /// host's `probe_view_limit` sets it) is laid out as [`MapLayout`] says,
    /// its map's raw view at the map slot; one no larger than the window
    /// keeps the fork's layout.
    #[test]
    fn a_view_heap_with_a_heap_map_keeps_the_dynamic_slots_after_the_map() {
        if host().is_none() {
            return;
        }
        TEST_MAP.set(Some((16, 24)));
        let mapped = HostHeap::new(false, 64);
        let fits = HostHeap::new(false, 8);
        TEST_MAP.set(None);
        let (mapped, fits) = (mapped.unwrap(), fits.unwrap());
        assert_eq!((mapped.count(), mapped.title_count, mapped.dynamic_first), (16, 64, 25));
        assert!(mapped.map.is_some());
        {
            let made = mapped.made.lock().unwrap();
            assert_eq!(made.keys.len(), 16 + 8, "the window's and the pool's");
            assert_eq!(made.map.as_ref().map(SlotMap::len), Some(48));
        }
        assert_eq!(mapped.tabled.len(), 64, "by title slot");
        assert_eq!(mapped.kept.lock().unwrap().len(), (25 + DYNAMIC) as usize);
        assert!(mapped.kept.lock().unwrap()[24].is_some(), "the map's raw view holds it");
        assert_eq!(mapped.dynamic(DYNAMIC), Some(25));
        assert_eq!(mapped.dynamic(1), None);
        assert_eq!((fits.count(), fits.dynamic_first), (8, 8));
        assert!(fits.map.is_none() && fits.made.lock().unwrap().map.is_none());
        assert_eq!(fits.dynamic(1), Some(8));
    }

    /// A title view heap of `count` slots and its first descriptor, made
    /// as the title makes them (its host heap is made at first use).
    fn title_heap(count: u32) -> (crate::object::This, u64) {
        #[repr(C)]
        struct Extended(super::super::HeapDesc, super::super::Placement);
        let desc = super::super::HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        };
        let none = super::super::Placement { address: 0, size: 0 };
        let extended = Extended(desc, none);
        let mut out = std::ptr::null();
        let made = super::super::create_heap(
            std::ptr::null(),
            &extended.0,
            &super::super::HEAP_IIDS[0],
            &mut out,
        );
        assert_eq!(made, crate::object::S_OK);
        (out, super::super::heap(out).start)
    }

    /// Writes a constant buffer view of `address` into title slot `slot`.
    fn title_cbv(start: u64, slot: u32, address: u64) {
        super::super::create_cbv(
            std::ptr::null(),
            &[address as u32, (address >> 32) as u32, 256, 0],
            start + u64::from(slot) * TITLE_SLOT_BYTES,
        );
    }

    fn release(object: crate::object::This) -> u32 {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object)
    }

    /// The heap map entry of title slot `title`, and its pool slot.
    fn entry(heap: &HostHeap, title: u32) -> (u32, Option<u32>) {
        let window = heap.count();
        let at = heap.map.as_ref().unwrap().at;
        // SAFETY: an entry of the mapped heap map.
        let entry = unsafe { (at as *const u32).add((title - window) as usize).read_volatile() };
        let pool = heap.made.lock().unwrap().map.as_ref().unwrap().host(title);
        (entry, pool)
    }

    /// Title slots past the identity window that the title writes take
    /// pool slots, published in the heap map (entries hold the slot's
    /// bitwise not) once made; slots below it keep their index. A slot the
    /// title writes again keeps its pool slot; one it empties gives it back
    /// (entry zeroed: the guard), quarantined while submissions run.
    #[test]
    fn title_slots_past_the_window_take_pool_slots() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        title_cbv(start, 3, 0x1_0000);
        title_cbv(start, 40, 0x2_0000);
        TEST_MAP.set(Some((16, 24)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        assert_eq!(heap.count(), 16);
        let (e40, pooled) = entry(heap, 40);
        let pooled = pooled.expect("a pool slot");
        assert!((16..24).contains(&pooled), "{pooled}");
        assert_eq!(e40, !pooled);
        assert_eq!(entry(heap, 41), (0, None));
        assert_eq!(heap.made.lock().unwrap().host(3), Some(3), "below the window: its own");
        assert_eq!(heap.title_start.load(std::sync::atomic::Ordering::Relaxed), start);

        // Written again: the same pool slot. Another slot: another one.
        title_cbv(start, 40, 0x3_0000);
        assert_eq!(entry(heap, 40), (!pooled, Some(pooled)));
        title_cbv(start, 63, 0x3_0000);
        let other = entry(heap, 63).1.expect("a pool slot");
        assert_ne!(other, pooled);

        // Emptied (as a write of no view finds it): the entry reads the
        // guard and the pool slot waits.
        let handle = start + 40 * TITLE_SLOT_BYTES;
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (handle as *mut u8).write_bytes(0, 32) };
        assert!(heap.mirror(40, handle, false).is_none());
        assert_eq!(entry(heap, 40), (0, None));
        let stats = heap.made.lock().unwrap().map.as_ref().unwrap().stats();
        assert_eq!((stats.mapped, stats.quarantined), (1, 1));
        assert_eq!(release(title), 0);
    }

    /// With the pool used up, the pool slots of title slots that hold no
    /// live view any more (overwritten as bytes: no descriptor write says
    /// so) are given back, and the slot that asked takes one.
    #[test]
    fn a_used_up_pool_gives_back_slots_without_views() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        TEST_MAP.set(Some((16, 20)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        for slot in 16..20 {
            title_cbv(start, slot, 0x1_0000 + u64::from(slot) * 0x100);
        }
        let freed = entry(heap, 17).1.expect("a pool slot");
        // SAFETY: a descriptor slot of the title heap.
        unsafe { ((start + 17 * TITLE_SLOT_BYTES) as *mut u8).write_bytes(0, 32) };
        title_cbv(start, 30, 0x9_0000);
        assert_eq!(entry(heap, 17), (0, None), "given back");
        assert!(entry(heap, 16).1.is_some() && entry(heap, 19).1.is_some());
        // Handed out once the submissions running when it was given back
        // have finished (other tests' may be).
        let (running, _) = crate::host::submissions();
        let waited = std::time::Instant::now();
        while entry(heap, 30).1.is_none() && waited.elapsed() < std::time::Duration::from_secs(20) {
            if crate::host::submissions().1 >= running {
                let _ = heap.mirror(30, start + 30 * TITLE_SLOT_BYTES, false);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        assert_eq!(entry(heap, 30), (!freed, Some(freed)));
        assert_eq!(release(title), 0);
    }

    /// Placed unordered texture views (2D and 3D) mirror onto host unordered
    /// access textures when written into a shader-visible heap, reuse an
    /// existing target at the same address, and feed shader views placed at
    /// that address in another format through `texture::alias`.
    #[test]
    fn placed_unordered_texture_view_mirrors_and_aliases_to_shader_view() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 8) else {
            return;
        };

        // 1. 3D volume UAV (4x4x4 R32_UINT, resource flags = 0) mirrored via
        // `HostHeap::mirror(..., for_draw = false)` as a shader-visible heap
        // copy does.
        let vol_at = 0x7fed_1000_0000u64;
        let vol_res = [4, 0, 0, 0, 4, 0, 4, 0x1_0004, 42, 1, 0, 0, 0, 0];
        let mut vol_uav = [0u32; 16];
        (vol_uav[0], vol_uav[1], vol_uav[4]) = (42, 8, 4);
        (vol_uav[10], vol_uav[11]) = (vol_at as u32, (vol_at >> 32) as u32);
        let slot_mem = [0u64; 4];
        let slot_handle = slot_mem.as_ptr() as u64;
        super::super::create_placed_uav(std::ptr::null(), &vol_res, &vol_uav, slot_handle);
        assert!(heap.mirror(0, slot_handle, false).is_some());
        {
            let kept = heap.kept.lock().unwrap();
            let tex = kept[0].as_ref().expect("3D placed UAV must mirror immediately");
            // SAFETY: live D3D12 resource.
            let d = unsafe { tex.GetDesc() };
            assert_eq!(d.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE3D);
            assert_ne!(d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0, 0);
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // Another placed UAV at the same address with compatible format reuses
        // the resource already recorded in `TARGETS_AT`.
        let Some(View::Unordered(first_vol, _, _)) =
            placed_texture(&View::Placed(true, vol_res, vol_uav))
        else {
            panic!("expected View::Unordered");
        };
        let mut explicit_flags = vol_res;
        explicit_flags[12] = 0x4;
        let Some(View::Unordered(second_vol, _, _)) =
            placed_texture(&View::Placed(true, explicit_flags, vol_uav))
        else {
            panic!("expected View::Unordered");
        };
        assert_eq!(first_vol, second_vol, "must reuse existing UAV texture at address");

        // 2. Cross-format aliasing: a 4x4 R9G9B9E5_SHAREDEXP (67) description
        // written through an R32_UINT (42) placed UAV and then read through an
        // R9G9B9E5_SHAREDEXP placed SRV (even if `ALLOW_UNORDERED_ACCESS` was
        // left on the SRV's resource description) copies the GPU texels across.
        let alias_at = 0x7fed_2000_0000u64;
        let rgb9e5_res = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 67, 1, 0, 0, 0x4, 0];
        let mut uav_desc = [0u32; 16];
        (uav_desc[0], uav_desc[1]) = (42, 4);
        (uav_desc[10], uav_desc[11]) = (alias_at as u32, (alias_at >> 32) as u32);
        let uav_placed = View::Placed(true, rgb9e5_res, uav_desc);
        let Some(View::Unordered(uav_obj, _, _)) = placed_texture(&uav_placed) else {
            panic!("expected placed UAV texture");
        };
        assert!(make(&uav_placed, heap.cpu(1)).flatten().is_some());

        let mut srv_words = [0u32; 16];
        (srv_words[0], srv_words[1], srv_words[2], srv_words[5]) =
            (67, 4, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 1);
        (srv_words[10], srv_words[11]) = (alias_at as u32, (alias_at >> 32) as u32);
        let srv_placed = View::Placed(false, rgb9e5_res, srv_words);
        let Some(View::Shader(srv_obj, _)) = placed_texture(&srv_placed) else {
            panic!("expected placed SRV texture");
        };
        assert!(make(&srv_placed, heap.cpu(2)).flatten().is_some());
        assert_eq!(
            crate::resource::alias_of(srv_obj as crate::object::This),
            Some(uav_obj as crate::object::This),
            "placed SRV must find the placed UAV texture in TARGETS_AT"
        );

        // Upload a known R32_UINT pattern into `uav_obj` on the GPU, run
        // `texture::alias` into `srv_obj`, and read `srv_obj` back.
        let uav_tex = crate::resource::host_resource(uav_obj as crate::object::This).unwrap();
        let srv_tex = crate::resource::host_resource(srv_obj as crate::object::This).unwrap();
        let pattern = 0x7c40_8210u32;
        let make_buf = |heap_type, state| {
            let hp = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, None, &mut out)
                    .unwrap();
            }
            out.unwrap()
        };
        let upload = make_buf(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let readback = make_buf(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY: 1024-byte upload buffer mapped on CPU.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut ptr)).unwrap();
            std::slice::from_raw_parts_mut(ptr.cast::<u32>(), 256).fill(pattern);
            upload.Unmap(0, None);
        }

        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, uav_obj as crate::object::This, 0x400); // COPY_DEST
        let buf_loc = |buf: &ID3D12Resource, format| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer not released by the copy location.
            pResource: unsafe { std::mem::transmute_copy(buf) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: 4,
                        Height: 4,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        let tex_loc = |tex: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer not released by the copy location.
            pResource: unsafe { std::mem::transmute_copy(tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: open list and live resources in matching copy states.
        unsafe {
            list.list.CopyTextureRegion(
                &tex_loc(uav_tex),
                0,
                0,
                0,
                &buf_loc(&upload, DXGI_FORMAT_R32_UINT),
                None,
            );
        }
        crate::list::ensure_state(&list, uav_obj as crate::object::This, 0x8); // UNORDERED_ACCESS
        assert!(crate::texture::alias(
            &list,
            srv_obj as crate::object::This,
            uav_obj as crate::object::This
        ));
        crate::list::ensure_state(&list, srv_obj as crate::object::This, 0x800); // COPY_SOURCE
        // SAFETY: open list and `srv_tex` transitioned to `COPY_SOURCE`.
        unsafe {
            list.list.CopyTextureRegion(
                &buf_loc(&readback, DXGI_FORMAT_R9G9B9E5_SHAREDEXP),
                0,
                0,
                0,
                &tex_loc(srv_tex),
                None,
            );
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: GPU execution finished; readback holds 1024 bytes.
        let got = unsafe {
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            let word = *ptr.cast::<u32>();
            readback.Unmap(0, None);
            word
        };
        assert_eq!(got, pattern);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
