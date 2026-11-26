//! Host copies into textures the title also fills, kept from an upload of
//! title memory that lands after them.
//!
//! The title makes some textures on the card: its character bake draws a
//! character's decorations, compresses them with a compute program into an
//! integer texture and copies that, level by level, into a plain
//! block-compressed texture (256x256 to 2048x2048, formats 77 and 78) that
//! nothing ever writes in title memory. To the host such a texture is one
//! "the title fills" until the first copy into it is recorded
//! (`crate::texture::host_written`), and the pass over the heap's views
//! uploads title memory into it as soon as its view is written
//! (`crate::draw`'s `heap_textures`): whatever that memory held before.
//!
//! Lists are recorded on several threads and run in the order the title
//! submits them. An upload recorded in one list before the copy was recorded
//! in another runs after the copy when its list is submitted after the
//! copy's: the baked texels are overwritten with the old bytes of title
//! memory, and nothing uploads or copies again (the frame listings of
//! all15b-on2 frame 923 and cx1-water-on1 frame 1160 show it: the bake's list
//! third of the submission, a list with an upload of the same texture sixth).
//! Which list comes first changes from run to run: a character's shirt was
//! its print in one run and coloured squares in the next.
//!
//! `VICTORY_COPY_KEEP=1` (off by default): the first host copy into a
//! texture the title fills that was asked to be uploaded within the last
//! [`WINDOW`] frames makes a second host texture of the same description
//! (the keeper); that copy and every later one into the texture is recorded
//! into the keeper too, and [`RESTORES`] frames later, when every list
//! recorded around the copy has run, the subresources copied whole are
//! copied from the keeper into the texture again ([`restore_due`]). The
//! keeper then goes.
//!
//! `VICTORY_TEXTURE_TRACE=<n>` (off by default): up to `n` lines of each
//! kind that name such textures: a first host copy into a texture asked to
//! be uploaded just before, a list submitted with an upload into a texture
//! another list has copied into since, and a restore.
//!
//! shortcut: by frames, not by which lists have run; a subresource copied
//! only in part is not restored (it keeps what the last upload or copy left).

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;

/// Frames after a texture was asked to be uploaded from title memory in
/// which a first host copy into it may be followed by that upload: lists
/// recorded in one frame are submitted in it or the next.
const WINDOW: u64 = 4;

/// Frames after the first copy at which the texture is copied from its
/// keeper again: once when the lists recorded around the copy have run, and
/// once more for a list held back longer.
const RESTORES: [u64; 2] = [6, 30];

/// `VICTORY_COPY_KEEP=1`.
pub(crate) fn keeping() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_KEEP").is_some_and(|v| v == "1"))
}

/// `VICTORY_TEXTURE_TRACE=<n>`: the most lines of each kind (0: none).
pub(crate) fn trace() -> u64 {
    static LINES: OnceLock<u64> = OnceLock::new();
    *LINES.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_TRACE").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
    })
}

/// Whether anything here is on: off, a copy pays one look at this.
pub(crate) fn active() -> bool {
    keeping() || trace() > 0
}

/// Whether one more trace line of kind `kind` may be written.
fn line(kind: usize) -> bool {
    static SAID: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    let most = trace();
    most > 0 && SAID[kind].fetch_add(1, Relaxed) < most
}

/// The frame each texture was last asked to be uploaded from title memory
/// in ([`asked`]), by title texture.
static ASKED: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// The list that recorded the first host copy into each texture, for the
/// trace ([`upload_submitted`]).
static COPIED_BY: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// A texture's keeper.
struct Kept {
    /// A host texture of the title texture's description, in the copy
    /// destination state between lists.
    texture: ID3D12Resource,
    /// The title texture's host copy the keeper was made for.
    host: usize,
    /// The subresources copied whole so far.
    whole: BTreeSet<u32>,
    /// The frame of the first copy.
    made: u64,
    /// How many of [`RESTORES`] are done.
    restored: usize,
}

// SAFETY: a device-local texture only lists record copies with.
unsafe impl Send for Kept {}

/// The keepers, by title texture.
static KEPT: Mutex<Option<HashMap<usize, Kept>>> = Mutex::new(None);

/// How many keepers there are: none on almost every call.
static KEPT_NOW: AtomicU64 = AtomicU64::new(0);

/// Since the start: keepers made, restores recorded.
static KEEPERS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// How many frames ago a texture asked to be uploaded in frame `noted` was
/// asked, if a first host copy into it in `frame` may still be followed by
/// that upload.
fn at_risk(noted: Option<u64>, frame: u64) -> Option<u64> {
    let ago = frame.checked_sub(noted?)?;
    (ago <= WINDOW).then_some(ago)
}

/// Whether a keeper made in frame `made` with `restored` restores done is
/// due for the next one in `frame`.
fn restore_at(made: u64, restored: usize, frame: u64) -> bool {
    RESTORES.get(restored).is_some_and(|&after| frame >= made + after)
}

/// A subresource's extent in texels, and its format's block width (4 for
/// block compression).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extent {
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) depth: u64,
    pub(crate) block: u64,
}

impl Extent {
    /// Of subresource `sub` of a host texture.
    pub(crate) fn of(texture: &ID3D12Resource, sub: u32) -> Option<Self> {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mips = u32::from(d.MipLevels.max(1));
        let mip = sub % mips;
        let volume = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        let (_, block) = crate::resource::element(d.Format.0 as u32)?;
        Some(Self {
            width: (d.Width >> mip).max(1),
            height: u64::from((d.Height >> mip).max(1)),
            depth: if volume { u64::from((d.DepthOrArraySize >> mip).max(1)) } else { 1 },
            block,
        })
    }

    /// In elements: texels, or blocks of a block-compressed format.
    fn elements(&self) -> Option<[u64; 3]> {
        (self.block > 0).then(|| [self.width.div_ceil(self.block), self.height.div_ceil(self.block), self.depth])
    }
}

/// Whether a copy of all of a source subresource (`boxed`: of a box of it)
/// to `corner` of a destination subresource writes every texel of the
/// destination: the host copies element by element, a texel of an integer
/// source into a block of a block-compressed destination.
pub(crate) fn whole(dst: Option<Extent>, src: Option<Extent>, corner: [u32; 3], boxed: bool) -> bool {
    !boxed
        && corner == [0, 0, 0]
        && match (dst.and_then(|e| e.elements()), src.and_then(|e| e.elements())) {
            (Some(dst), Some(src)) => dst == src,
            _ => false,
        }
}

/// `resource` (a texture the title fills) is asked to be uploaded from title
/// memory by a list being recorded: noted before the asking looks whether
/// the host has copied into it, so that a first copy recorded at the same
/// moment sees the note or the asking sees the copy.
pub(crate) fn asked(resource: This) {
    if !active() {
        return;
    }
    let frame = crate::frame::current();
    let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    let asked = asked.get_or_insert_with(HashMap::new);
    if asked.len() >= 1 << 14 {
        asked.retain(|_, &mut at| at + WINDOW >= frame);
    }
    asked.insert(resource as usize, frame);
}

/// A host copy into title texture `resource` (host copy `to`) was recorded
/// into `list`; `first`: the first into it. `subresource`: the one written
/// (none: all of them, a copy of the whole resource), `all_of_it`: every
/// texel of it ([`whole`]). `record` records the same copy into the texture
/// it is given: the keeper, if the texture has or now gets one.
pub(crate) fn copied(
    list: &HostList,
    resource: This,
    to: &ID3D12Resource,
    first: bool,
    subresource: Option<u32>,
    all_of_it: bool,
    record: impl FnOnce(&ID3D12Resource),
) {
    if !active() {
        return;
    }
    if first {
        made_first(list, resource, to);
    }
    if KEPT_NOW.load(Relaxed) == 0 {
        return;
    }
    let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = kept.as_mut() else { return };
    let Some(k) = map.get_mut(&(resource as usize)) else { return };
    if k.host != to.as_raw() as usize {
        // The host copy was made anew: the keeper is of the one before.
        map.remove(&(resource as usize));
        KEPT_NOW.fetch_sub(1, Relaxed);
        return;
    }
    record(&k.texture);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("the same copy into the keeper of {resource:?} (subresource {subresource:?}, whole {all_of_it})"));
    match subresource {
        Some(sub) if all_of_it => {
            k.whole.insert(sub);
        }
        Some(_) => {}
        None => {
            // SAFETY: a live resource.
            let d = unsafe { to.GetDesc() };
            let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(d.DepthOrArraySize.max(1)) };
            k.whole.extend(0..u32::from(d.MipLevels.max(1)) * layers);
        }
    }
    list.keep(&k.texture);
}

/// The first host copy into `resource`: its keeper, if an upload of title
/// memory may follow the copy ([`at_risk`]).
fn made_first(list: &HostList, resource: This, to: &ID3D12Resource) {
    if !crate::texture::title_written(to) {
        return;
    }
    let frame = crate::frame::current();
    let noted = ASKED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|asked| asked.get(&(resource as usize)).copied());
    let Some(ago) = at_risk(noted, frame) else {
        return;
    };
    if trace() > 0 {
        COPIED_BY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(resource as usize, std::ptr::from_ref(list) as usize);
    }
    // SAFETY: a live resource.
    let d = unsafe { to.GetDesc() };
    let keeper = keeping().then(|| keeper(&d)).flatten();
    if line(0) {
        tracing::info!(
            texture = ?resource,
            memory = ?crate::resource::texture_memory(resource).map(|(at, _)| at),
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            mips = d.MipLevels,
            frame,
            asked_frames_ago = ago,
            kept = keeper.is_some(),
            "texture trace: the first host copy into a texture the title fills that was asked to be uploaded from title memory just before: an upload recorded in a list that runs later lands after the copy (VICTORY_COPY_KEEP=1 keeps the copy)"
        );
    }
    let Some(texture) = keeper else { return };
    crate::host::name(&texture, || format!("keeper of {resource:?} {}x{} f{}", d.Width, d.Height, d.Format.0));
    let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
    let k = Kept { texture, host: to.as_raw() as usize, whole: BTreeSet::new(), made: frame, restored: 0 };
    if kept.get_or_insert_with(HashMap::new).insert(resource as usize, k).is_none() {
        KEPT_NOW.fetch_add(1, Relaxed);
    }
    let n = KEEPERS[0].fetch_add(1, Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(
            keepers = n,
            restores = KEEPERS[1].load(Relaxed),
            alive = KEPT_NOW.load(Relaxed),
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            "a texture the title fills is copied into by the host just after it was asked to be uploaded from title memory: the copy is kept and made again some frames later (VICTORY_COPY_KEEP=1)"
        );
    }
}

/// A host texture of description `d` in the copy destination state.
fn keeper(d: &D3D12_RESOURCE_DESC) -> Option<ID3D12Resource> {
    let host = crate::host::host()?;
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description of a live host texture; the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut texture)
    };
    if let Err(e) = made {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(error = %e, desc = ?d, "no keeper for a texture the host copies into"));
    }
    texture
}

/// The title released `resource`, or the host writes it in a way a keeper
/// does not follow (a resolve): its keeper and its notes go.
pub(crate) fn forget(resource: usize) {
    if !active() {
        return;
    }
    if let Some(asked) = ASKED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        asked.remove(&resource);
    }
    if let Some(by) = COPIED_BY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        by.remove(&resource);
    }
    if KEPT_NOW.load(Relaxed) == 0 {
        return;
    }
    if let Some(kept) = KEPT.lock().unwrap_or_else(|e| e.into_inner()).as_mut()
        && kept.remove(&resource).is_some()
    {
        KEPT_NOW.fetch_sub(1, Relaxed);
    }
}

/// Records into `list` (a list being recorded, before its next draw) the
/// restores that are due: the subresources copied whole into a texture with
/// a keeper, copied from the keeper again. The texture is left readable by
/// shaders, as the pass over the heap's views leaves the ones it uploads.
pub(crate) fn restore_due(list: &HostList) {
    list.meter_ours(crate::meters::Kind::Uploads);
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    if KEPT_NOW.load(Relaxed) == 0 {
        return;
    }
    let frame = crate::frame::current();
    let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = kept.as_mut() else { return };
    let due: Vec<usize> =
        map.iter().filter(|(_, k)| restore_at(k.made, k.restored, frame)).map(|(&resource, _)| resource).collect();
    for resource in due {
        let Some(k) = map.get_mut(&resource) else { continue };
        // The keeper is of this host copy; a texture the host copied into
        // keeps its host copy (`crate::resource::let_go_idle`).
        let to = crate::resource::has_host_copy(resource as This)
            .then(|| crate::resource::host_resource(resource as This))
            .flatten()
            .filter(|to| to.as_raw() as usize == k.host);
        let Some(to) = to else {
            map.remove(&resource);
            KEPT_NOW.fetch_sub(1, Relaxed);
            continue;
        };
        if !k.whole.is_empty() {
            crate::list::ensure_state(list, resource as This, COPY_DEST);
            let part = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
            };
            let (written, read) = (D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("restore of {:#x} from its keeper ({} subresources)", resource, k.whole.len()));
            // SAFETY: an open list and live textures of one description;
            // the keeper rests in the copy destination state between lists.
            unsafe {
                list.list.ResourceBarrier(&[crate::host::transition(&k.texture, written, read)]);
                for &sub in &k.whole {
                    list.list.CopyTextureRegion(&part(to, sub), 0, 0, 0, &part(&k.texture, sub), None);
                }
                list.list.ResourceBarrier(&[crate::host::transition(&k.texture, read, written)]);
            }
            crate::list::ensure_state(list, resource as This, SHADER_RESOURCE);
            list.keep(&k.texture);
            KEEPERS[1].fetch_add(1, Relaxed);
        }
        k.restored += 1;
        if line(2) {
            tracing::info!(
                texture = format_args!("{resource:#x}"),
                subresources = k.whole.len(),
                frame,
                frames_after_the_copy = frame.saturating_sub(k.made),
                restore = k.restored,
                "texture trace: a texture the host copied into is copied from its keeper again"
            );
        }
        if k.restored >= RESTORES.len() {
            map.remove(&resource);
            KEPT_NOW.fetch_sub(1, Relaxed);
        }
    }
}

/// For the trace: list `list` is submitted with an upload of title memory
/// into `resource` recorded in it (`memory`, `desc`: where the texture's
/// memory is, and its console description). Says so if another list has
/// recorded the first host copy into the texture since: that upload lands
/// after the copy if this list runs after that one.
pub(crate) fn upload_submitted(list: usize, resource: usize, memory: usize, desc: impl FnOnce() -> String) {
    if trace() == 0 {
        return;
    }
    let by = COPIED_BY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|by| by.get(&resource).copied());
    if let Some(by) = by
        && by != list
        && line(1)
    {
        tracing::info!(
            texture = format_args!("{resource:#x}"),
            memory = format_args!("{memory:#x}"),
            desc = desc(),
            list = format_args!("{list:#x}"),
            copied_by_list = format_args!("{by:#x}"),
            frame = crate::frame::current(),
            "texture trace: a list is submitted with an upload of title memory into a texture another list has copied into since: if this list runs after that one the copy is overwritten"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u64, height: u64, block: u64) -> Option<Extent> {
        Some(Extent { width, height, depth: 1, block })
    }

    #[test]
    fn a_first_copy_is_at_risk_only_just_after_an_upload_was_asked() {
        assert_eq!(at_risk(None, 100), None);
        assert_eq!(at_risk(Some(100), 100), Some(0));
        assert_eq!(at_risk(Some(100 - WINDOW), 100), Some(WINDOW));
        assert_eq!(at_risk(Some(99 - WINDOW), 100), None);
        // A note from a later frame than the copy's (frames are read apart).
        assert_eq!(at_risk(Some(101), 100), None);
    }

    #[test]
    fn a_keeper_restores_twice_and_then_no_more() {
        assert!(!restore_at(100, 0, 100));
        assert!(!restore_at(100, 0, 100 + RESTORES[0] - 1));
        assert!(restore_at(100, 0, 100 + RESTORES[0]));
        assert!(!restore_at(100, 1, 100 + RESTORES[0]));
        assert!(restore_at(100, 1, 100 + RESTORES[1]));
        assert!(!restore_at(100, RESTORES.len(), u64::MAX));
    }

    #[test]
    fn the_bakes_level_copies_write_whole_subresources() {
        // The bake: a 64x64 integer texture into the 256x256 level of a
        // block-compressed texture, a texel to a block; its 1x1 texture
        // into the 2x2 and 1x1 levels (one block each).
        assert!(whole(extent(256, 256, 4), extent(64, 64, 1), [0, 0, 0], false));
        assert!(whole(extent(2, 2, 4), extent(1, 1, 1), [0, 0, 0], false));
        assert!(whole(extent(1, 1, 4), extent(1, 1, 1), [0, 0, 0], false));
        // Like formats: texel for texel, block for block.
        assert!(whole(extent(512, 256, 1), extent(512, 256, 1), [0, 0, 0], false));
        assert!(whole(extent(6, 6, 4), extent(8, 8, 4), [0, 0, 0], false));
    }

    #[test]
    fn a_copy_of_a_part_is_not_a_whole_subresource() {
        // A smaller source, a corner, a box, an unknown format.
        assert!(!whole(extent(256, 256, 4), extent(32, 32, 1), [0, 0, 0], false));
        assert!(!whole(extent(256, 256, 1), extent(256, 256, 1), [4, 0, 0], false));
        assert!(!whole(extent(256, 256, 1), extent(256, 256, 1), [0, 0, 0], true));
        assert!(!whole(None, extent(64, 64, 1), [0, 0, 0], false));
        assert!(!whole(extent(64, 64, 1), None, [0, 0, 0], false));
        // A volume's depth counts.
        let deep = Some(Extent { width: 16, height: 16, depth: 4, block: 1 });
        assert!(!whole(deep, extent(16, 16, 1), [0, 0, 0], false));
        assert!(whole(deep, deep, [0, 0, 0], false));
    }

    /// On the card: keepers of the bake's textures are made (in the copy
    /// destination state), and every level of one is written whole by the
    /// level of the integer texture the bake copies into it.
    #[test]
    fn keepers_of_the_bakes_textures_are_made_on_this_card() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
        if crate::host::host().is_none() {
            // No GPU to make textures on: nothing to check.
            return;
        }
        let texture = |side: u32, mips: u16, format: i32| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(side),
            Height: side,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: DXGI_FORMAT(format),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // 256x256 and 2048x2048 in format 78, 512x512 in format 77, each
        // with all its levels (all15b-on2 frame 923, cx1-water-on1 frame
        // 1160).
        for (side, mips, format) in [(256, 9, 78), (2048, 12, 78), (512, 10, 77)] {
            let kept = keeper(&texture(side, mips, format)).expect("a keeper");
            // The integer texture the compute program writes: a texel for
            // each block (format 3), down to 1x1; the two smallest levels
            // come from 1x1 textures.
            let blocks = keeper(&texture(side / 4, mips - 2, 3)).expect("an integer texture");
            for level in 0..u32::from(mips) {
                let source = Extent::of(&blocks, level.min(u32::from(mips) - 3));
                let all = whole(Extent::of(&kept, level), source, [0, 0, 0], false);
                assert!(all, "level {level} of {side}x{side} f{format} is not written whole by {source:?}");
            }
            // A level too large for the destination's is not a whole copy.
            assert!(!whole(Extent::of(&kept, 0), Extent::of(&blocks, 1), [0, 0, 0], false));
        }
    }
}
