//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! (Geckomacabre/vx-layer eee5667d, GPL-3.0, after fxceless issue #863.)
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; `0` for none; by default the kinds the read
//! watch found the title reading), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.
//!
//! The texels land frames after the write was recorded. Where the texture
//! lay then is what they are for ([`Target`]): a landing whose memory the
//! runtime has mapped anew since, or whose texture the title has released,
//! is not made, and one that is made writes the bytes its texels cover and
//! no others (`VICTORY_TEXTURE_BACK_CHECK=0`: neither, as before).

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_else(|_| DEFAULT.to_owned());
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// `VICTORY_TEXTURE_BACK_STATE` (on by default; `0`: off): after the copy
/// a texture is moved back to the state the title's commands left it in
/// (fxceless/victory-fork #1170, BradRoland, GPL-3.0). Off, it stayed in
/// COPY_SOURCE: the list's end state, and so the texture's real state,
/// which the next list starts from where it does not name the texture
/// (shaders that pick it from the heap by a computed index).
fn restore_state() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_STATE").is_none_or(|v| v != "0"))
}

/// The states the copy is made between, from the one the list leaves the
/// host copy in so far (`last`; none: the list has not named it): the one
/// to move to before the copy and the one to move back to after it.
pub(crate) fn copy_states(last: Option<u32>) -> (Option<u32>, Option<u32>) {
    copy_states_with(last, restore_state())
}

/// [`copy_states`], restoring or not. A read state that already holds
/// COPY_SOURCE needs neither move; a texture the list does not name stays
/// in COPY_SOURCE (its prelude puts it there, its real state follows).
fn copy_states_with(last: Option<u32>, restore: bool) -> (Option<u32>, Option<u32>) {
    const COPY_SOURCE: u32 = 0x800;
    if !restore {
        return (Some(COPY_SOURCE), None);
    }
    match last {
        Some(state) if state & COPY_SOURCE != 0 => (None, None),
        Some(state) => (Some(COPY_SOURCE), Some(state)),
        None => (Some(COPY_SOURCE), None),
    }
}

/// `VICTORY_TEXTURE_BACK_VIEWS=1` (off by default): only the mips and
/// array slices the list's unordered access views of a texture expose are
/// copied back (fxceless/victory-fork #1152, BradRoland, GPL-3.0). Off,
/// every subresource is: those no view could write carry the host copy's
/// bytes over title memory, which may be older than what the title (or
/// another texture over the same bytes) put there.
fn by_views() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_VIEWS").is_some_and(|v| v == "1"))
}

/// The subresources (mip + slice * mips) the unordered access views
/// `views` of a texture with `mips` mips and `slices` array slices can
/// write: a view without a description is the resource's own (mip 0 of
/// every slice); 2D arrays name a mip and a run of slices (`u32::MAX`: the
/// rest); a 2D view of an array is its slice 0. Views of other kinds, of
/// a second plane or outside the texture add nothing.
pub(crate) fn view_subresources(
    views: &[Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>],
    mips: u32,
    slices: u32,
) -> std::collections::BTreeSet<u32> {
    let mut out = std::collections::BTreeSet::new();
    for view in views {
        let (mip, first, count, plane) = match view {
            None => (0, 0, slices, 0),
            Some(desc) => match desc.ViewDimension {
                D3D12_UAV_DIMENSION_TEXTURE2D => {
                    // SAFETY: the member ViewDimension names.
                    let v = unsafe { desc.Anonymous.Texture2D };
                    (v.MipSlice, 0, 1, v.PlaneSlice)
                }
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                    // SAFETY: the member ViewDimension names.
                    let v = unsafe { desc.Anonymous.Texture2DArray };
                    (v.MipSlice, v.FirstArraySlice, v.ArraySize, v.PlaneSlice)
                }
                _ => continue,
            },
        };
        let end = if count == u32::MAX { Some(slices) } else { first.checked_add(count) };
        if plane == 0
            && mip < mips
            && let Some(end) = end.filter(|&end| first < end && end <= slices)
        {
            out.extend((first..end).map(|slice| slice * mips + mip));
        }
    }
    out
}

/// `VICTORY_TEXTURE_BACK_CHECK` (on by default; `0`: off): a texture's
/// place in title memory is taken when its first writer is recorded
/// ([`Target`]), its texels land only if it still lies there, and they
/// overwrite only the bytes they cover (after DexNom/VIctory PR #63,
/// Cuebert, GPL-3.0). Off, as before: the address is taken when the list
/// closes, and the landing reads the texture's whole range from title
/// memory, lays the texels over it and writes all of it back, whatever
/// lives there by then; bytes the texels do not cover are rewritten with
/// what was read, which a title thread may have changed in between.
pub(crate) fn checked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_CHECK").is_none_or(|v| v != "0"))
}

/// A ticket's states: the texture lies where it lay; the runtime mapped or
/// unmapped memory there; the title released the texture.
const FRESH: u8 = 0;
const REMAPPED: u8 = 1;
const RELEASED: u8 = 2;

/// The tickets of the copies not landed yet: the title resource, its title
/// memory (first byte, end) and the ticket its [`Target`] and [`Back`]
/// hold. [`remapped`] and [`released`] mark the ones they meet.
static TICKETS: Mutex<Vec<(usize, usize, usize, Weak<AtomicU8>)>> = Mutex::new(Vec::new());

/// Where a texture copied back lay in title memory when a list's first
/// writer of it was recorded, and the ticket that says whether it still
/// does: the list's copy lands there or not at all.
pub(crate) struct Target {
    layout: victory_formats::texture::Layout,
    at: usize,
    size: usize,
    ticket: Arc<AtomicU8>,
}

impl Target {
    /// Title texture `resource` as it lies now; `None` for one that is not
    /// copied back (a buffer, planes, no memory).
    pub(crate) fn new(resource: This) -> Option<Self> {
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        Self::at(resource as usize, layout, at, size)
    }

    fn at(resource: usize, layout: victory_formats::texture::Layout, at: usize, size: usize) -> Option<Self> {
        let end = at.checked_add(size).filter(|_| at != 0 && size != 0)?;
        let ticket = Arc::new(AtomicU8::new(FRESH));
        let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
        tickets.retain(|t| t.3.strong_count() > 0);
        tickets.push((resource, at, end, Arc::downgrade(&ticket)));
        Some(Target { layout, at, size, ticket })
    }

    /// Whether the texture no longer lies where it lay (then logged): the
    /// list makes no copy of it.
    pub(crate) fn stale(&self) -> bool {
        rejected(&self.ticket, self.at, self.size, "before its copy was recorded")
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: copies
/// recorded for textures there do not land.
pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    mark(|_, first, last| first < end && at < last, REMAPPED);
}

/// The title released `resource`: its copies do not land (its memory is
/// the title's to use for something else from here on).
pub(crate) fn released(resource: usize) {
    mark(|r, _, _| r == resource, RELEASED);
}

fn mark(meets: impl Fn(usize, usize, usize) -> bool, state: u8) {
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if tickets.is_empty() {
        return;
    }
    tickets.retain(|(resource, first, end, ticket)| {
        let Some(ticket) = ticket.upgrade() else {
            return false;
        };
        if meets(*resource, *first, *end) {
            // The first reason stays.
            let _ = ticket.compare_exchange(FRESH, state, Relaxed, Relaxed);
        }
        true
    });
}

/// Landings not made since the start ([`checked`]).
pub(crate) static REJECTED: AtomicU64 = AtomicU64::new(0);

/// Whether `ticket` says the texture at `[at, at + size)` no longer lies
/// there. Logged: the first 32 times, then at each power of two.
fn rejected(ticket: &AtomicU8, at: usize, size: usize, when: &str) -> bool {
    let why = match ticket.load(Relaxed) {
        FRESH => return false,
        REMAPPED => "its memory was mapped anew",
        _ => "the title released it",
    };
    let n = REJECTED.fetch_add(1, Relaxed) + 1;
    if n <= 32 || n.is_power_of_two() {
        tracing::warn!(
            n,
            at = format_args!("{at:#x}"),
            size,
            why,
            when,
            copied = COPIED.load(Relaxed),
            "a texture copy-back is not landed: the texture no longer lies where its writer was recorded (VICTORY_TEXTURE_BACK_CHECK=0: landed all the same)"
        );
    }
    true
}

/// A texture's copy into a read-back buffer, recorded into a list, and where
/// its texels go once the list has run.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    /// The subresources copied: index, footprint in the buffer, rows.
    subresources: Vec<(u32, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32)>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    /// The texture's title memory and its layout's bytes.
    at: usize,
    size: usize,
    /// Whether the texture still lies there ([`Target`]; none: not asked,
    /// [`checked`] off).
    ticket: Option<Arc<AtomicU8>>,
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy, in the copy-source state) into a read-back buffer: with
/// [`by_views`], only of the subresources the list's unordered access views
/// of it (`views`, their descriptions' words) expose. `target`: where the
/// texture lay at its first writer in the list ([`checked`]; none: where
/// it lies now, unchecked).
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    resource: This,
    texture: &ID3D12Resource,
    views: &[Option<[u32; 10]>],
    target: Option<Target>,
) -> Option<Back> {
    let host = crate::host::host()?;
    let (layout, at, size, ticket) = match target {
        Some(t) => (t.layout, t.at, t.size, Some(t.ticket)),
        None => {
            let (layout, at, size) = crate::resource::title_texture(resource)?;
            (layout, at, size, None)
        }
    };
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = u32::from(desc.DepthOrArraySize.max(1));
    let all = mips * slices;
    let indices: Vec<u32> = if by_views() {
        let views: Vec<_> = views
            .iter()
            .map(|v| v.as_ref().map(crate::descriptor::host::uav_desc))
            .collect();
        view_subresources(&views, mips, slices).into_iter().collect()
    } else {
        (0..all).collect()
    };
    if indices.is_empty() {
        return None;
    }
    let n = all as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            all,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // shortcut: a buffer for every subresource also when some are copied
    // (a megabyte at most for the kinds copied back).
    let buffer = host.read_back_buffer(total)?;
    let subresources: Vec<_> = indices
        .into_iter()
        .filter_map(|i| Some((i, *footprints.get(i as usize)?, *rows.get(i as usize)?)))
        .collect();
    for (i, footprint, _) in &subresources {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: *i },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    if COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        tracing::info!(
            width = desc.Width,
            height = desc.Height,
            format = desc.Format.0,
            swizzle = layout.desc.swizzle.0,
            at = format_args!("{at:#x}"),
            subresources = subresources.len(),
            of = all,
            "a texture shaders wrote is copied back into title memory, the first (VICTORY_TEXTURE_BACK)"
        );
    }
    Some(Back { buffer, subresources, mips, layout, at, size, ticket })
}

/// Textures copied back since the start.
pub(crate) static COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Runs `tile` over two buffers of `size` bytes that differ in every byte:
/// where they agree afterwards `tile` wrote, whatever it wrote. The first
/// buffer, and those ranges in order.
fn tile_covered<E>(
    size: usize,
    mut tile: impl FnMut(&mut [u8]) -> Result<(), E>,
) -> Result<(Vec<u8>, Vec<std::ops::Range<usize>>), E> {
    let mut a = vec![0u8; size];
    let mut b = vec![0xffu8; size];
    tile(&mut a)?;
    tile(&mut b)?;
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut cover = |from: usize, to: usize| match runs.last_mut() {
        Some(last) if last.end == from => last.end = to,
        _ => runs.push(from..to),
    };
    let mut at = 0;
    // Eight bytes at a time where they all agree.
    for (x, y) in a.chunks(8).zip(b.chunks(8)) {
        if x == y {
            cover(at, at + x.len());
        } else {
            for (k, _) in x.iter().zip(y).enumerate().filter(|(_, (p, q))| p == q) {
                cover(at + k, at + k + 1);
            }
        }
        at += x.len();
    }
    Ok((a, runs))
}

/// Writes the bytes of `tiled` in `runs` into title memory at `at`, and
/// none between them.
fn write_covered(at: usize, tiled: &[u8], runs: &[std::ops::Range<usize>]) {
    for run in runs {
        crate::host::copy_mapped((at + run.start) as *mut u8, tiled[run.clone()].as_ptr(), run.len(), true);
    }
}

/// Once the list that recorded `back` has run: its texels, tiled into the
/// title's layout, into the texture's title memory, if the texture still
/// lies there ([`checked`]): only the bytes the texels cover.
pub(crate) fn land(back: &Back) {
    let Some(ticket) = &back.ticket else {
        return land_whole(back);
    };
    if rejected(ticket, back.at, back.size, "before its texels were tiled") {
        return;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    type Failed = (victory_formats::texture::Error, u32, u32);
    let tiled = tile_covered(back.size, |tiled: &mut [u8]| -> Result<(), Failed> {
        for &(i, ref footprint, rows) in &back.subresources {
            let (mip, slice) = (i % back.mips, i / back.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped buffer.
            let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
            back.layout
                .tile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
                .map_err(|e| (e, mip, slice))?;
        }
        Ok(())
    });
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    let (tiled, runs) = match tiled {
        Ok(tiled) => tiled,
        Err((e, mip, slice)) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, swizzle = back.layout.desc.swizzle.0, "a texture copied back could not be tiled"));
            return;
        }
    };
    // Asked again, right before the write: a landing takes the texture's
    // place as it is now. (shortcut: a remap between this and the copies
    // below is not seen; they take microseconds, and a destination page
    // that is gone is left alone, `copy_mapped`.)
    if rejected(ticket, back.at, back.size, "before its texels were written") {
        return;
    }
    write_covered(back.at, &tiled, &runs);
}

/// [`land`] as it was ([`checked`] off): the texture's whole range is read
/// from title memory, the texels laid over it and all of it written back.
fn land_whole(back: &Back) {
    let mut tiled = vec![0u8; back.size];
    // Bytes of title memory the subresources do not cover stay as they are.
    crate::host::copy_mapped(tiled.as_mut_ptr(), back.at as *const u8, back.size, false);
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for &(i, ref footprint, rows) in &back.subresources {
        let (mip, slice) = (i % back.mips, i / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        if let Err(e) = back.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, swizzle = back.layout.desc.swizzle.0, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    crate::host::copy_mapped(back.at as *mut u8, tiled.as_ptr(), back.size, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> victory_formats::texture::Layout {
        crate::resource::test_wave_map_layout().0
    }

    /// A copy lands only where its texture still lies: memory mapped anew
    /// over any byte of it, or the texture's release, makes its ticket
    /// stale; memory mapped next to it and other textures' releases do not.
    #[test]
    fn a_remap_or_a_release_makes_a_landing_stale() {
        // Addresses and resources no other test names: in a block of ours.
        let block = vec![0u8; 0x31_0000];
        let base = block.as_ptr() as usize + 0x2000;
        let target = |resource: usize, at: usize| Target::at(resource, layout(), at, 0x1000).unwrap();
        let before = REJECTED.load(Relaxed);

        let kept = target(base + 1, base);
        remapped(base - 0x1000, 0x1000);
        remapped(base + 0x1000, 0x4000);
        released(base + 2);
        assert!(!kept.stale(), "memory next to it, another texture");

        let gone = target(base + 3, base + 0x10_0000);
        remapped(base + 0x10_0fff, 1);
        assert!(gone.stale(), "its last byte was mapped anew");
        assert!(!kept.stale());

        let freed = target(base + 4, base + 0x20_0000);
        // A new writer of the texture at the address after the remap, and
        // of the next texture at a released one's address, is fresh.
        let again = target(base + 3, base + 0x10_0000);
        released(base + 4);
        assert!(freed.stale(), "the title released it");
        assert!(!again.stale());
        // The first reason stays, and a stale ticket stays stale.
        remapped(base + 0x20_0000, 0x1000);
        assert_eq!(freed.ticket.load(Relaxed), RELEASED);
        assert!(REJECTED.load(Relaxed) >= before + 2);

        // No memory: not a texture to copy back. A ticket no one holds is
        // forgotten.
        assert!(Target::at(base + 5, layout(), 0, 0x1000).is_none());
        assert!(Target::at(base + 5, layout(), usize::MAX - 8, 0x1000).is_none());
        drop((kept, gone, freed, again));
        remapped(base, 0x30_0000);
        let left = TICKETS.lock().unwrap();
        assert!(!left.iter().any(|t| (base..base + 0x30_0000).contains(&t.1)));
    }

    /// A landing writes the bytes its texels cover and leaves the others
    /// as title memory has them at that time, whatever they were when the
    /// copy was recorded; texels that equal either filler are covered too.
    #[test]
    fn a_landing_writes_only_the_bytes_its_texels_cover() {
        let tile = |tiled: &mut [u8]| -> Result<(), ()> {
            tiled[4..8].copy_from_slice(&[0x00, 0xff, 0x37, 0x00]);
            tiled[8..12].copy_from_slice(&[1, 2, 3, 4]);
            tiled[21..30].copy_from_slice(&[0xff; 9]);
            Ok(())
        };
        let (tiled, runs) = tile_covered(40, tile).unwrap();
        assert_eq!(runs, [4..12, 21..30]);
        // Title memory, changed by the title after the copy was recorded.
        let mut memory = vec![0x9bu8; 40];
        write_covered(memory.as_mut_ptr() as usize, &tiled, &runs);
        let mut expected = vec![0x9bu8; 40];
        expected[4..12].copy_from_slice(&[0x00, 0xff, 0x37, 0x00, 1, 2, 3, 4]);
        expected[21..30].copy_from_slice(&[0xff; 9]);
        assert_eq!(memory, expected);
        // Nothing covered: nothing written; a failure writes nothing.
        assert!(tile_covered(16, |_| Ok::<(), ()>(())).unwrap().1.is_empty());
        assert!(tile_covered(16, |_| Err(())).is_err());
    }

    /// A real layout (a 64x64 two-channel float texture with mips, as the
    /// title's wave maps are): every texel of every subresource lands, and
    /// the bytes of the layout no subresource covers are not written.
    #[test]
    fn a_tiled_texture_lands_whole_and_its_padding_is_left() {
        let (layout, mips, size) = crate::resource::test_wave_map_layout();
        let texel = 8usize;
        let (tiled, runs) = tile_covered(size, |tiled| {
            for mip in 0..mips {
                let side = (64usize >> mip).max(1);
                let linear = vec![0x40 + mip as u8; side * side * texel];
                layout.tile(tiled, mip, 0, &linear, side * texel, side * side * texel)?;
            }
            Ok::<(), victory_formats::texture::Error>(())
        })
        .unwrap();
        let covered: usize = runs.iter().map(|r| r.len()).sum();
        let texels: usize = (0..mips).map(|mip| (64usize >> mip).max(1).pow(2) * texel).sum();
        assert!(covered >= texels && covered <= size, "{covered} of {size}, {texels} of texels");
        // Mip 0's first texel is where the layout says, and holds its value.
        let first = layout.offset(0, 0, 0, 0).unwrap() as usize;
        assert!(runs.iter().any(|r| r.contains(&first)));
        assert_eq!(tiled[first], 0x40);
        let mut memory = vec![0x9bu8; size];
        write_covered(memory.as_mut_ptr() as usize, &tiled, &runs);
        let kept = memory.iter().filter(|&&b| b == 0x9b).count();
        assert_eq!(kept, size - covered, "bytes no texel covers stay");
    }

    #[test]
    fn the_copy_moves_back_to_the_state_the_title_left() {
        const SRV: u32 = 0xc0;
        const UAV: u32 = 0x8;
        const SOURCE: u32 = 0x800;
        assert_eq!(copy_states_with(Some(SRV), true), (Some(SOURCE), Some(SRV)));
        assert_eq!(copy_states_with(Some(UAV), true), (Some(SOURCE), Some(UAV)));
        // Already a read state with COPY_SOURCE in it: no barrier at all.
        assert_eq!(copy_states_with(Some(SRV | SOURCE), true), (None, None));
        assert_eq!(copy_states_with(Some(SOURCE), true), (None, None));
        // Not named by the list: its prelude brings it to COPY_SOURCE.
        assert_eq!(copy_states_with(None, true), (Some(SOURCE), None));
        // Off: as before, left in COPY_SOURCE.
        assert_eq!(copy_states_with(Some(UAV), false), (Some(SOURCE), None));
    }

    fn array(mip: u32, first: u32, size: u32) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
        Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: mip,
                    FirstArraySlice: first,
                    ArraySize: size,
                    PlaneSlice: 0,
                },
            },
            ..Default::default()
        })
    }

    fn two_d(mip: u32) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
        Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV { MipSlice: mip, PlaneSlice: 0 },
            },
            ..Default::default()
        })
    }

    #[test]
    fn only_the_subresources_a_view_exposes_are_copied_back() {
        let set = |views: &[_], mips, slices| view_subresources(views, mips, slices).into_iter().collect::<Vec<_>>();
        // The resource's own view: mip 0 of every slice.
        assert_eq!(set(&[None], 3, 4), [0, 3, 6, 9]);
        // Mip 1 of slices 1 and 2, overlapping slices 2 on (u32::MAX: the rest).
        assert_eq!(set(&[array(1, 1, 2), array(1, 2, u32::MAX)], 3, 4), [4, 7, 10]);
        // A 2D view of an array: its slice 0.
        assert_eq!(set(&[two_d(2)], 3, 4), [2]);
        assert_eq!(set(&[two_d(1)], 3, 1), [1]);
        // Outside the texture: nothing.
        assert_eq!(set(&[array(0, 4, u32::MAX), two_d(3), array(0, 3, 2)], 3, 4), Vec::<u32>::new());
    }
}
