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

use std::sync::OnceLock;

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
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy, in the copy-source state) into a read-back buffer: with
/// [`by_views`], only of the subresources the list's unordered access views
/// of it (`views`, their descriptions' words) expose.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    resource: This,
    texture: &ID3D12Resource,
    views: &[Option<[u32; 10]>],
) -> Option<Back> {
    let host = crate::host::host()?;
    let (layout, at, size) = crate::resource::title_texture(resource)?;
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
    Some(Back { buffer, subresources, mips, layout, at, size })
}

/// Textures copied back since the start.
pub(crate) static COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Once the list that recorded `back` has run: its texels, tiled into the
/// title's layout, into the texture's title memory.
pub(crate) fn land(back: &Back) {
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
