//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! The scratch buffer is left readable as indirect arguments after a clamp
//! and stays so until the next clamp into it, which moves it straight back
//! to unordered access; it goes back to the common state once, when the
//! list closes. The executions that read it record no barrier of their
//! own, so draws that read records clamped together (`crate::draw`'s
//! windows) follow each other with nothing between them. Before, each
//! execution was followed by a move of the scratch buffer back to the
//! common state, which drivers may treat as a full wait.
//!
//! With `VICTORY_INDIRECT_BATCH` the executions of one record without a
//! count buffer (nearly all the title's) are clamped in batches ([`batch`]):
//! the rest of this is for the others, and for every execution without it.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

pub(crate) mod batch;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bits 2 to 7: the part of
    // dispatch records over 65,535 groups in x this clamp writes (0: the
    // first; `with_part`). Bits 8 to 15: dispatches, host thread groups per
    // console thread group along z (a console group of several waves run as
    // one host group per wave, `with_split`; 0 or 1: as asked).
    uint has_count;
    uint count_offset;
    // The execution (`note_dispatch`), noted with a record it cuts down; bits
    // 28 to 30: the parts after the first its records run in (`parts`).
    uint execution;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64 threads racing for one word, cost a
// round for each record of a window, and in memory the CPU maps (where the
// budget was, and is with VICTORY_INDIRECT_CARD_BUDGET=0) every atomic
// crosses the bus on a card of its own.
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        // Bits 2 to 7: part `part` of records over 65,535 groups in x (`with_part`):
        // the groups from 65,535 x part on, run by the program translated with that base.
        uint part = (has_count >> 2) & 0x3fu;
        bool rest = part != 0;
        if (rest) {
            uint base = 65535u * part;
            asked.x = asked.x > base ? asked.x - base : 0u;
        }
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        bool huge = g.z != 0 && g.x * g.y > GROUPS / g.z;
        // Diagnostics, in the budget buffer's words 3 to 5: the largest
        // first dimension a compute record asked for (as the title wrote it),
        // the groups run in all, and the records cut down (a dimension over
        // 65,535, or more groups than a dispatch runs).
        uint was;
        if (!rest) {
            budget.InterlockedMax(12, asked.x, was);
            // (cut: past what the record's parts run, 65,535 groups in x each)
            uint covered = 65535u * (((execution >> 28) & 7u) + 1u);
            if (huge || asked.x > covered || any(asked.yz > g.yz)) {
                budget.InterlockedAdd(20, 1u, was);
                // word 6: the largest first dimension a cut record asked for (18
                // bits), and the execution it came from (14 bits)
                budget.InterlockedMax(24, (min(asked.x, 0x3ffffu) << 14) | (execution & 0x3fffu), was);
            }
        }
        if (huge) {
            g = uint3(0, 0, 0);
        }
        // One host group per console wave of a group the translation split
        // (`with_split`): past what a dispatch runs along z, empty. (The
        // counts and the charge below are of the console's groups, as
        // without the split; z is multiplied after them.)
        uint split = (has_count >> 8) & 0xffu;
        if (split > 1u && g.z * split > 65535u) {
            g = uint3(0, 0, 0);
        }
        budget.InterlockedAdd(16, g.x * g.y * g.z, was);
        // word 7: the groups run by the watched pipelines (`VICTORY_INDIRECT_WATCH`,
        // bit 31 of the execution)
        if ((execution & 0x80000000u) != 0) {
            budget.InterlockedAdd(28, g.x * g.y * g.z, was);
            // words 8-10: the watched records as asked (first parts only): how
            // many, how many ask for no group, the largest first dimension
            if (!rest) {
                budget.InterlockedAdd(32, 1u, was);
                if (asked.x == 0u || asked.y == 0u || asked.z == 0u) {
                    budget.InterlockedAdd(36, 1u, was);
                }
                budget.InterlockedMax(40, asked.x, was);
            }
        }
        cost = g.x * g.y * g.z * 64u;
        // (split: z counts the console waves of each group too)
        if (split > 1u) {
            g.z *= split;
        }
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        // Diagnostics, in the budget buffer's next words (below): the
        // largest group count a mesh record asked for, and the groups
        // asked in all.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    // The group's own records: none for a group past a count buffer's
    // count (an execution may ask for thousands and count a few), which
    // touches the budget not at all, as its threads did not before.
    uint first = id.x - gi;
    uint n = eff_count > first ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest.
        // Only the group's own records are walked (one, clamped alone).
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, SOURCE)
}

/// The clamp compiled from `source` (ours, or in the benchmark the one it
/// replaced).
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Executions of indirect dispatches by number (14 bits, a ring): what the
/// clamp of a dispatch's records notes with the largest record it cut down
/// (word 6 of the list's budget), so the frame line can name the pipeline
/// whose record asked for the most groups.
static DISPATCHES: std::sync::Mutex<Vec<Option<std::sync::Arc<str>>>> = std::sync::Mutex::new(Vec::new());
static NEXT_DISPATCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
const DISPATCH_RING: u32 = 1 << 14;

thread_local! {
    /// The execution the next clamp on this thread is recorded for ([`with_dispatch`]).
    static EXECUTION: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Runs `f` (the clamp of an indirect dispatch's records) with a new execution
/// number noted as `label` (the pipeline's shaders), which [`dispatch_label`]
/// gives back for the number the clamp wrote.
pub(crate) fn with_dispatch<R>(label: impl FnOnce() -> String, f: impl FnOnce() -> R) -> R {
    let was = EXECUTION.replace(note_dispatch(label));
    let r = f();
    EXECUTION.set(was);
    r
}

/// A new execution number noted as `label` (the pipeline's shaders), with
/// what the clamp of a dispatch's records is told about it besides: the
/// parts its records run in (bits 28 to 30) and whether its pipeline is
/// watched (bit 31). What [`with_dispatch`] sets for its clamp, and what a
/// batched clamp's job of a dispatch record carries ([`batch`]).
pub(crate) fn note_dispatch(label: impl FnOnce() -> String) -> u32 {
    let n = NEXT_DISPATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % DISPATCH_RING;
    let watched;
    {
        let mut ring = DISPATCHES.lock().unwrap_or_else(|e| e.into_inner());
        if ring.is_empty() {
            ring.resize(DISPATCH_RING as usize, None);
        }
        let label: std::sync::Arc<str> = label().into();
        watched = watching(&label);
        ring[n as usize] = Some(label);
    }
    // (the parts its records run in, for the clamp's count of records cut)
    let parts = parts(|| ring_label(n));
    n | ((parts - 1).min(7) << 28) | if watched { 1 << 31 } else { 0 }
}

thread_local! {
    /// The part of its dispatch records the clamp on this thread writes (0: the
    /// first; [`with_part`]).
    static PART: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The most parts a dispatch record is run in: 65,535 groups in x each, up to
/// 262,140 (the clamp notes what a cut record asked for in 18 bits).
pub(crate) const MOST_PARTS: u32 = 4;

/// Runs `f` (the clamp of an indirect dispatch's records) for part `part`
/// (1 and up) of the records: the groups from 65,535 x `part` on in x of a
/// record that asks for more, up to 65,535 of them (none of a record that does
/// not), which the pipeline translated with that group base runs
/// (`VICTORY_INDIRECT_SPLIT`, `crate::draw`).
pub(crate) fn with_part<R>(part: u32, f: impl FnOnce() -> R) -> R {
    let was = PART.replace(part);
    let r = f();
    PART.set(was);
    r
}

/// [`with_part`] for the second part (groups 65,535 to 131,069).
#[cfg(test)]
pub(crate) fn with_remainder<R>(f: impl FnOnce() -> R) -> R {
    with_part(1, f)
}

thread_local! {
    /// The host thread groups per console thread group of the dispatch whose records
    /// the clamp on this thread writes ([`with_split`]; 0 or 1: as the console's).
    static GROUP_SPLIT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Runs `f` (the clamp of an indirect dispatch's records) for a program the translation
/// split into `split` host thread groups per console thread group, numbered along z
/// (`victory_shader::emit::HostShader::group_split`, `VICTORY_GROUP_SPLIT`): the clamp
/// multiplies each record's z by it (a record that would pass 65,535 runs empty). The
/// title's arguments stay as it wrote them; only the clamped copy the host executes
/// changes. (Corgansm/victory-corgi's `with_split`.)
pub(crate) fn with_split<R>(split: u32, f: impl FnOnce() -> R) -> R {
    let was = GROUP_SPLIT.replace(split.min(255));
    let r = f();
    GROUP_SPLIT.set(was);
    r
}

/// `VICTORY_INDIRECT_SPLIT=<hash>,...`: pipelines (`compute 0123456789abcdef`) whose indirect
/// dispatches run a record over 65,535 groups in x in parts ([`parts`]).
fn split_list() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_INDIRECT_SPLIT")
            .map(|v| {
                v.split([',', ';', ' '])
                    .map(|h| h.trim().trim_start_matches('#').to_ascii_lowercase())
                    .filter(|h| !h.is_empty() && h != "0")
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Whether a pipeline whose indirect dispatch record was cut down (more than 65,535 groups in
/// x: the groups past them did not run) is split from then on ([`note_cut`]), in as many
/// parts as the record asked for (`VICTORY_INDIRECT_SPLIT_AUTO=0`: only the pipelines
/// `VICTORY_INDIRECT_SPLIT` names, in two parts, as before). The boat scene's black
/// rectangles were `649850fad69f9228`'s (97,368 groups asked), and 27 more pipelines were cut
/// in that run. A cut is seen once its list has run: the frames until then still lose it.
fn auto() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_SPLIT_AUTO").is_none_or(|v| v != "0"))
}

/// The pipelines [`note_cut`] has seen cut down, by the hash in their label, with the parts
/// their largest record needs.
static LEARNED: std::sync::Mutex<Vec<(String, u32)>> = std::sync::Mutex::new(Vec::new());

/// The parts a record of `asked` groups in x needs (65,535 each), at most [`MOST_PARTS`].
fn parts_for(asked: u32) -> u32 {
    asked.div_ceil(65_535).clamp(1, MOST_PARTS)
}

/// A dispatch record of the pipeline `label` names (`compute 0123456789abcdef`) asked for
/// `asked` groups in x and was cut down: with [`auto`], that pipeline's dispatches run in
/// parts from now on. Whether that is new, or more parts than before.
pub(crate) fn note_cut(label: &str, asked: u32) -> bool {
    if !auto() || asked <= 65_535 {
        return false;
    }
    let Some(hash) = label.split_whitespace().last().filter(|h| h.len() == 16) else {
        return false;
    };
    let parts = parts_for(asked);
    let mut learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    match learned.iter_mut().find(|(h, _)| h == hash) {
        Some((_, n)) if *n >= parts => false,
        Some((_, n)) => {
            *n = parts;
            tracing::info!(pipeline = %label, asked, parts, "an indirect dispatch record asked for more groups than its parts run: more parts from now on (VICTORY_INDIRECT_SPLIT_AUTO)");
            true
        }
        None => {
            learned.push((hash.to_owned(), parts));
            tracing::info!(pipeline = %label, asked, parts, "an indirect dispatch record was cut down: its pipeline's dispatches run in parts from now on (VICTORY_INDIRECT_SPLIT_AUTO)");
            true
        }
    }
}

/// How many parts the indirect dispatches of the pipeline `label` names (`compute
/// 0123456789abcdef`) run a record in: 1 (as it is), or 2 and up, the parts after the first by
/// the program translated with a group base of 65,535 x the part (exact up to 65,535 x the
/// parts groups). The pipelines `VICTORY_INDIRECT_SPLIT` names run in two parts at least; with
/// [`auto`] those seen cut run in as many as their largest record needed.
pub(crate) fn parts(label: impl FnOnce() -> String) -> u32 {
    let list = split_list();
    let learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    if list.is_empty() && learned.is_empty() {
        return 1;
    }
    let label = label();
    let mut parts = if list.iter().any(|h| label.contains(h.as_str())) { 2 } else { 1 };
    if let Some((_, n)) = learned.iter().find(|(h, _)| label.contains(h.as_str())) {
        parts = parts.max(*n);
    }
    parts
}

/// Whether any pipeline's indirect dispatches may run in parts ([`parts`]), for the frame line.
pub(crate) fn splitting() -> bool {
    !split_list().is_empty() || auto()
}

#[cfg(test)]
mod parts_tests {
    use super::{MOST_PARTS, parts_for};

    /// A record runs in one part per 65,535 groups in x, at most [`MOST_PARTS`].
    #[test]
    fn a_record_runs_in_a_part_per_65535_groups() {
        assert_eq!(parts_for(0), 1);
        assert_eq!(parts_for(65_535), 1);
        assert_eq!(parts_for(65_536), 2);
        assert_eq!(parts_for(97_368), 2);
        assert_eq!(parts_for(131_070), 2);
        assert_eq!(parts_for(131_071), 3);
        assert_eq!(parts_for(262_140), MOST_PARTS);
        assert_eq!(parts_for(262_143), MOST_PARTS);
    }
}

/// `VICTORY_INDIRECT_WATCH=<hash>,...`: whether the pipeline `label` names
/// (`compute 0123456789abcdef`) is one whose indirect dispatches' groups the
/// frame line counts (word 7 of the lists' budgets).
fn watching(label: &str) -> bool {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_INDIRECT_WATCH")
            .map(|v| {
                v.split([',', ';', ' '])
                    .map(|h| h.trim().trim_start_matches('#').to_ascii_lowercase())
                    .filter(|h| !h.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    !list.is_empty() && list.iter().any(|h| label.contains(h.as_str()))
}

/// `VICTORY_INDIRECT_WATCH`, on the CPU, since the last frame line: the
/// watched pipelines' indirect dispatches recorded, those with a count buffer,
/// the largest first dimension of their first record in title memory when
/// recorded (`u32::MAX`: unreadable), and the largest count in title memory
/// ([`note_watched`]).
pub(crate) static WATCHED_CPU: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Notes an indirect dispatch of a watched pipeline (`VICTORY_INDIRECT_WATCH`)
/// as the title's memory holds it when it is recorded: its first record at
/// `arguments`, its count at `count`. On the card the record may be another:
/// a dispatch earlier in the submission writes it (the frame line has both).
pub(crate) fn note_watched(label: impl FnOnce() -> String, arguments: u64, count: Option<u64>) {
    use std::sync::atomic::Ordering::Relaxed;
    if std::env::var_os("VICTORY_INDIRECT_WATCH").is_none() || !watching(&label()) {
        return;
    }
    WATCHED_CPU[0].fetch_add(1, Relaxed);
    let mut first = [0u8; 4];
    let x = if crate::host::read_into(arguments as *const u8, &mut first) { u32::from_le_bytes(first) } else { u32::MAX };
    WATCHED_CPU[2].fetch_max(u64::from(x), Relaxed);
    if let Some(at) = count {
        WATCHED_CPU[1].fetch_add(1, Relaxed);
        let mut n = [0u8; 4];
        let n = if crate::host::read_into(at as *const u8, &mut n) { u32::from_le_bytes(n) } else { u32::MAX };
        WATCHED_CPU[3].fetch_max(u64::from(n), Relaxed);
    }
}

/// The label [`with_dispatch`] noted for execution `n` (empty if none).
fn ring_label(n: u32) -> String {
    dispatch_label(n).map(|l| l.to_string()).unwrap_or_default()
}

/// What [`with_dispatch`] noted for execution `n`, if the ring still holds it.
pub(crate) fn dispatch_label(n: u32) -> Option<std::sync::Arc<str>> {
    let ring = DISPATCHES.lock().unwrap_or_else(|e| e.into_inner());
    ring.get((n % DISPATCH_RING) as usize).cloned().flatten()
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, at offset `0`. The scratch buffer is left in `INDIRECT_ARGUMENT`
/// until the next clamp into it: executions read it with no barrier of
/// their own. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp or alignment is not
/// available. (Tests; the draws use [`bound_at`].)
#[cfg(test)]
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    bound_at(list, src, count, spec, (0, 0))
}

/// [`bound`] into the scratch buffer from byte `at` on (a multiple of 4),
/// in a scratch buffer of at least `room` bytes (`crate::draw`'s windows
/// keep several runs of clamped records at once, each in a part of its
/// own: `VICTORY_INDIRECT_WINDOW=2`).
pub(crate) fn bound_at(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    (at, room): (u64, u64),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let size = at.checked_add(bytes)?.max(room);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(size)?);
    clamp_into(list, c, (&dst, at), budget, src, count, spec, false).then_some(dst)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before. (Tests; the draws use
/// [`bound_ahead_at`].)
#[cfg(test)]
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    bound_ahead_at(list, src, spec, (0, 0))
}

/// [`bound_ahead`] into the scratch buffer from byte `at` on, in one of at
/// least `room` bytes ([`bound_at`]).
pub(crate) fn bound_ahead_at(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
    (at, room): (u64, u64),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let size = at.checked_add(bytes)?.max(room);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(size)?);
    clamp_into(list, c, (&dst, at), budget, src, None, spec, true).then_some(dst)
}

/// [`record_clamp_as`] into `list`'s scratch buffer `dst`, from the state
/// the list left it in; afterwards the list knows it is readable as
/// indirect arguments.
#[allow(clippy::too_many_arguments)]
fn clamp_into(
    list: &HostList,
    c: &Clamp,
    (dst, at): (&ID3D12Resource, u64),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    let before = list.indirect_state(dst);
    let recorded = record_clamp_at(c, &list.list, (dst, before, at), budget, src, count, spec, ahead);
    if recorded {
        list.indirect_readable(dst);
    }
    recorded
}

/// Records the clamp into `dst`, which is in state `before` (common, or
/// indirect arguments after an earlier clamp into it), and leaves `dst` in
/// `INDIRECT_ARGUMENT`. `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]). False, with nothing recorded, if the
/// layout or the addresses do not fit the clamp.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    record_clamp_at(c, l, (dst, before, 0), budget, src, count, spec, ahead)
}

/// [`record_clamp_as`] into `dst` from byte `at` on: record `i` lands at
/// `at + i * stride`, and nothing else of `dst` is written.
#[allow(clippy::too_many_arguments)]
fn record_clamp_at(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before, at): (&ID3D12Resource, D3D12_RESOURCE_STATES, u64),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_base) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    let Some(dst_gpu) = dst_base.checked_add(at) else {
        return false;
    };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let execution = EXECUTION.with(std::cell::Cell::get);
    let split = GROUP_SPLIT.with(std::cell::Cell::get);
    let rest = ((PART.with(std::cell::Cell::get) & 0x3f) << 2) | if kind == 2 && split > 1 { split << 8 } else { 0 };
    let constants = [kind, stride, records, args_offset, has_count | (u32::from(ahead) << 1) | rest, count_offset, execution];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `before -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`. Moving
    // it out of `INDIRECT_ARGUMENT` orders this clamp's writes after every
    // earlier execution's reads of the scratch buffer; nothing else writes it.
    // Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [
        uav_barrier(),
        transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip = usize::from(!waits);
    // SAFETY: l is an open command list, dst is a live scratch buffer,
    // barriers transition dst correctly, and constants and root arguments
    // match c's compute pipeline layout.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&second[skip..]);
    }
    true
}

/// Whether an execution of at most `max` records runs nothing, whatever its
/// arguments and count say: `State::execute_indirect` ends it before the
/// argument refresh, the clamp (a wait for every earlier shader write and
/// two state changes) and the host's execution of no records. Only `max`
/// says so on the CPU: a count buffer is mostly written by the title's
/// passes on the card, which run after the call is recorded, so what its
/// memory holds at recording is not what the call reads.
pub(crate) fn executes_nothing(max: u32) -> bool {
    max == 0
}

/// The batched clamp's pipeline ([`batch`]).
fn batch_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let made = host().ok_or_else(windows::core::Error::empty).and_then(|h| make_from(&h.device, batch::SOURCE));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no batched argument clamp (VICTORY_INDIRECT_BATCH): records are clamped one at a time");
                    None
                }
            }
        })
        .as_ref()
}

/// Opens a batch in `list` ([`batch`]): a pass that clamps the records of
/// the executions recorded after it, of `source` (a host copy's number and
/// buffer) and of the `recent` ones after it. The table to add them to,
/// and the scratch buffer its places are in. Replaces the list's compute
/// root signature and pipeline (callers bind theirs after). `None` if the
/// clamp is not available.
///
/// Ported from avatarmanz/VIctory-7900XTX 037596241 (GPL-3.0-or-later).
pub(crate) fn open_batch(
    list: &HostList,
    source: (u64, &ID3D12Resource),
    recent: &[(u64, ID3D12Resource)],
) -> Option<(batch::Table, ID3D12Resource)> {
    let c = batch_clamp()?;
    let gpu = |b: &ID3D12Resource| {
        // SAFETY: a live buffer.
        Some(unsafe { b.GetGPUVirtualAddress() }).filter(|at| at.is_multiple_of(4))
    };
    // A source there is none of is the first again, which no job names.
    let mut ids = [0u64; batch::SOURCES];
    let mut sources = [gpu(source.1)?; batch::SOURCES];
    ids[0] = source.0;
    let budget = list.budget()?;
    let (jobs, at) = list.scratch(batch::TABLE)?;
    let (dst, base, before) = list.batch_place()?;
    let others = recent.iter().filter(|r| r.0 != source.0).take(batch::SOURCES - 1);
    for (i, (id, buffer)) in others.enumerate() {
        if let Some(at) = gpu(buffer) {
            // Bound, and read only if a job names it: kept alive like the
            // copies draws read (`Shadow::refresh`).
            list.keep(buffer);
            (ids[i + 1], sources[i + 1]) = (*id, at);
        }
    }
    // Diagnostics: a pass of ours after one wait for every earlier shader
    // write, counted with the clamps of one execution.
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    batch::record(c, &list.list, (&dst, before), budget, sources, jobs);
    // SAFETY: the list's own upload memory of that many bytes, 256-byte
    // aligned, until it is reset.
    Some((unsafe { batch::Table::new(at, base, ids) }, dst))
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
        /// One WARP device at a time (`crate::warp_alone`). The last field:
        /// let go after the device and everything above is released.
        _alone: std::sync::MutexGuard<'static, ()>,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            let alone = crate::warp_alone();
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                    _alone: alone,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the parts of pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) =
            warp.shared_buffer(16 * u64::from(PIECES * crate::shadow::CARD_PARTS));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(PIECES, 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        // In part 5 of the second piece: its dword gets bit 5.
        let at = PIECE + 5 * 4096 + 1234;
        // SAFETY: `at` is inside the second piece, strictly inside the mapped
        // buffer of size 3 * PIECE.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 32, 0]);
        assert_eq!(run(&mut warp, 1), [0, 32, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 32, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 32, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Every lane's slice is hashed: first byte of piece 0 (part 0),
        // last byte of piece 2 (part 15, its last lane), and swapping two
        // 16-byte blocks across lanes in piece 1 must all be detected.
        for &(offset, expected) in &[
            (0usize, [1, 0, 0]),
            (PIECE * 3 - 1, [0, 0, 1 << 15]),
            (63 * 16 + 7, [1, 0, 0]),
        ] {
            // SAFETY: offset is inside the mapped buffer.
            unsafe { now_ptr.add(offset).write_volatile(content[offset] ^ 0x01) };
            assert_eq!(run(&mut warp, 1), expected);
            // SAFETY: restore original byte.
            unsafe { now_ptr.add(offset).write_volatile(content[offset]) };
            assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        }
        // Swap lane 0's first 16 bytes with lane 1's first 16 bytes in piece 1.
        // SAFETY: both 16-byte blocks are inside piece 1 of the mapped buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(content[PIECE + 16..PIECE + 32].as_ptr(), now_ptr.add(PIECE), 16);
            std::ptr::copy_nonoverlapping(content[PIECE..PIECE + 16].as_ptr(), now_ptr.add(PIECE + 16), 16);
        }
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Restore piece 1, then flip the sign bit of the first u32 in both
        // two 16-byte loads of piece 0's last part (offsets 64,512 and
        // 65,024). The pair must not cancel out.
        // SAFETY: all offsets are inside the mapped buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(content[PIECE..PIECE + 32].as_ptr(), now_ptr.add(PIECE), 32);
            now_ptr.add(64_512 + 3).write_volatile(content[64_512 + 3] ^ 0x80);
            now_ptr.add(65_024 + 3).write_volatile(content[65_024 + 3] ^ 0x80);
        }
        assert_eq!(run(&mut warp, 1), [1 << 15, 0, 0]);
    }

    /// The fetch inside flights (`crate::host::CARD_FETCH`, fxceless
    /// 9376fd60c / pull request #648 by Jonirulah), on the software adapter,
    /// in one list behind the compare as `Host::card_before` records it:
    /// each flagged piece takes a ticket and, while it is within the slots
    /// of the lowest ticket in the air, its bytes go to its slot; an
    /// unflagged piece's entry says so; a ticket past the slots is written
    /// but copies nothing.
    #[test]
    fn warp_card_fetch_copies_flagged_pieces_to_their_slots() {
        use crate::host::{CARD_COMPARE, CARD_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, CARD_COMPARE).expect("the card compare must compile on WARP");
        let fetch = compile_cs(&warp.device, CARD_FETCH).expect("the card fetch must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        // The copy starts 40 bytes into its piece 0: piece 0 is short, the
        // others straddle title pieces and start off a multiple of 16.
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let pieces = PIECES + 1;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(pieces * crate::shadow::CARD_PARTS));
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 1u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mappings hold what is written.
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> Vec<u32> {
            // SAFETY: our open list, live buffers, the shaders' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(pieces, 1, 1);
                if mode == 2 {
                    warp.list.ResourceBarrier(&[uav_barrier()]);
                    warp.list.SetComputeRootSignature(&fetch.root);
                    warp.list.SetPipelineState(&fetch.pso);
                    // Two groups for four pieces: the loop over pieces.
                    let constants = [0u32, SKEW, size as u32, pieces, 0, landed, slots, 2];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, table_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, fetched_gpu);
                    warp.list.Dispatch(2, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            (0..=pieces as usize).map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() }).collect()
        };
        let set = |at: usize, value: u8| {
            // SAFETY: every `at` below is inside the mapping of the copy.
            unsafe { now_ptr.add(at).write_volatile(value) }
        };
        let slot = |len: usize| {
            // SAFETY: the slot's mapping holds `INFLIGHT_SLOT` bytes, and
            // no `len` below is more.
            unsafe { std::slice::from_raw_parts(fetched_ptr, len).to_vec() }
        };
        run(&mut warp, 0, 0);
        // Piece 2 of the copy (title piece 2: bytes 2 * 64 KiB - 40 on), its
        // slot from the multiple of 16 below to the one above its end.
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        assert_ne!(widened, from);
        set(from + 100, content[from + 100] ^ 0x40);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table, [1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted, "piece 2 in slot 0");
        // Two pieces flagged with one slot, ticket 0 not landed: tickets 1
        // and 2 get none and the slot keeps what it held.
        set(5, content[5] ^ 1);
        set(PIECE + 7, content[PIECE + 7] ^ 1);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table[0], 3);
        let mut tickets = [table[1], table[2]];
        tickets.sort_unstable();
        assert_eq!(tickets, [1, 2]);
        assert_eq!(&table[3..], [INFLIGHT_NONE, INFLIGHT_NONE]);
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted);
        // Every ticket up to 2 landed: the next one, 3, gets slot 0. Piece
        // 0 starts at the copy's start.
        set(9, content[9] ^ 2);
        let table = run(&mut warp, 2, 3);
        assert_eq!(table, [4, 3, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        let mut piece0 = content[..PIECE - SKEW as usize].to_vec();
        piece0[5] ^= 1;
        piece0[9] ^= 2;
        assert_eq!(slot(PIECE - SKEW as usize), piece0);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn only_an_execution_of_no_records_ends_early() {
        assert!(executes_nothing(0));
        assert!(!executes_nothing(1));
        assert!(!executes_nothing(64));
        assert!(!executes_nothing(u32::MAX));
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }

        // Snapshot `dst_buf` after Call #1 (copy 5 * STRIDE bytes, then leave
        // `dst_buf` readable as arguments again, as the clamp left it).
        // SAFETY: open command list; the clamp left `dst_buf` in `INDIRECT_ARGUMENT`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        // The scratch buffer goes straight from arguments back to unordered
        // access.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            // SAFETY: an open list; the scratch buffer, which the clamp left
            // readable as arguments, is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    /// `with_part`: a dispatch record over 65,535 groups in x is clamped to
    /// its first 65,535, its second part to the next 65,535 and its third to
    /// the rest, which the program translated with each part's group base
    /// runs; a record within the limit has empty further parts. y and z stay
    /// as asked.
    #[test]
    fn warp_a_dispatch_over_the_limit_is_clamped_in_parts() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let records: [[u32; 3]; 3] = [[86_568, 1, 1], [1_000, 2, 1], [140_000, 1, 1]];
        let bytes = (STRIDE as usize) * records.len();
        let (src_buf, src_ptr, _) = warp.shared_buffer(bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(3 * bytes as u64);
        let dst_buf = warp.default_uav(3 * bytes as u64);
        let words: Vec<u8> = records.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), src_ptr, bytes);
            std::ptr::write_bytes(budget_ptr, 0, 256);
        }
        let spec = (KIND_DISPATCH, STRIDE, records.len() as u32);
        let first = record_clamp_at(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON, 0),
            budget_gpu,
            (&src_buf, 0),
            None,
            spec,
            false,
        );
        let second = with_remainder(|| {
            record_clamp_at(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, bytes as u64),
                budget_gpu,
                (&src_buf, 0),
                None,
                spec,
                false,
            )
        });
        let third = with_part(2, || {
            record_clamp_at(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, 2 * bytes as u64),
                budget_gpu,
                (&src_buf, 0),
                None,
                spec,
                false,
            )
        });
        assert!(first && second && third);
        // SAFETY: open command list; the clamped records are copied back.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 3 * bytes as u64);
        }
        warp.submit_and_wait();
        // SAFETY: GPU work completed.
        let out: Vec<u32> = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>(), 3 * bytes / 4).to_vec() };
        assert_eq!(&out[..9], &[65_535, 1, 1, 1_000, 2, 1, 65_535, 1, 1], "the first parts");
        assert_eq!(&out[9..18], &[21_033, 1, 1, 0, 2, 1, 65_535, 1, 1], "the second parts");
        assert_eq!(&out[18..], &[0, 1, 1, 0, 2, 1, 8_930, 1, 1], "the third parts");
    }

    /// `with_split`: the records of a dispatch whose program runs each console thread
    /// group as several host groups along z come out with z multiplied; x and y stay as
    /// asked, a record whose z would pass 65,535 runs empty, and a clamp outside
    /// `with_split` leaves z alone.
    #[test]
    fn warp_a_split_dispatch_has_its_z_multiplied() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let records: [[u32; 3]; 3] = [[240, 135, 1], [7, 3, 4], [1, 1, 40_000]];
        let bytes = (STRIDE as usize) * records.len();
        let (src_buf, src_ptr, _) = warp.shared_buffer(bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(2 * bytes as u64);
        let dst_buf = warp.default_uav(2 * bytes as u64);
        let words: Vec<u8> = records.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), src_ptr, bytes);
            std::ptr::write_bytes(budget_ptr, 0, 256);
        }
        let spec = (KIND_DISPATCH, STRIDE, records.len() as u32);
        let plain = record_clamp_at(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON, 0),
            budget_gpu,
            (&src_buf, 0),
            None,
            spec,
            false,
        );
        let split = with_split(2, || {
            record_clamp_at(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, bytes as u64),
                budget_gpu,
                (&src_buf, 0),
                None,
                spec,
                false,
            )
        });
        assert!(plain && split);
        // SAFETY: open command list; the clamped records are copied back.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 2 * bytes as u64);
        }
        warp.submit_and_wait();
        // SAFETY: GPU work completed.
        let out: Vec<u32> = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>(), 2 * bytes / 4).to_vec() };
        assert_eq!(&out[..9], &[240, 135, 1, 7, 3, 4, 1, 1, 40_000], "as asked without a split");
        assert_eq!(&out[9..], &[240, 135, 2, 7, 3, 8, 0, 0, 0], "z times the split; past 65,535: empty");
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            false,
        ));

        // SAFETY: open command list; copy clamped scratch buffer (left
        // readable as arguments) for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, the scratch buffer's
    /// tracked state, `Host::execute`, and `HostList::reset` end-to-end on
    /// the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`, which stays readable
        // as arguments between them.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        assert_eq!(
            list.indirect_state(&b1),
            args,
            "a clamp leaves the scratch readable as arguments"
        );
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2), args);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let grown = list.indirect_scratch(5000 * 20).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            D3D12_RESOURCE_STATE_COMMON,
            "a new scratch buffer is common"
        );
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        assert_eq!(b3.as_raw(), grown.as_raw());
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3), args);
        assert_eq!(
            list.indirect_state(&b1),
            D3D12_RESOURCE_STATE_COMMON,
            "the replaced buffer went back to common"
        );

        list.close();
        // With the debug layer on (`VICTORY_D3D12_DEBUG=1`, diagnostics):
        // every barrier recorded named the state the buffer was in. Read
        // before the submission, which moves the messages to the log.
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations; the buffer decayed to common when the
        // submission ran.
        assert_eq!(
            list.indirect_state(&b3),
            D3D12_RESOURCE_STATE_COMMON,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4), args);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty. (Racing for the
    /// word record by record, which of them fitted was up to the GPU.)
    #[test]
    fn warp_a_group_charges_its_records_in_lane_order() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;
        const BUDGET_CAP: u32 = 1 << 27;
        let counts = [100u32, 250, 50, 100];
        let total = counts.len() * STRIDE as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            for (i, &count) in counts.iter().enumerate() {
                let words = [count, 1, 7, 9];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STRIDE as usize),
                    16,
                );
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (budget_ptr as *mut u32).write_volatile(BUDGET_CAP - 300);
        }
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DRAW, STRIDE, counts.len() as u32),
            false,
        ));
        // SAFETY: an open list; the clamped records are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                    .to_vec(),
            )
        };
        assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits");
        for i in 1..counts.len() {
            assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
        }
        assert_eq!(spent, BUDGET_CAP, "the record that did not fit took the rest");
    }

    /// The list's budget starts empty at each submission, wherever it
    /// lives (the card's copy is zeroed in the list): a recording whose
    /// clamps spend all of it is followed, after the reset, by one whose
    /// small record still runs.
    #[test]
    fn host_list_budget_starts_empty_at_each_submission() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        // Three records of 65,535 x 16 groups: about half the budget each.
        let Some((big, big_ptr)) = host.shared_buffer(3 * u64::from(STRIDE)) else {
            return;
        };
        let Some((small, small_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            for i in 0..3 {
                let words = [65535u32, 16, 1];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    big_ptr.add(i * STRIDE as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), small_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for round in 0..2 {
            // Spends the budget (clamped, not executed), then one group.
            bound(&list, (&big, 0), None, (KIND_DISPATCH, STRIDE, 3)).unwrap();
            let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
            // SAFETY: an open list; the worker's root signature and
            // pipeline; buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            // The small record comes after the big ones spent the budget:
            // it runs empty, in every round alike (none carries over).
            assert_eq!(hits, 0, "round {round}: the spent budget empties the record after it");
        }
        // A recording with only the small record: the budget is empty
        // again, so it runs.
        let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
        // SAFETY: as above.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, 1, "the budget starts empty at the next submission");
    }

    /// Records the dispatch records `groups` (three words each, `step`
    /// bytes apart, the rest of each record ones) into a fresh mapped
    /// buffer of `warp`'s device.
    fn dispatch_records(warp: &WarpHarness, groups: &[u32], step: u32) -> ID3D12Resource {
        let total = groups.len() * step as usize;
        let (buffer, ptr, _) = warp.shared_buffer(total as u64);
        let mut bytes = vec![0xffu8; total];
        for (i, &g) in groups.iter().enumerate() {
            for (w, word) in [g, 1, 1].into_iter().enumerate() {
                let at = i * step as usize + w * 4;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        // SAFETY: the mapping holds `total` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
        buffer
    }

    /// Two windows in one list, as `crate::draw` records them: each clamps
    /// its records ahead, then one execution per record reads the scratch
    /// buffer with no barrier between them; the second clamp moves the
    /// scratch buffer straight from arguments back to unordered access.
    /// Every record runs once, with the group count it asked for.
    #[test]
    fn warp_window_executions_read_their_records_with_no_barrier_between() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        // The title names 12 bytes and keeps its records 32 apart.
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let sources: Vec<ID3D12Resource> =
            windows.iter().map(|g| dispatch_records(&warp, g, STEP)).collect();
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(u64::from(STEP) * 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let mut state = D3D12_RESOURCE_STATE_COMMON;
        for (groups, src) in windows.iter().zip(&sources) {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, state),
                budget_gpu,
                (src, 0),
                None,
                (KIND_DISPATCH, STEP, groups.len() as u32),
                true,
            ));
            state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: an open list; the worker's root signature and
            // pipeline; the scratch buffer is readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                for i in 0..groups.len() as u64 {
                    warp.list.ExecuteIndirect(&sig, 1, &dst_buf, i * u64::from(STEP), None, 0);
                }
            }
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (hits, spent) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
            )
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        assert_eq!(hits, expected, "every record of both windows runs once");
        assert_eq!(spent, expected * 64, "and is charged what it asked for");
    }

    /// The windows of `warp_window_executions_read_their_records_with_no_barrier_between`
    /// through a host list (`bound_ahead`, the list's tracked scratch state,
    /// `Host::execute`), twice: on the first recording the scratch buffer
    /// starts common, on the second (after the reset) it starts common
    /// again. With the debug layer on (`VICTORY_D3D12_DEBUG=1` in a
    /// diagnostics build) no barrier names a state the buffer is not in.
    #[test]
    fn host_list_window_executions_need_no_barrier_of_their_own() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let mut sources = Vec::new();
        for groups in windows {
            let total = groups.len() * STEP as usize;
            let Some((buffer, ptr)) = host.shared_buffer(total as u64) else {
                return;
            };
            let mut bytes = vec![0xffu8; total];
            for (i, &g) in groups.iter().enumerate() {
                for (w, word) in [g, 1, 1].into_iter().enumerate() {
                    let at = i * STEP as usize + w * 4;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            // SAFETY: the mapping holds `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            for (groups, src) in windows.iter().zip(&sources) {
                let spec = (KIND_DISPATCH, STEP, groups.len() as u32);
                let scratch = bound_ahead(&list, (src, 0), spec).unwrap();
                assert_eq!(list.indirect_state(&scratch), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    for i in 0..groups.len() as u64 {
                        let at = i * u64::from(STEP);
                        list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0);
                    }
                }
            }
            list.close();
            // With the debug layer on: nothing it objects to was recorded.
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every record of both windows runs once");
        }
    }

    /// Two windows clamped into parts of their own of the scratch buffer
    /// (`bound_ahead_at`, as `crate::draw`'s wide windows do), then a clamp
    /// of an execution that is not a window after both: the first window's
    /// records are read after the later clamps and still run, each once.
    #[test]
    fn windows_in_parts_of_their_own_outlive_later_clamps() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        // As `crate::draw`'s parts: 16 KiB a window, the rest after 128 KiB.
        const PART: u64 = 16 << 10;
        const REST: u64 = 8 * PART;
        const ROOM: u64 = REST + (64 << 10);
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        let windows: [&[u32]; 3] = [&[1, 2, 3, 4], &[10, 11, 12, 13], &[100]];
        let mut sources = Vec::new();
        for groups in windows {
            let total = groups.len() * STEP as usize;
            let Some((buffer, ptr)) = host.shared_buffer(total as u64) else {
                return;
            };
            let mut bytes = vec![0xffu8; total];
            for (i, &g) in groups.iter().enumerate() {
                for (w, word) in [g, 1, 1].into_iter().enumerate() {
                    let at = i * STEP as usize + w * 4;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            // SAFETY: the mapping holds `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        // SAFETY: the mapping holds 256 bytes; nothing runs.
        unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
        // Window 0 in part 0, window 1 in part 1, then the third, not a
        // window, after the parts.
        let spec = |g: &[u32]| (KIND_DISPATCH, STEP, g.len() as u32);
        let first = bound_ahead_at(&list, (&sources[0], 0), spec(windows[0]), (0, ROOM)).unwrap();
        let second = bound_ahead_at(&list, (&sources[1], 0), spec(windows[1]), (PART, ROOM)).unwrap();
        let third = bound_at(&list, (&sources[2], 0), None, spec(windows[2]), (REST, ROOM)).unwrap();
        assert!(first == second && second == third, "one scratch buffer for all parts");
        assert_eq!(list.indirect_state(&first), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        // SAFETY: an open list; the worker's root signature and pipeline;
        // buffers the test holds until the list has run.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            for (part, groups) in [0, PART, REST].into_iter().zip(windows) {
                for i in 0..groups.len() as u64 {
                    list.list.ExecuteIndirect(&signature, 1, &first, part + i * u64::from(STEP), None, 0);
                }
            }
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, expected, "every record of every part runs once, the first after both later clamps");
    }

    /// A command signature of one argument of `kind` on `device`.
    fn one_argument(device: &ID3D12Device, kind: u32, stride: u32) -> ID3D12CommandSignature {
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: a valid single-argument command signature description.
        unsafe {
            device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig).unwrap();
        }
        sig.unwrap()
    }

    /// What a batch's table says of records added to it
    /// (`VICTORY_INDIRECT_BATCH`): one job per record, charged for every
    /// execution that reads it, and none for what the pass cannot read.
    /// (avatarmanz/VIctory-7900XTX 037596241.)
    #[test]
    fn batch_tables() {
        use batch::{JOBS, SLOT, TABLE, TIMES, Table, job};
        let mut memory = vec![0xffff_ffffu32; TABLE as usize / 4 + 1];
        // SAFETY: the vector holds the table and outlives its use.
        let mut t = unsafe { Table::new(memory.as_mut_ptr().cast(), 4096, [7, 9, 0, 0]) };
        assert_eq!(t.jobs(), 0);
        assert_eq!(t.place(7, 64, 10), Some(4096));
        assert_eq!(t.place(9, 64, 10), Some(4096 + u64::from(SLOT)));
        assert_eq!(t.place(7, 64, 2), Some(4096 + 2 * u64::from(SLOT)), "another kind is another record");
        // The same record again: the same place, charged twice.
        assert_eq!(t.place(9, 64, 10), Some(4096 + u64::from(SLOT)));
        assert_eq!(t.jobs(), 3);
        assert_eq!(&memory[..4], [3, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]);
        assert_eq!(memory[4..8], [job(0, 10, 1), 64, 4096, 0]);
        assert_eq!(memory[8..12], [job(1, 10, 2), 64, 4096 + SLOT, 0]);
        assert_eq!(job(1, 10, 2), 1 | 10 << 4 | 3 << 8 | 2 << 16);
        // SAFETY: as above.
        let mut t = unsafe { Table::new(memory.as_mut_ptr().cast(), 4096, [7, 9, 0, 0]) };
        assert_eq!(memory[0], 0, "a new table has no job");
        assert_eq!(t.place(7, 64, 10), Some(4096));
        assert_eq!(t.place(9, 64, 10), Some(4096 + u64::from(SLOT)));
        assert_eq!(t.place(7, 64, 2), Some(4096 + 2 * u64::from(SLOT)));
        // Not a copy of the pass, not on 4 bytes, past 32 bits, not a kind.
        for (id, offset, kind) in [(8, 64, 10), (7, 66, 10), (7, 1 << 32, 10), (7, u64::from(u32::MAX) & !3, 10), (7, 64, 3)] {
            assert_eq!(t.place(id, offset, kind), None);
        }
        // Read more often than one job may be charged for: another job.
        for _ in 1..TIMES {
            assert_eq!(t.place(7, 64, 10), Some(4096));
        }
        assert_eq!(t.place(7, 64, 10), Some(4096 + 3 * u64::from(SLOT)));
        assert_eq!(t.place(7, 64, 10), Some(4096 + 3 * u64::from(SLOT)));
        for i in 4..u64::from(JOBS) {
            assert_eq!(t.place(7, 1024 + 16 * i, 0), Some(4096 + i * u64::from(SLOT)));
        }
        assert_eq!(t.place(7, 1024, 1), None, "full");
        assert_eq!(t.place(7, 1024 + 16 * 5, 0), Some(4096 + 5 * u64::from(SLOT)));
        assert_eq!((memory[0], memory[TABLE as usize / 4]), (JOBS, 0xffff_ffff));
    }

    /// A dispatch's record is a job of each pipeline that reads it, noted
    /// with the execution that asked first (the fourth word): what the
    /// clamp of one execution is told ([`with_dispatch`]).
    #[test]
    fn batch_tables_note_dispatch_records_by_who_reads_them() {
        use batch::{SLOT, TABLE, Table, job};
        let mut memory = vec![0u32; TABLE as usize / 4];
        // SAFETY: the vector holds the table and outlives its use.
        let mut t = unsafe { Table::new(memory.as_mut_ptr().cast(), 0, [7, 0, 0, 0]) };
        let mut made = 0;
        let mut next = || {
            made += 1;
            0x8000_0000 | made
        };
        assert_eq!(t.place_noted(7, 128, 2, 0xA, &mut next), Some(0));
        // The same pipeline again: the same job, no new execution noted.
        assert_eq!(t.place_noted(7, 128, 2, 0xA, &mut next), Some(0));
        // Another pipeline reads the same record: a job and a note of its own.
        assert_eq!(t.place_noted(7, 128, 2, 0xB, &mut next), Some(u64::from(SLOT)));
        assert_eq!(memory[0], 2);
        assert_eq!(memory[4..8], [job(0, 2, 2), 128, 0, 0x8000_0001]);
        assert_eq!(memory[8..12], [job(0, 2, 1), 128, SLOT, 0x8000_0002]);
        assert_eq!(made, 2);
    }

    /// A batch clamps records exactly as the clamp of one execution does, of
    /// any kind, sensible or not; they are added to its table after the pass
    /// was recorded; and it leaves the list's budget and every note beside
    /// it (mesh groups, dispatch groups, records cut, the watched ones) as
    /// that clamp does. (After avatarmanz/VIctory-7900XTX 037596241.)
    #[test]
    fn warp_batch_clamps_as_the_clamp_of_one_does() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let pass = make_from(&warp.device, batch::SOURCE).expect("the batched clamp must compile on WARP");
        let kinds = [0u32, 1, 2, 10];
        let values = [0u32, 1, 3, 100, 65_535, 65_536, 1 << 20, 1 << 24, (1 << 26) + 1, u32::MAX];
        // Every kind with many records, in two sources.
        let mut records: Vec<(u32, [u32; 5])> = Vec::new();
        for kind in kinds {
            for (i, a) in values.into_iter().enumerate() {
                for (k, b) in values.into_iter().enumerate() {
                    records.push((kind, [a, b, values[(i * 3 + k) % 4], 77, 0xABCD]));
                }
            }
        }
        let (src, src_ptr, src_gpu) = warp.shared_buffer(64 << 10);
        let (_other, other_ptr, other_gpu) = warp.shared_buffer(64 << 10);
        for (i, (_, words)) in records.iter().enumerate() {
            for (w, word) in words.iter().enumerate() {
                // SAFETY: 400 records of 32 bytes in 64 KiB.
                unsafe {
                    src_ptr.cast::<u32>().add(i * 8 + w).write_volatile(*word);
                    other_ptr.cast::<u32>().add(i * 8 + w).write_volatile(*word);
                }
            }
        }
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(64 << 10);
        let dst = warp.default_uav(64 << 10);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        for (i, (kind, _)) in records.iter().enumerate().step_by(3) {
            let offset = i as u64 * 32;
            let words = batch::words(*kind).unwrap();
            let stride = words * 4;
            // A dispatch's execution: its number, the parts its pipeline
            // runs in (one or two) and whether it is watched.
            let execution = if *kind == 2 { (i as u32 & 0x3fff) | ((i as u32 / 3 % 2) << 28) | ((i as u32 / 6 % 2) << 31) } else { 0 };
            let read = |p: *mut u8| -> Vec<u32> {
                // SAFETY: the mapping holds more.
                (0..words as usize).map(|w| unsafe { p.cast::<u32>().add(w).read_volatile() }).collect()
            };
            let mut seen = Vec::new();
            for batched in [false, true] {
                // SAFETY: mapped buffers; nothing runs.
                unsafe { std::ptr::write_bytes(budget_ptr, 0, 256) };
                if batched {
                    batch::record(&pass, &warp.list, (&dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, [src_gpu, other_gpu, src_gpu, src_gpu], table_gpu);
                    // SAFETY: the mapped table; the list has not run.
                    let mut table = unsafe { batch::Table::new(table_ptr, 0, [1, 2, 0, 0]) };
                    assert_eq!(table.place_noted(1 + i as u64 % 2, offset, *kind, 0, || execution), Some(0));
                } else {
                    let was = EXECUTION.replace(execution);
                    assert!(record_clamp_as(
                        &warp.clamp,
                        &warp.list,
                        (&dst, D3D12_RESOURCE_STATE_COMMON),
                        budget_gpu,
                        (&src, offset),
                        None,
                        (*kind, stride, 1),
                        false,
                    ));
                    EXECUTION.set(was);
                }
                // SAFETY: an open list; the records are readable as arguments.
                unsafe {
                    warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    warp.list.CopyBufferRegion(&back, 0, &dst, 0, u64::from(stride));
                    warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                }
                warp.submit_and_wait();
                // SAFETY: the list has run; mapped words.
                let budget: Vec<u32> = (0..11).map(|k| unsafe { budget_ptr.cast::<u32>().add(k).read_volatile() }).collect();
                seen.push((read(back_ptr), budget));
            }
            assert_eq!(seen[0], seen[1], "record {i} of kind {kind}: {:?} (clamped alone, then in a batch)", records[i].1);
        }
    }

    /// Records past the list's budget come out empty, and one read by many
    /// executions is charged for each. (avatarmanz/VIctory-7900XTX 037596241.)
    #[test]
    fn warp_batch_charges_every_execution() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let pass = make_from(&warp.device, batch::SOURCE).expect("the batched clamp must compile on WARP");
        const BUDGET: u32 = 1 << 27;
        // 2^26 vertices each, as much as a draw is let run: two fill it.
        for (times, records, filled) in [(1u32, 3u64, 2usize), (2, 1, 1), (1, 1, 1), (3, 1, 0)] {
            let (_src, src_ptr, src_gpu) = warp.shared_buffer(64 << 10);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (back, back_ptr, _) = warp.shared_buffer(1024);
            let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
            let dst = warp.default_uav(64 << 10);
            batch::record(&pass, &warp.list, (&dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, [src_gpu; 4], table_gpu);
            // SAFETY: mapped buffers, which the list has not read yet.
            let mut table = unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
                for r in 0..records {
                    for (w, word) in [1u32 << 24, 4, 5, 6].into_iter().enumerate() {
                        src_ptr.cast::<u32>().add(r as usize * 4 + w).write_volatile(word);
                    }
                }
                batch::Table::new(table_ptr, 0, [1, 0, 0, 0])
            };
            for _ in 0..times {
                for r in 0..records {
                    assert_eq!(table.place(1, r * 16, 0), Some(r * u64::from(batch::SLOT)));
                }
            }
            // SAFETY: an open list; the records are readable as arguments.
            unsafe {
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&back, 0, &dst, 0, 256);
            }
            warp.submit_and_wait();
            // SAFETY: the list has run; mapped words.
            let (spent, out): (u32, Vec<u32>) = unsafe {
                (
                    budget_ptr.cast::<u32>().read_volatile(),
                    (0..records as usize).map(|r| back_ptr.cast::<u32>().add(r * batch::SLOT as usize / 4).read_volatile()).collect(),
                )
            };
            assert_eq!(out.iter().filter(|&&v| v == 1 << 24).count(), filled, "{times} times {records}: {out:?}");
            assert!(out.iter().all(|&v| v == 0 || v == 1 << 24));
            assert_eq!(spent, BUDGET.min((1 << 26) * times * records as u32));
        }
    }

    /// A pass of many jobs (more than one group of its threads), filled
    /// after it was recorded and out of order: every job's record lands at
    /// its own place, bounded, whatever the jobs beside it are.
    #[test]
    fn warp_batch_of_many_jobs_bounds_each_at_its_place() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let pass = make_from(&warp.device, batch::SOURCE).expect("the batched clamp must compile on WARP");
        const RECORDS: u32 = 200;
        let (_src, src_ptr, src_gpu) = warp.shared_buffer(64 << 10);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(64 << 10);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let dst = warp.default_uav(64 << 10);
        // Mesh records 32 bytes apart: record `r` asks for `r + 1` groups,
        // every fiftieth for all ones (a slot the title has not filled), which
        // is bounded to the most a mesh dispatch runs; all fit the budget.
        let asked = |r: u32| if r % 50 == 49 { u32::MAX } else { r + 1 };
        // SAFETY: mapped buffers, which no list reads yet.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            for r in 0..RECORDS {
                for (w, word) in [asked(r), 1, 1].into_iter().enumerate() {
                    src_ptr.cast::<u32>().add(r as usize * 8 + w).write_volatile(word);
                }
            }
        }
        batch::record(&pass, &warp.list, (&dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, [src_gpu; 4], table_gpu);
        // SAFETY: the mapped table; the list has not run.
        let mut table = unsafe { batch::Table::new(table_ptr, 0, [5, 0, 0, 0]) };
        // Backwards, as the title draws some of its buffers.
        let places: Vec<(u32, u64)> = (0..RECORDS).rev().map(|r| (r, table.place(5, u64::from(r) * 32, 10).unwrap())).collect();
        assert_eq!(table.jobs(), RECORDS);
        // SAFETY: an open list; the records are readable as arguments.
        unsafe {
            warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            warp.list.CopyBufferRegion(&back, 0, &dst, 0, u64::from(RECORDS * batch::SLOT));
        }
        warp.submit_and_wait();
        let mut spent = 0u32;
        for (r, at) in places {
            // SAFETY: the list has run; mapped words inside the copy.
            let got: Vec<u32> = (0..3).map(|w| unsafe { back_ptr.cast::<u32>().add(at as usize / 4 + w).read_volatile() }).collect();
            let groups = asked(r).min(65_535);
            assert_eq!(got, [groups, 1, 1], "record {r} at {at}");
            spent += groups * 128;
        }
        // SAFETY: as above.
        assert_eq!(unsafe { budget_ptr.cast::<u32>().read_volatile() }, spent, "each charged once");
    }

    /// Two batches through a host list, twice (the second recording finds
    /// the scratch buffer as the first one's end left it): executions with
    /// nothing between them read what they asked for, of two source buffers,
    /// the second batch while the first one's places are still read. With
    /// the debug layer on no barrier names a state the buffer is not in.
    /// (avatarmanz/VIctory-7900XTX 037596241.)
    #[test]
    fn host_list_batches() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        let signature = one_argument(&host.device, 2, 12);
        let mut sources = Vec::new();
        for first in [1u32, 100] {
            let Some((buffer, ptr)) = host.shared_buffer(1024) else {
                return;
            };
            for i in 0..16u32 {
                for (w, word) in [first + i, 1, 1].into_iter().enumerate() {
                    // SAFETY: inside the mapping.
                    unsafe { ptr.cast::<u32>().add(i as usize * 8 + w).write_volatile(word) };
                }
            }
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            let mut expected = 0;
            // More batches than one scratch buffer holds.
            for k in 0..12u32 {
                let (mut table, scratch) =
                    open_batch(&list, (1, &sources[0]), &[(1, sources[0].clone()), (2, sources[1].clone())]).unwrap();
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                }
                for (id, record) in [(1u64, 3 + k % 4), (2, 5), (1, 0), (2, 5)] {
                    let at = table.place(id, u64::from(record) * 32, 2).unwrap();
                    // SAFETY: as above.
                    unsafe { list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0) };
                    expected += record + if id == 1 { 1 } else { 100 };
                }
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every execution runs what it asked for");
        }
    }

    /// A clamp of one execution in the middle of a batch (a count buffer, a
    /// dispatch in parts: what batches leave alone) uses the list's other
    /// scratch buffer: the batch's records are read after it and still run.
    #[test]
    fn host_list_batch_outlives_a_clamp_of_one() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        let signature = one_argument(&host.device, 2, 12);
        let apart = one_argument(&host.device, 2, 32);
        let Some((source, ptr)) = host.shared_buffer(1024) else {
            return;
        };
        for i in 0..16u32 {
            for (w, word) in [i + 1, 1, 1].into_iter().enumerate() {
                // SAFETY: inside the mapping.
                unsafe { ptr.cast::<u32>().add(i as usize * 8 + w).write_volatile(word) };
            }
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        // SAFETY: the mapping holds 256 bytes; nothing runs.
        unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
        let bind = || {
            // SAFETY: an open list; the worker's root signature and pipeline;
            // buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            }
        };
        let (mut table, scratch) = open_batch(&list, (1, &source), &[]).unwrap();
        bind();
        let first = table.place(1, 2 * 32, 2).unwrap();
        // SAFETY: as in `bind`.
        unsafe { list.list.ExecuteIndirect(&signature, 1, &scratch, first, None, 0) };
        // One clamped on its own: records 4 and 5 (5 + 6 groups).
        let alone = bound(&list, (&source, 4 * 32), None, (2, 32, 2)).unwrap();
        assert!(alone != scratch, "not the batches' buffer");
        bind();
        // SAFETY: as in `bind`.
        unsafe { list.list.ExecuteIndirect(&apart, 2, &alone, 0, None, 0) };
        // The batch goes on: another record, and the first one again.
        let later = table.place(1, 9 * 32, 2).unwrap();
        assert_eq!(table.place(1, 2 * 32, 2), Some(first));
        // SAFETY: as in `bind`.
        unsafe {
            list.list.ExecuteIndirect(&signature, 1, &scratch, later, None, 0);
            list.list.ExecuteIndirect(&signature, 1, &scratch, first, None, 0);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, 3 + (5 + 6) + 10 + 3);
    }

    /// What the clamps of a frame cost on this card, each on its own and in
    /// batches (a measurement, ignored: run it by name with `--ignored
    /// --nocapture`): 2,400 indirect dispatches of one group that does next
    /// to nothing, through a host list, as many as a frame of the boat
    /// scene clamps. A batch every 1 is what `VICTORY_INDIRECT_BATCH=1`
    /// leaves of a run of dispatches; every 16 or 256, what runs of draws
    /// (or of dispatches with `=2`) come to. (After the measurement of
    /// avatarmanz/VIctory-7900XTX 037596241, `batch_costs`.)
    ///
    /// On the RTX 4060 Ti (a release build, the best of 8 submissions, twice,
    /// with the test queue's game running beside it): each on its own 8.2
    /// to 10.5 ms (3.4 to 4.4 microseconds a clamp); a batch every 1: 11.0
    /// to 11.6 ms (a pass of a batch costs more than a clamp of one, so
    /// runs of dispatches gain nothing with `=1`); every 3: 5.3; every 16:
    /// 2.2; every 256: 1.6 ms. A table of 64 jobs (one group of threads)
    /// instead of 256 changed nothing (11.0 ms every 1). The dispatches
    /// here do nothing: in a frame each pass also waits for real work.
    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn batch_costs_on_this_card() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        const EXECUTIONS: u32 = 2400;
        const RECORDS: u32 = 2048;
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        let signature = one_argument(&host.device, 2, 12);
        let Some((source, ptr)) = host.shared_buffer(u64::from(RECORDS) * 32) else {
            return;
        };
        for i in 0..RECORDS as usize {
            for w in 0..3 {
                // SAFETY: inside the mapping.
                unsafe { ptr.cast::<u32>().add(i * 8 + w).write_volatile(1) };
            }
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        let bind = || {
            // SAFETY: an open list; the worker's root signature and pipeline;
            // buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            }
        };
        for run in [0u32, 1, 3, 16, 256, 0, 1, 16] {
            let mut best = f64::MAX;
            for _ in 0..8 {
                // SAFETY: the mapping holds 256 bytes; nothing runs.
                unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
                let mut batch = None;
                for i in 0..EXECUTIONS {
                    let offset = u64::from(i % RECORDS) * 32;
                    let (scratch, at) = if run == 0 {
                        let scratch = bound(&list, (&source, offset), None, (2, 12, 1)).unwrap();
                        bind();
                        (scratch, 0)
                    } else {
                        if i % run == 0 {
                            batch = Some(open_batch(&list, (1, &source), &[]).unwrap());
                            bind();
                        }
                        let (table, scratch) = batch.as_mut().unwrap();
                        (scratch.clone(), table.place(1, offset, 2).unwrap())
                    };
                    // SAFETY: as in `bind`.
                    unsafe { list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0) };
                }
                list.close();
                let started = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(started.elapsed().as_secs_f64() * 1e3);
                list.reset();
                // SAFETY: the list has run; a mapped buffer.
                assert_eq!(unsafe { (hits_ptr as *const u32).read_volatile() }, EXECUTIONS);
            }
            let what = if run == 0 { "each clamped on its own".to_owned() } else { format!("a batch every {run}") };
            println!("{EXECUTIONS} indirect dispatches, {what}: {best:.2} ms (the best of 8 submissions)");
        }
    }
}

/// The benchmark of the clamp and its barriers (ignored by default).
#[cfg(test)]
mod bench;

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        /// Kept with the bench: the clears above wrote through it.
        _depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    _depth_written: depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
