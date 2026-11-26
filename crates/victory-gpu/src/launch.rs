//! Indirect mesh records bounded inside the draw: an amplification shader
//! of ours (`launch.hlsl`, embedded as bytes in `launch_code.rs`) in front
//! of every translated mesh shader (`VICTORY_MESH_CLAMP_IN_DRAW=1` or `2`;
//! off by default: each record is clamped by a pass of `crate::indirect`,
//! alone, in a window or in a batch, as before).
//!
//! Ported from fablinoo/kelvaroq native/patches/0041-mesh-launch-wide-hash.patch
//! (GPL-3.0; their `VICTORY_MESH_LAUNCH`), `launch.rs` and `launch.hlsl`.
//! The shader bytes are theirs, and are what the Windows SDK's dxc
//! (10.0.26100.0) makes of `launch.hlsl` here, byte for byte.
//!
//! The title draws its meshes one indirect record a call, each from a slot
//! of an argument buffer its own compute passes wrote. Each such record was
//! bounded by a compute dispatch of ours between two waits for every earlier
//! shader write (`crate::indirect::bound_at`): the card went from drawing to
//! compute and back, emptying itself twice, for every record (windows and
//! batches share one pass among several records, but a draw that follows a
//! pass still waits for it).
//!
//! Here the record is read by the draw itself: the pipeline's amplification
//! stage (one group of one thread) loads the three words where the title's
//! passes wrote them, bounds them as the clamp bounds a mesh record (65,535
//! groups, one-dimensional), charges the list's budget as the clamp does (a
//! record that does not fit whole runs empty and takes the rest), notes
//! the largest count asked and the groups asked in the budget's words 1 and
//! 2 as the clamp does, and launches the mesh groups. The title's own
//! barriers (which `crate::list` records as waits for every shader write)
//! order its passes before the draw. With `=2` a wait of ours for every
//! shader write is recorded before a launched record when a dispatch or a
//! draw whose shaders store has been recorded in the list since the last
//! one ([`State::launch_wait`](crate::draw)): what the clamp's first wait
//! ordered, without the pass, the second wait and the state changes.
//!
//! A mesh dispatch the CPU knows (`DispatchMesh`) gives its count as a root
//! constant and is launched the same way, unbounded by the budget as
//! before. An indirect call with a count buffer or more than one record
//! keeps the clamp and runs on the same pipeline made without the stage
//! ([`crate::pipeline::host::Launch::plain`], made the first time one
//! needs it): the translated mesh shader declares the payload either way,
//! which the driver takes with or without an amplification shader.
//!
//! What changes: the amplification groups of different draws may run in
//! another order than the list's, so when a list's budget runs out, which
//! of its records run empty is the card's order and not the list's; the
//! bound on the list's total work (the guard against a list that runs for
//! seconds) is the same.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;

#[path = "launch_code.rs"]
mod launch_code;

/// The amplification shader's code (see `launch.hlsl`).
const CODE: &[u8] = launch_code::CODE;

/// The root constant that has the card read the record.
pub(crate) const DIRECT_NONE: u32 = u32::MAX;

/// Where the amplification shader binds: `b0`, `t0`, `u0` in this space.
const SPACE: u32 = victory_shader::emit::LAUNCH_SPACE;

/// Whether mesh pipelines get the amplification stage and single indirect
/// mesh records are bounded by it (`VICTORY_MESH_CLAMP_IN_DRAW=1` or `2`).
/// The same switch has translated mesh shaders declare the payload.
pub(crate) fn on() -> bool {
    victory_shader::emit::mesh_clamp_in_draw() != 0
}

/// `VICTORY_MESH_CLAMP_IN_DRAW=2`: a wait for every earlier shader write
/// before a launched record when work that can store ran since the last.
pub(crate) fn waits() -> bool {
    victory_shader::emit::mesh_clamp_in_draw() == 2
}

/// The amplification stage's code.
pub(crate) fn code() -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: CODE.as_ptr().cast(), BytecodeLength: CODE.len() }
}

/// The bytes of [`code`], for the pipeline library's names.
pub(crate) fn bytes() -> &'static [u8] {
    CODE
}

/// The three root parameters the stage reads, in order: the direct group
/// count (one constant), the record (a root SRV), the list's budget (a
/// root UAV). Five root words.
pub(crate) fn params() -> [D3D12_ROOT_PARAMETER1; 3] {
    let descriptor = |kind| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
    };
    [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: SPACE, Num32BitValues: 1 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV),
    ]
}

/// Mesh draws launched since the last line: records read by the card,
/// dispatches whose count the CPU gave, waits of ours before records
/// (`=2`), and indirect mesh calls the stage does not take (clamped, on the
/// pipeline without it).
static LAUNCHED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
pub(crate) const L_RECORD: usize = 0;
pub(crate) const L_DIRECT: usize = 1;
pub(crate) const L_WAIT: usize = 2;
pub(crate) const L_PLAIN: usize = 3;

pub(crate) fn count(what: usize) {
    LAUNCHED[what].fetch_add(1, Ordering::Relaxed);
}

/// Records the launch of one mesh draw, the mesh pipeline with its stage
/// bound and the stage's parameters at `roots`: `Record` the record's
/// address on the card (the card reads it), or the group count the CPU
/// knows. False, with nothing recorded, if the list has no budget for a
/// record or no mesh dispatches.
pub(crate) fn record(list: &HostList, roots: [u32; 3], what: Launch) -> bool {
    record_into(&list.list, list.budget(), roots, what)
}

/// [`record`] into a plain command list with the budget at `budget`.
pub(crate) fn record_into(
    list: &ID3D12GraphicsCommandList,
    budget: Option<u64>,
    roots: [u32; 3],
    what: Launch,
) -> bool {
    // Bindings outside the draw state's memory of them (`VICTORY_SET_CACHE`).
    let _binds = crate::draw::sets::Announce;
    let Ok(l6) = windows::core::Interface::cast::<ID3D12GraphicsCommandList6>(list) else {
        return false;
    };
    let (direct, record, budget) = match what {
        Launch::Record(address) => {
            let Some(budget) = budget else {
                return false;
            };
            (DIRECT_NONE, address, budget)
        }
        // The stage reads neither the record nor the budget; live
        // addresses all the same where the list has them.
        Launch::Direct(x) => (x, budget.unwrap_or(0), budget.unwrap_or(0)),
    };
    // SAFETY: an open list with the mesh pipeline and its root signature
    // set; the record and the budget are live buffers the list keeps.
    unsafe {
        list.SetGraphicsRoot32BitConstant(roots[0], direct, 0);
        list.SetGraphicsRootShaderResourceView(roots[1], record);
        list.SetGraphicsRootUnorderedAccessView(roots[2], budget);
        l6.DispatchMesh(1, 1, 1);
    }
    count(if matches!(what, Launch::Direct(_)) { L_DIRECT } else { L_RECORD });
    true
}

/// What one launch reads.
#[derive(Clone, Copy)]
pub(crate) enum Launch {
    /// The record at this address on the card.
    Record(u64),
    /// A group count known on the CPU.
    Direct(u32),
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [records, direct, waits, plain] = [L_RECORD, L_DIRECT, L_WAIT, L_PLAIN].map(|i| LAUNCHED[i].swap(0, Ordering::Relaxed));
    tracing::info!(
        "mesh draws bounded in the draw per frame (VICTORY_MESH_CLAMP_IN_DRAW={}): {:.0} records read where the title \
         wrote them, {:.0} counts known on the CPU, {:.0} waits of ours before records (=2), {:.0} indirect mesh calls \
         clamped as before (a count buffer or several records)",
        victory_shader::emit::mesh_clamp_in_draw(),
        records as f64 / frames,
        direct as f64 / frames,
        waits as f64 / frames,
        plain as f64 / frames
    );
}
