//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// The loops a per-lane branch leaves (`VICTORY_AFTER_LOOP=1`), and a console wave's
/// operations after a loop its lanes leave apart, against the console's results.
#[path = "wave64_after_loop.rs"]
mod after_loop;

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: Vec::new(),
        group_split: 1,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

#[test]
fn local_waves_select_their_console_bits_from_a_constant_mask() {
    let words = [
        0xD760_000E,
        0x0001_0100, // v_readlane_b32 s14, v0, 0: require console lane numbering
        0xBE8C_0380, // s_mov_b32 s12, 0
        0xBE8D_03C1, // s_mov_b32 s13, -1
        0xD501_0001,
        0x0031_0280, // v_cndmask_b32 v1, 0, 1, s[12:13]
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lower::with_local_waves_any(|| lowered(&native, &p, &HashSet::new()))
        .expect("lower a constant mask on two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords() as usize]))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch the constant mask");
    assert_eq!(&out[0][..32], &[0; 32]);
    assert_eq!(&out[0][32..], &[1; 32]);
    eprintln!("ran on the hardware device: each half selected its own mask bits");
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
const MASK_OR_NUMBER: [u32; 8] = [
    0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
    0xBE94_036A, // s_mov_b32 s20, vcc_lo
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0001, // s_cbranch_scc1 over the next
    0xBE94_0385, // s_mov_b32 s20, 5
    0x7E00_0214, // v_mov_b32 v0, s20
    0x38 << 26 | 0x1C << 18,
    0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
];

/// The program above reads a lane mask as a number (on the path that kept it), so it needs
/// the console's 32 lanes in a host wave: it declares `WaveSize(32)`. Without it, a host of
/// 16-lane waves gave only 16 of the mask's bits (#356).
#[test]
fn a_lane_mask_joined_with_a_number_needs_the_consoles_wave_size() {
    let native = native_of(&MASK_OR_NUMBER, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    assert_eq!(lowered.expect("lower").wave_size, Some(32));
}

#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let native = native_of(&MASK_OR_NUMBER, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// Edge threading (`ir::thread_edges`) can resolve an unsupported register
/// error on a wave64 compute shader that still needs the 32-lane retry passes
/// (`without_skips` / `local_waves`). Calling `emit_once` after threading lets
/// those retries run instead of failing or emitting `WaveSize(64)`.
#[test]
fn threaded_wave64_compute_still_runs_wave32_retries() {
    // Uniform triangle on s0 == 0:
    //   fork: v1 = 0; if (s0 == 0) jump join (ttmp0 uninitialized on this edge)
    //   arm:  ttmp0 = 5; v1 = 1; fallthrough join
    // join: if (v1 == 0) branch S_ENDPGM
    // body: if (t < s1) { v_readlane_b32 s7, v0, 0 } (needs without_skips on wave32);
    //       store ttmp0 + s7 to UAV.
    let words = [
        0x7E02_0280, // 0: v_mov_b32 v1, 0
        0xBF06_8000, // 1: s_cmp_eq_u32 s0, 0
        0xBF85_0002, // 2: s_cbranch_scc1 +2 -> word 5 (join)
        0xBEEC_0385, // 3: s_mov_b32 ttmp0 (108), 5
        0x7E02_0281, // 4: v_mov_b32 v1, 1
        0x7D84_0280, // 5: join: v_cmp_eq_u32 vcc, 0, v1
        0xBF87_000B, // 6: s_cbranch_vccnz +11 -> S_ENDPGM (word 18)
        0xBE87_0380, // 7: body: s_mov_b32 s7, 0
        0x7D88_0001, // 8: v_cmp_gt_u32 vcc, s1, v0
        0xBE8C_246A, // 9: s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // 10: s_cbranch_execz +2 -> word 13
        0xD760_0007, // 11: v_readlane_b32 s7, v0, 0 (2 dwords)
        0x0001_0100, // 12:
        0xBEFE_040C, // 13: s_mov_b64 exec, s[12:13]
        0x8006_076C, // 14: s_add_u32 s6, ttmp0 (108), s7
        0x7E04_0206, // 15: v_mov_b32 v2, s6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 2 << 8, // 16-17: buffer_store_dword v2, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let edges = super::threading::constant_edges(&p, &st);
    assert!(!edges.is_empty(), "test program must trigger constant-edge threading");
    let io = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();
    // Confirm the threaded program alone still requires the wave32 retry path.
    let tp = ir::thread_edges(&p, &edges).expect("thread_edges");
    let tst = structure::structure(&tp).expect("threaded structure");
    let tio = io::raise(&tp, &native).expect("threaded io");
    let tres = resource::track(&tp, &native.layout, native.stage, Some(&tst)).expect("threaded res");
    let (plain_threaded, shader) = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(32));
        let plain = super::emit_program(&empty, &native, &tp, &tst, &tio, &tres);
        let full = super::emit(&empty, &native, &p, &st, &io, &res);
        cell.set(was);
        (plain, full)
    });
    assert!(
        matches!(
            &plain_threaded,
            Ok((_, Some(64))) | Err(ShaderError::Unsupported { .. })
        ),
        "expected plain threaded emit_program to need wave32 retries, got {:?}",
        plain_threaded.as_ref().map(|(_, w)| w)
    );
    let shader = shader.expect("threaded wave64 compute shader should succeed via wave32 retries");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
}

/// A 32-lane console wave. A loop counts s6 up from 1 and leaves on `s_cbranch_vccz` of
/// `v_cmp_eq_u32 vcc, s6, v0`: on the console VCC is tested for the whole wave, so the wave
/// leaves at the first count some lane meets (lane 1, at s6 = 1) and every lane stores 1.
/// Taken as each lane's own bit, lane 0 (whose index s6 never reaches) would keep looping up
/// to the iteration limit and store that count instead (the clustered-light pixel shaders,
/// fxceless #196). Every lane stores s6 at its own dword; lane 0's is checked.
#[test]
fn a_loop_on_a_vcc_test_of_a_lane_compare_leaves_with_the_wave() {
    let words = [
        0xBE86_0380, // s_mov_b32 s6, 0
        0x8006_8106, // loop: s_add_u32 s6, s6, 1
        0x7D84_0006, // v_cmp_eq_u32 vcc, s6, v0
        0xBF86_FFFD, // s_cbranch_vccz loop (no lane met it: once more)
        0x7E04_0206, // v_mov_b32 v2, s6
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 3, // buffer_store_dword v2, v3, s[8:11], 0 offen
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = on_the_card(&shader, [0, 0]) else { return };
    assert_eq!(got, 1, "lane 0 left the loop at count {got}, not with the wave at 1");
    eprintln!("ran on the hardware device: the wave left the loop together");
}

/// The wave size a translation asks of the host (PSV0's minimum and maximum lane counts).
fn psv0_wave(shader: &HostShader) -> (u32, u32) {
    let c = crate::Container::parse(&shader.dxil).expect("container");
    let d = c.part(crate::FourCC::PSV0).expect("PSV0").data;
    let at = |k: usize| u32::from_le_bytes(d[k..k + 4].try_into().unwrap());
    (at(20), at(24))
}

/// `VICTORY_WAVE_WIDE=last` keeps every form an earlier retry takes: a console wave64
/// program with a wave OR inside a per-lane branch ([`program`]) and one with an LDS write
/// and read inside a per-lane branch (lane t reads lane t ^ 32's word), on a 32-lane host,
/// under the whole `emit`, translate the same with and without it. (The programs that hung
/// the card under `VICTORY_WAVE_WIDE=1` had such an earlier form; only a program that would
/// keep 64 lanes reaches the wave-wide form last of all: 4 of the 6 a 32-lane card refuses
/// in the title.)
#[test]
fn the_wave_wide_form_last_of_all_keeps_the_forms_taken_before() {
    let mut lds = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0 (this lane's byte)
        0x3A04_02FF,
        0x0000_0080, // v_xor_b32 v2, 128, v1 (lane t ^ 32's byte)
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0008, // s_cbranch_execz to the restore of EXEC
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD8D8_0000,
        0x0300_0002, // ds_read_b32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 3 << 8 | 1, // buffer_store_dword v3, v1, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    if let StageIo::Cs(cs) = &mut lds.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    for native in [program(0xBF88_0000), lds] {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let io = io::raise(&p, &native).expect("io");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
        let empty_bytes = crate::dxbc::build(&[]);
        let empty = crate::Container::parse(&empty_bytes).unwrap();
        let run = |last: bool| {
            let was = lower::TEST_HOST_WAVE.replace(Some(32));
            let was_last = super::TEST_WIDE_LAST.replace(Some(last));
            let r = super::emit(&empty, &native, &p, &st, &io, &res);
            super::TEST_WIDE_LAST.set(was_last);
            lower::TEST_HOST_WAVE.set(was);
            r.expect("translates")
        };
        let (before, after) = (run(false), run(true));
        assert_ne!(psv0_wave(&before), (64, 64), "an earlier form takes it");
        assert_eq!(before.dxil, after.dxil, "and the switch keeps that form");
    }
}

/// A lane loop that takes the number of its first lane (`s_ff1` of EXEC) in every round, as
/// the tile programs over the window glass do (`e4a9f58e3acfcf7b`, `c4183d3cbee28e8e`: "a
/// wave's first-lane number used as data", "a lane mask exchanged where not every lane
/// runs"): lane t runs t rounds, and round k adds the first lane still in the loop (lane k)
/// to the lane's sum, so the console's lane t ends with 1 + 2 + ... + t. Every lane stores
/// its sum at its own dword.
const FIRST_LANE_ROUNDS: [u32; 14] = [
    0x7E02_0280, // v_mov_b32 v1, 0 (the sum)
    0x7E04_0300, // v_mov_b32 v2, v0 (rounds left)
    0xBE8C_047E, // s_mov_b64 s[12:13], exec
    0x7D8A_0480, // loop: v_cmp_ne_u32 vcc, 0, v2
    0x87FE_6A7E, // s_and_b64 exec, exec, vcc
    0xBF88_0004, // s_cbranch_execz done
    0xBE87_147E, // s_ff1_i32_b64 s7, exec
    0x4A02_0207, // v_add_nc_u32 v1, s7, v1
    0x4E04_0481, // v_subrev_nc_u32 v2, 1, v2
    0xBF82_FFF9, // s_branch loop
    0xBEFE_040C, // done: s_mov_b64 exec, s[12:13]
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0x38 << 26 | 0x1C << 18 | 1 << 12,
    0x80 << 24 | 2 << 16 | 1 << 8 | 3, // buffer_store_dword v1, v3, s[8:11], 0 offen
];

/// `VICTORY_WAVE_EXACT` (on by default): a console wave64 program that takes its first lane's
/// number inside a lane loop ([`FIRST_LANE_ROUNDS`]) has no exact form without barriers (no
/// skip to leave out: the loop's exit is the only branch), and the approximate one runs each
/// host wave as a wave of its own, whose first lane in the upper half is lane 32 at the
/// earliest. It now takes the wave-wide form before that: the loop taken by the whole wave,
/// the mask exchanged between the halves. Under the whole `emit` on a 32-lane host, and on a
/// card of 32-lane waves, every lane ends with the console's sum; the approximate form, still
/// taken with the switch off, gives the upper half's lanes another.
#[test]
fn a_first_lane_number_in_a_lane_loop_takes_the_exact_wave_wide_form() {
    let native = native(&FIRST_LANE_ROUNDS);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let io = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();
    let run = |exact: bool| {
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        let was_exact = super::TEST_WAVE_EXACT.replace(Some(exact));
        let r = super::emit(&empty, &native, &p, &st, &io, &res);
        super::TEST_WAVE_EXACT.set(was_exact);
        lower::TEST_HOST_WAVE.set(was);
        r.expect("translates")
    };
    let (approximate, exact) = (run(false), run(true));
    assert_ne!(psv0_wave(&exact), (64, 64), "the exact form runs on 32-lane waves");
    assert_ne!(approximate.dxil, exact.dxil, "the two orders take different forms");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for shader in [&approximate, &exact] {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let console: Vec<u32> = (0..64).map(|t| t * (t + 1) / 2).collect();
    let mut sums = Vec::new();
    for shader in [&exact, &approximate] {
        let Ok((root, pso)) = host.compute_pipeline(shader) else {
            eprintln!("the hardware device does not take this program (waves of 32?); not run");
            return;
        };
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch the rounds");
        sums.push(out[0][..64].to_vec());
    }
    assert_eq!(sums[0], console, "the exact form: every lane the console's sum");
    assert_ne!(sums[1][32..], console[32..], "the approximate form: the upper half its own");
    eprintln!("ran on the hardware device: the exact form gave every lane the console's sum");
}

/// `with_group_base`: the second part of a dispatch over 65,535 groups in x reads its
/// group id x past the first part's. A 32-lane console program with the group id x in s5
/// (TGID_X_EN) stores it; translated with a base of 65,535 and run as one group, it
/// stores 65,535, and as it is, 0.
#[test]
fn a_group_base_is_added_to_the_group_id_x() {
    let mut native = native_of(&[], false, 32);
    let words = [
        0x7E02_0205, // v_mov_b32 v1, s5 (the group id x)
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
        S_ENDPGM,
    ];
    native.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 7; // TGID_X_EN: the group id x after the user SGPRs
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let plain = lowered(&native, &p, &HashSet::new()).expect("lower");
    let based = lower::with_group_base(65_535, || lowered(&native, &p, &HashSet::new())).expect("lower");
    assert_ne!(plain.dxil, based.dxil, "the base is in the translation");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut based.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = on_the_card(&based, [0, 0]) else { return };
    assert_eq!(got, 65_535, "group 0 of the second part is group 65,535");
    assert_eq!(on_the_card(&plain, [0, 0]), Some(0));
    eprintln!("ran on the hardware device: the second part's group ids start at the base");
}

/// Each lane's end is 1 or 2, alternating across a wave. VCCNZ keeps
/// the whole wave looping while one lane still has work. VCCZ keeps it
/// looping until the first lane matches. Both paths terminate in at most
/// two iterations.
#[test]
fn a_vcc_loop_tests_the_mask_in_its_original_direction() {
    let Ok(host) = Host::new(true) else {
        eprintln!("no SM 6.6 WARP device; not run");
        return;
    };
    for (compare, branch, expected) in [
        (0x7D84_0406, 0xBF86_FFFD, 1), // s6 == v2, branch back on VCCZ
        (0x7D82_0406, 0xBF87_FFFD, 2), // s6 < v2, branch back on VCCNZ
    ] {
        let words = [
            0x3604_0081, // v_and_b32 v2, 1, v0
            0x4A04_0481, // v_add_nc_u32 v2, 1, v2
            0xBE86_0380, // s_mov_b32 s6, 0
            0x8006_8106, // s_add_u32 s6, s6, 1
            compare,
            branch,
            0x7E02_0206, // v_mov_b32 v1, s6
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xE070_1000,
            0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
        ];
        let native = native_of(&words, false, 32);
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let shader = lowered(&native, &p, &HashSet::new()).unwrap();
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                    )],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(out[0], [expected; 32], "branch {branch:#x}");
    }
}

/// A 64-lane `v_add_f64` whole-wave sum lowered on a 32-lane host:
/// - for a group of 32 threads (`e.1 == true`), one host wave holds the whole group so no
///   cross-half exchange is needed;
/// - inside a per-lane branch under `with_local_waves_any` (`self.local == true`), each
///   host wave takes its own 32-lane sum instead of failing with
///   `"a wave reduction where not every lane runs"`.
#[test]
fn wave64_f64_sum_on_wave32_host_respects_small_groups_and_local_waves() {
    let vop1 = |op: u16, vdst: u8, src0: u16| {
        (0x3Fu32 << 25) | (u32::from(vdst) << 17) | (u32::from(op) << 9) | u32::from(src0)
    };
    let vop3 = |op: u16, vdst: u8, src0: u16, src1: u16, src2: u16| {
        [
            (0x35u32 << 26) | (u32::from(op) << 16) | u32::from(vdst),
            u32::from(src0) | (u32::from(src1) << 9) | (u32::from(src2) << 18),
        ]
    };
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), u32::from(vsrc) | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            u32::from(vsrc) | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };
    let vgpr = |n: u8| 256u16 + u16::from(n);
    let sgpr = |n: u8| u16::from(n);

    // Whole-wave f64 sum of per-lane (s0, s1) inside `if (v0 < s2)`.
    let mut words = vec![
        vop1(0x001, 6, sgpr(0)), // v_mov_b32 v6, s0
        vop1(0x001, 7, sgpr(1)), // v_mov_b32 v7, s1
        0x7D88_0002,             // v_cmp_gt_u32 vcc, s2, v0
        0xBE92_246A,             // s_and_saveexec_b64 s[18:19], vcc
    ];
    let skip_at = words.len();
    words.push(0xBF88_0000); // s_cbranch_execz (patched below)
    let body_start = words.len();
    words.push(0xBE8C_25C1); // s_or_saveexec_b64 s[12:13], -1
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12))); // v_cndmask_b32 v2, 0, v6, s[12:13]
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12))); // v_cndmask_b32 v3, 0, v7, s[12:13]
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.extend_from_slice(&vop3(0x164, 2, sgpr(14), vgpr(2), 0)); // v_add_f64 v[2:3], s[14:15], v[2:3]
    words.extend_from_slice(&vop3(0x360, 16, vgpr(2), 0xA0, 0)); // v_readlane_b32 s16, v2, 32
    words.extend_from_slice(&vop3(0x360, 17, vgpr(3), 0xA0, 0)); // v_readlane_b32 s17, v3, 32
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    words.push(vop1(0x00F, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&[0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16]); // buffer_store_dword v0, off, s[8:11], 0
    let body_words = (words.len() - body_start) as u32;
    words[skip_at] = 0xBF88_0000 | body_words;
    words.push(0xBEFE_0412); // s_mov_b64 exec, s[18:19]

    // 1. 64-thread group: fails without local waves, succeeds under `with_local_waves_any`.
    let n64 = native_of(&words, true, 64);
    let p64 = ir::build(&n64.code, n64.wave64, n64.vgprs).expect("ir::build");
    let err = lowered(&n64, &p64, &HashSet::new());
    assert!(
        matches!(&err, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "expected divergent reduction error without local waves, got {:?}",
        err.err(),
    );
    let shader64 = lower::with_local_waves_any(|| lowered(&n64, &p64, &HashSet::new())).expect("local waves f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader64.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (64-thread local)");
    }

    // 2. 32-thread group (wave64 on wave32 host, `e.1 == true`): no other half to exchange with.
    let n32 = native_of(&words, true, 32);
    let p32 = ir::build(&n32.code, n32.wave64, n32.vgprs).expect("ir::build");
    let shader32 = lowered(&n32, &p32, &HashSet::new()).expect("32-thread wave64 f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader32.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (32-thread)");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    let Ok((root, pso)) = host.compute_pipeline(&shader32) else { return };
    let b = 1.25f64.to_bits();
    let inputs = [b as u32, (b >> 32) as u32, 4u32, 0u32];
    let mut ud = vec![0u32; shader32.user_data_dwords().max(1) as usize];
    for u in &shader32.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => inputs[u.dword as usize],
            (RootType::Table, 1) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    assert_eq!(out[0][0], 5.0f32.to_bits(), "32-thread wave64 f64 sum of 4 lanes * 1.25");
}

/// One console wave of 64. `v_writelane_b32` writes its lane whatever EXEC is: inside
/// `if (t > s0) { ... }`, lane 0 (never inside) takes the scalar as soon as any thread is.
/// The compiler parks scalars this way in a VGPR that has other uses too (`unspill` leaves
/// those alone). Every thread stores what lane 0 holds after the join: 5 once the body
/// ran, 7 when the wave skipped it. As it is the program is refused (lane 0 is not in the
/// block); the forms without the skip have every lane there.
#[test]
fn a_lane_write_reaches_its_lane_outside_exec() {
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0x7D82_0000, // v_cmp_lt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz to "join"
        0xBE86_0385, // s_mov_b32 s6, 5
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_040C, // join: s_mov_b64 exec, s[12:13]
        0xD760_0007,
        0x0001_0109, // v_readlane_b32 s7, v9, 0
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        eprintln!("  {:#06x} {}", i.machine.offset, i.name());
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let forms = [
        ("as is", Some(p.clone())),
        ("without skips", ir::without_skips(&p, false)),
        ("without skips, scalars followed", ir::without_skips(&p, true)),
    ];
    let mut wrong = Vec::new();
    for (name, q) in &forms {
        let Some(q) = q else {
            eprintln!("{name}: the skip is kept");
            continue;
        };
        lower::take_needs();
        let shader = match lowered(&native, q, &HashSet::new()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{name}: {e}");
                continue;
            }
        };
        // (threads above s0 inside) -> lane 0 after the join
        for (s0, expected) in [(10u32, 5u32), (62, 5), (63, 7)] {
            let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
            eprintln!("{name}: threads above {s0} inside -> {got} (the console: {expected})");
            if got != expected {
                wrong.push(format!("{name}: s0 = {s0}: {got}, not {expected}"));
            }
        }
    }
    // The same with lane 0 inside (`t < s0`): the write lands too.
    let mut inside = words;
    inside[1] = 0x7D88_0000; // v_cmp_gt_u32 vcc, s0, v0
    let native = native_of(&inside, true, 64);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let q = ir::without_skips(&p, false).expect("the skip is left out");
    lower::take_needs();
    let shader = lowered(&native, &q, &HashSet::new()).expect("lower");
    for (s0, expected) in [(10u32, 5u32), (0, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        eprintln!("lane 0 inside, without skips: threads below {s0} inside -> {got} (the console: {expected})");
        assert_eq!(got, expected, "lane 0 inside, s0 = {s0}");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A 32-lane console wave whose waves are the host's, as the title's trace programs: they
/// keep a small array of scalars in the lanes of a VGPR, written by constant lane and read
/// by a computed one (`shadowmask:CS_TraceShadowRays_*`, `rtdirectionalshadows:*`).
/// `if (t > s0) { v_writelane_b32 v9, 5, 0 }`, then every thread stores lane s1 (= 0) of v9.
/// Lane 0 is never inside: per lane the program is refused, and with the skip taken by the
/// wave lane 0 is there and takes the scalar once some lane is inside.
#[test]
fn a_lane_array_is_written_and_read_with_the_skip_wave_wide() {
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0x7D82_0000, // v_cmp_lt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0003, // s_cbranch_execz to "join"
        0xBE86_0385, // s_mov_b32 s6, 5
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_030D, // join: s_mov_b32 exec_lo, s13
        0xD760_0007,
        0x0000_0309, // v_readlane_b32 s7, v9, s1
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == lower::LANE_ELSEWHERE),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (threads above s0 inside) -> lane 0 after the join
    for (s0, expected) in [(10u32, 5u32), (30, 5), (31, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        assert_eq!(got, expected, "threads above {s0} inside");
    }
    // The last resort, among the lanes that are there: it lowers, and the write is lost.
    let lax = lower::with_lanes_as_they_stand(|| lowered(&native, &p, &HashSet::new())).expect("lower");
    let Some(got) = on_the_card(&lax, [10, 0]) else { return };
    assert_eq!(got, 7, "as before: lane 0 is not inside");
    eprintln!("ran on the hardware device: lane 0 takes the scalar once some lane is inside");
}

/// A console wave32 pixel shader run on a GPU that launches pixel shaders in
/// 64-lane waves (AMD RDNA). Each 32-lane half must keep its own lane index
/// (`v_mbcnt_lo`), ballot (`s_ff1`), lane read (`v_readlane_b32`) and
/// first-lane read (`v_readfirstlane_b32`), so upper-half lanes 32..63 never
/// read from lanes 0..31.
#[test]
fn wave32_pixel_shader_runs_as_32_lane_halves_on_wave64_hosts() {
    use super::container::SigElement;
    use super::host::Draw;
    let words: [u32; 22] = [
        // v0 = SV_Position.x, v1 = SV_Position.y -> v4 = y * 16 + x (0..255)
        0x7E04_0F00, // v_cvt_u32_f32 v2, v0
        0x7E06_0F01, // v_cvt_u32_f32 v3, v1
        0x3406_0684, // v_lshlrev_b32 v3, 4, v3
        0x3808_0702, // v_or_b32 v4, v2, v3
        // v5 = v_mbcnt_lo_u32_b32(-1, 0): console lane within its 32-lane wave
        0xD765_0005,
        0x0001_00C1,
        // s0 = s_ff1_i32_b32(ballot(v5 == 7)), s1 = v_readlane_b32(v4, s0)
        0x7D84_0A87, // v_cmp_eq_u32 vcc_lo, 7, v5
        0xBE80_136A, // s_ff1_i32_b32 s0, vcc_lo
        0xD760_0001,
        0x0000_0104, // v_readlane_b32 s1, v4, s0
        // narrow EXEC to v5 >= 3, then s3 = v_readfirstlane_b32(v4)
        0x7D86_0A83, // v_cmp_le_u32 vcc_lo, 3, v5
        0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
        0x7E06_0504, // v_readfirstlane_b32 s3, v4
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        // Export (my_lane, picked_lane7, first_ge3, pix_id) as floats to mrt0
        0x7E0C_0D05, // v_cvt_f32_u32 v6, v5
        0x7E0E_0C01, // v_cvt_f32_u32 v7, s1
        0x7E10_0C03, // v_cvt_f32_u32 v8, s3
        0x7E12_0D04, // v_cvt_f32_u32 v9, v4
        0xF800_080F,
        0x0908_0706, // exp mrt0 v6, v7, v8, v9 done
        S_ENDPGM,
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: (1 << 8) | (1 << 9),
            db_shader_control: 0,
            patches: vec![],
        },
        code: words[..21].iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let pos = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0x3,
        half: false,
    };
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![pos],
        outputs: vec![target],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("lower");
    let ps = HostShader {
        stage: HwStage::Ps,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: vec![],
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: vec![],
        half_views: Vec::new(),
        group_split: 1,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = ps.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let vs = {
        // Full-screen triangle from SV_VertexID (v0 = 0, 1, 2):
        // (-1,-1,0,1), (3,-1,0,1), (-1,3,0,1).
        let vs_words: [u32; 13] = [
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0282, // v_lshlrev_b32 v1, 2, v1
            0x7E02_0D01, // v_cvt_f32_u32 v1, v1
            0x0602_02F3, // v_add_f32 v1, -1.0, v1
            0x3604_0082, // v_and_b32 v2, 2, v0
            0x3404_0481, // v_lshlrev_b32 v2, 1, v2
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x0604_04F3, // v_add_f32 v2, -1.0, v2
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7E08_02F2, // v_mov_b32 v4, 1.0
            0xF800_08CF,
            0x0403_0201, // exp pos0 v1, v2, v3, v4 done
            S_ENDPGM,
        ];
        let vn = NativeProgram {
            stage: HwStage::Vs,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Vs {
                inputs: vec![],
                outputs: vec![],
            },
            code: vs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let vp = ir::build(&vn.code, vn.wave64, vn.vgprs).expect("vs ir::build");
        let vst = structure::structure(&vp).expect("vs structure");
        let vres = resource::track(&vp, &vn.layout, vn.stage, Some(&vst)).expect("vs track");
        let vio = io::raise(&vp, &vn).expect("vs io::raise");
        let vpos = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let vl = lower::lower(&lower::Parts {
            native: &vn,
            p: &vp,
            st: &vst,
            io: &vio,
            res: &vres,
            inputs: vec![],
            outputs: vec![vpos],
            prims: vec![],
            mesh_stride: 0,
        })
        .expect("vs lower");
        HostShader {
            stage: HwStage::Vs,
            dxil: vl.dxil,
            user_data: vl.user_data,
            samplers: vl.samplers,
            static_samplers: vec![],
            root_cbvs: vl.root_cbvs,
            root_srvs: vl.root_srvs,
            root_uavs: vl.root_uavs,
            root_indexed_views: vec![],
            half_views: Vec::new(),
            group_split: 1,
        }
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = host.pipeline(&vs, &ps).expect("graphics pipeline");
    let pixels = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: vec![],
                textures: &[],
                samplers: &[],
                vertices: &[],
                stride: 0,
                size: (16, 16),
                mesh_groups: 0,
            },
        )
        .expect("draw");
    assert_eq!(pixels.len(), 256);
    let mut lane7_seen = HashSet::new();
    let mut lane3_seen = HashSet::new();
    for (idx, px) in pixels.iter().enumerate() {
        let my_lane = px[0] as u32;
        let picked7 = px[1] as u32;
        let first_ge3 = px[2] as u32;
        let pix_id = px[3] as u32;
        assert_eq!(pix_id, idx as u32, "pixel {idx} id");
        assert!(
            my_lane < 32,
            "pixel {idx}: v_mbcnt_lo gave lane {my_lane} >= 32 (upper half of a 64-lane host wave?)"
        );
        if my_lane == 7 {
            assert_eq!(
                picked7, pix_id,
                "pixel {idx} (lane 7): v_readlane_b32 read {picked7} instead of own half's {pix_id}"
            );
            lane7_seen.insert(picked7);
        }
        if my_lane == 3 {
            assert_eq!(
                first_ge3, pix_id,
                "pixel {idx} (lane 3): v_readfirstlane_b32 read {first_ge3} instead of own half's {pix_id}"
            );
            lane3_seen.insert(first_ge3);
        }
    }
    assert!(
        lane7_seen.len() >= 8,
        "expected >= 8 distinct 32-lane half-waves for 256 pixels, saw {}",
        lane7_seen.len()
    );
    assert!(
        lane3_seen.len() >= 8,
        "expected >= 8 distinct 32-lane half-waves for v_readfirstlane, saw {}",
        lane3_seen.len()
    );
    eprintln!(
        "ran on the hardware device: {} distinct 32-lane halves across 256 pixels",
        lane7_seen.len()
    );
}

/// A spilled 64-thread wave64 compute program dispatched with `DispatchX`
/// `USE_THREAD_DIMENSIONS` (`USER_DISPATCH_THREADS`): reads `s_bcnt1_i32_b64(exec)`
/// on entry, computes a whole-wave OR of `1 << (lane >> 1)`, and stores
/// `(popcnt, wave_or)` unconditionally at `uav[lane * 2..+2]`.
///
/// Verified for both host wave sizes:
/// - `host_max_wave = 64`: entry early-exit masks hardware `EXEC` before `B0`
///   so `s_bcnt1_i32_b64(exec)` and the wave OR see only lanes `< limit`, and
///   lanes `>= limit` never store.
/// - `host_max_wave = 32`: predicated mode keeps both 32-lane host waves alive
///   for `exchange_halves` barriers (no hang even at `limit = 1` where the
///   upper half has zero active lanes) while masking ballots, reductions, and
///   stores identically.
#[test]
fn spilled_compute_masks_partial_dispatch_threads_on_wave64_and_wave32() {
    let mut words = vec![
        // Read descriptor-table offset (user_data 16) from spilled table s[0:1] + 0.
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0x3D << 26 | 4 << 6 | 1,
        125 << 25, // s_load_dword s4, s[2:3], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (lane * 8 bytes)
        0x7E04_0206, // v_mov_b32 v2, s6 (popcnt(exec))
        0x7E06_0207, // v_mov_b32 v3, s7 (wave OR)
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 1, // buffer_store_dwordx2 v[2:3], v1, s[8:11], 0 offen
        S_ENDPGM,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1, // s0..s1 spilled table pointer
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).ok();
    for host_wave in [64u32, 32] {
        let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        });
        lower::TEST_HOST_WAVE.set(was);
        let lowered = lowered.unwrap_or_else(|e| panic!("lower (host_wave={host_wave}): {e}"));
        assert!(
            lowered
                .user_data
                .iter()
                .any(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS) && u.dword == 0),
            "host_wave={host_wave}: missing USER_DISPATCH_THREADS(0)"
        );
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: res.root_indexed_views.clone(),
            half_views: Vec::new(),
            group_split: 1,
        };
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("DXIL validation failed (host_wave={host_wave}): {e}"));
        }
        let Some(host) = host.as_ref() else { continue };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("device skipped host_wave={host_wave} pipeline");
            continue;
        };
        for (limit, active, expected_or) in [
            (45u32, 45usize, 0x007F_FFFFu32),
            (1, 1, 1),
            (0, 64, u32::MAX),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Other(super::USER_DISPATCH_THREADS) && u.dword == 0 {
                    ud[u.index as usize] = limit;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(128)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed (host_wave={host_wave}, limit={limit}): {e}"));
            for lane in 0..64 {
                let (got_cnt, got_or) = (out[0][lane * 2], out[0][lane * 2 + 1]);
                if lane < active {
                    assert_eq!(
                        (got_cnt, got_or),
                        (active as u32, expected_or),
                        "host_wave={host_wave} limit={limit} lane={lane}"
                    );
                } else {
                    assert_eq!(
                        (got_cnt, got_or),
                        (0, 0),
                        "host_wave={host_wave} limit={limit}: inactive lane {lane} stored!"
                    );
                }
            }
        }
        eprintln!("ran on the hardware device (host_wave={host_wave}): partial dispatch limits 45, 1, 0 exact");
    }
}

/// `VICTORY_LDS_FIT` (on by default): a console wave64 program whose LDS takes the whole
/// 32 KiB, in a group of one console wave on a host of 32-lane waves, that reads lane numbers
/// (`v_mbcnt`, so the halves would exchange) and its LDS. Each host wave a wave of its own,
/// it reserves no exchange pair and its counter of console threads borrows the program's
/// first dword, so the group keeps within D3D12's 32 KiB (`6b82bc7d22016cb5` needed 32,900
/// bytes, which the validator refuses); on a 32-lane card every thread still stores its own
/// console thread index (which the counter hands out) through its LDS dword.
#[test]
fn a_program_with_all_32_kib_of_lds_keeps_within_it_on_32_lane_waves() {
    let mut native = native(&[
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1 (the console lane)
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0 (this thread's byte)
        0xD834_0000,
        0x0000_0002, // ds_write_b32 v2, v0 (this thread's console index)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD8D8_0000,
        0x0300_0002, // ds_read_b32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 3 << 8 | 2, // buffer_store_dword v3, v2, s[8:11], 0 offen
    ]);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 64 << 15; // 64 x 512 bytes: all 32 KiB of LDS
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lower::with_local_waves_any(|| lowered(&native, &p, &HashSet::new()))
        .expect("lower with each host wave a wave of its own");
    let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) else {
        eprintln!("no validator; the 32 KiB check needs it");
        return;
    };
    let mut copy = shader.dxil.clone();
    v.validate(&mut copy).expect("within 32 KiB of thread-group shared memory");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let lanes: Vec<u32> = (0..64).collect();
    assert_eq!(&out[0][..64], &lanes[..], "each console thread through its own LDS dword");
    eprintln!("ran on the hardware device: 32 KiB of LDS and the borrowed counter");
}

/// [`FIRST_LANE_ROUNDS`] for a thread group of two console waves (128 threads), with the
/// group id z (TGID_Z_EN, in s5) in the address and in the value: lane L of each console wave
/// (thread t, L = t & 63) runs L rounds, each adding the number of the first lane of its
/// console wave still in the loop, and stores `z << 16 | 1 + 2 + ... + L` at dword
/// `z * 128 + t`. `rows`: the group as 8 x 16 threads (the thread id in v0 and v1, as the
/// title's water mask writers have it) instead of 128 x 1.
fn two_wave_rounds(rows: bool) -> NativeProgram {
    let mut native = native_of(&[], true, 128);
    let mut words = vec![
        0x7E10_0205, // v_mov_b32 v8, s5 (the group id z)
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    if rows {
        words.extend_from_slice(&[
            0x3416_0283, // v_lshlrev_b32 v11, 3, v1 (y * 8)
            0x3800_010B, // v_or_b32 v0, v11, v0 (the thread's index in the group)
        ]);
    }
    words.extend_from_slice(&[
        0x7E02_0280, // v_mov_b32 v1, 0 (the sum)
        0x3604_00BF, // v_and_b32 v2, 63, v0 (rounds left: the lane in its console wave)
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7D8A_0480, // loop: v_cmp_ne_u32 vcc, 0, v2
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0004, // s_cbranch_execz done
        0xBE87_147E, // s_ff1_i32_b64 s7, exec
        0x4A02_0207, // v_add_nc_u32 v1, s7, v1
        0x4E04_0481, // v_subrev_nc_u32 v2, 1, v2
        0xBF82_FFF9, // s_branch loop
        0xBEFE_040C, // done: s_mov_b64 exec, s[12:13]
        0x3412_1087, // v_lshlrev_b32 v9, 7, v8 (z * 128)
        0x3812_0109, // v_or_b32 v9, v9, v0
        0x3406_1282, // v_lshlrev_b32 v3, 2, v9 (the thread's byte)
        0x3414_1090, // v_lshlrev_b32 v10, 16, v8
        0x3802_030A, // v_or_b32 v1, v10, v1 (z << 16 | sum)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 3, // buffer_store_dword v1, v3, s[8:11], 0 offen
        S_ENDPGM,
    ]);
    native.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 9; // TGID_Z_EN: the group id z after the user SGPRs
        if rows {
            cs.threads = [8, 16, 1];
            cs.rsrc2 |= 1 << 11; // two thread id components, v0 and v1
        }
    }
    native
}

/// `VICTORY_GROUP_SPLIT` (off by default; Corgansm/victory-corgi's group split): a console
/// wave64 program whose thread group holds two console waves that share nothing (no LDS, no
/// barrier) and that needs the wave-wide form ([`two_wave_rounds`]: a first-lane number in a
/// lane loop). Without the switch the wave-wide form is refused it ("thread groups of
/// several console waves") and it takes an approximate form or none, translated as before;
/// with it each console wave runs as a host group of its own (64 threads, two host waves of
/// 32 that meet inside the loop), `group_split` says 2, and on a card of 32-lane waves every
/// thread of every console group gets the console's sum, at its console thread and group
/// ids, when the dispatch's z is doubled.
#[test]
fn a_group_of_two_console_waves_is_split_into_a_host_group_per_wave() {
    for rows in [false, true] {
        let native = two_wave_rounds(rows);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let io = io::raise(&p, &native).expect("io");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
        let empty_bytes = crate::dxbc::build(&[]);
        let empty = crate::Container::parse(&empty_bytes).unwrap();
        let run = |split: Option<bool>| {
            let was = lower::TEST_HOST_WAVE.replace(Some(32));
            let was_split = super::TEST_GROUP_SPLIT.replace(split);
            let r = super::emit(&empty, &native, &p, &st, &io, &res);
            super::TEST_GROUP_SPLIT.set(was_split);
            lower::TEST_HOST_WAVE.set(was);
            r
        };
        // off, and unset (the default): the same translation, or the same refusal
        let (off, unset) = (run(Some(false)), run(None));
        assert_eq!(off.as_ref().ok().map(|s| &s.dxil), unset.as_ref().ok().map(|s| &s.dxil), "off is the default");
        assert_eq!(off.is_ok(), unset.is_ok());
        if let Ok(off) = &off {
            assert_eq!(off.group_split, 1, "not split without the switch");
        }
        let on = run(Some(true)).expect("split, it takes the wave-wide form");
        assert_eq!(on.group_split, 2, "one host group per console wave");
        assert_ne!(psv0_wave(&on), (64, 64), "on 32-lane waves");
        assert_ne!(off.as_ref().ok().map(|s| &s.dxil), Some(&on.dxil));
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = on.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        const GROUPS: u32 = 3;
        let console: Vec<u32> = (0..GROUPS * 128)
            .map(|i| {
                let (z, lane) = (i / 128, i % 64);
                (z << 16) | (lane * (lane + 1) / 2)
            })
            .collect();
        let on_card = |shader: &HostShader| {
            let (root, pso) = host.compute_pipeline(shader).ok()?;
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]))],
                        views: &[View::RawUav(GROUPS * 128)],
                        samplers: &[],
                        // (as the host does: z counts the console waves too)
                        groups: [1, 1, GROUPS * shader.group_split],
                    },
                )
                .expect("dispatch the rounds");
            Some(out[0][..(GROUPS * 128) as usize].to_vec())
        };
        let Some(split) = on_card(&on) else {
            eprintln!("the hardware device does not take this program (waves of 32?); not run");
            return;
        };
        assert_eq!(split, console, "rows {rows}: every thread of every console group the console's sum");
        if let Some(whole) = off.as_ref().ok().and_then(on_card) {
            assert_ne!(whole, console, "rows {rows}: the form taken without the split is not the console's");
        }
        eprintln!("ran on the hardware device (rows {rows}): split groups gave every thread the console's sum");
    }
}
