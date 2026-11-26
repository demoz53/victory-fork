//! Console wave64 compute programs in the paired form (`VICTORY_WAVE_PAIRS`: one host wave for
//! each console wave, two lanes a thread, `super::pairs` and `lower::paired`), whose LDS, lane
//! reads and masks cross the halves without barriers, checked against what the console's 64
//! lanes compute. Lowered, validated by `dxil.dll` when present, and run on the host GPU when
//! it takes waves of 32 (WARP's are 4 lanes: the form does not run there).
//!
//! Ported from fxceless/victory-fork PR #854 (commit 26600c6, by PiloteCode): its on-card
//! tests of the paired form in `wave64_tests.rs`, with the helpers they need. Changed for this
//! tree: the form is opt-in here, so the tests choose it for their thread (`TEST_PAIRS`), and
//! the test of `emit`'s choice checks the switch's three values instead of the approximate
//! form's picture. Added: a lane write under a guard, with lanes of either half in EXEC.

use super::host::{Dispatch, Host, Root, View};
use super::pairs::{Mode, TEST_PAIRS};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

// Operands as the hardware numbers them: SGPR n is n, VCC 106 (its high dword 107), EXEC 126,
// an inline integer n 128 + n (-1: 193), a literal (the next dword) 255, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

const fn ds(op: u32, vdst: u32, addr: u32, data: u32, offset: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18 | offset, vdst << 24 | data << 8 | addr]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// A compute program of `threads` threads in a row (the thread index in v0) with 4 root
/// constants in s0..s3, a raw UAV's descriptor in s[8:11] and `lds` bytes of LDS (512-byte
/// granules), then `body`.
fn native_with(body: &[u32], wave64: bool, threads: u32, lds: u32) -> NativeProgram {
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
            rsrc2: 5 << 1 | lds.div_ceil(512) << 15, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV (`wave64_tests::program`).
fn or_in_a_branch(skip: u32) -> NativeProgram {
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
    native_with(&words, true, 64, 0)
}

/// What a test needs of a program to emit it: its IR, the lanes' own structure, I/O and
/// resources.
struct Stages {
    p: ir::Program,
    st: structure::Structured,
    io: io::Io,
    res: resource::Resources,
}

fn stages(native: &NativeProgram) -> Stages {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    Stages { p, st, io, res }
}

/// [`super::emit`]'s choice for `native` on a host of 32-lane waves with `VICTORY_WAVE_PAIRS`
/// as `mode`.
fn emitted(native: &NativeProgram, s: &Stages, mode: Mode) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let wave = lower::TEST_HOST_WAVE.replace(Some(32));
    let pairs = TEST_PAIRS.replace(Some(mode));
    let r = super::emit(&c, native, &s.p, &s.st, &s.io, &s.res);
    TEST_PAIRS.set(pairs);
    lower::TEST_HOST_WAVE.set(wave);
    r
}

/// `native` in the paired form on a host of 32-lane waves, validated by `dxil.dll` when
/// present.
fn paired(native: &NativeProgram) -> HostShader {
    let s = stages(native);
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let shader = super::pairs::paired(&c, native, &s.p, &s.io, &s.res);
    lower::TEST_HOST_WAVE.set(was);
    let (shader, wave) = shader.expect("the paired form");
    assert_eq!(wave, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// `shader` on the hardware device: one group, root constants `s0..s3`, a raw UAV of `dwords`
/// zeroed dwords, which come back; None (and a line) without a device that takes it.
fn run_paired(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let mut out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.swap_remove(0))
}

/// Panics at the first dword of `got` that is not the console's (`want`), naming the thread
/// and which of its four dwords.
fn same_as_console(got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    if let Some(n) = (0..want.len()).find(|&n| got[n] != want[n]) {
        panic!("thread {}, dword {}: {:#x}, the console's {:#x}", n / 4, n % 4, got[n], want[n]);
    }
}

/// One console wave of 64 threads with 256 bytes of LDS. The even lanes write their dword in
/// one arm of a per-lane branch; the odd lanes theirs in the other arm, then, in that arm and
/// with no barrier, read their mirror lane's (63 - n, an even lane's); the even lanes read
/// theirs (an odd lane's) after the join. Then lanes 5 and 40 are read (`v_readlane_b32`), the
/// lanes holding more than 250 are counted (`s_bcnt1_i32_b64` of the compare's mask) and
/// counted below each lane (`v_mbcnt_lo/hi`): each of these takes lanes of both halves.
#[test]
fn paired_lanes_share_lds_and_lanes_across_halves_without_barriers() {
    let mut w = vec![vop2(0x1A, 1, k(2), 0)]; // v_lshlrev_b32 v1, 2, v0: this lane's dword
    w.extend(vop3(0x169, 2, v(0), k(7))); // v_mul_lo_u32 v2, v0, 7
    w.extend([
        vop2(0x26, 5, k(63), 0), // v_sub_nc_u32 v5, 63, v0: the mirror lane
        vop2(0x1A, 5, k(2), 5),  // v_lshlrev_b32 v5, 2, v5: its dword
        vop2(0x1B, 3, k(1), 0),  // v_and_b32 v3, 1, v0
        vopc(0xC2, k(0), 3),     // v_cmp_eq_u32 vcc, 0, v3: the even lanes
        sop1(0x24, 12, VCC),     // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 4),           // s_cbranch_execz past the even lanes' arm
        vop2(0x25, 4, LIT, 2),   // v_add_nc_u32 v4, 100, v2
        100,
    ]);
    w.extend(ds(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend([
        sop2(0x15, EXEC, 12, EXEC), // s_andn2_b64 exec, s[12:13], exec: the odd lanes
        sopp(0x08, 6),              // s_cbranch_execz past their arm
        vop2(0x25, 4, LIT, 2),      // v_add_nc_u32 v4, 200, v2
        200,
    ]);
    w.extend(ds(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend(ds(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an even lane's, from the other arm
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        sop1(0x24, 12, VCC),  // s_and_saveexec_b64 s[12:13], vcc: the even lanes again
    ]);
    w.extend(ds(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an odd lane's
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w.extend(vop3(0x360, 14, v(6), k(5))); // v_readlane_b32 s14, v6, 5
    w.extend(vop3(0x360, 15, v(6), k(40))); // v_readlane_b32 s15, v6, 40
    w.extend([
        sop2(0x00, 14, 14, 15), // s_add_u32 s14, s14, s15
        vopc(0xC1, LIT, 6),     // v_cmp_lt_u32 vcc, 250, v6
        250,
        sop1(0x10, 16, VCC), // s_bcnt1_i32_b64 s16, vcc
    ]);
    w.extend(vop3(0x365, 7, VCC, k(0))); // v_mbcnt_lo_u32_b32 v7, vcc_lo, 0
    w.extend(vop3(0x366, 7, VCC + 1, v(7))); // v_mbcnt_hi_u32_b32 v7, vcc_hi, v7
    w.extend([
        vop1(0x01, 9, 14),  // v_mov_b32 v9, s14
        vop1(0x01, 10, 16), // v_mov_b32 v10, s16
    ]);
    w.extend(stores([6, 9, 10, 7]));
    let shader = paired(&native_with(&w, true, 64, 256));
    // the console: lane n's dword is 7n + 100 (even n) or 7n + 200, and lane n reads 63 - n's
    let dword = |n: u32| 7 * n + if n.is_multiple_of(2) { 100 } else { 200 };
    let read: Vec<u32> = (0..64).map(|n| dword(63 - n)).collect();
    let big = (0..64).filter(|&n| read[n] > 250).fold(0u64, |m, n| m | 1 << n);
    assert!(big as u32 != 0 && big >> 32 != 0, "lanes of both halves counted");
    let want: Vec<u32> = (0..64)
        .flat_map(|n| [read[n], read[5] + read[40], big.count_ones(), (big & ((1 << n) - 1)).count_ones()])
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [0; 4], 256) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: LDS, lane reads and masks across the halves as on the console");
}

/// Two console waves of 64 in a group of 128 (two host waves of 32, 512 bytes of LDS). In
/// each, lane 0 zeroes the wave's dword of LDS; every lane loops (lane & 3) + 1 + wave times,
/// adding its lane number to its sum and 1 to the wave's dword each round (`ds_add_u32`; the
/// loop's branch is the wave's: it goes round while some lane is left); after the loop each
/// lane reads the wave's total, with no barrier; lanes 40-63 read their first lane's sum
/// (`v_readfirstlane_b32`: lane 40, an upper lane), then lanes 20-63 (lane 20, a lower one,
/// though upper lanes have one too); past the barrier, each lane reads the other wave's total.
#[test]
fn paired_lanes_loop_apart_and_meet_the_other_wave_at_the_barrier() {
    let mut w = vec![
        vop2(0x16, 1, k(6), 0),  // v_lshrrev_b32 v1, 6, v0: the console wave
        vop2(0x1B, 2, k(63), 0), // v_and_b32 v2, 63, v0: the lane in it
        vop2(0x1A, 11, k(2), 1), // v_lshlrev_b32 v11, 2, v1: the wave's dword
        vop1(0x01, 12, k(0)),    // v_mov_b32 v12, 0
        sop1(0x04, 22, EXEC),    // s_mov_b64 s[22:23], exec
        vopc(0xD2, k(0), 2),     // v_cmpx_eq_u32 exec, 0, v2: lane 0
    ];
    w.extend(ds(0x0D, 0, 11, 12, 0)); // ds_write_b32 v11, v12
    w.extend([
        sop1(0x04, EXEC, 22),   // s_mov_b64 exec, s[22:23]
        vop2(0x1B, 3, k(3), 2), // v_and_b32 v3, 3, v2
        vop2(0x25, 3, k(1), 3), // v_add_nc_u32 v3, 1, v3
        vop2(0x25, 3, v(1), 3), // v_add_nc_u32 v3, v1, v3: the rounds
        vop1(0x01, 4, k(0)),    // v_mov_b32 v4, 0
        vop1(0x01, 13, k(1)),   // v_mov_b32 v13, 1
        sop1(0x04, 20, EXEC),   // s_mov_b64 s[20:21], exec
        vop2(0x25, 4, v(2), 4), // loop: v_add_nc_u32 v4, v2, v4
    ]);
    w.extend(ds(0x00, 0, 11, 13, 0)); // ds_add_u32 v11, v13
    w.extend([
        vop2(0x25, 3, 193, 3),       // v_add_nc_u32 v3, -1, v3
        vopc(0xC5, k(0), 3),         // v_cmp_ne_u32 vcc, 0, v3
        sop2(0x0F, EXEC, EXEC, VCC), // s_and_b64 exec, exec, vcc
        sopp(0x09, -7),              // s_cbranch_execnz loop
        sop1(0x04, EXEC, 20),        // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(ds(0x36, 5, 11, 0, 0)); // ds_read_b32 v5, v11: the wave's total
    w.extend([
        sop1(0x04, 24, EXEC), // s_mov_b64 s[24:25], exec
        vopc(0xD1, k(39), 2), // v_cmpx_lt_u32 exec, 39, v2: lanes 40-63
        vop1(0x02, 6, v(4)),  // v_readfirstlane_b32 s6, v4
        sop1(0x04, EXEC, 24), // s_mov_b64 exec, s[24:25]
        vopc(0xD1, k(19), 2), // v_cmpx_lt_u32 exec, 19, v2: lanes 20-63
        vop1(0x02, 7, v(4)),  // v_readfirstlane_b32 s7, v4
        sop1(0x04, EXEC, 24), // s_mov_b64 exec, s[24:25]
        vop1(0x01, 6, 6),     // v_mov_b32 v6, s6
        vop1(0x01, 9, 7),     // v_mov_b32 v9, s7
        0xD400_0000 | 0x346 << 16 | 6, // v_lshl_add_u32 v6, v9, 16, v6
        v(9) | k(16) << 9 | v(6) << 18,
        sopp(0x0A, 0),            // s_barrier
        vop2(0x1D, 14, k(4), 11), // v_xor_b32 v14, 4, v11: the other wave's dword
    ]);
    w.extend(ds(0x36, 7, 14, 0, 0)); // ds_read_b32 v7, v14
    w.extend(stores([4, 5, 6, 7]));
    let shader = paired(&native_with(&w, true, 128, 512));
    let rounds = |wave: u32, lane: u32| (lane & 3) + 1 + wave;
    let total = |wave: u32| (0..64).map(|lane| rounds(wave, lane)).sum::<u32>();
    let first = |wave: u32, lane: u32| lane * rounds(wave, lane);
    let want: Vec<u32> = (0..128)
        .flat_map(|t| {
            let (wave, lane) = (t / 64, t % 64);
            [first(wave, lane), total(wave), first(wave, 20) << 16 | first(wave, 40), total(1 - wave)]
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [0; 4], 512) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: two console waves, their loops and LDS as on the console");
}

/// One console wave: `ds_bpermute_b32` from the mirror lane; a DPP row shift whose row mask
/// leaves rows 0 and 3 alone (rows 2 and 3 are lanes 32-63: rows are counted in console lanes); a
/// `ds_swizzle_b32` swapping the two 16-lane rows of each 32 lanes; `v_writelane_b32` of s0
/// into lane 45.
#[test]
fn paired_lanes_permute_shift_and_write_lanes_of_both_halves() {
    let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
    w.extend([
        vop2(0x25, 1, k(1), 1),  // v_add_nc_u32 v1, 1, v1
        vop2(0x26, 2, k(63), 0), // v_sub_nc_u32 v2, 63, v0
        vop2(0x1A, 2, k(2), 2),  // v_lshlrev_b32 v2, 2, v2: the mirror lane's address
    ]);
    w.extend(ds(0xB3, 3, 2, 1, 0)); // ds_bpermute_b32 v3, v2, v1
    w.extend([
        vop1(0x01, 4, LIT), // v_mov_b32 v4, 1000
        1000,
        vop1(0x01, 4, 0xFA), // v_mov_b32 v4, v1 row_shr:1 row_mask:0x6 bank_mask:0xf
        0x6 << 28 | 0xF << 24 | 0x111 << 8 | 1,
    ]);
    w.extend(ds(0x35, 5, 1, 0, 0x401F)); // ds_swizzle_b32 v5, v1 offset:swizzle(BITMASK_PERM, xor 16)
    w.push(vop1(0x01, 6, v(0))); // v_mov_b32 v6, v0
    w.extend(vop3(0x361, 6, 0, k(45))); // v_writelane_b32 v6, s0, 45
    w.extend(stores([3, 4, 5, 6]));
    let shader = paired(&native_with(&w, true, 64, 0));
    let s0 = 0xABCD_1234;
    let x = |n: u32| 3 * n + 1;
    let want: Vec<u32> = (0..64)
        .flat_map(|n| {
            let shifted = if matches!(n >> 4, 1 | 2) && n & 15 != 0 { x(n - 1) } else { 1000 };
            let swizzled = x(n & 32 | (n & 31) ^ 16);
            [x(63 - n), shifted, swizzled, if n == 45 { s0 } else { n }]
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [s0, 0, 0, 0], 256) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: permutes, row masks, swizzles and lane writes as on the console");
}

/// `v_writelane_b32` under a guard (EXEC narrowed by `v_cmpx`, no branch): the console writes
/// the lane whatever its own EXEC bit, once some lane of the wave is in EXEC. s0 picks the
/// lanes in EXEC (those at or above it): lane 10 (a lower lane, outside EXEC for s0 > 10) and
/// lane 50 (an upper one) take s1 while any lane of either half is left, and no lane does when
/// none is (s0 = 64).
#[test]
fn paired_lanes_write_a_lane_under_a_guard_while_some_lane_of_either_half_runs() {
    let mut w = vec![
        vop1(0x01, 6, v(0)),  // v_mov_b32 v6, v0
        vop1(0x01, 7, v(0)),  // v_mov_b32 v7, v0
        sop1(0x04, 12, EXEC), // s_mov_b64 s[12:13], exec
        vopc(0xD3, 0, 0),     // v_cmpx_le_u32 exec, s0, v0: the lanes at or above s0
    ];
    w.extend(vop3(0x361, 6, 1, k(10))); // v_writelane_b32 v6, s1, 10
    w.extend(vop3(0x361, 7, 1, k(50))); // v_writelane_b32 v7, s1, 50
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w.extend(stores([6, 7, 0, 0]));
    let native = native_with(&w, true, 64, 0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let names: Vec<&str> = p.blocks.iter().flat_map(|b| &b.insts).map(|i| i.name()).collect();
    assert!(names.contains(&"v_cmpx_le_u32"), "{names:?}");
    assert_eq!(names.iter().filter(|n| **n == "v_writelane_b32").count(), 2);
    let shader = paired(&native);
    let s1 = 0x5EED_0001;
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [0u32, 5, 11, 33, 51, 63, 64] {
        let Some(got) = run_paired(&shader, [s0, s1, 0, 0], 256) else { return };
        let some = s0 < 64;
        let want: Vec<u32> = (0..64u32)
            .flat_map(|n| {
                let wrote = |lane: u32| if n == lane && some { s1 } else { n };
                [wrote(10), wrote(50), n, n]
            })
            .collect();
        same_as_console(&got, &want);
    }
    eprintln!("ran on the hardware device: guarded lane writes reach lanes of both halves");
}

/// A wave OR inside a per-lane branch ([`or_in_a_branch`]), in the paired form: the OR of both
/// halves' lanes, whichever take the branch; nothing stored when none does.
#[test]
fn paired_lanes_or_a_wave_inside_a_lane_branch() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        let shader = paired(&or_in_a_branch(skip));
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let cases = [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1), (0, 0)];
        for (below, expected) in cases {
            let Some(got) = run_paired(&shader, [below, 0, 0, 0], 1) else { return };
            assert_eq!(got[0], expected, "{name}: {below} threads in the branch");
        }
    }
    eprintln!("ran on the hardware device: the OR of both halves in every case");
}

/// One console wave of 64 threads, thread t holding the bit `1 << (t >> 1)`: the threads below
/// s0 OR theirs over the wave inside `if (t < s0) { ... }` ([`super::wave::WAVE_OR_V7_TO_S7`]
/// into s7, which holds 0x77 before), and every thread stores s7 after the branch: the wave's
/// OR, or 0x77 when no thread took the branch and the wave skipped it. A wave-wide value used
/// past a per-lane branch, as the title's tile classification and light clustering programs
/// use theirs: it does not run as it is in two host waves (they would meet inside the
/// branch). `VICTORY_WAVE_PAIRS` unset: [`super::emit`] gives what it gave before, never the
/// paired form; `=first`: the paired form; `=1`: the paired form unless an exact form without
/// added barriers takes the program first.
#[test]
fn a_wave_value_used_past_a_lane_branch_takes_the_paired_form_when_asked() {
    let mut w = vec![
        sop1(0x03, 7, LIT), // s_mov_b32 s7, 0x77
        0x77,
        vop1(0x01, 5, k(1)),    // v_mov_b32 v5, 1
        vop2(0x16, 6, k(1), 0), // v_lshrrev_b32 v6, 1, v0
        vop2(0x1A, 7, v(6), 5), // v_lshlrev_b32 v7, v6, v5
        vopc(0xC4, 0, 0),       // v_cmp_gt_u32 vcc, s0, v0
        sop1(0x24, 12, VCC),    // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 20),         // s_cbranch_execz past the OR
    ];
    w.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        vop1(0x01, 8, 7),     // v_mov_b32 v8, s7
    ]);
    w.extend(stores([8, 7, 6, 0]));
    let native = native_with(&w, true, 64, 0);
    let pairs = paired(&native);
    let s = stages(&native);
    let off = emitted(&native, &s, Mode::Off).expect("a form without the switch");
    let on = emitted(&native, &s, Mode::BeforeWide).expect("a form with the switch");
    let first = emitted(&native, &s, Mode::First).expect("a form with the switch at first");
    assert!(first.dxil == pairs.dxil, "VICTORY_WAVE_PAIRS=first: emit's choice is not the paired form");
    assert!(off.dxil != pairs.dxil, "without the switch, emit gives the paired form anyway");
    assert!(on.dxil == pairs.dxil || on.dxil == off.dxil, "with the switch: the paired form, or an earlier exact one");
    // the console: every thread stores the OR of the threads below s0, or 0x77 with none
    let want = |s0: u32| -> Vec<u32> {
        let or = (0..64u32).filter(|&t| t < s0).fold(0u32, |m, t| m | 1 << (t >> 1));
        let s7 = if s0 == 0 { 0x77 } else { or };
        (0..64u32).flat_map(|t| [s7, 1 << (t >> 1), t >> 1, t]).collect()
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [64u32, 40, 33, 32, 10, 1, 0] {
        let Some(got) = run_paired(&pairs, [s0, 0, 0, 0], 256) else { return };
        same_as_console(&got, &want(s0));
    }
    eprintln!(
        "ran on the hardware device: the wave's OR past the branch in every lane; VICTORY_WAVE_PAIRS=1 gives {}",
        if on.dxil == pairs.dxil { "the paired form" } else { "an earlier exact form" }
    );
}

/// 64 threads in one console wave: thread t holds the 16-bit float with bits 0x3C00 + 37 t
/// (1.0 and up, with odd mantissas, so that additions round); all of them store the wave's sum
/// (`WaveActiveSum` of halves) to dword 0. In the paired form: each half's tree over its 32
/// lanes, then the two halves' sums, the console's last addition.
#[test]
fn paired_lanes_sum_halves_as_the_consoles_tree() {
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
    let shader = paired(&native_with(&words, true, 64, 0));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [0; 4], 1) else { return };
    assert_eq!(got[0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// One console wave of 64: the lanes below s2 sum the double (s0, s1) over the wave, as the
/// title's water volumetrics count their lanes (`v_add_f64` of each dword pair moved by DPP,
/// `v_permlanex16` and lane 32), and store it as a float (`wave64_tests`' program for the
/// 64-bit float sum). In the paired form: the lower lanes' sum, then the upper lanes' added.
#[test]
fn paired_lanes_sum_doubles_over_both_halves() {
    let vop3 = |op: u32, vdst: u32, src0: u32, src1: u32, src2: u32| {
        [0x35u32 << 26 | op << 16 | vdst, src0 | src1 << 9 | src2 << 18]
    };
    let v_mov_dpp8 = |vdst: u32, vsrc: u32, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc | sel << 8];
    let v_mov_row_mirror =
        |vdst: u32, vsrc: u32| [vop1(0x001, vdst, 0xFA), vsrc | 0x140 << 8 | 1 << 19 | 0xF << 24 | 0xF << 28];
    let mut words = vec![
        vop1(0x001, 6, 0), // v_mov_b32 v6, s0
        vop1(0x001, 7, 1), // v_mov_b32 v7, s1
        0x7D88_0002,       // v_cmp_gt_u32 vcc, s2, v0
        0xBE92_246A,       // s_and_saveexec_b64 s[18:19], vcc
    ];
    let skip_at = words.len();
    words.push(0xBF88_0000); // s_cbranch_execz (patched below)
    let body_start = words.len();
    words.push(0xBE8C_25C1); // s_or_saveexec_b64 s[12:13], -1
    words.extend_from_slice(&vop3(0x101, 2, 0x80, v(6), 12)); // v_cndmask_b32 v2, 0, v6, s[12:13]
    words.extend_from_slice(&vop3(0x101, 3, 0x80, v(7), 12)); // v_cndmask_b32 v3, 0, v7, s[12:13]
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, v(4), v(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, v(4), v(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, v(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, v(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, v(4), v(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, v(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, v(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.extend_from_slice(&vop3(0x164, 2, 14, v(2), 0)); // v_add_f64 v[2:3], s[14:15], v[2:3]
    words.extend_from_slice(&vop3(0x360, 16, v(2), 0xA0, 0)); // v_readlane_b32 s16, v2, 32
    words.extend_from_slice(&vop3(0x360, 17, v(3), 0xA0, 0)); // v_readlane_b32 s17, v3, 32
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    words.push(vop1(0x00F, 0, 16)); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&[0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16]); // buffer_store_dword v0, off, s[8:11], 0
    let body_words = (words.len() - body_start) as u32;
    words[skip_at] = 0xBF88_0000 | body_words;
    words.push(0xBEFE_0412); // s_mov_b64 exec, s[18:19]
    let shader = paired(&native_with(&words, true, 64, 0));
    let b = 1.25f64.to_bits();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for below in [64u32, 40, 33, 32, 4, 1] {
        let Some(got) = run_paired(&shader, [b as u32, (b >> 32) as u32, below, 0], 1) else { return };
        assert_eq!(got[0], (1.25f32 * below as f32).to_bits(), "{below} lanes of 1.25");
    }
    eprintln!("ran on the hardware device: the sum of doubles over the lanes of both halves");
}

/// What the paired form does not take is refused by name, and `emit` goes on to its other
/// forms: a thread group that does not fill its console waves, and a program outside compute.
#[test]
fn the_paired_form_refuses_what_it_does_not_take() {
    let body = stores([0, 0, 0, 0]);
    let native = native_with(&body, true, 48, 0);
    let s = stages(&native);
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = super::pairs::paired(&c, &native, &s.p, &s.io, &s.res);
    lower::TEST_HOST_WAVE.set(was);
    match r {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "paired lanes in a thread group that does not fill its console waves");
        }
        Err(e) => panic!("another error: {e}"),
        Ok(_) => panic!("48 threads taken in the paired form"),
    }
    // and the switch changes nothing for a program that runs as it is
    let whole = native_with(&body, true, 64, 0);
    let s = stages(&whole);
    let off = emitted(&whole, &s, Mode::Off).expect("emit");
    for mode in [Mode::BeforeWide, Mode::First] {
        let on = emitted(&whole, &s, mode).expect("emit");
        assert!(on.dxil == off.dxil, "{mode:?} changed a program that runs as it is");
    }
}
