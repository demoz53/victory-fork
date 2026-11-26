//! A console wave64 program's wave operations after a loop its lanes leave apart, and its
//! lane regions with the loops their branches leave (`VICTORY_AFTER_LOOP=1`), on a host of
//! 32-lane waves.
//!
//! Adapted from fxceless/victory-fork pull request #1249 by ByFxbian (GPL-3.0): its region
//! test, its two hardware comparisons against the console's results and their ISA encoders
//! (from fork pull request #993, Pilote Production); the loop fixture is the one of
//! darkworldhm/victory-fork-darkworldhm 08fdf0b47.

use super::*;
use crate::structure::{Atom, LaneExit};

/// [`program`]'s structured form with its flow replaced by `exits`, whose branches test a
/// per-lane condition of the fixture; the blocks of `wide` close on a branch the wave takes
/// together.
fn flow(exits: &[(usize, Option<usize>)], wide: &[usize]) -> structure::Structured {
    let native = super::program(0xBF88_0000);
    let ir = crate::spill::build(&native).unwrap();
    let mut st = structure::structure(&ir).unwrap();
    let when = st
        .exits
        .iter()
        .find_map(|exit| match exit {
            LaneExit::Branch { when, .. }
                if st.bools.support(*when).iter().any(|a| !matches!(a, Atom::Uniform(_))) =>
            {
                Some(*when)
            }
            _ => None,
        })
        .expect("the fixture supplies a per-lane condition");
    st.wide = wide.iter().copied().collect();
    st.exits = exits
        .iter()
        .map(|&(taken, other)| match other {
            Some(not_taken) => LaneExit::Branch { when, taken, not_taken },
            None if taken == usize::MAX => LaneExit::End,
            None => LaneExit::Jump(taken),
        })
        .collect();
    st.wave_any = st
        .exits
        .iter()
        .map(|e| matches!(e, LaneExit::Branch { .. }).then_some(true))
        .collect();
    st
}

/// `f` as a console wave64 compute program's lowering on a 32-lane host runs, with
/// `VICTORY_AFTER_LOOP` as `on`.
fn after_loop<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let native = super::program(0xBF88_0000);
    let was = (lower::TEST_AFTER_LOOP.replace(Some(on)), lower::TEST_HOST_WAVE.replace(Some(32)));
    let r = {
        let _scope = lower::after_loop_scope(&native);
        f()
    };
    lower::TEST_AFTER_LOOP.set(was.0);
    lower::TEST_HOST_WAVE.set(was.1);
    r
}

const END: (usize, Option<usize>) = (usize::MAX, None);

/// The pull request's flow: a loop of two blocks left per lane by its second. The leaving
/// block passes itself again on the way round, which our region search knows as a lane loop
/// already ([`lower::lane_loop_fix`]): the branch is found with the switch off too.
#[test]
fn a_lane_loop_exit_block_is_in_its_own_region_either_way() {
    let st = flow(&[(1, None), (2, None), (1, Some(3)), END], &[]);
    for on in [false, true] {
        assert_eq!(after_loop(on, || lower::branches_around(&st, &[2])), Some(vec![2]), "{on}");
        assert_eq!(after_loop(on, || lower::branches_around(&st, &[1])), Some(vec![2]), "{on}");
        assert_eq!(after_loop(on, || lower::branches_around(&st, &[3])), Some(vec![]), "{on}");
    }
}

/// What the switch adds: an inner loop (blocks 2 to 4) left per lane by block 3 for block 5,
/// from where the only way on is round the outer loop (blocks 1 to 5) into the inner one
/// again; the program is left from the inner loop's header, by the whole wave. Every way on
/// from the lane branch passes that header, so the region search stops there and takes the
/// header and the branch's own block for blocks every lane runs; with the switch the whole
/// inner loop is the branch's region.
#[test]
fn a_left_loop_is_its_lane_branchs_region_with_the_switch() {
    let st = flow(
        &[(1, None), (2, None), (6, Some(3)), (5, Some(4)), (2, None), (1, None), END],
        &[2],
    );
    let around = |on: bool, b: usize| after_loop(on, || lower::branches_around(&st, &[b]));
    for b in [1, 4, 5] {
        assert_eq!(around(false, b), Some(vec![3]), "block {b}: reached by some lanes only");
        assert_eq!(around(true, b), Some(vec![3]), "block {b}");
    }
    for b in [2, 3] {
        assert_eq!(around(false, b), Some(vec![]), "block {b}: as before");
        assert_eq!(around(true, b), Some(vec![3]), "block {b}: in the loop the branch leaves");
    }
    for b in [0, 6] {
        assert_eq!(around(true, b), Some(vec![]), "block {b}: outside the loops");
    }
    // (asked outside a wave64 compute program's lowering: as before)
    let was = lower::TEST_AFTER_LOOP.replace(Some(true));
    assert_eq!(lower::branches_around(&st, &[2]), Some(vec![]));
    lower::TEST_AFTER_LOOP.set(was);
}

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

/// A branch by `imm` dwords from the next instruction.
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

/// One console wave of 64 threads. Lane t goes round a loop `(t & s0) + 1` times, adding
/// `t & s1` to its sum each round; the loop is the compiler's usual one for lanes that leave
/// apart (a mask of the lanes still going round, the wave's branch back while some lane is
/// left). After it, over the lanes whose `t & s1` is not 0: their count (`s_bcnt1_i32_b64`),
/// the first of them (`v_readfirstlane_b32` of the thread index) and each one's number among
/// them (`v_mbcnt_lo/hi`). Every thread stores its sum, its number (0x55 where it has none),
/// the count and the first.
fn loop_then_wave() -> NativeProgram {
    let mut w = vec![
        vop2(0x1B, 4, 0, 0), // v_and_b32 v4, s0, v0: the lane's last round
        vop2(0x1B, 5, 1, 0), // v_and_b32 v5, s1, v0: what the lane adds a round
        vop1(0x01, 7, k(0)), // v_mov_b32 v7, 0
        vop1(0x01, 8, LIT),  // v_mov_b32 v8, 0x55
        0x55,
        sop1(0x03, 6, k(0)),      // s_mov_b32 s6, 0: the round
        sop1(0x04, 2, EXEC),      // s_mov_b64 s[2:3], exec
        sop1(0x04, 12, EXEC),     // s_mov_b64 s[12:13], exec: the lanes still going round
        sop1(0x04, 14, EXEC),     // s_mov_b64 s[14:15], exec
        sop2(0x0F, EXEC, 14, 12), // loop: s_and_b64 exec, s[14:15], s[12:13]
        vop2(0x25, 7, v(5), 7),   // v_add_nc_u32 v7, v5, v7
        sop1(0x04, EXEC, 14),     // s_mov_b64 exec, s[14:15]
        sop2(0x02, 6, 6, k(1)),   // s_add_i32 s6, s6, 1
    ];
    w.extend(vop3(0xC4, 14, 6, v(4))); // v_cmp_gt_u32 s[14:15], s6, v4: the lanes that are done
    w.extend([
        sop1(0x24, 14, 14),       // s_and_saveexec_b64 s[14:15], s[14:15]
        sop2(0x15, 12, 12, EXEC), // s_andn2_b64 s[12:13], s[12:13], exec
        sopp(0x05, -9),           // s_cbranch_scc1 loop
        sop1(0x04, EXEC, 2),      // s_mov_b64 exec, s[2:3]
        vopc(0xD5, k(0), 5),      // v_cmpx_ne_u32 exec, 0, v5
        sop1(0x10, 7, EXEC),      // s_bcnt1_i32_b64 s7, exec
        vop1(0x02, 16, v(0)),     // v_readfirstlane_b32 s16, v0
    ]);
    w.extend(vop3(0x365, 8, EXEC, k(0))); // v_mbcnt_lo_u32_b32 v8, exec_lo, 0
    w.extend(vop3(0x366, 8, EXEC + 1, v(8))); // v_mbcnt_hi_u32_b32 v8, exec_hi, v8
    w.extend([
        sop1(0x04, EXEC, 2), // s_mov_b64 exec, s[2:3]
        vop1(0x01, 9, 7),    // v_mov_b32 v9, s7
        vop1(0x01, 10, 16),  // v_mov_b32 v10, s16
    ]);
    w.extend(stores([7, 8, 9, 10]));
    native(&w)
}

/// What the console's 64 lanes store with the root constants `s0` and `s1`.
fn console(s0: u32, s1: u32) -> Vec<u32> {
    let counted: Vec<u32> = (0..64).filter(|t| t & s1 != 0).collect();
    (0..64u32)
        .flat_map(|t| {
            let number = if t & s1 != 0 { counted.iter().filter(|&&u| u < t).count() as u32 } else { 0x55 };
            [(t & s1) * ((t & s0) + 1), number, counted.len() as u32, counted[0]]
        })
        .collect()
}

/// The pull request's seven comparisons: the sums, the count, the first lane and the lanes'
/// numbers after the loop are the console's on a card of 32-lane waves, with the switch and
/// without it. (Validated by `dxil.dll` when present; not run without such a card.)
#[test]
fn a_wave_asked_after_a_loop_its_lanes_leave_apart_answers_as_the_console() {
    let cases = [(0, 1), (3, 1), (7, 5), (24, 63), (7, 32), (7, 16), (63, 9)];
    for on in [false, true] {
        on_32_lanes(&loop_then_wave(), on, &cases, console);
    }
}

/// The pull request's second comparison: lane t writes its LDS word and reads lane t ^ 32's,
/// with no barrier of the program's between them (one console wave runs them in order), and
/// takes its number in the wave. Each lane has the other half's word and its own number.
#[test]
fn lds_between_the_halves_of_a_console_wave_is_in_the_consoles_order() {
    let mut body = vec![
        vop2(0x1A, 1, k(2), 0), // v_lshlrev_b32 v1, 2, v0: this lane's byte
        vop2(0x25, 2, k(1), 0), // v_add_nc_u32 v2, 1, v0: the thread index + 1
        0xD834_0000,
        0x0000_0201,             // ds_write_b32 v1, v2
        vop2(0x1D, 3, k(32), 0), // v_xor_b32 v3, 32, v0: the partner lane
        vop2(0x1A, 3, k(2), 3),  // v_lshlrev_b32 v3, 2, v3
        0xD8D8_0000,
        0x0400_0003, // ds_read_b32 v4, v3
    ];
    body.extend(vop3(0x365, 5, 193, k(0))); // v_mbcnt_lo_u32_b32 v5, -1, 0
    body.extend(vop3(0x366, 5, 193, v(5))); // v_mbcnt_hi_u32_b32 v5, -1, v5
    body.extend(stores([4, 5, 2, 0]));
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // one 512-byte LDS unit
    }
    for on in [false, true] {
        on_32_lanes(&native, on, &[(0, 0)], |_, _| {
            (0..64).flat_map(|lane| [(lane ^ 32) + 1, lane, lane + 1, lane]).collect()
        });
    }
}

/// Translates `native` as for a host of 32-lane waves (the whole `emit`, with
/// `VICTORY_AFTER_LOOP` as `on`), and on a hardware device of such waves runs it with each
/// case's root constants `s0`, `s1` and compares the UAV's dwords with `expected`.
fn on_32_lanes(native: &NativeProgram, on: bool, cases: &[(u32, u32)], expected: impl Fn(u32, u32) -> Vec<u32>) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1};
    let p = crate::spill::build(native).expect("build native shader");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(&p, native).expect("I/O");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).unwrap();
    let was = (lower::TEST_AFTER_LOOP.replace(Some(on)), lower::TEST_HOST_WAVE.replace(Some(32)));
    let result = crate::emit::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_AFTER_LOOP.set(was.0);
    lower::TEST_HOST_WAVE.set(was.1);
    let shader = result.expect("a form for a host of 32-lane waves");
    assert_ne!(psv0_wave(&shader), (64, 64), "not one that needs 64 lanes");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation");
    }
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the query writes the options structure of the size given.
    let asked = unsafe {
        host.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut caps).cast(), size_of_val(&caps) as u32)
    };
    if asked.is_err() || caps.WaveLaneCountMax != 32 {
        eprintln!("the hardware device's waves are not 32 lanes ({}); not run", caps.WaveLaneCountMax);
        return;
    }
    let (root, pso) = host.compute_pipeline(&shader).expect("a compute pipeline on the card");
    for &(s0, s1) in cases {
        let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = [s0, s1, 0, 0][d as usize];
            }
        }
        let got = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(256)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(&got[0][..256], &expected(s0, s1)[..], "VICTORY_AFTER_LOOP {on}, s0={s0}, s1={s1}");
    }
    eprintln!(
        "ran on the hardware device (VICTORY_AFTER_LOOP {on}): {} comparisons with the console's results, waves {:?}",
        cases.len(),
        psv0_wave(&shader)
    );
}
