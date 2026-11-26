//! Lane semantics of a console wave64 on a host of 32-lane waves, against a CPU reference.
//!
//! [`model`] is a deliberately small model of one 64-lane wave, written from the public
//! RDNA 2 ISA and sharing nothing with the translator: vector registers of 64 slots, EXEC
//! as 64 bits, every instruction reading its sources before writing (source snapshots).
//! Where the ISA leaves a result undefined, or this model does not claim to know it, a lane
//! is `None` and is not compared.
//!
//! The GPU tests build synthetic GFX10 programs of one console wave (64 threads), lower
//! them for a host of 32-lane waves (each console wave becomes two host waves; cross-lane
//! code goes through the paired-wave exchange), validate the DXIL with `dxil.dll` when
//! present, run them on the hardware device and compare every lane with the model. Without
//! a hardware device of 32-lane waves they print "NOT RUN" and pass: report them as not
//! run, never as passed.
//!
//! Ported from DiogoPCastelos/victory-fork (GPL-3.0-or-later), `emit/lane_tests.rs` as of
//! 79717bfc0 by DiogoPCastelos (commits 0698344c3, 999e1e5e0, 79717bfc0): the model and the
//! tests of the paired-wave path. Its dual-bank runs of the same tests and its benchmark are
//! left out (this tree has no dual-bank form).

use super::*;

/// The reference: one console wave of 64 lanes.
pub(super) mod model {
    /// The handover's input: lane g holds `0x1000 + g`.
    pub fn input() -> [u32; 64] {
        std::array::from_fn(|g| 0x1000 + g as u32)
    }

    pub fn active(exec: u64, lane: usize) -> bool {
        exec >> lane & 1 != 0
    }

    /// `ds_bpermute_b32 dst, addr, data` for each active lane: the lane `addr[l] / 4 mod 64`
    /// is read. The ISA's rule for a source lane that is not active is not relied on here:
    /// such lanes are `None`. Lanes not active keep `old`.
    pub fn bpermute(old: &[u32; 64], addr: &[u32; 64], data: &[u32; 64], exec: u64) -> [Option<u32>; 64] {
        std::array::from_fn(|l| {
            if !active(exec, l) {
                return Some(old[l]);
            }
            let src = (addr[l] >> 2 & 63) as usize;
            active(exec, src).then(|| data[src])
        })
    }

    /// `v_readlane_b32`: lane `lane mod 64` of `v`, whatever EXEC is.
    pub fn readlane(v: &[u32; 64], lane: u32) -> u32 {
        v[(lane & 63) as usize]
    }

    /// `v_writelane_b32`: `value` into lane `lane mod 64`, whatever EXEC is.
    pub fn writelane(v: &[u32; 64], value: u32, lane: u32) -> [u32; 64] {
        let mut out = *v;
        out[(lane & 63) as usize] = value;
        out
    }

    /// `v_readfirstlane_b32`: the lowest active lane's value; `None` with EXEC empty.
    pub fn readfirstlane(v: &[u32; 64], exec: u64) -> Option<u32> {
        (exec != 0).then(|| v[exec.trailing_zeros() as usize])
    }

    /// A `v_cmp` into VCC: the predicate's bit for each active lane, 0 for the others.
    pub fn ballot(exec: u64, predicate: impl Fn(usize) -> bool) -> u64 {
        (0..64).filter(|&l| active(exec, l) && predicate(l)).fold(0, |m, l| m | 1 << l)
    }

    /// `s_ff1_i32_b64`: the lowest set bit, -1 for none.
    pub fn ff1(mask: u64) -> i32 {
        if mask == 0 { -1 } else { mask.trailing_zeros() as i32 }
    }

    /// `v_mbcnt_lo_u32_b32` then `v_mbcnt_hi_u32_b32` over `mask`: for lane g the set bits of
    /// `mask` below g (the exclusive prefix count).
    pub fn mbcnt(mask: u64) -> [u32; 64] {
        std::array::from_fn(|g| (mask & ((1u64 << g) - 1)).count_ones())
    }

    /// Lanes active by `exec` get `v`, the others keep `old`.
    pub fn masked(old: &[u32; 64], v: &[u32; 64], exec: u64) -> [u32; 64] {
        std::array::from_fn(|l| if active(exec, l) { v[l] } else { old[l] })
    }

    /// The address each lane of the shuffle programs gives: `((g ^ x) + y) mod 64`, in bytes.
    pub fn shuffle_addr(x: u32, y: u32) -> [u32; 64] {
        std::array::from_fn(|g| ((g as u32 ^ x).wrapping_add(y) & 63) << 2)
    }
}

/// The handover's deterministic expectations, on the model alone (always runs).
#[test]
fn the_model_gives_the_handovers_expected_values() {
    use model::*;
    let d = input();
    let all = u64::MAX;
    let shuffle = |x, y| bpermute(&[0; 64], &shuffle_addr(x, y), &d, all);
    assert_eq!(shuffle(0, 0)[17], Some(0x1011), "identity");
    assert_eq!(shuffle(0, 1)[31], Some(0x1020), "lane 31 reads lane 32");
    assert_eq!(shuffle(0, 63)[32], Some(0x101F), "lane 32 reads lane 31");
    for g in 0..64 {
        assert_eq!(shuffle(32, 0)[g], Some(0x1000 + (g as u32 ^ 32)), "XOR-32");
        assert_eq!(shuffle(1, 0)[g], Some(0x1000 + (g as u32 ^ 1)), "XOR-1");
    }
    let set = |lanes: &[usize]| lanes.iter().fold(0u64, |m, &l| m | 1 << l);
    let b = ballot(set(&[0, 31, 32, 63]), |_| true);
    assert_eq!((b as u32, (b >> 32) as u32), (0x8000_0001, 0x8000_0001), "ballot");
    assert_eq!(ff1(1 << 37), 37, "first active lane, high half only");
    assert_eq!(readfirstlane(&d, 1 << 37), Some(0x1025));
    assert_eq!(d.iter().map(|v| v - 0x1000).sum::<u32>(), 2016, "sum of lane ids");
    assert_eq!(mbcnt(u64::MAX), std::array::from_fn(|g| g as u32), "exclusive prefix of ones");
    // a source lane that is not active is not claimed
    assert_eq!(bpermute(&[7; 64], &shuffle_addr(0, 1), &d, !(1 << 32))[31], None);
    assert_eq!(bpermute(&[7; 64], &shuffle_addr(0, 1), &d, !(1 << 31))[31], Some(7));
}

// ---- GFX10 encodings (the decoder's tables name each one: `gfx10/opcodes.rs`) ----

/// Operand codes: SGPRs are their number, VGPRs 256 up, inline integers 128 + n.
const fn v(n: u32) -> u32 {
    256 + n
}
const fn int(n: u32) -> u32 {
    128 + n
}
const LITERAL: u32 = 255;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const EXEC: u32 = 126;

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}
const fn vop3(op: u32, vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | vdst, src1 << 9 | src0]
}
const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}
const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | ssrc0
}
const fn ds(op: u32, vdst: u32, addr: u32, data0: u32) -> [u32; 2] {
    [0x36 << 26 | op << 18, vdst << 24 | data0 << 8 | addr]
}

const fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0
}
/// A branch (its offset in dwords from the next instruction is ORed in later).
const fn sopp(op: u32) -> u32 {
    0xBF80_0000 | op << 16
}

const V_MOV_B32: u32 = 0x01;
const V_READFIRSTLANE_B32: u32 = 0x02;
const V_AND_B32: u32 = 0x1B;
const V_XOR_B32: u32 = 0x1D;
const V_LSHLREV_B32: u32 = 0x1A;
const V_ADD_NC_U32: u32 = 0x25;
const V_CMP_LE_U32: u32 = 0xC3;
const V_CMPX_GT_U32: u32 = 0xD4;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const V_MBCNT_LO: u32 = 0x365;
const V_MBCNT_HI: u32 = 0x366;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_I32_B64: u32 = 0x10;
const S_FF1_I32_B64: u32 = 0x14;
const DS_BPERMUTE_B32: u32 = 0xB3;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_AND_B64: u32 = 0x0F;
const S_OR_B64: u32 = 0x11;
const S_ANDN2_B64: u32 = 0x15;
const S_BRANCH: u32 = 0x02;
const S_CBRANCH_EXECZ: u32 = 0x08;
const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
/// Stores: `buffer_store_dword v12, v14, s[8:11], 0 offen` (v14 = lane * 4), and
/// `buffer_store_dword vN, off, s[8:11], 0 offset:o`.
const STORE_V12: [u32; 2] = [0xE070_1000, 0x8002_0C0E];
const fn store_at(vdata: u32, offset: u32) -> [u32; 2] {
    [0x38 << 26 | 0x1C << 18 | offset, 0x80 << 24 | 2 << 16 | vdata << 8]
}

/// What every program starts with: v1 = 0x1000 + lane, v14 = lane * 4, v12 = 0xDEAD0000
/// (what lanes left out of a write keep), EXEC saved in s[22:23].
fn prologue() -> Vec<u32> {
    vec![
        vop2(V_ADD_NC_U32, 1, LITERAL, 0),
        0x1000,
        vop2(V_LSHLREV_B32, 14, int(2), 0),
        vop1(V_MOV_B32, 12, LITERAL),
        0xDEAD_0000,
        sop1(S_MOV_B64, 22, EXEC),
    ]
}

/// EXEC back from s[22:23] and v12 stored at v14.
fn epilogue(words: &mut Vec<u32>) {
    words.push(sop1(S_MOV_B64, EXEC, 22));
    words.extend(STORE_V12);
}

const DEAD: u32 = 0xDEAD_0000;

/// A program lowered for a host of 32-lane waves (validated), and for 64-lane waves (validated
/// only: the RTX 3060 runs at most 32), on the hardware device.
struct Card {
    host: Host,
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    shader: HostShader,
    threads: u32,
}

fn validate(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
}

/// `body` (after [`prologue`], before the program's end) as one console wave; None (and a
/// line) without a hardware device of 32-lane waves that takes it.
fn card(what: &str, body: &[u32]) -> Option<Card> {
    card_of(what, body, 64)
}

/// [`card`] for a group of `threads` threads (console waves of 64, the last one partial).
fn card_of(what: &str, body: &[u32], threads: u32) -> Option<Card> {
    card_with(what, body, |n| {
        if let StageIo::Cs(cs) = &mut n.io {
            cs.threads = [threads, 1, 1];
        }
    })
}

/// [`card`] for a program as `adjust` changes it (group shape, thread-id VGPRs).
fn card_with(what: &str, body: &[u32], adjust: impl FnOnce(&mut NativeProgram)) -> Option<Card> {
    let mut console = native_of(body, true, 64);
    adjust(&mut console);
    let threads = match &console.io {
        StageIo::Cs(cs) => cs.threads.iter().product::<u32>(),
        _ => 64,
    };
    let p = ir::build(&console.code, console.wave64, console.vgprs).expect("ir::build");
    let wide = lowered_on(&console, &p, &HashSet::new(), 64).expect("lowered for 64-lane waves");
    validate(&wide);
    let shader = lowered(&console, &p, &HashSet::new())
        .unwrap_or_else(|e| panic!("{what}: not lowered for 32-lane waves: {e}"));
    validate(&shader);
    let host = match Host::new(false) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("{what}: no SM 6.6 hardware device; NOT RUN");
            return None;
        }
    };
    if host.wave_lanes().is_none_or(|lanes| lanes.1 > 32) {
        eprintln!("{what}: the hardware device runs waves over 32 lanes; the 32-lane path is NOT RUN here");
        return None;
    }
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("{what}: the hardware device refuses the 32-lane form: {e}"));
    eprintln!("{what}: lowered for 32-lane waves (PSV0 waves {:?}), running on the hardware device", psv0_wave(&shader));
    Some(Card { host, root, pso, shader, threads })
}

/// [`card_with`] through the whole of `emit` (its retries and forms, as the title's programs
/// are translated) for a host of 32-lane waves; also the 64-lane form, validated. A test's
/// switch (thread-local) set around the call applies to the lowering. `None` as [`card`].
fn card_emitted(what: &str, body: &[u32], adjust: impl FnOnce(&mut NativeProgram)) -> Option<Card> {
    let mut console = native_of(body, true, 64);
    adjust(&mut console);
    let threads = match &console.io {
        StageIo::Cs(cs) => cs.threads.iter().product::<u32>(),
        _ => 64,
    };
    let p = ir::build(&console.code, console.wave64, console.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let io = io::raise(&p, &console).expect("io::raise");
    let res = resource::track(&p, &console.layout, console.stage, Some(&st)).expect("resource::track");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();
    let on = |host_wave: u32| {
        let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
        let r = super::super::emit(&empty, &console, &p, &st, &io, &res);
        lower::TEST_HOST_WAVE.set(was);
        r
    };
    validate(&on(64).unwrap_or_else(|e| panic!("{what}: not emitted for 64-lane waves: {e}")));
    let shader = on(32).unwrap_or_else(|e| panic!("{what}: not emitted for 32-lane waves: {e}"));
    validate(&shader);
    let host = match Host::new(false) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("{what}: no SM 6.6 hardware device; NOT RUN");
            return None;
        }
    };
    if host.wave_lanes().is_none_or(|lanes| lanes.1 > 32) {
        eprintln!("{what}: the hardware device runs waves over 32 lanes; the 32-lane path is NOT RUN here");
        return None;
    }
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("{what}: the hardware device refuses the 32-lane form: {e}"));
    eprintln!("{what}: emitted for 32-lane waves (PSV0 waves {:?}), running on the hardware device", psv0_wave(&shader));
    Some(Card { host, root, pso, shader, threads })
}

impl Card {
    /// Runs the program with root constants s0..s3; the UAV's first 64 dwords.
    fn run(&self, s: [u32; 4]) -> Vec<u32> {
        let mut ud = vec![0u32; self.shader.user_data_dwords().max(1) as usize];
        for u in &self.shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = s[d as usize];
            }
        }
        let mut out = self
            .host
            .dispatch(
                &self.root,
                &self.pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(self.threads.max(64))],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        out.swap_remove(0)
    }
}

/// Every lane the model claims, compared; the first few differences named.
fn compare(what: &str, got: &[u32], want: &[Option<u32>; 64]) {
    let wrong: Vec<String> = (0..64)
        .filter_map(|l| match want[l] {
            Some(w) if got[l] != w => Some(format!("lane {l}: got {:#x}, want {w:#x}", got[l])),
            _ => None,
        })
        .take(6)
        .collect();
    assert!(wrong.is_empty(), "{what}: {}", wrong.join("; "));
}

fn some(v: [u32; 64]) -> [Option<u32>; 64] {
    v.map(Some)
}

/// xorshift64*: the same "random" cases every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    crate::emit::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// `ds_bpermute_b32` with lane g reading lane `((g ^ s0) + s1) mod 64`, under EXEC = s[2:3];
/// `in_place`: the destination is the data register itself (old values must be read first).
fn shuffle_program(in_place: bool) -> Vec<u32> {
    let mut w = prologue();
    w.extend([
        vop2(V_XOR_B32, 2, 0, 0),       // v2 = s0 ^ v0
        vop2(V_ADD_NC_U32, 2, 1, 2),    // v2 = s1 + v2
        vop2(V_AND_B32, 2, int(63), 2), // v2 &= 63
        vop2(V_LSHLREV_B32, 2, int(2), 2),
        sop1(S_MOV_B64, EXEC, 2), // EXEC = s[2:3]
    ]);
    if in_place {
        w.extend(ds(DS_BPERMUTE_B32, 1, 2, 1));
        w.push(S_WAITCNT_LGKM0);
        w.push(vop1(V_MOV_B32, 12, v(1)));
    } else {
        w.extend(ds(DS_BPERMUTE_B32, 12, 2, 1));
        w.push(S_WAITCNT_LGKM0);
    }
    epilogue(&mut w);
    w
}

fn shuffle_want(x: u32, y: u32, exec: u64) -> [Option<u32>; 64] {
    model::bpermute(&[DEAD; 64], &model::shuffle_addr(x, y), &model::input(), exec)
}

fn split(m: u64) -> (u32, u32) {
    (m as u32, (m >> 32) as u32)
}

/// Identity, the 31/32 boundary both ways, XOR-1, XOR-32, and every (source, destination)
/// pair of the wave (4,096: 64 rotations), all lanes active; then the same in place.
#[test]
fn a_shuffle_reads_every_lane_of_the_console_wave() {
    let _one = one_at_a_time();
    for in_place in [false, true] {
        let what = if in_place { "ds_bpermute in place" } else { "ds_bpermute" };
        let Some(card) = card(what, &shuffle_program(in_place)) else { return };
        let all = u64::MAX;
        for (x, y, name) in [(0, 0, "identity"), (0, 1, "31 reads 32"), (0, 63, "32 reads 31"), (1, 0, "XOR-1"), (32, 0, "XOR-32"), (63, 0, "XOR-63")] {
            let got = card.run([x, y, u32::MAX, u32::MAX]);
            compare(&format!("{what}, {name}"), &got, &shuffle_want(x, y, all));
        }
        for y in 0..64 {
            let got = card.run([0, y, u32::MAX, u32::MAX]);
            compare(&format!("{what}, rotation {y}"), &got, &shuffle_want(0, y, all));
        }
        eprintln!("{what}: identity, 31<->32, XOR-1/32/63 and all 4,096 pairs as the model");
    }
}

/// `VICTORY_PERMUTE_HALVES=0`: a shuffle across the two host waves is refused, as before.
#[test]
fn a_shuffle_across_the_halves_is_refused_with_the_switch_off() {
    let native = native_of(&shuffle_program(false), true, 64);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let was = lower::TEST_PERMUTE_HALVES.replace(Some(false));
    let off = lowered(&native, &p, &HashSet::new());
    lower::TEST_PERMUTE_HALVES.set(Some(true));
    let on = lowered(&native, &p, &HashSet::new());
    lower::TEST_PERMUTE_HALVES.set(was);
    let Err(why) = off else {
        panic!("refused with the switch off");
    };
    let why = why.to_string();
    assert!(why.contains("a lane permute across two host waves"), "{why}");
    assert!(on.is_ok(), "taken with it on: {:?}", on.err());
}

/// The same shuffles under random EXEC masks (inactive lanes keep their value; a lane whose
/// source is inactive is not compared), with random XOR and rotation.
#[test]
fn a_shuffle_under_random_masks_keeps_the_inactive_lanes() {
    let _one = one_at_a_time();
    for in_place in [false, true] {
        let what = if in_place { "ds_bpermute in place, random masks" } else { "ds_bpermute, random masks" };
        let Some(card) = card(what, &shuffle_program(in_place)) else { return };
        let mut rng = Rng(0x5EED_0000_0000_0001 + u64::from(in_place));
        for case in 0..64 {
            let exec = match case {
                0 => 0,
                1 => 0xFFFF_FFFF,
                2 => 0xFFFF_FFFF_0000_0000,
                3 => 1 << 31 | 1 << 32,
                _ => rng.next() & rng.next() | rng.next() & 1 << (rng.next() % 64),
            };
            let (x, y) = ((rng.next() % 64) as u32, (rng.next() % 64) as u32);
            let (lo, hi) = split(exec);
            let got = card.run([x, y, lo, hi]);
            compare(&format!("{what}, exec {exec:#018x}, x {x}, y {y}"), &got, &shuffle_want(x, y, exec));
        }
        eprintln!("{what}: 64 cases as the model");
    }
}

/// `v_readlane_b32` from lanes 0, 31, 32, 63 and random ones, under random EXEC (the lanes
/// active take the value), including lanes read while inactive.
#[test]
fn a_lane_read_reaches_both_halves() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_READLANE_B32, 20, v(1), 0));
    w.push(vop1(V_MOV_B32, 12, 20));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_readlane", &w) else { return };
    let mut rng = Rng(0x5EED_0000_0000_0002);
    for case in 0..40u32 {
        let lane = if case < 4 { [0, 31, 32, 63][case as usize] } else { (rng.next() % 64) as u32 };
        let exec = if case < 8 { u64::MAX } else { rng.next() | 1 };
        let (lo, hi) = split(exec);
        let got = card.run([lane, 0, lo, hi]);
        let value = model::readlane(&model::input(), lane);
        compare(&format!("v_readlane lane {lane}, exec {exec:#x}"), &got, &some(model::masked(&[DEAD; 64], &[value; 64], exec)));
    }
    eprintln!("v_readlane: 40 cases as the model");
}

/// `v_writelane_b32` into lanes 31, 32 and random ones, under random EXEC (it ignores EXEC).
#[test]
fn a_lane_write_reaches_both_halves() {
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 12, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_WRITELANE_B32, 12, 1, 0));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_writelane", &w) else { return };
    let mut rng = Rng(0x5EED_0000_0000_0003);
    for case in 0..40u32 {
        let lane = if case < 4 { [31, 32, 0, 63][case as usize] } else { (rng.next() % 64) as u32 };
        let exec = if case < 4 { u64::MAX } else { rng.next() };
        let (lo, hi) = split(exec);
        let value = 0xC0DE_0000 | case;
        let got = card.run([lane, value, lo, hi]);
        compare(&format!("v_writelane lane {lane}, exec {exec:#x}"), &got, &some(model::writelane(&model::input(), value, lane)));
    }
    eprintln!("v_writelane: 40 cases as the model");
}

/// `VICTORY_SHORT_WAVE_LAST`: a group that is a multiple of 32 threads translates the same
/// with and without the rule; one that is not (48 or 80 threads) only with it off as before.
#[test]
fn the_short_wave_rule_changes_only_groups_that_are_no_multiple_of_32() {
    let dxil = |threads: u32, on: bool| {
        let native = native_of(&shuffle_program(false), true, threads);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let was = lower::TEST_SHORT_WAVE_LAST.replace(Some(on));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_SHORT_WAVE_LAST.set(was);
        shader.expect("lowered").dxil
    };
    for threads in [64, 96, 128] {
        assert_eq!(dxil(threads, true), dxil(threads, false), "{threads} threads");
    }
    for threads in [48, 80] {
        assert_ne!(dxil(threads, true), dxil(threads, false), "{threads} threads");
    }
}

/// `v_readlane_b32` and `v_writelane_b32` with a lane number past the wave: the ISA takes the
/// low six bits of it (lane 64 is lane 0, lane 101 is lane 37, -1 is lane 63). The source's
/// tests name lanes below 64 only on this path (its dual-bank form found the rule); these
/// cases are added here.
#[test]
fn a_lane_number_is_taken_modulo_the_wave() {
    let lanes = [64u32, 95, 96, 127, 101, 0x1_0025, u32::MAX];
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_READLANE_B32, 20, v(1), 0));
    w.push(vop1(V_MOV_B32, 12, 20));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(read) = card("v_readlane, lanes past the wave", &w) else { return };
    for lane in lanes {
        let got = read.run([lane, 0, u32::MAX, u32::MAX]);
        let value = model::readlane(&model::input(), lane);
        compare(&format!("v_readlane lane {lane:#x}"), &got, &some([value; 64]));
    }
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 12, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_WRITELANE_B32, 12, 1, 0));
    epilogue(&mut w);
    let Some(write) = card("v_writelane, lanes past the wave", &w) else { return };
    for (case, lane) in lanes.into_iter().enumerate() {
        let value = 0xC0DE_1000 | case as u32;
        let got = write.run([lane, value, u32::MAX, u32::MAX]);
        compare(&format!("v_writelane lane {lane:#x}"), &got, &some(model::writelane(&model::input(), value, lane)));
    }
    eprintln!("v_readlane and v_writelane: 7 lanes past the wave each, as the model");
}

/// `VICTORY_LANE_MODULO`: a program whose lane numbers are constants (the compiler's
/// whole-wave sum reads lanes 32 and 0) translates the same with and without the rule; one
/// with a computed lane only with it off as before.
#[test]
fn the_lane_modulo_rule_changes_only_computed_lanes() {
    let dxil = |words: &[u32], on: bool| {
        let native = native_of(words, true, 64);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let was = lower::TEST_LANE_MODULO.replace(Some(on));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_LANE_MODULO.set(was);
        shader.expect("lowered").dxil
    };
    let mut sum = prologue();
    sum.push(vop1(V_MOV_B32, 7, v(0)));
    sum.extend(crate::emit::wave::WAVE_OR_V7_TO_S7);
    sum.push(vop1(V_MOV_B32, 12, 7));
    epilogue(&mut sum);
    assert_eq!(dxil(&sum, true), dxil(&sum, false), "constant lanes");
    let mut read = prologue();
    read.extend(vop3(V_READLANE_B32, 20, v(1), 0));
    read.push(vop1(V_MOV_B32, 12, 20));
    epilogue(&mut read);
    assert_ne!(dxil(&read, true), dxil(&read, false), "a computed lane");
}

/// `v_readfirstlane_b32` under masks whose first lane is in either half, among them the
/// handover's high-half-only mask (lane 37).
#[test]
fn the_first_active_lane_is_found_in_either_half() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.push(vop1(V_READFIRSTLANE_B32, 20, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 22));
    w.push(vop1(V_MOV_B32, 12, 20));
    w.extend(STORE_V12);
    let _one = one_at_a_time();
    let Some(card) = card("v_readfirstlane", &w) else { return };
    let mut rng = Rng(0x5EED_0000_0000_0004);
    for case in 0..40u32 {
        let exec = match case {
            0 => 1 << 37,
            1 => 1 << 32,
            2 => 1 << 31 | 1 << 63,
            3 => u64::MAX,
            _ => rng.next() >> (rng.next() % 64) | 1 << 63,
        };
        let (lo, hi) = split(exec);
        let got = card.run([0, 0, lo, hi]);
        let first = model::readfirstlane(&model::input(), exec);
        compare(&format!("v_readfirstlane exec {exec:#x}"), &got, &[first; 64]);
    }
    eprintln!("v_readfirstlane: 40 cases as the model");
}

/// A ballot through `v_cmp` (lanes g >= s0 among the active ones), read as data: its two
/// halves, `s_ff1_i32_b64` and `s_bcnt1_i32_b64` of it.
#[test]
fn a_ballot_holds_both_halves() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.push(vopc(V_CMP_LE_U32, 0, 0)); // vcc = s0 <= v0
    w.push(sop1(S_MOV_B64, EXEC, 22));
    w.push(sop1(S_FF1_I32_B64, 24, VCC_LO));
    w.push(sop1(S_BCNT1_I32_B64, 25, VCC_LO));
    w.push(vop1(V_MOV_B32, 3, VCC_LO));
    w.push(vop1(V_MOV_B32, 4, VCC_HI));
    w.push(vop1(V_MOV_B32, 5, 24));
    w.push(vop1(V_MOV_B32, 6, 25));
    for (i, r) in [3, 4, 5, 6].into_iter().enumerate() {
        w.extend(store_at(r, 4 * i as u32));
    }
    let _one = one_at_a_time();
    let Some(card) = card("ballot", &w) else { return };
    let set = |lanes: &[u32]| lanes.iter().fold(0u64, |m, &l| m | 1 << l);
    let mut rng = Rng(0x5EED_0000_0000_0005);
    for case in 0..48u32 {
        let (exec, from) = match case {
            0 => (set(&[0, 31, 32, 63]), 0),
            1 => (1 << 37, 0),
            2 => (u64::MAX, 0),
            3 => (u64::MAX, 32),
            4 => (0, 0),
            _ => (rng.next(), (rng.next() % 65) as u32),
        };
        let (lo, hi) = split(exec);
        let got = card.run([from, 0, lo, hi]);
        let b = model::ballot(exec, |l| from as usize <= l);
        let want = [b as u32, (b >> 32) as u32, model::ff1(b) as u32, b.count_ones()];
        assert_eq!(&got[..4], &want, "ballot of g >= {from} over exec {exec:#x}: [lo, hi, ff1, bcnt]");
    }
    eprintln!("ballot: 48 cases as the model (the handover's {{0,31,32,63}} gives 0x80000001 twice)");
}

/// `v_mbcnt_lo` then `v_mbcnt_hi` over a mask in s[0:1]: each lane's exclusive prefix count,
/// all ones first (lane g gets g).
#[test]
fn a_prefix_count_runs_across_the_halves() {
    let mut w = prologue();
    w.extend(vop3(V_MBCNT_LO, 15, 0, int(0)));
    w.extend(vop3(V_MBCNT_HI, 15, 1, v(15)));
    w.push(vop1(V_MOV_B32, 12, v(15)));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_mbcnt", &w) else { return };
    let mut rng = Rng(0x5EED_0000_0000_0006);
    for case in 0..40 {
        let mask = match case {
            0 => u64::MAX,
            1 => 0,
            2 => 1 << 31 | 1 << 32,
            _ => rng.next(),
        };
        let (lo, hi) = split(mask);
        let got = card.run([lo, hi, u32::MAX, u32::MAX]);
        compare(&format!("v_mbcnt over {mask:#x}"), &got, &some(model::mbcnt(mask)));
    }
    eprintln!("v_mbcnt: 40 cases as the model (all ones: lane g gets g)");
}

/// The console compiler's whole-wave sum (identity in inactive lanes, DPP, `v_permlanex16`,
/// `v_readlane` 32 and 0) of the lane ids: 2016 in every lane.
#[test]
fn a_whole_wave_sum_of_lane_ids_is_2016() {
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 7, v(0)));
    // (the compiler's wave OR with v_add_nc_u32 for v_or_b32)
    w.extend(crate::emit::wave::WAVE_OR_V7_TO_S7.iter().map(|&x| if x >> 25 == 0x1C { x & 0x01FF_FFFF | V_ADD_NC_U32 << 25 } else { x }));
    w.push(vop1(V_MOV_B32, 12, 7));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("wave sum", &w) else { return };
    let got = card.run([0; 4]);
    compare("wave sum of lane ids", &got, &[Some(2016); 64]);
    eprintln!("wave sum: 2016 in every lane");
}

/// Divergence: EXEC = the lanes below s0 (`v_cmpx`), a write, EXEC restored. The lanes left
/// out keep their old value, on both sides of the 31/32 boundary.
#[test]
fn lanes_left_out_by_exec_keep_their_values() {
    let mut w = prologue();
    w.push(vopc(V_CMPX_GT_U32, 0, 0)); // exec = s0 > v0
    w.push(vop1(V_MOV_B32, 12, v(1)));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_cmpx divergence", &w) else { return };
    for below in [0u32, 1, 31, 32, 33, 63, 64] {
        let got = card.run([below, 0, 0, 0]);
        let exec = if below >= 64 { u64::MAX } else { (1u64 << below) - 1 };
        compare(&format!("lanes below {below}"), &got, &some(model::masked(&[DEAD; 64], &model::input(), exec)));
    }
    eprintln!("v_cmpx: lanes 0, 1, 31, 32, 33, 63, 64 active as the model");
}

/// Partial groups and several console waves per group: 48 threads (one console wave whose
/// upper host wave has 16 threads) and 96 (two console waves, the second with 32). Each
/// console wave shuffles among its own lanes; a lane that reads a lane the group does not
/// have is not compared (not there on the console: outside EXEC). The barriers must not
/// hang with a host wave short.
#[test]
fn a_shuffle_in_partial_groups_stays_in_its_console_wave() {
    let _one = one_at_a_time();
    for threads in [48u32, 96] {
        let what = format!("ds_bpermute, {threads} threads");
        let Some(card) = card_of(&what, &shuffle_program(false), threads) else { return };
        let mut rng = Rng(0x5EED_0000_0000_0007 + u64::from(threads));
        for case in 0..32 {
            let (x, y) = match case {
                0 => (0, 1),
                1 => (0, 63),
                2 => (32, 0),
                _ => ((rng.next() % 64) as u32, (rng.next() % 64) as u32),
            };
            let got = card.run([x, y, u32::MAX, u32::MAX]);
            for wave in 0..threads.div_ceil(64) {
                let first = wave * 64;
                let lanes = (threads - first).min(64);
                let exec = if lanes == 64 { u64::MAX } else { (1u64 << lanes) - 1 };
                let data: [u32; 64] = std::array::from_fn(|l| 0x1000 + first + l as u32);
                let mut want = model::bpermute(&[DEAD; 64], &model::shuffle_addr(x, y), &data, exec);
                for (l, w) in want.iter_mut().enumerate() {
                    if l as u32 >= lanes {
                        *w = None;
                    }
                }
                let got_wave: Vec<u32> =
                    (0..64).map(|l| got.get((first + l) as usize).copied().unwrap_or(0)).collect();
                compare(&format!("{what}, console wave {wave}, x {x}, y {y}"), &got_wave, &want);
            }
        }
        eprintln!("{what}: 32 cases as the model in every console wave");
    }
}

/// A per-lane if/else as the console compiler writes it: `v_cmp`, `s_and_saveexec_b64`,
/// `s_cbranch_execz` past the `then` lanes, `s_andn2_b64` to the `else` lanes and
/// `s_cbranch_execz` past them, EXEC restored. Lanes below s0 take v1, the others 7, on
/// both sides of the 31/32 boundary, including when one side has no lane (both branches
/// taken).
#[test]
fn an_if_else_on_lane_masks_runs_each_side_in_its_lanes() {
    let execz = sopp(S_CBRANCH_EXECZ);
    let mut w = prologue();
    w.push(vopc(0xC4, 0, 0)); // v_cmp_gt_u32 vcc, s0, v0
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC_LO)); // s[20:21] = exec; exec &= vcc
    let skip_then = w.len();
    w.push(execz);
    w.push(vop1(V_MOV_B32, 12, v(1)));
    let else_at = w.len();
    w.push(sop2(S_ANDN2_B64, EXEC, 20, EXEC)); // exec = saved & ~exec
    let skip_else = w.len();
    w.push(execz);
    w.push(vop1(V_MOV_B32, 12, int(7)));
    let end = w.len();
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[skip_then] |= rel(skip_then, else_at);
    w[skip_else] |= rel(skip_else, end);
    let _one = one_at_a_time();
    let Some(card) = card("if/else", &w) else { return };
    for below in [0u32, 1, 31, 32, 33, 63, 64] {
        let got = card.run([below, 0, 0, 0]);
        let want: [u32; 64] = std::array::from_fn(|l| if (l as u32) < below { 0x1000 + l as u32 } else { 7 });
        compare(&format!("if/else, lanes below {below} take v1"), &got, &some(want));
    }
    eprintln!("if/else: lanes 0, 1, 31, 32, 33, 63, 64 as the model");
}

/// 3D thread groups: lanes follow the linear thread id `x + X * (y + Y * z)` (x fastest), so
/// each thread computes its linear id from v0, v1, v2 and shuffles by it. Groups of 4x4x4
/// (one console wave) and 8x4x4 (two). Every thread must have its own (x, y, z), and each
/// console wave's shuffle stays among its own lanes.
fn three_d(threads: [u32; 3]) {
    let [x, y, _] = threads;
    let (lx, ly) = (x.trailing_zeros(), y.trailing_zeros());
    let mut w = vec![
        vop2(V_LSHLREV_B32, 3, int(lx + ly), 2), // v3 = z << (lx + ly)
        vop2(V_LSHLREV_B32, 4, int(lx), 1),      // v4 = y << lx
        vop2(V_ADD_NC_U32, 3, v(4), 3),
        vop2(V_ADD_NC_U32, 3, v(0), 3), // v3 = the linear id
        vop2(V_ADD_NC_U32, 1, LITERAL, 3),
        0x1000,
        vop2(V_LSHLREV_B32, 14, int(2), 3),
        vop1(V_MOV_B32, 12, LITERAL),
        DEAD,
        sop1(S_MOV_B64, 22, EXEC),
        vop2(V_XOR_B32, 2, 0, 3), // v2 = s0 ^ lin
        vop2(V_ADD_NC_U32, 2, 1, 2),
        vop2(V_AND_B32, 2, int(63), 2),
        vop2(V_LSHLREV_B32, 2, int(2), 2),
        sop1(S_MOV_B64, EXEC, 2),
    ];
    w.extend(ds(DS_BPERMUTE_B32, 12, 2, 1));
    w.push(S_WAITCNT_LGKM0);
    epilogue(&mut w);
    let group: u32 = threads.iter().product();
    let what = format!("3D group {threads:?}");
    let Some(card) = card_with(&what, &w, move |n| {
        if let StageIo::Cs(cs) = &mut n.io {
            cs.threads = threads;
            cs.rsrc2 |= 2 << 11; // v0, v1, v2: x, y, z
        }
    }) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0008 + u64::from(group));
    for case in 0..24 {
        let (sx, sy) = match case {
            0 => (0, 0),
            1 => (0, 1),
            2 => (32, 0),
            _ => ((rng.next() % 64) as u32, (rng.next() % 64) as u32),
        };
        let got = card.run([sx, sy, u32::MAX, u32::MAX]);
        for wave in 0..group / 64 {
            let first = wave * 64;
            let data: [u32; 64] = std::array::from_fn(|l| 0x1000 + first + l as u32);
            let want = model::bpermute(&[DEAD; 64], &model::shuffle_addr(sx, sy), &data, u64::MAX);
            let got_wave: Vec<u32> = (0..64).map(|l| got[(first + l) as usize]).collect();
            compare(&format!("{what}, console wave {wave}, x {sx}, y {sy}"), &got_wave, &want);
        }
    }
    eprintln!("{what}: 24 cases as the model in every console wave");
}

#[test]
fn three_dimensional_groups_number_lanes_x_first() {
    let _one = one_at_a_time();
    three_d([4, 4, 4]);
    three_d([8, 4, 4]);
}

/// A loop whose trip count is per lane, `(lane & 7) + (lane >> 5)` (so the halves differ),
/// leaving it lane by lane (`s_and_b64 exec, exec, vcc`, `s_cbranch_execz`), with an if
/// inside it on odd iterations (`s_and_saveexec_b64`, `s_cbranch_execz`, `s_or_b64` back).
/// Each lane adds 1 per iteration and 100 more on odd ones.
#[test]
fn a_loop_with_an_if_inside_runs_each_lane_its_own_count() {
    let mut w = prologue();
    w.extend([
        vop1(V_MOV_B32, 12, int(0)),       // acc
        vop1(V_MOV_B32, 6, int(0)),        // i
        vop2(0x16, 8, int(5), 0),          // v_lshrrev_b32 v8, 5, v0
        vop2(V_AND_B32, 5, int(7), 0),     // v5 = lane & 7
        vop2(V_ADD_NC_U32, 5, v(8), 5),    // t = (lane & 7) + (lane >> 5)
    ]);
    let top = w.len();
    w.push(vopc(0xC1, v(6), 5)); // v_cmp_lt_u32 vcc, v6, v5: i < t
    w.push(sop2(S_AND_B64, EXEC, EXEC, VCC_LO));
    let leave = w.len();
    w.push(sopp(S_CBRANCH_EXECZ));
    w.push(vop2(V_AND_B32, 7, int(1), 6));
    w.push(vopc(0xC5, int(0), 7)); // v_cmp_ne_u32 vcc, 0, v7: i odd
    w.push(sop1(S_AND_SAVEEXEC_B64, 24, VCC_LO));
    let skip = w.len();
    w.push(sopp(S_CBRANCH_EXECZ));
    w.push(vop2(V_ADD_NC_U32, 12, int(64), 12));
    w.push(vop2(V_ADD_NC_U32, 12, int(36), 12)); // + 100
    let join = w.len();
    w.push(sop2(S_OR_B64, EXEC, EXEC, 24));
    w.push(vop2(V_ADD_NC_U32, 12, int(1), 12));
    w.push(vop2(V_ADD_NC_U32, 6, int(1), 6));
    let back = w.len();
    w.push(sopp(S_BRANCH));
    let end = w.len();
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[leave] |= rel(leave, end);
    w[skip] |= rel(skip, join);
    w[back] |= rel(back, top);
    let _one = one_at_a_time();
    let Some(card) = card("loop with an if", &w) else { return };
    let got = card.run([0; 4]);
    let want: [u32; 64] = std::array::from_fn(|l| {
        let t = (l as u32 & 7) + (l as u32 >> 5);
        (0..t).map(|i| 1 + if i & 1 == 1 { 100 } else { 0 }).sum()
    });
    compare("loop with an if inside", &got, &some(want));
    eprintln!("loop with an if: every lane its own count, as the model");
}

/// The model of a DPP16 `row_shr:n` compare `src0 > vsrc1` (unsigned) over a 64-lane wave:
/// lane l reads src0 of lane l - n in its row of 16; a lane whose source is past the row's
/// start takes 0 with bound_ctrl, else is not written (its bit 0); rows off in `rows` and
/// lanes outside EXEC are not written either. Returns the bits, and the lanes not claimed:
/// a source lane in the row but outside EXEC (with FI clear the ISA treats it as out of
/// range; the translator reads it as it is; not settled, so not compared).
fn dpp_shr_compare(src0: &[u32; 64], vsrc1: &[u32; 64], n: usize, bound: bool, rows: u32, exec: u64) -> (u64, u64) {
    let (mut bits, mut unknown) = (0u64, 0u64);
    for l in 0..64 {
        if !model::active(exec, l) || rows >> (l / 16) & 1 == 0 {
            continue;
        }
        let from = (l % 16).checked_sub(n).map(|k| l / 16 * 16 + k);
        let value = match from {
            Some(s) if model::active(exec, s) => Some(src0[s]),
            Some(_) => {
                unknown |= 1 << l;
                continue;
            }
            None if bound => Some(0),
            None => None,
        };
        if value.is_some_and(|v| v > vsrc1[l]) {
            bits |= 1 << l;
        }
    }
    (bits, unknown)
}

/// `v_cmp_gt_u32` (or `v_cmpx_gt_u32`) with DPP16 `row_shr:1` on src0 = the lane number,
/// vsrc1 = s0, under EXEC = s[2:3]. The compare: VCC's two halves at dwords 0 and 1. The
/// `v_cmpx`: v12 = 1 in the lanes the new EXEC keeps.
fn dpp_compare_program(cmpx: bool, bound: bool, rows: u32) -> Vec<u32> {
    let ext = 2 | 0x111 << 8 | u32::from(bound) << 19 | 0xF << 24 | rows << 28; // src0 v2, row_shr:1
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 2, v(0)));
    w.push(vop1(V_MOV_B32, 4, 0)); // v4 = s0
    w.push(sop1(S_MOV_B64, EXEC, 2));
    let op = if cmpx { V_CMPX_GT_U32 } else { 0xC4 };
    w.push(0x7C00_0000 | op << 17 | 4 << 9 | 0xFA);
    w.push(ext);
    if cmpx {
        w.push(vop1(V_MOV_B32, 12, int(1)));
        epilogue(&mut w);
    } else {
        w.push(sop1(S_MOV_B64, EXEC, 22));
        w.push(vop1(V_MOV_B32, 3, VCC_LO));
        w.push(vop1(V_MOV_B32, 5, VCC_HI));
        w.extend(store_at(3, 0));
        w.extend(store_at(5, 4));
    }
    w
}

#[test]
#[ignore = "fails here: a compare with DPP is refused on every path (not one of the four defects)"]
fn a_dpp_compare_reads_its_row_neighbour() {
    let _one = one_at_a_time();
    let lanes: [u32; 64] = std::array::from_fn(|l| l as u32);
    for cmpx in [false, true] {
        for (bound, rows) in [(true, 0xF), (false, 0xF), (true, 0b0101), (false, 0b1010)] {
            let what = format!("DPP row_shr:1 {} bound_ctrl {bound} rows {rows:#06b}", if cmpx { "v_cmpx" } else { "v_cmp" });
            let Some(card) = card(&what, &dpp_compare_program(cmpx, bound, rows)) else { return };
            let mut rng = Rng(0x5EED_0000_0000_0009 + u64::from(rows) + u64::from(bound) * 16 + u64::from(cmpx) * 32);
            for case in 0..24 {
                let exec = if case < 2 { u64::MAX } else { rng.next() };
                let threshold = [0u32, 7, 20, 40][case % 4];
                let (lo, hi) = split(exec);
                let got = card.run([threshold, 0, lo, hi]);
                let (want, unknown) = dpp_shr_compare(&lanes, &[threshold; 64], 1, bound, rows, exec);
                if cmpx {
                    let expect: [Option<u32>; 64] = std::array::from_fn(|l| {
                        (unknown >> l & 1 == 0).then_some(if want >> l & 1 == 1 { 1 } else { DEAD })
                    });
                    compare(&format!("{what}, exec {exec:#x}, s0 {threshold}"), &got, &expect);
                } else {
                    let got64 = u64::from(got[0]) | u64::from(got[1]) << 32;
                    assert_eq!(
                        got64 & !unknown,
                        want,
                        "{what}, exec {exec:#x}, s0 {threshold}: VCC (lanes {unknown:#x} not compared)"
                    );
                }
            }
            eprintln!("{what}: 24 cases as the model");
        }
    }
}

/// A console wave32 program with console lanes (`v_mbcnt`) in a group of 32 by 2: thread
/// (x, y) stores 1 at dword 32y + x. Every dword must be written: the host's relabelled thread
/// numbers are each given once. A 32-lane card fills a two-dimensional group from tiles of 16
/// by 2, so the first flat id of a host wave rounded down gave both waves the same numbers;
/// the waves now take their threads from a group counter (`VICTORY_THREAD_TICKET`). With the
/// switch off (the old rounding) the count of dwords written is printed, not checked: it
/// depends on the card. (fxceless/victory-fork pull request #1261 by Garu667, GPL-3.0.)
#[test]
fn the_console_threads_of_a_two_dimensional_group_are_each_given_once() {
    let words = [
        0xD765_0003, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x0001_007E,
        0x3408_0285, // v_lshlrev_b32 v4, 5, v1
        0x4A08_0104, // v_add_nc_u32 v4, v4, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1 | 1 << 11, // 5 user SGPRs, thread ids x and y in v0, v1
        threads: [32, 2, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let lower_with = |ticket: bool| {
        let was = lower::TEST_THREAD_TICKET.replace(Some(ticket));
        let shader = lowered_on(&native, &p, &HashSet::new(), 32);
        lower::TEST_THREAD_TICKET.set(was);
        shader.expect("lower")
    };
    let (shader, old) = (lower_with(true), lower_with(false));
    assert_ne!(shader.dxil, old.dxil, "the counter is used for a group of two host waves");
    validate(&shader);
    // (one host wave: no counter)
    let mut one = native_of(&words, false, 32);
    one.io = StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | 1 << 11, threads: [16, 2, 1] });
    let p1 = ir::build(&one.code, one.wave64, one.vgprs).expect("ir::build");
    let same = |ticket: bool| {
        let was = lower::TEST_THREAD_TICKET.replace(Some(ticket));
        let shader = lowered_on(&one, &p1, &HashSet::new(), 32);
        lower::TEST_THREAD_TICKET.set(was);
        shader.expect("lower").dxil
    };
    assert_eq!(same(true), same(false), "a group of one host wave translates as before");
    let _one = one_at_a_time();
    let Ok(host) = Host::new(false) else {
        eprintln!("thread ticket: no SM 6.6 hardware device; NOT RUN");
        return;
    };
    if host.wave_lanes().is_none_or(|lanes| lanes.1 > 32) {
        eprintln!("thread ticket: the hardware device runs waves over 32 lanes; NOT RUN here");
        return;
    }
    let run = |shader: &HostShader| -> Vec<u32> {
        let (root, pso) = host.compute_pipeline(shader).expect("a compute pipeline on the card");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        out[0][..64].to_vec()
    };
    let before = run(&old).iter().filter(|&&d| d == 1).count();
    eprintln!("thread ticket: the old rounding on this device writes {before} of 64 dwords");
    let got = run(&shader);
    let missing: Vec<usize> = (0..64).filter(|&i| got[i] != 1).collect();
    assert!(missing.is_empty(), "dwords not written: {missing:?}");
    eprintln!("thread ticket: 64 of 64 console threads, each its own, on the hardware device");
}

const V_OR_B32: u32 = 0x1C;
const V_LSHRREV_B32: u32 = 0x16;
const V_MUL_U32_U24: u32 = 0x0B;
const V_CNDMASK_B32: u32 = 0x01;
const V_CMP_LT_U32: u32 = 0xC1;
const V_CMP_EQ_U32: u32 = 0xC2;
const V_CMP_GT_U32: u32 = 0xC4;

/// Inside `if (lane < s0)`, each lane's mask v7 = 1 << (lane >> 2) is ORed over the wave and a
/// loop visits its bits with the bits still to visit joined in s8 itself (the scalar form of the
/// title's light loop, `own_bits`), each lane adding the numbers of its own bits to v5. The loop
/// is left with one bit to go, so s8 then holds the wave OR's highest bit; every lane stores
/// s8 | v5 << 16. Taken as each host wave's own OR (`VICTORY_OWN_BITS_SCALAR=0`), the lanes of
/// the lower host wave store their own half's highest bit instead (avatarmanz/VIctory-7900XTX
/// PR #286, sharan104).
#[test]
fn a_scalar_loops_last_bit_after_it_is_the_whole_waves() {
    let mut w = prologue();
    w.push(S_WAITCNT_LGKM0);
    // s[4:11] (the UAV's descriptor and its table) kept in s[24:31]: the loop uses them
    for k in [4, 6, 8, 10] {
        w.push(sop1(S_MOV_B64, 20 + k, k));
    }
    w.extend([
        vop2(V_LSHRREV_B32, 6, int(2), 0), // v6 = lane >> 2
        vop1(V_MOV_B32, 8, int(1)),
        vop2(V_LSHLREV_B32, 7, v(6), 8), // v7 = 1 << (lane >> 2)
        vop1(V_MOV_B32, 5, int(0)),
        vop1(V_MOV_B32, 13, int(0)),
        vopc(V_CMP_GT_U32, 0, 0), // v_cmp_gt_u32 vcc, s0, v0
        sop1(S_AND_SAVEEXEC_B64, 20, VCC_LO),
    ]);
    let skip = w.len();
    w.push(sopp(S_CBRANCH_EXECZ));
    w.extend(super::super::wave::WAVE_OR_V7_TO_S7);
    w.extend([
        0xBF06_8007, // s_cmp_eq_u32 s7, 0
        0xBF85_000E, // s_cbranch_scc1 past the loop
        0xBE88_0307, // s_mov_b32 s8, s7
        0xBF82_0001, // s_branch to the loop's head
        0xBE88_0309, // again: s_mov_b32 s8, s9
        0xBE8A_1308, // head: s_ff1_i32_b32 s10, s8
        0x8F0B_0A81, // s_lshl_b32 s11, 1, s10
        0x3608_0E0B, // v_and_b32 v4, s11, v7
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7DAA_0880, // v_cmpx_ne_u32 exec, 0, v4
        0xBF88_0001, // s_cbranch_execz past the body
        0x4A0A_0A0A, // v_add_nc_u32 v5, s10, v5
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x8909_080B, // s_xor_b32 s9, s11, s8
        0xBF06_0B08, // s_cmp_eq_u32 s8, s11
        0xBF84_FFF4, // s_cbranch_scc0 again
        vop1(V_MOV_B32, 13, 8), // after the loop: v13 = s8
    ]);
    let join = w.len();
    w.push(sop1(S_MOV_B64, EXEC, 20));
    for k in [4, 6, 8, 10] {
        w.push(sop1(S_MOV_B64, k, 20 + k));
    }
    w.push(vop2(V_LSHLREV_B32, 5, int(16), 5));
    w.push(vop2(V_OR_B32, 12, v(13), 5));
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[skip] |= rel(skip, join);
    let want = |s0: u32| -> [u32; 64] {
        let or = (0..64u32).filter(|&l| l < s0).fold(0u32, |m, l| m | 1 << (l >> 2));
        let last = if or == 0 { 0 } else { 1u32 << (31 - or.leading_zeros()) };
        std::array::from_fn(|l| if (l as u32) < s0 { last | (l as u32 >> 2) << 16 } else { 0 })
    };
    let _one = one_at_a_time();
    // the fault, with the check off: the lower host wave's own highest bit
    let what = "scalar own-bits loop, check off";
    if let Some(card) = super::super::own_bits::with_checks(false, true, || card_emitted(what, &w, |_| {})) {
        let got = card.run([64, 0, 0, 0]);
        let wrong = (0..32).filter(|&l| got[l] != want(64)[l]).count();
        eprintln!("{what}: {wrong} of the lower host wave's 32 lanes wrong (32 is the fault)");
        assert_eq!(wrong, 32, "the fault this test is for is not shown: {:#x} at lane 0", got[0]);
        assert_eq!(got[0] & 0xFFFF, 1 << 7, "the lower half's own highest bit");
    }
    let Some(card) = card_emitted("scalar own-bits loop", &w, |_| {}) else { return };
    for s0 in [64, 40, 1, 0] {
        let got = card.run([s0, 0, 0, 0]);
        compare(&format!("scalar own-bits loop, lanes below {s0}"), &got, &some(want(s0)));
    }
    eprintln!("scalar own-bits loop: the whole wave's last bit in every lane, as the model");
}

/// Inside `if (lane < s0)`, the lanes of the upper host wave first count to s2 (a loop whose
/// result feeds the increment, so it stays), then every lane takes the unsigned minimum of an
/// LDS word with 0 and adds 1 to it, both atomically; after the branch every lane stores the
/// word. The console runs the wave's lanes in step: all minimums, then all adds, the word is
/// the number of lanes below s0. Taken as writes only (`VICTORY_LDS_ATOMICS_READ=0`), the two
/// host waves meet only around the branch: the lower one gets through both atomics while the
/// upper one counts, and the upper one's minimum then clears its adds (avatarmanz/VIctory-7900XTX
/// PR #204, feanorsun).
#[test]
fn lds_atomics_in_a_lane_branch_keep_the_consoles_order() {
    let mut w = prologue();
    w.extend([
        vop1(V_MOV_B32, 9, int(0)), // the word's address, and the minimum
        vop1(V_MOV_B32, 17, int(2)),
        vop2(V_LSHRREV_B32, 11, int(5), 0), // v11 = lane >> 5
        vop2(V_MUL_U32_U24, 11, 2, 11),     // v11 = s2 in the upper host wave, 0 in the lower
        vopc(V_CMP_GT_U32, 0, 0),           // v_cmp_gt_u32 vcc, s0, v0
        sop1(S_AND_SAVEEXEC_B64, 12, VCC_LO),
    ]);
    let skip = w.len();
    w.push(sopp(S_CBRANCH_EXECZ));
    w.extend([vop1(V_MOV_B32, 16, int(0)), vop1(V_MOV_B32, 10, v(0)), sop1(S_MOV_B64, 14, EXEC)]);
    let top = w.len();
    w.push(vopc(V_CMP_LT_U32, v(16), 11)); // v_cmp_lt_u32 vcc, v16, v11
    w.push(sop2(S_AND_B64, EXEC, EXEC, VCC_LO));
    let leave = w.len();
    w.push(sopp(S_CBRANCH_EXECZ));
    for _ in 0..8 {
        w.extend([vop2(V_MUL_U32_U24, 10, LITERAL, 10), 0x0019_660D, vop2(V_ADD_NC_U32, 10, LITERAL, 10), 0x3C6E_F35F]);
    }
    w.push(vop2(V_ADD_NC_U32, 16, int(1), 16));
    let back = w.len();
    w.push(sopp(S_BRANCH));
    let end = w.len();
    w.push(sop1(S_MOV_B64, EXEC, 14));
    w.push(vopc(V_CMP_EQ_U32, 3, 10)); // v_cmp_eq_u32 vcc, s3, v10 (never: s3 = -1)
    w.push(vop2(V_CNDMASK_B32, 2, int(1), 17)); // v2 = 1 (2 if it were)
    w.extend(ds(0x07, 0, 9, 9)); // ds_min_u32 v9, v9
    w.extend(ds(0x00, 0, 9, 2)); // ds_add_u32 v9, v2
    let join = w.len();
    w.push(sop1(S_MOV_B64, EXEC, 12));
    w.extend(ds(0x36, 12, 9, 0)); // ds_read_b32 v12, v9
    w.push(S_WAITCNT_LGKM0);
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[skip] |= rel(skip, join);
    w[leave] |= rel(leave, end);
    w[back] |= rel(back, top);
    let lds = |n: &mut NativeProgram| {
        if let StageIo::Cs(cs) = &mut n.io {
            cs.rsrc2 |= 1 << 15; // 128 dwords of LDS
        }
    };
    let names: Vec<&str> = {
        let p = ir::build(&native_of(&w, true, 64).code, true, 64).expect("ir::build");
        p.blocks.iter().flat_map(|b| &b.insts).map(|i| i.name()).filter(|n| n.starts_with("ds_")).collect()
    };
    assert_eq!(names, ["ds_min_u32", "ds_add_u32", "ds_read_b32"]);
    // (under the translation's iteration limit for compute loops, 4,096: past it a lane is
    // taken out of the program)
    const COUNT: u32 = 4_000;
    let _one = one_at_a_time();
    // the race, with atomics taken as writes only
    let what = "LDS atomics in a lane branch, as writes only";
    let was = lower::TEST_LDS_ATOMICS_READ.replace(Some(false));
    let old = card_emitted(what, &w, lds);
    lower::TEST_LDS_ATOMICS_READ.set(was);
    if let Some(card) = old {
        for s0 in [64, 33] {
            let got = card.run([s0, 0, COUNT, u32::MAX]);
            eprintln!("{what}: lanes below {s0}: the word is {} (the console: {s0})", got[0]);
        }
    }
    let Some(card) = card_emitted("LDS atomics in a lane branch", &w, lds) else { return };
    for s0 in [64, 33, 1, 0] {
        let got = card.run([s0, 0, COUNT, u32::MAX]);
        compare(&format!("LDS atomics in a lane branch, lanes below {s0}"), &got, &some([s0; 64]));
    }
    eprintln!("LDS atomics in a lane branch: every minimum before every add, as the model");
}

/// A wave64 lane mask held as a number, its two dwords written apart (LLVM's two `s_mov_b32`
/// for a 64-bit constant it cannot inline; `vcc_lo` and `vcc_hi` set apart the same way), taken
/// as EXEC: `s_and_saveexec_b64 s[20:21], s[12:13]`, then `v_mov_b32 v12, 1` in the lanes it
/// holds, lanes 32-63 by the high dword's bits. Named by its low dword alone
/// (`VICTORY_MASK_PAIR=0`), lowering guessed the high dword as the register written beside the
/// low one: none for halves written apart (JMAN730/project-flamingo issue #806, pablogonz12;
/// Marvisak/VIctory PR #2836, kotekompotek), and a stale one where the high dword is rewritten
/// after a 64-bit write (the second form).
#[test]
fn a_mask_whose_halves_are_written_apart_tests_its_high_dword() {
    const S_MOV_B32: u32 = 0x03;
    const LO: u32 = 0x0F0F_00FF;
    const HI: u32 = 0xA5A5_5A5A;
    let under = |w: &mut Vec<u32>| {
        w.extend([
            sop1(S_AND_SAVEEXEC_B64, 20, 12),
            vop1(V_MOV_B32, 12, int(1)),
            sop1(S_MOV_B64, EXEC, 20),
        ]);
        epilogue(w);
    };
    let apart = {
        let mut w = prologue();
        w.extend([sop1(S_MOV_B32, 12, LITERAL), LO, sop1(S_MOV_B32, 13, LITERAL), HI]);
        under(&mut w);
        w
    };
    let rewritten = {
        let mut w = prologue();
        // s_mov_b64 s[12:13], s[0:1] (s0 = LO, s1 = 0), then s13 = HI
        w.extend([sop1(S_MOV_B64, 12, 0), sop1(S_MOV_B32, 13, LITERAL), HI]);
        under(&mut w);
        w
    };
    let bits = u64::from(HI) << 32 | u64::from(LO);
    let want: [u32; 64] = std::array::from_fn(|l| if bits >> l & 1 != 0 { 1 } else { DEAD });
    let _one = one_at_a_time();
    for (what, words) in [("a mask written in halves", &apart), ("a mask's high dword rewritten", &rewritten)] {
        // with the switch off: lanes 32-63 wrong where the dwords' bits differ
        let off = format!("{what}, named by its low dword");
        let was = crate::structure::TEST_MASK_PAIRS.replace(Some(false));
        let old = card_emitted(&off, words, |_| {});
        crate::structure::TEST_MASK_PAIRS.set(was);
        if let Some(card) = old {
            let got = card.run([LO, 0, 0, 0]);
            let wrong = (0..64).filter(|&l| got[l] != want[l]).count();
            eprintln!("{off}: {wrong} lanes wrong");
            assert!(wrong > 0, "{off}: the fault this test is for is not shown");
        }
        let Some(card) = card_emitted(what, words, |_| {}) else { return };
        let got = card.run([LO, 0, 0, 0]);
        compare(what, &got, &some(want));
        eprintln!("{what}: every lane its bit of the pair, as the model");
    }
}

/// A wave64 select on a lane mask held as a number, its halves written apart:
/// `v_cndmask_b32 v12, 0, 1, s[12:13]` gives each lane its bit of the pair, lanes 32-63 the
/// high dword's. The select alone asks for the console's lanes (avatarmanz/VIctory-7900XTX
/// PR #308, compaccount1-coder; `VICTORY_SELECT_LANES`): without them the upper host wave's
/// lanes took the bits of host lanes 0-31 (a debug build stops on `cross_lane`'s assertion).
#[test]
fn a_wave64_select_on_a_number_gives_each_lane_its_bit() {
    const S_MOV_B32: u32 = 0x03;
    const LO: u32 = 0x0F0F_00FF;
    const HI: u32 = 0xA5A5_5A5A;
    let mut w = prologue();
    w.extend([sop1(S_MOV_B32, 12, LITERAL), LO, sop1(S_MOV_B32, 13, LITERAL), HI]);
    // v_cndmask_b32 v12, 0, 1, s[12:13] (VOP3)
    w.extend([0xD400_0000 | 0x101 << 16 | 12, 12 << 18 | int(1) << 9 | int(0)]);
    epilogue(&mut w);
    let bits = u64::from(HI) << 32 | u64::from(LO);
    let want: [u32; 64] = std::array::from_fn(|l| (bits >> l & 1) as u32);
    let _one = one_at_a_time();
    let Some(card) = card_emitted("a select on a number", &w, |_| {}) else { return };
    compare("a wave64 select on a number", &card.run([0; 4]), &some(want));
    eprintln!("a wave64 select on a number: every lane its bit of the pair, as the model");
}
