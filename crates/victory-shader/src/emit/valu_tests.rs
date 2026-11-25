//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MUBUF `buffer_load_dword vdst, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_load_dword(vdst: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8,
    ]
}

/// MUBUF `buffer_atomic_<op> vdata, off, s[srsrc:srsrc+3], 0 [glc]`.
const fn buffer_atomic(op: u16, vdata: u8, srsrc: u8, glc: bool) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32 & 0x7F) << 18 | (glc as u32) << 14,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
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
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (mut ran, mut no_32_lanes) = (0, 0);
    for &(label, warp) in devices {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(made) => made,
            // A program that declares `WaveSize(32)` (one thread with LDS, say) on a device
            // without 32-lane waves (#356: 16 at most): refused, as it should be; not run.
            Err(e) if !host.runs_waves_of(32) => {
                let (min, max) = host.wave_lanes().unwrap_or_default();
                eprintln!("{label}: waves of {min} to {max} lanes, not 32; skipped ({e})");
                no_32_lanes += 1;
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
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
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    if ran == 0 && devices.iter().all(|&(_, warp)| !warp) {
        // A case for the hardware adapter alone, on a machine without one.
        eprintln!("no SM 6.6 hardware device; a hardware-only case not run");
        return;
    }
    assert!(ran + no_32_lanes > 0, "no SM 6.6 device available ({devices:?})");
}

/// Each device says which wave sizes it runs (what the skip above goes by): a power-of-two
/// range, which its own ends are in.
#[test]
fn devices_report_their_wave_sizes() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (min, max) = host.wave_lanes().expect("OPTIONS1");
        eprintln!("{label}: waves of {min} to {max} lanes");
        assert!(min.is_power_of_two() && max.is_power_of_two() && min <= max, "{min}..{max}");
        assert!(host.runs_waves_of(min) && host.runs_waves_of(max));
        assert!(!host.runs_waves_of(max * 2) && !host.runs_waves_of(min / 2));
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

/// The scalar bit searches and counts, ported from TuxBox (brcly, Apache-2.0,
/// `shader: restore s_flbit_i32_b64 and s_ff0_i32 lowering dropped by a merge`,
/// 1a50eb9): SFU-free, none of them writes SCC, and the 64-bit forms look at
/// the high word first.
#[test]
fn scalar_bit_searches_match_the_console() {
    // SOP1 `s_<op> s16, s0` (`src0` 0 is s[0:1] for a two-dword source).
    let sop1 = |op: u32, src0: u32| 0xBE80_0000 | 16 << 16 | op << 8 | src0;
    let read_back = |op: u32, src0: u32| [sop1(op, src0), vop1(0x001, 0, sgpr(16))];

    // s_flbit_i32_b64: 31 less the high word's first set bit from the top, else
    // 32 past the low word's, ~0 for a zero pair.
    let flbit = |[lo, hi, _, _]: [u32; 4]| {
        if hi != 0 {
            hi.leading_zeros()
        } else if lo != 0 {
            32 + lo.leading_zeros()
        } else {
            u32::MAX
        }
    };
    let cases = [
        [0, 1, 0, 0],
        [0, 0x8000_0000, 0, 0],
        [0x1234_5678, 0, 0, 0],
        [1, 0, 0, 0],
        [0x8000_0000, 0, 0, 0],
        [0, 0, 0, 0],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0],
    ];
    let shader = build_valu_cs(&read_back(0x16, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, flbit(inp), "{backend}: s_flbit_i32_b64{s:?}", s = inp);
    });

    // s_ff0_i32_b32: the first zero bit from the bottom, ~0 when there is none.
    let ff0_b32 = |[a, _, _, _]: [u32; 4]| {
        if a == u32::MAX {
            u32::MAX
        } else {
            (!a).trailing_zeros()
        }
    };
    let cases = [[0, 0, 0, 0], [1, 0, 0, 0], [0x8000_0001, 0, 0, 0], [0xFFFF_FFFF, 0, 0, 0], [0xFFFF_FFFE, 0, 0, 0]];
    let shader = build_valu_cs(&read_back(0x11, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, ff0_b32(inp), "{backend}: s_ff0_i32_b32{s:?}", s = inp);
    });

    // s_ff0_i32_b64: the same over the pair, low word first.
    let ff0_b64 = |[lo, hi, _, _]: [u32; 4]| {
        if !lo != 0 {
            (!lo).trailing_zeros()
        } else if !hi != 0 {
            32 + (!hi).trailing_zeros()
        } else {
            u32::MAX
        }
    };
    let cases = [
        [0, 0, 0, 0],
        [0xF000_0000, 0, 0, 0],
        [0xFFFF_FFFF, 0, 0, 0],
        [0xFFFF_FFFF, 0x7FFF_FFFF, 0, 0],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0],
    ];
    let shader = build_valu_cs(&read_back(0x12, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, ff0_b64(inp), "{backend}: s_ff0_i32_b64{s:?}", s = inp);
    });
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

#[test]
#[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
fn v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    let tie_even = 1.0f64 + f64::from(f32::EPSILON) / 2.0; // ties to 1.0
    let tie_odd = 1.0f64 + 1.5 * f64::from(f32::EPSILON); // ties up to 1 + 2 ulp
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::f64::consts::PI,
        1.0e-3,
        123_456_789.123,
        tie_even,
        tie_odd,
        1.0 + f64::from(f32::EPSILON) * 0.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_000_1, // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40, // a float denormal: flushed
        -1.0e-300,
        f64::from(f32::MIN_POSITIVE),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let x = values[i];
        if x.is_nan() {
            assert!(f32::from_bits(got).is_nan(), "{backend}: cvt({x}) = {got:#010x}, want NaN");
            return;
        }
        let want = cvt_f32_f64_reference(x);
        assert_eq!(got, want, "{backend}: cvt({x:e}) = {got:#010x}, want {want:#010x}");
    });
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

#[test]
fn v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1);
    // lane `read` of v1 into s12, then into v0 for the store.
    let selectors: [(u32, u32); 3] = [
        (0x7654_3210, 0xfedc_ba98), // identity within the row
        (0x89ab_cdef, 0x0123_4567), // reversed
        (0x1111_1111, 0xf0f0_f0f0), // repeats
    ];
    for (op, cross) in [(0x377u16, false), (0x378, true)] {
        for read in [0u8, 5, 15, 16, 22, 31] {
            let mut body = Vec::new();
            body.extend_from_slice(&vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            body.extend_from_slice(&vop3(0x360, 12, vgpr(1), 128 + u16::from(read), 0));
            body.push(vop1(0x001, 0, sgpr(12))); // v_mov_b32 v0, s12
            let shader = build_cs(&body, 32, 0);
            let inputs: Vec<[u32; 4]> = selectors.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
            // Cross-lane code declares WaveSize(32), which WARP's pipeline
            // creation refuses: hardware only.
            run_cases_on(&shader, &inputs, &[("hardware", false)], |backend, i, inp, got| {
                let want = permlane_source(u32::from(read), inp[0], inp[1], cross);
                assert_eq!(
                    got, want,
                    "{backend}: op {op:#x} selectors {:08x?}, lane {read}",
                    selectors[i]
                );
            });
        }
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }

    // f16: -0 = 0x8000, +0 = 0x0000. Low 16 bits hold the f16 result.
    let (pz16, nz16) = (0x0000_u32, 0x8000_u32);
    let cases16 = [
        [nz16, pz16, nz16, 0],
        [pz16, nz16, nz16, 0],
        [nz16, pz16, pz16, 0],
        [pz16, nz16, pz16, 0],
    ];
    let ops16 = [
        ("v_max_f16", 0x139, pz16),
        ("v_min_f16", 0x13A, nz16),
        ("v_max3_f16", 0x354, pz16),
        ("v_min3_f16", 0x351, nz16),
    ];
    for (name, op, want) in ops16 {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases16, |backend, _, inp, got| {
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The console's float mode reads f32 denormal operands of `v_min/max/min3/max3/med3_f32`
/// as zeros of their sign, and those orders put -0 below +0 (`v_med3_f32` too). The host's
/// own denormal mode varies (WARP and an RX 7700 flush, an RX 6700 XT does not), so both are
/// decided on the bits.
#[test]
fn min_max_and_med3_read_denormals_as_signed_zeros() {
    fn ftz(x: u32) -> u32 {
        if x & 0x7F80_0000 == 0 {
            x & 0x8000_0000
        } else {
            x
        }
    }
    /// `a < b` in the hardware's order, on flushed operands: -0 below +0.
    fn lt(a: u32, b: u32) -> bool {
        let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
        fa < fb || (fa == fb && a == 0x8000_0000 && b == 0)
    }
    fn min(a: u32, b: u32) -> u32 {
        if lt(b, a) { b } else { a }
    }
    fn max(a: u32, b: u32) -> u32 {
        if lt(a, b) { b } else { a }
    }
    let values = [
        0,           // +0
        0x8000_0000, // -0
        0x0000_0001, // +denormal
        0x8000_0001, // -denormal
        0x007F_FFFF, // largest +denormal
        0x0080_0000, // smallest +normal
        0xBF80_0000, // -1.0
    ];
    let mut inputs = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                inputs.push([a, b, c, 0]);
            }
        }
    }
    type Want = fn(u32, u32, u32) -> u32;
    let ops: [(&str, u16, Want); 5] = [
        ("v_min_f32", 0x10F, |a, b, _| min(a, b)),
        ("v_max_f32", 0x110, |a, b, _| max(a, b)),
        ("v_min3_f32", 0x151, |a, b, c| min(min(a, b), c)),
        ("v_max3_f32", 0x154, |a, b, c| max(max(a, b), c)),
        ("v_med3_f32", 0x157, |a, b, c| {
            max(min(a, b), min(max(a, b), c))
        }),
    ];
    // Per device and instruction: how many cases are wrong, and the first.
    let mut wrong = std::collections::BTreeMap::<String, (usize, String)>::new();
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &inputs, |backend, _, [a, b, c, _], got| {
            let w = want(ftz(a), ftz(b), ftz(c));
            if got != w {
                let first = format!(
                    "{name}({a:#010x}, {b:#010x}, {c:#010x}) = {got:#010x}, want {w:#010x}"
                );
                wrong
                    .entry(format!("{backend}: {name}"))
                    .or_insert((0, first))
                    .0 += 1;
            }
        });
    }
    let report: Vec<String> = wrong
        .iter()
        .map(|(k, (n, first))| format!("{k}: {n} wrong, first {first}"))
        .collect();
    assert!(wrong.is_empty(), "{}", report.join("\n"));
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// `v_cmp_f_*` is false and `v_cmp_tru_*` true whatever the operands, NaNs
/// included (RDNA2 ISA). As `fcmp false`/`fcmp true` WARP removed its device
/// at the first dispatch.
#[test]
fn float_compares_that_are_always_false_or_true() {
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;
    let f32_inputs = [
        [f(1.0), f(2.0), 0, 0],
        [f(1.0), f(1.0), 0, 0],
        [nan, f(1.0), 0, 0],
        [nan, nan, 0, 0],
    ];
    let f16_inputs = [
        [0x3C00, 0x4000, 0, 0],
        [0x3C00, 0x3C00, 0, 0],
        [0x7E00, 0x3C00, 0, 0],
        [0x7E00, 0x7E00, 0, 0],
    ];
    // v_cmp_*_e64 s20, v1, v2; v_cndmask_b32_e64 v0, 0, 1, s20
    for (op, what, inputs, want) in [
        (0x000u16, "v_cmp_f_f32", f32_inputs, 0),
        (0x00f, "v_cmp_tru_f32", f32_inputs, 1),
        (0x0c8, "v_cmp_f_f16", f16_inputs, 0),
        (0x0ef, "v_cmp_tru_f16", f16_inputs, 1),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// 64-bit integer compares order by the high dword (signed for i64), then
/// by the low one. AMD's D3D12 compiler runs an ordered i64 `icmp` as if the
/// dwords were swapped (RX 9070 and gfx1036 alike), so the lowering compares
/// the dwords; the values are picked so that the two orders disagree.
#[test]
fn v_cmp_u64_and_i64_order_by_the_high_dword() {
    type Order = fn(u64, u64) -> bool;
    let ops: [(u16, &str, Order); 12] = [
        (0x0a1, "v_cmp_lt_i64", |a, b| (a as i64) < (b as i64)),
        (0x0a2, "v_cmp_eq_i64", |a, b| a == b),
        (0x0a3, "v_cmp_le_i64", |a, b| (a as i64) <= (b as i64)),
        (0x0a4, "v_cmp_gt_i64", |a, b| (a as i64) > (b as i64)),
        (0x0a5, "v_cmp_ne_i64", |a, b| a != b),
        (0x0a6, "v_cmp_ge_i64", |a, b| (a as i64) >= (b as i64)),
        (0x0e1, "v_cmp_lt_u64", |a, b| a < b),
        (0x0e2, "v_cmp_eq_u64", |a, b| a == b),
        (0x0e3, "v_cmp_le_u64", |a, b| a <= b),
        (0x0e4, "v_cmp_gt_u64", |a, b| a > b),
        (0x0e5, "v_cmp_ne_u64", |a, b| a != b),
        (0x0e6, "v_cmp_ge_u64", |a, b| a >= b),
    ];
    let values: [u64; 10] = [
        0,
        1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x1_FFFF_FFFF,
        0x0000_0040_C010_FBFF,
        0x0001_8000_0000_0040,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0001,
        0xFFFF_FFFF_0000_0000,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .flat_map(|&a| values.iter().map(move |&b| (a, b)))
        .map(|(a, b)| [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32])
        .collect();
    for (op, name, order) in ops {
        // op s20, v[1:2], v[3:4] and, sharing v2, op s20, v[1:2], v[2:3];
        // v_cndmask_b32_e64 v0, 0, 1, s20
        for second in [3, 2] {
            let mut body = vop3(op, 20, vgpr(1), vgpr(second), 0x80).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
            let shader = build_valu_cs(&body);
            let at = usize::from(second) - 1;
            run_cases(&shader, &inputs, |backend, _, inp, got| {
                let a = u64::from(inp[0]) | u64::from(inp[1]) << 32;
                let b = u64::from(inp[at]) | u64::from(inp[at + 1]) << 32;
                let want = u32::from(order(a, b));
                assert_eq!(got, want, "{backend}: {name}({a:#018x}, {b:#018x})");
            });
        }
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// An f32 result: exactly these bits, any NaN, or within the 1 ULP the
/// RDNA2 ISA gives the transcendental ops (same sign, finite).
#[derive(Clone, Copy, Debug)]
enum F32Want {
    Bits(u32),
    Nan,
    Ulp1(f32),
}

impl F32Want {
    fn holds(self, got: u32) -> bool {
        match self {
            Self::Bits(b) => got == b,
            Self::Nan => f32::from_bits(got).is_nan(),
            Self::Ulp1(v) => {
                let v = v.to_bits();
                f32::from_bits(got).is_finite() && (got ^ v) >> 31 == 0 && got.abs_diff(v) <= 1
            }
        }
    }
}

#[test]
fn rcp_rsq_sqrt_log_exp_f32_special_values_match_rdna2() {
    use F32Want::{Bits, Nan, Ulp1};
    let f = f32::to_bits;
    let (pinf, ninf, nz, nan) = (0x7F80_0000, 0xFF80_0000, 0x8000_0000, 0x7FC0_0000);
    let rcp: &[(u32, F32Want)] = &[
        (0, Bits(pinf)),
        (nz, Bits(ninf)),
        (pinf, Bits(0)),
        (ninf, Bits(nz)),
        (nan, Nan),
        (f(2.0), Ulp1(0.5)),
        (f(-4.0), Ulp1(-0.25)),
        (f(3.0), Ulp1(1.0 / 3.0)),
        (f(1e-30), Ulp1((1.0 / f64::from(1e-30f32)) as f32)),
    ];
    // (VOP1 opcode, name, cases); v_log_f32 is log2, v_exp_f32 is 2^x.
    type Cases<'a> = &'a [(u32, F32Want)];
    let ops: &[(u16, &str, Cases)] = &[
        (0x02a, "v_rcp_f32", rcp),
        (0x02b, "v_rcp_iflag_f32", rcp),
        (
            0x02e,
            "v_rsq_f32",
            &[
                (0, Bits(pinf)),
                (nz, Bits(ninf)),
                (pinf, Bits(0)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (nan, Nan),
                (f(4.0), Ulp1(0.5)),
                (f(0.25), Ulp1(2.0)),
                (f(2.0), Ulp1(std::f32::consts::FRAC_1_SQRT_2)),
            ],
        ),
        (
            0x033,
            "v_sqrt_f32",
            &[
                (0, Bits(0)),
                (nz, Bits(nz)),
                (pinf, Bits(pinf)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (f(-1e-30), Nan),
                (nan, Nan),
                (f(4.0), Ulp1(2.0)),
                (f(2.0), Ulp1(std::f32::consts::SQRT_2)),
                (f(1e30), Ulp1(f64::from(1e30f32).sqrt() as f32)),
            ],
        ),
        (
            0x027,
            "v_log_f32",
            &[
                (0, Bits(ninf)),
                (nz, Bits(ninf)),
                (pinf, Bits(pinf)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (nan, Nan),
                (f(1.0), Ulp1(0.0)),
                (f(8.0), Ulp1(3.0)),
                (f(0.125), Ulp1(-3.0)),
                (f(3.0), Ulp1(3f64.log2() as f32)),
            ],
        ),
        (
            0x025,
            "v_exp_f32",
            &[
                (ninf, Bits(0)),
                (pinf, Bits(pinf)),
                (nan, Nan),
                (0, Ulp1(1.0)),
                (nz, Ulp1(1.0)),
                (f(3.0), Ulp1(8.0)),
                (f(-3.0), Ulp1(0.125)),
                (f(0.5), Ulp1(std::f32::consts::SQRT_2)),
                (f(128.0), Bits(pinf)),
                (f(-200.0), Bits(0)),
            ],
        ),
    ];
    for &(op, name, cases) in ops {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert!(
                want.holds(got),
                "{backend}: {name}({:#010x} = {:?}) = {got:#010x} ({:?}), want {want:?}",
                inp[0],
                f32::from_bits(inp[0]),
                f32::from_bits(got),
            );
        });
    }
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_i32_and_u32_f32_truncate_saturate_and_take_nan_as_zero() {
    // RDNA2: toward zero, out-of-range values (infinities too) saturate, NaN
    // gives 0 – what Rust's `as` does.
    let f = f32::to_bits;
    let inputs = [
        0,
        0x8000_0000, // -0
        f(1.5),
        f(-1.5),
        f(-0.75),
        f(0.999_999_94),
        f(16_777_215.0),
        f(2_147_483_520.0), // largest f32 below 2^31
        f(2_147_483_648.0), // 2^31
        f(-2_147_483_648.0),
        f(-2_147_483_904.0), // next f32 below -2^31
        f(4_294_967_040.0),  // largest f32 below 2^32
        f(4_294_967_296.0),  // 2^32
        f(1e30),
        f(-1e30),
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0000, // NaN
        0xFFC0_0000, // -NaN
        0x7F80_0001, // signalling NaN
        0x0000_0001, // smallest denormal
    ];
    let cases: Vec<[u32; 4]> = inputs.iter().map(|&x| [x, 0, 0, 0]).collect();
    for (op, name) in [(0x008, "v_cvt_i32_f32"), (0x007, "v_cvt_u32_f32")] {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let x = f32::from_bits(inp[0]);
            let want = if op == 0x008 {
                x as i32 as u32
            } else {
                x as u32
            };
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x} = {x:?}) = {got:#010x}, want {want:#010x}",
                inp[0],
            );
        });
    }
}

#[test]
fn v_cvt_f32_i32_u32_and_ubyte_round_to_nearest_even() {
    // Integers past 2^24 round to nearest even (the MODE default), as Rust's
    // `as` does; the ubyte forms convert one byte of the source.
    let inputs = [
        0,
        1,
        0xFF,
        0x0100_0001, // 2^24 + 1: tie, rounds down to even
        0x0100_0003, // 2^24 + 3: tie, rounds up to even
        0x0100_0005,
        0x7FFF_FFBF, // just below the halfway point to 2^31
        0x7FFF_FFC0, // halfway, rounds to 2^31
        0x7FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FF7F,
        0xFFFF_FF80, // u32: halfway, rounds to 2^32
        0xFFFF_FFFF,
        0x80FF_7F01,
        0x12AB_00FE,
    ];
    let ops = [
        (0x005, "v_cvt_f32_i32"),
        (0x006, "v_cvt_f32_u32"),
        (0x011, "v_cvt_f32_ubyte0"),
        (0x012, "v_cvt_f32_ubyte1"),
        (0x013, "v_cvt_f32_ubyte2"),
        (0x014, "v_cvt_f32_ubyte3"),
    ];
    let cases: Vec<[u32; 4]> = inputs.iter().map(|&x| [x, 0, 0, 0]).collect();
    for (op, name) in ops {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let x = inp[0];
            let want = match op {
                0x005 => x as i32 as f32,
                0x006 => x as f32,
                _ => f32::from((x >> (8 * (op - 0x011))) as u8),
            }
            .to_bits();
            // WARP's uitofp is not correctly rounded above 2^31 (it gave
            // 2^32 for 0xffffff7f); the hardware result must be exact.
            if backend == "warp" && op == 0x006 && x >= 1 << 31 && got != want {
                assert!(
                    got.abs_diff(want) <= 1,
                    "warp: {name}({x:#010x}) = {got:#010x}"
                );
                eprintln!("warp: {name}({x:#010x}) = {got:#010x}, RDNA2 gives {want:#010x}");
                return;
            }
            assert_eq!(
                got,
                want,
                "{backend}: {name}({:#010x}) = {got:#010x} ({:?}), want {want:#010x} ({:?})",
                inp[0],
                f32::from_bits(got),
                f32::from_bits(want),
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27); v_mov_b32 v0, s6
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 width is 7-bit (src1[22:16]), so width=32 keeps all remaining bits without sign extension.
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    let su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // +qNaN -> 0
        (0xFE00, 0),      // -qNaN -> 0
        (0x7C01, 0),      // +sNaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),           // -inf -> -32768
        (0xF8E2, 0x8000),           // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),                // +qNaN -> 0
        (0xFE00, 0),                // -qNaN -> 0
        (0x7C01, 0),                // +sNaN -> 0
        (0x57BC, 123),              // 123.75 -> 123
        (0x78E2, 0x7FFF),           // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),           // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// DS `op vdst, addr, data0, data1` at offset 0.
const fn ds2(op: u8, vdst: u8, addr: u8, data0: u8, data1: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data1 as u32) << 16 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_CMPST_F32: u8 = 0x11;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_ADD_F32: u8 = 0x15;
const DS_CMPST_RTN_F32: u8 = 0x31;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;
const DS_ADD_RTN_F32: u8 = 0x55;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    // Hardware only (see below): `VICTORY_WARP=1` makes the hardware device WARP too.
    if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") {
        eprintln!("skipped on WARP: it runs no fixed wave size of 32 for one thread");
        return;
    }
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS has no peers and needs no WaveSize(32), so it
            // runs on both hardware (including 64-only GCN GPUs) and WARP.
            let shader = build_cs(&body, 1, 512);
            run_cases(
                &shader,
                &cases,
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    let plain = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1 << 8;
    w1 |= 1 << 29;
    let neg_abs = build_valu_cs(&[w0, w1]);

    let values: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        f64::from_bits(0x3FF0_0000_1000_0000), // 1.0 + 0.5 ulp(f32) -> 1.0
        f64::from_bits(0x3FF0_0000_3000_0000), // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        f64::from_bits(0x3FF0_0000_1000_0001), // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from_bits(0x380F_FFFF_F000_0000), // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        f64::from_bits(0x3690_0000_0000_0000),
        f64::from_bits(0x3690_0000_0000_0001),
        f64::from_bits(0xB690_0000_0000_0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000), // +qNaN
        f64::from_bits(0x7FF0_0000_0000_0001), // +sNaN -> quieted
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&plain, &inputs, |backend, i, _, got| {
        let x = values[i];
        let want = (x as f32).to_bits();
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan() && (got & 0x0040_0000) != 0,
                "{backend}: v_cvt_f32_f64({:#018x}) = {got:#010x}, expected qNaN",
                x.to_bits(),
            );
        } else {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_f32_f64({x} / {:#018x}) = {got:#010x}, want {want:#010x}",
                x.to_bits(),
            );
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    let mod_values: &[f64] = &[2.5, -2.5, 0.0, f64::INFINITY];
    let mod_in: Vec<[u32; 4]> = mod_values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&neg_abs, &mod_in, |backend, i, _, got| {
        let x = mod_values[i];
        let want = (-(x.abs()) as f32).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_f32_f64(-|{x}|) = {got:#010x}, want {want:#010x}",
        );
    });
}

#[test]
fn whole_wave_f64_sum_and_cvt_f32_f64() {
    let s_or_saveexec_b32 =
        |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x3D << 8 | src0 as u32;
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc as u32 | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };

    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Per-lane f64 input in v[6:7] = (s0, s1).
    words.push(vop1(0x001, 6, sgpr(0)));
    words.push(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push(s_or_saveexec_b32(12, 0xC1));
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
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
    words.push(s_mov_b32(16, 14));
    words.push(s_mov_b32(17, 15));
    words.push(s_mov_b32(126, 12)); // restore exec_lo
    words.push(vop1(0x00f, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
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
            rsrc2: 5 << 1,
            threads: [4, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
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
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let cases: &[(f64, f32)] = &[(1.25, 5.0), (-0.75, -3.0)];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize(32) on SM 6.6.
            Err(e) if warp => {
                eprintln!("{label}: compute_pipeline skipped ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for &(x, want) in cases {
            let b = x.to_bits();
            let inputs = [b as u32, (b >> 32) as u32, 0, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
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
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0],
                want.to_bits(),
                "{label}: wave f64 sum({x}) = {:#010x}, want {:#010x}",
                out[0][0],
                want.to_bits(),
            );
        }
    }
}

/// A one-thread program that stores 7 through the raw UAV whose descriptor
/// it loads at heap index `s0` from the address in `s4`, which is the
/// resource heap itself (`ResourceHeap`: bindless, no used-resource entry)
/// or a table; lowered with the view heap limit and heap map `heap` gives
/// this thread (`TEST_VIEW_HEAP`).
fn lower_store_cs(root: RootType, heap: (u32, Option<(u32, u32)>)) -> HostShader {
    let words = [
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        // s_load_dwordx4 s[8:11], s[4:5], s6
        s_load_dwordx4(8, 4)[0],
        6 << 25,
        vop1(0x001, 0, 0x87), // v_mov_b32 v0, 7
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: if root == RootType::Table {
            vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }]
        } else {
            Vec::new()
        },
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    super::shader::TEST_VIEW_HEAP.set(Some(heap));
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
    super::shader::TEST_VIEW_HEAP.set(None);
    let lowered = lowered.expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Under a heap map, a view taken from the resource heap itself is looked
/// up (the translation changes with the window and the map's slot, which
/// the shader cache key holds), and nothing else changes: no user data,
/// root views or root parameters of its own, so the host's root signatures
/// stay as they were. A table's index is the host's already: a table is
/// translated the same with and without a map.
#[test]
fn only_views_from_the_resource_heap_are_looked_up_in_the_heap_map() {
    const LIMIT: u32 = 999_999;
    let bindless = |map| lower_store_cs(RootType::ResourceHeap, (LIMIT, map));
    let plain = bindless(None);
    assert_eq!(plain.dxil, bindless(None).dxil, "translations are repeatable");
    let mapped = bindless(Some((131_072, 737_854)));
    assert_ne!(plain.dxil, mapped.dxil, "a bindless view is looked up");
    assert_ne!(mapped.dxil, bindless(Some((131_072, 737_853))).dxil);
    assert_ne!(mapped.dxil, bindless(Some((65_536, 737_854))).dxil);
    assert_eq!(plain.user_data, mapped.user_data);
    assert_eq!(
        (&plain.root_cbvs, &plain.root_srvs, &plain.root_uavs),
        (&mapped.root_cbvs, &mapped.root_srvs, &mapped.root_uavs)
    );
    let table = |map| lower_store_cs(RootType::Table, (LIMIT, map));
    assert_eq!(table(None).dxil, table(Some((131_072, 737_854))).dxil);
}

/// On the hardware GPU and WARP, a bindless store at a title index below
/// the window lands at that index, one at a title index the heap map maps
/// lands at the host slot it names, and one the map leaves at 0 (never
/// written, or past the map's end) lands at the null slot the limit names
/// (here a raw UAV of its own, to see the stores it takes).
#[test]
fn a_heap_index_past_the_window_reads_the_slot_the_heap_map_gives() {
    // Heap: UAVs at 0 and 1, the map at 2, the limit's slot at 3. Window
    // 1: title index 0 is host slot 0; the map's entries are for title
    // indices 1, 2, 3: none, host slot 1, none.
    let shader = lower_store_cs(RootType::ResourceHeap, (3, Some((1, 2))));
    let map = [0, !1u32, 0];
    let cases: [(u32, [u32; 3]); 7] = [
        (0, [7, 0, 0]),           // below the window
        (2, [0, 7, 0]),           // mapped to host slot 1
        (1, [0, 0, 7]),           // an entry of 0: the limit's slot
        (3, [0, 0, 7]),           // likewise
        (4, [0, 0, 7]),           // past the map's end
        (1 << 25, [0, 0, 7]),     // far past it
        (0x07FF_FFFF, [0, 0, 7]), // the last index an address gives
    ];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (title, want) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.dword) {
                    (RootType::Constants, 0) => title,
                    (RootType::Constants, _) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[
                            View::RawUav(1),
                            View::RawUav(1),
                            View::Buffer {
                                data: &map,
                                stride: super::host::Stride::Raw,
                            },
                            View::RawUav(1),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got = [out[0][0], out[1][0], out[2][0]];
            assert_eq!(got, want, "{label}: title index {title:#x}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn wave_prefix_f32_uses_signed_flag_and_validates() {
    // 32-thread wave32 inclusive float prefix sum over v5 = float(v0):
    // lowers via `dx.op.wavePrefixOp.f32(val, 0 /*Sum*/, 0 /*signed*/)`.
    // Passing `unsigned = 1` on f32 is rejected by `dxil.dll`.
    let dpp16 = |src0: u8, ctrl: u16| -> u32 {
        0xF << 28 | 0xF << 24 | (ctrl as u32 & 0x1FF) << 8 | src0 as u32
    };
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s5 = 0
    words.extend_from_slice(&s_load_dwordx4(8, 4)); // s[8:11] = UAV
    words.push(vop1(0x006, 5, vgpr(0))); // v_cvt_f32_u32 v5, v0
    // Whole-wave inclusive f32 prefix sum of v5 into v1:
    words.push(0xBE80_0000 | 6 << 16 | 0x3D << 8 | 193); // s_or_saveexec_b32 s6, -1
    words.extend_from_slice(&vop3(0x101, 1, 0x80, vgpr(5), sgpr(6))); // v_cndmask_b32 v1, 0.0, v5, s6
    for shift in [1u16, 2, 4, 8] {
        words.push(vop2(0x03, 1, 0xFA, 1)); // v_add_f32_dpp v1, v1, v1 row_shr:shift
        words.push(dpp16(1, 0x110 + shift));
    }
    words.extend_from_slice(&vop3(0x360, 7, vgpr(1), 128 + 15, 0)); // v_readlane_b32 s7, v1, 15
    words.extend_from_slice(&[0xBEFE_03FF, 0xFFFF_0000]); // s_mov_b32 exec_lo, 0xFFFF0000
    words.push(vop2(0x03, 1, sgpr(7), 1)); // v_add_f32 v1, s7, v1
    words.push(0xBEFE_0306); // s_mov_b32 exec_lo, s6
    // Store each lane's prefix sum to UAV[v0].
    words.push(0x340E_0082); // v_lshlrev_b32 v7, 2, v0
    words.extend_from_slice(&[0xE070_1000, 0x8002_0107]); // buffer_store_dword v1, v7, s[8:11], 0 offen
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .unwrap();
    assert!(l.wave_ops, "expected wave prefix lowering");
    let mut dxil = l.dxil;
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut dxil).unwrap();
    }
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: l.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: Vec::new(),
    };
    #[cfg(windows)]
    {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(host) = Host::new(false)
            && let Ok((root, pso)) = host.compute_pipeline(&shader)
        {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0; 5]))],
                        views: &[View::RawUav(32)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            for lane in 0..32u32 {
                let got = f32::from_bits(out[0][lane as usize]);
                let want = (lane * (lane + 1) / 2) as f32;
                assert_eq!(got, want, "lane {lane}: got {got}, want {want}");
            }
        }
    }
}

/// EXP `exp target, v0, v1, v2, v3 [done] [vm]` (`en`: 4-bit channel mask).
const fn exp_inst(target: u8, en: u8, v: [u8; 4], done: bool, vm: bool) -> [u32; 2] {
    let w0 = 0x3E << 26
        | (vm as u32) << 12
        | (done as u32) << 11
        | (target as u32 & 0x3F) << 4
        | (en as u32 & 0xF);
    let w1 = (v[3] as u32) << 24 | (v[2] as u32) << 16 | (v[1] as u32) << 8 | (v[0] as u32);
    [w0, w1]
}

#[test]
fn vs_rt_and_viewport_index_exports_validate_and_create_pso() {
    use super::container::SigElement;
    use crate::dxbc::{Container, FourCC};

    let build_stage = |stage: HwStage,
                       words: &[u32],
                       layout: Vec<RootParam>,
                       stage_io: StageIo,
                       outputs: Vec<SigElement>|
     -> HostShader {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64: false,
            vgprs: 8,
            layout,
            used: Vec::new(),
            io: stage_io,
            code,
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res =
            resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        let shader = HostShader {
            stage,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: res.root_indexed_views,
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        shader
    };

    // Minimal pixel shader writing SV_Target0 = (1, 0, 0, 1).
    let ps_words = [
        vop1(0x001, 0, 0xF2), // v0 = 1.0
        vop1(0x001, 1, 0x80), // v1 = 0.0
        vop1(0x001, 2, 0x80), // v2 = 0.0
        vop1(0x001, 3, 0xF2), // v3 = 1.0
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[0],
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[1],
        S_ENDPGM,
    ];
    let ps = build_stage(
        HwStage::Ps,
        &ps_words,
        Vec::new(),
        StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }],
    );

    // Three VS variants: RT index only, viewport index only, and both + clip distance in pos2.
    let configs = [
        (true, false, false),
        (false, true, false),
        (true, true, true),
    ];
    let mut vs_shaders = Vec::new();
    for (has_rt, has_vp, has_clip) in configs {
        let mut words = vec![
            vop1(0x001, 0, 0x80),    // pos.x = 0.0
            vop1(0x001, 1, 0x80),    // pos.y = 0.0
            vop1(0x001, 2, 0x80),    // pos.z = 0.0
            vop1(0x001, 3, 0xF2),    // pos.w = 1.0
            vop1(0x001, 4, sgpr(0)), // pos1.z = packed (viewport << 16) | rt
            vop1(0x001, 5, sgpr(1)), // pos2.x = clip distance
        ];
        words.extend_from_slice(&exp_inst(12, 0xF, [0, 1, 2, 3], false, false)); // pos0
        words.extend_from_slice(&exp_inst(13, 0x4, [0, 0, 4, 0], !has_clip, false)); // pos1.z
        if has_clip {
            words.extend_from_slice(&exp_inst(14, 0x1, [5, 0, 0, 0], true, false)); // pos2.x
        }
        words.push(S_ENDPGM);

        let mut outputs = vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        let mut next_row = 1u32;
        if has_rt {
            outputs.push(SigElement {
                name: "SV_RenderTargetArrayIndex".into(),
                index: 0,
                kind: 4,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_vp {
            outputs.push(SigElement {
                name: "SV_ViewportArrayIndex".into(),
                index: 0,
                kind: 5,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_clip {
            outputs.push(SigElement {
                name: "SV_ClipDistance".into(),
                index: 0,
                kind: 6,
                comp: 9,
                interp: 2,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }

        let vs = build_stage(
            HwStage::Vs,
            &words,
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            outputs,
        );
        let c = Container::parse(&vs.dxil).expect("parse emitted VS DXIL");
        let sfi0 = u64::from_le_bytes(
            c.part(FourCC::SFI0).expect("SFI0").data[..8]
                .try_into()
                .unwrap(),
        );
        assert_ne!(
            sfi0 & 0x2000,
            0,
            "SFI0 should set ViewPortAndRTArrayIndexFromAnyShaderFeedingRasterizer"
        );
        vs_shaders.push(vs);
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (i, vs) in vs_shaders.iter().enumerate() {
            host.pipeline(vs, &ps)
                .unwrap_or_else(|e| panic!("{label}: graphics pipeline case {i} failed: {e}"));
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx8(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// MIMG `image_store vdata, vaddr, s[srsrc:srsrc+7] dmask:dmask dim:2D unorm [d16]`.
const fn image_store_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, d16: bool) -> [u32; 2] {
    let w0 = 0x3C << 26 | 0x08 << 18 | 1 << 12 | (dmask as u32 & 0xF) << 8 | 1 << 3;
    let w1 = (d16 as u32) << 31
        | ((srsrc as u32 >> 2) & 0x1F) << 16
        | (vdata as u32) << 8
        | (vaddr as u32);
    [w0, w1]
}

fn try_build_image_store_cs(dmask: u8, d16: bool) -> Result<HostShader, crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx8(8, 4));
    // Data in v1..v4 from root constants s0..s3; 2D texel coordinate (0, 0) in v5..v6.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.push(vop1(0x001, 5, 0x80)); // v5 = 0
    words.push(vop1(0x001, 6, 0x80)); // v6 = 0
    words.extend_from_slice(&image_store_2d(1, 5, 8, dmask, d16));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
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
    })?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

#[test]
fn image_store_partial_dmask_writes_expected_channels() {
    // (dmask, d16, input root constants, expected RGBA texel)
    let cases: &[(u8, bool, [u32; 4], [f32; 4])] = &[
        (0x0, false, [1.5f32.to_bits(), 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
        (0x1, false, [1.5f32.to_bits(), 0, 0, 0], [1.5, 0.0, 0.0, 0.0]),
        (
            0x3,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            [1.5, -2.25, 0.0, 0.0],
        ),
        (
            0x7,
            false,
            [
                1.5f32.to_bits(),
                (-2.25f32).to_bits(),
                4.0f32.to_bits(),
                0,
            ],
            [1.5, -2.25, 4.0, 0.0],
        ),
        // Sparse masks: RDNA2 packs enabled channels consecutively and zeros disabled ones.
        (
            0x5,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            [1.5, 0.0, -2.25, 0.0],
        ),
        (
            0x9,
            false,
            [1.5f32.to_bits(), 4.0f32.to_bits(), 0, 0],
            [1.5, 0.0, 0.0, 4.0],
        ),
        // d16: 1.5 (0x3E00) in low half, -2.0 (0xC000) in high half, 4.0 (0x4400) in next VGPR.
        (0x1, true, [0x0000_3E00, 0, 0, 0], [1.5, 0.0, 0.0, 0.0]),
        (0x3, true, [0xC000_3E00, 0, 0, 0], [1.5, -2.0, 0.0, 0.0]),
        (
            0x7,
            true,
            [0xC000_3E00, 0x0000_4400, 0, 0],
            [1.5, -2.0, 4.0, 0.0],
        ),
        (
            0xD,
            true,
            [0xC000_3E00, 0x0000_4400, 0, 0],
            [1.5, 0.0, -2.0, 4.0],
        ),
    ];

    let shaders: Vec<HostShader> = cases
        .iter()
        .map(|&(dmask, d16, ..)| {
            try_build_image_store_cs(dmask, d16)
                .unwrap_or_else(|e| panic!("dmask={dmask:#x} d16={d16}: {e}"))
        })
        .collect();

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (&(dmask, d16, inputs, want), shader) in cases.iter().zip(&shaders) {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: dmask={dmask:#x} d16={d16}: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
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
                        views: &[View::TextureUav(1, 1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got: [f32; 4] = [
                f32::from_bits(out[0][0]),
                f32::from_bits(out[0][1]),
                f32::from_bits(out[0][2]),
                f32::from_bits(out[0][3]),
            ];
            assert_eq!(
                got, want,
                "{label}: image_store dmask={dmask:#x} d16={d16} wrote {got:?}, want {want:?}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

#[test]
fn float_compares_that_are_always_false_or_true() {
    let cmp_shader = |op: u16| {
        let cmp = vop3(op, 6, vgpr(1), vgpr(2), 0);
        let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        build_valu_cs(&[cmp[0], cmp[1], sel[0], sel[1]])
    };
    let cmpx_shader = |op: u16| {
        let vopc = 0x7C00_0000 | (op as u32) << 17 | 2 << 9 | vgpr(1) as u32;
        build_valu_cs(&[
            s_mov_b32(6, 0x7E),   // s_mov_b32 s6, exec_lo
            vop1(0x001, 0, 0x80), // v_mov_b32 v0, 0
            vopc,                 // v_cmpx_* exec_lo, v1, v2
            vop1(0x001, 0, 0x81), // v_mov_b32 v0, 1 (guarded by narrowed EXEC)
            s_mov_b32(0x7E, 6),   // s_mov_b32 exec_lo, s6
        ])
    };

    let f32_inputs: &[[u32; 4]] = &[
        [0.0f32.to_bits(), 0.0f32.to_bits(), 0, 0],
        [(-0.0f32).to_bits(), 1.0f32.to_bits(), 0, 0],
        [(-2.5f32).to_bits(), 3.75f32.to_bits(), 0, 0],
        [f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 0, 0],
        [0x7FC0_0000, 1.0f32.to_bits(), 0, 0], // qNaN
        [0x7F80_0001, 0xFF80_0001, 0, 0],      // +sNaN, -sNaN
    ];
    let f16_inputs: &[[u32; 4]] = &[
        [0x0000, 0x8000, 0, 0], // +0.0, -0.0
        [0x3C00, 0xBC00, 0, 0], // +1.0, -1.0
        [0x7C00, 0xFC00, 0, 0], // +inf, -inf
        [0x7E00, 0x3C00, 0, 0], // qNaN, +1.0
        [0x7C01, 0xFC01, 0, 0], // +sNaN, -sNaN
    ];

    for (name, op, want, inputs) in [
        ("v_cmp_f_f32", 0x000, 0u32, f32_inputs),
        ("v_cmp_tru_f32", 0x00f, 1u32, f32_inputs),
        ("v_cmp_f_f16", 0x0c8, 0u32, f16_inputs),
        ("v_cmp_tru_f16", 0x0ef, 1u32, f16_inputs),
    ] {
        let cs = cmp_shader(op);
        run_cases(&cs, inputs, |backend, _, inp, got| {
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    for (name, op, want, inputs) in [
        ("v_cmpx_f_f32", 0x010, 0u32, f32_inputs),
        ("v_cmpx_tru_f32", 0x01f, 1u32, f32_inputs),
        ("v_cmpx_f_f16", 0x0d8, 0u32, f16_inputs),
        ("v_cmpx_tru_f16", 0x0ff, 1u32, f16_inputs),
    ] {
        let cs = cmpx_shader(op);
        run_cases(&cs, inputs, |backend, _, inp, got| {
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// An f64 rounded to the nearest-even half (65520 and up to infinity).
fn f16_rne(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a == 0.0 {
        return sign;
    }
    let e = (((a.to_bits() >> 52) & 0x7FF) as i32 - 1023).max(-14);
    let r = (a / 2f64.powi(e - 10)).round_ties_even() as i32;
    sign | (((e + 15) << 10) + r - 1024) as u16
}

#[test]
fn f16_nearest_switch_rounds_every_narrowing() {
    // With `VICTORY_F16_NEAREST=1` every f32-to-f16 narrowing rounds to nearest even,
    // not only the VOP1/VOP2/VOP3 16-bit results: v_cvt_f16_f32, the packed v_pk_*_f16
    // and v_fma_mixlo/mixhi_f16 too. The host's conversion rounds toward zero (NVIDIA,
    // AMD and WARP), so each case below is one where the two differ.
    fn h(x: u16) -> f64 {
        f64::from(f16_to_f32(x))
    }
    let pk = |op: u32, three: bool| {
        let src2 = if three { 0x103 << 18 } else { 0 };
        vec![0xCC00_4000 | op << 16, 0x1800_0000 | src2 | 0x102 << 9 | 0x101]
    };
    // v_mov_b32 v0, v4 (the destination's other half), then v_fma_mix* v0, v1, v2, v3
    // with all three sources f32 (op_sel_hi 0).
    let mix = |op: u32| {
        vec![vop1(0x001, 0, vgpr(4)), 0xCC00_0000 | op << 16, 0x103 << 18 | 0x102 << 9 | 0x101]
    };
    let pack = |lo: u16, hi: u16| u32::from(lo) | u32::from(hi) << 16;
    let f32b = |x: f32| x.to_bits();
    type Want = fn([u32; 4]) -> u32;
    let cases: Vec<(&str, Vec<u32>, Vec<[u32; 4]>, Want)> = vec![
        (
            "v_cvt_f16_f32",
            vec![vop1(0x001, 0, 0x80), vop1(0x00A, 0, vgpr(1))],
            [0x3F7F_FFFF, 0xBF7F_FFFF, 0x3F80_1001, 0x3F80_1000, 0x477F_F000, 0x3586_0000]
                .map(|x| [x, 0, 0, 0])
                .to_vec(),
            |i| u32::from(f16_rne(f64::from(f32::from_bits(i[0])))),
        ),
        (
            "v_pk_add_f16",
            pk(0x0F, false),
            vec![
                // Marvisak/VIctory#1614's example in the low lane: 0xf8b6 + 0x0001.
                [pack(0xF8B6, 0x3C01), pack(0x0001, 0x1001), 0, 0],
                [pack(0x7BFF, 0xC784), pack(0x7BFF, 0x2036), 0, 0],
            ],
            |i| {
                let lane = |k: u32| {
                    let (a, b) = ((i[0] >> k) as u16, (i[1] >> k) as u16);
                    u32::from(f16_rne(h(a) + h(b))) << k
                };
                lane(0) | lane(16)
            },
        ),
        (
            "v_pk_fma_f16",
            pk(0x0E, true),
            // f32 holds each product and sum exactly, so the only rounding is the narrowing.
            vec![[pack(0x3C01, 0x3BFF), pack(0x3C01, 0xBC00), pack(0x3C01, 0xBC00), 0]],
            |i| {
                let lane = |k: u32| {
                    let (a, b, c) = ((i[0] >> k) as u16, (i[1] >> k) as u16, (i[2] >> k) as u16);
                    u32::from(f16_rne(h(a) * h(b) + h(c))) << k
                };
                lane(0) | lane(16)
            },
        ),
        (
            "v_fma_mixlo_f16",
            mix(0x21),
            vec![
                [0x3F7F_FFFF, f32b(1.0), 0, 0xABCD_1234],
                [f32b(1.0), 0x3F80_1001, 0, 0x5555_AAAA],
            ],
            |i| {
                let r = f64::from(f32::from_bits(i[0])) * f64::from(f32::from_bits(i[1]));
                i[3] & 0xFFFF_0000 | u32::from(f16_rne(r))
            },
        ),
        (
            "v_fma_mixhi_f16",
            mix(0x22),
            vec![
                [0x3F7F_FFFF, f32b(1.0), 0, 0xABCD_1234],
                [f32b(1.0), 0x3F80_1001, 0, 0x5555_AAAA],
            ],
            |i| {
                let r = f64::from(f32::from_bits(i[0])) * f64::from(f32::from_bits(i[1]));
                i[3] & 0xFFFF | u32::from(f16_rne(r)) << 16
            },
        ),
    ];
    let mut wrong = Vec::new();
    for (name, words, inputs, want) in cases {
        let was = lower::TEST_F16_NEAREST.replace(Some(true));
        let shader = build_valu_cs(&words);
        lower::TEST_F16_NEAREST.set(was);
        let mask = if name == "v_cvt_f16_f32" { 0xFFFF } else { u32::MAX };
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            let w = want(inp);
            if got & mask != w {
                wrong.push(format!("{backend}: {name}({inp:08x?}) = {got:#010x}, want {w:#010x}"));
            }
        });
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn scalar_zero_counts_set_scc_for_nonzero_results() {
    let cases = [
        [0, 0, 0, 0],
        [u32::MAX, u32::MAX, 0, 0],
        [u32::MAX, 0, 0, 0],
        [0, u32::MAX, 0, 0],
        [0x8000_0001, 0xaaaa_5555, 0, 0],
    ];
    for (opcode, wide) in [(0x0d, false), (0x0e, true)] {
        // Count zero bits in s0 or s[0:1] into s6. Complement the count on SCC=0.
        let shader = build_valu_cs(&[
            0xbe86_0000 | opcode << 8,
            sop2(0x0a, 7, 128, 193),
            sop2(0x12, 7, 6, 7),
            vop1(0x001, 0, sgpr(7)),
        ]);
        run_cases(&shader, &cases, |backend, _, inputs, got| {
            let count = inputs[0].count_zeros() + if wide { inputs[1].count_zeros() } else { 0 };
            assert_eq!(got, if count == 0 { u32::MAX } else { count }, "{backend}");
        });
    }
}

#[test]
fn scalar_sign_extension_preserves_scc() {
    for (opcode, byte) in [(0x19, true), (0x1a, false)] {
        let shader = build_valu_cs(&[
            0xbf06_8001,               // s_cmp_eq_u32 s1, 0
            0xbe86_0000 | opcode << 8, // sign-extend s0 into s6
            sop2(0x0a, 7, 6, 2),       // s_cselect_b32 s7, s6, s2
            vop1(0x001, 0, sgpr(7)),
        ]);
        let mut cases = Vec::new();
        for input in [
            0,
            0xffff_007f,
            0x1234_0080,
            0xaaaa_7fff,
            0x5555_8000,
            u32::MAX,
        ] {
            for flag_input in [0, 1] {
                cases.push([input, flag_input, 0x1234_5678, 0]);
            }
        }
        run_cases(
            &shader,
            &cases,
            |backend, _, [input, flag_input, fallback, _], got| {
                let extended = if byte {
                    i32::from(input as i8)
                } else {
                    i32::from(input as i16)
                };
                assert_eq!(
                    got,
                    if flag_input == 0 {
                        extended as u32
                    } else {
                        fallback
                    },
                    "{backend}"
                );
            },
        );
    }
}

#[test]
fn bitwise_mov_and_cndmask_preserve_subnormals_and_snans() {
    let patterns: &[u32] = &[
        0x0000_0001, // +subnormal
        0x8000_0001, // -subnormal
        0x7F80_0001, // +sNaN
        0xFF80_0042, // -sNaN
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        (-1.5f32).to_bits(),
        3.25f32.to_bits(),
    ];

    for (abs, neg, expected) in [
        (1u8, 0u8, (|x: u32| x & 0x7FFF_FFFF) as fn(u32) -> u32),
        (0u8, 1u8, |x: u32| x ^ 0x8000_0000),
        (1u8, 1u8, |x: u32| x | 0x8000_0000),
    ] {
        let mov = build_valu_cs(&vop3_mod(0x181, 0, vgpr(1), 0, 0, abs, neg, 0, false));
        let cases: Vec<[u32; 4]> = patterns.iter().map(|&x| [x, 0, 0, 0]).collect();
        run_cases(&mov, &cases, |backend, _, inp, got| {
            let want = expected(inp[0]);
            assert_eq!(
                got, want,
                "{backend}: v_mov_b32(abs={abs}, neg={neg}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0],
            );
        });
    }

    // v_cmp_ne_u32_e64 s6, v3, 0; v_cndmask_b32_e64 v0, |v1|, -v2, s6
    let cmp = vop3(0x0c5, 6, vgpr(3), 0x80, 0);
    let sel = vop3_mod(0x101, 0, vgpr(1), vgpr(2), sgpr(6), 1, 2, 0, false);
    let cnd = build_valu_cs(&[cmp[0], cmp[1], sel[0], sel[1]]);
    let mut cases = Vec::new();
    for &x in patterns {
        cases.push([x, x, 0, 0]);
        cases.push([x, x, 1, 0]);
    }
    run_cases(&cnd, &cases, |backend, _, inp, got| {
        let want = if inp[2] != 0 {
            inp[1] ^ 0x8000_0000
        } else {
            inp[0] & 0x7FFF_FFFF
        };
        assert_eq!(
            got, want,
            "{backend}: v_cndmask_b32(|{:#010x}|, -{:#010x}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_cvt_pkrtz_f16_f32_honors_clamp() {
    // v_cvt_pkrtz_f16_f32 v0, v1, v2 clamp
    let clamped = build_valu_cs(&vop3_mod(0x12f, 0, vgpr(1), vgpr(2), 0, 0, 0, 0, true));
    let cases: &[([u32; 4], u32)] = &[
        ([(-1.0f32).to_bits(), 2.0f32.to_bits(), 0, 0], 0x3C00_0000),
        ([0.5f32.to_bits(), 0.25f32.to_bits(), 0, 0], 0x3400_3800),
        ([0x7FC0_0000, (-0.0f32).to_bits(), 0, 0], 0x0000_0000),
        (
            [f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 0, 0],
            0x0000_3C00,
        ),
        // 1.0 - 2^-24 stays strictly below 1.0 and rounds toward zero to 0x3BFF.
        ([0x3F7F_FFFF, 1.0f32.to_bits(), 0, 0], 0x3C00_3BFF),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&clamped, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cvt_pkrtz_f16_f32 clamp({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    // With source neg on both inputs: -(-2.0) -> 2.0 -> 1.0, -(0.5) -> -0.5 -> 0.0.
    let neg_clamped = build_valu_cs(&vop3_mod(0x12f, 0, vgpr(1), vgpr(2), 0, 0, 3, 0, true));
    run_cases(
        &neg_clamped,
        &[[(-2.0f32).to_bits(), 0.5f32.to_bits(), 0, 0]],
        |backend, _, _, got| {
            assert_eq!(
                got, 0x0000_3C00,
                "{backend}: v_cvt_pkrtz_f16_f32(-v1, -v2) clamp = {got:#010x}",
            );
        },
    );
}

#[test]
fn ds_add_f32_accumulates_atomically_across_waves() {
    // 64 threads: LDS[0] = v1, every thread adds (thread_id + 1) as f32 -> +2080.0.
    let mut body = vec![vop1(0x001, 5, 0x80)]; // v5 = 0
    body.extend(ds(DS_WRITE_B32, 0, 5, 1));
    body.push(S_BARRIER);
    body.push(vop2(0x25, 6, 0x81, 0)); // v_add_nc_u32 v6, 1, v0
    body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32 v6, v6
    body.extend(ds(DS_ADD_F32, 0, 5, 6));
    body.push(S_BARRIER);
    body.extend(ds(DS_READ_B32, 0, 5, 0));
    let shader = build_cs(&body, 64, 512);
    for (init, want) in [(0.0f32, 2080.0f32), (-100.5, 1979.5), (1024.0, 3104.0)] {
        run_cases(&shader, &[[init.to_bits(), 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                f32::from_bits(got),
                want,
                "{backend}: ds_add_f32 from {init} over 64 threads",
            );
        });
    }

    // Per-lane isolated slot (v5 = v0 << 2): check ds_add_rtn_f32 old value and updated value.
    let mut ret_old = vec![vop2(0x1A, 5, 0x82, 0)]; // v_lshlrev_b32 v5, 2, v0
    ret_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
    ret_old.extend(ds(DS_ADD_RTN_F32, 0, 5, 2));
    let mut ret_new = vec![vop2(0x1A, 5, 0x82, 0)];
    ret_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
    ret_new.extend(ds(DS_ADD_RTN_F32, 6, 5, 2));
    ret_new.extend(ds(DS_READ_B32, 0, 5, 0));
    let s_old = build_cs(&ret_old, 64, 512);
    let s_new = build_cs(&ret_new, 64, 512);
    let add_cases: &[(f32, f32, f32)] = &[
        (1.5, 2.25, 3.75),
        (-3.5, 3.5, 0.0),
        (f32::INFINITY, -1.0, f32::INFINITY),
    ];
    let inputs: Vec<[u32; 4]> = add_cases
        .iter()
        .map(|&(a, b, _)| [a.to_bits(), b.to_bits(), 0, 0])
        .collect();
    run_cases(&s_old, &inputs, |backend, i, _, got| {
        assert_eq!(
            got,
            add_cases[i].0.to_bits(),
            "{backend}: ds_add_rtn_f32 old value case {i}",
        );
    });
    run_cases(&s_new, &inputs, |backend, i, _, got| {
        assert_eq!(
            got,
            add_cases[i].2.to_bits(),
            "{backend}: ds_add_rtn_f32 new value case {i}",
        );
    });
}

#[test]
fn ds_cmpst_f32_compares_as_ieee_floats() {
    // Per-lane isolated slot (v5 = v0 << 2): LDS[v5] = v1; cmpst(cmp=v2, new=v3).
    let mut ret_old = vec![vop2(0x1A, 5, 0x82, 0)]; // v_lshlrev_b32 v5, 2, v0
    ret_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
    ret_old.extend(ds2(DS_CMPST_RTN_F32, 0, 5, 2, 3));
    let mut ret_new = vec![vop2(0x1A, 5, 0x82, 0)];
    ret_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
    ret_new.extend(ds2(DS_CMPST_F32, 0, 5, 2, 3));
    ret_new.extend(ds(DS_READ_B32, 0, 5, 0));
    let s_old = build_cs(&ret_old, 64, 512);
    let s_new = build_cs(&ret_new, 64, 512);

    // (init, cmp, new, want_after)
    let cases: &[(u32, u32, u32, u32)] = &[
        (1.5f32.to_bits(), 1.5f32.to_bits(), 4.25f32.to_bits(), 4.25f32.to_bits()),
        (1.5f32.to_bits(), 2.5f32.to_bits(), 4.25f32.to_bits(), 1.5f32.to_bits()),
        // -0.0 == +0.0 under IEEE-754 float comparison: swap succeeds!
        (0x8000_0000, 0x0000_0000, 3.0f32.to_bits(), 3.0f32.to_bits()),
        (0x0000_0000, 0x8000_0000, (-3.0f32).to_bits(), (-3.0f32).to_bits()),
        // Identical qNaN bit patterns must NOT match under IEEE-754 float equality.
        (0x7FC0_0000, 0x7FC0_0000, 1.0f32.to_bits(), 0x7FC0_0000),
        (f32::INFINITY.to_bits(), f32::INFINITY.to_bits(), 7.0f32.to_bits(), 7.0f32.to_bits()),
        (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 7.0f32.to_bits(), f32::INFINITY.to_bits()),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, c, _)| [a, b, c, 0]).collect();
    run_cases(&s_old, &inputs, |backend, i, _, got| {
        assert_eq!(
            got, cases[i].0,
            "{backend}: ds_cmpst_rtn_f32 old value case {i}",
        );
    });
    run_cases(&s_new, &inputs, |backend, i, _, got| {
        assert_eq!(
            got, cases[i].3,
            "{backend}: ds_cmpst_f32 new value case {i}",
        );
    });
}

#[test]
fn buffer_atomic_extended_ops_match_rdna2() {
    // Helper: write v1 to UAV[0], apply atomic op using v2 (or v[2:3]), return old (glc) or new (load).
    let run_op = |op: u16, name: &str, cases: &[([u32; 4], u32)]| {
        let mut body_old = Vec::new();
        body_old.extend(buffer_store_dword(1, 8, 0));
        body_old.extend(buffer_atomic(op, 2, 8, true));
        body_old.push(vop1(0x001, 0, vgpr(2)));
        let s_old = build_valu_cs(&body_old);

        let mut body_new = Vec::new();
        body_new.extend(buffer_store_dword(1, 8, 0));
        body_new.extend(buffer_atomic(op, 2, 8, false));
        body_new.extend(buffer_load_dword(0, 8, 0));
        let s_new = build_valu_cs(&body_new);

        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&s_old, &inputs, |backend, i, inp, got| {
            assert_eq!(
                got, inp[0],
                "{backend}: {name} (glc old) case {i} ({:#010x}, {:#010x}) = {got:#010x}, want {:#010x}",
                inp[0], inp[1], inp[0],
            );
        });
        run_cases(&s_new, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {name} (new) case {i} ({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1],
            );
        });
    };

    // buffer_atomic_sub (0x33): wrapping subtract.
    run_op(
        0x33,
        "buffer_atomic_sub",
        &[
            ([10, 3, 0, 0], 7),
            ([2, 5, 0, 0], 2u32.wrapping_sub(5)),
            ([0, 0, 0, 0], 0),
        ],
    );

    // buffer_atomic_csub (0x34): clamped unsigned subtract (saturating_sub).
    run_op(
        0x34,
        "buffer_atomic_csub",
        &[
            ([10, 3, 0, 0], 7),
            ([3, 3, 0, 0], 0),
            ([2, 5, 0, 0], 0),
            ([0x8000_0005, 0x8000_0000, 0, 0], 5),
        ],
    );

    // buffer_atomic_inc (0x3c): (old >= data) ? 0 : old + 1.
    run_op(
        0x3c,
        "buffer_atomic_inc",
        &[
            ([0, 5, 0, 0], 1),
            ([4, 5, 0, 0], 5),
            ([5, 5, 0, 0], 0),
            ([9, 5, 0, 0], 0),
            ([u32::MAX, u32::MAX, 0, 0], 0),
        ],
    );

    // buffer_atomic_dec (0x3d): (old == 0 || old > data) ? data : old - 1.
    run_op(
        0x3d,
        "buffer_atomic_dec",
        &[
            ([5, 5, 0, 0], 4),
            ([1, 5, 0, 0], 0),
            ([0, 5, 0, 0], 5),
            ([9, 5, 0, 0], 5),
            ([3, 0, 0, 0], 0),
        ],
    );

    // buffer_atomic_fcmpswap (0x3e): data = (new=v2, cmp=v3), IEEE-754 float equality.
    run_op(
        0x3e,
        "buffer_atomic_fcmpswap",
        &[
            ([1.5f32.to_bits(), 4.25f32.to_bits(), 1.5f32.to_bits(), 0], 4.25f32.to_bits()),
            ([1.5f32.to_bits(), 4.25f32.to_bits(), 2.5f32.to_bits(), 0], 1.5f32.to_bits()),
            ([0x8000_0000, 3.0f32.to_bits(), 0x0000_0000, 0], 3.0f32.to_bits()),
            ([0x7FC0_0000, 1.0f32.to_bits(), 0x7FC0_0000, 0], 0x7FC0_0000),
        ],
    );

    // buffer_atomic_fmin (0x3f) and buffer_atomic_fmax (0x40).
    let float_pairs: &[(u32, u32)] = &[
        (1.0f32.to_bits(), 2.0f32.to_bits()),
        ((-2.5f32).to_bits(), 1.5f32.to_bits()),
        (0x0000_0000, 0x8000_0000),
        (0x7FC0_0000, 3.0f32.to_bits()),
        (3.0f32.to_bits(), 0x7FC0_0000),
        (0x7F80_0001, 3.0f32.to_bits()), // sNaN in memory quieted
    ];
    let fmin_cases: Vec<([u32; 4], u32)> = float_pairs
        .iter()
        .map(|&(old, data)| ([old, data, 0, 0], lds_min_max_ref(old, data, false)))
        .collect();
    let fmax_cases: Vec<([u32; 4], u32)> = float_pairs
        .iter()
        .map(|&(old, data)| ([old, data, 0, 0], lds_min_max_ref(old, data, true)))
        .collect();
    run_op(0x3f, "buffer_atomic_fmin", &fmin_cases);
    run_op(0x40, "buffer_atomic_fmax", &fmax_cases);
}

#[test]
fn v_ldexp_f32_scales_and_handles_edge_cases() {
    // v_ldexp_f32 v0, v1, v2 (VOP3 opcode 0x362)
    let plain = build_valu_cs(&vop3(0x362, 0, vgpr(1), vgpr(2), 0));
    let cases: &[([u32; 4], u32)] = &[
        ([1.5f32.to_bits(), 3, 0, 0], 12.0f32.to_bits()),
        ([(-1.5f32).to_bits(), (-2i32) as u32, 0, 0], (-0.375f32).to_bits()),
        ([1.0f32.to_bits(), 127, 0, 0], 0x7F00_0000),
        ([1.0f32.to_bits(), 128, 0, 0], f32::INFINITY.to_bits()),
        ([(-1.0f32).to_bits(), i32::MAX as u32, 0, 0], f32::NEG_INFINITY.to_bits()),
        ([1.0f32.to_bits(), (-126i32) as u32, 0, 0], 0x0080_0000),
        ([(-1.0f32).to_bits(), (-127i32) as u32, 0, 0], (-0.0f32).to_bits()),
        ([1.0f32.to_bits(), i32::MIN as u32, 0, 0], 0.0f32.to_bits()),
        ([0.0f32.to_bits(), 100, 0, 0], 0.0f32.to_bits()),
        ([(-0.0f32).to_bits(), 100, 0, 0], (-0.0f32).to_bits()),
        ([0x8000_0001, 50, 0, 0], (-0.0f32).to_bits()), // subnormal flushed under f32 FTZ
        ([f32::INFINITY.to_bits(), (-100i32) as u32, 0, 0], f32::INFINITY.to_bits()),
        ([f32::NEG_INFINITY.to_bits(), (-100i32) as u32, 0, 0], f32::NEG_INFINITY.to_bits()),
        ([0x7FC0_0000, 5, 0, 0], 0x7FC0_0000),
        ([0x7F80_0001, 5, 0, 0], 0x7FC0_0001), // sNaN quieted
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&plain, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        if want & 0x7FFF_FFFF > 0x7F80_0000 {
            assert!(
                got & 0x7FFF_FFFF > 0x7F80_0000 && got & 0x0040_0000 != 0,
                "{backend}: v_ldexp_f32 case {i} expected qNaN, got {got:#010x}",
            );
        } else {
            assert_eq!(
                got, want,
                "{backend}: v_ldexp_f32({:#010x}, {}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1] as i32,
            );
        }
    });

    // Source neg + output clamp: -(-0.75) * 2^1 = 1.5 -> clamped to 1.0; -(0.25) * 2^1 = -0.5 -> 0.0.
    let neg_clamp = build_valu_cs(&vop3_mod(0x362, 0, vgpr(1), vgpr(2), 0, 0, 1, 0, true));
    let mod_cases: &[([u32; 4], u32)] = &[
        ([(-0.75f32).to_bits(), 1, 0, 0], 1.0f32.to_bits()),
        ([0.25f32.to_bits(), 1, 0, 0], 0.0f32.to_bits()),
        ([(-0.25f32).to_bits(), 1, 0, 0], 0.5f32.to_bits()),
    ];
    let mod_in: Vec<[u32; 4]> = mod_cases.iter().map(|c| c.0).collect();
    run_cases(&neg_clamp, &mod_in, |backend, i, _, got| {
        assert_eq!(
            got, mod_cases[i].1,
            "{backend}: v_ldexp_f32(-v1, v2) clamp case {i} = {got:#010x}, want {:#010x}",
            mod_cases[i].1,
        );
    });
}

#[test]
fn v_med3_min3_max3_integer_matches_rdna2() {
    let make = |op: u16| build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));

    let min3_i = make(0x152); // v_min3_i32
    let min3_u = make(0x153); // v_min3_u32
    let max3_i = make(0x155); // v_max3_i32
    let max3_u = make(0x156); // v_max3_u32
    let med3_i = make(0x158); // v_med3_i32
    let med3_u = make(0x159); // v_med3_u32

    // (inputs [a, b, c, 0], min_i32, max_i32, med_i32, min_u32, max_u32, med_u32)
    let cases: &[([u32; 4], u32, u32, u32, u32, u32, u32)] = &[
        // Plain ascending order
        ([1, 2, 3, 0], 1, 3, 2, 1, 3, 2),
        // Reversed
        ([3, 2, 1, 0], 1, 3, 2, 1, 3, 2),
        // All equal
        ([7, 7, 7, 0], 7, 7, 7, 7, 7, 7),
        // Two equal (low)
        ([1, 1, 5, 0], 1, 5, 1, 1, 5, 1),
        // Two equal (high)
        ([5, 9, 9, 0], 5, 9, 9, 5, 9, 9),
        // Signed vs unsigned with negative numbers
        (
            [(-3i32) as u32, (-1i32) as u32, 2, 0],
            (-3i32) as u32,
            2,
            (-1i32) as u32,
            2,
            (-1i32) as u32,
            (-3i32) as u32,
        ),
        // Signed extremes
        (
            [0x8000_0000, 0x7FFF_FFFF, 0, 0],
            0x8000_0000,
            0x7FFF_FFFF,
            0,
            0,
            0x8000_0000,
            0x7FFF_FFFF,
        ),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();

    let check = |name: &str,
                 shader: &HostShader,
                 extract: fn(&([u32; 4], u32, u32, u32, u32, u32, u32)) -> u32| {
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = extract(&cases[i]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2]
            );
        });
    };

    check("v_min3_i32", &min3_i, |c| c.1);
    check("v_max3_i32", &max3_i, |c| c.2);
    check("v_med3_i32", &med3_i, |c| c.3);
    check("v_min3_u32", &min3_u, |c| c.4);
    check("v_max3_u32", &max3_u, |c| c.5);
    check("v_med3_u32", &med3_u, |c| c.6);
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    // v_cmp_class_f32_e64 s6, v1, v2 (VOP3 encoding of VOPC 0x088)
    // v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp = build_valu_cs(&[
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x40_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN
        (0x7FC0_0000, 1), // +qNaN (quiet bit set)
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    cases.push(([0x3F80_0000, 0, 0, 0], 0u32));
    cases.push(([0x3F80_0000, 0xFFFF_FC00, 0, 0], 0u32));

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn scalar_lshl_add_and_minmax_set_scc() {
    // SOP2 opcodes: s_addc_u32 = 0x04, s_cselect_b32 = 0x0a,
    // s_lshl{1,2,3,4}_add_u32 = 0x2e..=0x31.
    for (shift, op) in [(1u32, 0x2Eu8), (2, 0x2F), (3, 0x30), (4, 0x31)] {
        // s6 = (s0 << shift) + s1, SCC = 32-bit unsigned carry-out;
        // s7 = s2 + 0 + SCC (64-bit address/pointer high-word carry propagation);
        // v0 = s7.
        let shader = build_valu_cs(&[
            sop2(op, 6, 0, 1),
            sop2(0x04, 7, 2, 0x80),
            vop1(0x001, 0, sgpr(7)),
        ]);
        let cases: &[([u32; 4], u32)] = &[
            // Non-zero sum, no carry -> SCC must be 0 (not `r != 0`).
            ([3, 4, 10, 0], 10),
            // Exact zero result with carry shifted out of s0 -> SCC must be 1.
            ([1u32 << (32 - shift), 0, 10, 0], 11),
            // Carry from addition wrap only (shift does not overflow).
            ([1, 0xFFFF_FFFF, 10, 0], 11),
            // Carry from shift overflow only (addition does not wrap).
            ([0x8000_0000, 5, 10, 0], 11),
            // Carry from both shift and addition.
            ([0xFFFF_FFFF, 0xFFFF_FFFF, 10, 0], 11),
            // Zero + zero -> no carry.
            ([0, 0, 10, 0], 10),
        ];
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            let expected_carry = u32::from((((u64::from(inp[0]) << shift) + u64::from(inp[1])) >> 32) != 0);
            assert_eq!(want, inp[2] + expected_carry);
            assert_eq!(
                got, want,
                "{backend}: s_lshl{shift}_add_u32({:#010x}, {:#010x}) + s_addc_u32({}) = {got}, want {want}",
                inp[0], inp[1], inp[2],
            );
        });
    }

    // s_min_i32 (0x06), s_min_u32 (0x07), s_max_i32 (0x08), s_max_u32 (0x09):
    // SCC = 1 iff src0 won the comparison (`src0 < src1` or `src0 > src1`).
    let minmax_ops: &[(u8, &str, fn(u32, u32) -> bool)] = &[
        (0x06, "s_min_i32", |a, b| (a as i32) < (b as i32)),
        (0x07, "s_min_u32", |a, b| a < b),
        (0x08, "s_max_i32", |a, b| (a as i32) > (b as i32)),
        (0x09, "s_max_u32", |a, b| a > b),
    ];
    let pairs: &[(u32, u32)] = &[
        (0, 5),
        (5, 0),
        (5, 3),
        (3, 5),
        (5, 5),
        (0, 0),
        (0, (-5i32) as u32),
        ((-5i32) as u32, 0),
        ((-5i32) as u32, (-5i32) as u32),
        (0x8000_0000, 1),
        (1, 0x8000_0000),
    ];
    let inputs: Vec<[u32; 4]> = pairs.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
    for &(op, name, pred_fn) in minmax_ops {
        // s6 = op(s0, s1), SCC = src0 won; s7 = SCC ? 1 : 0; v0 = s7.
        let shader = build_valu_cs(&[
            sop2(op, 6, 0, 1),
            sop2(0x0A, 7, 0x81, 0x80),
            vop1(0x001, 0, sgpr(7)),
        ]);
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = u32::from(pred_fn(inp[0], inp[1]));
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}) SCC = {got}, want {want} (case {i})",
                inp[0], inp[1],
            );
        });
    }
}

/// MTBUF instruction encoding (`tbuffer_load_format_*`, soffset = inline 0).
const fn mtbuf(op: u8, format: u8, vdata: u8, vaddr: u8, srsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x3A << 26
            | (format as u32 & 0x7F) << 19
            | (op as u32 & 0x7) << 16
            | (offset as u32 & 0xFFF),
        0x80 << 24
            | ((op as u32 >> 3) & 1) << 21
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

#[test]
fn tbuffer_load_format_d16_packs_float_and_int_channels() {
    // MTBUF opcodes: tbuffer_load_format_d16_xyz = 0x0a, tbuffer_load_format_d16_xyzw = 0x0b.
    // Input raw SRV at table offset 0 dwords (heap index 0); output raw UAV at 8 dwords (index 1).
    let cases: &[(u8, u8, &[u32], [u32; 2], &str)] = &[
        // 16_16_16_16_FLOAT (71), xyzw: [1.0, -2.0, 0.5, 4.0]
        (
            0x0B,
            71,
            &[0xC000_3C00, 0x4400_3800],
            [0xC000_3C00, 0x4400_3800],
            "16_16_16_16_FLOAT xyzw",
        ),
        // 16_16_FLOAT (29), xyzw: missing z=0.0 (0x0000), w=1.0 (0x3C00)
        (
            0x0B,
            29,
            &[0xBC00_3800, 0],
            [0xBC00_3800, 0x3C00_0000],
            "16_16_FLOAT xyzw defaults",
        ),
        // 16_16_16_16_FLOAT (71), xyz: odd 3rd channel leaves high half of v2 zero
        (
            0x0A,
            71,
            &[0xC000_3C00, 0x4400_3800],
            [0xC000_3C00, 0x0000_3800],
            "16_16_16_16_FLOAT xyz",
        ),
        // 8_8_8_8_UINT (60), xyzw: [0x01, 0x12, 0x80, 0xFF]
        (
            0x0B,
            60,
            &[0xFF80_1201],
            [0x0012_0001, 0x00FF_0080],
            "8_8_8_8_UINT xyzw",
        ),
        // 8_8_8_8_SINT (61), xyzw: [-1, -128, 127, 0]
        (
            0x0B,
            61,
            &[0x007F_80FF],
            [0xFF80_FFFF, 0x0000_007F],
            "8_8_8_8_SINT xyzw",
        ),
        // 32_32_32_32_UINT (75), xyzw: clamps values > 0xFFFF to 0xFFFF
        (
            0x0B,
            75,
            &[0, 65535, 70000, u32::MAX],
            [0xFFFF_0000, 0xFFFF_FFFF],
            "32_32_32_32_UINT xyzw clamp",
        ),
        // 32_32_32_32_SINT (76), xyzw: clamps to [-32768, 32767]
        (
            0x0B,
            76,
            &[(-40000i32) as u32, (-1i32) as u32, 32767, 50000],
            [0xFFFF_8000, 0x7FFF_7FFF],
            "32_32_32_32_SINT xyzw clamp",
        ),
    ];

    for &(op, fmt, input, want, label) in cases {
        let mut words = Vec::new();
        words.push(s_mov_b32(5, 0x80)); // s5 = 0
        words.extend_from_slice(&s_load_dwordx4(8, 4)); // s[8:11] = SRV (offset 0)
        // s_load_dwordx4 s[12:15], s[4:5], 0x20 (UAV at table offset 8 dwords)
        words.extend_from_slice(&[0xF408_0302, 0xFA00_0020]);
        words.push(vop1(0x001, 0, 0x80)); // v0 = 0
        words.extend_from_slice(&mtbuf(op, fmt, 1, 0, 8, 0)); // load into v[1:2]
        words.extend_from_slice(&buffer_store_dword(1, 12, 0));
        words.extend_from_slice(&buffer_store_dword(2, 12, 4));
        words.push(S_ENDPGM);

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
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
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let l = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut dxil).unwrap();
        }
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil,
            user_data: l.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: Vec::new(),
        };
        #[cfg(windows)]
        for (backend, warp) in [("hardware", false), ("WARP", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0, 0, 0, 0, 0]))],
                        views: &[
                            View::Buffer {
                                data: input,
                                stride: super::host::Stride::Raw,
                            },
                            View::RawUav(16),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(
                [out[0][0], out[0][1]],
                want,
                "{backend}: {label}: got [{:#010x}, {:#010x}], want [{:#010x}, {:#010x}]",
                out[0][0],
                out[0][1],
                want[0],
                want[1],
            );
        }
    }
}

/// MUBUF `buffer_load_*` with `offen = 1`, `soffset = 0`, `offset = 0`.
const fn mubuf_load_offen(op: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32 & 0x7F) << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

#[test]
fn cbv_narrow_buffer_loads_extract_and_sign_extend() {
    // CBV contents (64 bytes = 16 dwords, 16-byte rows):
    //   dword 0 (bytes 0..4):   [0x05, 0xFE, 0x7F, 0x81]
    //   dword 1 (bytes 4..8):   [0x01, 0x80, 0xAB, 0xCD]
    //   dword 2 (bytes 8..12):  [0x00, 0x00, 0x00, 0x00]
    //   dword 3 (bytes 12..16): [0x00, 0x00, 0x00, 0x80]
    //   dword 4 (bytes 16..20): [0xDE, 0xAD, 0xBE, 0xEF] (cross-16-byte-row straddle at 15)
    let cbv_data: [u32; 16] = [
        0x817F_FE05,
        0xCDAB_8001,
        0x0000_0000,
        0x8000_0000,
        0xEFBE_ADDE,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let bytes: Vec<u8> = cbv_data.iter().flat_map(|w| w.to_le_bytes()).collect();
    let offsets: &[u32] = &[0, 1, 2, 3, 4, 5, 7, 15, 16, 19];

    // MUBUF opcodes: ubyte=0x08, sbyte=0x09, ushort=0x0a, sshort=0x0b.
    let ops: &[(u8, u32, bool, &str)] = &[
        (0x08, 8, false, "buffer_load_ubyte"),
        (0x09, 8, true, "buffer_load_sbyte"),
        (0x0A, 16, false, "buffer_load_ushort"),
        (0x0B, 16, true, "buffer_load_sshort"),
    ];

    for &(op, bits, signed, name) in ops {
        let mut words = Vec::new();
        words.push(s_mov_b32(5, 0x80)); // s5 = 0
        words.extend_from_slice(&s_load_dwordx4(8, 4)); // s[8:11] = CBV at offset 0
        // s_load_dwordx4 s[12:15], s[4:5], 0x20 (UAV at table offset 8 dwords)
        words.extend_from_slice(&[0xF408_0302, 0xFA00_0020]);
        words.push(vop1(0x001, 1, sgpr(0))); // v1 = byte offset from root constant s0
        words.extend_from_slice(&mubuf_load_offen(op, 0, 1, 8)); // v0 = narrow load from CBV
        words.extend_from_slice(&buffer_store_dword(0, 12, 0));
        words.push(S_ENDPGM);

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
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
            used: vec![
                UsedResource {
                    kind: ResourceType::Cbv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let l = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut dxil).unwrap();
        }
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil,
            user_data: l.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: Vec::new(),
        };
        #[cfg(windows)]
        for (backend, warp) in [("hardware", false), ("WARP", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            for &off in offsets {
                let want = match (bits, signed) {
                    (8, false) => u32::from(bytes[off as usize]),
                    (8, true) => (bytes[off as usize] as i8) as i32 as u32,
                    (16, false) => u32::from(u16::from_le_bytes([
                        bytes[off as usize],
                        bytes[off as usize + 1],
                    ])),
                    (16, true) => (i16::from_le_bytes([
                        bytes[off as usize],
                        bytes[off as usize + 1],
                    ])) as i32 as u32,
                    _ => unreachable!(),
                };
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(vec![off, 0, 0, 0, 0]))],
                            views: &[View::Cbv(&cbv_data), View::RawUav(16)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                assert_eq!(
                    out[0][0], want,
                    "{backend}: {name} at CBV byte offset {off} = {:#010x}, want {want:#010x}",
                    out[0][0],
                );
            }
        }
    }
}
