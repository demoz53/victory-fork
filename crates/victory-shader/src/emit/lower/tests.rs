use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use super::half_input_partners;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_input_partners(&io, &sig);
    assert_eq!(p.get(&0), Some(&1));
    assert_eq!(p.get(&3), Some(&4));
    assert_eq!(p.get(&5), None);
}

/// The packed pedestrian layouts of the title: an element
/// behind another one in its register follows in the first parameter
/// with a dword left, and is not a half of its register's row.
#[test]
fn packed_elements_behind_another_take_the_first_free_dword() {
    use super::super::container::SigElement;
    use super::packed_half;
    let element = |index, row, col, cols| SigElement {
        name: "TEXCOORD".into(),
        index,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    };
    // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
    // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
    // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
    let sig = vec![
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
        element(6, 4, 3, 1),
        element(3, 5, 0, 4),
        element(5, 6, 0, 4),
        element(7, 7, 0, 2),
        element(9, 9, 0, 3),
        element(11, 10, 0, 3),
        element(15, 12, 0, 4),
    ];
    let packs = [
        (2, Some(3)),
        (4, Some(5)),
        (6, Some(7)),
        (9, Some(10)),
        (12, None),
    ];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
    assert_eq!(at(4, 2), Some((4, 2)));
    assert_eq!(at(4, 3), None);
    assert_eq!(at(4, 4), Some((5, 0)));
    assert_eq!(at(4, 7), Some((5, 3)));
    // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
    assert_eq!(at(6, 3), Some((6, 3)));
    assert_eq!(at(6, 5), Some((7, 1)));
    assert_eq!(at(6, 6), Some((4, 3)));
    assert_eq!(at(6, 7), None);
    // Parameters without such an element are as before.
    assert_eq!(at(9, 3), None);
    assert_eq!(at(9, 4), Some((10, 0)));
    assert_eq!(at(12, 3), Some((12, 3)));
    assert_eq!(at(12, 4), None);
    // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
    // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
    // TEXCOORD7.
    let sig = vec![
        element(0, 1, 0, 2),
        element(7, 1, 2, 2),
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
    ];
    let packs = [(3, Some(4)), (1, Some(2))];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    assert_eq!(at(1, 1), Some((1, 1)));
    assert_eq!(at(1, 2), Some((2, 0)));
    assert_eq!(at(1, 5), Some((2, 3)));
    assert_eq!(at(1, 6), Some((1, 2)));
    assert_eq!(at(1, 7), Some((1, 3)));
    assert_eq!(at(3, 6), Some((4, 2)));
    assert_eq!(at(3, 7), None);
    // A row no element holds gives nothing, as before.
    assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
}

/// A 16-bit operation's SDWA destination with the reserved dst_unused 3 is
/// refused (`VICTORY_SDWA_SEXT`, upstream pull request #3124), and taken as
/// before with the switch off.
#[test]
fn sdwa_16bit_reserved_dst_unused_is_refused() {
    let was = super::TEST_SDWA_SEXT.replace(Some(true));
    for dst_sel in [4u32, 5] {
        // v_cvt_i16_f16_sdwa v1, v0 dst_sel:{WORD_0,WORD_1} dst_unused:3 (reserved);
        // ds_write_b32 v0, v1 to keep v1 live.
        let sdwa_ext = (dst_sel << 8) | (3 << 11) | (4 << 16);
        let words = [0x7E02_A6F9, sdwa_ext, 0xD834_0000, 0x0000_0100, S_ENDPGM];
        let err = lowers(&words, 1 << 15).expect_err("reserved dst_unused must fail");
        assert!(
            matches!(err, crate::ShaderError::Unsupported { what: "an SDWA destination of a 16-bit operation", .. }),
            "unexpected error for dst_sel={dst_sel}: {err}"
        );
        super::TEST_SDWA_SEXT.set(Some(false));
        assert!(lowers(&words, 1 << 15).is_ok(), "VICTORY_SDWA_SEXT=0: taken as before (dst_sel={dst_sel})");
        super::TEST_SDWA_SEXT.set(Some(true));
    }
    super::TEST_SDWA_SEXT.set(was);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

#[test]
fn a_mask_joined_with_a_number_requires_the_console_wave_layout() {
    let mut words = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // skip the overwrite
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        S_ENDPGM,
    ];
    let emit = |words: &[u32]| {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 16,
            layout: vec![
                RootParam {
                    sgprs: 4,
                    ..root(RootType::Constants, 0, 0)
                },
                root(RootType::Table, 4, 1),
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
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap()
    };
    let wave32 = emit(&words);
    assert!(wave32.wave_ops);
    assert_eq!(wave32.wave_size, Some(32));

    words[4] = 0xBE94_0387; // join two ordinary numbers
    let scalars = emit(&words);
    assert!(!scalars.wave_ops);
    assert_eq!(scalars.wave_size, None);
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(p.blocks.iter()
        .flat_map(|b| &b.insts)
        .any(|i| i.name() == "v_perm_b32"));

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    }).unwrap();

    let container =
        crate::Container::parse(&emitted.dxil).unwrap();

    let features =
        container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// A buffer descriptor read through the resource heap's own root parameter
/// at a byte offset of `offset` (21 bits, signed) from the heap's start, and
/// stored through. With [`super::heap_below_zero`] a negative offset (here
/// -0x40) gives slot 0 by a signed compare and a select, so its DXIL differs
/// from the plain shift's; a positive one is the shift either way
/// (JordanFromIT/CGBJ-VIctory fa3aee394's case: 0x40 slot 2, 0xe0 slot 7,
/// 0x1fffc0 below the start).
#[test]
fn a_descriptor_below_the_heap_start_is_slot_zero() {
    let heap = RootParam {
        root: RootType::ResourceHeap,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 2,
        slot: 0,
    };
    let lowered = |offset: u32, on: bool| {
        let words = [
            0xBE83_0380u32, // s_mov_b32 s3, 0
            0xF408_0401,    // s_load_dwordx4 s[16:19], s[2:3], offset
            0xFA00_0000 | offset,
            0xE070_2000,
            0x8004_0000, // buffer_store_dword v0, v0, s[16:19], 0 idxen
            S_ENDPGM,
        ];
        super::with_heap_below_zero(on, || lower_with(&words, vec![heap], vec![]))
            .unwrap()
            .dxil
    };
    for offset in [0x40, 0xe0, 0x1f_ffc0] {
        // Each form lowers; the same switch gives the same code.
        assert_eq!(lowered(offset, true), lowered(offset, true));
        assert_eq!(lowered(offset, false), lowered(offset, false));
    }
    // The fix adds the compare and select: the DXIL changes with it, for
    // any offset (the offset is a run-time sum of the scalar base).
    assert_ne!(lowered(0x1f_ffc0, true), lowered(0x1f_ffc0, false));
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn scalar_bit_searches_lower() {
    // The searches and zero counts (TuxBox 1a50eb9): each result is
    // compared and branched on, so the value has to be built.
    let tail = [
        0xBF07_8010, // s_cmp_lg_u32 s16, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let heads: [(&[u32], &str); 3] = [
        (&[0xBE90_1600], "s_flbit_i32_b64"), // s16, s[0:1]
        (&[0xBE90_1200], "s_ff0_i32_b64"),   // s16, s[0:1]
        (&[0xBE90_1100], "s_ff0_i32_b32"),   // s16, s0
    ];
    for (head, name) in heads {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
        // The encodings above name s16 as the destination and s0 as the
        // source; a two-dword source starts at the even register 0.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 256).unwrap();
        let inst = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == name)
            .unwrap_or_else(|| panic!("{name} decoded"));
        assert!(inst.write(crate::ir::Reg::S(16)).is_some(), "{name}");
        assert!(inst.read(crate::ir::Reg::S(0)).is_some(), "{name}");
    }
}

#[test]
#[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn first_lane_read_of_uniform_lds_load_lowers_on_wave32() {
    // Wave64 compute program lowered for a 32-lane host: inside a divergent
    // branch (`v0 < 1`), read LDS at a uniform address (`v1 = s16`), add a
    // scalar (`v8 = s17 + v2`), and read the first lane into `s17`.
    // When `ds_read_b32` reads from the uniform `v1`, `v8` is wave-uniform
    // and `v_readfirstlane_b32` needs no cross-half exchange. When it reads
    // from per-lane `v0`, the divergent read must still be refused.
    let make = |addr_vgpr: u32| -> Vec<u32> {
        vec![
            0xBE90_0380,            // s_mov_b32 s16, 0
            0xBE91_0380,            // s_mov_b32 s17, 0
            0xD760_0004,            // v_readlane_b32 s4, v0, 0 (triggers wave64 exchange)
            0x0001_0100,
            0x7D88_0081,            // v_cmp_gt_u32_e32 vcc, 1, v0
            0xBE86_246A,            // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0005,            // s_cbranch_execz -> restore exec
            0x7E02_0210,            // v_mov_b32 v1, s16
            0xD8D8_0000,            // ds_read_b32 v2, v{addr}
            0x0200_0000 | addr_vgpr,
            0x4A10_0411,            // v_add_nc_u32 v8, s17, v2
            0x7E22_0508,            // v_readfirstlane_b32 s17, v8
            0xBEFE_0406,            // s_mov_b64 exec, s[6:7]
            0x8011_0411,            // s_add_u32 s17, s17, s4 (keep s17 and s4 live)
            0x7E06_0211,            // v_mov_b32 v3, s17
            0xD834_0000,            // ds_write_b32 v0, v3
            0x0000_0300,
            S_ENDPGM,
        ]
    };
    let run = |words: &[u32]| {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15, // 512 bytes LDS
                threads: [128, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let prev = super::TEST_HOST_WAVE.replace(Some(32));
        let out = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        super::TEST_HOST_WAVE.set(prev);
        out
    };

    if let Err(e) = run(&make(1)) {
        panic!("uniform ds_read_b32 + scalar add must be recognised as wave-uniform, got: {e}");
    }
    let err = run(&make(0))
        .err()
        .expect("per-lane ds_read_b32 must be refused")
        .to_string();
    assert!(
        err.contains("a first-lane read where not every lane runs"),
        "per-lane ds_read_b32 must still require all lanes, got: {err}"
    );
}

/// An LDS write and a read of another lane's word in one lane branch: two host waves
/// cannot be kept in the console's order there, so `lds_meetings` refuses it, unless
/// [`super::with_lds_regions_approx`] asks for the meetings around the region anyway.
#[test]
fn an_lds_write_and_read_in_one_lane_branch_meet_around_it_only_when_approximate() {
    let words: [u32; 13] = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x3A04_02FF,
        0x0000_0080, // v_xor_b32 v2, 128, v1 (lane t ^ 32's byte)
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0005, // s_cbranch_execz to the restore of EXEC
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0000,
        0x0300_0002, // ds_read_b32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 8).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let all = super::all_lane_blocks(&st).expect("all-lane blocks");
    assert!(all.iter().any(|a| !a), "the branch's blocks are not run by every lane");
    assert!(super::lds_meetings(&p, &st, &all).is_none(), "refused as it is");
    let meet = super::with_lds_regions_approx(|| super::lds_meetings(&p, &st, &all))
        .expect("meetings when approximate");
    assert!(meet.iter().any(|m| m.0 || m.1), "the host waves meet around the region");
    assert!(super::lds_meetings(&p, &st, &all).is_none(), "and refused again after it");
}

/// LDS atomics in one lane branch: a minimum then an add to the same word give 33 on the
/// console with 33 lanes, but 1 or 32 when one host wave gets through both before the other
/// starts. An atomic reads the word too, so the region is refused there (and the program takes
/// a form with the atomics where every lane runs); one kind of reduction without results is
/// not. Ported from avatarmanz/VIctory-7900XTX PR #204 (feanorsun, 5e200817b).
#[test]
fn lds_atomics_in_one_lane_branch_are_reads_and_writes() {
    const fn ds(op: u32, vdst: u32, addr: u32, data0: u32) -> [u32; 2] {
        [0x36 << 26 | op << 18, vdst << 24 | data0 << 8 | addr]
    }
    let program = |first: u32, second: u32| {
        let mut w = vec![
            0x7E12_0280, // v_mov_b32 v9, 0 (address and minimum)
            0x7E04_0281, // v_mov_b32 v2, 1 (increment)
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0006, // s_cbranch_execz past both atomics
        ];
        w.extend(ds(first, 0, 9, 9));
        w.extend([
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBF85_0002, // s_cbranch_scc1 past the second
        ]);
        w.extend(ds(second, 0, 9, 2));
        w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
        w.extend(ds(0x36, 5, 9, 0)); // ds_read_b32 v5, v9
        w.push(S_ENDPGM);
        let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        ir::build(&code, true, 16).expect("ir::build")
    };
    let meets = |p: &ir::Program| {
        let st = structure::structure(p).expect("structure");
        let all = super::all_lane_blocks(&st).expect("all-lane blocks");
        let names: Vec<&str> = p
            .blocks
            .iter()
            .zip(&all)
            .filter(|(_, full)| !**full)
            .flat_map(|(b, _)| &b.insts)
            .map(|i| i.name())
            .filter(|n| n.starts_with("ds_"))
            .collect();
        (names.join(" "), super::lds_meetings(p, &st, &all).is_some())
    };
    let off = |p: &ir::Program| {
        let was = super::TEST_LDS_ATOMICS_READ.replace(Some(false));
        let r = meets(p);
        super::TEST_LDS_ATOMICS_READ.set(was);
        r.1
    };
    // ds_min_u32 then ds_add_u32: refused; taken with the switch off (the race)
    let p = program(0x07, 0x00);
    assert_eq!(meets(&p), ("ds_min_u32 ds_add_u32".into(), false));
    assert!(off(&p));
    // ds_min_u32 then ds_add_rtn_u32 (the old value back: a slot claimed): refused
    let p = program(0x07, 0x20);
    assert_eq!(meets(&p), ("ds_min_u32 ds_add_rtn_u32".into(), false));
    // ds_add_rtn_u32 then ds_write_b32: an atomic and a store, refused
    let p = program(0x20, 0x0D);
    assert_eq!(meets(&p), ("ds_add_rtn_u32 ds_write_b32".into(), false));
    assert!(off(&p));
    // two ds_add_u32 (one kind of reduction, no result): any order gives the same word
    let p = program(0x00, 0x00);
    assert_eq!(meets(&p), ("ds_add_u32 ds_add_u32".into(), true));
    // ds_max_u32 then ds_add_u32: two kinds, refused
    let p = program(0x08, 0x00);
    assert!(!meets(&p).1);
    // two ds_write_b32: write-only as before
    let p = program(0x0D, 0x0D);
    assert_eq!(meets(&p), ("ds_write_b32 ds_write_b32".into(), true));
}

/// A lane loop (one block, left per lane on `s_cbranch_execnz`): after the first round only
/// the lanes still in it run its block, so it is not a block every lane runs, though it
/// post-dominates its own branch.
#[test]
fn a_lane_loop_is_not_a_block_every_lane_runs() {
    let words: [u32; 9] = [
        0xBE82_0480, // s_mov_b64 s[2:3], 0
        0x4A00_0081, // 4: v_add_nc_u32 v0, 1, v0
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0x8882_026A, // s_or_b64 s[2:3], vcc, s[2:3]
        0x8AFE_027E, // s_andn2_b64 exec, exec, s[2:3]
        0xBF89_FFFB, // s_cbranch_execnz -> 4
        0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
        0x7E02_0280, // v_mov_b32 v1, 0
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 8).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let all = super::all_lane_blocks(&st).expect("all-lane blocks");
    assert!(matches!(st.exits[1], structure::LaneExit::Branch { taken: 1, not_taken: 2, .. }), "{:?}", st.exits);
    assert_eq!(all, [true, false, true], "the entry, the loop, after the loop");
    let was = super::TEST_LANE_LOOP_FIX.replace(Some(false));
    let before = super::all_lane_blocks(&st).expect("all-lane blocks");
    super::TEST_LANE_LOOP_FIX.set(was);
    assert_eq!(before, [true, true, true], "VICTORY_LANE_LOOP_FIX=0: as before");
}

#[test]
fn lane_uniform_handles_v255_and_high_pair_operands() {
    // Regression test for operands ending at v255 (exclusive end 256):
    // 1. A single-word uniform v255 must still be recognised as wave-uniform.
    // 2. A 64-bit pair v[254:255] with uniform v254 and per-lane v255 shifted
    //    right by 32 copies v255 into v8 and must NOT be classified as uniform.
    let run = |body: &[u32]| {
        let mut words = vec![
            0xBE90_0380, // s_mov_b32 s16, 0
            0xBE91_0380, // s_mov_b32 s17, 0
            0xD760_0004, // v_readlane_b32 s4, v0, 0 (triggers wave64 exchange)
            0x0001_0100,
            0x7D88_0081, // v_cmp_gt_u32_e32 vcc, 1, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0000 | (body.len() as u32 + 1), // s_cbranch_execz -> restore exec
        ];
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0x7E22_0508, // v_readfirstlane_b32 s17, v8
            0xBEFE_0406, // s_mov_b64 exec, s[6:7]
            0x8011_0411, // s_add_u32 s17, s17, s4
            0x7E06_0211, // v_mov_b32 v3, s17
            0xD834_0000, // ds_write_b32 v0, v3
            0x0000_0300,
            S_ENDPGM,
        ]);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [128, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let prev = super::TEST_HOST_WAVE.replace(Some(32));
        let out = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        super::TEST_HOST_WAVE.set(prev);
        out
    };

    // Uniform v255 -> uniform v8 -> v_readfirstlane_b32 lowers on wave32.
    let uniform_v255 = [
        0x7FFE_0210, // v_mov_b32 v255, s16
        0x4A11_FE11, // v_add_nc_u32 v8, s17, v255
    ];
    if let Err(e) = run(&uniform_v255) {
        panic!("uniform v255 read must be recognised as wave-uniform, got: {e}");
    }

    // Uniform v254 + per-lane v255 -> v_lshrrev_b64 v[8:9], 32, v[254:255] -> per-lane v8.
    let varying_high = [
        0x7FFC_0210, // v_mov_b32 v254, s16
        0x7FFE_0300, // v_mov_b32 v255, v0
        0xD700_0008, // v_lshrrev_b64 v[8:9], 32, v[254:255]
        0x0003_FCA0,
    ];
    let err = run(&varying_high)
        .err()
        .expect("varying v255 high word must not be classified as uniform")
        .to_string();
    assert!(
        err.contains("a first-lane read where not every lane runs"),
        "varying v255 high word must require all lanes, got: {err}"
    );
}

#[test]
fn single_thread_lds_omits_wave_size() {
    // v_mov_b32 v5, 0; ds_write_b32 v5, v1; s_endpgm
    let words: &[u32] = &[
        0x7E0A_0280,
        0xD834_0000,
        0x0000_0105,
        0xBF81_0000,
    ];
    let lower_lds = |threads: u32| {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15, // 512 bytes of LDS
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = crate::structure::structure(&p).expect("structure");
        let res =
            crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = crate::io::raise(&p, &native).expect("io::raise");
        lower(&Parts {
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
        .expect("lower")
    };
    assert_eq!(lower_lds(1).wave_size, None);
    assert_eq!(lower_lds(2).wave_size, Some(32));
}

#[test]
fn only_latches_reached_through_a_loop_exit_lose_their_abort() {
    use crate::structure::{Bool, LaneExit::*};

    // Header 1 sends some lanes straight through latch 4. Only paths to
    // latch 3 must pass the natural exit in block 2.
    let branch = |taken, not_taken| Branch { when: Bool::TRUE, taken, not_taken };
    let exits = [Jump(1), branch(2, 4), branch(5, 3), Jump(1), Jump(1), End];
    let limit = super::limit_exits(&[0, 1, 4, 2, 3, 5], &exits);
    assert_eq!(limit.plain, std::collections::HashSet::from([3]));
    assert_eq!(limit.tests[&2], [1]);
    assert!(!limit.bodies[&1].contains(&5));
}

#[test]
fn an_endless_loop_keeps_its_iteration_limit_in_the_latch() {
    use crate::structure::LaneExit::Jump;
    let limit = super::limit_exits(&[0, 1], &[Jump(1), Jump(1)]);
    assert!(limit.plain.is_empty());
    assert!(limit.tests.is_empty());
}

#[test]
fn loop_exits_is_on_unless_switched_off() {
    // Unset (the default) and any value but "0" are on; the test hook stands in for "0".
    let was = super::TEST_LOOP_EXITS.replace(None);
    assert_eq!(super::loop_exits(), std::env::var_os("VICTORY_LOOP_EXITS").is_none_or(|v| v != "0"));
    super::TEST_LOOP_EXITS.set(Some(false));
    assert!(!super::loop_exits(), "VICTORY_LOOP_EXITS=0: the latch test as before");
    super::TEST_LOOP_EXITS.set(Some(true));
    assert!(super::loop_exits());
    super::TEST_LOOP_EXITS.set(was);
}
