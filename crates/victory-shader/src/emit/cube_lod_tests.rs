//! A cube sampled with explicit gradients selects its level by the footprint in texels
//! ([`super::lower::cube_grad_lod`]). After DexNom/VIctory 921d8d3b4's test (GPL-3.0).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};
use windows::Win32::Graphics::Direct3D12::*;

/// Every mip of the cube holds its own level number, so a trilinear sample returns the level
/// it took. `image_sample_d` with a footprint of 1, 2 and 16 texels takes levels 0, 1 and 4
/// (and 0 for none); `image_sample_d_cl` takes at least its minimum level; `image_sample_l`
/// and `_lz` take theirs. Without [`super::lower::cube_grad_lod`] a footprint of 2 took level
/// 1.44.
#[test]
fn a_cube_gradient_selects_the_level_of_its_footprint() {
    const SIZE: usize = 64;
    const MIPS: u16 = 7;
    let mut words = vec![
        0xBE81_0380,
        0xBE83_0380, // s1, s3 = 0 (upper halves of the tables)
        0xF40C_0400,
        0xFA00_0000, // cube SRV to s[16:23] from s[0:1]
        0xF408_0300,
        0xFA00_0020, // raw output UAV to s[12:15] from s[0:1]+32
        0xF408_0601,
        0xFA00_0000, // sampler to s[24:27] from s[2:3]
        0x7E1E_0280, // v15 = 0 for output byte offsets
    ];
    // (opcode, footprint in texels, level given, level wanted)
    let cases = [
        (0x22, 1.0, 0.0, 0.0),
        (0x22, 2.0, 0.0, 1.0),
        (0x22, 16.0, 0.0, 4.0),
        (0x22, 0.000001, 0.0, 0.0),
        (0x22, 0.0, 0.0, 0.0),
        (0x23, 1.0, 3.0, 3.0),
        (0x23, 16.0, 2.0, 4.0),
        (0x23, 0.0, 2.0, 2.0),
        (0x24, 0.0, 2.0, 2.0),
        (0x27, 0.0, 0.0, 0.0),
    ];
    for (i, &(op, footprint, lod, _)) in cases.iter().enumerate() {
        let delta = footprint / SIZE as f32;
        let mut addr = Vec::new();
        if op == 0x22 || op == 0x23 {
            addr.extend([delta, 0.0, 0.0, delta]); // gradients along the face
        }
        addr.extend([1.5f32, 1.5, 0.0]); // positive X face, centre
        if op == 0x23 || op == 0x24 {
            addr.push(lod);
        }
        for (reg, value) in addr.iter().enumerate() {
            words.extend([0x7E00_02FF | ((reg as u32) << 17), value.to_bits()]);
        }
        // MIMG, cube, dmask R, inputs from v0, SRV s16, sampler s24, result v10
        words.extend([0xF000_0118 | (op << 18), 0x00C4_0A00]);
        // buffer_store_dword v10, v15, s[12:15], 0 offen offset:i*4
        words.extend([0xE070_1000 | (i as u32 * 4), 0x8003_0A0F]);
    }
    words.push(0xBF81_0000); // s_endpgm
    let table = |slot, user_data| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let range = |kind, at, words, slot, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 0,
        dwords: words,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![table(0, 0), table(1, 2)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0, 0),
            range(ResourceType::Uav, 8, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 1, 2),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::with_cube_grad_lod(true, || {
        lower::lower(&lower::Parts {
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
    });
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
        half_views: lowered.half_views,
        group_split: 1,
    };
    let path = validator::Validator::default_path();
    if let Ok(mut v) = validator::Validator::load(path.as_deref()) {
        let mut dxil = shader.dxil.clone();
        v.validate(&mut dxil).expect("DXIL validation failed");
    }
    let texels: Vec<_> = (0..6)
        .flat_map(|_| {
            (0..MIPS).flat_map(|mip| {
                std::iter::repeat_n([f32::from(mip); 4], (SIZE >> mip).pow(2))
            })
        })
        .collect();
    let sampler = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxLOD: f32::MAX,
        MaxAnisotropy: 1,
        ..Default::default()
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords() as usize]))],
                    views: &[
                        View::Cube(SIZE as u32, MIPS, &texels),
                        View::RawUav(cases.len() as u32),
                    ],
                    samplers: &[sampler],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let levels: Vec<f32> = out[0].iter().copied().map(f32::from_bits).collect();
        let wanted = cases.map(|(_, _, _, level)| level);
        assert_eq!(levels.as_slice(), wanted, "{name}");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
