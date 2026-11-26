//! Everything specific to the title: its container formats, build
//! identities, and the hooks we place in its code (located by per-build
//! offsets or byte-pattern signatures). What may and may not be stored here
//! is defined in docs/asset-policy.md.

pub mod sgd5;

/// A specific build of the title executable we know how to run.
#[derive(Debug, Clone, Copy)]
pub struct KnownBuild {
    /// Our own label, e.g. "1.0.0 retail".
    pub label: &'static str,
    /// BLAKE3 of the original executable, hex.
    pub blake3: &'static str,
    /// Changes to its code, made in memory once it is mapped: the
    /// upscaler choice (see [`KNOWN_BUILDS`]).
    pub patches: &'static [CodePatch],
    /// Where its own upscaler's render scale is set (`VICTORY_RENDER_SCALE`).
    pub render_scale: Option<RenderScale>,
    /// Its lighting made to take its own programs without rays
    /// (`VICTORY_RT_FALLBACK=1`, [`KNOWN_BUILDS`]).
    pub rt_fallback: &'static [CodePatch],
    /// Its render feature mask without the two ray-tracing structures
    /// (`VICTORY_RT_FALLBACK=2`, with [`KnownBuild::rt_fallback`]).
    pub features_nort: &'static [CodePatch],
    /// Its hang watchdog made harmless (`VICTORY_NO_WATCHDOG=1`).
    pub watchdog: &'static [CodePatch],
    /// Its table of output sizes (`VICTORY_RESOLUTION`).
    pub resolution: Option<ResolutionLadder>,
}

/// The title's table of output sizes (the resolution "ladder"): five entries
/// of `width, height, 60, 1` (16 bytes each) at `table`, the largest the
/// console's. It picks its swap chain and render targets from it, so capping
/// the entries above a size makes it render at that size.
///
/// `budget_check` is the debug assertion after its render targets are made
/// (`cmp rax, 0x1fffff; jbe +1; int3`: what one memory scope of theirs used,
/// plus a running total, must stay below 2 MiB). With a capped table it
/// fires (soldsawn/victory-nvidia-8gb's runs); its `int3` becomes a `nop`.
/// It reads two stack values and one data word, nothing of the title's code
/// and no clock: an assertion, not an integrity check.
#[derive(Debug, Clone, Copy)]
pub struct ResolutionLadder {
    pub table: u32,
    pub sizes: [(u32, u32); 5],
    pub budget_check: CodePatch,
}

/// The places of a build's own upscaler that bound its render scale: the
/// minimum, an `f32` immediate (`expect`, 2/3 in the known build) at
/// `minimum` that becomes the scale asked for, and the instruction that
/// raises it to the maximum (`vmaxss`), made a plain load of the minimum
/// (`vmovss`), so the scale is fixed.
#[derive(Debug, Clone, Copy)]
pub struct RenderScale {
    pub minimum: u32,
    pub minimum_expect: [u8; 4],
    pub maximum: CodePatch,
}

/// A change to title code computed when it is made ([`render_scale_patches`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedPatch {
    pub name: &'static str,
    pub rva: u32,
    pub expect: Vec<u8>,
    pub write: Vec<u8>,
}

/// The scale `VICTORY_RENDER_SCALE` asks for, if it is one the title's own
/// upscaler takes (0.2 to 1 of the output size).
pub fn parse_render_scale(text: &str) -> Option<f32> {
    let v: f32 = text.trim().parse().ok()?;
    (0.2..=1.0).contains(&v).then_some(v)
}

/// The patches that fix `build`'s render scale at `scale` (none if it has
/// no known places). From vitutu1/victory-perf (`crates/victory-loader/src/
/// hooks.rs`, ported there from ZigZagBoi1/ScarlettPort; GPL-3.0): its own
/// upscaler's two places; FSR's are not needed, since the title keeps its
/// own upscaler ([`KNOWN_BUILDS`]' upscaler choice).
pub fn render_scale_patches(build: &KnownBuild, scale: f32) -> Vec<OwnedPatch> {
    let Some(r) = build.render_scale else {
        return Vec::new();
    };
    vec![
        OwnedPatch {
            name: "render scale: own upscaler minimum",
            rva: r.minimum,
            expect: r.minimum_expect.to_vec(),
            write: scale.to_le_bytes().to_vec(),
        },
        OwnedPatch {
            name: r.maximum.name,
            rva: r.maximum.rva,
            expect: r.maximum.expect.to_vec(),
            write: r.maximum.write.to_vec(),
        },
    ]
}

/// A size `VICTORY_RESOLUTION=<w>x<h>` asks for, if it is one the title's
/// table can be capped at: from 1280x720 to 3840x2160 (below 1920x1080
/// soldsawn saw FSR's largest upscale size refuse it on a Series S).
pub fn parse_resolution(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.trim().split_once(['x', 'X'])?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    ((1280..=3840).contains(&w) && (720..=2160).contains(&h)).then_some((w, h))
}

/// The patches that cap `build`'s table of output sizes at `w`x`h`: every
/// entry with more pixels becomes `w, h, 60, 1`, and the budget assertion
/// that then fires is stepped over. None if no entry is larger, or the build
/// has no known table. From soldsawn/victory-nvidia-8gb 037ef5893 (DKTTT,
/// `crates/victory-game/src/output.rs`, GPL-3.0), there found by pattern in
/// an image made writable whole; here at this build's places, each checked.
pub fn resolution_patches(build: &KnownBuild, w: u32, h: u32) -> Vec<OwnedPatch> {
    let Some(r) = build.resolution else {
        return Vec::new();
    };
    let entry = |w: u32, h: u32| [w, h, 60, 1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let target = u64::from(w) * u64::from(h);
    let mut patches: Vec<OwnedPatch> = (0u32..)
        .zip(r.sizes)
        .filter(|&(_, (lw, lh))| u64::from(lw) * u64::from(lh) > target)
        .map(|(i, (lw, lh))| OwnedPatch {
            name: "resolution: an output size above the cap",
            rva: r.table + 16 * i,
            expect: entry(lw, lh),
            write: entry(w, h),
        })
        .collect();
    if !patches.is_empty() {
        patches.extend(owned(&[r.budget_check]));
    }
    patches
}

/// What `VICTORY_RT_FALLBACK` asks for: `1` the set test alone, `2` it and
/// the feature mask; anything else nothing.
pub fn parse_rt_fallback(text: &str) -> Option<u8> {
    match text.trim() {
        "1" => Some(1),
        "2" => Some(2),
        _ => None,
    }
}

/// The patches `VICTORY_RT_FALLBACK=<level>` makes in `build` (see
/// [`KNOWN_BUILDS`]): the mask never without the set test.
pub fn rt_fallback_patches(build: &KnownBuild, level: u8) -> Vec<OwnedPatch> {
    let mut patches = owned(build.rt_fallback);
    if level >= 2 && !patches.is_empty() {
        patches.extend(owned(build.features_nort));
    }
    patches
}

/// A build's fixed patches as [`OwnedPatch`]es.
pub fn owned(patches: &[CodePatch]) -> Vec<OwnedPatch> {
    patches
        .iter()
        .map(|p| OwnedPatch { name: p.name, rva: p.rva, expect: p.expect.to_vec(), write: p.write.to_vec() })
        .collect()
}

/// A change to title code: at `rva`, the bytes `expect` (a short signature
/// checked before anything is written) become `write`, our own code.
#[derive(Debug, Clone, Copy)]
pub struct CodePatch {
    /// Our name for it, as logged.
    pub name: &'static str,
    pub rva: u32,
    pub expect: &'static [u8],
    pub write: &'static [u8],
}

/// `mov eax, 6; ret`: the upscaler a Series X uses.
const UPSCALER_6: &[u8] = &[0xb8, 0x06, 0x00, 0x00, 0x00, 0xc3];

/// Builds we support.
pub const KNOWN_BUILDS: &[KnownBuild] = &[KnownBuild {
    label: "Xbox Series master",
    blake3: "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93",
    // The title picks its upscaler by console: on a Series S with the
    // FidelityFX upscaler's capability, FSR 4 (type 7), else its own
    // (type 6, as on a Series X). FSR 4's output never reaches the screen
    // here (every world frame is black with `--console series-s`), and
    // without its DLL the title divides by zero. Both functions that make
    // the choice return 6, so a Series S runs its own render profile (about
    // 30% fewer milliseconds a frame than a Series X) with the upscaler that
    // works.
    patches: &[
        CodePatch {
            name: "upscaler choice",
            rva: 0x7e_93f0,
            expect: &[0x48, 0x83, 0xec, 0x28, 0x80, 0x3d],
            write: UPSCALER_6,
        },
        CodePatch {
            name: "upscaler choice (the per-frame copy)",
            rva: 0x540_0190,
            expect: &[0x48, 0x83, 0xec, 0x28, 0x48, 0x8b],
            write: UPSCALER_6,
        },
    ],
    render_scale: Some(RenderScale {
        minimum: 0x28f_1e7e,
        // 2/3
        minimum_expect: [0xab, 0xaa, 0x2a, 0x3f],
        maximum: CodePatch {
            name: "render scale: own upscaler maximum = minimum",
            rva: 0x28f_1e87,
            // vmaxss xmm0, xmm0, [rsi+0x90] -> vmovss xmm0, [rsi+0x90]
            expect: &[0xc5, 0xfa, 0x5f, 0x86, 0x90, 0, 0, 0],
            write: &[0xc5, 0xfa, 0x10, 0x86, 0x90, 0, 0, 0],
        },
    }),
    // The lighting picks its programs by one test, "is acceleration-
    // structure set i of this manager there" (+0x4fd8570: `mov rax,
    // [rcx+8]; mov edx, edx; cmp qword [rax+rdx*8], 0; jne +0x4fd8590`, else
    // `xor eax, eax; ret`), asked for the lighting manager through
    // +0x4fd8550 (45 calls) and for the shadow manager through +0x49be10 (9),
    // and called directly 14 more times. A yes takes the `_BVH` programs
    // (CS_ScreenProbes_TraceRays_RayBinning_BVH_*), a no the title's own
    // without rays (CS_ScreenProbes_TraceRays_Fallback,
    // CS_PPR_RaytraceFallback, CS_LaserLightShadowRaytracingFallback) and
    // leaves out the irradiance-cache and reflection trace dispatches. Our
    // card has no structure the console's builder made, so its rays find
    // nothing. The `jne` becomes a 6-byte `nop`: every set reads as absent.
    // SonicRaT/XNW `apps/xnw/src/patches.rs` `rtfallback` (2fa40fef6,
    // GPL-3.0); same place and bytes in this build (PE time stamp
    // 0x6af783d4, read from the executable).
    rt_fallback: &[CodePatch {
        name: "rt fallback: no acceleration-structure set is there",
        rva: 0x4fd_857b,
        // jne +0x4fd8590
        expect: &[0x0f, 0x85, 0x0f, 0x00, 0x00, 0x00],
        // nop word [rax+rax+0]
        write: &[0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00],
    }],
    // The render feature mask (`mov eax, 0xde4f`, stored to two words by
    // +0x1ca31f0, a function reached once by a jump) without bits 0x4 and
    // 0x8: the near and far ray-tracing structures, so none is made. XNW's
    // `features_nort` (same source): which programs then run is unproven
    // there.
    features_nort: &[CodePatch {
        name: "rt fallback: render feature mask without the ray-tracing structures",
        rva: 0x1ca_31f0,
        // mov eax, 0xde4f -> mov eax, 0xde43
        expect: &[0xb8, 0x4f, 0xde, 0x00, 0x00],
        write: &[0xb8, 0x43, 0xde, 0x00, 0x00],
    }],
    // The title's hang watchdog (thread function +0x2215a60): every second
    // it compares each registered thread's last beat with a limit (`>=
    // 300.0` scaled ticks, about 30 s) and, past it, calls a handler kept
    // in a data word, which writes to address 0 on purpose (`mov dword
    // [0], 0x4d8bd421` at +0x23cf683), and after it does the same itself
    // (`mov dword [0], 0; int3` at +0x2215c35): a deliberate crash for the
    // console's crash report, not a protection. The load of the handler
    // becomes a jump to the loop's next round (+0x2215c41), and the
    // handler's own write a `nop`. trustno1-coder/VIctory-NVIDIA afd84de12
    // (`crates/victory-loader/src/hooks.rs`, from vitutu1/victory-perf, from
    // ZigZagBoi1/ScarlettPort `runtime/victory`; GPL-3.0-or-later), also
    // XNW's `watchdog`; same places and bytes here.
    watchdog: &[
        CodePatch {
            name: "no watchdog: skip the handler and the deliberate fault",
            rva: 0x221_5c27,
            // mov rax, [rip+0x4fa071a] (the handler)
            expect: &[0x48, 0x8b, 0x05, 0x1a, 0x07, 0xfa, 0x04],
            // jmp +0x18 (to +0x2215c41); nops
            write: &[0xeb, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
        },
        CodePatch {
            name: "no watchdog: the handler's deliberate write to 0",
            rva: 0x23c_f683,
            // mov dword [0], 0x4d8bd421
            expect: &[0xc7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xd4, 0x8b, 0x4d],
            // an 11-byte nop
            write: &[0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
        },
    ],
    // In .rdata; its sizes are the console's own list of output modes.
    resolution: Some(ResolutionLadder {
        table: 0x615_c0c0,
        sizes: [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)],
        budget_check: CodePatch {
            name: "resolution: step over the render-target budget assertion",
            rva: 0x247_af25,
            // cmp rax, 0x1fffff; jbe +1; int3 -> ...; nop
            expect: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc],
            write: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0x90],
        },
    }),
}];

/// Looks up a build by the BLAKE3 hash of its executable.
pub fn identify(blake3_hex: &str) -> Option<&'static KnownBuild> {
    KNOWN_BUILDS
        .iter()
        .find(|b| b.blake3.eq_ignore_ascii_case(blake3_hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_builds_are_well_formed() {
        for build in KNOWN_BUILDS {
            assert_eq!(build.blake3.len(), 64);
            assert!(build.blake3.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(
                identify(&build.blake3.to_uppercase()).map(|b| b.label),
                Some(build.label)
            );
            let fixed = [build.patches, build.rt_fallback, build.features_nort, build.watchdog];
            let check = build.resolution.map(|r| r.budget_check);
            for patch in fixed.iter().flat_map(|p| p.iter()).chain(check.iter()) {
                assert!(!patch.write.is_empty(), "{}", patch.name);
                assert_eq!(patch.expect.len(), patch.write.len(), "{}", patch.name);
                assert_ne!(patch.expect, patch.write, "{}", patch.name);
            }
        }
        assert!(identify("00").is_none());
    }

    #[test]
    fn render_scale_patches_fix_the_scale() {
        assert_eq!(parse_render_scale("0.5"), Some(0.5));
        assert_eq!(parse_render_scale(" 1 "), Some(1.0));
        assert_eq!(parse_render_scale("0.1"), None);
        assert_eq!(parse_render_scale("1.5"), None);
        assert_eq!(parse_render_scale("half"), None);
        let build = &KNOWN_BUILDS[0];
        let p = render_scale_patches(build, 0.5);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].write, 0.5f32.to_le_bytes());
        assert_eq!(p[0].expect, (2.0f32 / 3.0).to_le_bytes());
        assert!(p.iter().all(|p| p.write.len() == p.expect.len() && p.write != p.expect));
        let none = KnownBuild { render_scale: None, ..*build };
        assert!(render_scale_patches(&none, 0.5).is_empty());
    }

    #[test]
    fn rt_fallback_levels() {
        assert_eq!(parse_rt_fallback("1"), Some(1));
        assert_eq!(parse_rt_fallback(" 2 "), Some(2));
        assert_eq!(parse_rt_fallback("0"), None);
        assert_eq!(parse_rt_fallback("yes"), None);
        let build = &KNOWN_BUILDS[0];
        let one = rt_fallback_patches(build, 1);
        assert_eq!(one.iter().map(|p| p.rva).collect::<Vec<_>>(), [0x4fd_857b]);
        // The removed jne went to the state test past the "absent" answer.
        assert_eq!(0x4fd_857b + 6 + u32::from_le_bytes(one[0].expect[2..6].try_into().unwrap()), 0x4fd_8590);
        let two = rt_fallback_patches(build, 2);
        assert_eq!(two.iter().map(|p| p.rva).collect::<Vec<_>>(), [0x4fd_857b, 0x1ca_31f0]);
        let none = KnownBuild { rt_fallback: &[], ..*build };
        assert!(rt_fallback_patches(&none, 2).is_empty(), "the mask never goes alone");
    }

    #[test]
    fn the_watchdog_jump_lands_on_the_loop() {
        let w = KNOWN_BUILDS[0].watchdog;
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].write[0], 0xeb);
        assert_eq!(w[0].rva + 2 + u32::from(w[0].write[1]), 0x221_5c41);
    }

    #[test]
    fn resolution_caps_the_larger_sizes() {
        assert_eq!(parse_resolution("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_resolution(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_resolution("640x360"), None);
        assert_eq!(parse_resolution("1920"), None);
        assert_eq!(parse_resolution("7680x4320"), None);
        let build = &KNOWN_BUILDS[0];
        let p = resolution_patches(build, 1920, 1080);
        let table = build.resolution.unwrap().table;
        assert_eq!(p.iter().map(|p| p.rva).collect::<Vec<_>>(), [table + 48, table + 64, 0x247_af25]);
        let size = |b: &[u8]| (u32::from_le_bytes(b[0..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap()));
        assert_eq!(size(&p[0].expect), (2560, 1440));
        assert_eq!(size(&p[1].expect), (3840, 2160));
        assert!(p[..2].iter().all(|p| size(&p.write) == (1920, 1080) && p.write[8..] == p.expect[8..]));
        assert_eq!(*p[2].write.last().unwrap(), 0x90);
        // Nothing larger: no patch, and the assertion stays.
        assert!(resolution_patches(build, 3840, 2160).is_empty());
        let none = KnownBuild { resolution: None, ..*build };
        assert!(resolution_patches(&none, 1920, 1080).is_empty());
    }
}
