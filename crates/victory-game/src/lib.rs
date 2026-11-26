//! Everything specific to the title: its container formats, build
//! identities, and the hooks we place in its code (located by per-build
//! offsets or byte-pattern signatures). What may and may not be stored here
//! is defined in docs/asset-policy.md.

pub mod cpu_sites;
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
            for patch in build.patches {
                assert!(!patch.write.is_empty(), "{}", patch.name);
                assert_eq!(patch.expect.len(), patch.write.len(), "{}", patch.name);
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
}
