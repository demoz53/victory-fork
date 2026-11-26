//! Everything specific to the title: its container formats, build
//! identities, and the hooks we place in its code (located by per-build
//! offsets or byte-pattern signatures). What may and may not be stored here
//! is defined in docs/asset-policy.md.

#[cfg(windows)]
pub mod hooks;
pub mod sgd5;

pub mod cpu_sites;

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
    /// Where its game clocks' largest step is loaded (`VICTORY_TIME_STEP`).
    pub time_step: Option<TimeStepCap>,
    /// The ceiling of its per-frame placement budget for streamed resources
    /// (`VICTORY_PLACE_BUDGET`).
    pub place_budget: Option<PlaceBudget>,
    /// Its per-frame cap of texture upgrades (`VICTORY_TEX_UPGRADES`).
    pub tex_upgrades: Option<TexUpgrades>,
}

/// The largest step the title's game clocks take in a frame: each clock
/// advances by `min(frame time, cap)` (times its time scale; the frame time
/// is a real `QueryPerformanceCounter` difference), and the cap is a float
/// constant shared with dozens of other uses, loaded by `loads` (`vmovss
/// xmm, [rip + disp32]`, 8 bytes each). Raising the cap points those loads
/// at another read-only constant of the image: `values` pairs a cap in
/// milliseconds with the RVA of a constant holding it, in `rdata` (the
/// image's .rdata: RVA and length), checked as mapped before use.
#[derive(Debug, Clone, Copy)]
pub struct TimeStepCap {
    pub loads: &'static [(u32, [u8; 8])],
    pub values: &'static [(u32, u32)],
    pub rdata: (u32, u32),
}

/// The ceiling of the title's per-frame time slice for placing streamed
/// resources: the clamp `min(budget, 1250)` then `max(it, 250)` in
/// microseconds, whose 1250 is two immediates (`mov ecx, 1250` at `mov_ecx`,
/// `cmp eax, 1250` at `cmp_eax`, 5 bytes each).
#[derive(Debug, Clone, Copy)]
pub struct PlaceBudget {
    pub mov_ecx: u32,
    pub cmp_eax: u32,
    pub ceiling: u32,
}

/// The title's cap of texture upgrades a frame (maxTexUpgradesPerFrame, 5):
/// a data word its settings code stores from a settings field (`mov eax,
/// [rsi+0x110]` at `load`, then the store), compared by two `jge` (`jges`,
/// 6 bytes each) that end the frame's upgrades.
#[derive(Debug, Clone, Copy)]
pub struct TexUpgrades {
    pub load: (u32, [u8; 6]),
    pub jges: &'static [(u32, [u8; 6])],
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

/// The cap `VICTORY_TIME_STEP` asks for, in milliseconds: `0.25`, `0.5` or
/// `1` (`1.0`) seconds (the title's own is 1/15 s).
pub fn parse_time_step(text: &str) -> Option<u32> {
    match text.trim() {
        "0.25" => Some(250),
        "0.5" | "0.50" => Some(500),
        "1" | "1.0" => Some(1000),
        _ => None,
    }
}

/// The patches that make `build`'s game clocks take at most `ms` a frame:
/// both loads of the cap (or neither) pointed at the build's constant of
/// that value, after checking that `rdata` (the image's .rdata as mapped,
/// from [`TimeStepCap::rdata`]'s RVA) holds it there. An error says why none
/// can be made. SonicRaT/XNW 2fa40fef6 (`apps/xnw/src/patches.rs`,
/// `timestep4`, `timestep2`, `timestep1`; GPL-3.0): the same places,
/// constants and bytes.
pub fn time_step_patches(build: &KnownBuild, ms: u32, rdata: &[u8]) -> Result<Vec<OwnedPatch>, String> {
    let Some(cap) = build.time_step else {
        return Err("this build has no places for it".to_owned());
    };
    let &(_, constant) = cap.values.iter().find(|(v, _)| *v == ms).ok_or("this build has no constant of that value")?;
    let want = (ms as f32 / 1000.0).to_le_bytes();
    let held = (constant.checked_sub(cap.rdata.0).map(|at| at as usize))
        .and_then(|at| rdata.get(at..at + 4));
    if held != Some(&want[..]) {
        return Err(format!("the constant at +{constant:#x} is not {} s (another build?)", ms as f32 / 1000.0));
    }
    cap.loads
        .iter()
        .map(|&(rva, expect)| {
            let disp = i32::try_from(i64::from(constant) - (i64::from(rva) + 8)).map_err(|_| "the constant is out of reach".to_owned())?;
            let mut write = expect;
            write[4..].copy_from_slice(&disp.to_le_bytes());
            Ok(OwnedPatch { name: "time step: the cap's load", rva, expect: expect.to_vec(), write: write.to_vec() })
        })
        .collect()
}

/// The ceiling `VICTORY_PLACE_BUDGET` asks for, in microseconds: a number
/// from 1250 (the title's own) to 100000, or `1`/`on` for XNW's 30000.
pub fn parse_place_budget(text: &str) -> Option<u32> {
    match text.trim() {
        "1" | "on" => Some(30_000),
        t => t.parse().ok().filter(|us| (1250..=100_000).contains(us)),
    }
}

/// The patches that raise `build`'s placement ceiling to `us`: both
/// immediates of the clamp. SonicRaT/XNW 2fa40fef6 `place_budget`.
pub fn place_budget_patches(build: &KnownBuild, us: u32) -> Vec<OwnedPatch> {
    let Some(b) = build.place_budget else {
        return Vec::new();
    };
    let imm = |op: u8, v: u32| [&[op][..], &v.to_le_bytes()].concat();
    vec![
        OwnedPatch { name: "place budget: the ceiling (mov ecx)", rva: b.mov_ecx, expect: imm(0xb9, b.ceiling), write: imm(0xb9, us) },
        OwnedPatch { name: "place budget: the ceiling (cmp eax)", rva: b.cmp_eax, expect: imm(0x3d, b.ceiling), write: imm(0x3d, us) },
    ]
}

/// What `VICTORY_TEX_UPGRADES` asks for: a cap from 1 to 100000 a frame, or
/// `unlimited`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexUpgradeCap {
    Most(u32),
    Unlimited,
}

pub fn parse_tex_upgrades(text: &str) -> Option<TexUpgradeCap> {
    match text.trim() {
        "unlimited" => Some(TexUpgradeCap::Unlimited),
        t => t.parse().ok().filter(|n| (1..=100_000).contains(n)).map(TexUpgradeCap::Most),
    }
}

/// The patches for `VICTORY_TEX_UPGRADES`: a number becomes the value the
/// title's settings code stores as its cap (`mov eax, <n>; nop` in place of
/// the load of the settings field, so every later store of the settings
/// keeps it); `unlimited` makes both `jge` of the cap 6-byte `nop`s
/// (SonicRaT/XNW 2fa40fef6 `tex_upgrades`).
pub fn tex_upgrade_patches(build: &KnownBuild, cap: TexUpgradeCap) -> Vec<OwnedPatch> {
    let Some(t) = build.tex_upgrades else {
        return Vec::new();
    };
    match cap {
        TexUpgradeCap::Most(n) => {
            let write = [&[0xb8][..], &n.to_le_bytes(), &[0x90]].concat();
            vec![OwnedPatch { name: "tex upgrades: the cap the settings store", rva: t.load.0, expect: t.load.1.to_vec(), write }]
        }
        TexUpgradeCap::Unlimited => t
            .jges
            .iter()
            .map(|&(rva, expect)| OwnedPatch { name: "tex upgrades: no cap (jge -> nop)", rva, expect: expect.to_vec(), write: vec![0x90; 6] })
            .collect(),
    }
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
    // 300.0` QueryPerformanceCounter ticks times 1/frequency: 300 s by this
    // reading; the sources say about 30 s) and, past it, calls a handler kept
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
    // The game clocks' step (+0x22264d0, once a frame): the frame time is
    // the difference of two QueryPerformanceCounter reads times 1/frequency,
    // and every clock (nine calls of the helper +0x2226040, and one inline
    // step at +0x2226833) advances by min(it, 1/15 s) times its time scale
    // (and at least 1/30 s). Both loads read the constant 1/15 at
    // +0x6159ec0, which 63 other places read too, so the loads move, not the
    // constant. SonicRaT/XNW 2fa40fef6 `timestep4/2/1`: the same places and
    // constants (0.25 at +0x6159ca4, 0.5 at +0x6159b10, 1 at +0x6159bc4),
    // read again from our executable.
    time_step: Some(TimeStepCap {
        loads: &[
            // vmovss xmm3, [rip + 0x3f33e73] (= +0x6159ec0)
            (0x222_6045, [0xc5, 0xfa, 0x10, 0x1d, 0x73, 0x3e, 0xf3, 0x03]),
            // vmovss xmm0, [rip + 0x3f33685] (= +0x6159ec0)
            (0x222_6833, [0xc5, 0xfa, 0x10, 0x05, 0x85, 0x36, 0xf3, 0x03]),
        ],
        values: &[(250, 0x615_9ca4), (500, 0x615_9b10), (1000, 0x615_9bc4)],
        // .rdata
        rdata: (0x5f4_1000, 0x8a_0c0c),
    }),
    // The main thread places streamed resources (+0x9d94a0) until a
    // deadline of a per-frame budget (+0x2576d40); +0x2294750 works the
    // budget out from two counts and the frame rate and clamps it to
    // 250-1250 us. SonicRaT/XNW 2fa40fef6 `place_budget` (30000 there).
    place_budget: Some(PlaceBudget { mov_ecx: 0x229_4814, cmp_eax: 0x229_4819, ceiling: 1250 }),
    // maxTexUpgradesPerFrame: dword +0x6849848 (5 in the file), stored by
    // the settings code at +0x224364b from its settings field +0x110, so a
    // value written in the data word would not last; compared at
    // +0x220392a and +0x2203986 against the frame's count (+0x71290ec).
    // XNW `tex_upgrades` NOPs both jge.
    tex_upgrades: Some(TexUpgrades {
        // mov eax, [rsi+0x110]
        load: (0x224_3645, [0x8b, 0x86, 0x10, 0x01, 0x00, 0x00]),
        jges: &[
            // jge +0x22038b0
            (0x220_3930, [0x0f, 0x8d, 0x7a, 0xff, 0xff, 0xff]),
            (0x220_398c, [0x0f, 0x8d, 0x1e, 0xff, 0xff, 0xff]),
        ],
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
    fn the_time_step_cap_moves_its_loads() {
        assert_eq!(parse_time_step("0.25"), Some(250));
        assert_eq!(parse_time_step(" 0.5 "), Some(500));
        assert_eq!(parse_time_step("1.0"), Some(1000));
        assert_eq!(parse_time_step("0.1"), None);
        let build = &KNOWN_BUILDS[0];
        let cap = build.time_step.unwrap();
        // Both loads read the title's own cap, 1/15 s, at +0x6159ec0.
        for &(rva, bytes) in cap.loads {
            let disp = i32::from_le_bytes(bytes[4..].try_into().unwrap());
            assert_eq!(i64::from(rva) + 8 + i64::from(disp), 0x615_9ec0);
        }
        // A fake .rdata holding the three constants where the build has them.
        let mut rdata = vec![0u8; cap.rdata.1 as usize];
        for &(ms, rva) in cap.values {
            let at = (rva - cap.rdata.0) as usize;
            rdata[at..at + 4].copy_from_slice(&(ms as f32 / 1000.0).to_le_bytes());
        }
        // XNW's bytes, both places.
        let quarter = time_step_patches(build, 250, &rdata).unwrap();
        assert_eq!(quarter[0].write, [0xc5, 0xfa, 0x10, 0x1d, 0x57, 0x3c, 0xf3, 0x03]);
        assert_eq!(quarter[1].write, [0xc5, 0xfa, 0x10, 0x05, 0x69, 0x34, 0xf3, 0x03]);
        let half = time_step_patches(build, 500, &rdata).unwrap();
        assert_eq!((&half[0].write[4..6], &half[1].write[4..6]), (&[0xc3, 0x3a][..], &[0xd5, 0x32][..]));
        let one = time_step_patches(build, 1000, &rdata).unwrap();
        assert_eq!((&one[0].write[4..6], &one[1].write[4..6]), (&[0x77, 0x3b][..], &[0x89, 0x33][..]));
        // The constant not there (another build), or no such value: nothing.
        let at = (0x615_9ca4 - cap.rdata.0) as usize;
        rdata[at] ^= 1;
        assert!(time_step_patches(build, 250, &rdata).is_err());
        assert!(time_step_patches(build, 100, &rdata).is_err());
        assert!(time_step_patches(build, 500, &[]).is_err());
        let none = KnownBuild { time_step: None, ..*build };
        assert!(time_step_patches(&none, 500, &rdata).is_err());
    }

    #[test]
    fn place_budget_and_tex_upgrades() {
        assert_eq!(parse_place_budget("1"), Some(30_000));
        assert_eq!(parse_place_budget("30000"), Some(30_000));
        assert_eq!(parse_place_budget("100"), None);
        let build = &KNOWN_BUILDS[0];
        let p = place_budget_patches(build, 30_000);
        assert_eq!(p.iter().map(|p| p.rva).collect::<Vec<_>>(), [0x229_4814, 0x229_4819]);
        assert_eq!((p[0].expect.as_slice(), p[0].write.as_slice()), (&[0xb9, 0xe2, 0x04, 0, 0][..], &[0xb9, 0x30, 0x75, 0, 0][..]));
        assert_eq!((p[1].expect.as_slice(), p[1].write.as_slice()), (&[0x3d, 0xe2, 0x04, 0, 0][..], &[0x3d, 0x30, 0x75, 0, 0][..]));

        assert_eq!(parse_tex_upgrades("unlimited"), Some(TexUpgradeCap::Unlimited));
        assert_eq!(parse_tex_upgrades("50"), Some(TexUpgradeCap::Most(50)));
        assert_eq!(parse_tex_upgrades("0"), None);
        let most = tex_upgrade_patches(build, TexUpgradeCap::Most(50));
        assert_eq!(most.len(), 1);
        assert_eq!(most[0].write, [0xb8, 50, 0, 0, 0, 0x90]);
        let all = tex_upgrade_patches(build, TexUpgradeCap::Unlimited);
        assert_eq!(all.iter().map(|p| p.rva).collect::<Vec<_>>(), [0x220_3930, 0x220_398c]);
        assert!(all.iter().all(|p| p.write == [0x90; 6]));
        // Both jge go to the same place: the end of the frame's upgrades.
        for p in &all {
            let rel = i32::from_le_bytes(p.expect[2..6].try_into().unwrap());
            assert_eq!(i64::from(p.rva) + 6 + i64::from(rel), 0x220_38b0);
        }
        let none = KnownBuild { place_budget: None, tex_upgrades: None, ..*build };
        assert!(place_budget_patches(&none, 30_000).is_empty());
        assert!(tex_upgrade_patches(&none, TexUpgradeCap::Unlimited).is_empty());
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
