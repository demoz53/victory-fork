//! A present's planes composed in our window (`VICTORY_COMPOSE_PLANES`).
//!
//! With `--console series-s` the title presents two planes: its game image
//! and its interface (HUD, menus, map, phone), whose empty texels are zero
//! with alpha zero. A window that shows the first plane alone has none of
//! the interface.
//!
//! The pass here draws each plane's source rectangle (else the whole
//! texture), sampled bilinearly, over its destination placement in the
//! output (else the whole of it), whatever the plane's own size: a game
//! plane at half resolution is stretched under a full-size interface. The
//! first plane is drawn opaque, each later one blended over what is there,
//! premultiplied unless the plane is marked straight (see [`Alpha`]).
//!
//! Ported (GPL-3.0) from Geckomacabre/vx-layer `compose.rs` (the pass, its
//! formats and the alpha flag), with the destination placement of
//! ZSlayerHQ/zslayer-pc-port-runtime #10 (Cleverdee: the planes and their
//! 72-byte stride) and the premultiplied default of vitutu1/victory-perf's
//! overlay as Kamiloj9 carries it (demoz53/victory-fork 1b4e37cc).
//!
//! shortcut: a plane's colour space and scale filter are not read (logged
//! only): every plane is taken as SDR and filtered bilinearly.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{Host, compile};

/// How a plane after the first is blended over what is under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Alpha {
    /// Colour already multiplied by alpha: `src + dst * (1 - a)` (what an
    /// interface drawn onto a cleared target with the usual blending
    /// holds, and how the console's display planes blend by default).
    Premultiplied,
    /// `src * a + dst * (1 - a)`.
    Straight,
}

impl Alpha {
    /// A plane's blending by its flags: straight when it has flag `0x2`
    /// (vx-layer's reading), else premultiplied; `forced` wins.
    pub(crate) fn of(flags: u32, forced: Option<Alpha>) -> Alpha {
        forced.unwrap_or(if flags & 0x2 != 0 {
            Alpha::Straight
        } else {
            Alpha::Premultiplied
        })
    }
}

/// One plane to show.
pub(crate) struct ShownPlane<'a> {
    pub(crate) texture: &'a ID3D12Resource,
    /// Its host state, which it is in and stays in.
    pub(crate) state: u32,
    /// The title's source rectangle, `[left, top, right, bottom]` texels.
    pub(crate) src: Option<[i32; 4]>,
    /// The title's destination placement, `[left, top, right, bottom]` as
    /// fractions of the output.
    pub(crate) dest: Option<[f32; 4]>,
    /// How it is blended (the first plane is drawn opaque).
    pub(crate) alpha: Alpha,
}

/// One triangle over the viewport (the plane's placement); each texel
/// samples the plane's source rectangle, kept half a texel inside it so
/// that the filter takes in nothing beside it.
const COMPOSE: &str = r#"
#define RS "RootConstants(num32BitConstants = 4, b0, visibility = SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP, visibility = SHADER_VISIBILITY_PIXEL)"
cbuffer Source : register(b0) { float4 source; };
Texture2D<float4> plane : register(t0);
SamplerState clamped : register(s0);
struct Corner { float4 p : SV_Position; float2 t : TEXCOORD0; };
Corner corner(uint i : SV_VertexID) {
    Corner c;
    c.t = float2(float(i & 1) * 2.0, float(i >> 1) * 2.0);
    c.p = float4(c.t.x * 2.0 - 1.0, 1.0 - c.t.y * 2.0, 0.0, 1.0);
    return c;
}
[RootSignature(RS)]
float4 compose(Corner c) : SV_Target {
    uint w, h;
    plane.GetDimensions(w, h);
    float2 inside = 0.5 / float2(w, h);
    float2 at = clamp(lerp(source.xy, source.zw, c.t), source.xy + inside, source.zw - inside);
    return plane.SampleLevel(clamped, at, 0);
}
"#;

/// Views of planes, in a heap of ours used as a ring: a view must last
/// until its list has run, and presents can be sent ahead.
///
/// shortcut: a fixed count, not derived from how many presents are in
/// flight: with up to four planes a present, 64 presents.
const VIEWS: u32 = 256;

/// The pass: its shaders, root signature, views and, per target format and
/// blend (0 opaque, 1 premultiplied, 2 straight), the pipeline.
struct Pass {
    vs: Vec<u8>,
    ps: Vec<u8>,
    root: ID3D12RootSignature,
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(i32, u8, Option<ID3D12PipelineState>)>>,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn pass(device: &ID3D12Device) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let made = || -> windows::core::Result<Pass> {
            use windows::core::s;
            let vs = compile(COMPOSE, s!("corner"), s!("vs_5_1"))?;
            let ps = compile(COMPOSE, s!("compose"), s!("ps_5_1"))?;
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            // SAFETY: the root signature is inside the pixel shader; plain
            // heap creation and queries.
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
                let views: ID3D12DescriptorHeap =
                    device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: VIEWS,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?;
                Ok(Pass {
                    cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                    gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                    step: device.GetDescriptorHandleIncrementSize(kind),
                    vs,
                    ps,
                    root,
                    views,
                    next: AtomicU32::new(0),
                    made: Mutex::default(),
                })
            }
        };
        made()
            .inspect(|_| tracing::info!("present planes are composed in the window"))
            .inspect_err(|e| tracing::warn!(error = %e, "no pass to compose present planes"))
            .ok()
    })
    .as_ref()
}

impl Pass {
    /// The pipeline drawing into `format` with blend `blend`.
    fn pipeline(
        &self,
        device: &ID3D12Device,
        format: DXGI_FORMAT,
        blend: u8,
    ) -> Option<ID3D12PipelineState> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (format.0, blend)) {
            return pipeline.clone();
        }
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend_state = D3D12_BLEND_DESC::default();
        blend_state.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: (blend != 0).into(),
            SrcBlend: if blend == 2 {
                D3D12_BLEND_SRC_ALPHA
            } else {
                D3D12_BLEND_ONE
            },
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = format;
        // SAFETY: the description points at live data and never releases
        // its pointer.
        let pipeline = unsafe {
            device.CreateGraphicsPipelineState::<ID3D12PipelineState>(
                &D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&self.root),
                    VS: code(&self.vs),
                    PS: code(&self.ps),
                    BlendState: blend_state,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: true.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                },
            )
        }
        .inspect_err(|e| {
            tracing::warn!(error = %e, format = format.0, blend, "no pipeline to compose present planes")
        })
        .ok();
        made.push((format.0, blend, pipeline.clone()));
        pipeline
    }
}

/// The format a plane of `format` is read with: typeless ones as their
/// UNORM or float member; `None` for what a float sampler cannot read.
pub(crate) fn view_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(DXGI_FORMAT(match format.0 {
        27 => 28,
        28 | 29 | 31 => format.0,
        87 | 91 => format.0,
        90 => 87,
        23 => 24,
        24 => 24,
        9 => 10,
        10 | 11 | 13 => format.0,
        _ => return None,
    }))
}

/// The back buffer view a plane of `format` is drawn through: sRGB when
/// the plane is (its texels are decoded when read and encoded again here,
/// so an opaque plane's bytes arrive as stored, as the plain copy has them).
fn target_format(plane: DXGI_FORMAT, back: DXGI_FORMAT) -> DXGI_FORMAT {
    match (plane.0, back) {
        (29, DXGI_FORMAT_R8G8B8A8_UNORM) => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        (91, DXGI_FORMAT_B8G8R8A8_UNORM) => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        _ => back,
    }
}

/// A plane's source rectangle as fractions of its texture, `[u0, v0, u1,
/// v1]`: the whole texture unless the title's rectangle lies inside it.
fn source_fractions(src: Option<[i32; 4]>, width: u64, height: u32) -> [f32; 4] {
    match src {
        Some([l, t, r, b])
            if 0 <= l && l < r && r as u64 <= width && 0 <= t && t < b && b as u32 <= height =>
        {
            let (w, h) = (width as f32, height as f32);
            [l as f32 / w, t as f32 / h, r as f32 / w, b as f32 / h]
        }
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

/// A plane's placement in pixels of an output of `size`, `[left, top,
/// right, bottom]`: the whole output unless the title's fractions are a
/// rectangle inside 0..1 (ZSlayer #10's reading).
fn placement(dest: Option<[f32; 4]>, (width, height): (u32, u32)) -> [f32; 4] {
    let (w, h) = (width as f32, height as f32);
    match dest {
        Some([l, t, r, b])
            if [l, t, r, b].iter().all(|v| v.is_finite())
                && 0.0 <= l
                && l < r
                && r <= 1.0
                && 0.0 <= t
                && t < b
                && b <= 1.0 =>
        {
            [l * w, t * h, r * w, b * h]
        }
        _ => [0.0, 0.0, w, h],
    }
}

/// `VICTORY_COMPOSE_TRACE=1`: log what each present composes (and what
/// presents of other shapes do, see `queue.rs`) whenever it changes.
pub(crate) fn trace_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_COMPOSE_TRACE").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// Logs `text` under `what` when it differs from the last one logged for
/// `what` (`slot`: 0 the composition, 1 the title's present call).
pub(crate) fn trace(slot: usize, what: &str, text: String) {
    static LAST: Mutex<[String; 2]> = Mutex::new([String::new(), String::new()]);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last[slot] != text {
        tracing::info!(trace = %text, "compose trace: {what}");
        last[slot] = text;
    }
}

/// Whether the pass can read `texture` as a plane: a single-sampled 2D
/// texture of one slice, readable by shaders, in a format [`view_format`]
/// reads.
pub(crate) fn composable(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.SampleDesc.Count == 1
        && d.DepthOrArraySize == 1
        && d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && view_format(d.Format).is_some()
}

/// Records into `list` the composition of `planes` (each readable by pixel
/// shaders and [`composable`]) into `target` (a render target, `width` by
/// `height`, of `format`): cleared to black where no plane is placed, the
/// first plane drawn opaque over its placement, each later one blended
/// over its own. False if the pass could not be made: nothing was recorded
/// then.
///
/// The pass sets the list's descriptor heap, root signature, pipeline,
/// target, viewport, scissor and topology and puts none of them back: it is
/// for a list of ours that records nothing of the title's after it.
pub(crate) fn record(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    target: &ID3D12Resource,
    format: DXGI_FORMAT,
    (width, height): (u32, u32),
    planes: &[ShownPlane<'_>],
) -> bool {
    // Bindings outside the draw state's memory of them (`VICTORY_SET_CACHE`).
    let _binds = crate::draw::sets::Announce;
    let device = &host.device;
    let Some(pass) = pass(device) else {
        return false;
    };
    // SAFETY: an open list; slots of our heap; live resources; the pass's
    // objects live as long as the process, and a list holds what it was
    // recorded with.
    unsafe {
        // A first plane placed inside the output leaves the rest of it:
        // black, not the buffer's last frame.
        if planes
            .first()
            .is_some_and(|p| placement(p.dest, (width, height)) != [0.0, 0.0, width as f32, height as f32])
        {
            let view = host.rtv(Some(target), None);
            list.ClearRenderTargetView(view, &[0.0, 0.0, 0.0, 1.0], None);
        }
        list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        list.SetGraphicsRootSignature(&pass.root);
        list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(
            4,
        ));
        let mut traced = format!("swap chain/target {width}x{height}, {} planes", planes.len());
        for (i, p) in planes.iter().enumerate() {
            let d = p.texture.GetDesc();
            let Some(view_format) = view_format(d.Format) else {
                continue;
            };
            let target_format = target_format(d.Format, format);
            let blend = match (i, p.alpha) {
                (0, _) => 0,
                (_, Alpha::Premultiplied) => 1,
                (_, Alpha::Straight) => 2,
            };
            let Some(pipeline) = pass.pipeline(device, target_format, blend) else {
                continue;
            };
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: view_format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            device.CreateShaderResourceView(
                p.texture,
                Some(&view),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: pass.cpu + (slot * pass.step) as usize,
                },
            );
            let rtv = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: target_format,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            let rtv = host.rtv(Some(target), Some(&rtv));
            let source = source_fractions(p.src, d.Width, d.Height);
            let [l, t, r, b] = placement(p.dest, (width, height));
            if trace_on() {
                traced += &format!(
                    "; plane {i}: texture {}x{} format {}, title src {:?} -> read fractions {:?} (whole texture if the title's was unusable), title dest {:?} -> pixels [{l}, {t}, {r}, {b}], blend {blend}",
                    d.Width, d.Height, d.Format.0, p.src, source, p.dest
                );
            }
            list.SetPipelineState(&pipeline);
            list.SetGraphicsRoot32BitConstants(0, 4, source.as_ptr().cast(), 0);
            list.SetGraphicsRootDescriptorTable(
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: pass.gpu + u64::from(slot * pass.step),
                },
            );
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: l,
                TopLeftY: t,
                Width: r - l,
                Height: b - t,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: l as i32,
                top: t as i32,
                right: r.ceil() as i32,
                bottom: b.ceil() as i32,
            }]);
            list.DrawInstanced(3, 1, 0, 0);
        }
        if trace_on() {
            trace(0, "composed present", traced);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangles_outside_the_texture_are_the_whole_of_it() {
        assert_eq!(source_fractions(None, 100, 50), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            source_fractions(Some([0, 0, 200, 50]), 100, 50),
            [0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            source_fractions(Some([50, 0, 100, 25]), 100, 50),
            [0.5, 0.0, 1.0, 0.5]
        );
        assert_eq!(
            source_fractions(Some([10, 0, 10, 25]), 100, 50),
            [0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn placements_outside_the_output_are_the_whole_of_it() {
        assert_eq!(placement(None, (10, 20)), [0.0, 0.0, 10.0, 20.0]);
        assert_eq!(
            placement(Some([0.0, 0.0, 0.5, 1.0]), (10, 20)),
            [0.0, 0.0, 5.0, 20.0]
        );
        assert_eq!(
            placement(Some([f32::NAN, 0.0, 0.5, 1.0]), (10, 20)),
            [0.0, 0.0, 10.0, 20.0]
        );
        assert_eq!(
            placement(Some([0.0, 0.0, 2.0, 1.0]), (10, 20)),
            [0.0, 0.0, 10.0, 20.0]
        );
        assert_eq!(placement(Some([0.0; 4]), (10, 20)), [0.0, 0.0, 10.0, 20.0]);
    }

    #[test]
    fn planes_are_read_and_drawn_as_stored() {
        assert_eq!(view_format(DXGI_FORMAT(27)), Some(DXGI_FORMAT(28)));
        assert_eq!(view_format(DXGI_FORMAT(90)), Some(DXGI_FORMAT(87)));
        assert_eq!(view_format(DXGI_FORMAT(30)), None);
        assert_eq!(
            target_format(DXGI_FORMAT(29), DXGI_FORMAT_R8G8B8A8_UNORM),
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        );
        assert_eq!(
            target_format(DXGI_FORMAT(28), DXGI_FORMAT_R8G8B8A8_UNORM),
            DXGI_FORMAT_R8G8B8A8_UNORM
        );
    }

    #[test]
    fn a_plane_is_straight_only_when_marked_or_told() {
        assert_eq!(Alpha::of(0, None), Alpha::Premultiplied);
        assert_eq!(Alpha::of(0x2, None), Alpha::Straight);
        assert_eq!(Alpha::of(0x1, None), Alpha::Premultiplied);
        assert_eq!(
            Alpha::of(0x2, Some(Alpha::Premultiplied)),
            Alpha::Premultiplied
        );
        assert_eq!(Alpha::of(0, Some(Alpha::Straight)), Alpha::Straight);
    }

    /// Composes on the host device and reads the 4 by 2 target back, `None`
    /// without a device. The first plane is a red texture whose left half
    /// is blue, shown through `src` (a quarter of the target's size: the
    /// half-resolution world under a full-size interface); the second, as
    /// large as the target, is empty in column 0, half-alpha green
    /// (premultiplied: 0, 128, 0, 128) in column 1, opaque red in column 2
    /// and full green at half alpha (0, 255, 0, 128) in column 3.
    fn composed(
        alpha: Alpha,
        src: Option<[i32; 4]>,
        dest: Option<[f32; 4]>,
    ) -> Option<[[u8; 16]; 2]> {
        use crate::host::{host, transition};
        const RGBA8: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
        let Some(host) = host() else {
            eprintln!("no D3D12 device; skipping the compose read-back");
            return None;
        };
        let rtv = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let texture = |width: u32, height: u32| {
            let mut resource: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a render target with one RGBA8
            // subresource.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: u64::from(width),
                            Height: height,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: RGBA8,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        rtv,
                        None,
                        &mut resource,
                    )
                    .unwrap();
            }
            resource.unwrap()
        };
        let game = texture(4, 1);
        let ui = texture(4, 2);
        let target = texture(4, 2);
        let readback = host.read_back_buffer(512).unwrap();
        let commands = host.new_list().unwrap();
        let list = &commands.list;
        let strip = |left, right, bottom| RECT {
            left,
            right,
            top: 0,
            bottom,
        };
        // SAFETY: the views and all resources live through the run of the
        // list (`execute` waits for it).
        unsafe {
            let game_view = host.rtv(Some(&game), None);
            list.ClearRenderTargetView(game_view, &[1.0, 0.0, 0.0, 1.0], None);
            list.ClearRenderTargetView(game_view, &[0.0, 0.0, 1.0, 1.0], Some(&[strip(0, 2, 1)]));
            let ui_view = host.rtv(Some(&ui), None);
            list.ClearRenderTargetView(ui_view, &[0.0; 4], None);
            let half = 128.0 / 255.0;
            list.ClearRenderTargetView(ui_view, &[0.0, half, 0.0, half], Some(&[strip(1, 2, 2)]));
            list.ClearRenderTargetView(ui_view, &[1.0, 0.0, 0.0, 1.0], Some(&[strip(2, 3, 2)]));
            list.ClearRenderTargetView(ui_view, &[0.0, 1.0, 0.0, half], Some(&[strip(3, 4, 2)]));
            // Garbage under the composition: the first plane covers it, or
            // it is cleared where the first plane is not placed.
            let target_view = host.rtv(Some(&target), None);
            list.ClearRenderTargetView(target_view, &[1.0, 1.0, 1.0, 1.0], None);
            let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            list.ResourceBarrier(&[transition(&game, rtv, read), transition(&ui, rtv, read)]);
            let planes = [
                ShownPlane {
                    texture: &game,
                    state: read.0 as u32,
                    src,
                    dest,
                    alpha,
                },
                ShownPlane {
                    texture: &ui,
                    state: read.0 as u32,
                    src: None,
                    dest: None,
                    alpha,
                },
            ];
            assert!(planes.iter().all(|p| composable(p.texture)));
            assert!(record(host, list, &target, RGBA8, (4, 2), &planes));
            list.ResourceBarrier(&[transition(&target, rtv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: RGBA8,
                            Width: 4,
                            Height: 2,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        commands.close();
        host.execute(&[&commands]);
        let mut ptr = std::ptr::null_mut();
        let mut pixels = [[0u8; 16]; 2];
        // SAFETY: the copy has run; each row holds four texels at a pitch
        // of 256 bytes.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            for (y, row) in pixels.iter_mut().enumerate() {
                row.copy_from_slice(std::slice::from_raw_parts(
                    ptr.cast::<u8>().add(y * 256),
                    16,
                ));
            }
            readback.Unmap(0, None);
        }
        Some(pixels)
    }

    fn near(got: &[u8; 16], want: [[u8; 4]; 4]) -> bool {
        got.iter()
            .zip(want.as_flattened())
            .all(|(&g, &w)| g.abs_diff(w) <= 1)
    }

    /// The first plane fills the target opaque (stretched from a smaller
    /// texture, through its source rectangle, nothing beside the rectangle
    /// filtered in); an empty texel of the second leaves it as it is; the
    /// second's other texels are blended as the plane's alpha says.
    #[test]
    fn planes_are_stretched_and_blended_on_the_host_device() {
        let blue = [0, 0, 255, 255];
        let red = [255, 0, 0, 255];
        for (alpha, want) in [
            (
                Alpha::Premultiplied,
                [blue, [0, 128, 127, 255], red, [0, 255, 127, 255]],
            ),
            (
                Alpha::Straight,
                [blue, [0, 64, 127, 255], red, [0, 128, 127, 255]],
            ),
        ] {
            let Some(pixels) = composed(alpha, Some([0, 0, 2, 1]), None) else {
                return;
            };
            for row in &pixels {
                assert!(near(row, want), "{alpha:?}: {row:?}, wanted {want:?}");
            }
        }
        // Without a source rectangle the whole first plane is shown: blue on
        // the left, and red on the right under the half-alpha green.
        let Some(pixels) = composed(Alpha::Premultiplied, None, None) else {
            return;
        };
        let want = [blue, [0, 128, 127, 255], red, [127, 255, 0, 255]];
        for row in &pixels {
            assert!(near(row, want), "whole plane: {row:?}, wanted {want:?}");
        }
    }

    /// A first plane placed in the left half of the output: the right half
    /// is black under the second plane, not what the target held.
    #[test]
    fn a_placed_plane_leaves_black_beside_it() {
        let blue = [0, 0, 255, 255];
        let Some(pixels) = composed(
            Alpha::Premultiplied,
            Some([0, 0, 2, 1]),
            Some([0.0, 0.0, 0.5, 1.0]),
        ) else {
            return;
        };
        // Columns 0 and 1: the blue half stretched (under the empty and the
        // half-alpha green texels); 2: opaque red; 3: half-alpha full green
        // over black.
        let want = [blue, [0, 128, 127, 255], [255, 0, 0, 255], [0, 255, 0, 255]];
        for row in &pixels {
            assert!(near(row, want), "placed: {row:?}, wanted {want:?}");
        }
    }
}
