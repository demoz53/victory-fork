//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(windows)]
mod compose;
pub mod console;
#[cfg(windows)]
mod cores;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod material_demand;
#[cfg(windows)]
mod inplace;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod guarded;
#[cfg(windows)]
mod shadow;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod texback;
#[cfg(windows)]
mod hostcopy;

#[cfg(windows)]
pub use queue::world_began;
pub use host::on_failure;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use watch::{
    KernelWrite, title_memory_reading, title_memory_remapped, title_memory_remapping, title_memory_written,
    title_writing as kernel_will_write,
};

/// Tests: held by a test from before it asks for a device on the software
/// adapter (WARP) until everything it made there is released, so that one
/// test at a time has work on that adapter.
///
/// Tests run side by side, and each test of the indirect clamp asked for a
/// WARP device, made a queue of its own on it, compiled the same two
/// shaders and ran them. With several of them at work at once, one of
/// WARP's own pool threads read a table at index 0xFFFFFFFF and the test
/// process died of an access violation (`d3d10warp!Ordinal199+0x7cca9` and
/// `+0x7dd03`, both under `+0x7bc55`: 2 of 30 runs of the whole suite on an
/// idle PC; with the seven clamp tests alone 2 of 160 side by side, 0 of 60
/// one after another, and 0 of 100 for one of them by itself). Whether WARP
/// or our use of it is at fault there is not settled; the product runs on
/// the card's driver, not on WARP (but for diagnostics' software mode).
#[cfg(all(test, windows))]
pub(crate) fn warp_alone() -> std::sync::MutexGuard<'static, ()> {
    static ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ALONE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Total installed physical RAM in bytes, queried once from the OS.
pub(crate) fn total_system_ram_bytes() -> Option<u64> {
    static TOTAL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TOTAL.get_or_init(|| {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::SystemInformation::{
                GlobalMemoryStatusEx, MEMORYSTATUSEX,
            };
            // SAFETY: plain data the call fills in.
            let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
            m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            // SAFETY: a sized structure.
            let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
            (ok && m.ullTotalPhys > 0).then_some(m.ullTotalPhys)
        }
        #[cfg(not(windows))]
        {
            None
        }
    })
}
