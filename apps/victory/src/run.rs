use std::path::Path;

use anyhow::{Context, Result};

use crate::{Console, Unimplemented};

#[cfg(windows)]
pub fn run(
    exe: &Path,
    unimplemented: Unimplemented,
    strict: bool,
    execute: bool,
    console: Console,
) -> Result<()> {
    use std::sync::Arc;
    use victory_hle::{HleResolver, HostPolicy};
    use victory_loader::{BindReport, UnimplementedPolicy, WithTitleDlls};

    // A title that relaunched itself: the old process must be gone first.
    victory_hle::modules::lifecycle::wait_for_previous();
    let cfg = victory_core::Config::load()?;
    let original = std::fs::read(exe).with_context(|| format!("reading {}", exe.display()))?;
    let prepared = victory_loader::prepare(&original)?;
    let hash = prepared.original_hash.to_hex();
    let build = victory_game::identify(&hash);
    match build {
        Some(build) => println!("build          {} ({hash})", build.label),
        None => println!("build          unknown ({hash})"),
    }

    let images = cfg.cache_subdir("images")?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let image = victory_loader::load(&prepared, &images)?;
    // In memory, before any of the title's code runs: no file is written.
    apply_title_patches(&image, build, console);
    // The program is mapped from the copy `load` wrote: the file's bytes
    // and their patched copy are not read again, and were held for the
    // whole run (two blocks of 114.7 MB each for the title, in
    // physical memory: run 65's memory map).
    drop(prepared);
    drop(original);
    victory_hle::runtime::set_main_image(image.base());
    victory_hle::runtime::set_image_path(image.base(), exe);
    println!(
        "mapped at      {:#x} (preferred {:#x})",
        image.base(),
        image.info().image_base
    );
    let resolver = HleResolver::new(if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    });
    let wanted = |dll: &str| title_dll(&resolver.host, dll);
    let title_dir = exe.parent().unwrap_or(Path::new("."));
    // A missing manifest leaves id 0 and no storage quota.
    let manifest =
        std::fs::read_to_string(title_dir.join("MicrosoftGame.config")).unwrap_or_default();
    let title = victory_hle::runtime::Title::from_manifest(&manifest);
    let storage = cfg.data_subdir(&format!("{:08X}/storage", title.id))?;
    println!(
        "title          {:08X}, storage {} ({} MB)",
        title.id,
        storage.display(),
        title.storage_bytes >> 20
    );
    // The console's fixed drives: the package as `G:`, temporary files `T:`.
    victory_hle::modules::files::mount_at('G', title_dir);
    let temp = cfg.data_subdir(&format!("{:08X}/temp", title.id))?;
    victory_hle::modules::files::mount_at('T', &temp);
    victory_hle::runtime::set_title(victory_hle::runtime::Title { storage, ..title });
    victory_hle::console::set(match console {
        Console::SeriesX => victory_hle::console::Console::SeriesX,
        Console::SeriesS => victory_hle::console::Console::SeriesS,
    });
    println!("console        {}", victory_hle::console::current().name());
    let dlls = victory_loader::load_title_dlls(image.info(), title_dir, &images, &wanted)?;
    for (name, dll) in &dlls {
        println!("title dll      {name} at {:#x}", dll.base());
        victory_hle::runtime::set_image_path(dll.base(), &title_dir.join(name));
    }

    let policy = match unimplemented {
        Unimplemented::Trap => UnimplementedPolicy::Trap,
        Unimplemented::Zero => UnimplementedPolicy::ReturnZero,
    };
    let with_dlls = WithTitleDlls {
        inner: &resolver,
        dlls: &dlls,
    };
    let with_dlls = victory_hle::bink::SkipMovies { inner: &with_dlls };
    let show = |name: &str, report: BindReport| {
        println!(
            "bound          {name}: {} hle, {} host, {} title, {} missing",
            report.hle,
            report.host,
            report.title,
            report.missing.len()
        );
        for m in &report.missing {
            println!("  missing      {}!{}", m.dll, m.symbol);
        }
    };
    for (name, dll) in &dlls {
        show(name, dll.bind_imports(&with_dlls, policy)?);
    }
    let exe_name = exe.file_name().unwrap_or_default().to_string_lossy();
    show(&exe_name, image.bind_imports(&with_dlls, policy)?);
    if !execute {
        println!("not executing (pass --execute to start the title)");
        return Ok(());
    }
    victory_loader::report_crashes();
    victory_loader::log_crashes_to(temp.with_file_name("crashes.log"));
    victory_hle::runtime::use_console_timers();
    #[cfg(feature = "diagnostics")]
    diagnostics(&image, &cfg);
    let late = LateDlls {
        host: resolver.host.clone(),
        title_dir: title_dir.to_owned(),
        cache_dir: images,
        policy,
        mapped: std::sync::Mutex::new(dlls.clone()),
        lock: LoaderLock::default(),
    };
    victory_hle::runtime::set_title_dll_loader(Box::new(move |file| {
        late.load(file, show).map_err(|e| format!("{e:#}"))
    }));
    // SAFETY: imports are bound; from here on we run title code.
    let main = unsafe { Arc::new(image).spawn_main(dlls)? };
    let code = main
        .join()
        .map_err(|_| anyhow::anyhow!("title main thread panicked"))??;
    victory_loader::profile_final_report();
    println!("title exited with {code}");
    Ok(())
}

/// The debugging aids (README): scheduled stack dumps, watched values and
/// memory saved on a crash.
#[cfg(all(windows, feature = "diagnostics"))]
fn diagnostics(image: &victory_loader::LoadedImage, cfg: &victory_core::Config) {
    victory_loader::dump_stacks_on_schedule();
    // `VICTORY_PROFILE=<file>`: where each thread's time goes, and every
    // thread's stack when frames stop.
    victory_loader::profile(victory_gpu::frame::current);
    // `VICTORY_PROFILE=<ms>`: the same question answered in the log, every
    // thread sampled every <ms>, by module and by our innermost function.
    victory_loader::sample_threads(
        |base| victory_hle::runtime::image_path(base).is_some(),
        victory_gpu::frame::current,
    );
    victory_hle::modules::audio::capture::start();
    watch(image.base(), image.info().size_of_image as usize);
    // Debugging aid: `VICTORY_CRASH_SAVE=<rva>:<len>,...` (hex) writes those
    // title ranges to `crash-<rva>.bin` in the cache dir on a crash.
    let size = image.info().size_of_image as usize;
    let ranges = std::env::var("VICTORY_CRASH_SAVE").unwrap_or_default();
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    victory_loader::save_on_crash(
        ranges
            .split(',')
            .filter_map(|r| r.split_once(':'))
            .filter_map(|(rva, len)| Some((hex(rva)?, hex(len)?)))
            .filter(|&(rva, len)| rva.checked_add(len).is_some_and(|end| end <= size))
            .map(|(rva, len)| {
                let file = cfg.cache_dir.join(format!("crash-{rva:x}.bin"));
                (image.base() + rva, len, file)
            })
            .collect(),
    );
}

/// Debugging aid: `VICTORY_WATCH=<rva>,...` (hex) logs the u32 at each
/// title+rva whenever it changes, polled every 5 ms (an engine's state, say).
#[cfg(all(windows, feature = "diagnostics"))]
fn watch(base: usize, size: usize) {
    let Ok(list) = std::env::var("VICTORY_WATCH") else {
        return;
    };
    let rvas: Vec<usize> = list
        .split(',')
        .filter_map(|r| usize::from_str_radix(r.trim().trim_start_matches("0x"), 16).ok())
        .filter(|&rva| rva + 4 <= size)
        .collect();
    std::thread::spawn(move || {
        let mut seen = vec![None; rvas.len()];
        loop {
            for (&rva, seen) in rvas.iter().zip(&mut seen) {
                // SAFETY: four bytes inside the mapped image, checked above.
                let value = unsafe { ((base + rva) as *const u32).read_volatile() };
                if *seen != Some(value) {
                    eprintln!("VIctory: title+{rva:#x} = {value:#x}");
                    *seen = Some(value);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
}

/// DLLs the host or our runtime provide are never taken from the title.
#[cfg(windows)]
fn title_dll(host: &victory_hle::HostPolicy, dll: &str) -> bool {
    !host.allows(dll) && !victory_hle::registry().iter().any(|m| m.serves(dll))
}

/// Title DLLs the title loads itself (`LoadLibrary`) that nothing imports,
/// e.g. The title's upscaler behind its FidelityFX loader.
#[cfg(windows)]
struct LateDlls {
    host: victory_hle::HostPolicy,
    title_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    policy: victory_loader::UnimplementedPolicy,
    /// Every title DLL mapped so far, to bind new ones against.
    mapped: std::sync::Mutex<Vec<(String, victory_loader::LoadedImage)>>,
    lock: LoaderLock,
}

#[cfg(windows)]
impl LateDlls {
    /// Every `LoadLibrary` of the title comes here first. Returns a title
    /// image called `file`, mapping it and the title DLLs it needs first,
    /// binding them and running their `DllMain` on the calling thread, as
    /// the Windows loader would; `None` if `file` is no title image.
    fn load(
        &self,
        file: &str,
        show: impl Fn(&str, victory_loader::BindReport),
    ) -> Result<Option<usize>> {
        use victory_hle::runtime::{forget_image, image_named, set_image_path};
        // Waits while another thread initialises a DLL; a `DllMain` that
        // loads another DLL re-enters.
        self.lock.hold(|| {
            if let Some(base) = image_named(file) {
                return Ok(Some(base));
            }
            let new = {
                let mut mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
                let wanted = |dll: &str| title_dll(&self.host, dll) && image_named(dll).is_none();
                let new = victory_loader::load_title_dll(
                    file,
                    &self.title_dir,
                    &self.cache_dir,
                    &wanted,
                )?;
                let hle = victory_hle::HleResolver::new(self.host.clone());
                let old = victory_loader::WithTitleDlls {
                    inner: &hle,
                    dlls: &mapped,
                };
                let resolver = victory_loader::WithTitleDlls {
                    inner: &old,
                    dlls: &new,
                };
                let resolver = victory_hle::bink::SkipMovies { inner: &resolver };
                for (name, dll) in &new {
                    println!(
                        "title dll      {name} at {:#x} (loaded by the title)",
                        dll.base()
                    );
                    show(name, dll.bind_imports(&resolver, self.policy)?);
                    set_image_path(dll.base(), &self.title_dir.join(name));
                }
                // Visible before their `DllMain`s run, which may load more.
                mapped.extend(new.iter().cloned());
                new
            };
            let failed = new.iter().enumerate().find_map(|(i, (name, dll))| {
                // SAFETY: bound above, dependencies attached first.
                let attached = unsafe { dll.attach() };
                attached
                    .err()
                    .map(|e| (i, e.context(format!("attaching {name}"))))
            });
            let Some((i, error)) = failed else {
                return Ok(new.last().map(|(_, dll)| dll.base()));
            };
            // As `LoadLibrary` does when a `DllMain` fails: detach what this
            // load attached (the failed one too) and unmap all of it.
            let bases: Vec<_> = new.iter().map(|(_, d)| d.base()).collect();
            self.mapped
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(_, d)| !bases.contains(&d.base()));
            for (_, done) in new[..=i].iter().rev() {
                // SAFETY: attached (or failing to) on this thread just now.
                unsafe { done.detach() };
            }
            for (_, dll) in new {
                forget_image(dll.base());
                // SAFETY: detached, and no title code holds it yet.
                unsafe { dll.unload() };
            }
            Err(error)
        })
    }
}

/// The Windows loader lock's shape: one thread at a time, re-entered by
/// the thread that holds it.
#[cfg(windows)]
#[derive(Default)]
struct LoaderLock {
    owner: std::sync::Mutex<(Option<std::thread::ThreadId>, usize)>,
    free: std::sync::Condvar,
}

#[cfg(windows)]
impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::sync::{Arc, Mutex};

    #[test]
    fn loader_lock_is_reentrant_and_exclusive() {
        let lock = Arc::new(super::LoaderLock::default());
        assert_eq!(lock.hold(|| lock.hold(|| 7)), 7);
        let order = Arc::new(Mutex::new(Vec::new()));
        let other = lock.hold(|| {
            let (theirs, their_order) = (lock.clone(), order.clone());
            let other = std::thread::spawn(move || {
                theirs.hold(|| their_order.lock().unwrap().push(2));
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push(1);
            other
        });
        other.join().unwrap();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
    }
}

#[cfg(not(windows))]
pub fn run(_: &Path, _: Unimplemented, _: bool, _: bool, _: Console) -> Result<()> {
    anyhow::bail!("running titles requires Windows")
}

/// Bytes as lower-case hex.
#[cfg(windows)]
fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The known build's code patches, made in the mapped image's memory
/// before any of its code runs (no file on disk is written or changed:
/// the copy the loader maps from stays as it was), each group all or
/// none ([`victory_loader::patch::patch_mapped`]):
///
/// - the title's upscaler choice made the one a Series X makes
///   (fxceless/victory-fork PR #798 by KyleS98, as in Geckomacabre/vx-layer
///   7bdd8c621). On a Series S the title picks FSR 4, whose output never
///   reaches the screen here (every world frame black, 20 of its compute
///   shaders without a DXIL part); its own upscaler, the one a Series X
///   runs, works. With `--console series-s`; `VICTORY_TITLE_PATCHES=1` on
///   any console (a Series X already chooses it).
/// - `VICTORY_RENDER_SCALE=<0.2..1>`: its own upscaler's render scale fixed
///   at that fraction of the output size (vitutu1/victory-perf, from
///   ZigZagBoi1/ScarlettPort), on any console. Off unless set.
/// - `VICTORY_RT_FALLBACK=1`: its lighting takes its own programs without
///   rays (SonicRaT/XNW `rtfallback`); `=2` also its feature mask without
///   the ray-tracing structures (XNW `features_nort`). Off unless set. Made,
///   it also stands the stand-in for the traced light down
///   (`victory_gpu::pipeline::set_title_fallbacks`).
/// - `VICTORY_NO_WATCHDOG=1`: its hang watchdog no longer ends the process
///   (trustno1-coder/VIctory-NVIDIA, from vitutu1/victory-perf and
///   ZigZagBoi1/ScarlettPort). Off unless set.
/// - `VICTORY_RESOLUTION=<w>x<h>`: its table of output sizes capped there
///   (soldsawn/victory-nvidia-8gb). Off unless set.
/// - `VICTORY_TIME_STEP=0.25|0.5|1`: its game clocks advance by up to that
///   many seconds a frame instead of 1/15 s (SonicRaT/XNW `timestep*`).
/// - `VICTORY_PLACE_BUDGET=<us>`: its per-frame time slice for placing
///   streamed resources may reach that instead of 1250 us (XNW
///   `place_budget`).
/// - `VICTORY_TEX_UPGRADES=<n>|unlimited`: at most n texture upgrades a
///   frame instead of 5, or no cap (XNW `tex_upgrades`).
///   All three off unless set.
///
/// `VICTORY_TITLE_PATCHES=0`: none. If a group's bytes are not there
/// (another build), nothing of it is written and that is said once; each
/// patch made is said once.
#[cfg(windows)]
fn apply_title_patches(
    image: &victory_loader::LoadedImage,
    build: Option<&victory_game::KnownBuild>,
    console: Console,
) {
    let var = |name: &str| std::env::var(name).ok();
    let size = image.info().size_of_image as usize;
    // The image's read-only constants as mapped (relocated), for the time
    // step's cap: only asked with its switch.
    let constants = build.and_then(|b| b.time_step).filter(|_| var("VICTORY_TIME_STEP").is_some()).and_then(|t| {
        let (start, len) = (t.rdata.0 as usize, t.rdata.1 as usize);
        // SAFETY: inside the mapped image (checked), its .rdata: readable,
        // and nothing writes it while we read.
        (start.checked_add(len).is_some_and(|end| end <= size))
            .then(|| unsafe { std::slice::from_raw_parts((image.base() + start) as *const u8, len) })
    });
    let Some(groups) = title_patch_groups(build, console, var, constants) else {
        return;
    };
    // SAFETY: the title's image, mapped and not yet run.
    let made = unsafe { apply_patch_groups(image.base(), size, &groups) };
    if made.contains(&RT_FALLBACK) {
        victory_gpu::pipeline::set_title_fallbacks(true);
    }
}

/// The group `VICTORY_RT_FALLBACK` makes.
#[cfg(windows)]
const RT_FALLBACK: &str = "rt fallback";

/// The groups of title patches the switches (`var`) ask for, or `None`
/// when none is asked for or the build is not known (said then).
#[cfg(windows)]
fn title_patch_groups(
    build: Option<&victory_game::KnownBuild>,
    console: Console,
    var: impl Fn(&str) -> Option<String>,
    constants: Option<&[u8]>,
) -> Option<Vec<(&'static str, Vec<victory_game::OwnedPatch>)>> {
    let switch = var("VICTORY_TITLE_PATCHES");
    if switch.as_deref() == Some("0") {
        return None;
    }
    let upscaler = switch.as_deref() == Some("1") || matches!(console, Console::SeriesS);
    let scale = var("VICTORY_RENDER_SCALE");
    let fallback = var("VICTORY_RT_FALLBACK").filter(|v| !matches!(v.trim(), "" | "0"));
    let no_watchdog = var("VICTORY_NO_WATCHDOG").is_some_and(|v| v.trim() == "1");
    let resolution = var("VICTORY_RESOLUTION").filter(|v| !v.trim().is_empty());
    let step = var("VICTORY_TIME_STEP").filter(|v| !matches!(v.trim(), "" | "0"));
    let place = var("VICTORY_PLACE_BUDGET").filter(|v| !matches!(v.trim(), "" | "0"));
    let upgrades = var("VICTORY_TEX_UPGRADES").filter(|v| !v.trim().is_empty());
    let asked = [scale.is_some(), fallback.is_some(), no_watchdog, resolution.is_some(), step.is_some(), place.is_some(), upgrades.is_some()];
    if !upscaler && !asked.contains(&true) {
        return None;
    }
    let Some(build) = build else {
        println!("title patch    none: not a known build (it runs unpatched)");
        return None;
    };
    let mut groups: Vec<(&'static str, Vec<victory_game::OwnedPatch>)> = Vec::new();
    if upscaler {
        groups.push(("upscaler choice", victory_game::owned(build.patches)));
    }
    if let Some(text) = scale {
        match victory_game::parse_render_scale(&text) {
            Some(scale) => groups.push(("render scale", victory_game::render_scale_patches(build, scale))),
            None => println!("title patch    none for VICTORY_RENDER_SCALE={text}: it wants a number from 0.2 to 1"),
        }
    }
    if let Some(text) = fallback {
        match victory_game::parse_rt_fallback(&text) {
            Some(level) => groups.push((RT_FALLBACK, victory_game::rt_fallback_patches(build, level))),
            None => println!("title patch    none for VICTORY_RT_FALLBACK={text}: it wants 1 or 2"),
        }
    }
    if no_watchdog {
        groups.push(("watchdog", victory_game::owned(build.watchdog)));
    }
    if let Some(text) = resolution {
        match victory_game::parse_resolution(&text) {
            Some((w, h)) => {
                let patches = victory_game::resolution_patches(build, w, h);
                if patches.is_empty() && build.resolution.is_some() {
                    println!("title patch    none for VICTORY_RESOLUTION={text}: no size of the title's is larger");
                } else {
                    groups.push(("resolution", patches));
                }
            }
            None => println!(
                "title patch    none for VICTORY_RESOLUTION={text}: it wants <w>x<h> from 1280x720 to 3840x2160"
            ),
        }
    }
    if let Some(text) = step {
        match victory_game::parse_time_step(&text) {
            Some(ms) => match victory_game::time_step_patches(build, ms, constants.unwrap_or_default()) {
                Ok(patches) => groups.push(("time step", patches)),
                Err(why) => println!("title patch    none for VICTORY_TIME_STEP={text}: {why}"),
            },
            None => println!("title patch    none for VICTORY_TIME_STEP={text}: it wants 0.25, 0.5 or 1"),
        }
    }
    if let Some(text) = place {
        match victory_game::parse_place_budget(&text) {
            Some(us) => groups.push(("place budget", victory_game::place_budget_patches(build, us))),
            None => println!("title patch    none for VICTORY_PLACE_BUDGET={text}: it wants microseconds from 1250 to 100000, or 1"),
        }
    }
    if let Some(text) = upgrades {
        match victory_game::parse_tex_upgrades(&text) {
            Some(cap) => groups.push(("tex upgrades", victory_game::tex_upgrade_patches(build, cap))),
            None => println!("title patch    none for VICTORY_TEX_UPGRADES={text}: it wants a number from 1 to 100000, or unlimited"),
        }
    }
    Some(groups)
}

/// Makes each group in the image at `base` (`size` bytes), all or none per
/// group, says each patch made and each group left out, and returns the
/// names of the groups made.
///
/// # Safety
/// As [`victory_loader::patch::patch_mapped`]: `base..base + size` is an
/// image mapped in this process and none of its code runs.
#[cfg(windows)]
unsafe fn apply_patch_groups(
    base: usize,
    size: usize,
    groups: &[(&'static str, Vec<victory_game::OwnedPatch>)],
) -> Vec<&'static str> {
    use victory_loader::patch::MemoryPatch;
    let mut made = Vec::new();
    for (group, patches) in groups {
        if patches.is_empty() {
            println!("title patch    none for the {group}: this build has no places for it");
            continue;
        }
        let memory: Vec<MemoryPatch<'_>> = patches
            .iter()
            .map(|p| MemoryPatch { name: p.name, rva: p.rva, expect: &p.expect, write: &p.write })
            .collect();
        // SAFETY: forwarded from this function's contract.
        match unsafe { victory_loader::patch::patch_mapped(base, size, &memory) } {
            Ok(()) => {
                for p in patches {
                    println!(
                        "title patch    +{:#x} {} (in memory): {} -> {}",
                        p.rva,
                        p.name,
                        hex_str(&p.expect),
                        hex_str(&p.write)
                    );
                }
                made.push(*group);
            }
            Err(e) => println!("title patch    none for the {group} ({e}): the title runs without it"),
        }
    }
    made
}

#[cfg(all(test, windows))]
mod patch_tests {
    use super::*;

    /// A fake image as large as the title's, zeros but for `sites`.
    fn image_with(sites: &[(u32, &[u8])]) -> Vec<u8> {
        let mut image = vec![0u8; 0x620_0000];
        for &(rva, bytes) in sites {
            image[rva as usize..][..bytes.len()].copy_from_slice(bytes);
        }
        image
    }

    fn vars(set: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| set.iter().find(|(n, _)| *n == name).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn only_asked_groups_are_built() {
        let build = &victory_game::KNOWN_BUILDS[0];
        assert!(title_patch_groups(Some(build), Console::SeriesX, vars(&[]), None).is_none());
        assert!(title_patch_groups(None, Console::SeriesX, vars(&[("VICTORY_NO_WATCHDOG", "1")]), None).is_none());
        let all = &[
            ("VICTORY_RT_FALLBACK", "2"),
            ("VICTORY_NO_WATCHDOG", "1"),
            ("VICTORY_RESOLUTION", "1920x1080"),
        ];
        let names = |g: Option<Vec<(&'static str, Vec<victory_game::OwnedPatch>)>>| {
            g.unwrap().iter().map(|(n, p)| (*n, p.len())).collect::<Vec<_>>()
        };
        assert_eq!(
            names(title_patch_groups(Some(build), Console::SeriesX, vars(all), None)),
            [(RT_FALLBACK, 2), ("watchdog", 2), ("resolution", 3)]
        );
        // The one switch that turns them all off.
        let off = &[("VICTORY_TITLE_PATCHES", "0"), ("VICTORY_RT_FALLBACK", "1"), ("VICTORY_NO_WATCHDOG", "1")];
        assert!(title_patch_groups(Some(build), Console::SeriesS, vars(off), None).is_none());
        // Values they do not take: nothing.
        let bad = &[("VICTORY_RT_FALLBACK", "3"), ("VICTORY_NO_WATCHDOG", "yes"), ("VICTORY_RESOLUTION", "640x360")];
        assert_eq!(names(title_patch_groups(Some(build), Console::SeriesX, vars(bad), None)), []);
        assert_eq!(names(title_patch_groups(Some(build), Console::SeriesX, vars(&[("VICTORY_RT_FALLBACK", "1")]), None)), [(RT_FALLBACK, 1)]);
    }

    #[test]
    fn the_time_step_cap_is_checked_and_made() {
        let build = &victory_game::KNOWN_BUILDS[0];
        let cap = build.time_step.unwrap();
        let sites: Vec<(u32, &[u8])> = cap.loads.iter().map(|(rva, b)| (*rva, &b[..])).collect();
        let mut image = image_with(&sites);
        // The fake image's .rdata holds the three constants where ours does.
        for &(ms, rva) in cap.values {
            image[rva as usize..][..4].copy_from_slice(&(ms as f32 / 1000.0).to_le_bytes());
        }
        let rdata = image[cap.rdata.0 as usize..].to_vec(); // the fake image ends before .rdata does
        let groups =
            title_patch_groups(Some(build), Console::SeriesX, vars(&[("VICTORY_TIME_STEP", "0.25")]), Some(&rdata)).unwrap();
        assert_eq!(groups.iter().map(|(n, p)| (*n, p.len())).collect::<Vec<_>>(), [("time step", 2)]);
        // SAFETY: our own buffer, nothing runs it.
        assert_eq!(unsafe { apply_patch_groups(image.as_mut_ptr() as usize, image.len(), &groups) }, ["time step"]);
        for &(rva, _) in cap.loads {
            let disp = i32::from_le_bytes(image[rva as usize + 4..][..4].try_into().unwrap());
            let to = (i64::from(rva) + 8 + i64::from(disp)) as usize;
            assert_eq!(f32::from_le_bytes(image[to..to + 4].try_into().unwrap()), 0.25, "the load reads the new cap");
        }
        // Made twice (or another build): the loads' bytes differ, left out.
        // SAFETY: as above.
        assert!(unsafe { apply_patch_groups(image.as_mut_ptr() as usize, image.len(), &groups) }.is_empty());
        // The constant is not there, or a value it does not take: no group.
        let none = title_patch_groups(Some(build), Console::SeriesX, vars(&[("VICTORY_TIME_STEP", "0.5")]), Some(&[]));
        assert!(none.unwrap().is_empty());
        let bad = title_patch_groups(Some(build), Console::SeriesX, vars(&[("VICTORY_TIME_STEP", "0.1")]), Some(&rdata));
        assert!(bad.unwrap().is_empty());
        let off = &[("VICTORY_TITLE_PATCHES", "0"), ("VICTORY_TIME_STEP", "1"), ("VICTORY_PLACE_BUDGET", "1")];
        assert!(title_patch_groups(Some(build), Console::SeriesX, vars(off), Some(&rdata)).is_none());
    }

    #[test]
    fn placement_and_upgrades_are_made_with_their_bytes() {
        let build = &victory_game::KNOWN_BUILDS[0];
        let set = &[("VICTORY_PLACE_BUDGET", "1"), ("VICTORY_TEX_UPGRADES", "unlimited")];
        let groups = title_patch_groups(Some(build), Console::SeriesX, vars(set), None).unwrap();
        assert_eq!(groups.iter().map(|(n, p)| (*n, p.len())).collect::<Vec<_>>(), [("place budget", 2), ("tex upgrades", 2)]);
        let every: Vec<&victory_game::OwnedPatch> = groups.iter().flat_map(|(_, p)| p).collect();
        let sites: Vec<(u32, &[u8])> = every.iter().map(|p| (p.rva, &p.expect[..])).collect();
        let mut image = image_with(&sites);
        // One jge is not there: the upgrade group is left out whole.
        image[0x220_398c] = 0x0e;
        // SAFETY: our own buffer, nothing runs it.
        let made = unsafe { apply_patch_groups(image.as_mut_ptr() as usize, image.len(), &groups) };
        assert_eq!(made, ["place budget"]);
        assert_eq!(&image[0x220_3930..][..6], [0x0f, 0x8d, 0x7a, 0xff, 0xff, 0xff], "left as it was");
        assert_eq!(&image[0x229_4814..][..5], [0xb9, 0x30, 0x75, 0, 0]);
        // A number instead: the settings' load is the one place.
        let n = title_patch_groups(Some(build), Console::SeriesX, vars(&[("VICTORY_TEX_UPGRADES", "40")]), None).unwrap();
        assert_eq!(n[0].1.iter().map(|p| p.rva).collect::<Vec<_>>(), [0x224_3645]);
    }

    #[test]
    fn groups_with_their_bytes_are_made_and_the_others_left_out() {
        let build = &victory_game::KNOWN_BUILDS[0];
        let set = &[("VICTORY_RT_FALLBACK", "2"), ("VICTORY_NO_WATCHDOG", "1"), ("VICTORY_RESOLUTION", "1920x1080")];
        let groups = title_patch_groups(Some(build), Console::SeriesX, vars(set), None).unwrap();
        let every: Vec<&victory_game::OwnedPatch> = groups.iter().flat_map(|(_, p)| p).collect();

        // The right bytes everywhere: every group made, each site ours now.
        let sites: Vec<(u32, &[u8])> = every.iter().map(|p| (p.rva, &p.expect[..])).collect();
        let mut image = image_with(&sites);
        // SAFETY: our own buffer, nothing runs it.
        let made = unsafe { apply_patch_groups(image.as_mut_ptr() as usize, image.len(), &groups) };
        assert_eq!(made, [RT_FALLBACK, "watchdog", "resolution"]);
        for p in &every {
            assert_eq!(&image[p.rva as usize..][..p.write.len()], &p.write[..], "{}", p.name);
        }

        // Another build: the feature mask's bytes differ. The whole rt
        // fallback group is left out (its set test too); the others are made.
        let mask = build.features_nort[0];
        let sites: Vec<(u32, &[u8])> =
            every.iter().map(|p| (p.rva, if p.rva == mask.rva { &[0xb8, 0x4f, 0xde, 0x01, 0x00][..] } else { &p.expect[..] })).collect();
        let mut image = image_with(&sites);
        // SAFETY: as above.
        let made = unsafe { apply_patch_groups(image.as_mut_ptr() as usize, image.len(), &groups) };
        assert_eq!(made, ["watchdog", "resolution"]);
        let test = build.rt_fallback[0];
        assert_eq!(&image[test.rva as usize..][..test.expect.len()], test.expect, "left as it was");
        for p in &groups[1].1 {
            assert_eq!(&image[p.rva as usize..][..p.write.len()], &p.write[..], "{}", p.name);
        }

        // An image too small for the table: that group is left out.
        let mut small = vec![0u8; 0x300_0000];
        let w = build.watchdog;
        for p in w {
            small[p.rva as usize..][..p.expect.len()].copy_from_slice(p.expect);
        }
        // SAFETY: as above.
        let made = unsafe { apply_patch_groups(small.as_mut_ptr() as usize, small.len(), &groups[1..]) };
        assert_eq!(made, ["watchdog"]);
    }
}
