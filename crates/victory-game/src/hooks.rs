//! Detours on the title's own functions that call other functions of the
//! title (the user allowed calling the title's functions on 2026-11-26).
//! Each is opt-in, for the known build only: the entry bytes of the hooked
//! function and of every function called are checked first
//! ([`Hook::entry`], [`Hook::calls`]), and nothing is installed otherwise.
//! The front end installs them in the mapped image before any title code
//! runs (`victory_loader::patch::detour`), so no thread can be in the code
//! while its entry changes; a hook then runs on whichever title thread
//! calls the function, and calls the original through its trampoline.
//!
//! After SonicRaT/XNW 2fa40fef6 (`apps/xnw/src/levers.rs`, `detours.rs`;
//! GPL-3.0, by SonicRaT), whose places and bytes are for this same build
//! (PE time stamp 0x6af783d4); read again from our executable.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

/// A detour: the function at `rva` whose first whole instructions are
/// `entry` (at least 14 bytes, position independent: they run again from
/// the trampoline), `hook` called in its place, `original` set to the
/// trampoline at install; `calls` the title's functions the hook calls and
/// the code each starts with (TLS loads compared as the loader left them).
pub struct Hook {
    pub name: &'static str,
    pub rva: u32,
    pub entry: &'static [u8],
    pub hook: usize,
    pub original: &'static AtomicUsize,
    pub calls: &'static [(u32, &'static [u8])],
}

/// The mapped image's base, set when a hook is configured.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// The frame counter of the log lines (the front end's).
static FRAME: AtomicUsize = AtomicUsize::new(0);

fn frame() -> u64 {
    match FRAME.load(Ordering::Relaxed) {
        0 => 0,
        f => {
            // SAFETY: set by `configure` from a `fn() -> u64`.
            let f: fn() -> u64 = unsafe { std::mem::transmute(f) };
            f()
        }
    }
}

// ---- VICTORY_CINE -----------------------------------------------------------

/// The cinematic instance tick `+0x50f89c0(instance, flag)` (reached by
/// tail jumps from +0x50f89a0 and +0x1ffec82): push r15; push r14; push rsi;
/// push rdi; push rbx; sub rsp, 20h; mov rsi, rcx.
const CINE_TICK_RVA: u32 = 0x50f_89c0;
const CINE_TICK_ENTRY: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce];
/// The title's by-handle skip +0x31602a0 takes these steps (read from its
/// code): if (in_context() && context_test(inst+0x48)) stop; lock(group);
/// if !(inst+0x49 & 0x18) { inst+0xe48 = stamp(inst+0x497); skip(inst) };
/// unlock(group). `skip` acts only when `can_skip` holds (+0x496 bit 5,
/// the window a scene event opens, and the definition's +0x158 bit 2).
const CINE_SKIP: u32 = 0x314_6250;
const CINE_CAN_SKIP: u32 = 0x314_62c0;
const CINE_IN_CONTEXT: u32 = 0x524_c060;
const CINE_CONTEXT_TEST: u32 = 0x524_c030;
const CINE_LOCK: u32 = 0x524_c040;
const CINE_UNLOCK: u32 = 0x52e_c1a0;
const CINE_STAMP: u32 = 0x522_8e90;
/// The group locks, 40 bytes each, indexed by the i8 at instance +0xe24.
const CINE_LOCKS: u32 = 0xab7_3240;
const CINE_CALLS: &[(u32, &[u8])] = &[
    (CINE_SKIP, &[0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x20, 0x48, 0x89, 0xce]),
    (CINE_CAN_SKIP, &[0x56, 0x48, 0x83, 0xec, 0x40, 0xf6, 0x81, 0x96, 0x04, 0x00, 0x00, 0x20]),
    (CINE_IN_CONTEXT, &[0x48, 0x83, 0xec, 0x28, 0x8b, 0x05]),
    (CINE_CONTEXT_TEST, &[0xc1, 0xe9, 0x0e, 0x83, 0x3d]),
    (CINE_LOCK, &[0x65, 0x48, 0x8b, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8b, 0x00, 0x80, 0xb8, 0x30, 0x14, 0, 0, 0]),
    (CINE_UNLOCK, &[0x65, 0x48, 0x8b, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8b, 0x00, 0x80, 0xb8, 0x30, 0x14, 0, 0, 0]),
    (CINE_STAMP, &[0xf6, 0xc1, 0x40, 0x48, 0x8d, 0x05]),
];

static CINE_ORIGINAL: AtomicUsize = AtomicUsize::new(0);
/// 0 off, 1 skip, 2 force.
static CINE_MODE: AtomicU8 = AtomicU8::new(0);
static CINE_SKIPS: AtomicU64 = AtomicU64::new(0);
/// Instances whose closed gate was logged last (a ring), so each is said once.
static CINE_CLOSED: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
static CINE_CLOSED_NEXT: AtomicUsize = AtomicUsize::new(0);

/// `VICTORY_CINE`: `skip` (1) or `force` (2). XNW's `all` (forcing scenes
/// not skippable by definition) broke the title there and is not taken.
pub fn parse_cine(v: &str) -> Option<u8> {
    match v.trim() {
        "skip" => Some(1),
        "force" => Some(2),
        _ => None,
    }
}

/// Whether an instance in `state` with flag bytes +0x496 and +0x497 is one
/// to try: playing (6 or 8), no skip under way (+0x497 bits 4-5) or done
/// (+0x496 bit 6).
pub fn cine_wants_skip(state: u32, f496: u8, f497: u8) -> bool {
    (state == 6 || state == 8) && f497 & 0x30 == 0 && f496 & 0x40 == 0
}

fn first_closed(inst: usize) -> bool {
    if CINE_CLOSED.iter().any(|c| c.load(Ordering::Relaxed) == inst) {
        return false;
    }
    CINE_CLOSED[CINE_CLOSED_NEXT.fetch_add(1, Ordering::Relaxed) % CINE_CLOSED.len()].store(inst, Ordering::Relaxed);
    true
}

/// The detour of the instance tick: the tick, then, for a playing
/// instance, the title's own skip as its by-handle skip takes it.
extern "system" fn cine_hook(inst: usize, flag: usize) -> usize {
    // SAFETY: the trampoline made at install: the tick's own entry.
    let original: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(CINE_ORIGINAL.load(Ordering::Relaxed)) };
    let result = original(inst, flag);
    let (base, mode) = (BASE.load(Ordering::Relaxed), CINE_MODE.load(Ordering::Relaxed));
    if base == 0 || mode == 0 || inst == 0 {
        return result;
    }
    // SAFETY: the instance the title just ticked, on the thread that ticks it.
    let (state, f496, f497) =
        unsafe { (((inst + 0x43c) as *const u32).read(), ((inst + 0x496) as *const u8).read(), ((inst + 0x497) as *const u8).read()) };
    if !cine_wants_skip(state, f496, f497) {
        return result;
    }
    // SAFETY: the title's functions, their entries checked at configure,
    // with the arguments its by-handle skip (+0x31602a0) gives them, on the
    // thread that ticks the instance.
    unsafe {
        if mode == 2 {
            ((inst + 0x496) as *mut u8).write(f496 | 0x20);
        }
        let call = |rva: u32| base + rva as usize;
        let can: extern "system" fn(usize) -> u8 = std::mem::transmute(call(CINE_CAN_SKIP));
        let time = ((inst + 0x460) as *const f32).read();
        if can(inst) == 0 {
            if first_closed(inst) {
                let def = ((inst + 8) as *const usize).read();
                let skippable = def != 0 && ((def + 0x158) as *const u8).read() & 4 != 0;
                let window = ((inst + 0x496) as *const u8).read() & 0x20 != 0;
                tracing::info!(
                    instance = format_args!("{inst:#x}"),
                    state,
                    scene_s = time,
                    window,
                    skippable,
                    frame = frame(),
                    "VICTORY_CINE: a playing cinematic's skip gate is closed (said once per instance)"
                );
            }
            return result;
        }
        let in_context: extern "system" fn() -> u8 = std::mem::transmute(call(CINE_IN_CONTEXT));
        let context_test: extern "system" fn(u32) -> u8 = std::mem::transmute(call(CINE_CONTEXT_TEST));
        if in_context() != 0 && context_test(((inst + 0x48) as *const u32).read()) != 0 {
            return result;
        }
        let group = call(CINE_LOCKS).wrapping_add((40 * isize::from(((inst + 0xe24) as *const i8).read())) as usize);
        let lock: extern "system" fn(usize) -> usize = std::mem::transmute(call(CINE_LOCK));
        let unlock: extern "system" fn(usize) -> usize = std::mem::transmute(call(CINE_UNLOCK));
        lock(group);
        let mut asked = false;
        if ((inst + 0x49) as *const u8).read() & 0x18 == 0 {
            let stamp: extern "system" fn(u8) -> u32 = std::mem::transmute(call(CINE_STAMP));
            ((inst + 0xe48) as *mut u32).write(stamp(f497));
            let skip: extern "system" fn(usize) -> usize = std::mem::transmute(call(CINE_SKIP));
            skip(inst);
            asked = true;
        }
        unlock(group);
        if asked {
            let n = CINE_SKIPS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                skip = n,
                instance = format_args!("{inst:#x}"),
                state,
                scene_s = time,
                frame = frame(),
                "VICTORY_CINE: the title's own skip asked for a playing cinematic"
            );
        }
    }
    result
}

/// The hooks the switches ask for (`var`), with what each wants set; an
/// unknown value is said and asks for nothing.
pub fn wanted(var: &dyn Fn(&str) -> Option<String>) -> Vec<Hook> {
    let mut hooks = Vec::new();
    if let Some(v) = var("VICTORY_CINE").filter(|v| !matches!(v.trim(), "" | "0")) {
        match parse_cine(&v) {
            Some(mode) => {
                CINE_MODE.store(mode, Ordering::Relaxed);
                hooks.push(Hook {
                    name: "cine",
                    rva: CINE_TICK_RVA,
                    entry: &CINE_TICK_ENTRY,
                    hook: cine_hook as *const () as usize,
                    original: &CINE_ORIGINAL,
                    calls: CINE_CALLS,
                });
            }
            None => println!("title hook     none for VICTORY_CINE={v}: it wants skip or force"),
        }
    }
    hooks
}

/// Gives the hooks the image's base and the frame counter for their lines;
/// call before installing any.
pub fn configure(base: usize, frame: fn() -> u64) {
    BASE.store(base, Ordering::Relaxed);
    FRAME.store(frame as usize, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cine_values_and_the_instances_tried() {
        assert_eq!(parse_cine("skip"), Some(1));
        assert_eq!(parse_cine(" force "), Some(2));
        assert_eq!(parse_cine("all"), None, "XNW's all broke the title: not taken");
        assert!(cine_wants_skip(6, 0x20, 0));
        assert!(cine_wants_skip(8, 0, 0));
        assert!(!cine_wants_skip(5, 0, 0), "not playing");
        assert!(!cine_wants_skip(6, 0, 0x10), "a skip under way");
        assert!(!cine_wants_skip(6, 0x40, 0), "skipped already");
    }

    #[test]
    fn the_hooks_asked_for() {
        let none = |_: &str| None;
        assert!(wanted(&none).is_empty());
        let bad = |n: &str| (n == "VICTORY_CINE").then(|| "all".to_owned());
        assert!(wanted(&bad).is_empty());
        let skip = |n: &str| (n == "VICTORY_CINE").then(|| "skip".to_owned());
        let h = wanted(&skip);
        assert_eq!(h.iter().map(|h| (h.name, h.rva)).collect::<Vec<_>>(), [("cine", CINE_TICK_RVA)]);
        assert!(h[0].entry.len() >= 14);
        assert_eq!(h[0].calls.len(), 7);
    }
}
