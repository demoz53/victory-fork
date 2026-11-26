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

// ---- the frame hook: VICTORY_PLAYER_POS_LOG, VICTORY_TELEPORT, keys-file teleports

/// The game timer update `+0x22264d0`: once a frame, on the thread that runs
/// the game's update (the main loop calls it at +0x8126c6, its other four
/// callers are update loops too); it reads the frame's QueryPerformanceCounter
/// difference and steps every game clock. Its 19-byte entry: push rsi; push
/// rdi; push rbx; sub rsp, 70h; vmovaps [rsp+60h], xmm9; vmovaps [rsp+50h],
/// xmm8 (stack-relative only).
const TIMER_UPDATE_RVA: u32 = 0x222_64d0;
const TIMER_UPDATE_ENTRY: [u8; 19] = [
    0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x70, 0xc5, 0x78, 0x29, 0x4c, 0x24, 0x60, 0xc5, 0x78, 0x29, 0x44, 0x24, 0x50,
];

/// The player: `FindPlayerPed` (+0x4b0330: `test cl, cl; jne; mov rax, [rip +
/// 0xa6a6afd]; ret`) reads the local `CPed*` from +0xab56e38 (0 while
/// loading or in menus). viper-community/viper e44573c6e
/// (`crates/viper-layout/src/game.rs` FIND_PLAYER_PED, LOCAL_PLAYER_PED;
/// LGPL-2.1-or-later), checked in our executable.
const FIND_PLAYER_PED: u32 = 0x04b_0330;
const LOCAL_PLAYER_PED: u32 = 0xab5_6e38;
/// The entity's packed position at +0x70 (`f32 xRem, f32 yRem, f32 z, i16
/// cellX, i16 cellY`, world = rem + 32 * cell) and its forward row at +0x50
/// (VIper `entity::POSITION`, `FORWARD`).
const POSITION: usize = 0x70;
const FORWARD: usize = 0x50;
/// The CPed vtable and its SetPosition (slot 0x178: `CEntity::SetPosition(
/// ent, const float4*, bool, bool, bool warp)`) and SetHeading (slot 0x180:
/// `(ent, radians)`) (VIper CPED_VTABLE, VT_SET_POSITION, CPED_SET_POSITION,
/// VT_SET_HEADING, CPED_SET_HEADING); a teleport calls them only through a
/// player whose vtable is this one and whose slots hold these.
const CPED_VTABLE: u32 = 0x62f_7170;
const VT_SET_POSITION: usize = 0x178;
const VT_SET_HEADING: usize = 0x180;
const SET_POSITION: u32 = 0x52a_e150;
const SET_HEADING: u32 = 0x1a3_a160;
const FRAME_CALLS: &[(u32, &[u8])] = &[
    // test cl, cl; jne +8; mov rax, [rip + 0x0a6a6afd]; ret
    (FIND_PLAYER_PED, &[0x84, 0xc9, 0x75, 0x08, 0x48, 0x8b, 0x05, 0xfd, 0x6a, 0x6a, 0x0a, 0xc3]),
    // VIper's ENTITY_SET_POSITION signature
    (SET_POSITION, &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x58, 0x40, 0x8a, 0xac, 0x24, 0xc0, 0, 0, 0]),
    // push rsi; push rdi; push rbx; sub rsp, 150h; vmovaps [rsp+140h], xmm15 (read from our exe)
    (SET_HEADING, &[0x56, 0x57, 0x53, 0x48, 0x81, 0xec, 0x50, 0x01, 0x00, 0x00, 0xc5, 0x78, 0x29, 0xbc, 0x24, 0x40, 0x01, 0x00, 0x00]),
];
/// A teleport this far or farther from where the player is now is refused:
/// the far form (place high, ask for collision there, find the ground,
/// drop) is not done.
const NEAR_M: f32 = 150.0;
/// VICTORY_TELEPORT waits for the player to stand this many frames.
const STILL_FRAMES: u32 = 30;

static FRAME_ORIGINAL: AtomicUsize = AtomicUsize::new(0);
static POS_LOG: AtomicU8 = AtomicU8::new(0);
/// The keys file's teleports (the front end's `take_teleport`).
static TELEPORT_SOURCE: AtomicUsize = AtomicUsize::new(0);

/// A place and an optional heading for the player.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Teleport {
    pub to: [f32; 3],
    pub heading: Option<f32>,
}

/// `VICTORY_TELEPORT=x,y,z[,heading]`.
pub fn parse_teleport_switch(v: &str) -> Option<Teleport> {
    let n: Vec<f32> = v.split(',').map(|w| w.trim().parse::<f32>().ok().filter(|v| v.is_finite())).collect::<Option<_>>()?;
    match n[..] {
        [x, y, z] => Some(Teleport { to: [x, y, z], heading: None }),
        [x, y, z, h] => Some(Teleport { to: [x, y, z], heading: Some(h) }),
        _ => None,
    }
}

/// The world position in an entity's packed position (16 bytes at +0x70).
pub fn unpack_position(b: &[u8; 16]) -> [f32; 3] {
    let f = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let c = |i: usize| f32::from(i16::from_le_bytes(b[i..i + 2].try_into().unwrap()));
    [f(0) + 32.0 * c(12), f(4) + 32.0 * c(14), f(8)]
}

/// The heading (radians, 0 north, counter-clockwise, in -pi..pi) of a
/// forward row's x and y (VIper `math::heading_of`).
pub fn heading_of(fx: f32, fy: f32) -> f32 {
    (-fx).atan2(fy)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Why a teleport is not made, or `None` to make it.
pub fn refuse(from: Option<[f32; 3]>, to: [f32; 3]) -> Option<&'static str> {
    match from {
        None => Some("no player"),
        Some(f) if distance(f, to) >= NEAR_M => Some("the target is 150 m or more away (only near teleports are made)"),
        Some(_) => None,
    }
}

/// This frame's teleport, if any: VICTORY_TELEPORT's once the player (at
/// `here`) has stood still for 30 frames, else the next from `take` (the
/// keys file); refused (with why) when [`refuse`] says so.
#[allow(clippy::type_complexity)]
pub fn decide(
    once: &mut Option<Teleport>,
    still: &mut (u32, Option<[f32; 3]>),
    here: Option<[f32; 3]>,
    take: impl FnOnce() -> Option<Teleport>,
) -> Option<Result<Teleport, (Teleport, &'static str)>> {
    let mut todo = None;
    if let Some(t) = *once {
        match here {
            Some(p) => {
                let stood = still.1.is_some_and(|q| distance(p, q) < 0.01);
                *still = (if stood { still.0 + 1 } else { 0 }, Some(p));
                if still.0 >= STILL_FRAMES {
                    *once = None;
                    todo = Some(t);
                }
            }
            None => *still = (0, None),
        }
    }
    let t = todo.or_else(take)?;
    Some(match refuse(here, t.to) {
        Some(why) => Err((t, why)),
        None => Ok(t),
    })
}

/// The frame hook's own state, on the one thread it acts on.
struct FrameState {
    home: Option<std::thread::ThreadId>,
    other_said: bool,
    frames: u64,
    last_log: Option<std::time::Instant>,
    /// VICTORY_TELEPORT, until done; the frames the player has stood still and where.
    once: Option<Teleport>,
    still: (u32, Option<[f32; 3]>),
}

static STATE: std::sync::Mutex<FrameState> = std::sync::Mutex::new(FrameState {
    home: None,
    other_said: false,
    frames: 0,
    last_log: None,
    once: None,
    still: (0, None),
});

/// The player's `CPed*`, or 0.
///
/// # Safety
/// `base` is the mapped title.
unsafe fn player(base: usize) -> usize {
    // SAFETY: a pointer-sized word of the title's .data.
    unsafe { ((base + LOCAL_PLAYER_PED as usize) as *const usize).read_volatile() }
}

/// Where `ped` is and which way it faces.
///
/// # Safety
/// `ped` is the live player, read on the thread that updates it.
unsafe fn place(ped: usize) -> ([f32; 3], f32) {
    // SAFETY: the entity's packed position and forward row (VIper's layout).
    unsafe {
        let packed = ((ped + POSITION) as *const [u8; 16]).read_unaligned();
        let fx = ((ped + FORWARD) as *const f32).read_unaligned();
        let fy = ((ped + FORWARD + 4) as *const f32).read_unaligned();
        (unpack_position(&packed), heading_of(fx, fy))
    }
}

#[repr(C, align(16))]
struct Float4([f32; 4]);

/// Moves the player through its own SetPosition with warp, and turns it,
/// as VIper's `set_ped_position` / `set_entity_heading` do; refused unless
/// its vtable and both slots are the known ones.
///
/// # Safety
/// On the thread that updates the player; `ped` the live player.
unsafe fn teleport_now(base: usize, ped: usize, t: Teleport) -> Result<(), &'static str> {
    // SAFETY: the player's vtable pointer and two of its slots (read only).
    let (vt, set, turn) = unsafe {
        let vt = (ped as *const usize).read();
        (vt, ((vt + VT_SET_POSITION) as *const usize).read(), ((vt + VT_SET_HEADING) as *const usize).read())
    };
    if vt != base + CPED_VTABLE as usize || set != base + SET_POSITION as usize || turn != base + SET_HEADING as usize {
        return Err("the player's vtable or its SetPosition / SetHeading slot is not the known one");
    }
    let at = Float4([t.to[0], t.to[1], t.to[2], 0.0]);
    // SAFETY: the title's SetPosition (entry checked at install) with VIper's arguments: (ped, &float4, 1, 1, warp 1).
    unsafe {
        let f: extern "C" fn(usize, *const Float4, u8, u8, u8) = std::mem::transmute(set);
        f(ped, &at, 1, 1, 1);
        if let Some(h) = t.heading {
            let g: extern "C" fn(usize, f32) = std::mem::transmute(turn);
            g(ped, h);
        }
    }
    Ok(())
}

/// The detour of the timer update: the update, then our once-a-frame work
/// on its thread.
extern "system" fn frame_hook(a: usize, b: usize, c: usize, d: usize) -> usize {
    // SAFETY: the trampoline made at install: the update's own entry.
    let original: extern "system" fn(usize, usize, usize, usize) -> usize = unsafe { std::mem::transmute(FRAME_ORIGINAL.load(Ordering::Relaxed)) };
    let result = original(a, b, c, d);
    let base = BASE.load(Ordering::Relaxed);
    if base != 0 {
        frame_work(base);
    }
    result
}

fn frame_work(base: usize) {
    let Ok(mut st) = STATE.try_lock() else { return };
    let me = std::thread::current().id();
    match st.home {
        None => {
            st.home = Some(me);
            tracing::info!(
                thread = ?me,
                os_thread = os_thread_id(),
                frame = frame(),
                "title frame hook: first call (the game timer update, +0x22264d0); it acts on this thread only"
            );
        }
        Some(home) if home != me => {
            if !std::mem::replace(&mut st.other_said, true) {
                tracing::warn!(thread = ?me, os_thread = os_thread_id(), "title frame hook: called on another thread too; nothing done there");
            }
            return;
        }
        Some(_) => {}
    }
    st.frames += 1;
    // SAFETY: the mapped title's player word.
    let ped = unsafe { player(base) };
    let here = if ped == 0 {
        None
    } else {
        // SAFETY: the live player, on the thread that updates it.
        Some(unsafe { place(ped) })
    };
    if POS_LOG.load(Ordering::Relaxed) != 0 {
        let now = std::time::Instant::now();
        if st.last_log.is_none_or(|t| now.duration_since(t).as_secs_f32() >= 1.0) {
            st.last_log = Some(now);
            match here {
                Some((p, h)) => tracing::info!(frame = frame(), "player at x={:.2} y={:.2} z={:.2} heading={:.3}", p[0], p[1], p[2], h),
                None => tracing::info!(frame = frame(), "no player"),
            }
        }
    }
    let from = here.map(|(p, _)| p);
    let source = TELEPORT_SOURCE.load(Ordering::Relaxed);
    let take = || {
        // SAFETY: set by `set_teleport_source` from a function of this type.
        (source != 0).then(|| unsafe { std::mem::transmute::<usize, fn() -> Option<Teleport>>(source) }()).flatten()
    };
    let st = &mut *st;
    let t = match decide(&mut st.once, &mut st.still, from, take) {
        None => return,
        Some(Err((t, why))) => {
            tracing::warn!(frame = frame(), from = ?from, to = ?t.to, "teleport refused: {why}");
            return;
        }
        Some(Ok(t)) => t,
    };
    // SAFETY: on the thread that updates the player (the timer update's), the live player.
    match unsafe { teleport_now(base, ped, t) } {
        Ok(()) => {
            // SAFETY: as above.
            let (now, h) = unsafe { place(ped) };
            tracing::info!(
                frame = frame(),
                from = ?from,
                to = ?t.to,
                heading = ?t.heading,
                "teleport done: the player is at x={:.2} y={:.2} z={:.2} heading={:.3}",
                now[0],
                now[1],
                now[2],
                h
            );
        }
        Err(why) => tracing::warn!(frame = frame(), from = ?from, to = ?t.to, "teleport refused: {why}"),
    }
}

fn os_thread_id() -> u32 {
    // SAFETY: no arguments; the calling thread's id.
    unsafe { GetCurrentThreadId() }
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThreadId() -> u32;
}

/// Where the frame hook takes the keys file's teleports from (the front
/// end's adapter of `victory_hle`'s queue).
pub fn set_teleport_source(take: fn() -> Option<Teleport>) {
    TELEPORT_SOURCE.store(take as usize, Ordering::Relaxed);
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
    let pos_log = var("VICTORY_PLAYER_POS_LOG").is_some_and(|v| v.trim() == "1");
    let commands = var("VICTORY_TITLE_COMMANDS").is_some_and(|v| v.trim() == "1");
    let once = var("VICTORY_TELEPORT").filter(|v| !v.trim().is_empty()).and_then(|v| {
        let t = parse_teleport_switch(&v);
        if t.is_none() {
            println!("title hook     none for VICTORY_TELEPORT={v}: it wants x,y,z or x,y,z,heading");
        }
        t
    });
    if pos_log || commands || once.is_some() {
        POS_LOG.store(u8::from(pos_log), Ordering::Relaxed);
        STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).once = once;
        hooks.push(Hook {
            name: "frame",
            rva: TIMER_UPDATE_RVA,
            entry: &TIMER_UPDATE_ENTRY,
            hook: frame_hook as *const () as usize,
            original: &FRAME_ORIGINAL,
            calls: FRAME_CALLS,
        });
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
    fn positions_headings_and_teleport_values() {
        // xRem 3.5, yRem -1.25, z 30, cells (2, -3): (67.5, -97.25, 30).
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&3.5f32.to_le_bytes());
        b[4..8].copy_from_slice(&(-1.25f32).to_le_bytes());
        b[8..12].copy_from_slice(&30.0f32.to_le_bytes());
        b[12..14].copy_from_slice(&2i16.to_le_bytes());
        b[14..16].copy_from_slice(&(-3i16).to_le_bytes());
        assert_eq!(unpack_position(&b), [67.5, -97.25, 30.0]);
        // North 0, west (forward -x) +pi/2.
        assert!(heading_of(0.0, 1.0).abs() < 1e-6);
        assert!((heading_of(-1.0, 0.0) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(parse_teleport_switch("1, 2.5,-3"), Some(Teleport { to: [1.0, 2.5, -3.0], heading: None }));
        assert_eq!(parse_teleport_switch("1,2,3,0.5").unwrap().heading, Some(0.5));
        for bad in ["", "1,2", "1,2,3,4,5", "a,b,c", "1,2,NaN"] {
            assert_eq!(parse_teleport_switch(bad), None, "{bad:?}");
        }
        assert_eq!(refuse(None, [0.0; 3]), Some("no player"));
        assert_eq!(refuse(Some([0.0; 3]), [100.0, 100.0, 0.0]), None, "141 m: made");
        assert!(refuse(Some([0.0; 3]), [120.0, 100.0, 0.0]).is_some(), "156 m: refused");
    }

    #[test]
    fn the_teleport_queue_with_a_fake_drain() {
        let t = |x: f32| Teleport { to: [x, 0.0, 0.0], heading: None };
        let (mut once, mut still) = (None, (0, None));
        // Nothing asked: nothing, and the drain is asked.
        assert_eq!(decide(&mut once, &mut still, Some([0.0; 3]), || None), None);
        // The keys file's teleport: made near, refused far or without a player.
        assert_eq!(decide(&mut once, &mut still, Some([0.0; 3]), || Some(t(5.0))), Some(Ok(t(5.0))));
        assert!(matches!(decide(&mut once, &mut still, Some([0.0; 3]), || Some(t(500.0))), Some(Err(_))));
        assert_eq!(decide(&mut once, &mut still, None, || Some(t(5.0))), Some(Err((t(5.0), "no player"))));
        // VICTORY_TELEPORT: after 30 still frames, once; a move starts the count again; the drain waits meanwhile.
        once = Some(t(3.0));
        let mut drained = 0;
        for i in 0..20 {
            assert_eq!(decide(&mut once, &mut still, Some([1.0, 0.0, 0.0]), || { drained += 1; None }), None, "{i}");
        }
        assert_eq!(decide(&mut once, &mut still, Some([2.0, 0.0, 0.0]), || None), None, "moved");
        let mut made = None;
        for _ in 0..31 {
            if let Some(r) = decide(&mut once, &mut still, Some([2.0, 0.0, 0.0]), || None) {
                made = Some(r);
                break;
            }
        }
        assert_eq!(made, Some(Ok(t(3.0))));
        assert_eq!(once, None, "only once");
        assert_eq!(drained, 20, "the keys file was still drained while VICTORY_TELEPORT waited");
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
