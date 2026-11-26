//! `VICTORY_PROFILE=<ms>`: a sampling profiler that writes to the log (every
//! line begins `PROFILE`), for the last 30 s every 30 s and for the whole run
//! at the title's exit. Unset (or a file name: `crate::profile`): no thread,
//! no cost.
//!
//! Every <ms> a sampler thread (above normal priority) looks at every other
//! thread of the process (the list refreshed once a second). A thread whose
//! cycle count has not moved since its last sample has not run since: it is
//! where it was then, and counts as waiting. Any other is suspended for as
//! long as it takes to read its registers, its cycle count and the top of its
//! stack (at most 64 KiB, into a buffer allocated beforehand), then resumed.
//! While a thread is suspended the sampler allocates nothing, logs nothing
//! and takes no lock: the thread may hold the heap's, the loader's, or the
//! unwind tables' lock that `RtlLookupFunctionEntry` takes. The copy is
//! unwound after the thread runs again (pointers into its stack moved into
//! the copy), up to 40 frames, stopping where a step would read outside it.
//!
//! Each sample is classed by the module of its innermost frame: `title` (the
//! title's images), `victory` (our executable), `system` (any other DLL in
//! the Windows folder), `d3d` (d3d12, D3D12Core, dxgi, ...), `nvidia`
//! (nv*.dll) or `other` (no module: our generated thunks; other folders). A
//! sample whose innermost frame is a wait or delay system call (an export
//! of ntdll or win32u with "Wait" in its name, `NtDelayExecution`,
//! `NtRemoveIoCompletion`) counts as waiting, not running.
//!
//! A sample is charged to the innermost function of OUR crates (a symbol
//! beginning `victory_` or `victory::`) nearer the leaf than its first title
//! frame; frames of the executable that are not ours (std, core, alloc,
//! hashbrown, windows wrappers, the allocator shims of copystat) are passed
//! over, and the outermost of them is named beside it as the library call it
//! made ("(in Condvar::wait)"). The next function of ours outwards is its
//! caller. Frames of ours outside every title frame are thread starts and
//! are not charged. Without a symbol of ours the innermost executable frame
//! is taken.
//!
//! Every `STATE_EVERY` (20 ms; less often where the query costs more than 2%
//! of that time) the scheduler's state of every thread
//! of the process is read (`NtQuerySystemInformation`,
//! SystemProcessInformation): running, ready but not running (ready,
//! standby, deferred ready, transition), or blocked, with its wait reason.
//!
//! What it cannot see: time in the kernel is counted where the call came from
//! (a running sample whose leaf is a system call that does not wait); code
//! without unwind data inside a stack (hand-written title code, the
//! driver's) can end the walk early or send it wrong; inlined functions count
//! as the function they were inlined into; the processor a thread ran on
//! (not in the state query); and a run ended from outside (`Stop-Process`)
//! has the 30 s reports only.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::Wdk::System::SystemInformation::{
    NtQuerySystemInformation, SystemProcessInformation,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, STILL_ACTIVE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, RtlPcToFileHeader,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
use windows_sys::Win32::System::Threading::{
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, GetCurrentProcess,
    GetCurrentProcessId, GetCurrentThread, GetCurrentThreadId, GetExitCodeThread,
    GetProcessAffinityMask, GetThreadGroupAffinity, GetThreadIdealProcessorEx, INFINITE,
    OpenThread, ResumeThread, SetThreadPriority, SetWaitableTimer, SuspendThread,
    THREAD_GET_CONTEXT, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_QUERY_INFORMATION,
    THREAD_SUSPEND_RESUME, TIMER_ALL_ACCESS, WaitForSingleObject,
};
use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;

use crate::crash::{Aligned16, MAX_FRAMES, unwind_into};
use crate::profile::{Names, committed_around, function_of, name_of};

/// Words of a stack copied per sample (64 KiB).
const COPY_WORDS: usize = 64 * 1024 / 8;
const REPORT_EVERY: Duration = Duration::from_secs(30);
const REFRESH_EVERY: Duration = Duration::from_secs(1);
/// How often the scheduler's states are read.
const STATE_EVERY: Duration = Duration::from_millis(20);
/// A thread that ran fewer cycles than this since the last pass has not run
/// (blocked, or ready and not given a core): suspending a waiting thread
/// wakes it for a kernel APC, so its count moves a little with no work of
/// its own (a test with an exact compare suspended every thread at every
/// pass). At 2 ms this is about 1.5% of a core.
const RAN: u64 = 100_000;
/// Threads given in full in a report; the others get one line, up to `BRIEF`.
const DETAILED: usize = 6;
const BRIEF: usize = 30;

const TITLE: usize = 0;
const VICTORY: usize = 1;
const SYSTEM: usize = 2;
const D3D: usize = 3;
const NVIDIA: usize = 4;
const OTHER: usize = 5;
const CLASSES: usize = 6;
const CLASS_NAMES: [&str; CLASSES] = ["title", "victory", "system", "d3d", "nvidia", "other"];
/// File stems (lower case) of the D3D runtime's modules.
const D3D_NAMES: [&str; 9] = [
    "d3d12", "d3d12core", "dxgi", "d3d12sdklayers", "dxcore", "d3d11", "d3d10warp", "dxil",
    "dxcompiler",
];

/// Scheduler states, as counted: running, ready but not running, blocked.
const RUNNING: usize = 0;
const READY: usize = 1;
const BLOCKED: usize = 2;
/// The kernel's wait reasons (KWAIT_REASON), by number.
const REASONS: [&str; 43] = [
    "Executive", "FreePage", "PageIn", "PoolAllocation", "DelayExecution", "Suspended",
    "UserRequest", "WrExecutive", "WrFreePage", "WrPageIn", "WrPoolAllocation",
    "WrDelayExecution", "WrSuspended", "WrUserRequest", "WrEventPair", "WrQueue",
    "WrLpcReceive", "WrLpcReply", "WrVirtualMemory", "WrPageOut", "WrRendezvous",
    "WrKeyedEvent", "WrTerminated", "WrProcessInSwap", "WrCpuRateControl", "WrCalloutStack",
    "WrKernel", "WrResource", "WrPushLock", "WrMutex", "WrQuantumEnd", "WrDispatchInt",
    "WrPreempted", "WrYieldExecution", "WrFastMutex", "WrGuardedMutex", "WrRundown",
    "WrAlertByThreadId", "WrDeferredPreempt", "WrPhysicalFault", "WrIoRing", "WrMdlCache",
    "WrRcu",
];

/// Starts the sampler if `VICTORY_PROFILE` is a number of milliseconds
/// (not 0). `is_title` says whether a module base is one of the title's
/// images; `frames` counts frames begun.
pub fn sample_threads(is_title: fn(usize) -> bool, frames: fn() -> u64) {
    let Ok(value) = std::env::var("VICTORY_PROFILE") else {
        return;
    };
    let Ok(ms) = value.trim().parse::<u64>() else {
        return;
    };
    if ms == 0 || SHARED.get().is_some() {
        return;
    }
    let shared = Arc::new(Shared::new(ms, frames));
    if SHARED.set(shared.clone()).is_err() {
        return;
    }
    match spawn(shared, Duration::from_millis(ms), is_title) {
        Ok(_) => tracing::info!(
            "PROFILE sampler started: every {ms} ms, a report every {} s and at exit (VICTORY_PROFILE={ms})",
            REPORT_EVERY.as_secs()
        ),
        Err(e) => tracing::warn!("PROFILE sampler not started: {e}"),
    }
}

/// The report of the whole run, once, if the sampler runs: stops it first
/// (waiting up to a second). For the title's exit.
pub fn final_report() {
    let Some(shared) = SHARED.get() else {
        return;
    };
    if shared.reported.swap(true, Ordering::AcqRel) {
        return;
    }
    shared.stop.store(true, Ordering::Release);
    let waited = Instant::now();
    while !shared.stopped.load(Ordering::Acquire) && waited.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let lines = {
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut names = shared.names.lock().unwrap_or_else(|e| e.into_inner());
        let mut lines = state.close_window(&mut names, (shared.frames)(), "the last");
        let state = &*state;
        lines.extend(report(&mut names, state, "the whole run", &state.total));
        lines
    };
    for line in lines {
        tracing::info!("{line}");
    }
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// What the sampler thread and the exit's report share. Lock order: `state`
/// before `names`.
struct Shared {
    state: Mutex<State>,
    names: Mutex<Names>,
    frames: fn() -> u64,
    stop: AtomicBool,
    stopped: AtomicBool,
    reported: AtomicBool,
}

impl Shared {
    fn new(interval_ms: u64, frames: fn() -> u64) -> Self {
        let (mut process, mut system) = (0usize, 0usize);
        // SAFETY: writes two masks.
        unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
        Shared {
            state: Mutex::new(State {
                window: Agg::default(),
                total: Agg::default(),
                info: HashMap::new(),
                interval_ms,
                window_start: Instant::now(),
                frames_at: frames(),
                process_mask: process as u64,
            }),
            names: Mutex::new(Names::default()),
            frames,
            stop: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            reported: AtomicBool::new(false),
        }
    }
}

/// A thread's name and where it may run.
#[derive(Default, Clone)]
struct ThreadInfo {
    name: String,
    group: u16,
    mask: u64,
    ideal: u8,
}

struct State {
    window: Agg,
    total: Agg,
    info: HashMap<u32, ThreadInfo>,
    interval_ms: u64,
    window_start: Instant,
    frames_at: u64,
    process_mask: u64,
}

impl State {
    /// Ends the window: its report, and its counts added to the total.
    fn close_window(&mut self, names: &mut Names, frames: u64, what: &str) -> Vec<String> {
        self.window.secs = self.window_start.elapsed().as_secs_f64();
        self.window.frames = frames.saturating_sub(self.frames_at);
        self.window_start = Instant::now();
        self.frames_at = frames;
        let window = std::mem::take(&mut self.window);
        let what = format!("{what} {:.0} s", window.secs);
        let lines = report(names, self, &what, &window);
        self.total.merge(&window);
        lines
    }
}

/// Samples of one place, by the class of their leaf.
#[derive(Default, Clone)]
struct Spot {
    n: u64,
    leaf: [u64; CLASSES],
}

impl Spot {
    fn add(&mut self, leaf: usize) {
        self.n += 1;
        self.leaf[leaf] += 1;
    }
    fn merge(&mut self, other: &Spot) {
        self.n += other.n;
        for (a, b) in self.leaf.iter_mut().zip(other.leaf) {
            *a += b;
        }
    }
}

/// (ours, caller of ours, library call): 0 where there is none.
type Site = (u64, u64, u64);

#[derive(Default, Clone)]
struct PerThread {
    running: u64,
    waiting: u64,
    /// Samples with a title frame.
    title: u64,
    /// Waiting samples with a function of ours.
    waiting_ours: u64,
    /// Running samples by the class of their leaf.
    classes: [u64; CLASSES],
    /// Running samples by our innermost function and its library call.
    ours: HashMap<(u64, u64), Spot>,
    /// Waiting samples by site.
    waits: HashMap<Site, u64>,
    /// Running samples in a contended lock, by site.
    spins: HashMap<Site, u64>,
    /// Scheduler states read, and the wait reasons of the blocked ones.
    states: [u64; 3],
    reasons: HashMap<u32, u64>,
}

#[derive(Default)]
struct Agg {
    passes: u64,
    secs: f64,
    frames: u64,
    pass_us: u64,
    pass_us_max: u64,
    suspended: u64,
    suspended_us_max: u64,
    /// Time threads spent suspended, in nanoseconds.
    held_ns: u64,
    state_reads: u64,
    state_us: u64,
    state_us_max: u64,
    threads: HashMap<u32, PerThread>,
    /// Running samples of all threads by our innermost function and its
    /// library call, else the leaf's function.
    spots: HashMap<(u64, u64), Spot>,
}

impl Agg {
    fn add(&mut self, id: u32, seen: Seen, unchanged: bool) {
        let t = self.threads.entry(id).or_default();
        let site = (seen.ours, seen.caller, seen.lib);
        if seen.title {
            t.title += 1;
        }
        if seen.waiting || unchanged {
            t.waiting += 1;
            if seen.ours != 0 {
                t.waiting_ours += 1;
            }
            *t.waits.entry(site).or_default() += 1;
        } else {
            t.running += 1;
            t.classes[seen.leaf] += 1;
            if seen.ours != 0 {
                t.ours.entry((seen.ours, seen.lib)).or_default().add(seen.leaf);
            }
            if seen.contended {
                *t.spins.entry(site).or_default() += 1;
            }
            self.spots.entry(seen.spot).or_default().add(seen.leaf);
        }
    }

    fn add_state(&mut self, id: u32, state: u32, reason: u32) {
        let t = self.threads.entry(id).or_default();
        let kind = match state {
            2 => RUNNING,
            1 | 3 | 6 | 7 => READY,
            _ => BLOCKED,
        };
        t.states[kind] += 1;
        if kind == BLOCKED {
            *t.reasons.entry(reason).or_default() += 1;
        }
    }

    fn merge(&mut self, other: &Agg) {
        self.passes += other.passes;
        self.secs += other.secs;
        self.frames += other.frames;
        self.pass_us += other.pass_us;
        self.pass_us_max = self.pass_us_max.max(other.pass_us_max);
        self.suspended += other.suspended;
        self.suspended_us_max = self.suspended_us_max.max(other.suspended_us_max);
        self.held_ns += other.held_ns;
        self.state_reads += other.state_reads;
        self.state_us += other.state_us;
        self.state_us_max = self.state_us_max.max(other.state_us_max);
        for (id, o) in &other.threads {
            let t = self.threads.entry(*id).or_default();
            t.running += o.running;
            t.waiting += o.waiting;
            t.title += o.title;
            t.waiting_ours += o.waiting_ours;
            for (a, b) in t.classes.iter_mut().zip(o.classes) {
                *a += b;
            }
            for (a, b) in t.states.iter_mut().zip(o.states) {
                *a += b;
            }
            for (r, n) in &o.reasons {
                *t.reasons.entry(*r).or_default() += n;
            }
            for (f, s) in &o.ours {
                t.ours.entry(*f).or_default().merge(s);
            }
            for (f, n) in &o.waits {
                *t.waits.entry(*f).or_default() += n;
            }
            for (f, n) in &o.spins {
                *t.spins.entry(*f).or_default() += n;
            }
        }
        for (f, s) in &other.spots {
            self.spots.entry(*f).or_default().merge(s);
        }
    }
}

/// One sample, classed. Function starts; 0 where there is none.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// The leaf is a wait or delay system call.
    waiting: bool,
    /// Class of the leaf.
    leaf: usize,
    /// A title frame is on the stack.
    title: bool,
    /// Our innermost function inside the first title frame.
    ours: u64,
    /// The next function of ours outwards.
    caller: u64,
    /// The library function `ours` called (the outermost not-ours frame of
    /// the executable inside it; a contended lock's if there is one).
    lib: u64,
    /// One of those library frames is a contended lock.
    contended: bool,
    /// (ours, lib), else (the leaf's function, 0).
    spot: (u64, u64),
}

struct Tracked {
    handle: HANDLE,
    /// The cycle count read at the last pass (while suspended, if it was).
    cycles: Option<u64>,
    last: Option<Seen>,
}

/// Module classes by base address.
struct Modules {
    own: usize,
    is_title: fn(usize) -> bool,
    known: HashMap<usize, usize>,
}

impl Modules {
    fn new(is_title: fn(usize) -> bool) -> Self {
        Modules {
            // SAFETY: the executable's own base.
            own: unsafe { GetModuleHandleW(std::ptr::null()) } as usize,
            is_title,
            known: HashMap::new(),
        }
    }

    /// The class of the module holding `pc` (no thread may be suspended:
    /// `RtlPcToFileHeader` takes the loader's table lock).
    fn class(&mut self, pc: u64) -> usize {
        let mut base = std::ptr::null_mut();
        // SAFETY: looks up the image holding `pc`; writes `base`.
        unsafe { RtlPcToFileHeader(pc as *const c_void, &mut base) };
        let base = base as usize;
        if base == 0 {
            return OTHER;
        }
        if let Some(&c) = self.known.get(&base) {
            return c;
        }
        let class = self.classify(base);
        self.known.insert(base, class);
        class
    }

    fn classify(&self, base: usize) -> usize {
        if base == self.own {
            return VICTORY;
        }
        if (self.is_title)(base) {
            return TITLE;
        }
        let mut path = [0u16; 520];
        // SAFETY: `base` is a loaded module's; the buffer's length is passed.
        let len = unsafe { GetModuleFileNameW(base as _, path.as_mut_ptr(), path.len() as u32) };
        let path = String::from_utf16_lossy(&path[..len as usize]).to_ascii_lowercase();
        let file = path.rsplit('\\').next().unwrap_or("");
        let stem = file.split('.').next().unwrap_or("");
        if stem.starts_with("nv") {
            NVIDIA
        } else if D3D_NAMES.contains(&stem) {
            D3D
        } else if path.contains("\\windows\\") {
            SYSTEM
        } else {
            OTHER
        }
    }
}

/// The system call stubs of ntdll and win32u (sorted starts), and whether
/// each waits.
fn system_calls() -> Vec<(u64, bool)> {
    let mut out = Vec::new();
    for dll in ["ntdll.dll", "win32u.dll"] {
        let wide: Vec<u16> = dll.encode_utf16().chain([0]).collect();
        // SAFETY: a module of the process, if loaded; no reference taken.
        let base = unsafe { GetModuleHandleW(wide.as_ptr()) } as usize;
        if base == 0 {
            continue;
        }
        // SAFETY: a loaded image's headers and export directory.
        unsafe {
            exports(base, &mut |name, at| {
                if name.starts_with(b"Nt") {
                    out.push((at, waits(name)));
                }
            })
        };
    }
    out.sort_unstable();
    out.dedup_by_key(|e| e.0);
    out
}

/// Whether a system call of this name blocks the thread.
fn waits(name: &[u8]) -> bool {
    name.windows(4).any(|w| w == b"Wait")
        || name == b"NtDelayExecution"
        || name.starts_with(b"NtRemoveIoCompletion")
}

/// Whether `pc` lies in a waiting system call's stub (32 bytes each).
fn in_wait(calls: &[(u64, bool)], pc: u64) -> bool {
    let i = calls.partition_point(|e| e.0 <= pc);
    i > 0 && pc - calls[i - 1].0 < 32 && calls[i - 1].1
}

/// Calls `f` with each named export (not forwarded) of the image at `base`
/// and its address.
///
/// # Safety
/// `base` is a mapped PE32+ image.
unsafe fn exports(base: usize, f: &mut impl FnMut(&[u8], u64)) {
    let b = base as *const u8;
    // SAFETY: each read is inside the image's headers and export directory,
    // which the caller's image has mapped.
    let u32_at = |p: *const u8| unsafe { (p as *const u32).read_unaligned() } as usize;
    // SAFETY: as above.
    let nt = unsafe { b.add(u32_at(b.add(0x3c))) };
    // SAFETY: as above.
    let optional = unsafe { nt.add(24) };
    // SAFETY: as above.
    if u32_at(nt) != 0x4550 || unsafe { (optional as *const u16).read_unaligned() } != 0x20b {
        return;
    }
    // SAFETY: the export entry of the data directories (PE32+).
    let (rva, size) = unsafe { (u32_at(optional.add(112)), u32_at(optional.add(116))) };
    if rva == 0 {
        return;
    }
    // SAFETY: as above.
    let dir = unsafe { b.add(rva) };
    // SAFETY: IMAGE_EXPORT_DIRECTORY's fields.
    let (count, functions, names, ordinals) = unsafe {
        (u32_at(dir.add(0x18)), u32_at(dir.add(0x1c)), u32_at(dir.add(0x20)), u32_at(dir.add(0x24)))
    };
    for i in 0..count {
        // SAFETY: the i-th name and ordinal, and the function they name.
        let (name, at) = unsafe {
            let name = b.add(u32_at(b.add(names + 4 * i)));
            let ordinal = usize::from((b.add(ordinals + 2 * i) as *const u16).read_unaligned());
            (std::ffi::CStr::from_ptr(name.cast()).to_bytes(), u32_at(b.add(functions + 4 * ordinal)))
        };
        if (rva..rva + size).contains(&at) {
            continue;
        }
        f(name, (base + at) as u64);
    }
}

/// A thread's registers and the top of its stack, read while suspended.
struct Captured {
    context: Aligned16,
    /// Where the copy began in the thread's stack, and its words.
    from: u64,
    words: usize,
    cycles: u64,
    held: Duration,
}

/// Suspends `thread`, reads its registers, its cycle count and the top of
/// its stack into `stack`, and resumes it. While it is suspended nothing
/// here allocates, logs or takes a lock: only system calls and a copy.
fn capture(thread: HANDLE, stack: &mut [u64]) -> Option<Captured> {
    let mut context = Aligned16(
        // SAFETY: zeroed is a valid CONTEXT.
        unsafe { std::mem::zeroed::<CONTEXT>() },
    );
    context.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
    let (mut from, mut words, mut cycles) = (0u64, 0usize, 0u64);
    // SAFETY: a thread of this process that is not this one (the sampler
    // never lists itself); resumed below on every path.
    if unsafe { SuspendThread(thread) } == u32::MAX {
        return None;
    }
    let start = Instant::now();
    // SAFETY: an open handle with THREAD_GET_CONTEXT; an aligned context.
    let got = unsafe { GetThreadContext(thread, &mut context.0) } != 0;
    if got {
        let rsp = context.0.Rsp as usize & !7;
        if let Some((_, end)) = committed_around(rsp) {
            words = ((end - rsp) / 8).min(stack.len());
            // SAFETY: committed, readable memory of a suspended thread's
            // stack (`VirtualQuery` above), into a buffer with room.
            unsafe { std::ptr::copy_nonoverlapping(rsp as *const u64, stack.as_mut_ptr(), words) };
            from = rsp as u64;
        }
        // SAFETY: an open handle; writes one u64.
        unsafe { QueryThreadCycleTime(thread, &mut cycles) };
    }
    let held = start.elapsed();
    // SAFETY: suspended above.
    unsafe { ResumeThread(thread) };
    got.then_some(Captured { context, from, words, cycles, held })
}

/// The return addresses of a captured stack, from its copy: every word and
/// register pointing into the copied part of the thread's stack is moved
/// into the copy first, then the copy is unwound.
fn unwind_copy(c: &Captured, stack: &mut [u64], frames: &mut Vec<u64>) {
    frames.clear();
    if c.words == 0 {
        frames.push(c.context.0.Rip);
        return;
    }
    let (from, to) = (c.from, c.from + 8 * c.words as u64);
    let copy = stack.as_ptr() as u64;
    let moved = |v: u64| if (from..=to).contains(&v) { v - from + copy } else { v };
    for w in &mut stack[..c.words] {
        *w = moved(*w);
    }
    let mut x = c.context.0;
    for r in [
        &mut x.Rax, &mut x.Rcx, &mut x.Rdx, &mut x.Rbx, &mut x.Rsp, &mut x.Rbp, &mut x.Rsi,
        &mut x.Rdi, &mut x.R8, &mut x.R9, &mut x.R10, &mut x.R11, &mut x.R12, &mut x.R13,
        &mut x.R14, &mut x.R15,
    ] {
        *r = moved(*r);
    }
    unwind_into(x, copy as usize, (copy + 8 * c.words as u64) as usize, frames);
}

/// What a function of our executable is to the attribution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// A function of our crates.
    Ours,
    /// std, core, alloc, hashbrown, windows wrappers, allocator shims, C
    /// runtime pieces, functions without a symbol.
    Library,
    /// A library function that spins or waits on a contended lock.
    Contended,
}

/// The kind of a function by its symbol (`module!path`).
fn kind_of(name: &str) -> Kind {
    let symbol = name.split_once('!').map_or(name, |(_, s)| s);
    let last = symbol.rsplit("::").next().unwrap_or(symbol);
    let shim = symbol.contains("::copystat::")
        && matches!(
            last,
            "alloc" | "realloc" | "dealloc" | "alloc_zeroed" | "__rust_alloc" | "__rust_dealloc"
                | "__rust_realloc" | "__rust_alloc_zeroed"
        );
    if (symbol.starts_with("victory_") || symbol.starts_with("victory::")) && !shim {
        Kind::Ours
    } else if symbol.contains("_contended") {
        Kind::Contended
    } else {
        Kind::Library
    }
}

/// Kinds by function start, named through dbghelp once each.
#[derive(Default)]
struct Kinds {
    known: HashMap<u64, Kind>,
}

impl Kinds {
    fn kind(&mut self, names: &Mutex<Names>, f: u64) -> Kind {
        if let Some(&k) = self.known.get(&f) {
            return k;
        }
        let name = names.lock().unwrap_or_else(|e| e.into_inner()).name(f);
        let k = kind_of(&name);
        self.known.insert(f, k);
        k
    }
}

/// Classes a sample by its frames (innermost first). No thread may be
/// suspended (module lookups and dbghelp take locks).
fn classify(
    modules: &mut Modules,
    calls: &[(u64, bool)],
    kinds: &mut Kinds,
    names: &Mutex<Names>,
    frames: &[u64],
) -> Seen {
    let rip = frames[0];
    let leaf = modules.class(rip);
    let waiting = leaf == SYSTEM && in_wait(calls, rip);
    let mut seen = Seen {
        waiting,
        leaf,
        title: false,
        ours: 0,
        caller: 0,
        lib: 0,
        contended: false,
        spot: (0, 0),
    };
    // Functions of the executable inside the first title frame.
    let (mut exe, mut n) = ([0u64; MAX_FRAMES], 0);
    for (i, &pc) in frames.iter().enumerate() {
        // A return address: the call before it is in the caller.
        let pc = if i == 0 { pc } else { pc.saturating_sub(1) };
        match if i == 0 { leaf } else { modules.class(pc) } {
            TITLE => {
                seen.title = true;
                break;
            }
            VICTORY if n < exe.len() => {
                exe[n] = function_of(pc);
                n += 1;
            }
            _ => {}
        }
    }
    let exe = &exe[..n];
    match exe.iter().position(|&f| kinds.kind(names, f) == Kind::Ours) {
        Some(j) => {
            seen.ours = exe[j];
            let below = &exe[..j];
            let lock = below.iter().rposition(|&f| kinds.kind(names, f) == Kind::Contended);
            seen.contended = lock.is_some();
            seen.lib = lock.map(|i| below[i]).or(below.last().copied()).unwrap_or(0);
            seen.caller = exe[j + 1..]
                .iter()
                .copied()
                .find(|&f| f != seen.ours && kinds.kind(names, f) == Kind::Ours)
                .unwrap_or(0);
        }
        None => seen.ours = exe.first().copied().unwrap_or(0),
    }
    seen.spot = if seen.ours != 0 { (seen.ours, seen.lib) } else { (function_of(rip), 0) };
    seen
}

const INFO_LENGTH_MISMATCH: i32 = 0xC000_0004_u32 as i32;

/// The scheduler's view of this process's threads.
struct StateQuery {
    buf: Vec<u64>,
    pid: u64,
}

impl StateQuery {
    fn new() -> Self {
        StateQuery {
            buf: vec![0; 1 << 17],
            // SAFETY: plain query.
            pid: u64::from(unsafe { GetCurrentProcessId() }),
        }
    }

    /// (thread id, state, wait reason) of every thread of this process.
    fn read(&mut self, out: &mut Vec<(u32, u32, u32)>) -> bool {
        out.clear();
        for _ in 0..4 {
            let mut len = 0u32;
            let bytes = (self.buf.len() * 8).min(u32::MAX as usize) as u32;
            // SAFETY: writes at most `bytes` into our buffer and its length.
            let status = unsafe {
                NtQuerySystemInformation(SystemProcessInformation, self.buf.as_mut_ptr().cast(), bytes, &mut len)
            };
            if status == INFO_LENGTH_MISMATCH {
                self.buf.resize((len as usize + (1 << 16)) / 8, 0);
                continue;
            }
            return status >= 0 && self.parse(out).is_some();
        }
        false
    }

    /// SYSTEM_PROCESS_INFORMATION entries (x64): the next entry's offset at
    /// 0, the thread count at 4, the process id at 0x50, then from 0x100 a
    /// SYSTEM_THREAD_INFORMATION of 0x50 bytes per thread: its thread id at
    /// 0x30, state at 0x44 and wait reason at 0x48.
    fn parse(&self, out: &mut Vec<(u32, u32, u32)>) -> Option<()> {
        // SAFETY: the buffer's own bytes.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.buf.as_ptr().cast(), self.buf.len() * 8) };
        let u32_at = |o: usize| Some(u32::from_le_bytes(bytes.get(o..o + 4)?.try_into().ok()?));
        let u64_at = |o: usize| Some(u64::from_le_bytes(bytes.get(o..o + 8)?.try_into().ok()?));
        let mut at = 0usize;
        loop {
            let (next, count, pid) = (u32_at(at)?, u32_at(at + 4)?, u64_at(at + 0x50)?);
            if pid == self.pid {
                for i in 0..count as usize {
                    let t = at + 0x100 + i * 0x50;
                    out.push((u64_at(t + 0x30)? as u32, u32_at(t + 0x44)?, u32_at(t + 0x48)?));
                }
                return Some(());
            }
            if next == 0 {
                return None;
            }
            at += next as usize;
        }
    }
}

struct Sampler {
    shared: Arc<Shared>,
    me: u32,
    process: u32,
    threads: HashMap<u32, Tracked>,
    modules: Modules,
    calls: Vec<(u64, bool)>,
    kinds: Kinds,
    stack: Box<[u64]>,
    frames: Vec<u64>,
    results: Vec<(u32, Seen, bool)>,
    query: StateQuery,
    states: Vec<(u32, u32, u32)>,
    next_state: Instant,
}

fn spawn(
    shared: Arc<Shared>,
    interval: Duration,
    is_title: fn(usize) -> bool,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().name("victory sampler".into()).spawn(move || {
        let mut sampler = Sampler {
            shared,
            // SAFETY: plain queries.
            me: unsafe { GetCurrentThreadId() },
            // SAFETY: as above.
            process: unsafe { GetCurrentProcessId() },
            threads: HashMap::new(),
            modules: Modules::new(is_title),
            calls: system_calls(),
            kinds: Kinds::default(),
            stack: vec![0u64; COPY_WORDS].into_boxed_slice(),
            frames: Vec::with_capacity(MAX_FRAMES),
            results: Vec::new(),
            query: StateQuery::new(),
            states: Vec::new(),
            next_state: Instant::now(),
        };
        sampler.run(interval);
    })
}

impl Sampler {
    fn run(&mut self, interval: Duration) {
        // SAFETY: this thread's own priority; a timer of our own.
        let timer = unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            CreateWaitableTimerExW(
                std::ptr::null(),
                std::ptr::null(),
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS,
            )
        };
        let mut refreshed: Option<Instant> = None;
        let mut next = Instant::now();
        while !self.shared.stop.load(Ordering::Acquire) {
            let pass = Instant::now();
            if refreshed.is_none_or(|t| t.elapsed() >= REFRESH_EVERY) {
                self.refresh();
                refreshed = Some(pass);
            }
            self.pass();
            let lines = {
                let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                (state.window_start.elapsed() >= REPORT_EVERY).then(|| {
                    let mut names = self.shared.names.lock().unwrap_or_else(|e| e.into_inner());
                    state.close_window(&mut names, (self.shared.frames)(), "the last")
                })
            };
            for line in lines.into_iter().flatten() {
                tracing::info!("{line}");
            }
            // On a fixed beat (a late pass does not push the next ones back,
            // a pass more than a beat late is not made up).
            next = (next + interval).max(Instant::now());
            let rest = next.saturating_duration_since(Instant::now());
            if timer.is_null() {
                std::thread::sleep(rest);
            } else {
                // Relative, in 100 ns units.
                let due = -(rest.as_nanos() as i64 / 100).max(1);
                // SAFETY: our timer; waits for it.
                unsafe {
                    SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0);
                    WaitForSingleObject(timer, INFINITE);
                }
            }
        }
        for t in self.threads.values() {
            // SAFETY: opened by `refresh`.
            unsafe { CloseHandle(t.handle) };
        }
        self.threads.clear();
        if !timer.is_null() {
            // SAFETY: ours.
            unsafe { CloseHandle(timer) };
        }
        self.shared.stopped.store(true, Ordering::Release);
    }

    /// Opens threads that started and closes those that ended; names the
    /// new and still unnamed ones and reads where each may run.
    fn refresh(&mut self) {
        let mut alive = Vec::new();
        // SAFETY: a snapshot of the system's threads.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return;
        }
        // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
        let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        // SAFETY: `snapshot` is open; `entry` is sized.
        let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
        while more {
            if entry.th32OwnerProcessID == self.process && entry.th32ThreadID != self.me {
                alive.push(entry.th32ThreadID);
            }
            // SAFETY: as above.
            more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
        }
        // SAFETY: opened above.
        unsafe { CloseHandle(snapshot) };
        // Those that ended since (a thread id may be used again only after
        // its last handle, ours, is closed).
        self.threads.retain(|_, t| {
            let mut code = 0;
            // SAFETY: an open thread handle.
            let ok = unsafe { GetExitCodeThread(t.handle, &mut code) } != 0;
            let running = ok && code == STILL_ACTIVE as u32;
            if !running {
                // SAFETY: opened by us.
                unsafe { CloseHandle(t.handle) };
            }
            running
        });
        for id in alive {
            if self.threads.contains_key(&id) {
                continue;
            }
            // SAFETY: a thread of this process; it may have ended since the
            // snapshot, and then the open fails.
            let handle = unsafe {
                OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id)
            };
            if handle.is_null() {
                continue;
            }
            self.threads.insert(id, Tracked { handle, cycles: None, last: None });
        }
        let unnamed: Vec<u32> = {
            let state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            self.threads
                .keys()
                .filter(|id| state.info.get(id).is_none_or(|i| i.name.is_empty()))
                .copied()
                .collect()
        };
        let mut seen: Vec<(u32, Option<String>, ThreadInfo)> = Vec::with_capacity(self.threads.len());
        for (&id, t) in &self.threads {
            let name = unnamed.contains(&id).then(|| name_of(t.handle));
            // SAFETY: zeroed is valid for both.
            let (mut affinity, mut ideal): (GROUP_AFFINITY, PROCESSOR_NUMBER) = unsafe { std::mem::zeroed() };
            // SAFETY: an open handle with THREAD_QUERY_INFORMATION; each writes its struct.
            unsafe {
                GetThreadGroupAffinity(t.handle, &mut affinity);
                GetThreadIdealProcessorEx(t.handle, &mut ideal);
            }
            let info = ThreadInfo {
                name: String::new(),
                group: affinity.Group,
                mask: affinity.Mask as u64,
                ideal: ideal.Number,
            };
            seen.push((id, name, info));
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        for (id, name, info) in seen {
            let kept = state.info.entry(id).or_default();
            if let Some(name) = name {
                kept.name = name;
            }
            (kept.group, kept.mask, kept.ideal) = (info.group, info.mask, info.ideal);
        }
    }

    /// One pass over every thread, then its counts added under the lock.
    fn pass(&mut self) {
        // The scheduler's states: every STATE_EVERY, and taking no more than 2%
        // of a core (the query walks every thread of the system).
        let state_read = Instant::now() >= self.next_state;
        let mut state_us = None;
        if state_read {
            let t = Instant::now();
            let read = self.query.read(&mut self.states);
            let took = t.elapsed();
            self.next_state = Instant::now() + STATE_EVERY.max(took * 50);
            if read {
                state_us = Some(took.as_micros() as u64);
            }
        }
        let started = Instant::now();
        let Sampler { threads, modules, calls, kinds, stack, frames, results, shared, .. } = self;
        results.clear();
        let (mut suspended, mut held_max, mut held) = (0u64, Duration::ZERO, Duration::ZERO);
        for (&id, t) in threads.iter_mut() {
            if let (Some(cycles), Some(last)) = (t.cycles, t.last) {
                let mut now = 0;
                // SAFETY: an open thread handle; writes one u64.
                if unsafe { QueryThreadCycleTime(t.handle, &mut now) } != 0 {
                    t.cycles = Some(now);
                    if now.saturating_sub(cycles) < RAN {
                        // Not run since (beyond our last suspension's own
                        // cost): where it was.
                        results.push((id, last, true));
                        continue;
                    }
                }
            }
            let Some(c) = capture(t.handle, stack) else {
                continue;
            };
            suspended += 1;
            held_max = held_max.max(c.held);
            held += c.held;
            unwind_copy(&c, stack, frames);
            let seen = classify(modules, calls, kinds, &shared.names, frames);
            t.cycles = Some(c.cycles);
            t.last = Some(seen);
            results.push((id, seen, false));
        }
        let took = started.elapsed().as_micros() as u64;
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let w = &mut state.window;
        w.passes += 1;
        w.pass_us += took;
        w.pass_us_max = w.pass_us_max.max(took);
        w.suspended += suspended;
        w.suspended_us_max = w.suspended_us_max.max(held_max.as_micros() as u64);
        w.held_ns += held.as_nanos() as u64;
        for &(id, seen, unchanged) in &self.results {
            w.add(id, seen, unchanged);
        }
        if let Some(us) = state_us {
            w.state_reads += 1;
            w.state_us += us;
            w.state_us_max = w.state_us_max.max(us);
            for &(id, s, reason) in &self.states {
                if self.threads.contains_key(&id) {
                    w.add_state(id, s, reason);
                }
            }
        }
    }
}

/// A symbol without its module (`victory!`).
fn bare(name: &str) -> &str {
    name.split_once('!').map_or(name, |(_, s)| s)
}

/// A library function's last two path parts (`Condvar::wait`).
fn lib_name(name: &str) -> String {
    let parts: Vec<&str> = bare(name).split("::").collect();
    parts[parts.len().saturating_sub(2)..].join("::")
}

/// `ours <- caller (in lib)`.
fn place(names: &mut Names, (ours, caller, lib): Site) -> String {
    if ours == 0 {
        return "(none of ours)".into();
    }
    let mut s = bare(&names.name(ours)).to_string();
    if caller != 0 {
        s += " <- ";
        s += bare(&names.name(caller));
    }
    if lib != 0 {
        s += &format!(" (in {})", lib_name(&names.name(lib)));
    }
    s
}

/// What an unnamed title thread does, by the names of its functions of ours.
fn role(names: &mut Names, t: &PerThread) -> &'static str {
    const RULES: [(&str, &[&str]); 4] = [
        ("submits", &["submit", "execute_command", "present", "card_before", "fence"]),
        ("records", &["draw::", "list::", "descriptor", "pipeline", "record", "dispatch"]),
        ("audio", &["audio", "xma", "xaudio", "sound"]),
        ("loads", &["dstorage", "files::", "inflate", "bcpack", "decompress"]),
    ];
    let mut items: Vec<(u64, u64)> = t
        .ours
        .iter()
        .map(|(&(f, _), s)| (f, s.n))
        .chain(t.waits.iter().map(|(&(f, _, _), &n)| (f, n)))
        .filter(|&(f, _)| f != 0)
        .collect();
    items.sort_by_key(|&(_, n)| Reverse(n));
    let mut weight = [0u64; RULES.len()];
    for (f, n) in items.into_iter().take(24) {
        let name = names.name(f).to_ascii_lowercase();
        if let Some(r) = RULES.iter().position(|(_, keys)| keys.iter().any(|k| name.contains(k))) {
            weight[r] += n;
        }
    }
    match (0..RULES.len()).max_by_key(|&r| weight[r]) {
        Some(r) if weight[r] > 0 => RULES[r].0,
        _ => "other",
    }
}

/// Processors as ranges: `8-12,16`.
fn processors(mask: u64) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < 64 {
        if mask >> i & 1 == 1 {
            let start = i;
            while i + 1 < 64 && mask >> (i + 1) & 1 == 1 {
                i += 1;
            }
            out.push(if start == i { format!("{i}") } else { format!("{start}-{i}") });
        }
        i += 1;
    }
    out.join(",")
}

/// The report's lines for `a`.
fn report(names: &mut Names, st: &State, what: &str, a: &Agg) -> Vec<String> {
    let interval_ms = st.interval_ms;
    let running: u64 = a.threads.values().map(|t| t.running).sum();
    let waiting: u64 = a.threads.values().map(|t| t.waiting).sum();
    let passes = a.passes.max(1) as f64;
    let pct = |n: u64, of: u64| n as f64 * 100.0 / of.max(1) as f64;
    let fps = a.frames as f64 / a.secs.max(1e-3);
    let mut out = vec![format!(
        "PROFILE {what}: {} passes asked every {interval_ms} ms ({:.0} a second), {} frames begun ({fps:.1} a second); the sampler took {:.0} us a pass (most {} us), suspended {:.1} threads a pass for {:.1} us each (the longest {} us), read the scheduler's states {} times for {:.0} us each (most {} us); {} threads; running {:.2} cores, waiting {:.1} threads",
        a.passes,
        a.passes as f64 / a.secs.max(1e-3),
        a.frames,
        a.pass_us as f64 / passes,
        a.pass_us_max,
        a.suspended as f64 / passes,
        a.held_ns as f64 / 1000.0 / a.suspended.max(1) as f64,
        a.suspended_us_max,
        a.state_reads,
        a.state_us as f64 / a.state_reads.max(1) as f64,
        a.state_us_max,
        a.threads.len(),
        running as f64 / passes,
        waiting as f64 / passes,
    )];
    if running == 0 {
        return out;
    }
    out.push(
        "PROFILE how to read: A <- B = our function A, called from our function B; (in L) = the library call (std, core, alloc, windows, ...) A made; [leaf ...] = modules of the innermost frames; states = the scheduler's, read every 20 ms or less often: running / ready but not running / blocked".into(),
    );
    let split = |s: &Spot| -> String {
        let mut c: Vec<usize> = (0..CLASSES).filter(|&c| s.leaf[c] > 0).collect();
        c.sort_by_key(|&c| Reverse(s.leaf[c]));
        c.iter().take(2).map(|&c| format!("{} {:.0}%", CLASS_NAMES[c], pct(s.leaf[c], s.n))).collect::<Vec<_>>().join(", ")
    };
    let state_share = |t: &PerThread, k: usize| t.states[k] as f64 / t.states.iter().sum::<u64>().max(1) as f64;
    // Labels: names, or "title/<role>" for unnamed title threads.
    let mut labels: HashMap<u32, String> = HashMap::new();
    for (&id, t) in &a.threads {
        let name = st.info.get(&id).map(|i| i.name.clone()).unwrap_or_default();
        let label = if !name.is_empty() {
            name
        } else if t.title > 0 {
            format!("title/{}", role(names, t))
        } else {
            "unnamed".into()
        };
        labels.insert(id, label);
    }
    // The thread with the most time running or blocked in our code.
    if let Some((&id, t)) = a.threads.iter().max_by_key(|(_, t)| t.running + t.waiting_ours) {
        let samples = (t.running + t.waiting).max(1) as f64;
        // Milliseconds a frame, or a second when no frame was begun.
        let per_frame = |n: u64| n as f64 / samples * 1000.0 / if a.frames > 0 { fps } else { 1.0 };
        let mut waits: Vec<(&Site, &u64)> = t.waits.iter().collect();
        waits.sort_by_key(|(_, n)| Reverse(**n));
        let blocked: Vec<String> = waits
            .iter()
            .take(3)
            .map(|&(site, &n)| format!("{:.1} ms {}", per_frame(n), place(names, *site)))
            .collect();
        let frames = if a.frames > 0 {
            format!(
                "; per frame ({fps:.1} a second): running {:.1} ms, waiting {:.1} ms, of it {}",
                per_frame(t.running),
                per_frame(t.waiting),
                blocked.join("; ")
            )
        } else {
            format!("; no frames: waiting a second in {}", if blocked.is_empty() { "-".to_string() } else { blocked.join("; ") })
        };
        out.push(format!(
            "PROFILE critical thread {id} \"{}\": per second running {:.0} ms, not running {:.0} ms (states: ready {:.0} ms, blocked {:.0} ms){frames}",
            labels[&id],
            t.running as f64 / samples * 1000.0,
            t.waiting as f64 / samples * 1000.0,
            state_share(t, READY) * 1000.0,
            state_share(t, BLOCKED) * 1000.0,
        ));
    }
    // Totals by role (names with their numbers taken out).
    let mut roles: HashMap<String, (u32, f64, f64, f64)> = HashMap::new();
    for (id, t) in &a.threads {
        let key: String = labels[id].chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect();
        let key = key.replace("##", "#").replace("##", "#");
        let r = roles.entry(key).or_default();
        r.0 += 1;
        r.1 += t.running as f64 / passes;
        r.2 += state_share(t, READY);
        r.3 += state_share(t, BLOCKED);
    }
    let mut roles: Vec<(String, (u32, f64, f64, f64))> = roles.into_iter().collect();
    roles.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    for (role, (n, run, ready, blocked)) in roles.iter().take(12) {
        out.push(format!(
            "PROFILE role \"{role}\" x{n}: running {run:.2} cores, ready but not running {ready:.2} threads, blocked {blocked:.1} threads"
        ));
    }
    let mut threads: Vec<(u32, &PerThread)> =
        a.threads.iter().filter(|(_, t)| t.running * 100 > running).map(|(&id, t)| (id, t)).collect();
    threads.sort_by_key(|(_, t)| Reverse(t.running));
    for (rank, (id, t)) in threads.into_iter().take(DETAILED + BRIEF).enumerate() {
        let reason = t
            .reasons
            .iter()
            .max_by_key(|(_, n)| **n)
            .map_or("-", |(&r, _)| REASONS.get(r as usize).copied().unwrap_or("other"));
        let runs_on = match st.info.get(&id) {
            Some(i) if i.mask != 0 && i.mask != st.process_mask => {
                format!("group {} processors {} (ideal {})", i.group, processors(i.mask), i.ideal)
            }
            Some(i) => format!("any processor (ideal {})", i.ideal),
            None => "?".into(),
        };
        let classes: Vec<String> = (0..CLASSES)
            .filter(|&c| t.classes[c] > 0)
            .map(|c| format!("{} {:.0}%", CLASS_NAMES[c], pct(t.classes[c], t.running)))
            .collect();
        out.push(format!(
            "PROFILE thread {id} \"{}\": {:.1}% of all running ({:.2} cores); samples running {:.0}%, waiting {:.0}%; states running {:.0}%, ready {:.0}%, blocked {:.0}% (mostly {reason}); runs on {runs_on}; running by leaf module: {}",
            labels[&id],
            pct(t.running, running),
            t.running as f64 / passes,
            pct(t.running, t.running + t.waiting),
            pct(t.waiting, t.running + t.waiting),
            state_share(t, RUNNING) * 100.0,
            state_share(t, READY) * 100.0,
            state_share(t, BLOCKED) * 100.0,
            classes.join(", ")
        ));
        if rank >= DETAILED {
            continue;
        }
        let mut ours: Vec<(&(u64, u64), &Spot)> = t.ours.iter().collect();
        ours.sort_by_key(|(_, s)| Reverse(s.n));
        let charged: u64 = ours.iter().map(|(_, s)| s.n).sum();
        for &(&(f, lib), s) in ours.iter().take(12) {
            out.push(format!(
                "PROFILE thread {id}   {:5.1}%  {}  [leaf {}]",
                pct(s.n, t.running),
                place(names, (f, 0, lib)),
                split(s)
            ));
        }
        if charged < t.running {
            out.push(format!(
                "PROFILE thread {id}   {:5.1}%  (none of ours inside the title's frames)",
                pct(t.running - charged, t.running)
            ));
        }
        let mut waits: Vec<(&Site, &u64)> = t.waits.iter().collect();
        waits.sort_by_key(|(_, n)| Reverse(**n));
        for &(site, &n) in waits.iter().take(8) {
            out.push(format!("PROFILE thread {id}   waits {:5.1}%  {}", pct(n, t.waiting), place(names, *site)));
        }
        let mut spins: Vec<(&Site, &u64)> = t.spins.iter().collect();
        spins.sort_by_key(|(_, n)| Reverse(**n));
        for &(site, &n) in spins.iter().take(5) {
            out.push(format!(
                "PROFILE thread {id}   contended lock, running {:5.1}%  {}",
                pct(n, t.running),
                place(names, *site)
            ));
        }
    }
    let mut spots: Vec<(&(u64, u64), &Spot)> = a.spots.iter().collect();
    spots.sort_by_key(|(_, s)| Reverse(s.n));
    out.push(format!(
        "PROFILE top 25 over all threads ({running} running samples): our innermost function under the title's frames (and its library call), else the leaf's function"
    ));
    for (i, &(&(f, lib), s)) in spots.iter().take(25).enumerate() {
        let at = if kind_of(&names.name(f)) == Kind::Ours || lib != 0 {
            place(names, (f, 0, lib))
        } else {
            names.name(f)
        };
        out.push(format!("PROFILE top {:2} {:5.1}%  {at}  [leaf {}]", i + 1, pct(s.n, running), split(s)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// Spins in this function's own instructions until `stop`.
    #[inline(never)]
    fn busy_spin(stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) {
            // SAFETY: a countdown in one register.
            unsafe { std::arch::asm!("2:", "dec {n}", "jnz 2b", n = inout(reg) 2_000_000u64 => _) };
        }
    }

    /// Waits on `rx` from a frame of its own.
    #[inline(never)]
    fn wait_inside(rx: &std::sync::mpsc::Receiver<()>) {
        let _ = std::hint::black_box(rx).recv();
    }

    /// Calls `wait_inside`, so that has a caller of ours.
    #[inline(never)]
    fn wait_outside(rx: &std::sync::mpsc::Receiver<()>) {
        wait_inside(std::hint::black_box(rx));
        std::hint::black_box(());
    }

    fn not_title(_: usize) -> bool {
        false
    }

    fn no_frames() -> u64 {
        0
    }

    #[test]
    fn wait_calls_are_found() {
        let calls = system_calls();
        assert!(calls.iter().filter(|c| c.1).count() > 10, "{} calls", calls.len());
        assert!(waits(b"NtWaitForSingleObject") && waits(b"NtDelayExecution"));
        assert!(!waits(b"NtYieldExecution") && !waits(b"NtReadFile"));
    }

    #[test]
    fn library_frames_are_told_from_ours() {
        assert_eq!(kind_of("victory!victory_gpu::host::Host::settle"), Kind::Ours);
        assert_eq!(kind_of("victory!victory_guarded_copy_at"), Kind::Ours);
        assert_eq!(kind_of("victory!victory::run::run"), Kind::Ours);
        assert_eq!(kind_of("victory!std::sys::sync::condvar::futex::Condvar::wait"), Kind::Library);
        assert_eq!(kind_of("victory!std::sys::sync::mutex::futex::Mutex::lock_contended"), Kind::Contended);
        assert_eq!(kind_of("victory!std::sys::sync::rwlock::futex::RwLock::write_contended"), Kind::Contended);
        assert_eq!(kind_of("victory!victory_gpu::copystat::impl$0::realloc"), Kind::Library);
        assert_eq!(kind_of("victory!victory_gpu::copystat::_::__rust_dealloc"), Kind::Library);
        assert_eq!(kind_of("victory!hashbrown::raw::RawTable::reserve_rehash"), Kind::Library);
        assert_eq!(kind_of("victory!windows_core::unknown::impl$1::clone"), Kind::Library);
        assert_eq!(kind_of("victory.exe+0x1234"), Kind::Library);
        assert_eq!(lib_name("victory!std::sys::sync::condvar::futex::Condvar::wait"), "Condvar::wait");
        assert_eq!(processors(0x1f00), "8-12");
        assert_eq!(processors(0b1011), "0-1,3");
    }

    /// A thread blocked in a function of ours is unwound from its stack's
    /// copy through that function, and charged to it past the library's
    /// frames, with its caller.
    #[test]
    fn a_copied_stack_unwinds_through_its_callers() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (id_tx, id_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            // SAFETY: plain query.
            id_tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            wait_outside(&rx);
        });
        let id = id_rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        // SAFETY: a thread of this process, opened for sampling.
        let thread = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, 0, id) };
        assert!(!thread.is_null());
        let mut stack = vec![0u64; COPY_WORDS];
        let mut frames = Vec::with_capacity(MAX_FRAMES);
        let c = capture(thread, &mut stack).expect("captured");
        unwind_copy(&c, &mut stack, &mut frames);
        let mut modules = Modules::new(not_title);
        let calls = system_calls();
        let names = Mutex::new(Names::default());
        let seen = classify(&mut modules, &calls, &mut Kinds::default(), &names, &frames);
        let target = function_of(wait_inside as *const () as u64);
        let found = frames.iter().skip(1).any(|&pc| function_of(pc - 1) == target);
        tx.send(()).unwrap();
        waiter.join().unwrap();
        // SAFETY: ours.
        unsafe { CloseHandle(thread) };
        assert!(found, "{} frames: {frames:x?}", frames.len());
        assert!(seen.waiting, "{seen:?}");
        let mut names = names.into_inner().unwrap();
        let place = place(&mut names, (seen.ours, seen.caller, seen.lib));
        assert!(place.contains("wait_inside <- ") && place.contains("wait_outside") && place.contains("(in "), "{place}");
    }

    /// The scheduler's states: this thread runs, a sleeper is blocked in a
    /// delay, and spinners held to one processor wait their turn ready.
    #[test]
    fn thread_states_are_read() {
        let stop = Arc::new(AtomicBool::new(false));
        let (id_tx, id_rx) = std::sync::mpsc::channel();
        let s = stop.clone();
        let tx = id_tx.clone();
        let sleeper = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            while !s.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        let sleeper_id = id_rx.recv().unwrap();
        let spinners: Vec<_> = (0..3)
            .map(|_| {
                let (s, tx) = (stop.clone(), id_tx.clone());
                std::thread::spawn(move || {
                    // SAFETY: this thread's own affinity: processor 1 alone.
                    unsafe { windows_sys::Win32::System::Threading::SetThreadAffinityMask(GetCurrentThread(), 2) };
                    // SAFETY: plain query.
                    tx.send(unsafe { GetCurrentThreadId() }).unwrap();
                    busy_spin(&s);
                })
            })
            .collect();
        let spinner_ids: Vec<u32> = (0..3).map(|_| id_rx.recv().unwrap()).collect();
        std::thread::sleep(Duration::from_millis(50));
        let mut query = StateQuery::new();
        let mut out = Vec::new();
        // SAFETY: plain query.
        let me = unsafe { GetCurrentThreadId() };
        let (mut me_running, mut sleeper_blocked, mut ready) = (0, 0, 0);
        for _ in 0..40 {
            assert!(query.read(&mut out));
            for &(id, state, reason) in &out {
                if id == me && state == 2 {
                    me_running += 1;
                }
                // std's sleep waits on a high-resolution timer: UserRequest.
                if id == sleeper_id && state == 5 && (reason as usize) < REASONS.len() {
                    sleeper_blocked += 1;
                }
                if spinner_ids.contains(&id) && matches!(state, 1 | 3 | 7) {
                    ready += 1;
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        stop.store(true, Ordering::Relaxed);
        sleeper.join().unwrap();
        for s in spinners {
            s.join().unwrap();
        }
        assert_eq!(me_running, 40);
        assert!(sleeper_blocked >= 35, "{sleeper_blocked} of 40");
        // Two of three should be ready at any time; tests beside this one and
        // the scheduler's own choices make it fewer (17 seen at the least).
        assert!(ready >= 3, "{ready} ready of 120");
    }

    /// A busy thread's running samples fall in its busy function, a sleeping
    /// thread counts as waiting, and threads that come and go meanwhile do
    /// no harm.
    #[test]
    fn the_sampler_finds_a_busy_function_and_a_sleeper() {
        let stop = Arc::new(AtomicBool::new(false));
        let ids = Arc::new(Mutex::new([0u32; 2]));
        let (s, i) = (stop.clone(), ids.clone());
        let busy = std::thread::Builder::new()
            .name("sampled busy".into())
            .spawn(move || {
                // SAFETY: plain query.
                i.lock().unwrap()[0] = unsafe { GetCurrentThreadId() };
                busy_spin(&s);
            })
            .unwrap();
        let (s, i) = (stop.clone(), ids.clone());
        let sleeper = std::thread::Builder::new()
            .name("sampled sleeper".into())
            .spawn(move || {
                // SAFETY: plain query.
                i.lock().unwrap()[1] = unsafe { GetCurrentThreadId() };
                while !s.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                }
            })
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let shared = Arc::new(Shared::new(2, no_frames));
        let sampler = spawn(shared.clone(), Duration::from_millis(2), not_title).unwrap();
        // Threads that start and end while sampled.
        let churn = Instant::now();
        while churn.elapsed() < Duration::from_millis(1500) {
            let short: Vec<_> = (0..8)
                .map(|_| std::thread::spawn(|| std::thread::sleep(Duration::from_millis(3))))
                .collect();
            for t in short {
                t.join().unwrap();
            }
        }
        shared.stop.store(true, Ordering::Release);
        sampler.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        busy.join().unwrap();
        sleeper.join().unwrap();
        let [busy_id, sleeper_id] = *ids.lock().unwrap();
        let mut state = shared.state.lock().unwrap();
        let mut names = shared.names.lock().unwrap();
        let lines = state.close_window(&mut names, 0, "test");
        for line in &lines {
            eprintln!("{line}");
        }
        let a = &state.total;
        assert!(a.passes > 200, "{} passes", a.passes);
        assert!(a.state_reads >= 2, "{} state reads", a.state_reads);
        let b = &a.threads[&busy_id];
        let target = function_of(busy_spin as *const () as u64);
        let spot = b.ours.get(&(target, 0)).map_or(0, |s| s.n);
        assert!(b.running > 200 && spot * 10 >= b.running * 8, "busy: {} running, {spot} in busy_spin", b.running);
        assert!(b.classes[VICTORY] * 10 >= b.running * 9);
        // Other tests run beside this one: the busy thread's own top place.
        let most = b.ours.iter().max_by_key(|(_, s)| s.n).map(|(f, _)| *f);
        assert_eq!(most, Some((target, 0)));
        assert!(a.spots.get(&(target, 0)).is_some_and(|s| s.n >= spot));
        assert!(b.states[RUNNING] + b.states[READY] >= b.states.iter().sum::<u64>() * 9 / 10, "{:?}", b.states);
        let s = &a.threads[&sleeper_id];
        assert!(s.waiting > 200 && s.running * 20 <= s.waiting, "sleeper: {} running, {} waiting", s.running, s.waiting);
        assert!(s.states[BLOCKED] * 10 >= s.states.iter().sum::<u64>() * 8, "{:?}", s.states);
        assert!(lines.iter().any(|l| l.contains("busy_spin")), "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains("\"sampled busy\"")), "{lines:#?}");
        assert!(lines.iter().any(|l| l.starts_with("PROFILE critical thread")), "{lines:#?}");
        assert!(lines.len() < 250);
    }
}
