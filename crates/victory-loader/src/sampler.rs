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
//! `NtRemoveIoCompletion`) counts as waiting, not running. A running sample
//! with a frame of ours inside (nearer the leaf than) its first title frame
//! is charged to the innermost such function of ours, so time in the driver
//! or in ntdll that our code called is ours; frames of ours outside every
//! title frame are thread starts and are not charged.
//!
//! What it cannot see: time in the kernel is counted where the call came from
//! (a running sample whose leaf is a system call that does not wait); code
//! without unwind data inside a stack (hand-written title code, the
//! driver's) can end the walk early or send it wrong; inlined functions count
//! as the function they were inlined into; and a run ended from outside
//! (`Stop-Process`) has the 30 s reports only.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, STILL_ACTIVE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, RtlPcToFileHeader,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
use windows_sys::Win32::System::Threading::{
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, GetCurrentProcessId,
    GetCurrentThread, GetCurrentThreadId, GetExitCodeThread, INFINITE, OpenThread, ResumeThread,
    SetThreadPriority, SetWaitableTimer, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_PRIORITY_ABOVE_NORMAL, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
    TIMER_ALL_ACCESS, WaitForSingleObject,
};
use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;

use crate::crash::{Aligned16, MAX_FRAMES, unwind_into};
use crate::profile::{Names, committed_around, function_of, name_of};

/// Words of a stack copied per sample (64 KiB).
const COPY_WORDS: usize = 64 * 1024 / 8;
const REPORT_EVERY: Duration = Duration::from_secs(30);
const REFRESH_EVERY: Duration = Duration::from_secs(1);
/// A thread that ran fewer cycles than this since the last pass has not run
/// (blocked, or ready and not given a core): suspending a waiting thread
/// wakes it for a kernel APC, so its count moves a little with no work of
/// its own (a test with an exact compare suspended every thread at every
/// pass). At 2 ms this is about 1.5% of a core.
const RAN: u64 = 100_000;

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

/// Starts the sampler if `VICTORY_PROFILE` is a number of milliseconds
/// (not 0). `is_title` says whether a module base is one of the title's
/// images.
pub fn sample_threads(is_title: fn(usize) -> bool) {
    let Ok(value) = std::env::var("VICTORY_PROFILE") else {
        return;
    };
    let Ok(ms) = value.trim().parse::<u64>() else {
        return;
    };
    if ms == 0 || SHARED.get().is_some() {
        return;
    }
    let shared = Arc::new(Shared::new(ms));
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
        let mut lines = state.close_window("the last");
        let state = &mut *state;
        lines.extend(report(&mut state.names, &state.thread_names, "the whole run", &state.total, state.interval_ms));
        lines
    };
    for line in lines {
        tracing::info!("{line}");
    }
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// What the sampler thread and the exit's report share.
struct Shared {
    state: Mutex<State>,
    stop: AtomicBool,
    stopped: AtomicBool,
    reported: AtomicBool,
}

impl Shared {
    fn new(interval_ms: u64) -> Self {
        Shared {
            state: Mutex::new(State {
                window: Agg::default(),
                total: Agg::default(),
                names: Names::default(),
                thread_names: HashMap::new(),
                interval_ms,
                window_start: Instant::now(),
            }),
            stop: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            reported: AtomicBool::new(false),
        }
    }
}

struct State {
    window: Agg,
    total: Agg,
    names: Names,
    thread_names: HashMap<u32, String>,
    interval_ms: u64,
    window_start: Instant,
}

impl State {
    /// Ends the window: its report, and its counts added to the total.
    fn close_window(&mut self, what: &str) -> Vec<String> {
        self.window.secs = self.window_start.elapsed().as_secs_f64();
        self.window_start = Instant::now();
        let window = std::mem::take(&mut self.window);
        let what = format!("{what} {:.0} s", window.secs);
        let lines = report(&mut self.names, &self.thread_names, &what, &window, self.interval_ms);
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

#[derive(Default, Clone)]
struct PerThread {
    running: u64,
    waiting: u64,
    /// Running samples by the class of their leaf.
    classes: [u64; CLASSES],
    /// Running samples by our innermost function.
    ours: HashMap<u64, Spot>,
    /// Waiting samples by our innermost function (0: none).
    waits: HashMap<u64, u64>,
}

#[derive(Default)]
struct Agg {
    passes: u64,
    secs: f64,
    pass_us: u64,
    pass_us_max: u64,
    suspended: u64,
    suspended_us_max: u64,
    /// Time threads spent suspended, in nanoseconds.
    held_ns: u64,
    threads: HashMap<u32, PerThread>,
    /// Running samples of all threads by our innermost function, else the
    /// leaf's function.
    spots: HashMap<u64, Spot>,
}

impl Agg {
    fn add(&mut self, id: u32, seen: Seen, unchanged: bool) {
        let t = self.threads.entry(id).or_default();
        if seen.waiting || unchanged {
            t.waiting += 1;
            *t.waits.entry(seen.ours.unwrap_or(0)).or_default() += 1;
        } else {
            t.running += 1;
            t.classes[seen.leaf] += 1;
            if let Some(f) = seen.ours {
                t.ours.entry(f).or_default().add(seen.leaf);
            }
            self.spots.entry(seen.spot).or_default().add(seen.leaf);
        }
    }

    fn merge(&mut self, other: &Agg) {
        self.passes += other.passes;
        self.secs += other.secs;
        self.pass_us += other.pass_us;
        self.pass_us_max = self.pass_us_max.max(other.pass_us_max);
        self.suspended += other.suspended;
        self.suspended_us_max = self.suspended_us_max.max(other.suspended_us_max);
        self.held_ns += other.held_ns;
        for (id, o) in &other.threads {
            let t = self.threads.entry(*id).or_default();
            t.running += o.running;
            t.waiting += o.waiting;
            for (a, b) in t.classes.iter_mut().zip(o.classes) {
                *a += b;
            }
            for (f, s) in &o.ours {
                t.ours.entry(*f).or_default().merge(s);
            }
            for (f, n) in &o.waits {
                *t.waits.entry(*f).or_default() += n;
            }
        }
        for (f, s) in &other.spots {
            self.spots.entry(*f).or_default().merge(s);
        }
    }
}

/// One sample, classed.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// The leaf is a wait or delay system call.
    waiting: bool,
    /// Class of the leaf.
    leaf: usize,
    /// Our innermost function inside the first title frame.
    ours: Option<u64>,
    /// `ours`, else the leaf's function.
    spot: u64,
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

/// Classes a sample by its frames (innermost first).
fn classify(modules: &mut Modules, calls: &[(u64, bool)], frames: &[u64]) -> Seen {
    let rip = frames[0];
    let leaf = modules.class(rip);
    let waiting = leaf == SYSTEM && in_wait(calls, rip);
    let mut ours = None;
    for (i, &pc) in frames.iter().enumerate() {
        // A return address: the call before it is in the caller.
        let pc = if i == 0 { pc } else { pc.saturating_sub(1) };
        match if i == 0 { leaf } else { modules.class(pc) } {
            TITLE => break,
            VICTORY => {
                ours = Some(function_of(pc));
                break;
            }
            _ => {}
        }
    }
    let spot = ours.unwrap_or_else(|| function_of(rip));
    Seen { waiting, leaf, ours, spot }
}

struct Sampler {
    shared: Arc<Shared>,
    me: u32,
    process: u32,
    threads: HashMap<u32, Tracked>,
    modules: Modules,
    calls: Vec<(u64, bool)>,
    stack: Box<[u64]>,
    frames: Vec<u64>,
    results: Vec<(u32, Seen, bool)>,
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
            stack: vec![0u64; COPY_WORDS].into_boxed_slice(),
            frames: Vec::with_capacity(MAX_FRAMES),
            results: Vec::new(),
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
                (state.window_start.elapsed() >= REPORT_EVERY).then(|| state.close_window("the last"))
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
    /// new and still unnamed ones.
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
        let mut named = Vec::new();
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
        {
            let state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            for (&id, t) in &self.threads {
                if state.thread_names.get(&id).is_none_or(String::is_empty) {
                    named.push((id, t.handle));
                }
            }
        }
        let named: Vec<(u32, String)> = named.into_iter().map(|(id, h)| (id, name_of(h))).collect();
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        for (id, name) in named {
            state.thread_names.insert(id, name);
        }
    }

    /// One pass over every thread, then its counts added under the lock.
    fn pass(&mut self) {
        let started = Instant::now();
        let Sampler { threads, modules, calls, stack, frames, results, .. } = self;
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
            let seen = classify(modules, calls, frames);
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
    }
}

/// The report's lines for `a`.
fn report(
    names: &mut Names,
    thread_names: &HashMap<u32, String>,
    what: &str,
    a: &Agg,
    interval_ms: u64,
) -> Vec<String> {
    let running: u64 = a.threads.values().map(|t| t.running).sum();
    let waiting: u64 = a.threads.values().map(|t| t.waiting).sum();
    let passes = a.passes.max(1) as f64;
    let pct = |n: u64, of: u64| n as f64 * 100.0 / of.max(1) as f64;
    let mut out = vec![format!(
        "PROFILE {what}: {} passes asked every {interval_ms} ms ({:.0} a second); the sampler took {:.0} us a pass (most {} us), suspended {:.1} threads a pass for {:.1} us each (the longest {} us); {} threads; running {:.2} cores, waiting {:.1} threads",
        a.passes,
        a.passes as f64 / a.secs.max(1e-3),
        a.pass_us as f64 / passes,
        a.pass_us_max,
        a.suspended as f64 / passes,
        a.held_ns as f64 / 1000.0 / a.suspended.max(1) as f64,
        a.suspended_us_max,
        a.threads.len(),
        running as f64 / passes,
        waiting as f64 / passes,
    )];
    if running == 0 {
        return out;
    }
    let split = |s: &Spot| -> String {
        let mut c: Vec<usize> = (0..CLASSES).filter(|&c| s.leaf[c] > 0).collect();
        c.sort_by_key(|&c| Reverse(s.leaf[c]));
        c.iter().take(2).map(|&c| format!("{} {:.0}%", CLASS_NAMES[c], pct(s.leaf[c], s.n))).collect::<Vec<_>>().join(", ")
    };
    let mut threads: Vec<(u32, &PerThread)> =
        a.threads.iter().filter(|(_, t)| t.running * 100 > running).map(|(&id, t)| (id, t)).collect();
    threads.sort_by_key(|(_, t)| Reverse(t.running));
    for (id, t) in threads {
        let name = thread_names.get(&id).map(String::as_str).unwrap_or("");
        let classes: Vec<String> = (0..CLASSES)
            .filter(|&c| t.classes[c] > 0)
            .map(|c| format!("{} {:.0}%", CLASS_NAMES[c], pct(t.classes[c], t.running)))
            .collect();
        out.push(format!(
            "PROFILE thread {id} \"{name}\": {:.1}% of all running ({:.2} cores); its samples running {:.0}%, waiting {:.0}%; running by leaf module: {}",
            pct(t.running, running),
            t.running as f64 / passes,
            pct(t.running, t.running + t.waiting),
            pct(t.waiting, t.running + t.waiting),
            classes.join(", ")
        ));
        let mut ours: Vec<(&u64, &Spot)> = t.ours.iter().collect();
        ours.sort_by_key(|(_, s)| Reverse(s.n));
        let charged: u64 = ours.iter().map(|(_, s)| s.n).sum();
        for (f, s) in ours.iter().take(12) {
            out.push(format!(
                "PROFILE thread {id}   {:5.1}%  {}  [leaf {}]",
                pct(s.n, t.running),
                names.name(**f),
                split(s)
            ));
        }
        if charged < t.running {
            out.push(format!(
                "PROFILE thread {id}   {:5.1}%  (none of ours inside the title's frames)",
                pct(t.running - charged, t.running)
            ));
        }
        if t.waiting > 0 {
            let mut waits: Vec<(&u64, &u64)> = t.waits.iter().collect();
            waits.sort_by_key(|(_, n)| Reverse(**n));
            let waits: Vec<String> = waits
                .iter()
                .take(3)
                .map(|&(&f, &n)| {
                    let at = if f == 0 { "(none of ours)".to_string() } else { names.name(f) };
                    format!("{:.0}% {at}", pct(n, t.waiting))
                })
                .collect();
            out.push(format!("PROFILE thread {id}   waits in: {}", waits.join("; ")));
        }
    }
    let mut spots: Vec<(&u64, &Spot)> = a.spots.iter().collect();
    spots.sort_by_key(|(_, s)| Reverse(s.n));
    out.push(format!(
        "PROFILE top 25 over all threads ({running} running samples): our innermost function under the title's frames, else the leaf's function"
    ));
    for (i, (f, s)) in spots.iter().take(25).enumerate() {
        out.push(format!(
            "PROFILE top {:2} {:5.1}%  {}  [leaf {}]",
            i + 1,
            pct(s.n, running),
            names.name(**f),
            split(s)
        ));
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

    fn not_title(_: usize) -> bool {
        false
    }

    #[test]
    fn wait_calls_are_found() {
        let calls = system_calls();
        assert!(calls.iter().filter(|c| c.1).count() > 10, "{} calls", calls.len());
        assert!(waits(b"NtWaitForSingleObject") && waits(b"NtDelayExecution"));
        assert!(!waits(b"NtYieldExecution") && !waits(b"NtReadFile"));
    }

    /// A thread blocked in a function of ours is unwound from its stack's
    /// copy through that function.
    #[test]
    fn a_copied_stack_unwinds_through_its_callers() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (id_tx, id_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            // SAFETY: plain query.
            id_tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            wait_inside(&rx);
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
        let seen = classify(&mut modules, &calls, &frames);
        let target = function_of(wait_inside as *const () as u64);
        let found = frames.iter().skip(1).any(|&pc| function_of(pc - 1) == target);
        tx.send(()).unwrap();
        waiter.join().unwrap();
        // SAFETY: ours.
        unsafe { CloseHandle(thread) };
        assert!(found, "{} frames: {frames:x?}", frames.len());
        assert!(seen.waiting, "{seen:?}");
        assert!(seen.ours.is_some(), "{seen:?}");
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
        let shared = Arc::new(Shared::new(2));
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
        let lines = state.close_window("test");
        for line in &lines {
            eprintln!("{line}");
        }
        let a = &state.total;
        assert!(a.passes > 200, "{} passes", a.passes);
        let b = &a.threads[&busy_id];
        let target = function_of(busy_spin as *const () as u64);
        let spot = b.ours.get(&target).map_or(0, |s| s.n);
        assert!(b.running > 200 && spot * 10 >= b.running * 8, "busy: {} running, {spot} in busy_spin", b.running);
        assert!(b.classes[VICTORY] * 10 >= b.running * 9);
        // Other tests run beside this one: the busy thread's own top place.
        let most = b.ours.iter().max_by_key(|(_, s)| s.n).map(|(f, _)| *f);
        assert_eq!(most, Some(target));
        assert!(a.spots.get(&target).is_some_and(|s| s.n >= spot));
        let s = &a.threads[&sleeper_id];
        assert!(s.waiting > 200 && s.running * 20 <= s.waiting, "sleeper: {} running, {} waiting", s.running, s.waiting);
        assert!(lines.iter().any(|l| l.contains("busy_spin")), "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains("\"sampled busy\"")), "{lines:#?}");
    }
}
