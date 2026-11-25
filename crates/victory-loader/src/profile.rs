//! What every thread spends its time on, and where every thread is when
//! frames stop: `VICTORY_PROFILE=<file>` samples each thread that ran since
//! the last pass (`VICTORY_PROFILE_MS`, 10 by default), unwinds its stack,
//! and every `VICTORY_PROFILE_WINDOW_S` seconds (10) writes to the file, per
//! busy thread, its share of a core and its costliest stacks, with our own
//! functions and the system's named (dbghelp, from `victory.pdb` beside the
//! executable and the system DLLs' exports) and the title's as module+rva of
//! the function. When no frame begins for `VICTORY_STALL_MS` (750; 0: never),
//! every thread's stack goes to the file at once and every 2 s after while
//! the stall lasts (three at most), grouped by where they are.
//!
//! A stack is shown as the functions where it crosses from one module into
//! another, innermost first: `ntdll!NtWaitForSingleObject < KERNELBASE!… <
//! victory!victory_hle::…::wait … victory!…::WaitForSingleObjectEx <
//! title+0x8a1230 … title+0x12340` (`…`: more frames of that module).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use windows_sys::Wdk::System::Threading::{NtQueryInformationThread, ThreadBasicInformation};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, STILL_ACTIVE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext, RtlLookupFunctionEntry, SYMBOL_INFOW,
    SYMOPT_DEFERRED_LOADS, SYMOPT_FAIL_CRITICAL_ERRORS, SYMOPT_NO_PROMPTS, SYMOPT_UNDNAME,
    SymFromAddrW, SymInitializeW, SymSetOptions,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId, GetExitCodeThread,
    GetThreadDescription, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};
use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;

use crate::crash::{MAX_FRAMES, unwind_into};

/// Stack frames kept per sample (the innermost).
const KEPT: usize = 24;
/// A thread that ran fewer cycles than this since the last pass is idle.
const RAN: u64 = 50_000;

/// Starts the profiler if `VICTORY_PROFILE` names a file. `progress` counts
/// frames begun: a stall is a time it does not move.
pub fn profile(progress: fn() -> u64) {
    let Ok(path) = std::env::var("VICTORY_PROFILE") else {
        return;
    };
    let number = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let interval = Duration::from_millis(number("VICTORY_PROFILE_MS", 10).max(1));
    let window = Duration::from_secs(number("VICTORY_PROFILE_WINDOW_S", 10).max(1));
    let stall = number("VICTORY_STALL_MS", 750);
    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("VIctory: VICTORY_PROFILE: {path}: {e}");
            return;
        }
    };
    let spawned = std::thread::Builder::new()
        .name("victory profiler".into())
        .spawn(move || {
            let mut p = Profiler::new(std::io::BufWriter::new(file), progress, stall);
            p.run(interval, window);
        });
    if let Err(e) = spawned {
        eprintln!("VIctory: VICTORY_PROFILE: {e}");
    }
}

struct Tracked {
    handle: HANDLE,
    name: String,
    cycles: u64,
    /// Cycles run in this window.
    window: u64,
    /// Cycles run since the last stall check (half a second or so).
    recent: u64,
}

/// One stack's weight in a window: cycles and samples.
#[derive(Default, Clone, Copy)]
struct Weight {
    cycles: u64,
    samples: u32,
}

struct Profiler<W: std::io::Write> {
    out: W,
    progress: fn() -> u64,
    stall_ms: u64,
    me: u32,
    process: u32,
    threads: HashMap<u32, Tracked>,
    /// (thread, functions of the stack) → weight, this window.
    stacks: HashMap<(u32, Vec<u64>), Weight>,
    names: Names,
    frames: Vec<u64>,
    started: Instant,
}

impl<W: std::io::Write> Profiler<W> {
    fn new(out: W, progress: fn() -> u64, stall_ms: u64) -> Self {
        Profiler {
            out,
            progress,
            stall_ms,
            // SAFETY: plain queries.
            me: unsafe { GetCurrentThreadId() },
            // SAFETY: as above.
            process: unsafe { GetCurrentProcessId() },
            threads: HashMap::new(),
            stacks: HashMap::new(),
            names: Names::default(),
            frames: Vec::with_capacity(MAX_FRAMES),
            started: Instant::now(),
        }
    }

    fn run(&mut self, interval: Duration, window: Duration) {
        let _ = writeln!(
            self.out,
            "VIctory profile: a sample every {} ms of each thread that ran, a report every {} s, \
             stacks of all threads after {} ms without a frame (0: never)",
            interval.as_millis(),
            window.as_secs(),
            self.stall_ms
        );
        let mut window_start = Instant::now();
        let mut window_frames = (self.progress)();
        let mut last_refresh: Option<Instant> = None;
        let (mut frame, mut frame_at) = ((self.progress)(), Instant::now());
        let mut stall_dumps = 0u32;
        let mut next_dump = Instant::now();
        let tsc = Tsc::start();
        loop {
            let pass = Instant::now();
            if last_refresh.is_none_or(|t| t.elapsed() >= Duration::from_secs(1)) {
                self.refresh();
                last_refresh = Some(pass);
            }
            self.sample();
            // Stalls: no frame begun for a while.
            let now_frame = (self.progress)();
            if now_frame != frame {
                if stall_dumps > 0 {
                    let _ = writeln!(
                        self.out,
                        "== {} ({:.1} s): the stall ended after {} ms (frame {now_frame})\n",
                        clock(),
                        self.seconds(),
                        frame_at.elapsed().as_millis()
                    );
                    let _ = self.out.flush();
                }
                (frame, frame_at, stall_dumps) = (now_frame, Instant::now(), 0);
                for t in self.threads.values_mut() {
                    t.recent = 0;
                }
            } else if self.stall_ms > 0
                && frame_at.elapsed() >= Duration::from_millis(self.stall_ms)
                && stall_dumps < 3
                && Instant::now() >= next_dump
            {
                let waited = frame_at.elapsed().as_millis();
                self.dump_all(waited, frame);
                stall_dumps += 1;
                next_dump = Instant::now() + Duration::from_secs(2);
                for t in self.threads.values_mut() {
                    t.recent = 0;
                }
            }
            if window_start.elapsed() >= window {
                let frames = (self.progress)();
                let hz = tsc.hz();
                self.report(
                    window_start.elapsed(),
                    frames.saturating_sub(window_frames),
                    frames,
                    hz,
                );
                window_start = Instant::now();
                window_frames = frames;
            }
            std::thread::sleep(interval.saturating_sub(pass.elapsed()));
        }
    }

    fn seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Opens threads that started and closes those that ended.
    fn refresh(&mut self) {
        let mut alive = Vec::new();
        // SAFETY: a snapshot of this process's threads.
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
        // SAFETY: we opened it.
        unsafe { CloseHandle(snapshot) };
        for id in alive {
            if self.threads.contains_key(&id) {
                continue;
            }
            // SAFETY: opens a thread of our own process.
            let handle = unsafe {
                OpenThread(
                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                    0,
                    id,
                )
            };
            if handle.is_null() {
                continue;
            }
            let mut cycles = 0;
            // SAFETY: an open thread handle; writes one u64.
            unsafe { QueryThreadCycleTime(handle, &mut cycles) };
            let name = name_of(handle);
            self.threads.insert(
                id,
                Tracked {
                    handle,
                    name,
                    cycles,
                    window: 0,
                    recent: 0,
                },
            );
        }
        self.threads.retain(|_, t| {
            let mut code = 0;
            // SAFETY: an open thread handle.
            let ok = unsafe { GetExitCodeThread(t.handle, &mut code) } != 0;
            let running = ok && code == STILL_ACTIVE as u32;
            if !running {
                // SAFETY: we opened it.
                unsafe { CloseHandle(t.handle) };
            }
            running
        });
    }

    /// One pass: each thread that ran since the last is unwound, weighted
    /// by the cycles it ran.
    fn sample(&mut self) {
        let ids: Vec<u32> = self.threads.keys().copied().collect();
        for id in ids {
            let Some(t) = self.threads.get_mut(&id) else {
                continue;
            };
            let mut cycles = 0;
            // SAFETY: an open thread handle; writes one u64.
            if unsafe { QueryThreadCycleTime(t.handle, &mut cycles) } == 0 {
                continue;
            }
            let ran = cycles.saturating_sub(t.cycles);
            t.cycles = cycles;
            t.window += ran;
            t.recent += ran;
            if ran < RAN {
                continue;
            }
            let handle = t.handle;
            if !stack_of(handle, &mut self.frames) {
                continue;
            }
            let key: Vec<u64> = self
                .frames
                .iter()
                .take(KEPT)
                .map(|&pc| function_of(pc))
                .collect();
            let w = self.stacks.entry((id, key)).or_default();
            w.cycles += ran;
            w.samples += 1;
        }
    }

    fn report(&mut self, took: Duration, frames: u64, frame: u64, hz: f64) {
        let secs = took.as_secs_f64();
        let core = |cycles: u64| cycles as f64 / (hz * secs) * 100.0;
        let total: u64 = self.threads.values().map(|t| t.window).sum();
        let now = self.seconds();
        let mut text = format!(
            "== {} ({:.1}-{now:.1} s): {frames} frames begun ({:.1}/s, frame {frame}); all threads {:.2} cores\n",
            clock(),
            now - secs,
            frames as f64 / secs,
            core(total) / 100.0
        );
        let mut busy: Vec<(u32, u64)> =
            self.threads.iter().map(|(&id, t)| (id, t.window)).collect();
        busy.sort_by_key(|b| std::cmp::Reverse(b.1));
        // Where our own layer's time goes, over all threads: the function
        // of ours the title (or a system DLL) called, inclusive.
        let mut entries: HashMap<u64, u64> = HashMap::new();
        for ((_, stack), w) in &self.stacks {
            if let Some(e) = self.names.entry_of(stack) {
                *entries.entry(e).or_default() += w.cycles;
            }
        }
        for &(id, cycles) in busy.iter().filter(|&&(_, c)| core(c) >= 2.0).take(16) {
            let name = self
                .threads
                .get(&id)
                .map(|t| t.name.clone())
                .unwrap_or_default();
            let mut mine: Vec<(&Vec<u64>, Weight)> = self
                .stacks
                .iter()
                .filter(|((t, _), _)| *t == id)
                .map(|((_, s), w)| (s, *w))
                .collect();
            mine.sort_by_key(|m| std::cmp::Reverse(m.1.cycles));
            let sampled: u64 = mine.iter().map(|(_, w)| w.cycles).sum::<u64>().max(1);
            let _ = writeln!(
                text,
                "-- thread {id} \"{name}\": {:.0}% of a core",
                core(cycles)
            );
            for (stack, w) in mine.iter().take(5) {
                let line = self.names.chain(stack);
                let _ = writeln!(
                    text,
                    "   {:>3.0}%  {line}",
                    w.cycles as f64 / sampled as f64 * 100.0
                );
            }
        }
        let mut entries: Vec<(u64, u64)> = entries.into_iter().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        if !entries.is_empty() {
            let _ = writeln!(
                text,
                "-- our code by its outermost function (where the title or a thread start called in), all threads:"
            );
            for (f, cycles) in entries.into_iter().take(10) {
                let _ = writeln!(
                    text,
                    "   {:>5.1}% of a core  {}",
                    core(cycles),
                    self.names.name(f)
                );
            }
        }
        let _ = writeln!(self.out, "{text}");
        let _ = self.out.flush();
        self.stacks.clear();
        for t in self.threads.values_mut() {
            t.window = 0;
        }
    }

    /// Every thread's stack, grouped by where it is, busiest first.
    fn dump_all(&mut self, waited: u128, frame: u64) {
        let now = self.seconds();
        let mut groups: Vec<(Vec<u64>, Vec<(u32, String, u64)>)> = Vec::new();
        let ids: Vec<(u32, HANDLE, String, u64)> = self
            .threads
            .iter()
            .map(|(&id, t)| (id, t.handle, t.name.clone(), t.recent))
            .collect();
        for (id, handle, name, recent) in ids {
            if !stack_of(handle, &mut self.frames) {
                continue;
            }
            let key: Vec<u64> = self
                .frames
                .iter()
                .take(KEPT)
                .map(|&pc| function_of(pc))
                .collect();
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, list)) => list.push((id, name, recent)),
                None => groups.push((key, vec![(id, name, recent)])),
            }
        }
        let busiest = |list: &Vec<(u32, String, u64)>| list.iter().map(|x| x.2).max().unwrap_or(0);
        groups.sort_by(|a, b| {
            busiest(&b.1)
                .cmp(&busiest(&a.1))
                .then(b.1.len().cmp(&a.1.len()))
        });
        let mut text = format!(
            "== {} ({now:.1} s): STALL, no frame begun for {waited} ms (frame {frame}); every thread, \
             grouped by stack, busiest since the last frame first\n",
            clock()
        );
        for (stack, list) in &groups {
            let who: Vec<String> = list
                .iter()
                .take(6)
                .map(|(id, name, recent)| {
                    let ran = if *recent >= RAN {
                        format!(", ran {}M cycles", recent / 1_000_000)
                    } else {
                        String::new()
                    };
                    format!("{id} \"{name}\"{ran}")
                })
                .collect();
            let more = if list.len() > 6 {
                format!(" and {} more", list.len() - 6)
            } else {
                String::new()
            };
            let _ = writeln!(
                text,
                "-- {} thread(s): {}{more}",
                list.len(),
                who.join("; ")
            );
            let _ = writeln!(text, "   {}", self.names.chain(stack));
        }
        let _ = writeln!(self.out, "{text}");
        let _ = self.out.flush();
    }
}

/// The time of day (UTC, as the log's), `hh:mm:ss.mmm`.
fn clock() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let (s, ms) = (now.as_secs() % 86_400, now.subsec_millis());
    format!("{:02}:{:02}:{:02}.{ms:03}", s / 3600, s / 60 % 60, s % 60)
}

/// Suspends the thread just long enough to unwind it into `frames`
/// (allocated beforehand: it may hold the heap lock).
fn stack_of(thread: HANDLE, frames: &mut Vec<u64>) -> bool {
    frames.clear();
    let mut info = [0usize; 6];
    // SAFETY: writes THREAD_BASIC_INFORMATION (48 bytes) into `info`.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            info.as_mut_ptr().cast(),
            size_of_val(&info) as u32,
            std::ptr::null_mut(),
        )
    };
    let teb = info[1] as *const usize;
    if status != 0 || teb.is_null() {
        return false;
    }
    // SAFETY: the TEB of a live thread of this process: its NT_TIB holds
    // the stack base at +8 and the stack limit at +0x10.
    let (high, low) = unsafe { (teb.add(1).read(), teb.add(2).read()) };
    let mut context = Aligned(unsafe {
        // SAFETY: zeroed is a valid CONTEXT.
        std::mem::zeroed::<CONTEXT>()
    });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: suspends, reads and resumes a thread that is not this one;
    // nothing in between allocates or takes a lock the thread could hold
    // besides the unwinder's own.
    unsafe {
        if SuspendThread(thread) == u32::MAX {
            return false;
        }
        if GetThreadContext(thread, &mut context.0) != 0 {
            // Only the committed, readable memory around the stack pointer:
            // the TEB's limits are not enough (all11i-warm, 2026-11-25: the
            // unwinder read a reserved, never committed page and the
            // profiler took the process down). A stack pointer outside such
            // memory gives the leaf alone.
            match committed_around(context.0.Rsp as usize) {
                Some((start, end)) => {
                    unwind_into(context.0, low.max(start), high.min(end), frames);
                }
                None => frames.push(context.0.Rip),
            }
        }
        ResumeThread(thread);
    }
    !frames.is_empty()
}

/// The committed, readable run of pages holding `at` (one region as
/// `VirtualQuery` sees it), or `None`.
fn committed_around(at: usize) -> Option<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    // SAFETY: zeroed is a valid MEMORY_BASIC_INFORMATION.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: VirtualQuery writes `info` (its size is passed) and reads
    // nothing at `at`.
    let got = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    let readable = info.State == MEM_COMMIT && info.Protect & (PAGE_GUARD | PAGE_NOACCESS) == 0 && info.Protect != 0;
    (got != 0 && readable).then(|| {
        let start = info.BaseAddress as usize;
        (start, start + info.RegionSize)
    })
}

/// The start of the function holding `pc`, or `pc` itself where no unwind
/// table covers it (a leaf, generated code).
fn function_of(pc: u64) -> u64 {
    let mut image = 0;
    // SAFETY: looks up the function table entry covering `pc`, if any.
    let entry = unsafe { RtlLookupFunctionEntry(pc, &mut image, std::ptr::null_mut()) };
    if entry.is_null() {
        pc
    } else {
        // SAFETY: a valid entry; its first field is the function's start RVA.
        image + u64::from(unsafe { (*entry).BeginAddress })
    }
}

#[repr(C, align(16))]
struct Aligned(CONTEXT);

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        windows_sys::Win32::Foundation::LocalFree(name.cast::<c_void>());
        text
    }
}

/// The TSC rate, for turning a thread's cycles into a share of a core.
struct Tsc(u64, Instant);
impl Tsc {
    fn start() -> Self {
        // SAFETY: reads the time-stamp counter.
        Tsc(unsafe { core::arch::x86_64::_rdtsc() }, Instant::now())
    }
    fn hz(&self) -> f64 {
        // SAFETY: as above.
        let ticks = unsafe { core::arch::x86_64::_rdtsc() }.saturating_sub(self.0);
        (ticks as f64 / self.1.elapsed().as_secs_f64().max(1e-3)).max(1.0)
    }
}

/// Function names: ours and the system's from dbghelp, the title's as
/// module+rva. Only the profiler thread uses dbghelp.
#[derive(Default)]
struct Names {
    ready: bool,
    /// Our executable's module name (`victory`).
    own: String,
    known: HashMap<u64, (String, String)>,
}

impl Names {
    fn init(&mut self) {
        if self.ready {
            return;
        }
        self.ready = true;
        self.own = std::env::current_exe()
            .ok()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_default();
        // Symbols only from beside the executable and the modules' own
        // exports: no symbol server, no environment search path.
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let path: Vec<u16> = dir.encode_utf16().chain([0]).collect();
        // SAFETY: dbghelp, used from this thread only.
        unsafe {
            SymSetOptions(
                SYMOPT_UNDNAME
                    | SYMOPT_DEFERRED_LOADS
                    | SYMOPT_FAIL_CRITICAL_ERRORS
                    | SYMOPT_NO_PROMPTS,
            );
            SymInitializeW(GetCurrentProcess(), path.as_ptr(), 1);
        }
    }

    /// (module, function) for a function start.
    fn lookup(&mut self, f: u64) -> (String, String) {
        if let Some(k) = self.known.get(&f) {
            return k.clone();
        }
        self.init();
        let located = crate::crash::locate(f as usize);
        let file = located.split('+').next().unwrap_or("");
        let module = match file.rsplit_once('.') {
            Some((stem, ext))
                if ext.eq_ignore_ascii_case("exe") || ext.eq_ignore_ascii_case("dll") =>
            {
                stem
            }
            _ => file,
        }
        .to_string();
        let mut name = located.clone();
        if !module.is_empty() && !located.starts_with("0x") {
            #[repr(C)]
            struct Buffer(SYMBOL_INFOW, [u16; 512]);
            // SAFETY: zeroed is valid for both parts.
            let mut buffer: Buffer = unsafe { std::mem::zeroed() };
            buffer.0.SizeOfStruct = size_of::<SYMBOL_INFOW>() as u32;
            buffer.0.MaxNameLen = 512;
            let mut displacement = 0;
            // SAFETY: dbghelp from this thread only; the buffer has room
            // for MaxNameLen more characters after the struct.
            let found =
                unsafe { SymFromAddrW(GetCurrentProcess(), f, &mut displacement, &mut buffer.0) }
                    != 0;
            if found {
                let len = (buffer.0.NameLen as usize).min(512);
                // SAFETY: the name starts at `Name` and runs into the room after it.
                let units = unsafe { std::slice::from_raw_parts(buffer.0.Name.as_ptr(), len) };
                let symbol = String::from_utf16_lossy(units);
                // An export far before the function is not its name.
                if displacement < 0x1000 || module == self.own {
                    name = format!("{module}!{}", short(&symbol));
                }
            }
        }
        let k = (module, name);
        self.known.insert(f, k.clone());
        k
    }

    fn name(&mut self, f: u64) -> String {
        self.lookup(f).1
    }

    /// The stack's module crossings, innermost first.
    fn chain(&mut self, stack: &[u64]) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut i = 0;
        while i < stack.len() && parts.len() < 8 {
            let (module, first) = self.lookup(stack[i]);
            let mut j = i;
            while j + 1 < stack.len() && self.lookup(stack[j + 1]).0 == module {
                j += 1;
            }
            if j > i {
                let last = self.name(stack[j]);
                let gap = if j > i + 1 { " … " } else { " < " };
                parts.push(format!("{first}{gap}{last}"));
            } else {
                parts.push(first);
            }
            i = j + 1;
        }
        parts.join(" < ")
    }

    /// The function of ours outermost in the stack's first run of ours,
    /// where something else called into us.
    fn entry_of(&mut self, stack: &[u64]) -> Option<u64> {
        let start = stack.iter().position(|&f| self.lookup(f).0 == self.own)?;
        let mut last = stack[start];
        for &f in &stack[start + 1..] {
            if self.lookup(f).0 != self.own {
                break;
            }
            last = f;
        }
        Some(last)
    }
}

/// A symbol cut to a readable length: generic arguments and closure
/// paths dropped.
fn short(symbol: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in symbol.chars() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let out = out.replace("::{{closure}}", "{closure}");
    if out.len() > 110 {
        let cut = out
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= 110)
            .last()
            .unwrap_or(0);
        format!("{}…", &out[..cut])
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_drop_generics() {
        assert_eq!(short("a::b<c::d<e>>::f"), "a::b::f");
        assert_eq!(short("x::{{closure}}"), "x{closure}");
    }

    #[test]
    fn a_busy_thread_shows_in_a_report() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let busy = std::thread::Builder::new()
            .name("busy one".into())
            .spawn(move || {
                let mut x = 0u64;
                while !s.load(std::sync::atomic::Ordering::Relaxed) {
                    x = std::hint::black_box(x.wrapping_mul(31).wrapping_add(7));
                }
                x
            })
            .unwrap();
        fn frames() -> u64 {
            0
        }
        let mut p = Profiler::new(Vec::new(), frames, 50);
        p.refresh();
        let tsc = Tsc::start();
        let start = Instant::now();
        for _ in 0..30 {
            p.sample();
            std::thread::sleep(Duration::from_millis(5));
        }
        p.report(start.elapsed(), 0, 0, tsc.hz());
        p.dump_all(60, 0);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        busy.join().unwrap();
        let text = String::from_utf8(p.out).unwrap();
        eprintln!("{text}");
        assert!(text.contains("\"busy one\""), "{text}");
        assert!(text.contains("STALL"), "{text}");
    }

    /// A stack aligned past 16 bytes: the function keeps a frame register.
    #[repr(C, align(64))]
    struct Aligned64([u8; 64]);

    /// Spins with garbage in `rbp` (all ones, then zero) in the middle of a
    /// function whose unwind information keeps its frame in `rbp` (its
    /// 64-byte aligned local), restoring it between spins; until `stop`.
    #[inline(never)]
    fn spin_with_garbage_frame(stop: &std::sync::atomic::AtomicBool) -> u8 {
        let local = Aligned64([1; 64]);
        let local = std::hint::black_box(&local);
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            // SAFETY: `rbp` is saved and restored around the spin, and
            // nothing in between reads the stack or calls anything.
            unsafe {
                std::arch::asm!(
                    "push rbp",
                    "mov rbp, -1",
                    "mov {n}, 20000",
                    "2:",
                    "dec {n}",
                    "jnz 2b",
                    "xor ebp, ebp",
                    "mov {n}, 20000",
                    "3:",
                    "dec {n}",
                    "jnz 3b",
                    "pop rbp",
                    n = out(reg) _,
                );
            }
        }
        local.0[0]
    }

    /// The profiler samples a thread caught with garbage in its frame
    /// register mid-function, again and again, and comes back each time
    /// with the frames it could read: the unwinder never follows the
    /// garbage (`crate::crash::step_reads_fit`).
    #[test]
    fn a_thread_with_garbage_in_its_frame_register_is_sampled_safely() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let spinner = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            spin_with_garbage_frame(&s)
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process, opened for sampling.
        let thread = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, 0, id) };
        assert!(!thread.is_null());
        let mut frames = Vec::with_capacity(MAX_FRAMES);
        let mut sampled = 0;
        for _ in 0..3000 {
            if stack_of(thread, &mut frames) {
                sampled += 1;
                assert!(frames.len() <= MAX_FRAMES);
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        spinner.join().unwrap();
        // SAFETY: our handle.
        unsafe { CloseHandle(thread) };
        assert!(sampled > 2900, "{sampled} of 3000 samples");
    }

    /// The unwind of a context with garbage in every register but `Rip`
    /// and `Rsp`, from many places in this thread's stack, stops where a
    /// step would read outside it.
    #[test]
    fn garbage_registers_stop_the_unwind_inside_the_stack() {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureContext;
        let mut context = Aligned(unsafe {
            // SAFETY: zeroed is a valid CONTEXT.
            std::mem::zeroed::<CONTEXT>()
        });
        // SAFETY: fills our aligned context with this thread's registers.
        unsafe { RtlCaptureContext(&mut context.0) };
        let (low, high) = committed_around(context.0.Rsp as usize).expect("this thread's stack");
        let mut frames = Vec::with_capacity(MAX_FRAMES);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for k in 0..20_000u64 {
            let mut c = context.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let garbage = [u64::MAX, 0, 8, x, x | 0xffff_0000_0000_0000][k as usize % 5];
            (c.Rbp, c.Rbx, c.Rsi, c.Rdi, c.R12, c.R13, c.R14, c.R15) =
                (garbage, garbage, garbage, garbage, garbage, garbage, garbage, garbage);
            // Rsp anywhere in the stack (8-aligned), and sometimes at its ends.
            c.Rsp = match k % 3 {
                0 => (low as u64 + (x % (high - low) as u64)) & !7,
                1 => high as u64 - 8,
                _ => low as u64,
            };
            frames.clear();
            crate::crash::unwind_into(c, low, high, &mut frames);
            assert!(!frames.is_empty() && frames.len() <= MAX_FRAMES);
        }
    }
}
