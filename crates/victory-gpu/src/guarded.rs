//! Reads of title memory that may meet an unmapped page, as a plain copy
//! (`rep movsb`) that an exception handler stops at the first page that
//! faults. `ReadProcessMemory`, which the reads went through, goes into
//! the kernel and copies through it: about a fifth of the submitting
//! thread's time in the world (the refreshes of host copies).
//! `VICTORY_GUARDED_READS=0` reads through the kernel, as before.

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, NTSTATUS};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};

// `victory_guarded_copy(dst, src, len) -> 1`, or 0 once the handler below
// moved a fault at `victory_guarded_copy_at` to `victory_guarded_copy_fail`
// (the bytes before the faulting one are copied then).
core::arch::global_asm!(
    ".globl victory_guarded_copy",
    ".globl victory_guarded_copy_at",
    ".globl victory_guarded_copy_fail",
    "victory_guarded_copy:",
    "push rdi",
    "push rsi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "victory_guarded_copy_at:",
    "rep movsb",
    "mov eax, 1",
    "pop rsi",
    "pop rdi",
    "ret",
    "victory_guarded_copy_fail:",
    "xor eax, eax",
    "pop rsi",
    "pop rdi",
    "ret",
);

// The same for a write into title memory (`VICTORY_COPY_PLAIN=1`): 0 once
// a fault at `victory_guarded_write_at` (a destination page that is not
// there or not writable) was moved to `victory_guarded_write_fail`.
core::arch::global_asm!(
    ".globl victory_guarded_write",
    ".globl victory_guarded_write_at",
    ".globl victory_guarded_write_fail",
    "victory_guarded_write:",
    "push rdi",
    "push rsi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "victory_guarded_write_at:",
    "rep movsb",
    "mov eax, 1",
    "pop rsi",
    "pop rdi",
    "ret",
    "victory_guarded_write_fail:",
    "xor eax, eax",
    "pop rsi",
    "pop rdi",
    "ret",
);

unsafe extern "C" {
    fn victory_guarded_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_guarded_copy_at: u8;
    static victory_guarded_copy_fail: u8;
    fn victory_guarded_write(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_guarded_write_at: u8;
    static victory_guarded_write_fail: u8;
}

/// Whether reads go through the guarded copy (`VICTORY_GUARDED_READS`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_GUARDED_READS").is_none_or(|v| v != "0") && {
            install();
            true
        }
    })
}

/// Turns a read fault of the guarded copy into its failure return. First
/// in line, ahead of the crash reporter: it looks at nothing but the
/// faulting instruction's address.
fn install() {
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let (r, c) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        // Only the symbols' addresses are taken.
        let (at, fail) = (
            std::ptr::addr_of!(victory_guarded_copy_at) as u64,
            std::ptr::addr_of!(victory_guarded_copy_fail) as u64,
        );
        const READ: usize = 0;
        if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && c.Rip == at
            && r.NumberParameters >= 2
            && r.ExceptionInformation[0] == READ
        {
            c.Rip = fail;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // The guarded write: a destination page that is not there, not
        // writable, or a guard page (the system has taken its guard off).
        const WRITE: usize = 1;
        let violation = r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && r.ExceptionInformation[0] == WRITE;
        if c.Rip == std::ptr::addr_of!(victory_guarded_write_at) as u64
            && (violation || r.ExceptionCode == EXCEPTION_GUARD_PAGE as NTSTATUS)
        {
            c.Rip = std::ptr::addr_of!(victory_guarded_write_fail) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    // SAFETY: the handler only moves the guarded copy's own faults.
    unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    }
}

/// Copies `len` bytes from `src` to `dst`; false if a page of the source is
/// not readable (the bytes before it are copied then).
///
/// # Safety
///
/// `dst` must be writable host memory of `len` bytes, not overlapping the
/// source, and [`on`] true (the handler installed).
pub(crate) unsafe fn copy(src: *const u8, dst: *mut u8, len: usize) -> bool {
    // SAFETY: as the caller vouches; the source is checked by the fault.
    unsafe { victory_guarded_copy(dst, src, len) != 0 }
}

/// `VICTORY_COPY_PLAIN=1` (off by default; needs the guarded reads on):
/// the rest of soldsawn/victory-nvidia-8gb's switch of that name that the
/// guarded reads above do not already do (fxceless/victory-fork a25370661
/// and PR #1032):
/// - what shaders wrote goes back into title memory with a plain copy
///   too ([`write`]), the kernel's write behind it for a page that is not
///   there or not writable;
/// - reads of under [`LEAST`] bytes, and reads from a 64 KiB block a
///   guarded read faulted in during the last two seconds, go straight to
///   the kernel ([`kernel_first`]): a fault and its handler cost tens of
///   microseconds where the kernel's failed read costs one.
pub(crate) fn plain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = on() && std::env::var_os("VICTORY_COPY_PLAIN").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "writes into title memory are made in place, the kernel's write behind them; small reads and reads \
                 where one faulted lately go to the kernel (VICTORY_COPY_PLAIN=1)"
            );
        }
        on
    })
}

/// Writes in place, their bytes, writes that met a page and fell back;
/// reads sent to the kernel as small, and as faulted lately; read faults.
static COUNT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The fewest bytes read in place with [`plain`]: a read of a byte or a
/// word is how the layer asks whether a page is there, and is expected to
/// fail.
const LEAST: usize = 256;

/// The 64 KiB blocks of title memory a guarded read faulted in lately, by
/// the block's number in the place its low bits name. Emptied every two
/// seconds: the title maps the memory in the end.
static RECENT: [AtomicUsize; 1024] = [const { AtomicUsize::new(0) }; 1024];
/// When [`RECENT`] was last emptied (`crate::host::now_ms`).
static EMPTIED: AtomicU64 = AtomicU64::new(0);

/// The numbers of the first and last 64 KiB block of `[at, at + len)`,
/// never 0 (an empty place).
fn blocks(at: usize, len: usize) -> (usize, usize) {
    ((at >> 16) | 1 << 63, ((at + len.max(1) - 1) >> 16) | 1 << 63)
}

/// With [`plain`]: whether a read of `len` bytes at `src` goes to the
/// kernel without being tried in place.
pub(crate) fn kernel_first(src: *const u8, len: usize) -> bool {
    if !plain() {
        return false;
    }
    if len < LEAST {
        COUNT[3].fetch_add(1, Relaxed);
        return true;
    }
    let now = crate::host::now_ms();
    let emptied = EMPTIED.load(Relaxed);
    if now.saturating_sub(emptied) >= 2_000 && EMPTIED.compare_exchange(emptied, now, Relaxed, Relaxed).is_ok() {
        for place in &RECENT {
            place.store(0, Relaxed);
        }
        return false;
    }
    let (first, last) = blocks(src as usize, len);
    let lately = RECENT[first & 1023].load(Relaxed) == first || RECENT[last & 1023].load(Relaxed) == last;
    if lately {
        COUNT[4].fetch_add(1, Relaxed);
    }
    lately
}

/// With [`plain`]: a guarded read of `len` bytes at `src` faulted.
pub(crate) fn note_fault(src: *const u8, len: usize) {
    if !plain() {
        return;
    }
    COUNT[5].fetch_add(1, Relaxed);
    let (first, last) = blocks(src as usize, len);
    RECENT[first & 1023].store(first, Relaxed);
    RECENT[last & 1023].store(last, Relaxed);
}

/// With [`plain`]: copies `len` bytes of host memory at `src` into title
/// memory at `dst` in place. False if it is off, or a destination page was
/// not there or not writable (the pages before it are written then, as by
/// the kernel's write, which the caller goes on to).
pub(crate) fn write(dst: *mut u8, src: *const u8, len: usize) -> bool {
    if !plain() {
        return false;
    }
    // SAFETY: the source is the caller's mapped host memory of `len` bytes;
    // a destination page that cannot be written stops the copy ([`install`],
    // which `plain` has run).
    let whole = unsafe { victory_guarded_write(dst, src, len) != 0 };
    if whole {
        COUNT[0].fetch_add(1, Relaxed);
        COUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        COUNT[2].fetch_add(1, Relaxed);
    }
    whole
}

/// Once a second (`crate::host` `sweep`): every ten, what [`plain`] did.
pub(crate) fn tick() {
    static TICKS: AtomicU64 = AtomicU64::new(0);
    if !plain() || TICKS.fetch_add(1, Relaxed) % 10 != 9 {
        return;
    }
    let [writes, bytes, fell, small, lately, faults] = [0, 1, 2, 3, 4, 5].map(|i| COUNT[i].swap(0, Relaxed));
    tracing::info!(
        "title memory in place over ten seconds (VICTORY_COPY_PLAIN=1): {writes} writes back, {:.1} MB, {fell} met a \
         page and went to the kernel's write; reads left to the kernel: {small} of under {LEAST} bytes, {lately} where \
         one faulted in the last two seconds; {faults} guarded reads faulted",
        bytes as f64 / 1e6
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A write in place stops at a page that is not writable, having
    /// written the pages before it, and says so; with the switch off
    /// `write` does not try.
    #[test]
    fn a_guarded_write_stops_at_a_page_that_is_not_writable() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            VirtualProtect,
        };
        if !on() {
            return; // VICTORY_GUARDED_READS=0
        }
        const PAGE: usize = 4096;
        let src = vec![7u8; 2 * PAGE];
        // SAFETY: our own fresh pages.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            assert!(victory_guarded_write(p, src.as_ptr(), 2 * PAGE) != 0);
            assert_eq!((p.read(), p.add(2 * PAGE - 1).read()), (7, 7));
            p.write_bytes(0, 2 * PAGE);
            let mut old = 0;
            assert!(VirtualProtect(p.add(PAGE).cast(), PAGE, PAGE_READONLY, &mut old) != 0);
            assert!(victory_guarded_write(p, src.as_ptr(), 2 * PAGE) == 0);
            assert_eq!((p.read(), p.add(PAGE - 1).read(), p.add(PAGE).read()), (7, 7, 0));
            // The switch decides whether `write` tries at all.
            p.write_bytes(0, PAGE);
            let wrote = write(p, src.as_ptr(), PAGE);
            assert_eq!(wrote, plain());
            assert_eq!(p.read(), if wrote { 7 } else { 0 });
            VirtualFree(p.cast(), 0, MEM_RELEASE);
        }
    }

    /// Reads under [`LEAST`] bytes and reads where one faulted lately are
    /// the kernel's with the switch, and none is without it.
    #[test]
    fn small_and_lately_faulted_reads_are_the_kernels_with_the_switch() {
        let at = 0x1234_5678_0000usize as *const u8;
        if !plain() {
            assert!(!kernel_first(at, 1));
            note_fault(at, 4096);
            assert!(!kernel_first(at, 4096));
            return;
        }
        assert!(kernel_first(at, LEAST - 1));
        note_fault(at, 4096);
        // Within two seconds of the note, unless the table was just emptied.
        let lately = kernel_first(at, 4096) || {
            note_fault(at, 4096);
            kernel_first(at, 4096)
        };
        assert!(lately);
    }

    #[test]
    fn a_guarded_copy_stops_at_an_unreadable_page() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            VirtualProtect,
        };
        if !on() {
            return; // VICTORY_GUARDED_READS=0
        }
        const PAGE: usize = 4096;
        // SAFETY: our own fresh pages.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            for i in 0..2 * PAGE {
                p.add(i).write(i as u8);
            }
            let mut out = vec![0u8; 2 * PAGE];
            assert!(copy(p, out.as_mut_ptr(), 2 * PAGE));
            assert_eq!(out[PAGE + 7], (PAGE + 7) as u8);
            let mut old = 0;
            assert!(VirtualProtect(p.add(PAGE).cast(), PAGE, PAGE_NOACCESS, &mut old) != 0);
            let mut out = vec![0u8; 2 * PAGE];
            assert!(!copy(p, out.as_mut_ptr(), 2 * PAGE));
            assert_eq!(out[PAGE - 1], (PAGE - 1) as u8);
            assert!(copy(p.add(16), out.as_mut_ptr(), 100));
            assert_eq!(out[0], 16);
            VirtualFree(p.cast(), 0, MEM_RELEASE);
        }
    }

    /// Benchmark (ignored): 256 MiB read in pieces of 512 bytes, 4 KiB,
    /// 64 KiB and 1 MiB, through the guarded copy and through the kernel.
    /// `cargo test --release ... guarded_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn guarded_bench() {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        if !on() {
            return;
        }
        const SIZE: usize = 256 << 20;
        let src = vec![1u8; SIZE];
        let mut dst = vec![0u8; SIZE];
        dst.fill(2);
        for piece in [512usize, 4 << 10, 64 << 10, 1 << 20] {
            let started = std::time::Instant::now();
            for at in (0..SIZE).step_by(piece) {
                // SAFETY: both buffers hold SIZE bytes.
                assert!(unsafe { copy(src.as_ptr().add(at), dst.as_mut_ptr().add(at), piece) });
            }
            let guarded = started.elapsed();
            let started = std::time::Instant::now();
            for at in (0..SIZE).step_by(piece) {
                let mut done = 0;
                // SAFETY: both buffers hold SIZE bytes.
                assert!(unsafe {
                    ReadProcessMemory(GetCurrentProcess(), src.as_ptr().add(at).cast(), dst.as_mut_ptr().add(at).cast(), piece, &mut done)
                } != 0);
            }
            let kernel = started.elapsed();
            println!("256 MiB in pieces of {piece} bytes: guarded {guarded:?}, kernel {kernel:?}");
        }
    }
}
