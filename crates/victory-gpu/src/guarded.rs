//! Reads of title memory that may meet an unmapped page, as a plain copy
//! (`rep movsb`) that an exception handler stops at the first page that
//! faults. `ReadProcessMemory`, which the reads went through, goes into
//! the kernel and copies through it: about a fifth of the submitting
//! thread's time in the world (the refreshes of host copies).
//! `VICTORY_GUARDED_READS=0` reads through the kernel, as before.

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
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

unsafe extern "C" {
    fn victory_guarded_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_guarded_copy_at: u8;
    static victory_guarded_copy_fail: u8;
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

#[cfg(test)]
mod tests {
    use super::*;

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
