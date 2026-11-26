//! Title code changed in memory once it is mapped, never in a file: the
//! mapped pages are made writable for the write (the process's own copy of
//! them; the image file on disk is not touched) and given their protection
//! back. Each group of changes is all or none: every expected run is
//! checked before anything is written, so a change made for another build
//! of the title writes nothing.

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// One change: the name it is said by, where it goes (from the image's
/// base), the bytes that must be there, and ours.
#[derive(Debug, Clone, Copy)]
pub struct MemoryPatch<'a> {
    pub name: &'a str,
    pub rva: u32,
    pub expect: &'a [u8],
    pub write: &'a [u8],
}

/// Why a group of changes was not made. Nothing of it was written, except
/// after [`PatchError::Protect`] (the changes before it in the group were).
#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error("{0}: its old and new bytes differ in length, or are none")]
    Length(String),
    #[error("{0}: outside the image")]
    Outside(String),
    #[error("{0}: the expected bytes are not there (another build of the title?)")]
    Mismatch(String),
    #[error("{0}: its bytes could not be made writable")]
    Protect(String),
}

/// Makes `patches` in the image mapped at `base` (`size` bytes), all or
/// none (see the module).
///
/// # Safety
/// `base..base + size` is an image mapped in this process, and no thread
/// runs the code being changed.
pub unsafe fn patch_mapped(base: usize, size: usize, patches: &[MemoryPatch<'_>]) -> Result<(), PatchError> {
    for p in patches {
        if p.expect.len() != p.write.len() || p.write.is_empty() {
            return Err(PatchError::Length(p.name.to_owned()));
        }
        let at = p.rva as usize;
        if at.checked_add(p.expect.len()).is_none_or(|end| end > size) {
            return Err(PatchError::Outside(p.name.to_owned()));
        }
        // SAFETY: inside the mapped image (checked above), read only.
        let now = unsafe { std::slice::from_raw_parts((base + at) as *const u8, p.expect.len()) };
        if now != p.expect {
            return Err(PatchError::Mismatch(p.name.to_owned()));
        }
    }
    for p in patches {
        let at = (base + p.rva as usize) as *mut u8;
        let mut old = 0;
        // SAFETY: those bytes of the mapped image only; restored below.
        if unsafe { VirtualProtect(at.cast(), p.write.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            return Err(PatchError::Protect(p.name.to_owned()));
        }
        // SAFETY: writable now, and the bytes checked above are there.
        unsafe {
            std::ptr::copy_nonoverlapping(p.write.as_ptr(), at, p.write.len());
            VirtualProtect(at.cast(), p.write.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at.cast(), p.write.len());
        }
    }
    Ok(())
}

/// `mov r64, gs:[0x58]` (the title's load of its thread's TLS array), 9
/// bytes: the loader turns each into `call thunk; nop4` when the image got
/// a TLS index other than 0 (`crate::tls`).
const TLS_LOAD_LEN: usize = 9;

fn is_tls_load(b: &[u8]) -> bool {
    b.len() >= TLS_LOAD_LEN
        && b[0] == 0x65
        && (b[1] == 0x48 || b[1] == 0x4c)
        && b[2] == 0x8b
        && b[3] & 0xc7 == 0x04
        && b[4..TLS_LOAD_LEN] == [0x25, 0x58, 0, 0, 0]
}

/// Whether mapped code `now` is the executable's `expect`, where each TLS
/// load of `expect` may have become the loader's `call rel32; nop4`
/// (SonicRaT/XNW 2fa40fef6 `levers.rs` `code_is`, GPL-3.0).
pub fn code_matches(now: &[u8], expect: &[u8]) -> bool {
    if now.len() != expect.len() {
        return false;
    }
    let mut i = 0;
    while i < expect.len() {
        if is_tls_load(&expect[i..]) && now[i] == 0xe8 && now[i + 5..i + TLS_LOAD_LEN] == [0x0f, 0x1f, 0x40, 0x00] {
            i += TLS_LOAD_LEN;
        } else if now[i] == expect[i] {
            i += 1;
        } else {
            return false;
        }
    }
    true
}

/// Whether the code at `base + rva` (inside an image of `size` bytes) is
/// `expect`, TLS loads as the loader left them ([`code_matches`]).
///
/// # Safety
/// `base..base + size` is an image mapped in this process.
pub unsafe fn code_at(base: usize, size: usize, rva: u32, expect: &[u8]) -> bool {
    let at = rva as usize;
    if at.checked_add(expect.len()).is_none_or(|end| end > size) {
        return false;
    }
    // SAFETY: inside the mapped image (checked above), read only.
    code_matches(unsafe { std::slice::from_raw_parts((base + at) as *const u8, expect.len()) }, expect)
}

/// `jmp qword [rip+0]` then the target: 14 bytes, reaches anywhere.
fn far_jump(to: usize) -> [u8; 14] {
    let mut j = [0xff, 0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    j[6..].copy_from_slice(&(to as u64).to_le_bytes());
    j
}

/// Puts a far jump to `hook` at the function at `base + rva`, whose first
/// bytes must be `entry` (whole instructions, at least 14 bytes, none of
/// them position dependent: they run again elsewhere), and returns the
/// trampoline that runs them and goes on after them: calling it is calling
/// the original function. The entry's bytes past the jump become `int3`
/// (never run). The trampoline is a block of ours anywhere in memory
/// (absolute jumps both ways), made read-and-execute.
///
/// From SonicRaT/XNW 2fa40fef6 `apps/xnw/src/detours.rs` (GPL-3.0).
///
/// # Safety
/// `base..base + size` is an image mapped in this process, no thread runs
/// the code at the entry, and `hook` is a function of the original's type.
pub unsafe fn detour(base: usize, size: usize, rva: u32, entry: &[u8], hook: usize, name: &str) -> Result<usize, PatchError> {
    if entry.len() < 14 || entry.len() > 48 {
        return Err(PatchError::Length(name.to_owned()));
    }
    let at = rva as usize;
    if at.checked_add(entry.len()).is_none_or(|end| end > size) {
        return Err(PatchError::Outside(name.to_owned()));
    }
    let code = (base + at) as *mut u8;
    // SAFETY: inside the mapped image, read only.
    if unsafe { std::slice::from_raw_parts(code, entry.len()) } != entry {
        return Err(PatchError::Mismatch(name.to_owned()));
    }
    // SAFETY: a fresh block of our own.
    let tramp = unsafe { VirtualAlloc(std::ptr::null(), 64, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
    if tramp.is_null() {
        return Err(PatchError::Protect(name.to_owned()));
    }
    let back = far_jump(code as usize + entry.len());
    let mut old = 0;
    // SAFETY: 64 bytes of ours hold the moved bytes (at most 48) and a jump.
    unsafe {
        std::ptr::copy_nonoverlapping(entry.as_ptr(), tramp, entry.len());
        std::ptr::copy_nonoverlapping(back.as_ptr(), tramp.add(entry.len()), back.len());
        VirtualProtect(tramp.cast(), 64, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), tramp.cast(), 64);
    }
    let mut new = vec![0xcc; entry.len()];
    new[..14].copy_from_slice(&far_jump(hook));
    let p = MemoryPatch { name, rva, expect: entry, write: &new };
    // SAFETY: forwarded from this function's contract.
    unsafe { patch_mapped(base, size, &[p]) }?;
    Ok(tramp as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualQuery,
    };

    /// Two pages of "code" (a pattern), read-and-execute as an image's are.
    struct Code(usize);

    impl Code {
        fn new() -> Code {
            // SAFETY: a fresh allocation of our own.
            let at = unsafe { VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
            assert_ne!(at, 0);
            for i in 0..0x2000 {
                // SAFETY: inside the allocation, writable.
                unsafe { ((at + i) as *mut u8).write((i % 251) as u8) };
            }
            let mut old = 0;
            // SAFETY: our allocation.
            assert_ne!(unsafe { VirtualProtect(at as _, 0x2000, PAGE_EXECUTE_READ, &mut old) }, 0);
            Code(at)
        }

        fn bytes(&self, rva: usize, n: usize) -> Vec<u8> {
            // SAFETY: inside the allocation, readable.
            unsafe { std::slice::from_raw_parts((self.0 + rva) as *const u8, n).to_vec() }
        }

        fn protection(&self, rva: usize) -> u32 {
            // SAFETY: a plain-data structure, all zeros is valid.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a plain query of our own pages.
            let got = unsafe {
                VirtualQuery((self.0 + rva) as _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>())
            };
            assert_ne!(got, 0);
            info.Protect
        }
    }

    impl Drop for Code {
        fn drop(&mut self) {
            // SAFETY: our allocation, no longer used.
            unsafe { VirtualFree(self.0 as _, 0, MEM_RELEASE) };
        }
    }

    static TRAMP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    extern "system" fn plus_one(a: usize, b: usize) -> usize {
        // SAFETY: the trampoline of the test function, of this type.
        let original: extern "system" fn(usize, usize) -> usize =
            unsafe { std::mem::transmute(TRAMP.load(std::sync::atomic::Ordering::Relaxed)) };
        original(a, b) + 1
    }

    /// A function made in memory with a 15-byte prologue (`push r15; push
    /// r14; push rsi; push rdi; push rbp; push rbx; sub rsp, 28h; mov rsi,
    /// rdx`) returning `a + b`: detoured, it runs the hook, which runs the
    /// original through the trampoline.
    #[test]
    fn a_detour_runs_the_hook_then_the_original() {
        let prologue = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x28, 0x48, 0x89, 0xd6];
        let body = [
            0x48, 0x89, 0xc8, // mov rax, rcx
            0x48, 0x01, 0xf0, // add rax, rsi
            0x48, 0x83, 0xc4, 0x28, // add rsp, 28h
            0x5b, 0x5d, 0x5f, 0x5e, // pop rbx; pop rbp; pop rdi; pop rsi
            0x41, 0x5e, 0x41, 0x5f, 0xc3, // pop r14; pop r15; ret
        ];
        // SAFETY: a fresh block of our own.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) } as usize;
        assert_ne!(code, 0);
        // SAFETY: inside the block.
        unsafe {
            std::ptr::copy_nonoverlapping(prologue.as_ptr(), (code + 16) as *mut u8, prologue.len());
            std::ptr::copy_nonoverlapping(body.as_ptr(), (code + 16 + prologue.len()) as *mut u8, body.len());
        }
        // SAFETY: the function just written, of this type.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(code + 16) };
        assert_eq!(f(2, 3), 5);
        // Other bytes than expected: nothing changes.
        let mut wrong = prologue;
        wrong[3] ^= 1;
        // SAFETY: our block, nothing runs it now.
        assert!(matches!(unsafe { detour(code, 4096, 16, &wrong, plus_one as *const () as usize, "w") }, Err(PatchError::Mismatch(_))));
        assert_eq!(f(2, 3), 5);
        // SAFETY: as above.
        let tramp = unsafe { detour(code, 4096, 16, &prologue, plus_one as *const () as usize, "f") }.unwrap();
        TRAMP.store(tramp, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(f(2, 3), 6, "the hook ran and called the original");
        // SAFETY: as above: the bytes are ours now, not the expected ones.
        assert!(unsafe { detour(code, 4096, 16, &prologue, plus_one as *const () as usize, "f") }.is_err());
        // SAFETY: our block.
        unsafe { windows_sys::Win32::System::Memory::VirtualFree(code as _, 0, MEM_RELEASE) };
    }

    #[test]
    fn code_matches_takes_the_loaders_tls_rewrite() {
        // mov rax, gs:[0x58]; mov rax, [rax]
        let exe = [0x65, 0x48, 0x8b, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8b, 0x00];
        let rewritten = [0xe8, 1, 2, 3, 4, 0x0f, 0x1f, 0x40, 0x00, 0x48, 0x8b, 0x00];
        assert!(code_matches(&exe, &exe));
        assert!(code_matches(&rewritten, &exe));
        let mut other = rewritten;
        other[10] = 0x8a;
        assert!(!code_matches(&other, &exe));
        assert!(!code_matches(&rewritten[..11], &exe));
    }

    #[test]
    fn a_group_is_written_whole_or_not_at_all() {
        let code = Code::new();
        let (a, b) = (0x10usize, 0x1ff0usize);
        let (old_a, old_b) = (code.bytes(a, 6), code.bytes(b, 4));
        let new_a = [0xb8, 0x06, 0, 0, 0, 0xc3];
        let new_b: Vec<u8> = old_b.iter().map(|x| !x).collect();
        let good_a = MemoryPatch { name: "a", rva: a as u32, expect: &old_a, write: &new_a };
        let wrong_b = MemoryPatch { name: "b", rva: b as u32, expect: &new_b, write: &old_b };

        // Another build: the second's bytes are not there; the first is not
        // written either.
        // SAFETY: our allocation, nothing runs it.
        let e = unsafe { patch_mapped(code.0, 0x2000, &[good_a, wrong_b]) };
        assert!(matches!(e, Err(PatchError::Mismatch(ref n)) if n == "b"));
        assert_eq!(code.bytes(a, 6), old_a);
        // Outside the image, or lengths that differ: nothing either.
        let outside = MemoryPatch { name: "c", rva: 0x1ffe, expect: &old_b, write: &new_b };
        // SAFETY: as above.
        assert!(matches!(unsafe { patch_mapped(code.0, 0x2000, &[good_a, outside]) }, Err(PatchError::Outside(_))));
        let short = MemoryPatch { name: "d", rva: a as u32, expect: &old_a, write: &new_a[..5] };
        // SAFETY: as above.
        assert!(matches!(unsafe { patch_mapped(code.0, 0x2000, &[short]) }, Err(PatchError::Length(_))));
        assert_eq!(code.bytes(a, 6), old_a);

        // The bytes are there: both written, the pages read-and-execute again.
        let good_b = MemoryPatch { name: "b", rva: b as u32, expect: &old_b, write: &new_b };
        // SAFETY: as above.
        unsafe { patch_mapped(code.0, 0x2000, &[good_a, good_b]) }.unwrap();
        assert_eq!(code.bytes(a, 6), new_a);
        assert_eq!(code.bytes(b, 4), new_b);
        assert_eq!(code.protection(a), PAGE_EXECUTE_READ);
        assert_eq!(code.protection(b), PAGE_EXECUTE_READ);
        // Done twice: the bytes are ours now, not the expected ones.
        // SAFETY: as above.
        assert!(matches!(unsafe { patch_mapped(code.0, 0x2000, &[good_a]) }, Err(PatchError::Mismatch(_))));
    }
}
