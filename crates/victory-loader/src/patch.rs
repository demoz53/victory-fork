//! Title code changed in memory once it is mapped, never in a file: the
//! mapped pages are made writable for the write (the process's own copy of
//! them; the image file on disk is not touched) and given their protection
//! back. Each group of changes is all or none: every expected run is
//! checked before anything is written, so a change made for another build
//! of the title writes nothing.

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
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
