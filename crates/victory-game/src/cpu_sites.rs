//! Where a build's code uses instructions of the console's CPU (a Zen 2)
//! that a PC's CPU may lack: `insertq` (SSE4a), `clzero`, `clwb`,
//! `monitorx` and `mwaitx`. On a CPU without one, every execution is an
//! invalid-opcode exception that the loader emulates in its handler; the
//! loader can instead rewrite the sites listed here once, at load
//! (`victory_loader::rewrite`, `VICTORY_CPU_REWRITE=1`).
//!
//! The list is SonicRaT/XNW's (`crates/xnw-loader/src/rewrite_table.rs`,
//! commit 2fa40fef6, GPL-3.0): 41 sites it found by an offline scan of the
//! same executable (SHA-256 e95020b2fa4a721d...d5a08993, which is our
//! BLAKE3 b99f876b...04239e93, PE time stamp 0x6af783d4). Each row was
//! checked again against that file before it was entered here:
//!
//! - the bytes at the site (`expect`: the instruction and, for a `clzero`,
//!   the whole instructions after it that a 5-byte jump also covers) are
//!   the ones given, and a disassembly from the start of the function that
//!   holds them has the instruction on its boundary;
//! - `targets`: bit n is set when a branch of that function lands n bytes
//!   past the site (1 to 15). None lands inside a row's bytes. XNW's table
//!   agrees for every offset inside them;
//! - nothing else in the file refers to a byte inside a row's bytes after
//!   its first. Every byte offset of the code section was tried as a
//!   `rel32` or `[rip + disp32]` operand (with 0, 1, 2 or 4 bytes of
//!   immediate after it) and as a 32-bit relative address, and the bytes
//!   near each row as `rel8` branches: the 88 places that would refer to
//!   one were each read in the disassembly of their function, and none is
//!   an operand that does. Outside the code section no 32-bit relative
//!   address (jump tables, unwind and exception data), no 64-bit address
//!   and no relocation names one; and the functions that hold the rows
//!   have no indirect jump.
//!
//! What such a scan cannot rule out is an address computed at run time
//! from something that is none of those.

/// The instruction at a [`CpuSite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuKind {
    /// `clwb m8`: a write-back hint (Intel CPUs since Ice Lake have it).
    Clwb,
    /// `monitorx`: names the line a following `mwaitx` waits on.
    Monitorx,
    /// `mwaitx`: waits for a write to that line, or a timer.
    Mwaitx,
    /// `insertq xmm, xmm, len, index` (SSE4a).
    Insertq,
    /// `clzero`: zeroes the 64-byte line `rax` points into.
    Clzero,
}

/// One use of such an instruction in a build's code.
#[derive(Debug, Clone, Copy)]
pub struct CpuSite {
    pub rva: u32,
    pub kind: CpuKind,
    /// The instruction's length in bytes.
    pub len: u8,
    /// The bytes at `rva`: the instruction, then any whole instructions
    /// that a rewrite moves with it (a short signature, checked before
    /// anything is written).
    pub expect: &'static [u8],
    /// Bit n: a branch lands n bytes past `rva`.
    pub targets: u16,
}

/// The builds whose sites are known: (BLAKE3 of the executable, sites).
const BUILDS: &[(&str, &[CpuSite])] =
    &[("b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93", SERIES_MASTER)];

/// The sites of the build whose executable has that BLAKE3 hash (hex);
/// none for a build nobody scanned.
pub fn cpu_sites(blake3_hex: &str) -> &'static [CpuSite] {
    BUILDS.iter().find(|(hash, _)| hash.eq_ignore_ascii_case(blake3_hex)).map_or(&[], |(_, sites)| sites)
}

#[rustfmt::skip]
const SERIES_MASTER: &[CpuSite] = &[
    CpuSite { rva: 0x27c8f2, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x44, 0x0f, 0x78, 0xcf, 0x08, 0x00], targets: 0x0 },
    CpuSite { rva: 0x27e9f1, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x44, 0x0f, 0x78, 0xc7, 0x18, 0x00], targets: 0x0 },
    CpuSite { rva: 0x2e7ff1, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x7f1ac9, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x45, 0x0f, 0x78, 0xdc, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x7f1b29, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xe3, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x7f1b38, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xeb, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x7f1b44, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xf3, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0xf75fb3, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x45, 0x0f, 0x78, 0xf5, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0xf762b7, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x45, 0x0f, 0x78, 0xf5, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0xf765b5, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x45, 0x0f, 0x78, 0xf5, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0xf768b0, kind: CpuKind::Insertq, len: 7, expect: &[0xf2, 0x45, 0x0f, 0x78, 0xf5, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x269dcb2, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xe5, 0x10, 0x30], targets: 0x0 },
    CpuSite { rva: 0x32bc150, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc2, 0x10, 0x10], targets: 0x0 },
    CpuSite { rva: 0x43d440e, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc0, 0x08, 0x08], targets: 0x0 },
    CpuSite { rva: 0x43d491b, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc0, 0x08, 0x08], targets: 0x0 },
    CpuSite { rva: 0x455bac7, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc9, 0x08, 0x10], targets: 0x0 },
    CpuSite { rva: 0x4e4b680, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x31], targets: 0x0 },
    CpuSite { rva: 0x4e8aa70, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x0 },
    CpuSite { rva: 0x4e8aa90, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x0 },
    CpuSite { rva: 0x4e8aab0, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x0 },
    CpuSite { rva: 0x4e8aad0, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x0 },
    CpuSite { rva: 0x4e8aaf0, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x1000 },
    CpuSite { rva: 0x4e8ab2d, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40], targets: 0x1000 },
    CpuSite { rva: 0x4e8ab53, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xda], targets: 0x100 },
    CpuSite { rva: 0x4e8ab72, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xd1], targets: 0x100 },
    CpuSite { rva: 0x4e8ab91, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xc8], targets: 0x100 },
    CpuSite { rva: 0x4e8abb4, kind: CpuKind::Clzero, len: 3, expect: &[0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2], targets: 0x0 },
    CpuSite { rva: 0x4ecc84a, kind: CpuKind::Clwb, len: 5, expect: &[0x66, 0x41, 0x0f, 0xae, 0x31], targets: 0x4000 },
    CpuSite { rva: 0x4fb0bd7, kind: CpuKind::Monitorx, len: 3, expect: &[0x0f, 0x01, 0xfa], targets: 0x0 },
    CpuSite { rva: 0x4fb0c00, kind: CpuKind::Mwaitx, len: 3, expect: &[0x0f, 0x01, 0xfb], targets: 0x0 },
    CpuSite { rva: 0x4fb0d21, kind: CpuKind::Monitorx, len: 3, expect: &[0x0f, 0x01, 0xfa], targets: 0x0 },
    CpuSite { rva: 0x4fb0d52, kind: CpuKind::Mwaitx, len: 3, expect: &[0x0f, 0x01, 0xfb], targets: 0x0 },
    CpuSite { rva: 0x5084782, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc0, 0x08, 0x18], targets: 0x0 },
    CpuSite { rva: 0x50f75f5, kind: CpuKind::Clwb, len: 5, expect: &[0x66, 0x41, 0x0f, 0xae, 0x30], targets: 0x0 },
    CpuSite { rva: 0x532f1b0, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x30], targets: 0x2000 },
    CpuSite { rva: 0x532f1f0, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x30], targets: 0x2000 },
    CpuSite { rva: 0x532f230, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x30], targets: 0x2000 },
    CpuSite { rva: 0x532f280, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x30], targets: 0x8000 },
    CpuSite { rva: 0x532f2ba, kind: CpuKind::Clwb, len: 4, expect: &[0x66, 0x0f, 0xae, 0x30], targets: 0x0 },
    CpuSite { rva: 0x546d4a2, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc0, 0x08, 0x18], targets: 0x0 },
    CpuSite { rva: 0x55b1811, kind: CpuKind::Insertq, len: 6, expect: &[0xf2, 0x0f, 0x78, 0xc9, 0x08, 0x18], targets: 0x0 },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_known_build_has_its_sites() {
        let build = &crate::KNOWN_BUILDS[0];
        let sites = cpu_sites(build.blake3);
        assert_eq!(cpu_sites(&build.blake3.to_uppercase()).len(), sites.len());
        assert!(cpu_sites("00").is_empty());
        let count = |kind| sites.iter().filter(|s| s.kind == kind).count();
        assert_eq!(sites.len(), 41);
        assert_eq!(count(CpuKind::Insertq), 19);
        assert_eq!(count(CpuKind::Clzero), 10);
        assert_eq!(count(CpuKind::Clwb), 8);
        assert_eq!(count(CpuKind::Monitorx), 2);
        assert_eq!(count(CpuKind::Mwaitx), 2);
    }

    #[test]
    fn rows_are_well_formed() {
        let sites = SERIES_MASTER;
        for pair in sites.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(a.rva as usize + a.expect.len() <= b.rva as usize, "{:#x} overlaps {:#x}", a.rva, b.rva);
        }
        for s in sites {
            let (len, window) = (usize::from(s.len), s.expect.len());
            assert!(len >= 3 && len <= window && window <= 15, "{:#x}", s.rva);
            // No branch lands inside the bytes a rewrite covers.
            assert_eq!(u32::from(s.targets) & ((1 << window) - 1), 0, "{:#x}", s.rva);
            // Only a clzero (3 bytes, too short for a jump) takes more with it.
            assert_eq!(window > len, s.kind == CpuKind::Clzero, "{:#x}", s.rva);
            let first = match s.kind {
                CpuKind::Clwb => &[0x66][..],
                CpuKind::Monitorx => &[0x0f, 0x01, 0xfa],
                CpuKind::Mwaitx => &[0x0f, 0x01, 0xfb],
                CpuKind::Insertq => &[0xf2],
                CpuKind::Clzero => &[0x0f, 0x01, 0xfc],
            };
            assert!(s.expect.starts_with(first), "{:#x}", s.rva);
        }
    }
}
