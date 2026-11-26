//! The console CPU's instructions that this PC's CPU lacks, rewritten in
//! the title's mapped image once, at load, so that they no longer cost an
//! exception each. [`crate::crash`] emulates them in the fault handler:
//! that stays, and takes every site this module leaves alone.
//!
//! The method is SonicRaT/XNW's (`crates/xnw-loader/src/rewrite.rs`, commit
//! 2fa40fef6, GPL-3.0), ported: the planner, the in-place forms, the two
//! stubs and the search for an arena are its; the table rows carry their
//! expected bytes here, the arena is made executable before any site
//! points into it, and each site is written through [`crate::patch`].
//!
//! Per site, and only where this CPU lacks the instruction (its CPUID bit:
//! a site the CPU runs itself is left as it is):
//!
//! - in place, same length: `clwb m8` becomes `nop m8` (the PC's memory is
//!   coherent, a write-back hint has nothing to do); `monitorx` becomes
//!   `nop eax` (`0f 1f c0`); `mwaitx` becomes `pause; nop` (`f3 90 90`).
//!   The title's only user of the pair is its fence wait, which reads the
//!   fence again after the `mwaitx` and reads no register or flag it sets;
//! - a stub: `insertq` and `clzero` become a 5-byte jump to a stub in an
//!   arena within 2 GiB of the image. The stub does the instruction with
//!   SSE2 on the site's own registers, keeps what it borrows (one xmm
//!   register, and `rcx` for `clzero`) in stack it takes below `rsp`,
//!   changes no flag (no arithmetic instruction: the masks are `pand`s with
//!   constants beside the stub), and jumps back. A `clzero` is 3 bytes, so
//!   the whole register-only instructions after it that the jump covers
//!   (`add rax, 40h`, or a `cmp`) run at the end of its stub.
//!
//! A site is rewritten only if its bytes are the row's, they decode as the
//! row's instruction, what would move is register-only (so it runs the
//! same anywhere: no `rip`, no memory, no branch) and no branch lands
//! inside the bytes covered. Anything else is left to the exception path,
//! and said. Nothing here runs once the title has started: the caller
//! rewrites before any of its code does.

use std::sync::OnceLock;

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::patch::{MemoryPatch, PatchError, patch_mapped};

/// An instruction of the console's CPU that a PC's may lack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Insertq,
    Clzero,
    Monitorx,
    Mwaitx,
    Clwb,
}

impl Kind {
    pub const ALL: [Kind; 5] =
        [Kind::Insertq, Kind::Clzero, Kind::Monitorx, Kind::Mwaitx, Kind::Clwb];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Insertq => "insertq",
            Kind::Clzero => "clzero",
            Kind::Monitorx => "monitorx",
            Kind::Mwaitx => "mwaitx",
            Kind::Clwb => "clwb",
        }
    }

    /// Whether its rewrite is a jump to a stub (else new bytes in place).
    fn stubbed(self) -> bool {
        matches!(self, Kind::Insertq | Kind::Clzero)
    }
}

/// Whether this CPU runs `kind` itself (CPUID): SSE4a is 8000_0001h ECX
/// bit 6, MONITORX/MWAITX bit 29 of the same, CLZERO 8000_0008h EBX bit 0,
/// CLWB leaf 7 EBX bit 24.
pub fn native(kind: Kind) -> bool {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    let extended = __cpuid(0x8000_0000).eax;
    let features = if extended >= 0x8000_0001 { __cpuid(0x8000_0001).ecx } else { 0 };
    match kind {
        Kind::Insertq => features & 1 << 6 != 0,
        Kind::Monitorx | Kind::Mwaitx => features & 1 << 29 != 0,
        Kind::Clzero => extended >= 0x8000_0008 && __cpuid(0x8000_0008).ebx & 1 != 0,
        Kind::Clwb => __cpuid(0).eax >= 7 && __cpuid_count(7, 0).ebx & 1 << 24 != 0,
    }
}

/// One use of such an instruction in the title's code.
#[derive(Debug, Clone, Copy)]
pub struct Site<'a> {
    pub rva: u32,
    pub kind: Kind,
    /// The instruction's length.
    pub len: u8,
    /// The bytes the rewrite covers, as they must be found: the
    /// instruction, then any whole instructions moved into its stub.
    pub expect: &'a [u8],
    /// Bit n: a branch lands n bytes past `rva`.
    pub targets: u16,
}

/// Why a site was not rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// The CPU runs the instruction itself: no rewrite, and no exception.
    Native,
    /// Its kind is held back by a switch ([`Options::hold`]).
    Held,
    Malformed,
    Outside,
    Bytes,
    Decode,
    Moved,
    Target,
    Short,
    NoArena,
    Reach,
    Protect,
}

impl Why {
    pub fn text(self) -> &'static str {
        match self {
            Why::Native => "the CPU has the instruction",
            Why::Held => "held back by a switch",
            Why::Malformed => "a malformed row",
            Why::Outside => "outside the image",
            Why::Bytes => "the expected bytes are not there",
            Why::Decode => "the bytes are not the instruction the row names",
            Why::Moved => "an instruction that would move with it is not register-only",
            Why::Target => "a branch lands inside its bytes",
            Why::Short => "fewer than 5 bytes for the jump",
            Why::NoArena => "no free memory within 2 GiB of the image",
            Why::Reach => "its stub is out of a jump's reach",
            Why::Protect => "memory protection could not be changed",
        }
    }
}

/// What to do at one site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// New bytes over the instruction.
    InPlace(Vec<u8>),
    /// A stub; the bytes covered become `jmp stub` and `nop`s.
    Stub,
}

/// The plan for `site` given the bytes now at it (`code` starts at the
/// site), or why it is refused.
pub fn plan(site: &Site<'_>, code: &[u8]) -> Result<Plan, Why> {
    let (len, window) = (usize::from(site.len), site.expect.len());
    if len == 0 || len > window || window > 15 {
        return Err(Why::Malformed);
    }
    let code = code.get(..window).ok_or(Why::Outside)?;
    if code != site.expect {
        return Err(Why::Bytes);
    }
    if !decodes_as(site.kind, &code[..len]) {
        return Err(Why::Decode);
    }
    if !moved_ok(&code[len..]) {
        return Err(Why::Moved);
    }
    // A branch to the site's first byte finds the rewritten instruction;
    // one to any later byte covered would land inside the jump.
    if u32::from(site.targets) & ((1u32 << window) - 1) & !1 != 0 {
        return Err(Why::Target);
    }
    if site.kind.stubbed() {
        return if window >= 5 { Ok(Plan::Stub) } else { Err(Why::Short) };
    }
    if window != len {
        // An in-place rewrite moves nothing.
        return Err(Why::Malformed);
    }
    in_place(site.kind, code).map(Plan::InPlace).ok_or(Why::Decode)
}

/// Whether `code` is exactly one instruction of `kind`.
pub fn decodes_as(kind: Kind, code: &[u8]) -> bool {
    match kind {
        Kind::Clwb => crate::crash::clwb_len(code) == Some(code.len()),
        Kind::Monitorx => code == [0x0f, 0x01, 0xfa],
        Kind::Mwaitx => code == [0x0f, 0x01, 0xfb],
        Kind::Clzero => code == [0x0f, 0x01, 0xfc],
        Kind::Insertq => insertq_fields(code).is_some(),
    }
}

/// Whether `code` is whole register-only instructions (`[rex] op modrm`
/// with mod 11, then an imm8 for `83`: the two-operand arithmetic and
/// logic forms, `test` and `mov`), which touch no memory and no `rip` and
/// so run the same from a stub.
fn moved_ok(mut code: &[u8]) -> bool {
    while !code.is_empty() {
        let at = usize::from(code[0] & 0xf0 == 0x40);
        let Some(&[op, modrm]) = code.get(at..at + 2) else {
            return false;
        };
        let imm = match op {
            0x83 => 1,
            0x01 | 0x03 | 0x09 | 0x0b | 0x21 | 0x23 | 0x29 | 0x2b | 0x31 | 0x33 | 0x39 | 0x3b
            | 0x85 | 0x89 | 0x8b => 0,
            _ => return false,
        };
        let n = at + 2 + imm;
        if modrm >> 6 != 3 || code.len() < n {
            return false;
        }
        code = &code[n..];
    }
    true
}

/// The same-length replacement for a hint (`code` is the one instruction).
fn in_place(kind: Kind, code: &[u8]) -> Option<Vec<u8>> {
    match kind {
        // nop eax
        Kind::Monitorx => Some(vec![0x0f, 0x1f, 0xc0]),
        // pause; nop
        Kind::Mwaitx => Some(vec![0xf3, 0x90, 0x90]),
        Kind::Clwb => {
            // `66 [rex] 0f ae /6` to `66 [rex] 0f 1f /0`: a nop with the
            // same memory operand, which it does not touch.
            crate::crash::clwb_len(code)?;
            let at = if code[1] & 0xf0 == 0x40 { 2 } else { 1 };
            let mut new = code.to_vec();
            new[at + 1] = 0x1f;
            new[at + 2] &= !0x38;
            Some(new)
        }
        Kind::Insertq | Kind::Clzero => None,
    }
}

/// `insertq xmm(dst), xmm(src), len, index` (`f2 [rex] 0f 78 /r ib ib`,
/// register operands): (dst, src, len, index).
fn insertq_fields(code: &[u8]) -> Option<(u8, u8, u8, u8)> {
    let (rex, at) = if code.get(1)? & 0xf0 == 0x40 { (code[1], 2) } else { (0, 1) };
    if code[0] != 0xf2 || code.get(at..at + 2)? != [0x0f, 0x78] {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    if modrm >> 6 != 3 || code.len() != at + 5 {
        return None;
    }
    Some((
        (modrm >> 3) & 7 | (rex & 4) << 1,
        modrm & 7 | (rex & 1) << 3,
        code[at + 3] & 63,
        code[at + 4] & 63,
    ))
}

/// Code being put together for a known address.
struct Asm {
    at: usize,
    code: Vec<u8>,
}

impl Asm {
    fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// The start of a legacy-SSE instruction: `prefix [rex] 0f op`.
    fn sse(&mut self, prefix: u8, op: u8, reg: u8, rm: u8) {
        self.code.push(prefix);
        let rex = 0x40 | (reg >> 3) << 2 | rm >> 3;
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.bytes(&[0x0f, op]);
    }

    /// `prefix [rex] 0f op /r` between registers.
    fn sse_rr(&mut self, prefix: u8, op: u8, reg: u8, rm: u8) {
        self.sse(prefix, op, reg, rm);
        self.code.push(0xc0 | (reg & 7) << 3 | (rm & 7));
    }

    /// The same with `[rsp + disp]` as the other operand.
    fn sse_rsp(&mut self, prefix: u8, op: u8, reg: u8, disp: u8) {
        self.sse(prefix, op, reg, 0);
        self.bytes(&[0x44 | (reg & 7) << 3, 0x24, disp]);
    }

    /// The same with `[rip + disp32]`, naming `target`.
    fn sse_rip(&mut self, prefix: u8, op: u8, reg: u8, target: usize) -> Result<(), Why> {
        self.sse(prefix, op, reg, 0);
        self.code.push(0x05 | (reg & 7) << 3);
        self.rel32(target)
    }

    /// `jmp target`.
    fn jmp(&mut self, target: usize) -> Result<(), Why> {
        self.code.push(0xe9);
        self.rel32(target)
    }

    /// A 32-bit displacement to `target` from the end of itself.
    fn rel32(&mut self, target: usize) -> Result<(), Why> {
        let end = self.at + self.code.len() + 4;
        self.bytes(&rel32(end, target)?);
        Ok(())
    }
}

/// The 32-bit displacement that reaches `target` from `end` (the address
/// after the displacement), if one does.
fn rel32(end: usize, target: usize) -> Result<[u8; 4], Why> {
    let distance = (target as i64).wrapping_sub(end as i64);
    i32::try_from(distance).map(i32::to_le_bytes).map_err(|_| Why::Reach)
}

/// A stub, as made for its place.
struct Built {
    /// The two 16-byte constants its `pand`s use, at the slot's start.
    constants: [[u64; 2]; 2],
    code: Vec<u8>,
    /// The bytes of stack it takes below the title's `rsp` while it works.
    frame: usize,
    /// Offset in `code` just past the instruction that gives them back.
    restored: usize,
}

/// Bytes per stub in the arena: 32 of constants, then the code. A multiple
/// of 16, as the constants are `pand` operands.
const STUB_ROOM: usize = 160;
const STUB_CODE: usize = 32;

/// The stub for `site` (planned as one: its bytes are the row's) in the
/// slot at `slot`, ending in a jump back to `back`.
fn stub(site: &Site<'_>, slot: usize, back: usize) -> Result<Built, Why> {
    let len = usize::from(site.len);
    let mut a = Asm { at: slot + STUB_CODE, code: Vec::with_capacity(STUB_ROOM - STUB_CODE) };
    let mut constants = [[0u64; 2]; 2];
    let frame: u8 = match site.kind {
        Kind::Insertq => {
            let (dst, src, n, index) = insertq_fields(&site.expect[..len]).ok_or(Why::Decode)?;
            let mask = if n == 0 { u64::MAX } else { (1u64 << n) - 1 };
            constants[0] = [mask, 0];
            constants[1] = [!(mask << index), u64::MAX];
            // The register borrowed: any but the two the instruction names.
            let t = (0..16u8).find(|&r| r != dst && r != src).ok_or(Why::Malformed)?;
            a.bytes(&[0x48, 0x8d, 0x64, 0x24, 0xf0]); // lea rsp, [rsp - 16]
            a.sse_rsp(0xf3, 0x7f, t, 0); // movdqu [rsp], xmm(t)
            a.sse_rr(0xf3, 0x7e, t, src); // movq xmm(t), xmm(src)
            a.sse_rip(0x66, 0xdb, t, slot)?; // pand xmm(t), [the field's mask]
            if index != 0 {
                a.sse_rr(0x66, 0x73, 6, t); // psllq xmm(t), index
                a.bytes(&[index]);
            }
            a.sse_rip(0x66, 0xdb, dst, slot + 16)?; // pand xmm(dst), [all but the field]
            a.sse_rr(0x66, 0xeb, dst, t); // por xmm(dst), xmm(t)
            a.sse_rsp(0xf3, 0x6f, t, 0); // movdqu xmm(t), [rsp]
            16
        }
        Kind::Clzero => {
            constants[0] = [!63, 0];
            a.bytes(&[0x48, 0x8d, 0x64, 0x24, 0xe8]); // lea rsp, [rsp - 24]
            a.bytes(&[0x48, 0x89, 0x0c, 0x24]); // mov [rsp], rcx
            a.sse_rsp(0xf3, 0x7f, 0, 8); // movdqu [rsp + 8], xmm0
            a.bytes(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
            a.sse_rip(0x66, 0xdb, 0, slot)?; // pand xmm0, [not 63]
            a.bytes(&[0x66, 0x48, 0x0f, 0x7e, 0xc1]); // movq rcx, xmm0
            a.bytes(&[0x66, 0x0f, 0xef, 0xc0]); // pxor xmm0, xmm0
            a.bytes(&[0x66, 0x0f, 0x7f, 0x01]); // movdqa [rcx], xmm0
            for at in [16u8, 32, 48] {
                a.bytes(&[0x66, 0x0f, 0x7f, 0x41, at]); // movdqa [rcx + at], xmm0
            }
            a.sse_rsp(0xf3, 0x6f, 0, 8); // movdqu xmm0, [rsp + 8]
            a.bytes(&[0x48, 0x8b, 0x0c, 0x24]); // mov rcx, [rsp]
            24
        }
        Kind::Monitorx | Kind::Mwaitx | Kind::Clwb => return Err(Why::Malformed),
    };
    a.bytes(&[0x48, 0x8d, 0x64, 0x24, frame]); // lea rsp, [rsp + frame]
    let restored = a.code.len();
    // The instructions moved (register-only, so the same here), then back.
    a.bytes(&site.expect[len..]);
    a.jmp(back)?;
    if STUB_CODE + a.code.len() > STUB_ROOM {
        return Err(Why::Malformed);
    }
    Ok(Built { constants, code: a.code, frame: usize::from(frame), restored })
}

/// How far from any byte of the image any byte of the arena may be: 2 GiB
/// (a jump's reach) less a margin.
const REACH: usize = 0x7ff0_0000;
/// The grain addresses are reserved at.
const GRAIN: usize = 0x1_0000;

/// The lowest and the highest start of an arena of `size` bytes from which
/// each of its bytes is within [`REACH`] of each byte of `base..end`.
fn span(base: usize, end: usize, size: usize) -> Option<(usize, usize)> {
    let lowest = end.saturating_sub(REACH).checked_next_multiple_of(GRAIN)?;
    let highest = base.checked_add(REACH)?.checked_sub(size)? & !(GRAIN - 1);
    (lowest <= highest).then_some((lowest, highest))
}

/// Memory of ours for the stubs: read-write while they are written.
struct Arena {
    at: usize,
    size: usize,
}

impl Arena {
    /// Free address space for `size` bytes within reach of the image at
    /// `base..end`, committed read-write: above the image first, then below.
    fn near(base: usize, end: usize, size: usize) -> Option<Arena> {
        let (lowest, highest) = span(base, end, size)?;
        let above = end.checked_next_multiple_of(GRAIN)?;
        for (from, to) in [(above, highest), (lowest, base.saturating_sub(size).min(highest))] {
            let mut at = from;
            while at <= to {
                // SAFETY: plain data filled in by the query.
                let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: a query about our own address space.
                if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                    break;
                }
                let region_end = (m.BaseAddress as usize).saturating_add(m.RegionSize);
                if m.State == MEM_FREE && region_end >= at + size {
                    // SAFETY: reserves and commits free address space.
                    let got = unsafe {
                        VirtualAlloc(at as *const _, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
                    };
                    if !got.is_null() {
                        return Some(Arena { at: got as usize, size });
                    }
                }
                at = region_end.max(at + GRAIN).checked_next_multiple_of(GRAIN)?;
            }
        }
        None
    }

    /// Written: read-and-execute from here on. False if that failed.
    fn seal(&self) -> bool {
        let mut old = 0;
        // SAFETY: our own allocation, written and not yet run.
        unsafe {
            if VirtualProtect(self.at as *const _, self.size, PAGE_EXECUTE_READ, &mut old) == 0 {
                return false;
            }
            FlushInstructionCache(GetCurrentProcess(), self.at as *const _, self.size);
        }
        true
    }

    fn free(self) {
        // SAFETY: our own allocation, which nothing points into.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

/// How [`apply`] is to go about it.
pub struct Options<'a> {
    /// Plan only: nothing is allocated or written.
    pub list: bool,
    /// Kinds whose sites stay on the exception path whatever the CPU.
    pub hold: &'a [Kind],
    /// Whether the CPU runs a kind itself ([`native`]).
    pub native: &'a dyn Fn(Kind) -> bool,
    /// Whether the stubs are made known to crash reports and stack walks
    /// (`stub_at`): once a process, for the title's image.
    pub register: bool,
}

/// What became of one site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// New bytes over the instruction.
    InPlace(Vec<u8>),
    /// A jump to its stub, whose code is at this address (0 in a plan).
    Stub(usize),
    /// Left as it was.
    Left(Why),
}

/// What [`apply`] did (or, in a plan, would do), site by site.
#[derive(Debug)]
pub struct Report {
    /// Whether this is a plan only.
    pub listed: bool,
    pub sites: Vec<(u32, Kind, Outcome)>,
    /// The stubs' memory, if any was needed.
    pub arena: Option<(usize, usize)>,
}

impl Report {
    /// How many sites of `kind` ended as `pick` says.
    fn count(&self, kind: Kind, pick: impl Fn(&Outcome) -> bool) -> usize {
        self.sites.iter().filter(|(_, k, o)| *k == kind && pick(o)).count()
    }

    /// Sites rewritten (or, in a plan, to be).
    pub fn rewritten(&self) -> usize {
        self.sites.iter().filter(|(_, _, o)| !matches!(o, Outcome::Left(_))).count()
    }

    /// Sites that stay on the exception path: left, and not because the
    /// CPU has the instruction.
    pub fn trapped(&self) -> usize {
        self.sites
            .iter()
            .filter(|(_, _, o)| matches!(o, Outcome::Left(why) if *why != Why::Native))
            .count()
    }

    /// One line: how many sites were rewritten per kind and how, how many
    /// the CPU runs itself, how many are left to the exception path and why.
    pub fn summary(&self) -> String {
        let per_kind = |pick: &dyn Fn(&Outcome) -> bool, how: &str| -> Vec<String> {
            Kind::ALL
                .iter()
                .filter_map(|&k| {
                    let n = self.count(k, pick);
                    (n > 0).then(|| format!("{} {n}{how}", k.name()))
                })
                .collect()
        };
        let or_none = |list: Vec<String>| if list.is_empty() { "none".into() } else { list.join(", ") };
        let mut made = per_kind(&|o| matches!(o, Outcome::Stub(_)), " (stubs)");
        made.extend(per_kind(&|o| matches!(o, Outcome::InPlace(_)), " (in place)"));
        let native = per_kind(&|o| *o == Outcome::Left(Why::Native), "");
        let mut trapped = Vec::new();
        for &kind in &Kind::ALL {
            let mut reasons: Vec<Why> = Vec::new();
            for (_, k, o) in &self.sites {
                if let Outcome::Left(why) = o
                    && *k == kind
                    && *why != Why::Native
                    && !reasons.contains(why)
                {
                    reasons.push(*why);
                }
            }
            for why in reasons {
                let n = self.count(kind, |o| *o == Outcome::Left(why));
                trapped.push(format!("{} {n} ({})", kind.name(), why.text()));
            }
        }
        format!(
            "{} sites: {} {}; the CPU has {}; left to the exception path: {}{}",
            self.sites.len(),
            if self.listed { "would rewrite" } else { "rewritten" },
            or_none(made),
            or_none(native),
            or_none(trapped),
            self.arena.map_or(String::new(), |(from, to)| format!("; stubs at {from:#x}..{to:#x}")),
        )
    }

    /// One line a site: where, what, the bytes covered and what became of it.
    pub fn lines<'a>(&'a self, sites: &'a [Site<'a>]) -> impl Iterator<Item = String> + 'a {
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        self.sites.iter().zip(sites).map(move |((rva, kind, outcome), site)| {
            let what = match outcome {
                Outcome::InPlace(new) => format!("in place -> {}", hex(new)),
                Outcome::Stub(0) => "a jump to a stub".to_owned(),
                Outcome::Stub(at) => format!("a jump to its stub at {at:#x}"),
                Outcome::Left(why) => format!("left: {}", why.text()),
            };
            let moved = site.expect.len() - usize::from(site.len);
            let with = if moved > 0 { format!(" (+{moved} bytes moved)") } else { String::new() };
            format!("+{rva:#x} {} {}{with}: {what}", kind.name(), hex(site.expect))
        })
    }
}

/// Rewrites `sites` in the image mapped at `base` (`size` bytes), each on
/// its own: one that cannot be is left as it was, with the reason. With
/// [`Options::list`] nothing is written and the report is the plan.
///
/// The stubs are written into fresh read-write memory, that memory is made
/// read-and-execute, and only then is any site made to jump into it.
///
/// # Safety
/// `base..base + size` is an image mapped in this process and no thread
/// runs its code (it has not been started).
pub unsafe fn apply(base: usize, size: usize, sites: &[Site<'_>], options: &Options<'_>) -> Report {
    let mut outcomes: Vec<Outcome> = sites
        .iter()
        .map(|site| {
            if (options.native)(site.kind) {
                return Outcome::Left(Why::Native);
            }
            if options.hold.contains(&site.kind) {
                return Outcome::Left(Why::Held);
            }
            let (at, window) = (site.rva as usize, site.expect.len());
            if at.checked_add(window).is_none_or(|end| end > size) {
                return Outcome::Left(Why::Outside);
            }
            // SAFETY: inside the mapped image (checked above), read only.
            let code = unsafe { std::slice::from_raw_parts((base + at) as *const u8, window) };
            match plan(site, code) {
                Ok(Plan::InPlace(new)) => Outcome::InPlace(new),
                Ok(Plan::Stub) => Outcome::Stub(0),
                Err(why) => Outcome::Left(why),
            }
        })
        .collect();
    let report = |outcomes: Vec<Outcome>, arena| Report {
        listed: options.list,
        sites: sites.iter().zip(outcomes).map(|(s, o)| (s.rva, s.kind, o)).collect(),
        arena,
    };
    if options.list {
        return report(outcomes, None);
    }

    // The stubs, whole and executable before any site points at one.
    let wanted = outcomes.iter().filter(|o| matches!(o, Outcome::Stub(_))).count();
    let mut arena = None;
    let mut infos = Vec::new();
    if wanted > 0 {
        let leave = |outcomes: &mut Vec<Outcome>, why| {
            for o in outcomes.iter_mut().filter(|o| matches!(o, Outcome::Stub(_))) {
                *o = Outcome::Left(why);
            }
        };
        match Arena::near(base, base + size, wanted * STUB_ROOM) {
            None => leave(&mut outcomes, Why::NoArena),
            Some(a) => {
                let mut slot = a.at;
                for (site, outcome) in sites.iter().zip(outcomes.iter_mut()) {
                    if !matches!(outcome, Outcome::Stub(_)) {
                        continue;
                    }
                    let at = base + site.rva as usize;
                    let code = slot + STUB_CODE;
                    let built = stub(site, slot, at + site.expect.len())
                        .and_then(|built| rel32(at + 5, code).map(|_| built));
                    match built {
                        Ok(built) => {
                            // SAFETY: the slot is inside the arena, which is
                            // ours and read-write; a stub fits its slot.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    built.constants.as_ptr().cast::<u8>(),
                                    slot as *mut u8,
                                    STUB_CODE,
                                );
                                std::ptr::copy_nonoverlapping(
                                    built.code.as_ptr(),
                                    code as *mut u8,
                                    built.code.len(),
                                );
                            }
                            infos.push(StubInfo {
                                code,
                                end: code + built.code.len(),
                                site: at,
                                rva: site.rva,
                                kind: site.kind,
                                frame: built.frame,
                                restored: code + built.restored,
                            });
                            *outcome = Outcome::Stub(code);
                            slot += STUB_ROOM;
                        }
                        Err(why) => *outcome = Outcome::Left(why),
                    }
                }
                if a.seal() {
                    arena = Some((a.at, a.at + a.size));
                } else {
                    leave(&mut outcomes, Why::Protect);
                    infos.clear();
                    a.free();
                }
            }
        }
    }

    // The sites, each checked again and written as a group of one.
    for (site, outcome) in sites.iter().zip(outcomes.iter_mut()) {
        let new = match outcome {
            Outcome::InPlace(new) => new.clone(),
            Outcome::Stub(code) => {
                let at = base + site.rva as usize;
                let mut new = vec![0x90; site.expect.len()];
                new[0] = 0xe9;
                match rel32(at + 5, *code) {
                    Ok(distance) => new[1..5].copy_from_slice(&distance),
                    Err(why) => {
                        *outcome = Outcome::Left(why);
                        continue;
                    }
                }
                new
            }
            Outcome::Left(_) => continue,
        };
        let patch = MemoryPatch { name: site.kind.name(), rva: site.rva, expect: site.expect, write: &new };
        // SAFETY: the caller's image, which no thread runs.
        if let Err(e) = unsafe { patch_mapped(base, size, &[patch]) } {
            infos.retain(|i| i.rva != site.rva);
            *outcome = Outcome::Left(match e {
                PatchError::Length(_) => Why::Malformed,
                PatchError::Outside(_) => Why::Outside,
                PatchError::Mismatch(_) => Why::Bytes,
                PatchError::Protect(_) => Why::Protect,
            });
        }
    }
    if options.register && !infos.is_empty() {
        infos.sort_by_key(|i| i.code);
        let _ = STUBS.set(infos);
    }
    let report = report(outcomes, arena);
    // Beside the exception path's own counts ("AMD-only instructions
    // emulated per second"), which the sites rewritten no longer add to.
    tracing::info!("cpu rewrite: {}", report.summary());
    report
}

/// A stub in use, for crash reports and stack walks: its code is not in
/// any module and has no unwind data, and while it runs the thread is, for
/// every purpose but the next few instructions, at its site in the title.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StubInfo {
    code: usize,
    end: usize,
    /// The site's address in the image.
    site: usize,
    pub(crate) rva: u32,
    pub(crate) kind: Kind,
    frame: usize,
    /// The address just past the instruction that gives `frame` back.
    restored: usize,
}

impl StubInfo {
    /// Where a thread at `rip` inside this stub with stack pointer `rsp`
    /// is as the title's own code has it: at the site, with the stack
    /// pointer the title had there (the stub lowers it by its frame with
    /// its first instruction, 5 bytes, and raises it again before the
    /// moved instructions).
    fn leave(&self, rip: u64, rsp: u64) -> (u64, u64) {
        let lowered = rip as usize >= self.code + 5 && (rip as usize) < self.restored;
        (self.site as u64, if lowered { rsp + self.frame as u64 } else { rsp })
    }
}

/// The title image's stubs, sorted by address; set once by [`apply`].
static STUBS: OnceLock<Vec<StubInfo>> = OnceLock::new();

/// The stub whose code holds `address`, if it is one's. Takes no lock and
/// allocates nothing: a stack walk of a suspended thread asks.
pub(crate) fn stub_at(address: usize) -> Option<&'static StubInfo> {
    let stubs = STUBS.get()?;
    let i = stubs.partition_point(|s| s.end <= address);
    stubs.get(i).filter(|s| s.code <= address)
}

/// For a stack walk: the title's own place and stack pointer for a thread
/// that is inside a stub ([`StubInfo::leave`]).
pub(crate) fn step_out(rip: u64, rsp: u64) -> Option<(u64, u64)> {
    stub_at(rip as usize).map(|s| s.leave(rip, rsp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, M128A};

    // The harness that runs rewritten code with every register chosen by
    // the test and reads every register back.
    //
    // `victory_rewrite_test_enter(state)` keeps the caller's registers on
    // its stack, loads the sixteen xmm registers, the flags and the general
    // registers (the stack pointer too) from `state`, and goes to
    // `state.target`. The code under test ends in a thunk the test wrote
    // (`Lab::place`), which stores `rax` into `state`, loads `rax` with
    // `state` and jumps to `victory_rewrite_test_capture`: that stores all
    // the other registers and the flags, takes the caller's stack and
    // registers back, and returns to `enter`'s caller. Only `mov`, `push`,
    // `pop`, `ret` and `movdqu` run between the flags' load and their
    // store: none changes a flag.
    core::arch::global_asm!(
        ".globl victory_rewrite_test_enter",
        ".globl victory_rewrite_test_capture",
        "victory_rewrite_test_enter:",
        "push rbx",
        "push rbp",
        "push rsi",
        "push rdi",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "sub rsp, 0xa8",
        "movdqu xmmword ptr [rsp + 0x00], xmm6",
        "movdqu xmmword ptr [rsp + 0x10], xmm7",
        "movdqu xmmword ptr [rsp + 0x20], xmm8",
        "movdqu xmmword ptr [rsp + 0x30], xmm9",
        "movdqu xmmword ptr [rsp + 0x40], xmm10",
        "movdqu xmmword ptr [rsp + 0x50], xmm11",
        "movdqu xmmword ptr [rsp + 0x60], xmm12",
        "movdqu xmmword ptr [rsp + 0x70], xmm13",
        "movdqu xmmword ptr [rsp + 0x80], xmm14",
        "movdqu xmmword ptr [rsp + 0x90], xmm15",
        "mov qword ptr [rcx + 0x90], rsp",
        "movdqu xmm0, xmmword ptr [rcx + 0xa0]",
        "movdqu xmm1, xmmword ptr [rcx + 0xb0]",
        "movdqu xmm2, xmmword ptr [rcx + 0xc0]",
        "movdqu xmm3, xmmword ptr [rcx + 0xd0]",
        "movdqu xmm4, xmmword ptr [rcx + 0xe0]",
        "movdqu xmm5, xmmword ptr [rcx + 0xf0]",
        "movdqu xmm6, xmmword ptr [rcx + 0x100]",
        "movdqu xmm7, xmmword ptr [rcx + 0x110]",
        "movdqu xmm8, xmmword ptr [rcx + 0x120]",
        "movdqu xmm9, xmmword ptr [rcx + 0x130]",
        "movdqu xmm10, xmmword ptr [rcx + 0x140]",
        "movdqu xmm11, xmmword ptr [rcx + 0x150]",
        "movdqu xmm12, xmmword ptr [rcx + 0x160]",
        "movdqu xmm13, xmmword ptr [rcx + 0x170]",
        "movdqu xmm14, xmmword ptr [rcx + 0x180]",
        "movdqu xmm15, xmmword ptr [rcx + 0x190]",
        "push qword ptr [rcx + 0x80]",
        "popfq",
        "mov rax, qword ptr [rcx + 0x00]",
        "mov rdx, qword ptr [rcx + 0x10]",
        "mov rbx, qword ptr [rcx + 0x18]",
        "mov rbp, qword ptr [rcx + 0x28]",
        "mov rsi, qword ptr [rcx + 0x30]",
        "mov rdi, qword ptr [rcx + 0x38]",
        "mov r8, qword ptr [rcx + 0x40]",
        "mov r9, qword ptr [rcx + 0x48]",
        "mov r10, qword ptr [rcx + 0x50]",
        "mov r11, qword ptr [rcx + 0x58]",
        "mov r12, qword ptr [rcx + 0x60]",
        "mov r13, qword ptr [rcx + 0x68]",
        "mov r14, qword ptr [rcx + 0x70]",
        "mov r15, qword ptr [rcx + 0x78]",
        "mov rsp, qword ptr [rcx + 0x20]",
        "push qword ptr [rcx + 0x88]",
        "mov rcx, qword ptr [rcx + 0x08]",
        "ret",
        "victory_rewrite_test_capture:",
        "mov qword ptr [rax + 0x08], rcx",
        "mov qword ptr [rax + 0x10], rdx",
        "mov qword ptr [rax + 0x18], rbx",
        "mov qword ptr [rax + 0x20], rsp",
        "mov qword ptr [rax + 0x28], rbp",
        "mov qword ptr [rax + 0x30], rsi",
        "mov qword ptr [rax + 0x38], rdi",
        "mov qword ptr [rax + 0x40], r8",
        "mov qword ptr [rax + 0x48], r9",
        "mov qword ptr [rax + 0x50], r10",
        "mov qword ptr [rax + 0x58], r11",
        "mov qword ptr [rax + 0x60], r12",
        "mov qword ptr [rax + 0x68], r13",
        "mov qword ptr [rax + 0x70], r14",
        "mov qword ptr [rax + 0x78], r15",
        "movdqu xmmword ptr [rax + 0xa0], xmm0",
        "movdqu xmmword ptr [rax + 0xb0], xmm1",
        "movdqu xmmword ptr [rax + 0xc0], xmm2",
        "movdqu xmmword ptr [rax + 0xd0], xmm3",
        "movdqu xmmword ptr [rax + 0xe0], xmm4",
        "movdqu xmmword ptr [rax + 0xf0], xmm5",
        "movdqu xmmword ptr [rax + 0x100], xmm6",
        "movdqu xmmword ptr [rax + 0x110], xmm7",
        "movdqu xmmword ptr [rax + 0x120], xmm8",
        "movdqu xmmword ptr [rax + 0x130], xmm9",
        "movdqu xmmword ptr [rax + 0x140], xmm10",
        "movdqu xmmword ptr [rax + 0x150], xmm11",
        "movdqu xmmword ptr [rax + 0x160], xmm12",
        "movdqu xmmword ptr [rax + 0x170], xmm13",
        "movdqu xmmword ptr [rax + 0x180], xmm14",
        "movdqu xmmword ptr [rax + 0x190], xmm15",
        "mov rsp, qword ptr [rax + 0x90]",
        "pushfq",
        "pop qword ptr [rax + 0x80]",
        "movdqu xmm6, xmmword ptr [rsp + 0x00]",
        "movdqu xmm7, xmmword ptr [rsp + 0x10]",
        "movdqu xmm8, xmmword ptr [rsp + 0x20]",
        "movdqu xmm9, xmmword ptr [rsp + 0x30]",
        "movdqu xmm10, xmmword ptr [rsp + 0x40]",
        "movdqu xmm11, xmmword ptr [rsp + 0x50]",
        "movdqu xmm12, xmmword ptr [rsp + 0x60]",
        "movdqu xmm13, xmmword ptr [rsp + 0x70]",
        "movdqu xmm14, xmmword ptr [rsp + 0x80]",
        "movdqu xmm15, xmmword ptr [rsp + 0x90]",
        "add rsp, 0xa8",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rdi",
        "pop rsi",
        "pop rbp",
        "pop rbx",
        "ret",
    );

    unsafe extern "C" {
        fn victory_rewrite_test_enter(state: *mut State);
        fn victory_rewrite_test_capture();
    }

    /// The registers the harness loads and reads back; the offsets are the
    /// ones its assembly uses.
    #[repr(C, align(16))]
    #[derive(Clone, Copy, PartialEq, Debug)]
    struct State {
        /// In the order instructions number them: rax, rcx, rdx, rbx, rsp,
        /// rbp, rsi, rdi, r8 to r15.
        gpr: [u64; 16],
        flags: u64,
        target: u64,
        host_rsp: u64,
        capture: u64,
        xmm: [[u64; 2]; 16],
    }
    const _: () = assert!(std::mem::offset_of!(State, flags) == 0x80);
    const _: () = assert!(std::mem::offset_of!(State, target) == 0x88);
    const _: () = assert!(std::mem::offset_of!(State, host_rsp) == 0x90);
    const _: () = assert!(std::mem::offset_of!(State, capture) == 0x98);
    const _: () = assert!(std::mem::offset_of!(State, xmm) == 0xa0);

    const RAX: usize = 0;
    const RCX: usize = 1;
    const RDX: usize = 2;
    const RSP: usize = 4;
    /// CF, PF, AF, ZF, SF and OF: the flags a test sets, and with DF the
    /// ones it compares.
    const STATUS: u64 = 0x8d5;
    const COMPARED: u64 = STATUS | 0x400;

    /// xorshift64*: test inputs that differ from round to round, the same
    /// every run of the test.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const LAB_SIZE: usize = 0x4_0000;
    const STACK_WORDS: usize = 0x2000;
    const STACK_FILL: u64 = 0x5a5a_a5a5_5a5a_a5a5;

    /// A made-up image: code the tests write, read-and-execute as an
    /// image's is, with a stack and a set of registers to run it on.
    struct Lab {
        image: usize,
        used: usize,
        sealed: bool,
        state: Box<State>,
        stack: Vec<u64>,
    }

    impl Lab {
        fn new() -> Lab {
            // SAFETY: a fresh allocation of our own.
            let image = unsafe {
                VirtualAlloc(std::ptr::null(), LAB_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            } as usize;
            assert_ne!(image, 0);
            // SAFETY: inside the allocation, writable.
            unsafe { (image as *mut u8).write_bytes(0xcc, LAB_SIZE) };
            // SAFETY: plain numbers, all zeros is valid.
            let state: Box<State> = Box::new(unsafe { std::mem::zeroed() });
            Lab { image, used: 0x40, sealed: false, state, stack: vec![STACK_FILL; STACK_WORDS] }
        }

        /// Puts `code` into the image, followed by the thunk that ends a
        /// run; returns where it starts.
        fn place(&mut self, code: &[u8]) -> u32 {
            assert!(!self.sealed);
            let rva = self.used;
            let state = (&raw const *self.state) as u64;
            let mut bytes = code.to_vec();
            bytes.extend([0x48, 0xa3]); // mov [state.gpr[0]], rax
            bytes.extend(state.to_le_bytes());
            bytes.extend([0x48, 0xb8]); // mov rax, state
            bytes.extend(state.to_le_bytes());
            bytes.extend([0xff, 0xa0, 0x98, 0, 0, 0]); // jmp [rax + 0x98]
            assert!(rva + bytes.len() <= LAB_SIZE);
            // SAFETY: inside the allocation, still writable.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), (self.image + rva) as *mut u8, bytes.len());
            }
            self.used = (rva + bytes.len()).next_multiple_of(16) + 16;
            rva as u32
        }

        /// Written: read-and-execute, as a mapped image's code is.
        fn seal(&mut self) {
            let mut old = 0;
            // SAFETY: our allocation.
            let ok = unsafe { VirtualProtect(self.image as _, LAB_SIZE, PAGE_EXECUTE_READ, &mut old) };
            assert_ne!(ok, 0);
            self.sealed = true;
        }

        fn bytes(&self, rva: u32, n: usize) -> Vec<u8> {
            // SAFETY: inside the allocation, readable.
            unsafe { std::slice::from_raw_parts((self.image + rva as usize) as *const u8, n).to_vec() }
        }

        /// A stack pointer inside the lab's stack, `shift` bytes off a
        /// 16-byte boundary.
        fn rsp(&self, shift: usize) -> u64 {
            self.stack.as_ptr() as u64 + (STACK_WORDS as u64 / 2) * 8 + shift as u64
        }

        /// Rewrites `sites` as a CPU with none of the instructions would.
        fn rewrite(&self, sites: &[Site<'_>]) -> Report {
            assert!(self.sealed);
            let options = Options { list: false, hold: &[], native: &|_| false, register: false };
            // SAFETY: the lab's image, which nothing runs now.
            unsafe { apply(self.image, LAB_SIZE, sites, &options) }
        }

        /// Runs the code at `rva` from `from` to its thunk; the registers
        /// after. The stack at and above the stack pointer must come back
        /// untouched.
        fn run(&mut self, rva: u32, from: &State) -> State {
            assert!(self.sealed);
            *self.state = *from;
            self.state.target = (self.image + rva as usize) as u64;
            self.state.capture = victory_rewrite_test_capture as *const () as u64;
            let low = self.stack.as_ptr() as u64;
            let high = low + STACK_WORDS as u64 * 8;
            let rsp = from.gpr[RSP];
            assert!(rsp >= low + 0x1000 && rsp + 0x1000 <= high);
            self.stack.fill(STACK_FILL);
            // SAFETY: the harness described at its assembly: `state` is
            // the lab's own, its target code ends in the thunk `place`
            // wrote, and its stack pointer has room on both sides.
            unsafe { victory_rewrite_test_enter(&raw mut *self.state) };
            let after = *self.state;
            let kept = ((after.gpr[RSP] - low) as usize).div_ceil(8);
            assert!(self.stack[kept..].iter().all(|&w| w == STACK_FILL), "the stack above rsp changed");
            after
        }

        /// Registers a test starts from: everything random, the stack
        /// pointer in the lab's stack at any alignment.
        fn random_state(&self, rng: &mut Rng) -> State {
            // SAFETY: plain numbers.
            let mut s: State = unsafe { std::mem::zeroed() };
            s.gpr.iter_mut().for_each(|r| *r = rng.next());
            s.xmm.iter_mut().for_each(|x| *x = [rng.next(), rng.next()]);
            s.gpr[RSP] = self.rsp(rng.below(16) as usize);
            s.flags = 0x202 | (rng.next() & STATUS);
            s
        }
    }

    impl Drop for Lab {
        fn drop(&mut self) {
            // SAFETY: our allocation, no longer run. (A stub arena made
            // for it stays: a few pages for the test process's life.)
            unsafe { VirtualFree(self.image as _, 0, MEM_RELEASE) };
        }
    }

    /// `state` as the exception handler would be given it.
    fn context_of(state: &State) -> Box<crate::crash::Aligned16> {
        // SAFETY: a zeroed CONTEXT is valid.
        let mut boxed = Box::new(crate::crash::Aligned16(unsafe { std::mem::zeroed::<CONTEXT>() }));
        let c = &mut boxed.0;
        let g = &state.gpr;
        (c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi) = (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
        (c.R8, c.R9, c.R10, c.R11) = (g[8], g[9], g[10], g[11]);
        (c.R12, c.R13, c.R14, c.R15) = (g[12], g[13], g[14], g[15]);
        c.EFlags = state.flags as u32;
        for (i, x) in state.xmm.iter().enumerate() {
            // SAFETY: the legacy save area is the union's plain data.
            unsafe { c.Anonymous.FltSave.XmmRegisters[i] = M128A { Low: x[0], High: x[1] as i64 } };
        }
        boxed
    }

    /// The registers in `c`, the rest as in `like`.
    fn state_of(c: &CONTEXT, like: &State) -> State {
        let mut s = *like;
        s.gpr = [
            c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13,
            c.R14, c.R15,
        ];
        s.flags = u64::from(c.EFlags);
        for (i, x) in s.xmm.iter_mut().enumerate() {
            // SAFETY: as in `context_of`.
            let r = unsafe { c.Anonymous.FltSave.XmmRegisters[i] };
            *x = [r.Low, r.High as u64];
        }
        s
    }

    /// What the exception handler's emulation makes of `code[at..]` from
    /// `from`: the registers after, and where it goes on.
    fn emulated(code: &[u8], at: usize, from: &State) -> (State, usize) {
        let mut context = context_of(from);
        assert!(crate::crash::emulate_in(code, at, &mut context.0), "{code:02x?} not emulated");
        (state_of(&context.0, from), context.0.Rip as usize)
    }

    #[track_caller]
    fn assert_same(got: &State, want: &State, what: &str) {
        assert_eq!(got.gpr, want.gpr, "{what}: general registers");
        assert_eq!(got.xmm, want.xmm, "{what}: xmm registers");
        assert_eq!(got.flags & COMPARED, want.flags & COMPARED, "{what}: flags");
    }

    fn insertq(dst: u8, src: u8, len: u8, index: u8) -> Vec<u8> {
        let mut code = vec![0xf2];
        let rex = 0x40 | (dst >> 3) << 2 | src >> 3;
        if rex != 0x40 {
            code.push(rex);
        }
        code.extend([0x0f, 0x78, 0xc0 | (dst & 7) << 3 | (src & 7), len, index]);
        code
    }

    fn row(kind: Kind, len: usize, expect: &[u8]) -> Site<'_> {
        Site { rva: 0, kind, len: len as u8, expect, targets: 0 }
    }

    /// Made-up instances of each kind: (kind, instruction, moved after it).
    const FORMS: [(Kind, &[u8], &[u8]); 6] = [
        (Kind::Clwb, &[0x66, 0x0f, 0xae, 0x72, 0x10], &[]), // clwb [rdx + 10h]
        (Kind::Clwb, &[0x66, 0x43, 0x0f, 0xae, 0x34, 0x18], &[]), // clwb [r8 + r11]
        (Kind::Monitorx, &[0x0f, 0x01, 0xfa], &[]),
        (Kind::Mwaitx, &[0x0f, 0x01, 0xfb], &[]),
        (Kind::Insertq, &[0xf2, 0x0f, 0x78, 0xd3, 0x05, 0x07], &[]), // insertq xmm2, xmm3, 5, 7
        (Kind::Clzero, &[0x0f, 0x01, 0xfc], &[0x48, 0x83, 0xe9, 0x40]), // clzero; sub rcx, 40h
    ];

    #[test]
    fn each_kind_decodes_and_refuses_the_others() {
        for (kind, ins, moved) in FORMS {
            let code = [ins, moved].concat();
            assert!(decodes_as(kind, ins), "{kind:?}");
            let p = plan(&row(kind, ins.len(), &code), &code).unwrap_or_else(|e| panic!("{kind:?}: {e:?}"));
            assert_eq!(p == Plan::Stub, kind.stubbed(), "{kind:?}");
            for (other, ins2, moved2) in FORMS {
                if other == kind {
                    continue;
                }
                let code2 = [ins2, moved2].concat();
                assert!(!decodes_as(kind, ins2), "{kind:?} accepted {other:?}");
                let p = plan(&row(kind, ins2.len(), &code2), &code2);
                assert!(p.is_err(), "{kind:?} planned over {other:?}");
            }
        }
        // extrq has no rewrite, and is not an insertq.
        assert!(!decodes_as(Kind::Insertq, &[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x00]));
        // The in-place forms: the same length, the hint's own operand.
        let in_place = |kind, ins: &[u8]| match plan(&row(kind, ins.len(), ins), ins) {
            Ok(Plan::InPlace(new)) => new,
            other => panic!("{other:?}"),
        };
        // nop word [rdx + 10h]; nop word [r8 + r11]; nop eax; pause, nop.
        assert_eq!(in_place(Kind::Clwb, FORMS[0].1), [0x66, 0x0f, 0x1f, 0x42, 0x10]);
        assert_eq!(in_place(Kind::Clwb, FORMS[1].1), [0x66, 0x43, 0x0f, 0x1f, 0x04, 0x18]);
        assert_eq!(in_place(Kind::Monitorx, FORMS[2].1), [0x0f, 0x1f, 0xc0]);
        assert_eq!(in_place(Kind::Mwaitx, FORMS[3].1), [0xf3, 0x90, 0x90]);
    }

    #[test]
    fn a_site_is_planned_only_as_its_row_says() {
        let clzero = [0x0f, 0x01, 0xfc];
        // What moves with a clzero must be register-only.
        for (moved, ok) in [
            (&[0x48, 0x83, 0xc3, 0x20][..], true),                // add rbx, 20h
            (&[0x4d, 0x39, 0xd3], true),                          // cmp r11, r10
            (&[0x48, 0x8b, 0x01], false),                         // mov rax, [rcx]
            (&[0x48, 0x8b, 0x05, 0x10, 0x00, 0x00, 0x00], false), // mov rax, [rip + 10h]
            (&[0x48, 0x3b, 0x05, 0x10, 0x00, 0x00, 0x00], false), // cmp rax, [rip + 10h]
            (&[0xe8, 0, 0, 0, 0], false),                         // call rel32
            (&[0x72, 0xf4, 0x90], false),                         // jb; nop
            (&[0x48, 0x83, 0xc0], false),                         // cut off
        ] {
            let code = [&clzero[..], moved].concat();
            let p = plan(&row(Kind::Clzero, 3, &code), &code);
            assert_eq!(p, if ok { Ok(Plan::Stub) } else { Err(Why::Moved) }, "{moved:02x?}");
        }
        let code = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40];
        let site = row(Kind::Clzero, 3, &code);
        assert_eq!(plan(&site, &code), Ok(Plan::Stub));
        // One byte that is not the row's: nothing is planned.
        for i in 0..code.len() {
            let mut other = code;
            other[i] ^= 0x01;
            assert_eq!(plan(&site, &other), Err(Why::Bytes), "byte {i}");
        }
        assert_eq!(plan(&site, &code[..6]), Err(Why::Outside));
        // The row names another instruction than its bytes are.
        assert_eq!(plan(&Site { kind: Kind::Insertq, ..site }, &code), Err(Why::Decode));
        // A branch that lands inside the bytes covered (its first byte is
        // where the loops come back to, and past the last is outside).
        for (targets, ok) in [(0x0001u16, true), (0x0080, true), (0x0008, false), (0x0040, false), (0x0002, false)] {
            let p = plan(&Site { targets, ..site }, &code);
            assert_eq!(p, if ok { Ok(Plan::Stub) } else { Err(Why::Target) }, "{targets:#x}");
        }
        // A clzero alone has no room for a jump; a hint moves nothing.
        assert_eq!(plan(&row(Kind::Clzero, 3, &clzero), &clzero), Err(Why::Short));
        let code = [0x0f, 0x01, 0xfa, 0x48, 0x89, 0xc1];
        assert_eq!(plan(&row(Kind::Monitorx, 3, &code), &code), Err(Why::Malformed));
        assert_eq!(plan(&row(Kind::Clzero, 0, &clzero), &clzero), Err(Why::Malformed));
        assert_eq!(plan(&row(Kind::Clzero, 4, &clzero), &clzero), Err(Why::Malformed));
    }

    #[test]
    fn the_arena_is_within_a_jumps_reach_of_the_whole_image() {
        // The span of allowed starts: from either end of it, the farthest
        // byte of the image is a 32-bit displacement away.
        for (base, end, size) in [
            (0x1_4000_0000usize, 0x1_4b06_d000usize, 29 * STUB_ROOM),
            (0x7ff6_0000_0000, 0x7ff6_0b06_d000, 0x1_0000),
            (0x1_0000, 0x4_0000, 0x1000),
            (0x1_4000_0000, 0x1_8000_0000, 0x1000),
        ] {
            let (lowest, highest) = span(base, end, size).unwrap();
            assert!(lowest <= highest && lowest % GRAIN == 0 && highest % GRAIN == 0);
            for start in [lowest, highest] {
                for (from, to) in [(start, end), (start + size, base), (end, start), (base, start + size)] {
                    assert!(rel32(from, to).is_ok(), "{from:#x} to {to:#x}");
                }
            }
            assert!(end.saturating_sub(lowest) <= REACH && highest + size - base <= REACH);
        }
        // An image wider than the reach has no such place.
        assert_eq!(span(0x1_0000_0000, 0x1_ffff_0000, 0x1000), None);
        // A displacement is exactly 32 bits, or the site is left.
        assert_eq!(rel32(0x1000, 0x1000 + 0x7fff_ffff), Ok(0x7fff_ffffi32.to_le_bytes()));
        assert_eq!(rel32(0x1000, 0x1000 + 0x8000_0000), Err(Why::Reach));
        assert_eq!(rel32(0x9000_0000, 0x1000_0000), Ok(i32::MIN.to_le_bytes()));
        assert_eq!(rel32(0x9000_0001, 0x1000_0000), Err(Why::Reach));
        // A real one, beside a real allocation: inside the span,
        // read-write until sealed, read-and-execute after.
        let lab = Lab::new();
        let (base, end) = (lab.image, lab.image + LAB_SIZE);
        let arena = Arena::near(base, end, 3 * STUB_ROOM).expect("free memory near the lab");
        let (lowest, highest) = span(base, end, 3 * STUB_ROOM).unwrap();
        assert!((lowest..=highest).contains(&arena.at), "{:#x} outside {lowest:#x}..={highest:#x}", arena.at);
        assert!(arena.at >= end || arena.at + arena.size <= base);
        assert_eq!(protection(arena.at), PAGE_READWRITE);
        assert!(arena.seal());
        assert_eq!(protection(arena.at), PAGE_EXECUTE_READ);
        arena.free();
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: a plain-data structure, all zeros is valid.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a plain query of our own pages.
        assert_ne!(unsafe { VirtualQuery(at as _, &mut info, size_of_val(&info)) }, 0);
        info.Protect
    }

    #[test]
    fn a_made_up_image_is_rewritten_site_by_site() {
        let ins = insertq(2, 3, 5, 7);
        let clzero_add = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40];
        let clzero_rip = [0x0f, 0x01, 0xfc, 0x48, 0x8b, 0x05, 0x10, 0x00, 0x00, 0x00];
        let (monitorx, mwaitx, clwb) = ([0x0f, 0x01, 0xfa], [0x0f, 0x01, 0xfb], [0x66, 0x0f, 0xae, 0x31]);
        let mut wrong = ins.clone();
        wrong[5] ^= 0x10;
        let mut lab = Lab::new();
        let codes: [&[u8]; 8] =
            [&ins, &wrong, &clzero_rip, &clzero_add, &clzero_add, &monitorx, &mwaitx, &clwb];
        let at: Vec<u32> = codes.iter().map(|code| lab.place(code)).collect();
        lab.seal();
        let sites = [
            Site { rva: at[0], ..row(Kind::Insertq, 6, &ins) },
            // One byte of the image is not the row's.
            Site { rva: at[1], ..row(Kind::Insertq, 6, &ins) },
            // A rip-relative instruction would move.
            Site { rva: at[2], ..row(Kind::Clzero, 3, &clzero_rip) },
            // A branch lands on the instruction that would move.
            Site { rva: at[3], targets: 0x8, ..row(Kind::Clzero, 3, &clzero_add) },
            Site { rva: at[4], ..row(Kind::Clzero, 3, &clzero_add) },
            Site { rva: at[5], ..row(Kind::Monitorx, 3, &monitorx) },
            Site { rva: at[6], ..row(Kind::Mwaitx, 3, &mwaitx) },
            Site { rva: at[7], ..row(Kind::Clwb, 4, &clwb) },
            Site { rva: (LAB_SIZE - 2) as u32, ..row(Kind::Monitorx, 3, &monitorx) },
        ];
        let before: Vec<Vec<u8>> = sites[..8].iter().map(|s| lab.bytes(s.rva, s.expect.len())).collect();
        let unchanged =
            |lab: &Lab, i: usize| assert_eq!(lab.bytes(sites[i].rva, before[i].len()), before[i], "site {i}");
        // This CPU's clwb is its own (as this PC's is); mwaitx is held back.
        let native = |kind| kind == Kind::Clwb;
        let hold = [Kind::Mwaitx];

        // A plan writes nothing.
        let options = Options { list: true, hold: &hold, native: &native, register: false };
        // SAFETY: the lab's image, which nothing runs.
        let listed = unsafe { apply(lab.image, LAB_SIZE, &sites, &options) };
        (0..8).for_each(|i| unchanged(&lab, i));
        assert!(listed.listed && listed.arena.is_none());
        assert_eq!(listed.sites[0].2, Outcome::Stub(0));
        assert_eq!(listed.lines(&sites).count(), sites.len());

        // SAFETY: as above.
        let done = unsafe { apply(lab.image, LAB_SIZE, &sites, &Options { list: false, ..options }) };
        let outcome = |i: usize| &done.sites[i].2;
        let (from, to) = done.arena.expect("two stubs");
        for i in [0, 4] {
            let Outcome::Stub(code) = *outcome(i) else { panic!("site {i}: {:?}", outcome(i)) };
            assert!((from..to).contains(&code));
            let now = lab.bytes(sites[i].rva, before[i].len());
            let rel = i32::from_le_bytes(now[1..5].try_into().unwrap());
            assert_eq!(now[0], 0xe9);
            assert_eq!((lab.image + sites[i].rva as usize + 5).wrapping_add_signed(rel as isize), code);
            assert!(now[5..].iter().all(|&b| b == 0x90));
        }
        assert_eq!(*outcome(1), Outcome::Left(Why::Bytes));
        assert_eq!(*outcome(2), Outcome::Left(Why::Moved));
        assert_eq!(*outcome(3), Outcome::Left(Why::Target));
        assert_eq!(*outcome(5), Outcome::InPlace(vec![0x0f, 0x1f, 0xc0]));
        assert_eq!(lab.bytes(at[5], 3), [0x0f, 0x1f, 0xc0]);
        assert_eq!(*outcome(6), Outcome::Left(Why::Held));
        assert_eq!(*outcome(7), Outcome::Left(Why::Native));
        assert_eq!(*outcome(8), Outcome::Left(Why::Outside));
        [1, 2, 3, 6, 7].into_iter().for_each(|i| unchanged(&lab, i));
        // The stubs' memory is read-and-execute, never writable again; the
        // image's code is as it was mapped.
        assert_eq!(protection(from), PAGE_EXECUTE_READ);
        assert_eq!(protection(lab.image + at[0] as usize), PAGE_EXECUTE_READ);
        assert_eq!((done.rewritten(), done.trapped()), (3, 5));
        assert_eq!(
            done.summary(),
            format!(
                "9 sites: rewritten insertq 1 (stubs), clzero 1 (stubs), monitorx 1 (in place); the CPU has clwb 1; \
                 left to the exception path: insertq 1 (the expected bytes are not there), clzero 1 (an instruction \
                 that would move with it is not register-only), clzero 1 (a branch lands inside its bytes), \
                 monitorx 1 (outside the image), mwaitx 1 (held back by a switch); stubs at {from:#x}..{to:#x}"
            )
        );
        assert!(listed.summary().starts_with("9 sites: would rewrite insertq 1 (stubs), clzero 1 (stubs), monitorx 1"));
        // Done twice: the bytes are ours now, so nothing more is written.
        // SAFETY: as above.
        let again = unsafe { apply(lab.image, LAB_SIZE, &sites, &Options { list: false, ..options }) };
        assert_eq!(again.rewritten(), 0);
    }

    /// The insertq forms the title's 19 sites have, as (dst, src, len,
    /// index).
    const TITLE_INSERTQ: [(u8, u8, u8, u8); 14] = [
        (9, 7, 8, 0),
        (8, 7, 24, 0),
        (0, 1, 16, 16),
        (11, 12, 16, 16),
        (4, 3, 16, 16),
        (5, 3, 16, 16),
        (6, 3, 16, 16),
        (14, 13, 16, 16),
        (4, 5, 16, 48),
        (0, 2, 16, 16),
        (0, 0, 8, 8),
        (1, 1, 8, 16),
        (0, 0, 8, 24),
        (1, 1, 8, 24),
    ];

    #[test]
    fn insertq_stubs_do_what_the_emulation_does() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut forms = TITLE_INSERTQ.to_vec();
        // Every pair of registers once, then any fields: a length of 0 is
        // 64 bits, and a field may run past bit 63.
        for dst in 0..16u8 {
            for src in 0..16u8 {
                forms.push((dst, src, rng.below(64) as u8, rng.below(64) as u8));
            }
        }
        forms.extend([(3, 5, 0, 0), (3, 5, 0, 17), (6, 6, 0, 63), (15, 15, 63, 63), (0, 15, 1, 0), (15, 0, 64, 64)]);
        let codes: Vec<Vec<u8>> = forms.iter().map(|&(d, s, n, i)| insertq(d, s, n, i)).collect();
        let mut lab = Lab::new();
        let at: Vec<u32> = codes.iter().map(|code| lab.place(code)).collect();
        lab.seal();
        let sites: Vec<Site<'_>> = codes
            .iter()
            .zip(&at)
            .map(|(code, &rva)| Site { rva, ..row(Kind::Insertq, code.len(), code) })
            .collect();
        let report = lab.rewrite(&sites);
        assert_eq!(report.rewritten(), sites.len(), "{}", report.summary());
        for (code, &rva) in codes.iter().zip(&at) {
            for round in 0..24 {
                let mut from = lab.random_state(&mut rng);
                if round % 4 == 0 {
                    // Fields at the edges: all ones, all zeros.
                    let fill = if round % 8 == 0 { [u64::MAX; 2] } else { [0; 2] };
                    from.xmm.iter_mut().for_each(|x| *x = fill);
                    from.xmm[rng.below(16) as usize] = [rng.next(), rng.next()];
                }
                let (want, next) = emulated(code, 0, &from);
                assert_eq!(next, code.len());
                let got = lab.run(rva, &from);
                assert_same(&got, &want, &format!("{code:02x?} round {round}"));
            }
        }
    }

    /// A buffer of whole cache lines for a clzero to work in.
    #[repr(align(64))]
    #[derive(Clone, PartialEq)]
    struct Lines([u8; 64 * 40]);

    impl Lines {
        fn random(rng: &mut Rng) -> Box<Lines> {
            let mut lines = Box::new(Lines([0; 64 * 40]));
            lines.0.iter_mut().for_each(|b| *b = rng.next() as u8 | 1);
            lines
        }

        fn at(&self, offset: usize) -> u64 {
            self.0.as_ptr() as u64 + offset as u64
        }
    }

    #[test]
    fn clzero_stubs_do_what_the_emulation_does() {
        // The instructions the title's ten sites take with them (a jump
        // needs 5 bytes, a clzero is 3).
        // Each sets the flags anew, so two more that leave them alone show
        // what the stub itself does to them: nothing.
        let moved: [&[u8]; 7] = [
            &[0x48, 0x83, 0xc0, 0x40], // add rax, 40h
            &[0x4d, 0x39, 0xda],       // cmp r10, r11
            &[0x4d, 0x39, 0xd1],       // cmp r9, r10
            &[0x4d, 0x39, 0xc8],       // cmp r8, r9
            &[0x4c, 0x39, 0xc2],       // cmp rdx, r8
            &[0x48, 0x89, 0xc3],       // mov rbx, rax
            &[0x4c, 0x8b, 0xd9],       // mov r11, rcx
        ];
        let clzero = [0x0f, 0x01, 0xfc];
        let codes: Vec<Vec<u8>> = moved.iter().map(|m| [&clzero[..], m].concat()).collect();
        let mut lab = Lab::new();
        let at: Vec<u32> = codes.iter().map(|code| lab.place(code)).collect();
        // The moved instructions alone, as the title would run them after
        // a clzero of its own CPU.
        let alone: Vec<u32> = moved.iter().map(|m| lab.place(m)).collect();
        lab.seal();
        let sites: Vec<Site<'_>> =
            codes.iter().zip(&at).map(|(code, &rva)| Site { rva, ..row(Kind::Clzero, 3, code) }).collect();
        let report = lab.rewrite(&sites);
        assert_eq!(report.rewritten(), sites.len(), "{}", report.summary());
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for i in 0..codes.len() {
            for round in 0..64 {
                let mut lines = Lines::random(&mut rng);
                let before = lines.clone();
                let mut from = lab.random_state(&mut rng);
                // Anywhere in a line, its first and last byte too.
                let offset = 64 * (1 + rng.below(38) as usize) + [0, 63, rng.below(64) as usize][round % 3];
                let line = offset & !63;
                from.gpr[RAX] = lines.at(offset);
                // The emulation: one line (nothing after the clzero makes
                // it a loop), then the moved instruction on the CPU.
                let (after_clzero, next) = emulated(&clzero, 0, &from);
                assert_eq!(next, 3);
                let want = lab.run(alone[i], &after_clzero);
                let want_lines = lines.clone();
                assert!(want_lines.0[line..line + 64].iter().all(|&b| b == 0));
                assert!(want_lines.0[..line] == before.0[..line]);
                assert!(want_lines.0[line + 64..] == before.0[line + 64..]);
                // The stub, from the same registers and the same memory.
                lines.0 = before.0;
                let got = lab.run(at[i], &from);
                assert_same(&got, &want, &format!("{:02x?} round {round}", codes[i]));
                assert!(*lines == *want_lines, "{:02x?} round {round}: memory", codes[i]);
            }
        }
    }

    #[test]
    fn clzero_loops_run_through_their_stubs_as_the_emulation_ends_them() {
        // The two loops compilers build around a clzero, with the title's
        // registers: (code, where the clzero is, the bytes covered, the
        // register that counts, the limit).
        let first = |cmp: [u8; 3]| [&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40][..], &cmp, &[0x72, 0xf4]].concat();
        let second = |load: [u8; 3], add: [u8; 4], cmp: [u8; 3]| {
            [&load[..], &add, &[0x0f, 0x01, 0xfc], &cmp, &[0x72, 0xf1]].concat()
        };
        let loops: [(Vec<u8>, usize, usize, usize, usize); 8] = [
            (first([0x4c, 0x39, 0xd0]), 0, 7, 0, 10), // cmp rax, r10
            (first([0x4c, 0x39, 0xc8]), 0, 7, 0, 9),  // cmp rax, r9
            (first([0x4c, 0x39, 0xc0]), 0, 7, 0, 8),  // cmp rax, r8
            (first([0x48, 0x39, 0xd0]), 0, 7, 0, 2),  // cmp rax, rdx
            // mov rax, r10; add r10, 40h; clzero; cmp r10, r11; jb
            (second([0x4c, 0x89, 0xd0], [0x49, 0x83, 0xc2, 0x40], [0x4d, 0x39, 0xda]), 7, 6, 10, 11),
            (second([0x4c, 0x89, 0xc8], [0x49, 0x83, 0xc1, 0x40], [0x4d, 0x39, 0xd1]), 7, 6, 9, 10),
            (second([0x4c, 0x89, 0xc0], [0x49, 0x83, 0xc0, 0x40], [0x4d, 0x39, 0xc8]), 7, 6, 8, 9),
            (second([0x48, 0x89, 0xd0], [0x48, 0x83, 0xc2, 0x40], [0x4c, 0x39, 0xc2]), 7, 6, 2, 8),
        ];
        let mut lab = Lab::new();
        let at: Vec<u32> = loops.iter().map(|(code, ..)| lab.place(code)).collect();
        lab.seal();
        let sites: Vec<Site<'_>> = loops
            .iter()
            .zip(&at)
            .map(|((code, clzero, window, ..), &rva)| Site {
                rva: rva + *clzero as u32,
                ..row(Kind::Clzero, 3, &code[*clzero..clzero + window])
            })
            .collect();
        let report = lab.rewrite(&sites);
        assert_eq!(report.rewritten(), sites.len(), "{}", report.summary());
        let mut rng = Rng(0xfeed_face_cafe_beef);
        for (i, (code, clzero, _, counter, limit)) in loops.iter().enumerate() {
            for round in 0..48 {
                let mut lines = Lines::random(&mut rng);
                let before = lines.clone();
                let mut from = lab.random_state(&mut rng);
                // From a few lines to none at all (a limit at or below the
                // start still does one pass), limits off a line's edge too.
                let start = 64 * (2 + rng.below(8) as usize);
                let passes = [0usize, 1, 2, 5, 17][round % 5];
                let ragged = if round % 3 == 0 { rng.below(64) } else { 0 };
                from.gpr[*counter] = lines.at(start);
                from.gpr[*limit] = lines.at(start + 64 * passes).wrapping_sub(ragged);
                // The emulation is given the thread as it faults at the
                // clzero: for the second loop, after the two instructions
                // before it.
                let mut faulting = from;
                if *clzero != 0 {
                    faulting.gpr[RAX] = from.gpr[*counter];
                    faulting.gpr[*counter] += 0x40;
                }
                let (want, next) = emulated(code, *clzero, &faulting);
                assert_eq!(next, code.len(), "the emulation ends the whole loop");
                let want_lines = lines.clone();
                assert!(*want_lines != *before);
                lines.0 = before.0;
                let got = lab.run(at[i], &from);
                assert_same(&got, &want, &format!("loop {i} round {round}"));
                assert!(*lines == *want_lines, "loop {i} round {round}: memory");
            }
        }
    }

    #[test]
    fn hints_rewritten_in_place_change_nothing() {
        // monitorx, mwaitx, clwb [rcx], clwb [r9], clwb [rdx + 10h].
        let codes: [(Kind, &[u8]); 5] = [
            (Kind::Monitorx, &[0x0f, 0x01, 0xfa]),
            (Kind::Mwaitx, &[0x0f, 0x01, 0xfb]),
            (Kind::Clwb, &[0x66, 0x0f, 0xae, 0x31]),
            (Kind::Clwb, &[0x66, 0x41, 0x0f, 0xae, 0x31]),
            (Kind::Clwb, &[0x66, 0x0f, 0xae, 0x72, 0x10]),
        ];
        let mut lab = Lab::new();
        let at: Vec<u32> = codes.iter().map(|(_, code)| lab.place(code)).collect();
        lab.seal();
        let sites: Vec<Site<'_>> = codes
            .iter()
            .zip(&at)
            .map(|((kind, code), &rva)| Site { rva, ..row(*kind, code.len(), code) })
            .collect();
        let report = lab.rewrite(&sites);
        assert_eq!(report.rewritten(), sites.len(), "{}", report.summary());
        assert!(report.arena.is_none(), "no stub for a hint");
        let mut rng = Rng(0x1357_9bdf_0246_8ace);
        for &rva in &at {
            for _ in 0..16 {
                // Registers that point nowhere: a nop with a memory
                // operand reads nothing.
                let mut from = lab.random_state(&mut rng);
                from.gpr[RCX] |= 0xdead_0000_0000_0000;
                from.gpr[RDX] |= 0xdead_0000_0000_0000;
                from.gpr[9] |= 0xdead_0000_0000_0000;
                let got = lab.run(rva, &from);
                assert_same(&got, &from, "a hint");
            }
        }
    }

    #[test]
    fn a_stack_walk_leaves_a_stub_at_its_site() {
        let info = StubInfo {
            code: 0x7000_0020,
            end: 0x7000_0070,
            site: 0x1_44e8_aa70,
            rva: 0x4e8_aa70,
            kind: Kind::Clzero,
            frame: 24,
            restored: 0x7000_0060,
        };
        let rsp = 0x10_0000u64;
        // Before its first instruction and after the one that raises the
        // stack pointer again, the stack is the title's; between, 24 lower.
        for (rip, lowered) in [
            (0x7000_0020u64, false),
            (0x7000_0025, true),
            (0x7000_005f, true),
            (0x7000_0060, false),
            (0x7000_006f, false),
        ] {
            let want = (0x1_44e8_aa70, if lowered { rsp + 24 } else { rsp });
            assert_eq!(info.leave(rip, rsp), want, "{rip:#x}");
        }
        // The stubs a test makes are not the process's.
        assert!(stub_at(0x7000_0020).is_none());
        // What the generator says of its own code: the frame is given
        // back by the instruction that ends at `restored`.
        let code = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40];
        let built = stub(&row(Kind::Clzero, 3, &code), 0x1_3000_0000, 0x1_4000_1007).unwrap();
        assert_eq!(built.frame, 24);
        assert_eq!(built.code[..5], [0x48, 0x8d, 0x64, 0x24, 0xe8]);
        assert_eq!(built.code[built.restored - 5..built.restored], [0x48, 0x8d, 0x64, 0x24, 0x18]);
        assert_eq!(built.code[built.restored..built.restored + 4], code[3..]);
        assert_eq!(built.code[built.restored + 4], 0xe9);
        let ins = insertq(2, 3, 5, 7);
        let built = stub(&row(Kind::Insertq, 6, &ins), 0x1_3000_0000, 0x1_4000_1006).unwrap();
        assert_eq!((built.frame, built.restored + 5), (16, built.code.len()));
        assert_eq!(built.constants, [[0x1f, 0], [!(0x1f << 7), u64::MAX]]);
    }

    #[test]
    fn this_cpu_is_asked_per_instruction() {
        // Whatever the answers, they are CPUID's: the pair of hints comes
        // together, and an Intel CPU (this PC's) has neither SSE4a nor
        // clzero nor the pair.
        assert_eq!(native(Kind::Monitorx), native(Kind::Mwaitx));
        // With `--nocapture`: what a rewrite on this machine would leave alone.
        let has: Vec<String> = Kind::ALL.iter().map(|&k| format!("{} {}", k.name(), native(k))).collect();
        println!("this CPU has: {}", has.join(", "));
        let vendor = std::arch::x86_64::__cpuid(0);
        let name: Vec<u8> = [vendor.ebx, vendor.edx, vendor.ecx].iter().flat_map(|r| r.to_le_bytes()).collect();
        if name == b"GenuineIntel" {
            assert!(!native(Kind::Insertq) && !native(Kind::Clzero) && !native(Kind::Mwaitx));
        }
    }
}
