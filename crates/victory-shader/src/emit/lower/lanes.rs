//! Lane semantics of a console wave64 compute program run as two host waves of 32: what the
//! lane tests (`emit/lane_tests.rs`, every lane against a CPU model of the console wave)
//! found wrong in the 32-lane lowering, each behind its own switch.
//!
//! Ported from DiogoPCastelos/victory-fork (GPL-3.0-or-later), commits 0698344c3 ("A CPU
//! model of a 64-lane wave and lane tests on the card; three translator fixes they found")
//! and 999e1e5e0 by DiogoPCastelos; the thread ticket ([`thread_ticket`]) from
//! fxceless/victory-fork pull request #1261 by Garu667 (GPL-3.0).

use super::*;

/// Whether `ds_bpermute_b32` in a console wave64 program run as two host waves reads the lane
/// its address names in either of them ([`Lower::permute_across_halves`]), through 64 LDS
/// dwords a console wave between two group barriers. `VICTORY_PERMUTE_HALVES=0`: refused
/// ("a lane permute across two host waves"), as before. Part of the translation: the shader
/// cache keys on it (`victory-gpu`'s does).
pub fn permute_halves() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_PERMUTE_HALVES.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PERMUTE_HALVES").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`permute_halves`] for this thread.
    pub(in crate::emit) static TEST_PERMUTE_HALVES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether, in a group that is not a multiple of 32 threads, only the full host waves take
/// their run of 32 console threads from the counter of [`Lower::wave_base`] and the short one
/// takes the last run (`group & !31`), the only one it can be. `VICTORY_SHORT_WAVE_LAST=0`:
/// every host wave takes the next run in the order it arrives, as before, where a short wave
/// that arrived first took console lanes 0..31 and left lanes with no thread (a group of 48:
/// the wave of 16 threads took lanes 0..15, the full one 32..63, and lanes 16..31 read 0).
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s does).
pub fn short_wave_last() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_SHORT_WAVE_LAST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHORT_WAVE_LAST").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`short_wave_last`] for this thread.
    pub(in crate::emit) static TEST_SHORT_WAVE_LAST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether `v_readlane_b32` and `v_writelane_b32` take a computed lane number modulo the
/// console wave's size, as the instruction does (its low six bits in a 64-lane wave: lane 64
/// is lane 0, -1 is lane 63; [`Lower::lane_mod`]). `VICTORY_LANE_MODULO=0`: the number is
/// compared as it stands, as before, where one past the wave named no lane (a read gave 0, a
/// write went nowhere). Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn lane_modulo() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_LANE_MODULO.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LANE_MODULO").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`lane_modulo`] for this thread.
    pub(in crate::emit) static TEST_LANE_MODULO: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a program with console lanes in a group of several host waves of 32 lanes, off
/// the path of a console wave64 program in host waves of 32 (which has its counter already),
/// takes its console threads from an LDS counter too ([`ticket`], through
/// [`Lower::wave_base`]): every thread zeroes the dword, the group meets, and each host wave
/// takes the next run of 32. `VICTORY_THREAD_TICKET=0`: the flat id of the wave's first lane
/// rounded down to the wave size, plus the lane, as before. That is each thread's own number
/// only while every host wave holds a wave-aligned run of flat ids, and a 32-lane card fills a
/// two-dimensional group from tiles of 16 by 2: in a group of 32 by 2 both host waves round
/// down to the same base, and half the threads take the numbers (and the places they store
/// to) of the other half. Part of the translation: the shader cache keys on it
/// (`victory-gpu`'s does).
pub fn thread_ticket() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_THREAD_TICKET.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THREAD_TICKET").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`thread_ticket`] for this thread.
    pub(in crate::emit) static TEST_THREAD_TICKET: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The LDS dword of the counter a program with console lanes takes its console threads from
/// ([`thread_ticket`]), reserved past `lds_dwords`: for a group of `group` threads in several
/// host waves of `wave` = 32 lanes that is not on the halves path (`halves`: it has its
/// counter). None otherwise: one host wave, waves of 64 (not taken: the counter counts in
/// 32s), the switch off, or no room left in the program's LDS ([`MAX_LDS_DWORDS`]).
pub(super) fn ticket(lds_dwords: &mut u32, halves: bool, console_lanes: bool, group: u32, wave: u32) -> Option<u32> {
    if halves || !console_lanes || wave != 32 || group <= wave || !thread_ticket() || *lds_dwords >= MAX_LDS_DWORDS {
        return None;
    }
    let slot = *lds_dwords;
    *lds_dwords += 1;
    Some(slot)
}

/// The first of the LDS dwords a group of `group` threads keeps for `ds_bpermute_b32` across
/// its host waves (64 a console wave, [`Lower::permute_across_halves`]), reserved past
/// `lds_dwords`. None where there is no such instruction, where the group is one host wave,
/// with each host wave a wave of its own (`local`: nothing goes through it), with the switch
/// off, or where the program's LDS leaves no room for it ([`MAX_LDS_DWORDS`]).
pub(super) fn permute_area(lds_dwords: &mut u32, group: u32, shuffles: bool, local: bool) -> Option<u32> {
    let dwords = 64 * group.div_ceil(64);
    if !shuffles || group <= 32 || local || !permute_halves() || *lds_dwords + dwords > MAX_LDS_DWORDS {
        return None;
    }
    let area = *lds_dwords;
    *lds_dwords += dwords;
    Some(area)
}

impl<'a> Lower<'a> {
    /// A lane number as `v_readlane_b32` and `v_writelane_b32` take it: modulo the console
    /// wave's size ([`lane_modulo`]). Left as it is: a constant (the console compiler's, in
    /// range; the lowering tells constants from computed lanes), and a computed lane of a
    /// pixel or vertex program, which is a lane of the host's wave (from a mask or `v_mbcnt`
    /// over it), whatever its size.
    pub(super) fn lane_mod(&mut self, lane: V) -> V {
        if matches!(lane, V::Const(_)) || self.mask_data == MaskData::Host || !lane_modulo() {
            return lane;
        }
        let last = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.cur, Bin::And, lane, last)
    }

    /// For [`Self::wave_base`], in the entry block: whether this host wave is full (every one
    /// of its 32 lanes holds a thread), asked only in a group that is not a multiple of 32
    /// threads ([`short_wave_last`]; `None`: every wave is full, or the rule is off), and
    /// `first` (this wave's first lane, which counts the wave off the counter) for full waves
    /// only.
    pub(super) fn full_waves_count(&mut self, first: V) -> (V, Option<V>) {
        let group = self.s.threads.map_or(0, |t| t.iter().product::<u32>());
        if group.is_multiple_of(32) || !short_wave_last() {
            return (first, None);
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let here = self.s.m.bool(true);
        let ballot = self.s.op(e, "waveActiveBallot", Attr::NoUnwind, four, 116, &[here]);
        let lo = self.s.m.extract(e, ballot, 0);
        let all = self.s.m.i32(u32::MAX);
        let full = self.s.m.cmp(e, pred::EQ, lo, all);
        (self.s.m.bin(e, Bin::And, first, full), Some(full))
    }

    /// For [`Self::wave_base`]: `base` (what the counter gave) for a full host wave, the
    /// group's last run of 32 console threads for the short one (`full` is
    /// [`Self::full_waves_count`]'s).
    pub(super) fn short_wave_base(&mut self, full: Option<V>, base: V) -> V {
        let Some(full) = full else { return base };
        let group = self.s.threads.map_or(0, |t| t.iter().product::<u32>());
        let last = self.s.m.i32(group & !31);
        self.s.m.select(self.s.entry, full, base, last)
    }

    /// `ds_bpermute_b32` in a console wave of 64 run as two host waves: the lane an address
    /// names may be in the other one. Every lane writes its data (0 where EXEC leaves it out,
    /// as the console reads a lane outside EXEC) to its console wave's 64 dwords, the group
    /// meets, every lane reads the dword of the lane its address names, and the group meets
    /// again before the dwords can be written anew. The meetings are group barriers, so every
    /// thread of the group must be here: elsewhere it is refused, and the block named for
    /// [`super::super::widen`].
    pub(super) fn permute_across_halves(
        &mut self,
        inst: &Inst,
        addr: Option<&Operand>,
        data: Option<&Operand>,
        offset: u32,
        dst_reg: Option<u8>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let (Some(area), false) = (self.permute_area, self.local) else {
            return Err("a lane permute across two host waves");
        };
        if !self.all_lanes(self.cur_ir) {
            need(self.cur_ir);
            return Err("a lane permute across two host waves where not every lane runs");
        }
        let c = self.cur;
        let at = self.vgprs(inst, addr.ok_or("a permute address")?, 1)?[0];
        let x = self.vgprs(inst, data.ok_or("a permute source")?, 1)?[0];
        let x = match guard {
            Some(g) => {
                let z = self.s.m.i32(0);
                self.s.m.select(c, g, x, z)
            }
            None => x,
        };
        let o = self.s.m.i32(offset);
        let at = self.s.m.bin(c, Bin::Add, at, o);
        let two = self.s.m.i32(2);
        let src = self.s.m.bin(c, Bin::LShr, at, two);
        let k63 = self.s.m.i32(63);
        let src = self.s.m.bin(c, Bin::And, src, k63);
        // thread t of the group writes dword area + t; its console wave's lane `src` is at
        // area + (t & !63) + src
        let thread = self.console_thread();
        let lds = self.s.lds();
        let base = self.s.m.i32(area);
        let mine = self.s.m.bin(c, Bin::Add, base, thread);
        let p = self.s.m.element(c, lds, mine);
        self.s.m.store(c, p, x);
        self.group_sync(c);
        let not63 = self.s.m.i32(!63);
        let wave = self.s.m.bin(c, Bin::And, thread, not63);
        let theirs_thread = self.s.m.bin(c, Bin::Or, wave, src);
        let theirs = self.s.m.bin(c, Bin::Add, base, theirs_thread);
        let p = self.s.m.element(c, lds, theirs);
        let r = self.s.m.load(c, p);
        self.group_sync(c);
        // a lane past the end of the group is not there on the console: 0
        let group = self.s.threads.map_or(64, |t| t.iter().product::<u32>());
        let r = if group.is_multiple_of(64) {
            r
        } else {
            let n = self.s.m.i32(group);
            let there = self.s.m.cmp(c, pred::IULT, theirs_thread, n);
            let z = self.s.m.i32(0);
            self.s.m.select(c, there, r, z)
        };
        let reg = dst_reg.ok_or("a permute result")?;
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }
}
