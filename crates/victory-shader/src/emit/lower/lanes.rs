//! Lane semantics of a console wave64 compute program run as two host waves of 32: what the
//! lane tests (`emit/lane_tests.rs`, every lane against a CPU model of the console wave)
//! found wrong in the 32-lane lowering, each behind its own switch.
//!
//! Ported from DiogoPCastelos/victory-fork (GPL-3.0-or-later), commits 0698344c3 ("A CPU
//! model of a 64-lane wave and lane tests on the card; three translator fixes they found")
//! and 999e1e5e0 by DiogoPCastelos.

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
