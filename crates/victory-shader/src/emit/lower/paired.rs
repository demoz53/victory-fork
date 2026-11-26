//! The paired form of console wave64 compute programs on hosts of 32-lane waves
//! (`VICTORY_WAVE_PAIRS`, [`super::super::pairs`]): each console wave runs as one host wave of
//! 32 lanes, and host lane n runs console lanes n and n + 32.
//!
//! Ported from fxceless/victory-fork PR #854 (commit 26600c6, by PiloteCode), read beside its
//! adapted ports in DexNom/VIctory PR #64 (commits 40c2f29 and a65db62, by Castyan) and in
//! Corgansm/victory-corgi (`VICTORY_WAVE_PAIRS`). Only the paired form is taken: what the
//! lowering keeps per console lane, the instruction loop for two lanes, and the cross-lane
//! operations over both halves. Kept in a module of its own with narrow hooks in `lower.rs`.
//! Added here for this tree: `v_writelane_b32` under a guard (the wave's "some lane is in
//! EXEC" over both halves), the 64-bit float wave sum in this tree's shape, and the refusals
//! of what the form does not take (a scatter `ds_permute_b32`, spilled user data).

use super::*;

thread_local! {
    /// See [`with_paired_lanes`].
    static PAIRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A branch the wave does not take together, on which the two console lanes of a host thread
/// could go different ways.
pub(super) const APART: Why = "a branch the two lanes of a thread may not take together";

/// Runs `f` with console wave64 compute programs on 32-lane hosts lowered in the paired form:
/// each console wave is one host wave of 32 lanes, and host lane n runs console lanes n and
/// n + 32. The group has half the console's threads; each thread runs what a lane does twice,
/// once for each of its lanes, in the order of the console's instructions, so the wave's lanes
/// stay together as on the console: what one writes to LDS, the next instruction of another
/// reads (a fence between the accesses keeps the host compiler from reordering them; no thread
/// waits for another). Cross-lane operations reach any of the 64 lanes through the host wave's
/// own operations on either half (lane n + 32 is lane n's upper half), and a lane mask is the
/// two ballots. The caller makes every per-lane branch one the wave takes together, as the
/// console does (`structure::structure_wide`): the wave goes where some lane has to, and EXEC
/// stays each lane's guard. So control flow is the same in every thread of a host wave, and
/// the program's barriers are reached as on the console.
// shortcut: D3D12 orders no groupshared accesses of two threads without a group barrier; the
// host is taken to run a wave's threads in step where they all go the same way (as for the
// console's 32-lane programs that share LDS within a wave), the fences only keeping its
// compiler from moving one thread's accesses past each other.
pub(in crate::emit) fn with_paired_lanes<R>(f: impl FnOnce() -> R) -> R {
    let was = PAIRED.replace(true);
    let r = f();
    PAIRED.set(was);
    r
}

/// Whether `native` is lowered in the paired form ([`with_paired_lanes`]).
pub(super) fn wanted(native: &NativeProgram) -> bool {
    PAIRED.get() && native.wave64 && host_uses_wave32()
}

/// The paired form's thread group: a host wave of 32 for each console wave of the group, its
/// threads in a row (the caller has checked that the console's waves are full). Returns the
/// LDS dword of `Lower::wave_base`'s counter for a group of several console waves, and whether
/// that dword is the program's own first, borrowed where the program takes all a host group
/// has.
pub(super) fn group(s: &mut Shader, threads: [u32; 3], lds_dwords: u32) -> (Option<u32>, bool) {
    let waves = threads.iter().product::<u32>().div_ceil(64);
    s.threads = Some([32 * waves, 1, 1]);
    s.lds_dwords = lds_dwords;
    s.wave_size = Some(32);
    if waves == 1 {
        return (None, false);
    }
    if s.lds_dwords >= MAX_LDS_DWORDS {
        (Some(0), true)
    } else {
        s.lds_dwords += 1;
        (Some(s.lds_dwords - 1), false)
    }
}

/// What the lowering keeps per console lane, for the paired form ([`with_paired_lanes`]): each
/// host thread runs two lanes of one console wave, and what a lane runs is lowered once for
/// each with this state exchanged between them ([`Lower::swap_lanes`]): the values (scalar ones
/// too, the same in both), the lane booleans and their joins, the handles of descriptors a lane
/// may have its own of, the phis still to fill.
#[derive(Default)]
pub(super) struct LaneState {
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    lane_fail: HashMap<Value, Why>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    strides: HashMap<(usize, Site, u8), V>,
    slot_words: HashMap<(usize, Site, u8), V>,
    clane: Option<V>,
}

/// The paired form: scalar values each console lane has its own of in the lowering, a waterfall
/// loop's first-lane reads (the lane's own dword, `resource::Waterfall`) and what scalar code
/// and joins make of them.
pub(super) fn lane_scalars(p: &Program, res: &Resources) -> HashSet<Value> {
    let mut own: HashSet<Value> = res.waterfall.reads.iter().copied().collect();
    if own.is_empty() {
        return own;
    }
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let before = own.len();
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                if !matches!(phi.reg, Reg::V(_)) && phi.args.iter().any(|a| own.contains(a)) {
                    own.insert(phi.value);
                }
            }
            for inst in &block.insts {
                if matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. })
                    && inst.reads.iter().any(|r| own.contains(&r.1))
                {
                    own.extend(inst.writes.iter().map(|w| w.1));
                }
            }
        }
        if own.len() == before {
            return own;
        }
    }
}

impl<'a> Lower<'a> {
    /// The paired form: makes the other console lane of this thread the one being lowered.
    pub(super) fn swap_lanes(&mut self) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        std::mem::swap(&mut self.vals, &mut t.vals);
        std::mem::swap(&mut self.floats, &mut t.floats);
        std::mem::swap(&mut self.bools, &mut t.bools);
        std::mem::swap(&mut self.lane_bits, &mut t.lane_bits);
        std::mem::swap(&mut self.lane_fail, &mut t.lane_fail);
        std::mem::swap(&mut self.mask_phis, &mut t.mask_phis);
        std::mem::swap(&mut self.pending_masks, &mut t.pending_masks);
        std::mem::swap(&mut self.data_phis, &mut t.data_phis);
        std::mem::swap(&mut self.handles, &mut t.handles);
        std::mem::swap(&mut self.strides, &mut t.strides);
        std::mem::swap(&mut self.slot_words, &mut t.slot_words);
        std::mem::swap(&mut self.clane, &mut t.clane);
        self.upper = !self.upper;
    }

    /// The paired form: `f` for each console lane of this thread, the lower one first; the lane
    /// being lowered stays the same.
    pub(super) fn both<T>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, Why>) -> Result<[T; 2], Why> {
        let was = self.upper;
        if was {
            self.swap_lanes();
        }
        let lo = f(self);
        self.swap_lanes();
        let hi = f(self);
        if !was {
            self.swap_lanes();
        }
        Ok([lo?, hi?])
    }

    /// The paired form: `lo` as the lower lane's value of `v`, `hi` as the upper lane's.
    fn put_lanes(&mut self, v: Value, lo: Val, hi: Val) {
        let (mine, other) = if self.upper { (hi, lo) } else { (lo, hi) };
        self.vals.insert(v, mine);
        if let Some(t) = self.twin.as_deref_mut() {
            t.vals.insert(v, other);
        }
    }

    /// The paired form: the upper lanes' inputs (their own thread ids).
    pub(super) fn paired_inputs(&mut self) -> Result<(), Stop> {
        self.swap_lanes();
        let r = self.inputs();
        self.swap_lanes();
        r
    }

    /// The paired form: the upper lane's joins at the start of IR block `b`, as `Lower::run`
    /// makes the lower lane's (filled once every block is lowered).
    pub(super) fn paired_block_phis(&mut self, b: usize) {
        self.swap_lanes();
        let p = self.x.p;
        for (i, phi) in p.blocks[b].phis.iter().enumerate() {
            let val = match phi.reg {
                Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                Reg::Scc => {
                    let v = self.s.m.phi(self.cur, self.s.t.i1);
                    self.data_phis.push((v, b, i));
                    Val::Bit(v)
                }
                _ => {
                    let v = self.s.m.phi(self.cur, self.s.t.i32);
                    self.data_phis.push((v, b, i));
                    Val::I32(v)
                }
            };
            self.vals.insert(phi.value, val);
        }
        self.swap_lanes();
    }

    /// The paired form: the other lane's joins too, after the lower lane's
    /// (`Lower::fill_phis`); filling one lane's can ask for lane masks joined in either lane,
    /// until neither has any left.
    pub(super) fn paired_fill_phis(&mut self) -> Result<(), Stop> {
        loop {
            self.swap_lanes();
            let r = self.fill_phis();
            self.swap_lanes();
            r?;
            let left = self.twin.as_deref().is_some_and(|t| !t.pending_masks.is_empty());
            if self.pending_masks.is_empty() && !left {
                return Ok(());
            }
            self.fill_phis()?;
        }
    }

    /// `Lower::section_inst` in the paired form: a value over the wave's lanes takes both
    /// lanes of every thread ([`Self::paired_shape`]), a lane's own value each lane's.
    pub(super) fn paired_section_inst(&mut self, b: usize, i: usize, start: usize) {
        use crate::emit::wave::Shape;
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let shape = match self.section_out.contains(&w) {
                true => self.sections[&b][&start].shapes.get(&w).copied(),
                false => None,
            };
            let made = match shape {
                None => Err(UNKNOWN),
                Some(sh @ (Shape::Active(..) | Shape::Prefix(..) | Shape::ActiveF64(..))) => self.paired_shape(sh),
                Some(sh) => self.both(|l| l.shape(sh, false)),
            };
            match made {
                Ok([lo, hi]) => self.put_lanes(w, Val::I32(lo), Val::I32(hi)),
                Err(why) => self.put_lanes(w, Val::Missing(why), Val::Missing(why)),
            }
        }
    }

    /// `op` of `arg` (`float`: a float, else bits) over this host wave's active lanes; with
    /// `prefix` the sum over the lanes below each lane (`Some(true)`: and the lane's own).
    fn host_wave_op(
        &mut self,
        op: crate::emit::wave::Op,
        arg: V,
        float: bool,
        prefix: Option<bool>,
    ) -> Result<V, Why> {
        use crate::emit::wave::Op;
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        Ok(match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
                let r = self.s.op(c, &format!("wavePrefixOp.{sfx}"), Attr::NoUnwind, ty, 121, &[arg, k0, k1]);
                if inclusive { self.s.m.bin(c, Bin::Add, r, arg) } else { r }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s.op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(c, &format!("waveActiveOp.{sfx}"), Attr::NoUnwind, ty, 119, &[arg, k, u])
            }
        })
    }

    /// The results of a reduction by `op` (`float`: floats, else bits) over the two halves of
    /// a console wave, joined into the wave's.
    fn join_halves(&mut self, op: crate::emit::wave::Op, lo: V, hi: V, float: bool) -> V {
        use crate::emit::wave::Op;
        let c = self.cur;
        let ty = if float { self.s.t.f32 } else { self.s.t.i32 };
        let sfx = if float { "f32" } else { "i32" };
        match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            _ => {
                // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                let code = match op {
                    Op::MaxF => 35,
                    Op::MinF => 36,
                    Op::MinU => 40,
                    Op::MaxU => 39,
                    Op::MinI => 38,
                    _ => 37,
                };
                self.s.op(c, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[lo, hi])
            }
        }
    }

    /// `Lower::shape` of a reduction or prefix in the paired form, over the console wave's 64
    /// lanes: for the lower and the upper lane of this thread. Each half is a host wave
    /// operation over the lower or the upper lanes, and the halves join as two host waves' do
    /// in `shape`: a reduction of the two results, a prefix of the upper lanes after the lower
    /// lanes' total.
    fn paired_shape(&mut self, shape: crate::emit::wave::Shape) -> Result<[V; 2], Why> {
        use crate::emit::wave::{Op, Shape};
        if let Shape::ActiveF64(lo, hi, mask, high) = shape {
            return self.paired_sum64(lo, hi, mask, high);
        }
        let (op, src, mask, ident, prefix) = match shape {
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Prefix(op, src, mask, ident, inclusive) => (op, src, mask, ident, Some(inclusive)),
            _ => return Err("a whole-wave section value of another shape"),
        };
        let &m = self.x.st.masks.get(&mask).ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        let [lo, hi] = self.both(|l| {
            let on = l.boolean(m, c)?;
            let x = l.i32_of(src)?;
            let id = l.s.m.i32(ident);
            Ok(l.s.m.select(c, on, x, id))
        })?;
        self.s.flags |= flags::WAVE_OPS;
        if op == Op::AddH {
            if prefix.is_some() {
                return Err("a wave prefix of an operation other than a sum");
            }
            let (lo, hi) = (self.half_tree(lo, 5), self.half_tree(hi, 5));
            let r = self.join_half_sums(lo, hi);
            return Ok([r, r]);
        }
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (lo, hi) = if float { (self.as_float(lo), self.as_float(hi)) } else { (lo, hi) };
        let r = match prefix {
            Some(inclusive) => {
                let mine = self.host_wave_op(op, lo, float, Some(inclusive))?;
                let theirs = self.host_wave_op(op, hi, float, Some(inclusive))?;
                let below = self.host_wave_op(op, lo, float, None)?;
                [mine, self.s.m.bin(c, Bin::Add, theirs, below)]
            }
            None => {
                let lo = self.host_wave_op(op, lo, float, None)?;
                let hi = self.host_wave_op(op, hi, float, None)?;
                let r = self.join_halves(op, lo, hi, float);
                [r, r]
            }
        };
        Ok(if float { r.map(|x| self.as_int(x)) } else { r })
    }

    /// The 64-bit float `WaveActiveSum` (`Lower::active_f64_sum`) in the paired form: the
    /// lower lanes' sum, then the upper lanes' added, as two host waves add theirs; dword
    /// `high` of it, for both lanes of this thread.
    fn paired_sum64(&mut self, lo: Value, hi: Value, mask: Value, high: bool) -> Result<[V; 2], Why> {
        let &m = self.x.st.masks.get(&mask).ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        let f64t = self.s.m.double();
        let i32t = self.s.t.i32;
        let split_ty = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let terms = self.both(|l| {
            let on = l.boolean(m, c)?;
            let (dl, dh) = (l.i32_of(lo)?, l.i32_of(hi)?);
            let z = l.s.m.i32(0);
            let (dl, dh) = (l.s.m.select(c, on, dl, z), l.s.m.select(c, on, dh, z));
            Ok(l.s.op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[dl, dh]))
        })?;
        self.s.flags |= flags::WAVE_OPS | flags::DOUBLES;
        let sums = terms.map(|x| {
            let (k0, k0b) = (self.s.m.i8(0), self.s.m.i8(0));
            self.s.op(c, "waveActiveOp.f64", Attr::NoUnwind, f64t, 119, &[x, k0, k0b])
        });
        let total = self.s.m.bin(c, Bin::Add, sums[0], sums[1]);
        let split = self.s.op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[total]);
        let x = self.s.m.extract(c, split, u32::from(high));
        Ok([x, x])
    }

    /// The first `steps` levels of `Lower::half_sum`'s tree over this host wave: its lanes'
    /// sum when they are all of its waves' lanes (5 steps for 32), in the low half; the high
    /// half lane 0's.
    fn half_tree(&mut self, x: V, steps: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let mut sum = x;
        for step in 0..steps {
            let bit = self.s.m.i32(1 << step);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let theirs = self.read_lane(sum, partner);
            sum = self.add_halves(sum, theirs);
        }
        let (high, low, zero) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF), self.s.m.i32(0));
        let first = self.read_lane(x, zero);
        let first = self.s.m.bin(c, Bin::And, first, high);
        let sum = self.s.m.bin(c, Bin::And, sum, low);
        self.s.m.bin(c, Bin::Or, first, sum)
    }

    /// The last addition of `Lower::half_sum`'s tree: the sums of a console wave's two halves
    /// (from [`Self::half_tree`]), with the high half of the lower one's (lane 0's).
    fn join_half_sums(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let total = self.add_halves(lo, hi);
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let total = self.s.m.bin(c, Bin::And, total, low);
        let first = self.s.m.bin(c, Bin::And, lo, high);
        self.s.m.bin(c, Bin::Or, first, total)
    }

    /// `Lower::mask_dword`'s ballot in the paired form, taken in DXIL block `at`: the lower
    /// lanes' bits (lanes 0-31) and the upper lanes' (lanes 32-63), each a host wave ballot.
    /// The host wave runs every block with all its lanes, so a mask that holds lanes outside
    /// EXEC is whole too.
    pub(super) fn paired_ballot(&mut self, lo: Value, m: Bool, at: usize) -> Result<(V, V), Why> {
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bits = self.both(|l| l.boolean(m, at));
        self.balloting.remove(&lo);
        let [lower, upper] = bits?;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        self.s.flags |= flags::WAVE_OPS;
        let r = self.s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[lower]);
        let x = self.s.m.extract(at, r, 0);
        let r = self.s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[upper]);
        let y = self.s.m.extract(at, r, 0);
        Ok((x, y))
    }

    /// `Lower::console_thread` in the paired form, of the lane being lowered: its console
    /// wave's first thread (`Lower::wave_base`, in a group of several console waves), plus the
    /// host lane `lane`, plus 32 for the upper lane.
    pub(super) fn paired_thread(&mut self, lane: V) -> V {
        let e = self.s.entry;
        let half = self.s.m.i32(if self.upper { 32 } else { 0 });
        let t = self.s.m.bin(e, Bin::Add, lane, half);
        match self.wave_base() {
            Some(base) => self.s.m.bin(e, Bin::Add, base, t),
            None => t,
        }
    }

    /// `Lower::scc_some` in the paired form: the host wave asks both lanes of each thread.
    pub(super) fn paired_scc_some(&mut self, scc: Value, m: Bool) -> Result<V, Why> {
        let at = match self.def_block.get(&scc).copied() {
            None => self.body_entry,
            Some(b) if b == self.cur_ir => self.cur,
            Some(b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        let [lo, hi] = self.both(|l| l.boolean(m, at))?;
        let either = self.s.m.bin(at, Bin::Or, lo, hi);
        self.s.flags |= flags::WAVE_OPS;
        let i1t = self.s.t.i1;
        Ok(self.s.op(at, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[either]))
    }

    /// `Lower::wave_wide` in the paired form: the host wave is the console's, each thread with
    /// two of its lanes; every lane is here (control flow is the wave's), and a lane outside
    /// EXEC says what it says on the console (no set bit for "some lane", no objection to
    /// "every lane").
    pub(super) fn paired_wave_wide(&mut self, when: Bool, any: bool) -> Result<V, Why> {
        let c = self.cur;
        let [lo, hi] = self.both(|l| l.boolean(when, c))?;
        let pair = self.s.m.bin(c, if any { Bin::Or } else { Bin::And }, lo, hi);
        self.s.flags |= flags::WAVE_OPS;
        let (name, code) = if any { ("waveAnyTrue", 113) } else { ("waveAllTrue", 114) };
        let i1t = self.s.t.i1;
        Ok(self.s.op(c, name, Attr::NoUnwind, i1t, code, &[pair]))
    }

    /// The paired form: whether the branch on `when`, which the wave does not take together,
    /// is one the two lanes of a thread could take different ways ([`APART`]): anything but
    /// scalar conditions that are the same in both.
    pub(super) fn lanes_apart(&self, when: Bool) -> bool {
        !self
            .x
            .st
            .bools
            .support(when)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(v) if !self.lane_scalars.contains(v)))
    }

    /// Instruction `i` of IR block `b` in the paired form ([`with_paired_lanes`]), for both
    /// console lanes of the thread: what a lane does is lowered for the lower lane, then for
    /// the upper one, which keeps the console's order of the wave's instructions; scalar
    /// control (the title's barrier) and scalar memory writes once; a read of other lanes once
    /// with what both lanes hold ([`Self::paired_cross`]). An LDS access that follows another
    /// gets a fence first unless both read: what one lane wrote (or still read) must not move
    /// past another lane's access the console made after it.
    pub(super) fn paired_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let name = inst.name();
        let once = matches!(inst.fields, Fields::Sopp { .. })
            || matches!(inst.fields, Fields::Smem { .. })
                && (name.contains("store") || name.contains("atomic"));
        if once {
            self.inst(b, i)?;
            self.mirror(inst);
            return Ok(());
        }
        let cross = match name {
            "v_readlane_b32" | "v_writelane_b32" | "ds_bpermute_b32" | "ds_permute_b32" => true,
            // (not a waterfall loop's read, the lane's own dword: `Lower::valu`)
            "v_readfirstlane_b32" => {
                !inst.writes.first().is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1))
            }
            _ => false,
        };
        if cross {
            if let Err(why) = self.paired_cross(inst, b, i) {
                for &(_, w) in &inst.writes {
                    self.put_lanes(w, Val::Missing(why), Val::Missing(why));
                }
            }
            return Ok(());
        }
        if matches!(inst.fields, Fields::Ds { .. })
            && !name.starts_with("ds_swizzle")
            && !name.contains("permute")
        {
            let write = !name.starts_with("ds_read");
            if write || self.lds_last != Some(false) {
                let (mode, void) = (self.s.m.i32(8), self.s.t.void);
                self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
            }
            self.lds_last = Some(write);
        }
        self.inst(b, i)?;
        self.swap_lanes();
        let r = self.inst(b, i);
        self.swap_lanes();
        r
    }

    /// The paired form: what `inst` wrote in the lane being lowered, for the other lane too (a
    /// scalar result, the same in both).
    fn mirror(&mut self, inst: &Inst) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        for &(_, w) in &inst.writes {
            if let Some(&v) = self.vals.get(&w) {
                t.vals.insert(w, v);
            }
        }
    }

    /// The paired form: an instruction that reads or writes other lanes of the console wave,
    /// any of its 64: `v_readlane_b32` and `v_readfirstlane_b32` (scalar results, the same in
    /// both lanes of the thread), `v_writelane_b32` and `ds_bpermute_b32`. Lane n and lane
    /// n + 32 are the lower and the upper lane of host lane n.
    fn paired_cross(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.cross_lane(false)?;
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        // a lane's value from the lower or upper value of the host lane `at` (of 0..63)
        let pick = |l: &mut Self, lo: V, hi: V, at: V| {
            let host = l.s.m.bin(c, Bin::And, at, k31);
            let upper = l.s.m.bin(c, Bin::And, at, k32);
            let upper = l.s.m.cmp(c, pred::NE, upper, z);
            let (lo, hi) = (l.read_lane(lo, host), l.read_lane(hi, host));
            l.s.m.select(c, upper, hi, lo)
        };
        match inst.name() {
            // A scatter keeps a scratch dword per lane and two group barriers around its
            // writes (`Lower::ds`): not taken here.
            "ds_permute_b32" => return Err("a scatter permute in paired lanes"),
            "v_readlane_b32" => {
                // the lane: a scalar or a constant
                let lane = self.src(inst, 1)?;
                let [lo, hi] = self.both(|l| l.src(inst, 0))?;
                let r = pick(self, lo, hi, lane);
                self.write_i(inst, r, None)?;
            }
            "v_readfirstlane_b32" => {
                let [(lo, in_lo), (hi, in_hi)] = self.both(|l| {
                    let x = l.src(inst, 0)?;
                    let guard = l.guard(b, i)?;
                    Ok(l.first_lane(x, guard))
                })?;
                // the first lane in EXEC is a lower one if there is one; with none, lane 0
                let hi = self.s.m.select(c, in_hi, hi, lo);
                let r = self.s.m.select(c, in_lo, lo, hi);
                self.write_i(inst, r, None)?;
            }
            "v_writelane_b32" => {
                // The lane given (the low six bits of a wave64 lane select) takes the scalar;
                // the others keep their value. The console writes the lane whatever EXEC is,
                // once the wave gets here: under a guard, when some lane of the wave (of
                // either half) is in EXEC, as `Lower::valu` has it for one host wave.
                let Fields::Valu(v) = &inst.fields else {
                    return Err("a lane write");
                };
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                if v.clamp || v.omod != 0 {
                    return Err("integer output modifiers");
                }
                let k63 = self.s.m.i32(63);
                let parts = self.both(|l| {
                    let (a, lane) = (l.src(inst, 0)?, l.src(inst, 1)?);
                    let old = match inst.read(Reg::V(reg)) {
                        Some(o) => l.i32_of(o)?,
                        None => l.s.m.i32(0),
                    };
                    let guard = l.guard(b, i)?;
                    let me = l.console_lane();
                    let lane = l.s.m.bin(c, Bin::And, lane, k63);
                    let is = l.s.m.cmp(c, pred::EQ, me, lane);
                    Ok((a, old, guard, is))
                })?;
                let some = match (parts[0].2, parts[1].2) {
                    (Some(g0), Some(g1)) => {
                        let either = self.s.m.bin(c, Bin::Or, g0, g1);
                        let i1t = self.s.t.i1;
                        Some(self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[either]))
                    }
                    _ => None,
                };
                self.both(|l| {
                    let (a, old, _, is) = parts[usize::from(l.upper)];
                    let is = match some {
                        Some(s) => l.s.m.bin(c, Bin::And, is, s),
                        None => is,
                    };
                    let r = l.s.m.select(c, is, a, old);
                    l.write(inst, Reg::V(reg), r, None);
                    Ok(())
                })?;
                return Ok(());
            }
            _ => {
                // `ds_bpermute_b32`: each lane reads the lane its address names, which gives
                // its data, or 0 outside EXEC
                let Fields::Ds {
                    dst: Some(Operand::Vector { reg, .. }),
                    addr: Some(addr),
                    data,
                    offset0,
                    offset1,
                    ..
                } = &inst.fields
                else {
                    return Err("a permute's operands");
                };
                let data = data.first().ok_or("a permute source")?;
                let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
                let [(from_lo, x_lo, g_lo), (from_hi, x_hi, g_hi)] = self.both(|l| {
                    let at = l.vgprs(inst, addr, 1)?[0];
                    let x = l.vgprs(inst, data, 1)?[0];
                    let guard = l.guard(b, i)?;
                    let (o, two) = (l.s.m.i32(offset), l.s.m.i32(2));
                    let at = l.s.m.bin(c, Bin::Add, at, o);
                    let from = l.s.m.bin(c, Bin::LShr, at, two);
                    let x = match guard {
                        Some(g) => l.s.m.select(c, g, x, z),
                        None => x,
                    };
                    Ok((from, x, guard))
                })?;
                // each half reads within itself ([`super::lanes::permute_half_local`]): the lower
                // lane from the lower half, the upper from the upper
                let (from_lo, from_hi) = if super::lanes::permute_half_local() {
                    let (k31, k32) = (self.s.m.i32(31), self.s.m.i32(32));
                    let lo = self.s.m.bin(c, Bin::And, from_lo, k31);
                    let hi = self.s.m.bin(c, Bin::And, from_hi, k31);
                    (lo, self.s.m.bin(c, Bin::Or, hi, k32))
                } else {
                    (from_lo, from_hi)
                };
                let read = [pick(self, x_lo, x_hi, from_lo), pick(self, x_lo, x_hi, from_hi)];
                let guards = [g_lo, g_hi];
                let reg = *reg;
                self.both(|l| {
                    let n = usize::from(l.upper);
                    l.write(inst, Reg::V(reg), read[n], guards[n]);
                    Ok(())
                })?;
                return Ok(());
            }
        }
        self.mirror(inst);
        Ok(())
    }
}
