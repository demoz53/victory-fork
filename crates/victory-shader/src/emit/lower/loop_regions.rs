//! `VICTORY_AFTER_LOOP=1`: the natural loops a per-lane branch leaves count as part of its
//! region ([`super::lane_regions`]) in a console wave64 compute program on a host of 32-lane
//! waves.
//!
//! A lane that leaves a loop by its own branch leaves the others to go round again, so no
//! block of that loop is one every lane runs the same number of times: two host waves cannot
//! meet there (a group barrier, an exchange of halves), nor is LDS in order there. The
//! region search of [`super::lane_regions`] finds most of such a loop by itself (what the
//! branch reaches before a block every way out of it passes, and the branch's own block:
//! [`super::lane_loop_fix`]); it stops at a block of the loop that every way on from the
//! branch passes again, the header of an inner loop whose every way out leads back into it,
//! say, and leaves that block and what only it reaches as blocks every lane runs. With the
//! switch the whole loop is the branch's region.
//!
//! Ported from fxceless/victory-fork pull request #1249 by ByFxbian (GPL-3.0), "NVIDIA
//! compatibility: Wave64 control flow and lossless d16 buffer accesses", which adapts the
//! region rule of fork pull request #993 (Pilote Production) and takes its loop fixture from
//! darkworldhm/victory-fork-darkworldhm 08fdf0b47.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;

use crate::native::{HwStage, NativeProgram};

/// Whether the loops a per-lane branch leaves are part of its region (see the module).
/// Off unless `VICTORY_AFTER_LOOP=1`. Part of the translation: the shader cache keys on the
/// raw switch (`victory-gpu`'s does).
pub fn after_loop() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_AFTER_LOOP.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_AFTER_LOOP").is_some_and(|v| v == "1"))
}

thread_local! {
    /// Whether the program being emitted on this thread is one the rule is for ([`scope`]).
    static IN_SCOPE: Cell<bool> = const { Cell::new(false) };
    /// `VICTORY_WAVE_WHY`: the branches (block, blocks added) the program being emitted has
    /// been told of already ([`add_left`] is asked many times a program).
    static TOLD: RefCell<HashSet<(usize, usize)>> = RefCell::new(HashSet::new());
}

#[cfg(test)]
thread_local! {
    /// A test's setting of [`after_loop`] for this thread.
    pub(in crate::emit) static TEST_AFTER_LOOP: Cell<Option<bool>> = const { Cell::new(None) };
}

/// See [`scope`]: puts back what was there when it goes.
pub(in crate::emit) struct Scope(bool);

impl Drop for Scope {
    fn drop(&mut self) {
        IN_SCOPE.set(self.0);
    }
}

/// While the result lives, the lane regions asked for on this thread are `native`'s: with
/// its left loops when [`after_loop`] is on and it is a console wave64 compute program on a
/// host of 32-lane waves (the programs whose host waves must meet), as they were otherwise.
pub(in crate::emit) fn scope(native: &NativeProgram) -> Scope {
    let on = after_loop() && native.stage == HwStage::Cs && native.wave64 && super::host_uses_wave32();
    if on {
        TOLD.with_borrow_mut(HashSet::clear);
    }
    Scope(IN_SCOPE.replace(on))
}

/// The natural loops of the flow `succ`, for [`add_left`]; none where the rule is off or the
/// program is not one it is for ([`scope`]).
pub(super) fn loops(succ: &[Vec<usize>]) -> Vec<Vec<bool>> {
    if IN_SCOPE.get() { natural_loops(succ) } else { Vec::new() }
}

/// Adds to `region`, the region of the per-lane branch closing block `d`, every block of the
/// loops (of [`loops`]) that the branch leaves: the ones `d` is in with a successor outside.
/// `VICTORY_WAVE_WHY=1`: a line for each branch whose region grows by it (the form of the
/// program is not named: its retries number their blocks alike or nearly).
pub(super) fn add_left(d: usize, succ: &[Vec<usize>], loops: &[Vec<bool>], region: &mut [bool]) {
    let mut more = 0;
    for body in loops.iter().filter(|body| body[d] && succ[d].iter().any(|&s| !body[s])) {
        for (divergent, &member) in region.iter_mut().zip(body) {
            more += usize::from(member && !*divergent);
            *divergent |= member;
        }
    }
    if more > 0 && std::env::var_os("VICTORY_WAVE_WHY").is_some() && TOLD.with_borrow_mut(|t| t.insert((d, more))) {
        let program = crate::emit::per_program::current();
        eprintln!(
            "AFTER-LOOP #{} the lane branch closing block {d}: {more} more blocks of the loops it leaves are its region",
            program.as_deref().unwrap_or("?")
        );
    }
}

/// The natural loops of the blocks reachable from block 0, one per header (the loops of
/// several back edges to one header together): per loop, whether each block is in it. For
/// the reducible flow `ir::build` gives; a retreating edge that is no back edge gives a set
/// that is too large, which only keeps more blocks from being every lane's.
fn natural_loops(succ: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = succ.len();
    if n == 0 {
        return Vec::new();
    }
    let (mut seen, mut post, mut stack) = (vec![false; n], Vec::new(), vec![(0, 0)]);
    seen[0] = true;
    while let Some((b, edge)) = stack.last_mut() {
        let from = *b;
        match succ[from].get(*edge) {
            Some(&to) => {
                *edge += 1;
                if !std::mem::replace(&mut seen[to], true) {
                    stack.push((to, 0));
                }
            }
            None => {
                post.push(from);
                stack.pop();
            }
        }
    }
    let mut rpo = vec![usize::MAX; n];
    for (rank, b) in post.into_iter().rev().enumerate() {
        rpo[b] = rank;
    }
    let mut pred = vec![Vec::new(); n];
    for (b, edges) in succ.iter().enumerate().filter(|(b, _)| seen[*b]) {
        for &t in edges {
            pred[t].push(b);
        }
    }
    let mut loops = std::collections::BTreeMap::new();
    for (tail, edges) in succ.iter().enumerate().filter(|(b, _)| seen[*b]) {
        for &header in edges.iter().filter(|&&t| rpo[t] <= rpo[tail]) {
            let body = loops.entry(header).or_insert_with(|| {
                let mut body = vec![false; n];
                body[header] = true;
                body
            });
            let mut pending = vec![tail];
            while let Some(b) = pending.pop() {
                if !std::mem::replace(&mut body[b], true) {
                    pending.extend(&pred[b]);
                }
            }
        }
    }
    loops.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two loops, one inside the other, and a block after them: each loop's blocks by its
    /// header; an unreachable block is in none.
    #[test]
    fn natural_loops_are_found_by_header() {
        // 0 -> 1 -> 2 -> 3 -> (2 | 4), 4 -> (1 | 5); 6 -> 2 is not reachable
        let succ = vec![vec![1], vec![2], vec![3], vec![2, 4], vec![1, 5], vec![], vec![2]];
        let loops = natural_loops(&succ);
        let blocks = |body: &[bool]| -> Vec<usize> { (0..body.len()).filter(|&b| body[b]).collect() };
        assert_eq!(loops.iter().map(|l| blocks(l)).collect::<Vec<_>>(), [vec![1, 2, 3, 4], vec![2, 3]]);
        // the inner loop is left by block 3 and the outer one by block 4
        let mut region = vec![false; succ.len()];
        add_left(3, &succ, &loops, &mut region);
        assert_eq!(blocks(&region), [2, 3]);
        add_left(4, &succ, &loops, &mut region);
        assert_eq!(blocks(&region), [1, 2, 3, 4]);
        // (a block in no loop, or one that leaves none)
        let mut none = vec![false; succ.len()];
        add_left(0, &succ, &loops, &mut none);
        add_left(2, &succ, &loops, &mut none);
        assert!(none.iter().all(|b| !b));
        assert!(natural_loops(&[]).is_empty());
    }

    /// Off unless asked for, and only for a console wave64 compute program on a host of
    /// 32-lane waves.
    #[test]
    fn the_rule_is_for_wave64_compute_on_32_lane_hosts_when_asked_for() {
        let succ = vec![vec![1], vec![1, 2], vec![]];
        let native = |stage, wave64| NativeProgram {
            stage,
            wave64,
            vgprs: 1,
            layout: Vec::new(),
            used: Vec::new(),
            io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 0, threads: [64, 1, 1] }),
            code: Vec::new(),
        };
        let found = |on: bool, stage, wave64, host: u32| {
            let was = (TEST_AFTER_LOOP.replace(Some(on)), super::super::TEST_HOST_WAVE.replace(Some(host)));
            let n = {
                let _scope = scope(&native(stage, wave64));
                loops(&succ).len()
            };
            TEST_AFTER_LOOP.set(was.0);
            super::super::TEST_HOST_WAVE.set(was.1);
            // (and nothing is left behind)
            assert!(loops(&succ).is_empty());
            n
        };
        assert_eq!(found(true, HwStage::Cs, true, 32), 1);
        assert_eq!(found(false, HwStage::Cs, true, 32), 0, "off: as before");
        assert_eq!(found(true, HwStage::Cs, false, 32), 0, "a console wave32 program");
        assert_eq!(found(true, HwStage::Cs, true, 64), 0, "a host of 64-lane waves");
        assert_eq!(found(true, HwStage::Ps, true, 32), 0, "not compute");
        // (the process's own setting: the switch as it is)
        assert_eq!(after_loop(), std::env::var_os("VICTORY_AFTER_LOOP").is_some_and(|v| v == "1"));
    }
}
