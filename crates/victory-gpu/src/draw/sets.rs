//! What the host list was last given for the bindings `draw.rs` sets on it
//! for every draw and dispatch: the vertex and index buffers, and per bind
//! point the root constants and root views, by root parameter. A call that
//! would give the list what it provably still holds is skipped
//! (`VICTORY_SET_CACHE=1`; off by default: every call is made, as before).
//!
//! After zocram4cc/victory-linux-perf d6bce43b6 (zocram4cc; GPL-3.0; also
//! in Proudnoob4/VIctory-NVIDIA nvidia-hybrid-v2 bf285c59f), which measured
//! 23.6% of these calls skipped. Rebuilt for our tree; what it holds is
//! exact because a binding keeps the value it was given until something
//! gives it another or voids it, and giving the same value again changes
//! nothing (a view or constant is read by the card at the draw, not at the
//! call). What is known is forgotten:
//!
//! 1. for another host list, or the same one reset or new: a [`Sets`] is
//!    valid for one value of the host list's binds serial
//!    (`HostList::binds_serial`), which no other list or recording has;
//! 2. at any binding a pass of ours records on any list outside this
//!    cache (clamps, the ambient stand-in, launches, depth plane passes and
//!    twin copies, the card's hash and compare passes, the in-place probe,
//!    the composer, rectangle expansion) and after every `ExecuteIndirect`
//!    of the title's (its command signature may set vertex, index or root
//!    arguments): each calls [`foreign_binds`], one count for the process,
//!    and every [`Sets`] forgets all it knows at its next use. The test
//!    `every_binding_outside_the_cache_is_announced` reads the crate's
//!    sources and fails on a binding call that neither goes through this
//!    cache nor announces itself;
//! 3. a root signature set on a bind point: its root arguments;
//! 4. a descriptor heap set: the root arguments of both bind points;
//! 5. a root view or table set by `draw.rs` outside this cache (root UAVs,
//!    root SRV tables): that parameter;
//! 6. a replay of the title's state after a list cut or a depth plane's
//!    pass (`State::replay`): everything;
//! 7. a title list reset or detached: the draw state is made anew.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::{D3D12_INDEX_BUFFER_VIEW, D3D12_VERTEX_BUFFER_VIEW};

/// Vertex buffer slots a list has.
const SLOTS: usize = 32;

/// Bindings recorded outside the cache since the start ([`foreign_binds`]).
static FOREIGN: AtomicU64 = AtomicU64::new(0);

/// A pass of ours recorded bindings on a host list (or the title's indirect
/// work may have set some): every [`Sets`] forgets what it knows at its next
/// use. Called on the thread that records, before or after the bindings and
/// before that list's next draw, so the count is seen there.
pub(crate) fn foreign_binds() {
    FOREIGN.fetch_add(1, Ordering::Release);
}

/// [`foreign_binds`] when dropped: a function that records bindings outside
/// the cache holds one from its start, so the announcement comes after its
/// last binding, on every way out of it.
pub(crate) struct Announce;

impl Drop for Announce {
    fn drop(&mut self) {
        foreign_binds();
    }
}

/// Whether the cache is on (`VICTORY_SET_CACHE=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SET_CACHE").is_some_and(|v| v == "1");
        if on {
            tracing::info!("VICTORY_SET_CACHE=1: vertex, index and root binds the host list provably holds are not given again");
        }
        on
    })
}

/// A root argument as the host list has it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bound {
    Cbv(u64),
    Srv(u64),
    Uav(u64),
}

/// Calls given to host lists and skipped since the last [`take_counts`]:
/// each thread adds its own every 1,024 decisions.
static COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    static HERE: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

fn count(sent: bool) {
    HERE.with(|c| {
        let mut n = c.get();
        n[usize::from(!sent)] += 1;
        if n[0] + n[1] >= 1024 {
            COUNTS[0].fetch_add(n[0], Ordering::Relaxed);
            COUNTS[1].fetch_add(n[1], Ordering::Relaxed);
            n = [0; 2];
        }
        c.set(n);
    });
}

/// Calls given (sent) and skipped since the last call (for the log line).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn take_counts() -> (u64, u64) {
    (COUNTS[0].swap(0, Ordering::Relaxed), COUNTS[1].swap(0, Ordering::Relaxed))
}

/// What one host list has bound, as far as this draw state knows it.
pub(crate) struct Sets {
    enabled: bool,
    /// The host list's binds serial and the foreign count this is of.
    serial: u64,
    foreign: u64,
    vertex: [Option<D3D12_VERTEX_BUFFER_VIEW>; SLOTS],
    index: Option<D3D12_INDEX_BUFFER_VIEW>,
    /// Graphics, compute; by root parameter.
    constants: [Vec<Option<Vec<u32>>>; 2],
    views: [Vec<Option<Bound>>; 2],
}

impl Default for Sets {
    fn default() -> Self {
        Sets::new(on())
    }
}

fn slot<T>(of: &mut Vec<Option<T>>, param: u32) -> &mut Option<T> {
    let at = param as usize;
    if of.len() <= at {
        of.resize_with(at + 1, || None);
    }
    &mut of[at]
}

impl Sets {
    pub(crate) fn new(enabled: bool) -> Self {
        Sets {
            enabled,
            serial: 0,
            foreign: 0,
            vertex: [None; SLOTS],
            index: None,
            constants: [Vec::new(), Vec::new()],
            views: [Vec::new(), Vec::new()],
        }
    }

    /// Before a decision for the host list of binds serial `serial`:
    /// another list, a reset or a binding of ours since makes all unknown.
    fn sync(&mut self, serial: u64) {
        let foreign = FOREIGN.load(Ordering::Acquire);
        if self.serial != serial || self.foreign != foreign {
            self.forget_all();
            (self.serial, self.foreign) = (serial, foreign);
        }
    }

    fn decide(&mut self, known: bool) -> bool {
        if self.enabled {
            count(!known);
        }
        !known
    }

    /// Whether the vertex buffers in slots `0..views.len()` must be given.
    pub(crate) fn vertex_buffers(&mut self, serial: u64, views: &[D3D12_VERTEX_BUFFER_VIEW]) -> bool {
        if !self.enabled {
            return true;
        }
        self.sync(serial);
        let known = views.iter().zip(&self.vertex).all(|(v, bound)| bound.as_ref() == Some(v));
        if !known {
            for (bound, v) in self.vertex.iter_mut().zip(views) {
                *bound = Some(*v);
            }
        }
        self.decide(known)
    }

    /// Whether the index buffer must be given.
    pub(crate) fn index_buffer(&mut self, serial: u64, view: &D3D12_INDEX_BUFFER_VIEW) -> bool {
        if !self.enabled {
            return true;
        }
        self.sync(serial);
        let known = self.index.as_ref() == Some(view);
        if !known {
            self.index = Some(*view);
        }
        self.decide(known)
    }

    /// Whether the root constants `values` of parameter `param` must be given.
    pub(crate) fn constants(&mut self, serial: u64, compute: bool, param: u32, values: &[u32]) -> bool {
        if !self.enabled {
            return true;
        }
        self.sync(serial);
        let at = slot(&mut self.constants[usize::from(compute)], param);
        let known = at.as_deref() == Some(values);
        if !known {
            *at = Some(values.to_vec());
            // A parameter holds constants or a view, not both.
            *slot(&mut self.views[usize::from(compute)], param) = None;
        }
        self.decide(known)
    }

    /// Whether root view `value` of parameter `param` must be given.
    pub(crate) fn view(&mut self, serial: u64, compute: bool, param: u32, value: Bound) -> bool {
        if !self.enabled {
            return true;
        }
        self.sync(serial);
        let at = slot(&mut self.views[usize::from(compute)], param);
        let known = *at == Some(value);
        if !known {
            *at = Some(value);
            *slot(&mut self.constants[usize::from(compute)], param) = None;
        }
        self.decide(known)
    }

    /// A root signature was set on a bind point (or a heap, which voids
    /// tables): none of its root arguments is known.
    pub(crate) fn forget_roots(&mut self, compute: bool) {
        self.constants[usize::from(compute)].clear();
        self.views[usize::from(compute)].clear();
    }

    /// One root parameter was set outside the cache.
    pub(crate) fn forget_root(&mut self, compute: bool, param: u32) {
        if let Some(at) = self.constants[usize::from(compute)].get_mut(param as usize) {
            *at = None;
        }
        if let Some(at) = self.views[usize::from(compute)].get_mut(param as usize) {
            *at = None;
        }
    }

    /// Nothing the list has set is known.
    pub(crate) fn forget_all(&mut self) {
        self.vertex = [None; SLOTS];
        self.index = None;
        self.forget_roots(false);
        self.forget_roots(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

    /// A fake host list: what it has bound, set only by the calls the
    /// cache lets through (and by "foreign" passes, which write it behind
    /// the cache's back). A draw reads what it has; with the cache wrong it
    /// would read another buffer than the title's.
    #[derive(Default)]
    struct FakeList {
        serial: u64,
        vertex: Vec<u64>,
        index: Option<u64>,
        constants: [std::collections::HashMap<u32, Vec<u32>>; 2],
        views: [std::collections::HashMap<u32, Bound>; 2],
    }

    fn vertex(address: u64) -> D3D12_VERTEX_BUFFER_VIEW {
        D3D12_VERTEX_BUFFER_VIEW { BufferLocation: address, SizeInBytes: 64, StrideInBytes: 16 }
    }

    fn index(address: u64) -> D3D12_INDEX_BUFFER_VIEW {
        D3D12_INDEX_BUFFER_VIEW { BufferLocation: address, SizeInBytes: 64, Format: DXGI_FORMAT(57) }
    }

    impl FakeList {
        /// A title draw binding vertex buffer `v`, index buffer `i`,
        /// constants `c` at graphics parameter 0 and a view `w` at 1,
        /// through the cache; then what the list holds is what the draw
        /// asked for.
        fn draw(&mut self, sets: &mut Sets, v: u64, i: u64, c: u32, w: u64) {
            if sets.vertex_buffers(self.serial, &[vertex(v)]) {
                self.vertex = vec![v];
            }
            if sets.index_buffer(self.serial, &index(i)) {
                self.index = Some(i);
            }
            if sets.constants(self.serial, false, 0, &[c, c + 1]) {
                self.constants[0].insert(0, vec![c, c + 1]);
            }
            if sets.view(self.serial, false, 1, Bound::Srv(w)) {
                self.views[0].insert(1, Bound::Srv(w));
            }
            assert_eq!(self.vertex, [v], "the draw reads the title's vertex buffer");
            assert_eq!(self.index, Some(i), "the draw reads the title's index buffer");
            assert_eq!(self.constants[0].get(&0), Some(&vec![c, c + 1]), "the title's constants");
            assert_eq!(self.views[0].get(&1), Some(&Bound::Srv(w)), "the title's root view");
        }

        /// A dispatch: compute constants at 0 and a UAV at 2.
        fn dispatch(&mut self, sets: &mut Sets, c: u32, u: u64) {
            if sets.constants(self.serial, true, 0, &[c]) {
                self.constants[1].insert(0, vec![c]);
            }
            if sets.view(self.serial, true, 2, Bound::Uav(u)) {
                self.views[1].insert(2, Bound::Uav(u));
            }
            assert_eq!(self.constants[1].get(&0), Some(&vec![c]));
            assert_eq!(self.views[1].get(&2), Some(&Bound::Uav(u)));
        }

        /// A pass of ours binding its own arguments on the list.
        fn foreign_pass(&mut self) {
            self.vertex = vec![0xbad];
            self.index = Some(0xbad);
            self.constants[1].insert(0, vec![0xbad]);
            self.views[1].insert(2, Bound::Uav(0xbad));
            self.constants[0].insert(0, vec![0xbad, 0xbad]);
            self.views[0].insert(1, Bound::Srv(0xbad));
            foreign_binds();
        }

        /// A root signature set on a bind point voids its arguments.
        fn root_signature(&mut self, sets: &mut Sets, compute: bool) {
            self.constants[usize::from(compute)].clear();
            self.views[usize::from(compute)].clear();
            sets.forget_roots(compute);
        }
    }

    /// The same draw again skips every call (and reads right); another
    /// buffer is given.
    #[test]
    fn repeated_binds_are_skipped_and_changed_ones_given() {
        let mut sets = Sets::new(true);
        let mut list = FakeList { serial: 1, ..Default::default() };
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        // Nothing reaches the list a second time: poison it to be sure.
        let before = (list.vertex.clone(), list.index);
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        assert_eq!((list.vertex.clone(), list.index), before);
        list.draw(&mut sets, 0x110, 0x200, 8, 0x300);
        list.draw(&mut sets, 0x110, 0x210, 8, 0x310);
    }

    /// Invalidation 2: a pass of ours that binds on the list (a clamp, the
    /// ambient pass, a launch, the card's passes, an ExecuteIndirect of the
    /// title's) makes the next draw give everything again.
    #[test]
    fn a_pass_of_ours_on_the_list_makes_everything_given_again() {
        let mut sets = Sets::new(true);
        let mut list = FakeList { serial: 2, ..Default::default() };
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        list.dispatch(&mut sets, 3, 0x400);
        list.foreign_pass();
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        list.dispatch(&mut sets, 3, 0x400);
    }

    /// Invalidation 1: another host list (a cut, a reset, a list from the
    /// pool: each a new serial) starts with nothing known.
    #[test]
    fn another_or_reset_host_list_is_given_everything() {
        let mut sets = Sets::new(true);
        let mut first = FakeList { serial: 10, ..Default::default() };
        first.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        let mut second = FakeList { serial: 11, ..Default::default() };
        second.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        // The first list reset: same object, new serial, nothing bound.
        first = FakeList { serial: 12, ..Default::default() };
        first.draw(&mut sets, 0x100, 0x200, 7, 0x300);
    }

    /// Invalidations 3 and 4: a root signature (or heap) set on one bind
    /// point voids its arguments only.
    #[test]
    fn a_root_signature_voids_its_bind_points_arguments() {
        let mut sets = Sets::new(true);
        let mut list = FakeList { serial: 20, ..Default::default() };
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        list.dispatch(&mut sets, 3, 0x400);
        list.root_signature(&mut sets, false);
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        let compute = (list.constants[1].clone(), list.views[1].clone());
        list.dispatch(&mut sets, 3, 0x400);
        assert_eq!((list.constants[1].clone(), list.views[1].clone()), compute, "compute untouched, skipped");
        list.root_signature(&mut sets, true);
        list.dispatch(&mut sets, 3, 0x400);
    }

    /// Invalidation 5: a parameter set outside the cache (a root UAV or a
    /// root SRV table of `draw.rs`) is given again, the others not.
    #[test]
    fn a_parameter_set_outside_the_cache_is_given_again() {
        let mut sets = Sets::new(true);
        let mut list = FakeList { serial: 30, ..Default::default() };
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        // A root SRV table at parameter 1, behind the cache.
        list.views[0].insert(1, Bound::Srv(0x999));
        sets.forget_root(false, 1);
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
    }

    /// Invalidation 6: a replay forgets everything (the list has nothing).
    #[test]
    fn a_replay_makes_everything_given_again() {
        let mut sets = Sets::new(true);
        let mut list = FakeList { serial: 40, ..Default::default() };
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
        list = FakeList { serial: 40, ..Default::default() };
        sets.forget_all();
        list.draw(&mut sets, 0x100, 0x200, 7, 0x300);
    }

    /// A parameter that held constants and then a view (another root
    /// signature) is not taken for either; switched off, every call goes.
    #[test]
    fn kinds_do_not_mix_and_off_gives_every_call() {
        let mut sets = Sets::new(true);
        assert!(sets.constants(50, false, 3, &[1]));
        assert!(sets.view(50, false, 3, Bound::Cbv(0x10)));
        assert!(sets.constants(50, false, 3, &[1]), "the view replaced the constants");
        assert!(sets.view(50, false, 3, Bound::Srv(0x10)), "another kind at the same address");
        let mut off = Sets::new(false);
        for _ in 0..3 {
            assert!(off.vertex_buffers(50, &[vertex(1)]));
            assert!(off.index_buffer(50, &index(1)));
            assert!(off.constants(50, false, 0, &[1]));
            assert!(off.view(50, false, 1, Bound::Srv(1)));
        }
    }

    /// The patterns of a binding call on a host list.
    const BINDINGS: [&str; 6] = [
        "IASetVertexBuffers(",
        "IASetIndexBuffer(",
        "SetGraphicsRoot",
        "SetComputeRoot",
        "SetDescriptorHeaps(",
        "ExecuteIndirect(",
    ];

    /// What makes a binding call outside the cache safe: the function it is
    /// in announces it (`foreign_binds()`), or it is one of the cache's own
    /// (a decision or a forget of `self.sets` within the lines just above).
    fn announced(lines: &[&str], at: usize) -> bool {
        let start = (0..=at)
            .rev()
            .find(|&k| {
                let l = lines[k].trim_start();
                ["fn ", "pub(crate) fn ", "pub(super) fn ", "pub fn ", "unsafe fn ", "pub(crate) unsafe fn ", "pub(crate) extern ", "extern "]
                    .iter()
                    .any(|p| l.starts_with(p))
            })
            .unwrap_or(0);
        let end = (at + 60).min(lines.len());
        let body = &lines[start..end];
        if body.iter().any(|l| l.contains("foreign_binds()") || l.contains("sets::Announce") || l.contains("sets: own list")) {
            return true;
        }
        // A parameter forgotten where the function starts (root UAVs and
        // root SRV tables of `draw.rs`).
        if lines[start..at].iter().any(|l| l.contains("self.sets.forget_root(")) {
            return true;
        }
        let near = at.saturating_sub(16);
        lines[near..at].iter().any(|l| l.contains("self.sets."))
    }

    /// Every binding call on a host list in the crate's sources (test
    /// modules aside) goes through the cache or announces itself, so a new
    /// pass of ours cannot leave the cache believing what the list no
    /// longer holds.
    #[test]
    fn every_binding_outside_the_cache_is_announced() {
        fn files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    files(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs")
                    // (Modules built only for tests: `mod bench;` under cfg(test).)
                    && !path.file_stem().is_some_and(|s| s.to_string_lossy().ends_with("bench") || s.to_string_lossy().ends_with("tests"))
                {
                    out.push(path);
                }
            }
        }
        let mut all = Vec::new();
        files(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut all);
        let mut missing = Vec::new();
        let mut seen = 0;
        for path in all {
            let text = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            // Lines inside test code (a `mod ...tests` block, or the block of
            // an item under a `#[cfg(...test...)]`) are left out.
            let mut skip_depth: Option<i64> = None;
            let mut depth: i64 = 0;
            let mut pending_test = false;
            for (k, line) in lines.iter().enumerate() {
                let trimmed = line.trim_start();
                let opens = line.matches('{').count() as i64;
                let closes = line.matches('}').count() as i64;
                if skip_depth.is_none() && !trimmed.starts_with("//") {
                    let is_mod = trimmed.starts_with("mod ") || trimmed.starts_with("pub(crate) mod ") || trimmed.starts_with("pub mod ");
                    let named_test = is_mod && (trimmed.contains("tests") || trimmed.contains("bench"));
                    if trimmed.ends_with('{') && (pending_test || named_test) {
                        skip_depth = Some(depth);
                        pending_test = false;
                    }
                }
                if trimmed.starts_with("#[cfg(") && trimmed.contains("test") && !trimmed.contains("not(test") {
                    pending_test = true;
                } else if pending_test && trimmed.ends_with(';') {
                    pending_test = false;
                }
                let in_test = skip_depth.is_some();
                depth += opens - closes;
                if let Some(d) = skip_depth
                    && depth <= d
                {
                    skip_depth = None;
                }
                if in_test || trimmed.starts_with("//") {
                    continue;
                }
                if BINDINGS.iter().any(|b| line.contains(b)) {
                    seen += 1;
                    if !announced(&lines, k) {
                        missing.push(format!("{}:{}: {}", path.display(), k + 1, trimmed));
                    }
                }
            }
        }
        assert!(seen >= 40, "the scan found only {seen} binding calls: it reads nothing");
        assert!(missing.is_empty(), "binding calls neither through the cache nor announced (crate::draw::sets::foreign_binds):\n{}", missing.join("\n"));
    }
}
