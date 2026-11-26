//! `VICTORY_WAVE_PAIRS`: the paired form of console wave64 compute programs on a host of
//! 32-lane waves, and where [`super::emit_once`] tries it.
//!
//! Ported from fxceless/victory-fork PR #854 (commit 26600c6, by PiloteCode), read beside
//! DexNom/VIctory PR #64 (Castyan) and Corgansm/victory-corgi's `VICTORY_WAVE_PAIRS`. There
//! the form is on by default; here it is opt-in, and with the switch unset nothing changes.

use super::{HostShader, emit_program, lower};
use crate::dxbc::Container;
use crate::native::{NativeProgram, StageIo};
use crate::{ShaderError, io, ir, resource, structure};

/// Where the paired form is tried among a stuck wave64 program's forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Not at all (the switch unset, or any other value): translations as before.
    Off,
    /// `VICTORY_WAVE_PAIRS=1`: after the exact forms without added barriers (each host wave
    /// on its own where that is the allocation idiom, the forms without early-outs and
    /// skips), before the wave-wide form with its group barriers and before every
    /// approximate form. The wave-wide form stays the last resort.
    BeforeWide,
    /// `VICTORY_WAVE_PAIRS=first`: before every other retry form, for every wave64 program
    /// the host cannot run as it is (those that today take the form of two host waves that
    /// exchange through LDS too).
    First,
}

/// The switch. Part of the translation: the shader cache keys on it (`victory-gpu`'s does).
pub fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = TEST_PAIRS.get() {
        return mode;
    }
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_WAVE_PAIRS").as_deref() {
        Ok("1") => Mode::BeforeWide,
        Ok("first") => Mode::First,
        _ => Mode::Off,
    })
}

#[cfg(test)]
thread_local! {
    /// A test's `VICTORY_WAVE_PAIRS`, for its thread only.
    pub(crate) static TEST_PAIRS: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

/// `p` (a console wave64 compute program on a host of 32-lane waves) in the paired form
/// ([`lower::with_paired_lanes`]): each console wave run by one host wave of 32 lanes, each
/// thread two of its lanes, with every per-lane branch taken by the whole wave as the console
/// takes it ([`structure::structure_wide`]), so that a host wave's threads go the same way and
/// EXEC stays each lane's guard. The program's barriers are reached by whole host waves where
/// the console's waves reach them; the only barriers added are on entry, before the program,
/// where a group of several console waves numbers its waves (`Lower::wave_base`). A thread
/// group whose console waves are not full would need the lanes past its end kept out of every
/// effect: refused. So is a program with spilled user data (the ray-tracing structure
/// builder's), whose dispatches may run part of a thread group: each lane of a thread would
/// need its own "is this thread dispatched" (DexNom/VIctory a65db62 takes those; not ported).
/// Descriptors and addresses stay those traced on the lanes' own paths (`res`, as for every
/// other form): the per-lane form rests on a lane that reads a scalar after a join having
/// come the way the wave did, and knowing less of a lane's way the wave-wide form would miss
/// some of them (a sampler set in a skip that only a wave with no lane left for it skips).
pub(super) fn paired(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let refuse = |what| Err(ShaderError::Unsupported { offset: 0, what });
    let StageIo::Cs(cs) = native.io else {
        return refuse("paired lanes outside compute");
    };
    if cs.threads.iter().product::<u32>() % 64 != 0 {
        return refuse("paired lanes in a thread group that does not fill its console waves");
    }
    if crate::spill::is_spilled(native) {
        return refuse("paired lanes in a program whose dispatches may run part of a thread group");
    }
    // every per-lane branch the wave's; again while that leaves another branch per lane
    let mut wide = std::collections::HashSet::new();
    let st = loop {
        let st = structure::structure_wide(p, &wide)?;
        let lanes: Vec<usize> = (0..st.exits.len())
            .filter(|b| !wide.contains(b))
            .filter(|&b| {
                matches!(st.exits[b], structure::LaneExit::Branch { when, .. }
                    if !st.bools.support(when).iter().all(|a| matches!(a, structure::Atom::Uniform(_))))
            })
            .collect();
        if lanes.is_empty() {
            break st;
        }
        wide.extend(lanes);
    };
    lower::with_paired_lanes(|| emit_program(c, native, p, &st, io, res))
}
