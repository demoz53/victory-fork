//! `VICTORY_WAVE_PAIRS`: the paired form of console wave64 compute programs on a host of
//! 32-lane waves, and where [`super::emit_once`] tries it.
//!
//! Ported from fxceless/victory-fork PR #854 (commit 26600c6, by PiloteCode), read beside
//! DexNom/VIctory PR #64 (Castyan) and Corgansm/victory-corgi's `VICTORY_WAVE_PAIRS`. There
//! the form is on by default; here it is opt-in, and with the switch unset nothing changes.

use super::{HostShader, emit_program, lower, per_program};
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

/// The switch for the program whose hash is `hash` (`None`: not known, as for a translation
/// outside [`super::for_program`]): [`mode`], unless the lists leave the program out.
/// - `VICTORY_WAVE_PAIRS_ONLY=<hash>,...`: the paired form for the listed programs alone;
/// - `VICTORY_WAVE_PAIRS_NOT=<hash>,...`: not for the listed programs.
///
/// The hashes are those a run's log and `victory translate --list` name a shader by, as for
/// `VICTORY_WAVE_DEEP_SKIPS_ONLY` and `_NOT` ([`per_program`]). Part of the translation: a
/// wave64 compute program's answer is in its shader cache key (`victory-gpu`'s
/// `shader_cache`), not the switch or the lists, so changing them translates again only the
/// programs whose answer changes.
pub fn mode_for(hash: Option<&str>) -> Mode {
    struct Lists {
        not: Option<Vec<String>>,
        only: Option<Vec<String>>,
    }
    static LISTS: std::sync::OnceLock<Lists> = std::sync::OnceLock::new();
    let lists = LISTS.get_or_init(|| {
        let list = |name: &str| std::env::var(name).ok().map(|v| per_program::parse_hash_list(&v));
        Lists { not: list("VICTORY_WAVE_PAIRS_NOT"), only: list("VICTORY_WAVE_PAIRS_ONLY") }
    });
    pick(mode(), lists.not.as_deref(), lists.only.as_deref(), hash)
}

/// [`mode_for`] on the switches' values. Pure, for testing.
fn pick(mode: Mode, not: Option<&[String]>, only: Option<&[String]>, hash: Option<&str>) -> Mode {
    if per_program::decide(mode != Mode::Off, not, only, hash) { mode } else { Mode::Off }
}

/// The switch itself, for every program.
fn mode() -> Mode {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lists_pick_the_programs_that_take_the_paired_form() {
        const A: &str = "5dc3b25986622949";
        const B: &str = "6f225b641c5b0708";
        let list = |v: &str| per_program::parse_hash_list(v);
        for mode in [Mode::BeforeWide, Mode::First] {
            assert_eq!(pick(mode, None, None, Some(A)), mode);
            assert_eq!(pick(mode, None, None, None), mode, "an unknown program follows the switch");
            let only = list(&format!("#{A}"));
            assert_eq!(pick(mode, None, Some(&only), Some(A)), mode);
            assert_eq!(pick(mode, None, Some(&only), Some(B)), Mode::Off);
            assert_eq!(pick(mode, None, Some(&only), None), Mode::Off, "an unknown program is not listed");
            let not = list(B);
            assert_eq!(pick(mode, Some(&not), None, Some(A)), mode);
            assert_eq!(pick(mode, Some(&not), None, Some(B)), Mode::Off);
        }
        // the lists turn nothing on
        let only = list(A);
        assert_eq!(pick(Mode::Off, None, Some(&only), Some(A)), Mode::Off);
    }
}
