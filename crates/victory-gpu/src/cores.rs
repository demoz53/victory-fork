//! Which processors the threads a frame waits for run on, where a processor
//! has two kinds of cores (`VICTORY_FAST_CORES`; off by default).
//!
//! Ported from avatarmanz/VIctory-7900XTX (commits 96f2f32e8 and b6202840a,
//! GPL-3.0-or-later, as this), whose `cpu_sets` and `keep_on_fast_cores` are
//! fxceless/victory-fork's (`framelog.rs`, commit 0b0074e0). That fork's
//! submitting thread is one of its own; here the title's thread submits and
//! our preparing runs on it, so it has a stage of its own
//! ([`keep_submitter_on_fast_cores`]).

/// The system's processor sets of group 0: each one's id and its efficiency
/// class (higher is faster).
fn cpu_sets() -> Vec<(u32, u8)> {
    cpu_sets_with_processor().into_iter().map(|(id, _, class)| (id, class)).collect()
}

/// [`cpu_sets`] with each set's logical processor: id, logical processor,
/// efficiency class.
fn cpu_sets_with_processor() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_of(sets: &[(u32, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// The slow logical processors of group 0 as a mask (a lower efficiency
/// class than the best): 0 where all cores are alike. fxceless/victory-fork's
/// `framelog::slow_processors` (commit 39731795d), for `VICTORY_SPREAD`.
pub(crate) fn slow_processors() -> usize {
    slow_of(&cpu_sets_with_processor())
}

fn slow_of(sets: &[(u32, u8, u8)]) -> usize {
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    sets.iter().filter(|c| c.2 < best && u32::from(c.1) < usize::BITS).fold(0, |mask, c| mask | 1usize << c.1)
}

/// `VICTORY_FAST_CORES` as a stage; anything else than 1 to 4 is 0.
fn stage_of(value: Option<&str>) -> u32 {
    value.and_then(|v| v.trim().parse().ok()).filter(|s| (1..=4).contains(s)).unwrap_or(0)
}

/// `VICTORY_FAST_CORES`: unset or 0: no thread asks (as before); 1: our
/// threads a frame waits for one by one (the one that lands flights, the one
/// that settles them); 2: the threads they share their work with too (the
/// helpers of the wide refreshes, fills and snapshots); 3: also each title
/// thread that submits lists, on which our preparing runs; 4: those that
/// work ahead of the frame as well (the texture makers): more threads by
/// then than the fast cores may run at a time.
///
/// Nothing is skipped and nothing runs in another order: only where the
/// scheduler puts these threads changes.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| stage_of(std::env::var("VICTORY_FAST_CORES").ok().as_deref()))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. fxceless' submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.)
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for the calling title thread, asked once, at its
/// first submission: the refreshes, fills and sends of a submission run on
/// it (58 ms a frame of running in the boat scene of run all15b-on1, the
/// card idle behind it for 49 of the frame's 252 ms).
pub(crate) fn keep_submitter_on_fast_cores() {
    thread_local! {
        static ASKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if stage() < 3 || ASKED.replace(true) {
        return;
    }
    ask("a title thread that submits", 3);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 4);
}

/// The fast sets and how many sets there are; no fast ones where all are
/// alike or nothing is known.
fn fast() -> &'static (Vec<u32>, usize) {
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    FAST.get_or_init(|| {
        let sets = cpu_sets();
        (fast_of(&sets), sets.len())
    })
}

/// Selects the fast sets for the calling thread: `None` where there is
/// nothing to ask for, else whether the system took it.
fn select_fast() -> Option<bool> {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    let (fast, _) = fast();
    if fast.is_empty() {
        return None;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    Some(unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) } != 0)
}

fn ask(what: &str, from: u32) {
    if stage() < from {
        return;
    }
    let (fast, of) = fast();
    match select_fast() {
        None => {}
        Some(false) => {
            tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
        }
        Some(true) => {
            tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1), (257, 1), (258, 0)]), [256, 257]);
        assert!(fast_of(&[(256, 0), (257, 0)]).is_empty());
        assert!(fast_of(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
    }

    #[test]
    fn the_slow_processors_are_those_of_a_worse_class() {
        // This PC's layout: 0-15 fast (class 1), 16-31 slow (class 0).
        let sets: Vec<(u32, u8, u8)> = (0..32u8).map(|p| (256 + u32::from(p), p, u8::from(p < 16))).collect();
        assert_eq!(slow_of(&sets), 0xFFFF_0000);
        assert_eq!(slow_of(&[(256, 0, 0), (257, 1, 0)]), 0, "all alike");
        assert_eq!(slow_of(&[]), 0);
        // The system's own answer: slow ones only where there are fast ones too.
        assert_eq!(slow_processors() == 0, fast().0.is_empty());
    }

    #[test]
    fn the_switch_is_off_unless_it_names_a_stage() {
        assert_eq!(stage_of(None), 0);
        assert_eq!(stage_of(Some("0")), 0);
        assert_eq!(stage_of(Some(" 2 ")), 2);
        assert_eq!(stage_of(Some("4")), 4);
        assert_eq!(stage_of(Some("5")), 0);
        assert_eq!(stage_of(Some("on")), 0);
    }

    /// On a processor with two kinds of cores the system takes the
    /// selection; on one with cores alike there is nothing to ask.
    #[test]
    fn a_thread_selects_the_fast_cores_where_there_are_slow_ones() {
        let took = std::thread::spawn(select_fast).join().unwrap();
        assert_eq!(took.is_some(), !fast().0.is_empty());
        assert_ne!(took, Some(false));
    }
}
