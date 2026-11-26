//! `VICTORY_CPU_REWRITE`: the title's uses of instructions this PC's CPU
//! lacks (the console's is a Zen 2: `insertq`, `clzero`, `monitorx`,
//! `mwaitx`, `clwb`) rewritten in the mapped image before any of its code
//! runs, instead of an exception at every execution
//! ([`victory_loader::rewrite`]; the sites are the build's,
//! [`victory_game::cpu_sites`]). A title patch like the others: in memory
//! only, each site's bytes checked first, none with
//! `VICTORY_TITLE_PATCHES=0`.

use victory_game::cpu_sites::{CpuKind, CpuSite};
use victory_loader::rewrite::{Kind, Site};

/// What the switches ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Nothing: every site stays on the exception path, as before.
    Off,
    /// Say what would be done at each site; write nothing.
    List,
    On,
}

/// What `var` (the environment) asks for: the mode, and whether the
/// `monitorx`/`mwaitx` pair stays on the exception path. `Err`: a value of
/// `VICTORY_CPU_REWRITE` that means nothing (then nothing is done).
///
/// - `VICTORY_CPU_REWRITE`: unset, empty or `0` off; `1` on; `list` the
///   plan only.
/// - `VICTORY_TITLE_PATCHES=0`: off, with every other title patch.
/// - `VICTORY_CPU_REWRITE_MWAITX=0`: the two `monitorx` and the two
///   `mwaitx` sites are left as they are. The pair goes together: the
///   exception path's `mwaitx` waits on the line its `monitorx` named
///   (`VICTORY_MWAITX_WAIT`), and with the `monitorx` gone it would only
///   yield.
fn asked(var: impl Fn(&str) -> Option<String>) -> Result<(Mode, bool), String> {
    let hold_wait = var("VICTORY_CPU_REWRITE_MWAITX").is_some_and(|v| v.trim() == "0");
    if var("VICTORY_TITLE_PATCHES").is_some_and(|v| v.trim() == "0") {
        return Ok((Mode::Off, hold_wait));
    }
    let text = var("VICTORY_CPU_REWRITE").unwrap_or_default();
    let mode = match text.trim().to_ascii_lowercase().as_str() {
        "" | "0" => Mode::Off,
        "1" => Mode::On,
        "list" => Mode::List,
        _ => return Err(text),
    };
    Ok((mode, hold_wait))
}

/// The build's row as the loader takes it.
fn site(row: &CpuSite) -> Site<'static> {
    let kind = match row.kind {
        CpuKind::Clwb => Kind::Clwb,
        CpuKind::Monitorx => Kind::Monitorx,
        CpuKind::Mwaitx => Kind::Mwaitx,
        CpuKind::Insertq => Kind::Insertq,
        CpuKind::Clzero => Kind::Clzero,
    };
    Site { rva: row.rva, kind, len: row.len, expect: row.expect, targets: row.targets }
}

/// Rewrites the sites known for the build whose executable has the BLAKE3
/// hash `blake3_hex` in `image`, as the switches ask ([`asked`]), and says
/// what was done: one line, a line more for each site refused, and with
/// `VICTORY_CPU_REWRITE=list` a line for every site.
///
/// Called where the other title patches are made: the image is mapped
/// (as a library whose entry point and TLS callbacks the loader took
/// away), its imports are not yet bound, its DLLs are not yet loaded and
/// its main thread does not exist, so no thread can be inside a site.
pub fn rewrite(image: &victory_loader::LoadedImage, blake3_hex: &str) {
    use victory_loader::rewrite::{Options, Outcome, Why};
    let (mode, hold_wait) = match asked(|name| std::env::var(name).ok()) {
        Ok(asked) => asked,
        Err(text) => {
            println!("cpu rewrite    none for VICTORY_CPU_REWRITE={text}: it wants 1 or list");
            return;
        }
    };
    if mode == Mode::Off {
        return;
    }
    let rows = victory_game::cpu_sites::cpu_sites(blake3_hex);
    if rows.is_empty() {
        println!("cpu rewrite    none: no sites are known for this build (the exception path does them all)");
        return;
    }
    let sites: Vec<Site<'_>> = rows.iter().map(site).collect();
    let hold: &[Kind] = if hold_wait { &[Kind::Monitorx, Kind::Mwaitx] } else { &[] };
    let options = Options {
        list: mode == Mode::List,
        hold,
        native: &victory_loader::rewrite::native,
        register: true,
    };
    let size = image.info().size_of_image as usize;
    // SAFETY: the title's image, mapped and not yet run.
    let report = unsafe { victory_loader::rewrite::apply(image.base(), size, &sites, &options) };
    for (line, (_, _, outcome)) in report.lines(&sites).zip(&report.sites) {
        // A refusal is said whatever the mode; the rest only in a plan.
        let refused = matches!(outcome, Outcome::Left(why) if !matches!(why, Why::Native | Why::Held));
        if refused || mode == Mode::List {
            println!("cpu rewrite    {line}");
        }
    }
    let switches = if hold_wait { ", VICTORY_CPU_REWRITE_MWAITX=0" } else { "" };
    let which = if mode == Mode::List { "list: nothing written" } else { "1" };
    println!("cpu rewrite    {} (VICTORY_CPU_REWRITE={which}{switches})", report.summary());
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_loader::rewrite::{Plan, plan};

    fn with(vars: &[(&str, &str)]) -> Result<(Mode, bool), String> {
        asked(|name| vars.iter().find(|(n, _)| *n == name).map(|(_, v)| (*v).to_owned()))
    }

    #[test]
    fn the_switches_ask_for_a_mode() {
        assert_eq!(with(&[]), Ok((Mode::Off, false)));
        assert_eq!(with(&[("VICTORY_CPU_REWRITE", "0")]), Ok((Mode::Off, false)));
        assert_eq!(with(&[("VICTORY_CPU_REWRITE", "")]), Ok((Mode::Off, false)));
        assert_eq!(with(&[("VICTORY_CPU_REWRITE", "1")]), Ok((Mode::On, false)));
        assert_eq!(with(&[("VICTORY_CPU_REWRITE", " List ")]), Ok((Mode::List, false)));
        assert_eq!(with(&[("VICTORY_CPU_REWRITE", "yes")]), Err("yes".to_owned()));
        // Every title patch off: this one too.
        let off = [("VICTORY_CPU_REWRITE", "1"), ("VICTORY_TITLE_PATCHES", "0")];
        assert_eq!(with(&off), Ok((Mode::Off, false)));
        let on = [("VICTORY_CPU_REWRITE", "1"), ("VICTORY_TITLE_PATCHES", "1")];
        assert_eq!(with(&on), Ok((Mode::On, false)));
        // The wait pair held back, only by 0.
        let held = [("VICTORY_CPU_REWRITE", "1"), ("VICTORY_CPU_REWRITE_MWAITX", "0")];
        assert_eq!(with(&held), Ok((Mode::On, true)));
        let not_held = [("VICTORY_CPU_REWRITE", "1"), ("VICTORY_CPU_REWRITE_MWAITX", "1")];
        assert_eq!(with(&not_held), Ok((Mode::On, false)));
    }

    #[test]
    fn every_row_of_the_known_build_plans() {
        // Each row against its own bytes, as on a CPU with none of the
        // instructions: all 41 are planned, 29 of them as stubs.
        let build = &victory_game::KNOWN_BUILDS[0];
        let rows = victory_game::cpu_sites::cpu_sites(build.blake3);
        let mut stubs = 0;
        for row in rows {
            let s = site(row);
            match plan(&s, s.expect) {
                Ok(Plan::Stub) => stubs += 1,
                Ok(Plan::InPlace(new)) => assert_eq!(new.len(), s.expect.len()),
                Err(why) => panic!("+{:#x}: {}", row.rva, why.text()),
            }
        }
        assert_eq!((rows.len(), stubs), (41, 29));
    }

    /// `VICTORY_TITLE_EXE=<the title's executable>`: the table is that
    /// file's (by its BLAKE3), and every row is found in it and plans.
    /// Reads the file, changes nothing, prints one line a site.
    #[test]
    #[ignore = "reads the title's executable named by VICTORY_TITLE_EXE"]
    fn the_table_fits_the_title_executable() {
        let path = std::env::var("VICTORY_TITLE_EXE").expect("VICTORY_TITLE_EXE");
        let file = std::fs::read(&path).unwrap();
        let hash = blake3::hash(&file).to_hex();
        let rows = victory_game::cpu_sites::cpu_sites(&hash);
        assert!(!rows.is_empty(), "no sites for a file with BLAKE3 {hash}");
        let image = victory_pe::ImageInfo::parse(&file).unwrap();
        for row in rows {
            let s = site(row);
            let at = image.rva_to_file_offset(row.rva).expect("in a section of the file") as usize;
            let planned = plan(&s, &file[at..]);
            println!("+{:#x} {:?} {:02x?}: {planned:02x?}", row.rva, row.kind, row.expect);
            assert!(planned.is_ok(), "+{:#x}", row.rva);
        }
        println!("{} sites fit {path} ({hash})", rows.len());
    }
}
