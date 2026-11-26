//! `victory shader-cache <folder>`: what is in a translated-shader cache
//! folder (a run's is `<cache>/shaders`), which of it is dead, and, only
//! when asked twice, its removal.
//!
//! `victory-gpu`'s `shader_cache` keeps each run key (a translator and its
//! settings) in its own folder, `k-<16 hex digits>`, with a `key.txt`
//! saying what the key is made of. Entries written before that sit in 256
//! folders named by two hex digits, each keyed by the executable that wrote
//! it: nothing in them says which, so they are shown by the day they were
//! written. No build from the per-key folders on reads them.
//!
//! - `--list` (the default): one line per key folder, with its number of
//!   files, their size, the newest file's time (UTC) and the key's
//!   identity; then the older entries by day.
//! - `--prune-older-than <days>`: the key folders whose newest file is
//!   older than that, and the older entries written before then, are
//!   listed as what would go. Nothing is removed without `--yes` as well.
//! - `--warm <path>...`: translates the containers found (`*.cdxil`, as
//!   `VICTORY_SHADER_SAVE` writes them) through this build's cache folder,
//!   under this process's translation switches, as a run does: a build
//!   whose translator is new then finds its shaders translated at its
//!   first run. Says how many came from the cache.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

pub struct Args {
    pub dir: PathBuf,
    pub prune_older_than: Option<u64>,
    pub yes: bool,
    pub warm: Vec<PathBuf>,
    pub jobs: Option<usize>,
    pub limit: Option<usize>,
}

/// What a group of entries is, for removal.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// A run key's folder.
    Key,
    /// Entries of the layout before key folders, written on one day.
    Old,
    /// Anything else in the folder: counted, never removed.
    Other,
}

#[derive(Clone, Debug)]
struct Group {
    kind: Kind,
    name: String,
    /// The key's identity and switches (`key.txt`), if it says.
    what: String,
    files: u64,
    bytes: u64,
    newest: SystemTime,
    /// The files, for removal (a key folder: the folder alone).
    paths: Vec<PathBuf>,
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// A key folder's name: `k-` and 16 hex digits.
fn is_key_folder(name: &str) -> bool {
    name.strip_prefix("k-").is_some_and(|hex| is_hex(hex, 16))
}

/// The civil date of a day counted from 1970-01-01 (the proleptic
/// Gregorian calendar; days before 1970 are not asked for).
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + u64::from(month <= 2), month, day)
}

fn secs(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn day_text(t: SystemTime) -> String {
    let (y, m, d) = civil(secs(t) / 86_400);
    format!("{y:04}-{m:02}-{d:02}")
}

fn time_text(t: SystemTime) -> String {
    let s = secs(t) % 86_400;
    format!("{} {:02}:{:02}", day_text(t), s / 3600, s % 3600 / 60)
}

fn size_text(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / f64::from(1u32 << 30))
    } else {
        format!("{:.1} MB", bytes as f64 / f64::from(1u32 << 20))
    }
}

/// Every file under `dir`: count, bytes and the newest modification time.
fn tally(dir: &Path, files: &mut u64, bytes: &mut u64, newest: &mut SystemTime) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            tally(&entry.path(), files, bytes, newest)?;
        } else {
            *files += 1;
            *bytes += meta.len();
            *newest = (*newest).max(meta.modified().unwrap_or(UNIX_EPOCH));
        }
    }
    Ok(())
}

/// The groups of entries in a cache folder, newest first. Reads names,
/// sizes and times, and each key folder's `key.txt`; changes nothing.
fn scan(dir: &Path) -> Result<Vec<Group>> {
    let mut groups = Vec::new();
    let mut old: BTreeMap<String, Group> = BTreeMap::new();
    let mut other = Group {
        kind: Kind::Other,
        name: "other files".to_owned(),
        what: "not the cache's: left alone".to_owned(),
        files: 0,
        bytes: 0,
        newest: UNIX_EPOCH,
        paths: Vec::new(),
    };
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let meta = entry.metadata()?;
        if meta.is_dir() && is_key_folder(&name) {
            let note = std::fs::read_to_string(entry.path().join(victory_gpu::shader_cache::KEY_NOTE))
                .unwrap_or_default();
            let line = |start: &str| {
                note.lines()
                    .find_map(|l| l.strip_prefix(start))
                    .unwrap_or("?")
                    .to_owned()
            };
            let mut g = Group {
                kind: Kind::Key,
                name,
                what: format!("{}; switches set: {}", line("identity: "), line("switches set: ")),
                files: 0,
                bytes: 0,
                newest: UNIX_EPOCH,
                paths: vec![entry.path()],
            };
            tally(&entry.path(), &mut g.files, &mut g.bytes, &mut g.newest)?;
            groups.push(g);
        } else if meta.is_dir() && is_hex(&name, 2) {
            // The layout before key folders: `<2 hex>/<62 hex>`.
            for file in std::fs::read_dir(entry.path())? {
                let file = file?;
                let meta = file.metadata()?;
                let entry_name = file.file_name().to_string_lossy().into_owned();
                if meta.is_file() && is_hex(&entry_name, 62) {
                    let written = meta.modified().unwrap_or(UNIX_EPOCH);
                    let g = old.entry(day_text(written)).or_insert_with(|| Group {
                        kind: Kind::Old,
                        name: format!("before key folders, written {}", day_text(written)),
                        what: "keyed by the executables of that day: read by no build with key folders".to_owned(),
                        files: 0,
                        bytes: 0,
                        newest: UNIX_EPOCH,
                        paths: Vec::new(),
                    });
                    g.files += 1;
                    g.bytes += meta.len();
                    g.newest = g.newest.max(written);
                    g.paths.push(file.path());
                } else {
                    other.files += 1;
                    other.bytes += meta.len();
                }
            }
        } else if meta.is_dir() {
            tally(&entry.path(), &mut other.files, &mut other.bytes, &mut other.newest)?;
        } else {
            other.files += 1;
            other.bytes += meta.len();
            other.newest = other.newest.max(meta.modified().unwrap_or(UNIX_EPOCH));
        }
    }
    groups.sort_by_key(|g| std::cmp::Reverse(g.newest));
    groups.extend(old.into_values().rev());
    if other.files > 0 {
        groups.push(other);
    }
    Ok(groups)
}

fn line(g: &Group) -> String {
    format!(
        "{:<40} {:>8} files {:>10}  newest {}  {}",
        g.name,
        g.files,
        size_text(g.bytes),
        time_text(g.newest),
        g.what
    )
}

/// The groups `--prune-older-than days` takes at the time `now`: key
/// folders and older-layout days whose newest file is older than that.
/// Never the other files.
fn prunable(groups: &[Group], days: u64, now: SystemTime) -> Vec<&Group> {
    let cutoff = now
        .checked_sub(Duration::from_secs(days.saturating_mul(86_400)))
        .unwrap_or(UNIX_EPOCH);
    groups
        .iter()
        .filter(|g| g.kind != Kind::Other && g.files > 0 && g.newest < cutoff)
        .collect()
}

pub fn run(a: Args) -> Result<()> {
    if !a.warm.is_empty() {
        if a.prune_older_than.is_some() {
            bail!("--warm and --prune-older-than: one at a time");
        }
        return warm(&a);
    }
    if a.yes && a.prune_older_than.is_none() {
        bail!("--yes goes with --prune-older-than <days>");
    }
    let groups = scan(&a.dir)?;
    println!("{} (times in UTC)", a.dir.display());
    for g in &groups {
        println!("{}", line(g));
    }
    let (files, bytes) = groups.iter().fold((0, 0), |(f, b), g| (f + g.files, b + g.bytes));
    println!("total: {files} files, {}", size_text(bytes));
    let Some(days) = a.prune_older_than else {
        return Ok(());
    };
    let take = prunable(&groups, days, SystemTime::now());
    let (files, bytes) = take.iter().fold((0, 0), |(f, b), g| (f + g.files, b + g.bytes));
    println!();
    for g in &take {
        println!("{} {}", if a.yes { "removing:" } else { "would remove:" }, line(g));
    }
    if !a.yes {
        println!(
            "would remove {files} files, {} (groups with no file newer than {days} days). Nothing was removed: add --yes to remove them.",
            size_text(bytes)
        );
        return Ok(());
    }
    for g in &take {
        match g.kind {
            Kind::Key => {
                for folder in &g.paths {
                    std::fs::remove_dir_all(folder).with_context(|| format!("removing {}", folder.display()))?;
                }
            }
            Kind::Old => {
                for file in &g.paths {
                    std::fs::remove_file(file).with_context(|| format!("removing {}", file.display()))?;
                }
            }
            Kind::Other => {}
        }
    }
    println!("removed {files} files, {}", size_text(bytes));
    Ok(())
}

/// Container files (`*.cdxil`) under the given paths, in name order.
fn containers(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    fn walk(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_dir() {
            for entry in std::fs::read_dir(path).with_context(|| format!("listing {}", path.display()))? {
                walk(&entry?.path(), out)?;
            }
        } else if path.extension().is_some_and(|e| e == "cdxil") {
            out.push(path.to_owned());
        }
        Ok(())
    }
    let mut out = Vec::new();
    for path in paths {
        walk(path, &mut out)?;
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn warm(a: &Args) -> Result<()> {
    let mut files = containers(&a.warm)?;
    if let Some(limit) = a.limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        bail!("no container files (*.cdxil) under the given paths");
    }
    victory_gpu::shader_cache::use_shader_cache(a.dir.clone());
    // (the host device is made here, as at a run's first translation: its
    // wave width and view heap limit are part of the key)
    let (folder, made_of) = victory_gpu::shader_cache::run_key();
    println!("key folder   {}", a.dir.join(&folder).display());
    for l in made_of.lines() {
        println!("             {l}");
    }
    let threads = a
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let started = std::time::Instant::now();
    let (next, refused) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..threads.min(files.len()) {
            s.spawn(|| {
                while let Some(file) = files.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let translated = std::fs::read(file)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| victory_gpu::shader_cache::translate_cached(&bytes).map(drop));
                    if translated.is_err() {
                        refused.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    let (hits, misses) = victory_gpu::shader_cache::hits_and_misses();
    if hits + misses == 0 {
        // (`victory-pipeline` refuses a folder inside the source tree, for one)
        bail!("the cache folder could not be used (see the warning above): nothing was kept");
    }
    println!(
        "warm         {} containers in {:.1} s: {hits} from the cache, {misses} translated now, {} not translated",
        files.len(),
        started.elapsed().as_secs_f64(),
        refused.load(Ordering::Relaxed)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system's temporary one, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("victory-shader-cache-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, path: &str, bytes: usize) -> PathBuf {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, vec![0u8; bytes]).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn args(dir: &Path, days: Option<u64>, yes: bool) -> Args {
        Args {
            dir: dir.to_owned(),
            prune_older_than: days,
            yes,
            warm: Vec::new(),
            jobs: None,
            limit: None,
        }
    }

    #[test]
    fn days_since_1970_are_civil_dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(59), (1970, 3, 1));
        assert_eq!(civil(11_016), (2000, 2, 29));
        assert_eq!(civil(20_782), (2026, 11, 25));
        assert_eq!(day_text(UNIX_EPOCH + Duration::from_secs(20_782 * 86_400 + 3 * 3600 + 5 * 60)), "2026-11-25");
        assert_eq!(time_text(UNIX_EPOCH + Duration::from_secs(20_782 * 86_400 + 3 * 3600 + 5 * 60)), "2026-11-25 03:05");
    }

    #[test]
    fn only_the_caches_own_names_are_key_folders() {
        assert!(is_key_folder("k-0123456789abcdef"));
        for other in ["k-0123456789abcde", "k-0123456789abcdeg", "0123456789abcdef", "k-", "00", "pipelines"] {
            assert!(!is_key_folder(other), "{other}");
        }
    }

    #[test]
    fn a_folder_is_listed_by_key_and_by_day_and_other_files_are_left_alone() {
        let s = Scratch::new("list");
        s.file("k-00000000000000aa/12/entry-one", 1000);
        s.file("k-00000000000000aa/34/entry-two", 500);
        std::fs::write(
            s.0.join("k-00000000000000aa").join(victory_gpu::shader_cache::KEY_NOTE),
            "identity: translator abc, entries def\nsettings: x\nswitches set: VICTORY_LOOP_EXITS=1\n",
        )
        .unwrap();
        s.file("k-00000000000000bb/56/entry", 10);
        s.file(&format!("7f/{}", "a".repeat(62)), 2000);
        s.file("7f/not-an-entry", 7);
        s.file("notes.txt", 3);
        let groups = scan(&s.0).unwrap();
        let aa = groups.iter().find(|g| g.name == "k-00000000000000aa").unwrap();
        // (its key.txt is a file of the folder too)
        assert_eq!((aa.kind.clone(), aa.files), (Kind::Key, 3));
        assert!(aa.bytes >= 1500);
        assert_eq!(aa.what, "translator abc, entries def; switches set: VICTORY_LOOP_EXITS=1");
        let bb = groups.iter().find(|g| g.name == "k-00000000000000bb").unwrap();
        assert_eq!((bb.files, bb.bytes, bb.what.as_str()), (1, 10, "?; switches set: ?"));
        let old: Vec<_> = groups.iter().filter(|g| g.kind == Kind::Old).collect();
        assert_eq!(old.len(), 1);
        assert_eq!((old[0].files, old[0].bytes, old[0].paths.len()), (1, 2000, 1));
        let other = groups.iter().find(|g| g.kind == Kind::Other).unwrap();
        assert_eq!((other.files, other.bytes), (2, 10));
        assert!(other.paths.is_empty());
        // Listing changes nothing.
        run(args(&s.0, None, false)).unwrap();
        assert!(s.0.join("7f").join("a".repeat(62)).exists());
    }

    #[test]
    fn pruning_takes_only_old_groups_and_nothing_without_yes() {
        let s = Scratch::new("prune");
        let entry = s.file("k-00000000000000aa/12/entry", 100);
        let old = s.file(&format!("00/{}", "b".repeat(62)), 200);
        let other = s.file("notes.txt", 3);
        let groups = scan(&s.0).unwrap();
        let now = SystemTime::now();
        // Everything was written just now: nothing is older than a day...
        assert!(prunable(&groups, 1, now).is_empty());
        // ...and from ten days on, the key folder and the old entries are
        // older than a week, the other file is never taken.
        let later = now + Duration::from_secs(10 * 86_400);
        let take = prunable(&groups, 7, later);
        assert_eq!(take.len(), 2);
        assert!(take.iter().all(|g| g.kind != Kind::Other));
        assert!(prunable(&groups, 30, later).is_empty());
        // Without --yes nothing goes, whatever the age asked for.
        run(args(&s.0, Some(0), false)).unwrap();
        assert!(entry.exists() && old.exists() && other.exists());
        // --yes alone is refused.
        assert!(run(args(&s.0, None, true)).is_err());
        assert!(entry.exists() && old.exists() && other.exists());
        // With both, and files older than "0 days" by the time it runs:
        std::thread::sleep(Duration::from_millis(1100));
        run(args(&s.0, Some(0), true)).unwrap();
        assert!(!entry.exists() && !old.exists());
        assert!(!s.0.join("k-00000000000000aa").exists());
        assert!(other.exists());
    }

    #[test]
    fn containers_are_found_by_their_extension_once_and_in_order() {
        let s = Scratch::new("warm");
        let b = s.file("b/2.compute.cdxil", 1);
        let a = s.file("a/1.pixel.cdxil", 1);
        s.file("a/1.pixel.dxil", 1);
        assert_eq!(containers(&[s.0.clone(), a.clone()]).unwrap(), vec![a, b]);
    }
}
